"""Skill backends: how a physical skill actually runs underneath the harness.

ScriptedBackend  - the original: the skill happens instantly with a fixed failure probability.
PolicyBackend    - VLA-LIKE simulation of a learned manipulation policy. This simulates the *interface
                   characteristics* of such a policy, not its quality. Success odds are made up.

A policy skill is an episode, not an event:
  * it runs over simulated time and emits motion in chunks (frames every 0.5 s)
  * it has NO done signal: it keeps acting, or stalls, or quietly finishes having achieved nothing
  * its odds depend on the situation (distance, fragility, preconditions), not on a flat coin flip
  * a supervisor judges it online through the Evaluator at segment boundaries (every 2 s), and may
    stop it early on a stall instead of waiting out the step budget

Navigation and charging stay scripted (classical stack); only pick, place and open run on the policy.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from scout.warehouse.world import ROOMS, Warehouse, dist

FRAME_S = 0.5        # one action chunk
SEGMENT_S = 2.0      # the evaluator checks this often
BATTERY_PER_S = 0.04
MANIP_TOOLS = {"pick", "place", "open"}
OUTCOMES = {"success", "stall", "silent_miss", "drop"}


@dataclass
class SkillRun:
    raw: dict
    frames: list = field(default_factory=list)   # {t, arm_ext, gripper_closed, effect} for the 3D viewer
    checks: list = field(default_factory=list)   # {t, status, progress, kind} at each segment boundary
    duration: float = 0.0
    stopped: str | None = None                   # success | stall | ended | timeout


class ScriptedBackend:
    name = "scripted"

    def run(self, world: Warehouse, name: str, args: dict, evaluator, before: dict) -> SkillRun:
        return SkillRun(raw=world.skill(name, args))


@dataclass
class _Episode:
    outcome: str   # success | silent_miss | drop | stall | flail
    T: float       # when a self-terminating episode finishes, in seconds
    ends: bool     # does the policy stop on its own at T?


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _state(tool: str, outcome: str, t: float, T: float, closed0: bool) -> dict:
    """Observable arm / gripper / effect state at time t of an episode."""
    f = min(1.0, t / T) if T > 0 else 1.0
    if outcome in ("stall", "flail"):
        base = 0.55 if outcome == "stall" else 0.3
        ext = min(base, base * t / 3.0) + (0.04 * math.sin(2.1 * t) if t > 3.0 else 0.0)
        eff = 0.1 if (tool == "open" and outcome == "stall") else 0.0
        return {"arm_ext": _clamp(ext), "gripper_closed": closed0, "effect": eff}
    if tool == "pick":
        ext, closed, eff = _clamp(f / 0.35), f >= 0.5, _clamp((f - 0.6) / 0.4)
        if outcome == "silent_miss":  # closes on nothing, "lifts" nothing, retracts as if done
            eff, ext = 0.0, ext * (1 - 0.8 * _clamp((f - 0.75) / 0.25))
        return {"arm_ext": ext, "gripper_closed": closed, "effect": eff}
    if tool == "place":  # success and drop look identical until the object is released
        return {"arm_ext": _clamp(f / 0.4), "gripper_closed": f < 0.7, "effect": _clamp((f - 0.7) / 0.3)}
    return {"arm_ext": _clamp(f / 0.4), "gripper_closed": closed0, "effect": _clamp((f - 0.4) / 0.6)}  # open


class PolicyBackend:
    name = "policy"

    def __init__(self, seed: int = 0, early_stop: bool = True, force: list[str] | None = None):
        self.rng = random.Random(seed * 7919 + 13)
        self.early_stop = early_stop
        self.force = [o for o in (force or []) if o in OUTCOMES]  # outcomes for the next manipulation episodes

    # ---- planning an episode (hidden from the agent and the evaluator) ----
    def _plan(self, world: Warehouse, name: str, args: dict) -> _Episode:
        r = world.robot
        u1, u2, u3 = self.rng.random(), self.rng.random(), self.rng.random()
        if name == "pick":
            o = world.objects[args["object"]]
            feasible = world.pick_ok(o)
            d = dist((r.x, r.y), (o.x, o.y))
            p = 0.80 - 0.15 * max(0.0, d - 1.2) / 0.8 - (0.05 if o.fragile else 0) - (0.10 if o.weight > 3 else 0)
            T = self.rng.uniform(4, 8)
        elif name == "place":
            o = world.objects[args["object"]]
            feasible, p, T = world.place_ok(o, args["target"]), 0.90, self.rng.uniform(3, 6)
        else:
            feasible, p, T = world.open_ok(args["target"]), 0.95, self.rng.uniform(2, 4)
        forced = self.force.pop(0) if self.force else None
        if not feasible:
            outcome = "flail"  # the preconditions are false: the policy moves but nothing can happen
        elif forced:
            outcome = forced
            if (outcome == "silent_miss" and name != "pick") or (outcome == "drop" and name != "place"):
                outcome = "stall"
        elif name == "place" and o.fragile and u3 < 0.25:
            outcome = "drop"
        elif u1 < max(0.2, p):
            outcome = "success"
        else:
            outcome = "silent_miss" if (name == "pick" and u2 >= 0.6) else "stall"
        return _Episode(outcome, T, ends=outcome in ("success", "silent_miss", "drop"))

    @staticmethod
    def _validate(world: Warehouse, name: str, args: dict) -> str | None:
        try:
            if name in ("pick", "place") and args["object"] not in world.objects:
                return f"unknown object '{args['object']}'"
            if name == "place" and args["target"] not in world.containers and args["target"] not in ROOMS:
                return f"unknown target '{args['target']}'"
            if name == "open" and args["target"] not in world.doors and args["target"] not in world.containers:
                return f"unknown target '{args['target']}'"
        except KeyError as e:
            return f"bad arguments: missing {e}"
        return None

    @staticmethod
    def _commit(world: Warehouse, name: str, args: dict, ep: _Episode) -> None:
        if ep.outcome == "success":
            if name == "pick":
                world.commit_pick(world.objects[args["object"]])
            elif name == "place":
                world.commit_place(world.objects[args["object"]], args["target"])
            else:
                world.commit_open(args["target"])
        elif ep.outcome == "drop":
            world.commit_place(world.objects[args["object"]], args["target"], dropped=True)
        # silent_miss / stall / flail: the world is left unchanged

    # ---- running and supervising an episode ----
    def run(self, world: Warehouse, name: str, args: dict, evaluator, before: dict) -> SkillRun:
        if name not in MANIP_TOOLS:
            return ScriptedBackend().run(world, name, args, evaluator, before)
        if (err := self._validate(world, name, args)):
            return SkillRun(raw={"error": err})
        r = world.robot
        ep = self._plan(world, name, args)
        closed0 = r.holding is not None
        tracker = evaluator.tracker(name, args, before)
        r.manip = {"mode": "policy", "arm_ext": 0.0, "gripper_closed": closed0, "effect": 0.0}
        frames = [{"t": 0.0, **{k: r.manip[k] for k in ("arm_ext", "gripper_closed", "effect")}}]
        checks: list[dict] = []
        t, stopped = 0.0, None
        while stopped is None:
            t = round(t + FRAME_S, 3)
            world._spend(BATTERY_PER_S * FRAME_S, FRAME_S)
            r.manip.update(_state(name, ep.outcome, t, ep.T, closed0))
            frames.append({"t": t, **{k: round(r.manip[k], 3) if k != "gripper_closed" else r.manip[k]
                                      for k in ("arm_ext", "gripper_closed", "effect")}})
            ended = ep.ends and t >= ep.T
            if ended:
                self._commit(world, name, args, ep)
            if ended or abs(t / SEGMENT_S - round(t / SEGMENT_S)) < 1e-6:
                status, prog, _reason, kind = tracker.update(world.eval_view(), ended=ended)
                if status == "failure" and kind == "stall" and not self.early_stop:
                    status = "in_progress"  # "wait for the end": ignore stall signals, only the budget stops it
                checks.append({"t": t, "status": status, "progress": round(prog, 2), "kind": kind})
                if status == "success":
                    stopped = "success"
                elif status == "failure":
                    stopped = kind
        world._after_action()
        world.log.append({"t": world.t, "kind": name, "policy": True, "frames": frames, "checks": checks,
                          "ok": stopped == "success", **{k: v for k, v in args.items() if k in ("object", "target")}})
        return SkillRun(raw={"executed": True}, frames=frames, checks=checks, duration=t, stopped=stopped)

