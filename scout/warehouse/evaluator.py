"""Evaluation as exit codes.

The evaluator runs as a post-hook after every physical skill. It is reasoning-blind: its only inputs
are the tool call, the per-tool post-condition, and the before/after observations. It returns a
structured verdict (status, evidence, failure_reason) and never recommends a fix; recovery is the
agent's job.

`ground_truth` is exact (the sim knows the truth). `judge` wraps it with an optional noise model,
using the error shape reported for VLM judges: true failures occasionally reported as successes.
"""
from __future__ import annotations

import random

from scout.warehouse.world import MAX_PAYLOAD, REACH, ROOMS, dist, plan_doors

PHYSICAL = {"navigate_to", "open", "pick", "place", "charge"}

# Online judging of a long-running skill (policy backend). The evaluator checks at segment boundaries
# and answers in_progress / success / failure, inferring progress from observations only.
STALL_CHECKS = 2        # this many consecutive checks without new best progress => stalled
PROGRESS_EPS = 0.05
STEP_BUDGET_S = 24.0    # hard cap on one skill run, in simulated seconds


def progress_of(tool: str, manip: dict) -> float:
    """How far along a manipulation looks, from arm extension, gripper state and the effect on the world."""
    ext, closed, eff = manip.get("arm_ext", 0.0), 1.0 if manip.get("gripper_closed") else 0.0, manip.get("effect", 0.0)
    if tool == "pick":
        return 0.4 * ext + 0.3 * closed + 0.3 * eff
    if tool == "place":
        return 0.5 * ext + 0.5 * eff
    return 0.4 * ext + 0.6 * eff  # open


class SegmentTracker:
    """One per skill run. Fed an observation at every segment boundary."""

    def __init__(self, ev: "Evaluator", tool: str, args: dict, before: dict):
        self.ev, self.tool, self.args, self.before = ev, tool, args, before
        self.best, self.stalled = 0.0, 0

    def update(self, now: dict, ended: bool = False) -> tuple[str, float, str | None, str | None]:
        """-> (status, progress, failure_reason, kind) with kind in {None, 'stall', 'ended', 'timeout'}."""
        ok, _evidence, reason = self.ev.ground_truth(self.tool, self.args, self.before, now)
        prog = progress_of(self.tool, now["robot"].get("manip") or {})
        if ok:
            return "success", 1.0, None, None
        if prog > self.best + PROGRESS_EPS:
            self.best, self.stalled = prog, 0
        else:
            self.stalled += 1
        if ended:
            return "failure", prog, reason, "ended"
        if now["t"] - self.before["t"] >= STEP_BUDGET_S:
            return "failure", prog, f"{reason} (no completion within the {STEP_BUDGET_S:.0f}s step budget)", "timeout"
        if self.stalled >= STALL_CHECKS:
            return "failure", prog, reason, "stall"
        return "in_progress", prog, None, None


class Evaluator:
    def __init__(self, false_success_rate: float = 0.0, false_failure_rate: float = 0.0, seed: int = 1):
        self.fs, self.ff = false_success_rate, false_failure_rate
        self.rng = random.Random(seed)

    def tracker(self, tool: str, args: dict, before: dict) -> SegmentTracker:
        return SegmentTracker(self, tool, args, before)

    # ---- exact post-condition check ----
    def ground_truth(self, tool: str, args: dict, before: dict, after: dict) -> tuple[bool, list[str], str | None]:
        return getattr(self, f"_pc_{tool}")(args, before, after)

    def judge(self, tool: str, args: dict, before: dict, after: dict) -> tuple[dict, bool]:
        ok, evidence, reason = self.ground_truth(tool, args, before, after)
        truth = ok
        if not ok and self.rng.random() < self.fs:
            ok, evidence, reason = True, ["post-condition appears satisfied"], None
        elif ok and self.rng.random() < self.ff:
            ok, evidence, reason = False, ["could not confirm the post-condition"], "outcome could not be confirmed"
        verdict = {"status": "success" if ok else "failure", "evidence": evidence, "failure_reason": reason}
        return verdict, truth

    # ---- helpers on eval views ----
    @staticmethod
    def _loc(view: dict, target: str):
        r = view["robot"]
        if target in ROOMS:
            return target, None
        if target in view["containers"]:
            c = view["containers"][target]
            return c["room"], (c["x"], c["y"])
        for d in view["doors"]:
            if d["id"] == target:
                return (r["room"] if r["room"] in (d["a"], d["b"]) else d["a"]), (d["x"], d["y"])
        if target in view["objects"]:
            o = view["objects"][target]
            return (r["room"], (r["x"], r["y"])) if o["held"] else (o["room"], (o["x"], o["y"]))
        return None, None

    # ---- post-conditions: (satisfied, evidence, failure_reason) ----
    def _pc_navigate_to(self, args, before, after):
        r, target = after["robot"], args["target"]
        room, pos = self._loc(after, target)
        in_room = r["room"] == room
        if in_room and (pos is None or dist((r["x"], r["y"]), pos) <= REACH):
            return True, [f"robot is in {r['room']} at the target"], None
        if r["battery"] <= 0:
            return False, ["battery is at 0%"], "battery depleted"
        for d in plan_doors(after["doors"], r["room"], room) or []:
            if d["state"] != "open":
                why = f"stopped before {d['id']} ({d['a']}<->{d['b']}) which is {d['state']}"
                return False, [f"robot is in {r['room']}, target is in {room}", why], why
        return False, [f"robot is in {r['room']}"], "did not reach the target"

    def _pc_open(self, args, before, after):
        t = args["target"]
        d = next((d for d in after["doors"] if d["id"] == t), None)
        st = d["state"] if d else after["containers"][t]["state"]
        if st == "open":
            return True, [f"{t} is open"], None
        room, pos = self._loc(after, t)
        r = after["robot"]
        far = pos is not None and (r["room"] != room and not (d and r["room"] in (d["a"], d["b"])) or dist((r["x"], r["y"]), pos) > REACH)
        if far:
            return False, [f"{t} is still {st}"], f"robot is too far from {t} (reach is {REACH}m)"
        if st == "locked":
            return False, [f"{t} is still locked"], f"{t} is locked and the robot does not hold its key"
        if (after["robot"].get("manip") or {}).get("mode") == "policy":
            return False, [f"{t} is still {st}"], f"policy stalled: the arm moved but {t} did not open"
        return False, [f"{t} is still {st}"], f"{t} did not open"

    def _pc_pick(self, args, before, after):
        oid, r = args["object"], after["robot"]
        o = after["objects"][oid]
        if r["holding"] == oid:
            return True, [f"gripper holds {oid}"], None
        ev = [f"{oid} is not in the gripper"]
        if before["robot"]["holding"] and before["robot"]["holding"] != oid:
            return False, ev, f"gripper is already holding {before['robot']['holding']}"
        if o["weight"] > MAX_PAYLOAD:
            return False, ev, f"{oid} is too heavy ({o['weight']}kg, max payload {MAX_PAYLOAD}kg)"
        cont = after["containers"].get(o["container"]) if o["container"] else None
        if cont and cont["state"] == "closed":
            return False, ev, f"{oid} is inside {cont['id']} which is closed"
        room, pos = self._loc(after, oid)
        if r["room"] != room:
            return False, ev, f"{oid} is in {room}, robot is in {r['room']}"
        d = dist((r["x"], r["y"]), pos)
        if d > REACH:
            return False, ev + [f"robot is {d:.1f}m from {oid}"], f"robot is too far from {oid} (reach is {REACH}m)"
        m = r.get("manip") or {}
        if m.get("mode") == "policy" and not m.get("gripper_closed"):
            return (False, ev + [f"arm extended {m['arm_ext']:.0%} but the gripper never closed"],
                    "policy stalled near the object: the arm hovered but the gripper never closed")
        return False, ev + ["gripper closed without lifting"], "grasp slipped: the gripper closed but the object was not lifted"

    def _pc_place(self, args, before, after):
        oid, tgt, r = args["object"], args["target"], after["robot"]
        o = after["objects"][oid]
        at_target = (not o["held"]) and ((o["container"] == tgt) or (tgt in ROOMS and o["room"] == tgt and o["container"] is None))
        if at_target and not o["broken"]:
            return True, [f"{oid} is at {tgt}"], None
        if o["broken"]:
            return False, [f"{oid} is on the floor and damaged"], f"{oid} was dropped and is broken"
        ev = [f"{oid} is not at {tgt}"]
        if before["robot"]["holding"] != oid:
            return False, ev, f"robot was not holding {oid}"
        if tgt in after["containers"]:
            c = after["containers"][tgt]
            if c["state"] == "closed":
                return False, ev, f"{tgt} is closed"
            if r["room"] != c["room"]:
                return False, ev, f"robot is in {r['room']} but {tgt} is in {c['room']}"
            if dist((r["x"], r["y"]), (c["x"], c["y"])) > REACH:
                return False, ev, f"robot is too far from {tgt} (reach is {REACH}m)"
        elif r["room"] != tgt:
            return False, ev, f"robot is in {r['room']}, not {tgt}"
        m = r.get("manip") or {}
        if m.get("mode") == "policy":
            return False, ev + ["the gripper never released"], f"policy stalled over {tgt}: the object was never released"
        return False, ev, "placement did not complete"

    def _pc_charge(self, args, before, after):
        if after["robot"]["battery"] >= 99:
            return True, ["battery is full"], None
        return False, [f"battery is {after['robot']['battery']:.0f}%"], "robot is not at the charger (charger is in the charging room)"
