"""Agent loop for the warehouse: one tool call per turn, scene-graph brief refreshed every turn.

Planners:
  * BaselinePlanner - deterministic rule-based agent; reads the same brief/verdicts. It is the offline
                      stand-in and the reference the model-backed agents are compared against, not an
                      NL parser. Model-backed planners (Claude, OpenAI) live in planners.py.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from scout.harness.trace import Trace
from scout.warehouse.embodiment import TOOLS, Embodiment
from scout.warehouse.evaluator import PHYSICAL
from scout.warehouse.scene_graph import SceneGraph
from scout.warehouse.world import REACH, ROOMS, dist, plan_doors

SYSTEM = """You control one mobile manipulator robot in a warehouse, following a human's instruction.

How the harness works:
- Each turn you issue exactly ONE tool call, then see the result. Physical outcomes are uncertain, so decide the next step from what actually happened.
- Physical skills return only "executed". The real outcome is the `verdict` from an independent evaluator: status, evidence, failure_reason. Treat failure_reason as the diagnosis and choose the recovery yourself (reposition, open something, retry, change approach). The evaluator can occasionally be wrong; if the scene graph contradicts a verdict, trust fresh observations.
- After every call you get a SCENE GRAPH brief: your memory of the world. Entries marked STALE or MISSING may be wrong; use look() or go check. Closed containers hide their contents.
- Rooms are connected by doors that may be closed or locked. Keys are ordinary objects. The gripper holds one object at a time.
- Call update_plan first for any multi-step task (it is shown to the operator). Then put the optional `step` argument (the 1-based plan step) on each physical action, which advances the operator's progress display without costing you a turn. Call update_plan again only if the plan itself changes.
- If the instruction is ambiguous, an item does not exist, or the task cannot be done safely, call query_user instead of guessing.
- When the task is complete (or you have concluded it cannot be), reply with a short plain-text report and no tool call."""

Emit = Callable[[dict], Awaitable[None]]


@dataclass
class Call:
    id: str
    name: str
    input: dict


class BaselinePlanner:
    """Goal-directed reactive agent. `goals` = [(object selector, destination ref)]."""
    name = "baseline"

    def __init__(self, goals: list[tuple[dict, str]], max_retries: int = 4):
        self.goals = goals
        self.max_retries = max_retries
        self.fails = 0
        self.graph: SceneGraph | None = None

    async def start(self, instruction: str, graph: SceneGraph):
        self.graph = graph
        self.fails = 0
        self.want_key: str | None = None

    # -- helpers on the (shared) scene graph --
    def _near(self, x, y) -> bool:
        r = self.graph.robot
        return dist((r["x"], r["y"]), (x, y)) <= REACH

    def _pos_of(self, ref: str):
        g = self.graph
        if ref in g.objects:
            n = g.objects[ref]
            return n.room, n.x, n.y
        if ref in g.containers:
            c = g.containers[ref]
            return c["room"], c["x"], c["y"]
        if ref in g.doors:
            d = g.doors[ref]
            return g.robot["room"], d["x"], d["y"]
        return None, None, None

    def _go(self, ref: str) -> Call:
        return Call("b", "navigate_to", {"target": ref})

    def _reach(self, ref: str, then: Call) -> Call:
        room, x, y = self._pos_of(ref)
        if room != self.graph.robot["room"] or not self._near(x, y):
            return self._go(ref)
        return then

    def _match(self, sel: dict):
        return [n for n in self.graph.find(sel.get("color"), sel.get("kind")) if not n.missing]

    def _done(self, sel: dict, dest: str) -> bool:
        return any(n.container == dest or (dest in ROOMS and n.room == dest and not n.held) for n in self._match(sel))

    async def decide(self, ctx: dict) -> tuple[str, Call | None]:
        g = self.graph
        last = ctx["last"]
        if ctx["turn"] == 0 and self.goals:
            steps = []
            for sel, dest in self.goals:
                what = " ".join(x for x in (sel.get("color"), sel.get("kind")) if x)
                steps += [{"text": f"Pick up the {what}", "status": "todo"},
                          {"text": f"Deliver it to {dest}", "status": "todo"}]
            return "Planning.", Call("b", "update_plan", {"steps": steps})
        if last:
            call, res = last
            v = res.get("verdict") or {}
            if v.get("status") == "failure":
                self.fails += 1
                if self.fails > self.max_retries * 3:
                    return "Too many failures; giving up.", None
                reason = v.get("failure_reason") or ""
                m = re.search(r"before (d_\w+) .* which is (closed|locked)", reason)
                if m:
                    return self._unblock(m.group(1), m.group(2))
            else:
                self.fails = 0
        for idx, (sel, dest) in enumerate(self.goals):
            if self._done(sel, dest):
                continue
            text, call = self._work(sel, dest)
            matches = self._match(sel)
            hold = self.graph.robot["holding"]
            detour = (self.want_key and self.graph.doors.get(self.want_key, {}).get("state") == "locked") \
                or (hold and (not matches or hold != matches[0].ref))
            if call.name in PHYSICAL and not detour:  # detours (fetching a key, clearing the gripper) carry no step
                carrying = bool(matches) and hold == matches[0].ref
                call.input["step"] = 2 * idx + (2 if carrying else 1)
            return text, call
        return "All objects delivered.", None

    def _unblock(self, door: str, state: str) -> tuple[str, Call]:
        if state == "locked":
            self.want_key = door
            return self._get_key_and_open(door)
        return f"Route blocked at {door} (closed).", self._reach(door, Call("b", "open", {"target": door}))

    def _work(self, sel: dict, dest: str) -> tuple[str, Call]:
        g, hold = self.graph, self.graph.robot["holding"]
        matches = self._match(sel)
        target = matches[0] if matches else None
        if self.want_key and g.doors.get(self.want_key, {}).get("state") == "locked":
            return self._get_key_and_open(self.want_key)
        if hold and (target is None or hold != target.ref):
            return f"Putting down {hold}.", Call("b", "place", {"object": hold, "target": g.robot["room"]})
        if target is None:
            return self._explore()
        if hold == target.ref:
            return f"Delivering {target.ref}.", self._reach_dest(target.ref, dest)
        if target.container and g.containers.get(target.container, {}).get("state") == "closed":
            return f"Opening {target.container}.", self._reach(target.container, Call("b", "open", {"target": target.container}))
        return f"Picking {target.ref}.", self._reach(target.ref, Call("b", "pick", {"object": target.ref}))

    def _reach_dest(self, ref: str, dest: str) -> Call:
        g = self.graph
        if dest in g.containers:
            c = g.containers[dest]
            if c["state"] == "closed":
                return self._reach(dest, Call("b", "open", {"target": dest}))
            return self._reach(dest, Call("b", "place", {"object": ref, "target": dest}))
        if g.robot["room"] != dest:
            return self._go(dest)
        return Call("b", "place", {"object": ref, "target": dest})

    def _get_key_and_open(self, door: str) -> tuple[str, Call]:
        g = self.graph
        keys = [n for n in g.find(kind="keycard") if not n.missing]
        hold = g.robot["holding"]
        if hold and hold.startswith("keycard"):
            return f"Using key on {door}.", self._reach(door, Call("b", "open", {"target": door}))
        if hold:
            return "Putting down cargo to fetch the key.", Call("b", "place", {"object": hold, "target": g.robot["room"]})
        if not keys:
            return self._explore()
        k = keys[0]
        if k.container and g.containers.get(k.container, {}).get("state") == "closed":
            return f"Opening {k.container} for the key.", self._reach(k.container, Call("b", "open", {"target": k.container}))
        return f"Fetching {k.ref}.", self._reach(k.ref, Call("b", "pick", {"object": k.ref}))

    def _explore(self) -> tuple[str, Call]:
        g = self.graph
        for room in ROOMS:
            if room in g.visited:
                continue
            route = plan_doors(list(g.doors.values()), g.robot["room"], room)
            if route is None or any(d["state"] == "locked" for d in route):
                continue  # known to be behind a locked door; look elsewhere first
            return f"Exploring {room}.", self._go(room)
        for c in g.containers.values():
            if c["state"] == "closed":
                return f"Searching {c['id']}.", self._reach(c["id"], Call("b", "open", {"target": c["id"]}))
        return "Nothing left to search.", Call("b", "look", {})


async def run_task(instruction: str, emb: Embodiment, planner, trace: Trace, emit: Emit, *, max_turns: int = 60) -> dict:
    await planner.start(instruction, emb.graph)
    trace.log("task", instruction=instruction, planner=planner.name)
    await emit({"type": "task", "text": instruction, "planner": planner.name})
    summary = {"turns": 0, "final": None, "calls": [], "asked_user": False}
    last = None
    for turn in range(max_turns):
        ctx = {"turn": turn, "brief": emb.brief(), "last": last}
        text, call = await planner.decide(ctx)
        if text:
            trace.log("thought", turn=turn, text=text)
            await emit({"type": "thought", "text": text})
        if call is None:
            summary["final"] = text
            emb.close_plan()
            await emit({"type": "plan", "steps": emb.plan})
            break
        result = emb.execute(call.name, call.input)
        public = {k: v for k, v in result.items() if not k.startswith("_")}
        trace.log("tool", turn=turn, name=call.name, input=call.input, result=public,
                  truth_ok=result.get("_truth_ok"), brief=emb.brief())
        await emit({"type": "tool", "turn": turn, "name": call.name, "input": call.input, "result": public})
        if emb.plan_dirty:
            emb.plan_dirty = False
            await emit({"type": "plan", "steps": emb.plan})
        await emit({"type": "scene", "brief": emb.brief(), "t": emb.world.t})
        summary["calls"].append({"name": call.name, "input": call.input, "result": public, "truth_ok": result.get("_truth_ok")})
        summary["asked_user"] |= call.name == "query_user"
        last = (call, public)
        summary["turns"] = turn + 1
        await asyncio.sleep(0)
    else:
        summary["final"] = "(turn budget exhausted)"
    await emit({"type": "done", "final": summary["final"]})
    return summary
