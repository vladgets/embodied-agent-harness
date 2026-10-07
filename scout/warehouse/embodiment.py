"""Tool registry + the execution pipeline: skill -> evaluator post-hook -> gated scene-graph update.

`use_evaluator` / `use_scene_graph` exist for ablations. With the evaluator off the agent gets no
verdict and the graph optimistically assumes every executed skill worked (open loop).
"""
from __future__ import annotations

from typing import Callable

from scout.warehouse.evaluator import PHYSICAL, Evaluator
from scout.warehouse.scene_graph import SceneGraph
from scout.warehouse.world import Warehouse

_S = {"type": "string"}

TOOLS = [
    {"name": "navigate_to",
     "description": "Drive to a target: a room name, or an object/container/door ref. Stops ~1.2m short of "
                    "entities. Precondition: closed doors on the route must already be open, otherwise the "
                    "robot stops in front of the first one. Costs battery by distance. Reliable, but "
                    "returns no success signal; read the verdict.",
     "input_schema": {"type": "object", "properties": {"target": _S}, "required": ["target"]}},
    {"name": "open",
     "description": "Open a door or container. Precondition: within 2m of it. Locked doors open only if the "
                    "robot is holding the matching key object. Reliable. Contents of a container are only "
                    "visible once it is open.",
     "input_schema": {"type": "object", "properties": {"target": _S}, "required": ["target"]}},
    {"name": "pick",
     "description": "Pick up one object. Preconditions: within 2m of it, gripper empty, object not inside a "
                    "closed container, object under 5kg. The grasp is imperfect: it can fail even when "
                    "preconditions hold (about 1 in 4). Check the verdict and retry, possibly after "
                    "repositioning.",
     "input_schema": {"type": "object", "properties": {"object": _S}, "required": ["object"]}},
    {"name": "place",
     "description": "Place the held object onto a container (shelf, bin, table, cabinet) or onto the floor "
                    "of the current room (target = room name). Preconditions: the object is held, the "
                    "container is open and within 2m. Fragile objects can be dropped and broken (roughly "
                    "1 in 3 placements onto a container); there is no undo.",
     "input_schema": {"type": "object", "properties": {"object": _S, "target": _S}, "required": ["object", "target"]}},
    {"name": "charge",
     "description": "Recharge to 100%. Only works at the charger in the charging room (navigate there first).",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "look",
     "description": "Scan the current room and refresh the scene graph. Use after entering a room, or to "
                    "re-check something that may have changed. Contents of closed containers are not visible.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "query_user",
     "description": "Ask the human a question. Use when the instruction is ambiguous, an item is unavailable, "
                    "or the task cannot safely be done as stated. Do not guess in those cases.",
     "input_schema": {"type": "object", "properties": {"question": _S}, "required": ["question"]}},
    {"name": "update_plan",
     "description": "Record your current plan as an ordered checklist. Call first for any multi-step task and "
                    "update as you learn. Costs nothing and does not move the robot.",
     "input_schema": {"type": "object", "properties": {"steps": {"type": "array", "items": {
         "type": "object", "properties": {"text": _S, "status": {"type": "string", "enum": ["todo", "doing", "done", "blocked"]}},
         "required": ["text", "status"]}}}, "required": ["steps"]}},
]

# Every physical tool accepts an optional `step`: the 1-based plan step the action serves. The harness
# advances the plan checklist from it, so progress shows up without the model spending a turn on update_plan.
_STEP = {"type": "integer", "description": "Optional: 1-based index of the plan step this action serves."}
for _t in TOOLS:
    if _t["name"] in PHYSICAL:
        _t["input_schema"]["properties"]["step"] = _STEP

UserReply = Callable[[str], str]


class Embodiment:
    def __init__(self, world: Warehouse, evaluator: Evaluator | None = None, *, use_evaluator: bool = True,
                 use_scene_graph: bool = True, user_reply: UserReply | None = None):
        self.world = world
        self.evaluator = evaluator or Evaluator()
        self.graph = SceneGraph()
        self.use_evaluator, self.use_scene_graph = use_evaluator, use_scene_graph
        self.user_reply = user_reply or (lambda q: "No preference; use your best judgment.")
        self.plan: list[dict] = []
        self.plan_dirty = False
        self.stats = {"physical": 0, "true_failures": 0, "false_successes": 0, "false_failures": 0}
        self.graph.update_from_view(world.view())

    def _advance_plan(self, step, ok: bool) -> None:
        """Action for plan step k: earlier steps are done, step k is doing (or blocked if its verdict failed)."""
        if not self.plan or not isinstance(step, int) or not 1 <= step <= len(self.plan):
            return
        for j in range(step - 1):
            self.plan[j]["status"] = "done"
        self.plan[step - 1]["status"] = "doing" if ok else "blocked"
        self.plan_dirty = True

    def close_plan(self) -> None:
        """Run ended normally: whatever was in progress is finished; blocked steps stay blocked."""
        for s in self.plan:
            if s.get("status") == "doing":
                s["status"] = "done"
        self.plan_dirty = True

    def brief(self) -> str:
        if self.use_scene_graph:
            return self.graph.brief()
        r = self.world.view()["robot"]
        return f"ROBOT: room={self.world.robot.room} battery={r['battery']:.0f}% holding={self.world.robot.holding or 'nothing'}\n(no persistent scene graph: only what look() returns)"

    def execute(self, name: str, args: dict) -> dict:
        if name == "update_plan":
            self.plan = [dict(s) for s in args.get("steps", [])]
            self.plan_dirty = True
            return {"ok": True, "plan_recorded": len(self.plan)}
        if name == "query_user":
            return {"ok": True, "user_reply": self.user_reply(args.get("question", ""))}
        if name == "look":
            self.graph.update_from_view(self.world.view())
            v = self.world.view()
            return {"ok": True, "room": v["room"], "visible_objects": [o["id"] for o in v["objects"]]}
        if name not in PHYSICAL:
            return {"ok": False, "error": f"unknown tool {name}"}

        args = dict(args)
        step = args.pop("step", None)
        before = self.world.eval_view()
        raw = self.world.skill(name, args)
        if "error" in raw:
            return {"ok": False, "error": raw["error"]}
        after = self.world.eval_view()

        truth_ok, _, _ = self.evaluator.ground_truth(name, args, before, after)
        if self.use_evaluator:
            verdict, _ = self.evaluator.judge(name, args, before, after)
            believed_ok = verdict["status"] == "success"
        else:
            verdict, believed_ok = {"status": "unknown", "evidence": [], "failure_reason": None}, True
        s = self.stats
        s["physical"] += 1
        s["true_failures"] += not truth_ok
        s["false_successes"] += (not truth_ok) and believed_ok
        s["false_failures"] += truth_ok and not believed_ok

        if self.use_evaluator:
            self._advance_plan(step, believed_ok)
        else:
            self._advance_plan(step, True)  # no verdicts: progress is only what the agent claims
        if believed_ok:
            self.graph.update_from_outcome(name, args)
        self.graph.update_from_view(self.world.view())  # perception refresh runs every turn
        return {"ok": True, "executed": True, "verdict": verdict, "_truth_ok": truth_ok}
