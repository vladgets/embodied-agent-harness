"""Tool schemas + the connector that turns model output into validated robot commands.

The agent never touches the World. Every action passes `Connector.execute`, which enforces
safety limits and returns a structured result the model can reason about.
"""
from __future__ import annotations

import math

from scout.sim.world import HOME, NO_GO, WORLD_SIZE, World

MIN_BATTERY_TO_MOVE = 15.0

TOOLS = [
    {
        "name": "list_robots",
        "description": "List all robots with position, mode, battery and comms status.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "move_to",
        "description": "Command one robot to move to (x, y). World spans -50..50 on both axes. "
                       "Rejected if the point is out of bounds, inside a no-go zone, the robot "
                       "has no comms, or its battery is too low.",
        "input_schema": {
            "type": "object",
            "properties": {"robot_id": {"type": "string"}, "x": {"type": "number"}, "y": {"type": "number"}},
            "required": ["robot_id", "x", "y"],
        },
    },
    {
        "name": "hold",
        "description": "Stop a robot and hold position (overwatch).",
        "input_schema": {"type": "object", "properties": {"robot_id": {"type": "string"}}, "required": ["robot_id"]},
    },
    {
        "name": "return_home",
        "description": "Send a robot back to the home base.",
        "input_schema": {"type": "object", "properties": {"robot_id": {"type": "string"}}, "required": ["robot_id"]},
    },
    {
        "name": "observe",
        "description": "Return targets currently within a robot's sensor range.",
        "input_schema": {"type": "object", "properties": {"robot_id": {"type": "string"}}, "required": ["robot_id"]},
    },
    {
        "name": "request_clarification",
        "description": "Ask the commander a question instead of acting. Use when the order is "
                       "ambiguous, or unsafe and you cannot find a safe interpretation.",
        "input_schema": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    },
    {
        "name": "report",
        "description": "Send the final status report to the commander. Call this last.",
        "input_schema": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    },
]


class Connector:
    def __init__(self, world: World):
        self.world = world

    def execute(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name}"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"ok": False, "error": f"bad arguments: {e}"}

    # ---- guards ----
    def _robot(self, rid: str):
        r = self.world.robots.get(rid)
        return (r, None) if r else (None, {"ok": False, "error": f"no such robot '{rid}'"})

    @staticmethod
    def check_point(x: float, y: float) -> str | None:
        lim = WORLD_SIZE / 2
        if not (-lim <= x <= lim and -lim <= y <= lim):
            return f"({x}, {y}) is outside the operating area (±{lim})"
        for zx, zy, zr in NO_GO:
            if math.hypot(x - zx, y - zy) <= zr:
                return f"({x}, {y}) is inside a no-go zone centred at ({zx}, {zy}) r={zr}"
        return None

    def _comms_guard(self, r):
        if not r.comms_up:
            return {"ok": False, "error": f"{r.id} has no comms link; command not delivered"}
        return None

    # ---- tools ----
    def _t_list_robots(self) -> dict:
        return {"ok": True, "robots": [
            {k: v for k, v in r.to_dict().items() if k in ("id", "x", "y", "mode", "battery", "comms_up")}
            for r in self.world.robots.values()
        ]}

    def _t_move_to(self, robot_id: str, x: float, y: float) -> dict:
        r, err = self._robot(robot_id)
        if err:
            return err
        if (g := self._comms_guard(r)):
            return g
        if (why := self.check_point(x, y)):
            return {"ok": False, "error": why}
        if r.battery < MIN_BATTERY_TO_MOVE:
            return {"ok": False, "error": f"{r.id} battery {r.battery:.0f}% below {MIN_BATTERY_TO_MOVE:.0f}% minimum"}
        self.world.set_goal(r.id, x, y)
        r.last_cmd = f"move_to({x:.0f},{y:.0f})"
        eta = math.hypot(x - r.x, y - r.y) / r.speed
        return {"ok": True, "robot": r.id, "eta_s": round(eta, 1)}

    def _t_hold(self, robot_id: str) -> dict:
        r, err = self._robot(robot_id)
        if err:
            return err
        if (g := self._comms_guard(r)):
            return g
        self.world.hold(r.id)
        r.last_cmd = "hold"
        return {"ok": True, "robot": r.id}

    def _t_return_home(self, robot_id: str) -> dict:
        r, err = self._robot(robot_id)
        if err:
            return err
        if (g := self._comms_guard(r)):
            return g
        self.world.set_goal(r.id, *HOME, mode="returning")
        r.last_cmd = "return_home"
        return {"ok": True, "robot": r.id}

    def _t_observe(self, robot_id: str) -> dict:
        r, err = self._robot(robot_id)
        if err:
            return err
        return {"ok": True, "robot": r.id, "visible": self.world.visible_to(r.id)}

    def _t_request_clarification(self, question: str) -> dict:
        return {"ok": True, "delivered": question}

    def _t_report(self, summary: str) -> dict:
        return {"ok": True, "delivered": summary}
