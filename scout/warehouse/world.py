"""Ground-truth warehouse. The agent never reads this directly.

Skills return only `{"executed": True}`: like a VLA policy, the controller has no "done" signal and
does not know whether the grasp worked. Outcomes are judged separately by the Evaluator, which reads
`eval_view()`. The agent's own perception comes from `view()` (current room only), via the scene graph.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable

# room -> (x0, y0, x1, y1)
ROOMS = {
    "storage": (0, 0, 20, 14), "packing": (20, 0, 40, 14), "dock": (40, 0, 60, 14),
    "office": (0, 14, 20, 28), "charging": (20, 14, 40, 28),
}
CHARGER = ("charging", 30.0, 25.0)
REACH = 2.0
STOP = 1.2
MAX_PAYLOAD = 5.0
SPEED = 2.0
BATTERY_PER_M = 0.3


@dataclass
class Door:
    id: str
    a: str
    b: str
    x: float
    y: float
    state: str  # open | closed | locked
    key: str | None = None


@dataclass
class Container:
    id: str
    kind: str  # shelf | bin | cabinet | table
    room: str
    x: float
    y: float
    state: str = "open"


@dataclass
class Obj:
    id: str
    kind: str
    color: str
    weight: float
    fragile: bool
    room: str | None
    x: float
    y: float
    container: str | None = None
    held: bool = False
    broken: bool = False


@dataclass
class Robot:
    room: str = "storage"
    x: float = 5.0
    y: float = 5.0
    battery: float = 100.0
    holding: str | None = None


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def plan_doors(doors: list, a: str, b: str) -> list | None:
    """Shortest door sequence from room a to room b, ignoring door state. Works on Door or dict."""
    g = lambda d, k: d[k] if isinstance(d, dict) else getattr(d, k)  # noqa: E731
    if a == b:
        return []
    prev: dict[str, tuple[str, object]] = {}
    q = deque([a])
    seen = {a}
    while q:
        cur = q.popleft()
        for d in doors:
            nxt = g(d, "b") if g(d, "a") == cur else g(d, "a") if g(d, "b") == cur else None
            if nxt and nxt not in seen:
                seen.add(nxt)
                prev[nxt] = (cur, d)
                if nxt == b:
                    path, node = [], b
                    while node != a:
                        node, d2 = prev[node][0], prev[node][1]
                        path.append(d2)
                    return path[::-1]
                q.append(nxt)
    return None


def approach(frm: tuple[float, float], to: tuple[float, float], stop: float = STOP) -> tuple[float, float]:
    d = dist(frm, to)
    if d <= stop:
        return frm
    t = (d - stop) / d
    return frm[0] + (to[0] - frm[0]) * t, frm[1] + (to[1] - frm[1]) * t


class Warehouse:
    def __init__(self, seed: int = 0, slip_p: float = 0.25, drop_p: float = 0.3):
        self.rng = random.Random(seed)
        self.slip_p, self.drop_p = slip_p, drop_p
        self.t = 0.0
        self.action_count = 0
        self.perturbations: list[tuple[int, str, Callable[["Warehouse"], None]]] = []
        self.log: list[dict] = []  # animation/replay events for the viewer
        self.robot = Robot()
        self.doors = {d.id: d for d in [
            Door("d_sp", "storage", "packing", 20, 7, "closed"),
            Door("d_pd", "packing", "dock", 40, 7, "locked", key="keycard_5"),
            Door("d_so", "storage", "office", 10, 14, "open"),
            Door("d_pc", "packing", "charging", 30, 14, "closed"),
            Door("d_oc", "office", "charging", 20, 21, "closed"),
        ]}
        self.containers = {c.id: c for c in [
            Container("shelf_1", "shelf", "storage", 3, 2),
            Container("bin_1", "bin", "storage", 17, 12, "closed"),
            Container("cabinet_1", "cabinet", "office", 3, 26, "closed"),
            Container("table_pack", "table", "packing", 35, 3),
            Container("table_dock", "table", "dock", 55, 10),
            Container("table_office", "table", "office", 15, 25),
        ]}
        self.objects = {o.id: o for o in [
            Obj("box_1", "box", "red", 1.5, False, "storage", 3, 2, "shelf_1"),
            Obj("box_2", "box", "blue", 2.0, False, "storage", 12, 6),
            Obj("crate_3", "crate", "yellow", 8.0, False, "packing", 25, 10),
            Obj("vase_4", "vase", "white", 0.8, True, "storage", 17, 12, "bin_1"),
            Obj("keycard_5", "keycard", "black", 0.1, False, "office", 3, 26, "cabinet_1"),
            Obj("wrench_6", "wrench", "gray", 1.0, False, "storage", 17, 12, "bin_1"),
            Obj("box_7", "box", "green", 1.0, False, "packing", 28, 10),
        ]}

    # ---- lookup ----
    def locate(self, target: str) -> tuple[str, float, float] | None:
        if target in ROOMS:
            x0, y0, x1, y1 = ROOMS[target]
            return target, (x0 + x1) / 2, (y0 + y1) / 2
        if target in self.containers:
            c = self.containers[target]
            return c.room, c.x, c.y
        if target in self.doors:
            d = self.doors[target]
            return self.robot.room if self.robot.room in (d.a, d.b) else d.a, d.x, d.y
        if target in self.objects:
            o = self.objects[target]
            return (self.robot.room, self.robot.x, self.robot.y) if o.held else (o.room, o.x, o.y)
        return None

    def _spend(self, battery: float, seconds: float) -> None:
        self.robot.battery = max(0.0, self.robot.battery - battery)
        self.t += seconds

    def _after_action(self) -> None:
        self.action_count += 1
        for n, _desc, fn in list(self.perturbations):
            if n == self.action_count:
                fn(self)

    def _pos(self) -> tuple[float, float]:
        return self.robot.x, self.robot.y

    def _near(self, x: float, y: float) -> bool:
        return dist(self._pos(), (x, y)) <= REACH

    # ---- skills (raw execution; no success semantics) ----
    def skill(self, name: str, args: dict) -> dict:
        fn = getattr(self, f"_s_{name}", None)
        if fn is None:
            return {"error": f"unknown skill {name}"}
        try:
            out = fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
        if "error" not in out:
            self._after_action()
        return out

    def _s_navigate_to(self, target: str) -> dict:
        loc = self.locate(target)
        if loc is None:
            return {"error": f"unknown target '{target}'"}
        r = self.robot
        if r.battery <= 0:
            return {"executed": True}
        troom, tx, ty = loc
        pos, total, path = self._pos(), 0.0, [self._pos()]
        blocked = False
        for d in plan_doors(list(self.doors.values()), r.room, troom) or []:
            if d.state != "open":
                stop = approach(pos, (d.x, d.y))
                total += dist(pos, stop)
                pos = stop
                path.append(pos)
                blocked = True
                break
            total += dist(pos, (d.x, d.y))
            pos = (d.x, d.y)
            path.append(pos)
            r.room = d.b if r.room == d.a else d.a
        if not blocked:
            final = (tx, ty) if target in ROOMS else approach(pos, (tx, ty))
            total += dist(pos, final)
            pos = final
            path.append(pos)
        r.x, r.y = pos
        cost = total * BATTERY_PER_M
        self._spend(cost, total / SPEED)
        self.log.append({"t": self.t, "kind": "navigate", "path": path, "room": r.room})
        return {"executed": True}

    def _s_open(self, target: str) -> dict:
        if target not in self.doors and target not in self.containers:
            return {"error": f"unknown target '{target}'"}
        self._spend(0.3, 2.0)
        r = self.robot
        if r.battery <= 0:
            return {"executed": True}
        if target in self.doors:
            d = self.doors[target]
            if r.room in (d.a, d.b) and self._near(d.x, d.y):
                if d.state == "closed" or (d.state == "locked" and r.holding == d.key):
                    d.state = "open"
        else:
            c = self.containers[target]
            if r.room == c.room and self._near(c.x, c.y) and c.state == "closed":
                c.state = "open"
        self.log.append({"t": self.t, "kind": "open", "target": target})
        return {"executed": True}

    def _s_pick(self, object: str) -> dict:  # noqa: A002
        o = self.objects.get(object)
        if o is None:
            return {"error": f"unknown object '{object}'"}
        self._spend(0.5, 4.0)
        r = self.robot
        cont = self.containers.get(o.container) if o.container else None
        ok = (r.battery > 0 and r.holding is None and not o.held and r.room == o.room
              and self._near(o.x, o.y) and o.weight <= MAX_PAYLOAD
              and (cont is None or cont.state != "closed"))
        slipped = ok and self.rng.random() < self.slip_p
        if ok and not slipped:
            r.holding, o.held, o.room, o.container = o.id, True, None, None
        self.log.append({"t": self.t, "kind": "pick", "object": object, "ok": ok and not slipped})
        return {"executed": True}

    def _s_place(self, object: str, target: str) -> dict:  # noqa: A002
        if object not in self.objects:
            return {"error": f"unknown object '{object}'"}
        if target not in ROOMS and target not in self.containers:
            return {"error": f"unknown target '{target}'"}
        self._spend(0.5, 3.0)
        r, o = self.robot, self.objects[object]
        if r.battery <= 0 or r.holding != object:
            return {"executed": True}
        if target in self.containers:
            c = self.containers[target]
            if not (r.room == c.room and self._near(c.x, c.y) and c.state != "closed"):
                return {"executed": True}
            if o.fragile and self.rng.random() < self.drop_p:
                o.broken, o.room, o.x, o.y, o.container = True, r.room, r.x, r.y, None
            else:
                o.room, o.x, o.y, o.container = c.room, c.x, c.y, c.id
        else:
            if r.room != target:
                return {"executed": True}
            o.room, o.x, o.y, o.container = r.room, r.x, r.y, None
        o.held, r.holding = False, None
        self.log.append({"t": self.t, "kind": "place", "object": object, "target": target})
        return {"executed": True}

    def _s_charge(self) -> dict:
        self._spend(0.0, 20.0)
        r = self.robot
        if r.room == CHARGER[0] and dist(self._pos(), CHARGER[1:]) <= 3.0:
            r.battery = 100.0
        self.log.append({"t": self.t, "kind": "charge"})
        return {"executed": True}

    # ---- observation ----
    def view(self) -> dict:
        """What the robot's own perception sees: its current room only, nothing inside closed containers."""
        r = self.robot
        return {
            "t": round(self.t, 1), "room": r.room,
            "robot": {"x": round(r.x, 1), "y": round(r.y, 1), "battery": round(r.battery, 1)},
            "objects": [
                {"id": o.id, "kind": o.kind, "color": o.color, "weight": o.weight, "fragile": o.fragile,
                 "x": o.x, "y": o.y, "container": o.container}
                for o in self.objects.values()
                if o.room == r.room and not o.held
                and (o.container is None or self.containers[o.container].state != "closed")
            ],
            "containers": [vars(c).copy() for c in self.containers.values() if c.room == r.room],
            "doors": [vars(d).copy() | {"key": None} for d in self.doors.values() if r.room in (d.a, d.b)],
        }

    def eval_view(self) -> dict:
        """Simulated sensor suite for the evaluator: full state, including what the gripper holds."""
        r = self.robot
        return {
            "t": round(self.t, 1),
            "robot": {"room": r.room, "x": r.x, "y": r.y, "battery": r.battery, "holding": r.holding},
            "objects": {o.id: vars(o).copy() for o in self.objects.values()},
            "containers": {c.id: vars(c).copy() for c in self.containers.values()},
            "doors": [vars(d).copy() | {"key": None} for d in self.doors.values()],
        }

    # ---- scenario helpers ----
    def add_perturbation(self, after_actions: int, description: str, fn: Callable[["Warehouse"], None]) -> None:
        self.perturbations.append((after_actions, description, fn))
