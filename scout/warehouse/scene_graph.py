"""Persistent symbolic world state, the agent's memory of the warehouse.

Two kinds of update reach it:
  * observation-derived: what the robot sees in its current room (`update_from_view`)
  * execution-derived: what an action changed (`update_from_outcome`), applied ONLY when the
    evaluator confirmed the outcome. A wrong verdict therefore corrupts the graph, which is the
    point of making evaluator accuracy a measurable knob.
"""
from __future__ import annotations

from dataclasses import dataclass

from scout.warehouse.world import ROOMS

STALE_AFTER_S = 60.0


@dataclass
class ObjNode:
    ref: str
    kind: str
    color: str
    weight: float
    fragile: bool
    room: str | None
    x: float | None
    y: float | None
    container: str | None
    last_seen: float
    missing: bool = False
    held: bool = False


class SceneGraph:
    def __init__(self):
        self.t = 0.0
        self.robot = {"room": "storage", "x": 0.0, "y": 0.0, "battery": 100.0, "holding": None}
        self.objects: dict[str, ObjNode] = {}
        self.containers: dict[str, dict] = {}
        self.doors: dict[str, dict] = {}
        self.visited: set[str] = set()

    # ---- observation-derived ----
    def update_from_view(self, view: dict) -> None:
        self.t = view["t"]
        room = view["room"]
        self.visited.add(room)
        self.robot.update(room=room, **view["robot"])
        for c in view["containers"]:
            self.containers[c["id"]] = {k: c[k] for k in ("id", "kind", "room", "x", "y", "state")}
        for d in view["doors"]:
            self.doors[d["id"]] = {k: d[k] for k in ("id", "a", "b", "x", "y", "state")}
        seen = set()
        for o in view["objects"]:
            seen.add(o["id"])
            self.objects[o["id"]] = ObjNode(
                ref=o["id"], kind=o["kind"], color=o["color"], weight=o["weight"], fragile=o["fragile"],
                room=room, x=o["x"], y=o["y"], container=o["container"], last_seen=self.t)
            if self.robot["holding"] == o["id"]:  # seen out in the world: the belief was wrong
                self.robot["holding"] = None
        for n in self.objects.values():
            if n.room == room and not n.held and not n.missing and n.ref not in seen:
                cont = self.containers.get(n.container) if n.container else None
                if cont and cont["state"] == "closed":
                    continue  # still plausibly inside, just not visible
                n.missing = True

    # ---- execution-derived (gated on a confirmed outcome) ----
    def update_from_outcome(self, tool: str, args: dict) -> None:
        if tool == "pick":
            n = self.objects.get(args["object"])
            if n:
                n.held, n.room, n.container, n.missing = True, None, None, False
                self.robot["holding"] = n.ref
        elif tool == "place":
            n = self.objects.get(args["object"])
            if n:
                n.held, n.missing = False, False
                self.robot["holding"] = None
                tgt = args["target"]
                if tgt in self.containers:
                    c = self.containers[tgt]
                    n.room, n.container, n.x, n.y = c["room"], tgt, c["x"], c["y"]
                else:
                    n.room, n.container, n.x, n.y = self.robot["room"], None, self.robot["x"], self.robot["y"]
                n.last_seen = self.t
        elif tool == "open":
            t = args["target"]
            if t in self.doors:
                self.doors[t]["state"] = "open"
            elif t in self.containers:
                self.containers[t]["state"] = "open"
        elif tool == "charge":
            self.robot["battery"] = 100.0

    # ---- queries ----
    def find(self, color: str | None = None, kind: str | None = None) -> list[ObjNode]:
        return [n for n in self.objects.values()
                if (color is None or n.color == color) and (kind is None or n.kind == kind)]

    def age(self, n: ObjNode) -> float:
        return self.t - n.last_seen

    def brief(self) -> str:
        r = self.robot
        hold = r["holding"] or "nothing"
        lines = [f"ROBOT: room={r['room']} pos=({r['x']:.0f},{r['y']:.0f}) battery={r['battery']:.0f}% holding={hold}  [t={self.t:.0f}s]"]
        unvisited = [x for x in ROOMS if x not in self.visited]
        lines.append(f"ROOMS visited: {', '.join(sorted(self.visited)) or 'none'}; never seen: {', '.join(unvisited) or 'none'}")
        if self.doors:
            lines.append("DOORS: " + "; ".join(f"{d['id']} {d['a']}<->{d['b']} {d['state']}" for d in self.doors.values()))
        if self.containers:
            lines.append("CONTAINERS: " + "; ".join(
                f"{c['id']}({c['kind']}) in {c['room']} {c['state']}" for c in self.containers.values()))
        lines.append("OBJECTS:")
        if not self.objects:
            lines.append("  (none seen yet)")
        for n in self.objects.values():
            if n.held:
                lines.append(f"  {n.ref} ({n.color} {n.kind}) HELD by robot")
                continue
            where = f"in {n.container}" if n.container else f"on floor of {n.room}"
            if n.container:
                where += f" ({n.room})"
            if n.missing:
                lines.append(f"  {n.ref} ({n.color} {n.kind}) MISSING: last seen {where} {self.age(n):.0f}s ago, no longer there")
                continue
            age = self.age(n)
            tag = f"seen {age:.0f}s ago" + (" STALE" if age > STALE_AFTER_S else "")
            extra = ("heavy " if n.weight > 5 else "") + ("fragile" if n.fragile else "")
            lines.append(f"  {n.ref} ({n.color} {n.kind}{', ' + extra.strip() if extra.strip() else ''}) {where} @({n.x:.0f},{n.y:.0f}) {tag}")
        return "\n".join(lines)
