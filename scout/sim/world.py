"""Deterministic 2.5D world: terrain, robots, targets, faults.

The sim knows nothing about agents. It exposes `step(dt)`, `snapshot()` and a small
command surface (`set_goal`, `hold`, `inject_fault`) that the connector layer calls.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

WORLD_SIZE = 100.0
HOME = (-40.0, -40.0)

# Circular no-go zones: (x, y, radius)
NO_GO = [(10.0, 15.0, 9.0)]


def terrain_height(x: float, y: float) -> float:
    return (
        3.0 * math.sin(x * 0.09) * math.cos(y * 0.07)
        + 1.6 * math.sin(x * 0.2 + 1.3) * math.sin(y * 0.17)
        + 2.2 * math.exp(-(((x - 25) ** 2 + (y - 30) ** 2) / 260.0)) * 4  # "the ridge"
    )


@dataclass
class Robot:
    id: str
    x: float
    y: float
    speed: float = 6.0  # units / s
    battery: float = 100.0
    sensor_range: float = 18.0
    comms_up: bool = True
    mode: str = "idle"  # idle | moving | holding | returning | lost
    goal: tuple[float, float] | None = None
    heading: float = 0.0
    last_cmd: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "x": round(self.x, 2), "y": round(self.y, 2),
            "z": round(terrain_height(self.x, self.y), 2),
            "heading": round(self.heading, 3), "battery": round(self.battery, 1),
            "comms_up": self.comms_up, "mode": self.mode, "range": self.sensor_range,
            "goal": self.goal, "last_cmd": self.last_cmd,
        }


@dataclass
class Target:
    id: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    detected_by: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "id": self.id, "x": round(self.x, 2), "y": round(self.y, 2),
            "z": round(terrain_height(self.x, self.y), 2),
            "detected": bool(self.detected_by),
        }


class World:
    def __init__(self, seed: int = 7, n_robots: int = 3):
        self.rng = random.Random(seed)
        self.t = 0.0
        self.robots: dict[str, Robot] = {}
        names = ["alpha", "bravo", "charlie", "delta", "echo"]
        for i in range(n_robots):
            r = Robot(id=names[i], x=HOME[0] + i * 4, y=HOME[1] + i * 2)
            self.robots[r.id] = r
        self.targets: dict[str, Target] = {
            "t1": Target("t1", 28, 34, 0.4, 0.1),  # on the ridge
            "t2": Target("t2", 42, -10, -0.3, 0.5),
            "t3": Target("t3", -20, 38, 0.0, -0.4),
        }
        self.explored: set[tuple[int, int]] = set()  # 5x5 cells revealed
        self.events: list[dict] = []

    # ---- commands (called via the connector, never by the agent directly) ----
    def set_goal(self, rid: str, x: float, y: float, mode: str = "moving") -> None:
        r = self.robots[rid]
        r.goal, r.mode = (x, y), mode

    def hold(self, rid: str) -> None:
        r = self.robots[rid]
        r.goal, r.mode = None, "holding"

    def inject_fault(self, rid: str, kind: str) -> None:
        r = self.robots[rid]
        if kind == "comms_loss":
            r.comms_up = False
            self.events.append({"t": self.t, "kind": "fault", "robot": rid, "fault": kind})
        elif kind == "comms_restore":
            r.comms_up = True
            self.events.append({"t": self.t, "kind": "fault", "robot": rid, "fault": kind})
        elif kind == "battery_drain":
            r.battery = min(r.battery, 12.0)
            self.events.append({"t": self.t, "kind": "fault", "robot": rid, "fault": kind})

    # ---- simulation ----
    def step(self, dt: float) -> None:
        self.t += dt
        for tg in self.targets.values():
            tg.x += tg.vx * dt
            tg.y += tg.vy * dt
            if abs(tg.x) > WORLD_SIZE / 2 - 5:
                tg.vx *= -1
            if abs(tg.y) > WORLD_SIZE / 2 - 5:
                tg.vy *= -1

        for r in self.robots.values():
            if r.mode in ("moving", "returning") and r.goal:
                dx, dy = r.goal[0] - r.x, r.goal[1] - r.y
                dist = math.hypot(dx, dy)
                step = r.speed * dt
                if dist <= step:
                    r.x, r.y = r.goal
                    r.goal = None
                    r.mode = "holding"
                    self.events.append({"t": self.t, "kind": "arrived", "robot": r.id})
                else:
                    r.heading = math.atan2(dy, dx)
                    r.x += dx / dist * step
                    r.y += dy / dist * step
                    r.battery -= 0.05 * step
                if r.battery <= 0:
                    r.mode, r.goal = "lost", None
            self._sense(r)

    def _sense(self, r: Robot) -> None:
        cx, cy = int((r.x + 50) // 5), int((r.y + 50) // 5)
        reach = int(r.sensor_range // 5)
        for i in range(cx - reach, cx + reach + 1):
            for j in range(cy - reach, cy + reach + 1):
                px, py = i * 5 - 47.5, j * 5 - 47.5
                if math.hypot(px - r.x, py - r.y) <= r.sensor_range:
                    self.explored.add((i, j))
        for tg in self.targets.values():
            seen = math.hypot(tg.x - r.x, tg.y - r.y) <= r.sensor_range
            if seen and r.id not in tg.detected_by:
                tg.detected_by.add(r.id)
                self.events.append({"t": self.t, "kind": "detection", "robot": r.id, "target": tg.id,
                                    "x": round(tg.x, 1), "y": round(tg.y, 1)})
            elif not seen:
                tg.detected_by.discard(r.id)

    def visible_to(self, rid: str) -> list[dict]:
        r = self.robots[rid]
        return [
            {"target": t.id, "x": round(t.x, 1), "y": round(t.y, 1),
             "range": round(math.hypot(t.x - r.x, t.y - r.y), 1)}
            for t in self.targets.values()
            if math.hypot(t.x - r.x, t.y - r.y) <= r.sensor_range
        ]

    def snapshot(self) -> dict:
        return {
            "t": round(self.t, 2),
            "robots": [r.to_dict() for r in self.robots.values()],
            "targets": [t.to_dict() for t in self.targets.values()],
            "explored": [list(c) for c in self.explored],
            "no_go": NO_GO, "home": HOME, "size": WORLD_SIZE,
        }

    def drain_events(self) -> list[dict]:
        ev, self.events = self.events, []
        return ev
