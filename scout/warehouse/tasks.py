"""Eval tasks: instruction + ground-truth success predicate + (for the baseline) structured goals."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from scout.warehouse.world import Warehouse


@dataclass
class Task:
    id: str
    level: str
    instruction: str
    success: Callable[[Warehouse], bool]
    goals: list[tuple[dict, str]] = field(default_factory=list)  # for BaselinePlanner only
    setup: Callable[[Warehouse], None] | None = None
    max_turns: int = 60
    expects_user_query: bool = False


def _on(w: Warehouse, oid: str, dest: str) -> bool:
    o = w.objects[oid]
    return o.container == dest and not o.broken


def _human_moves_red_box(w: Warehouse) -> None:
    # After the 3rd physical action a human takes the red box off the shelf and leaves it in the office.
    def move(world: Warehouse) -> None:
        o = world.objects["box_1"]
        if not o.held:
            o.container, o.room, o.x, o.y = None, "office", 8.0, 20.0
    w.add_perturbation(3, "human moves the red box to the office floor", move)


TASKS = [
    Task("L1_pick_place", "L1", "Put the red box on the packing table.",
         lambda w: _on(w, "box_1", "table_pack"), [({"color": "red", "kind": "box"}, "table_pack")]),
    Task("L2_two_boxes", "L2", "Move the red box and the blue box to the packing table.",
         lambda w: _on(w, "box_1", "table_pack") and _on(w, "box_2", "table_pack"),
         [({"color": "red", "kind": "box"}, "table_pack"), ({"color": "blue", "kind": "box"}, "table_pack")], max_turns=80),
    Task("L3_locked_dock", "L3", "Deliver the blue box to the dock table.",
         lambda w: _on(w, "box_2", "table_dock"), [({"color": "blue", "kind": "box"}, "table_dock")], max_turns=100),
    Task("L4_fragile_hidden", "L4", "Put the white vase on the office table.",
         lambda w: _on(w, "vase_4", "table_office"), [({"color": "white", "kind": "vase"}, "table_office")]),
    Task("L5_search", "L5", "Bring the keycard to the packing table.",
         lambda w: _on(w, "keycard_5", "table_pack"), [({"kind": "keycard"}, "table_pack")], max_turns=80),
    Task("L6_world_changes", "L6", "Put the red box on the packing table.",
         lambda w: _on(w, "box_1", "table_pack"), [({"color": "red", "kind": "box"}, "table_pack")],
         setup=_human_moves_red_box, max_turns=80),
    Task("L7_too_heavy", "L7", "Move the yellow crate to the dock table.",
         lambda w: not w.objects["crate_3"].held and w.objects["crate_3"].container is None,
         expects_user_query=True),
]
