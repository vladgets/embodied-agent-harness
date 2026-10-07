import asyncio

import pytest

from scout.harness.trace import Trace
from scout.warehouse.agent import BaselinePlanner, run_task
from scout.warehouse.embodiment import Embodiment
from scout.warehouse.evaluator import Evaluator
from scout.warehouse.tasks import TASKS
from scout.warehouse.world import Warehouse


def run(task, seed=0, tmp_path=None, **kw):
    w = Warehouse(seed=seed, **kw.pop("world_kw", {}))
    if task.setup:
        task.setup(w)
    emb = Embodiment(w, Evaluator(seed=seed, **kw.pop("ev_kw", {})), **kw)

    async def emit(_):
        pass

    summ = asyncio.run(run_task(task.instruction, emb, BaselinePlanner(task.goals), Trace(str(tmp_path)), emit,
                                max_turns=task.max_turns))
    return w, emb, summ


def test_skills_report_no_success_signal():
    w = Warehouse()
    emb = Embodiment(w)
    # pick from far away: executed, but the verdict (not the skill) says failure and why
    res = emb.execute("pick", {"object": "box_1"})
    assert res["executed"] is True
    assert res["verdict"]["status"] == "failure"
    assert "too far" in res["verdict"]["failure_reason"]


def test_closed_door_is_diagnosed():
    emb = Embodiment(Warehouse())
    res = emb.execute("navigate_to", {"target": "packing"})
    assert res["verdict"]["status"] == "failure"
    assert "d_sp" in res["verdict"]["failure_reason"]


def test_scene_graph_updates_only_on_confirmed_outcome():
    w = Warehouse(slip_p=1.0)  # every grasp slips
    emb = Embodiment(w)
    emb.execute("navigate_to", {"target": "box_1"})
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "failure"
    assert emb.graph.robot["holding"] is None
    assert "grasp slipped" in res["verdict"]["failure_reason"]


def test_false_success_corrupts_graph_then_perception_repairs_it():
    w = Warehouse(slip_p=1.0)
    emb = Embodiment(w, Evaluator(false_success_rate=1.0))
    emb.execute("navigate_to", {"target": "box_1"})
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "success"  # the judge was wrong
    assert emb.stats["false_successes"] == 1
    # the same-turn perception refresh sees the box still on the shelf and corrects the belief
    assert emb.graph.robot["holding"] is None


def test_human_move_marks_object_missing():
    w = Warehouse()
    emb = Embodiment(w)
    w.objects["box_1"].container, w.objects["box_1"].room = None, "office"
    emb.execute("look", {})
    assert emb.graph.objects["box_1"].missing


@pytest.mark.parametrize("task", TASKS[:6], ids=lambda t: t.id)
def test_baseline_solves_tasks_without_noise(task, tmp_path):
    ok = 0
    for seed in range(5):
        w, emb, summ = run(task, seed=seed, tmp_path=tmp_path, world_kw={"drop_p": 0.0})
        ok += task.success(w)
    assert ok >= 4, f"{task.id}: {ok}/5"
