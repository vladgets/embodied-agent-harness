import asyncio

import pytest

from scout.harness.trace import Trace
from scout.warehouse.agent import BaselinePlanner, run_task
from scout.warehouse.backends import PolicyBackend, ScriptedBackend
from scout.warehouse.embodiment import Embodiment
from scout.warehouse.evaluator import STEP_BUDGET_S, Evaluator
from scout.warehouse.tasks import TASKS
from scout.warehouse.world import Warehouse


def setup(force=None, seed=0, early_stop=True, near="box_1"):
    w = Warehouse(seed=seed)
    emb = Embodiment(w, Evaluator(seed=seed), backend=PolicyBackend(seed=seed, early_stop=early_stop, force=force))
    if near:
        emb.execute("navigate_to", {"target": near})  # navigation stays scripted
    return w, emb


def test_success_episode_takes_time_and_commits():
    w, emb = setup(force=["success"])
    t0 = w.t
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "success" and w.robot.holding == "box_1"
    assert 4 <= res["duration_s"] <= 8.5 and w.t - t0 >= res["duration_s"] - 0.01
    assert res["segments"][-1]["status"] == "success"
    assert all(s["status"] == "in_progress" for s in res["segments"][:-1])


def test_stall_is_cut_off_early_with_a_diagnosis():
    w, emb = setup(force=["stall"])
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "failure" and w.robot.holding is None
    assert "stalled" in res["verdict"]["failure_reason"]
    assert res["duration_s"] < STEP_BUDGET_S and res["segments"][-1]["status"] == "failure"


def test_without_early_stop_a_stall_runs_to_the_step_budget():
    _, emb = setup(force=["stall"], early_stop=False)
    res = emb.execute("pick", {"object": "box_1"})
    assert res["duration_s"] >= STEP_BUDGET_S
    assert "step budget" in res["verdict"]["failure_reason"]


def test_early_stop_saves_simulated_time_and_battery():
    w1, e1 = setup(force=["stall"], early_stop=True)
    w2, e2 = setup(force=["stall"], early_stop=False)
    t1, t2, b1, b2 = w1.t, w2.t, w1.robot.battery, w2.robot.battery
    e1.execute("pick", {"object": "box_1"})
    e2.execute("pick", {"object": "box_1"})
    assert (w1.t - t1) < (w2.t - t2) and (b1 - w1.robot.battery) < (b2 - w2.robot.battery)


def test_silent_miss_closes_on_nothing_and_says_so():
    w, emb = setup(force=["silent_miss"])
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "failure" and w.robot.holding is None
    assert "closed" in res["verdict"]["failure_reason"] and "lift" in res["verdict"]["failure_reason"]


def test_infeasible_preconditions_are_diagnosed_not_just_stalled():
    w, emb = setup(near=None)                       # robot is far from the box
    res = emb.execute("pick", {"object": "box_1"})
    assert res["verdict"]["status"] == "failure" and "too far" in res["verdict"]["failure_reason"]
    assert res["duration_s"] < STEP_BUDGET_S


def test_fragile_drop_is_silent_until_judged():
    w, emb = setup(near=None, force=["drop"])
    v = w.objects["vase_4"]
    v.held, v.room, v.container, w.robot.holding = True, None, None, "vase_4"
    w.robot.room, w.robot.x, w.robot.y = "storage", 3.0, 3.0   # next to the open shelf
    res = emb.execute("place", {"object": "vase_4", "target": "shelf_1"})
    assert res["verdict"]["status"] == "failure" and "broken" in res["verdict"]["failure_reason"]
    assert v.broken and w.robot.holding is None


@pytest.mark.parametrize("tool,args,prep", [
    ("pick", {"object": "box_1"}, None),
    ("open", {"target": "d_sp"}, "d_sp"),
])
def test_successful_episodes_are_never_cut_off_as_stalls(tool, args, prep):
    """Early stopping must not make things worse: 150 forced-success episodes, zero false failures."""
    bad = 0
    for seed in range(150):
        w, emb = setup(force=["success"], seed=seed, near=prep or "box_1")
        res = emb.execute(tool, args)
        bad += res["verdict"]["status"] != "success"
    assert bad == 0


def test_episodes_are_deterministic_per_seed():
    def trace(seed):
        w, emb = setup(seed=seed)
        res = emb.execute("pick", {"object": "box_1"})
        return res["verdict"]["status"], res["duration_s"], res["segments"]
    assert trace(3) == trace(3)


def test_unknown_object_is_a_hard_error_with_no_effect():
    w, emb = setup()
    t0 = w.t
    res = emb.execute("pick", {"object": "unicorn_9"})
    assert res["ok"] is False and w.t == t0


def test_scripted_backend_reports_no_segments():
    emb = Embodiment(Warehouse(), backend=ScriptedBackend())
    emb.execute("navigate_to", {"target": "box_1"})
    assert "segments" not in emb.execute("pick", {"object": "box_1"})


def test_navigation_stays_scripted_under_the_policy_backend():
    w, emb = setup(near=None)
    res = emb.execute("navigate_to", {"target": "box_1"})
    assert res["verdict"]["status"] == "success" and "segments" not in res


@pytest.mark.parametrize("task_id", ["L1_pick_place", "L3_locked_dock", "L4_fragile_hidden"])
def test_baseline_completes_tasks_on_the_policy_backend(task_id, tmp_path):
    task = next(t for t in TASKS if t.id == task_id)
    ok = 0
    for seed in range(5):
        w = Warehouse(seed=seed)
        if task.setup:
            task.setup(w)
        emb = Embodiment(w, Evaluator(seed=seed), backend=PolicyBackend(seed=seed))

        async def emit(_):
            pass

        asyncio.run(run_task(task.instruction, emb, BaselinePlanner(task.goals), Trace(str(tmp_path)), emit, max_turns=120))
        ok += bool(task.success(w))
    # L4 can legitimately fail when the vase is dropped; the others should almost always succeed
    assert ok >= (2 if task_id == "L4_fragile_hidden" else 4), f"{task_id}: {ok}/5"
