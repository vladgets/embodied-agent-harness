import asyncio

from scout.harness.trace import Trace
from scout.warehouse.agent import BaselinePlanner, run_task
from scout.warehouse.embodiment import TOOLS, Embodiment
from scout.warehouse.tasks import TASKS
from scout.warehouse.world import Warehouse


def test_step_argument_advances_plan_without_reaching_the_skill():
    emb = Embodiment(Warehouse())
    emb.execute("update_plan", {"steps": [{"text": "a", "status": "todo"}, {"text": "b", "status": "todo"}, {"text": "c", "status": "todo"}]})
    emb.execute("navigate_to", {"target": "box_1", "step": 1})
    assert [s["status"] for s in emb.plan] == ["doing", "todo", "todo"]
    res = emb.execute("pick", {"object": "box_1", "step": 2})  # `step` must not break the skill call
    assert res["ok"] and [s["status"] for s in emb.plan][0] == "done"
    emb.plan_dirty = False
    emb.execute("place", {"object": "box_1", "target": "table_pack", "step": 3})  # fails: door closed, far away
    assert emb.plan[2]["status"] == "blocked" and emb.plan_dirty


def test_step_is_in_every_physical_tool_schema_but_never_required():
    for t in TOOLS:
        props = t["input_schema"]["properties"]
        if t["name"] in {"navigate_to", "open", "pick", "place", "charge"}:
            assert "step" in props and "step" not in t["input_schema"].get("required", [])


def test_baseline_produces_a_plan_that_completes(tmp_path):
    task = next(t for t in TASKS if t.id == "L2_two_boxes")
    w = Warehouse(seed=1, drop_p=0.0)
    emb = Embodiment(w)
    plans = []

    async def emit(m):
        if m["type"] == "plan":
            plans.append([s["status"] for s in m["steps"]])

    asyncio.run(run_task(task.instruction, emb, BaselinePlanner(task.goals), Trace(str(tmp_path)), emit, max_turns=80))
    assert task.success(w)
    assert plans[0] == ["todo"] * 4                       # initial plan
    assert any("doing" in p for p in plans)               # progress was visible mid-run
    assert plans[-1] == ["done"] * 4                      # and finished
