import asyncio

from scout.harness.agent import ScriptedPlanner, run_order
from scout.harness.tools import Connector
from scout.harness.trace import Trace
from scout.sim.world import World


def make(tmp_path):
    w = World()
    return w, Connector(w), Trace(str(tmp_path))


def test_move_rejected_in_no_go_zone(tmp_path):
    w, c, _ = make(tmp_path)
    res = c.execute("move_to", {"robot_id": "alpha", "x": 10, "y": 15})
    assert not res["ok"] and "no-go" in res["error"]


def test_move_rejected_out_of_bounds(tmp_path):
    w, c, _ = make(tmp_path)
    assert not c.execute("move_to", {"robot_id": "alpha", "x": 80, "y": 0})["ok"]


def test_comms_loss_blocks_commands(tmp_path):
    w, c, _ = make(tmp_path)
    w.inject_fault("bravo", "comms_loss")
    res = c.execute("move_to", {"robot_id": "bravo", "x": 0, "y": 0})
    assert not res["ok"] and "comms" in res["error"]


def test_robot_reaches_goal_and_detects_target(tmp_path):
    w, c, _ = make(tmp_path)
    assert c.execute("move_to", {"robot_id": "alpha", "x": 28, "y": 34})["ok"]
    for _ in range(1200):
        w.step(0.1)
    assert w.robots["alpha"].mode == "holding"
    assert w.targets["t1"].detected_by


def test_scripted_order_dispatches_and_reports(tmp_path):
    w, c, tr = make(tmp_path)
    events = []

    async def emit(m):
        events.append(m)

    summary = asyncio.run(run_order("send two scouts to the ridge", c, tr, emit, ScriptedPlanner()))
    assert summary["report"]
    assert sum(1 for x in summary["calls"] if x["name"] == "move_to" and x["result"]["ok"]) == 2


def test_ambiguous_order_asks_for_clarification(tmp_path):
    w, c, tr = make(tmp_path)

    async def emit(m):
        pass

    summary = asyncio.run(run_order("go look around", c, tr, emit, ScriptedPlanner()))
    assert summary["clarification"]
