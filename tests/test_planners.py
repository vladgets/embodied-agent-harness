"""Planner plumbing tests with fake provider clients: no network, no keys, no cost."""
import asyncio
import json
from types import SimpleNamespace as NS

import pytest

from scout.harness.trace import Trace
from scout.warehouse.agent import run_task
from scout.warehouse.embodiment import Embodiment
from scout.warehouse.models import Usage, get_spec
from scout.warehouse.planners import BudgetExceeded, ClaudePlanner, OpenAIPlanner
from scout.warehouse.world import Warehouse


class FakeAnthropic:
    """Returns: navigate_to box_1, then a final text answer."""
    def __init__(self):
        self.calls = 0
        self.messages = self

    async def create(self, **kw):
        self.calls += 1
        self.last = kw
        usage = NS(input_tokens=1000, output_tokens=200, cache_read_input_tokens=500, cache_creation_input_tokens=0)
        if self.calls == 1:
            blk = NS(type="tool_use", id="tu_1", name="navigate_to", input={"target": "box_1"})
            return NS(content=[blk], usage=usage)
        return NS(content=[NS(type="text", text="Done.")], usage=usage)


class FakeOpenAI:
    """Responses API shape: .responses.create(...) -> .id, .output items, .output_text, .usage"""
    def __init__(self):
        self.calls = 0
        self.responses = self
        self.history = []

    async def create(self, **kw):
        self.calls += 1
        self.history.append(kw)
        usage = NS(input_tokens=1500, output_tokens=100, input_tokens_details=NS(cached_tokens=500))
        if self.calls == 1:
            item = NS(type="function_call", call_id="call_1", name="navigate_to",
                      arguments=json.dumps({"target": "box_1"}))
            return NS(id="resp_1", output=[NS(type="reasoning"), item], output_text="", usage=usage)
        return NS(id="resp_2", output=[NS(type="message")], output_text="Done.", usage=usage)


def drive(planner, tmp_path):
    emb = Embodiment(Warehouse())

    async def emit(_):
        pass
    return asyncio.run(run_task("Go to the red box.", emb, planner, Trace(str(tmp_path)), emit, max_turns=5)), emb


def test_claude_planner_loop_usage_and_append_only(tmp_path):
    client = FakeAnthropic()
    p = ClaudePlanner(get_spec("claude-sonnet-5-5"), client=client)
    summ, emb = drive(p, tmp_path)
    assert summ["final"] == "Done." and summ["calls"][0]["name"] == "navigate_to"
    assert p.usage.calls == 2 and p.usage.input == 2000 and p.usage.cache_read == 1000
    assert client.last["tool_choice"]["disable_parallel_tool_use"] is True
    assert client.last["cache_control"] == {"type": "ephemeral"}
    # append-only: tool_result content was never rewritten after being appended
    tr = client.last["messages"][2]["content"][0]
    assert tr["tool_use_id"] == "tu_1" and "scene_graph" in json.loads(tr["content"])


def test_openai_planner_loop_and_usage(tmp_path):
    client = FakeOpenAI()
    p = OpenAIPlanner(get_spec("gpt-5.5"), client=client)
    summ, _ = drive(p, tmp_path)
    assert summ["final"] == "Done." and p.usage.cache_read == 1000 and p.usage.input == 2000
    first, second = client.history
    assert first["parallel_tool_calls"] is False and "previous_response_id" not in first
    assert first["tools"][0]["name"] == "navigate_to" or first["tools"][0]["type"] == "function"
    # turn 2 chains on the previous response and sends only the tool output
    assert second["previous_response_id"] == "resp_1"
    assert second["input"][0] == {"type": "function_call_output", "call_id": "call_1",
                                  "output": second["input"][0]["output"]}
    assert "scene_graph" in json.loads(second["input"][0]["output"])


def test_budget_cap_aborts_run(tmp_path):
    p = ClaudePlanner(get_spec("claude-sonnet-5-5"), budget_usd=0.0000001, client=FakeAnthropic())
    with pytest.raises(BudgetExceeded):
        drive(p, tmp_path)


def test_cost_math_and_unverified_price():
    u = Usage(input=1_000_000, output=1_000_000, cache_read=1_000_000)
    assert u.cost_usd(get_spec("claude-sonnet-5-5")) == pytest.approx(2 + 10 + 0.2)
    assert u.cost_usd(get_spec("gpt-5.4-nano")) is None  # no price known -> unknown, never guessed
