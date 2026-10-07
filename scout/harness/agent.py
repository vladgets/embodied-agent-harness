"""The agent loop. One `run_order` call = one commander order driven to a report.

Two planners share the same loop and the same Connector:
  * ClaudePlanner  - real model with tool use
  * ScriptedPlanner - deterministic keyword planner so the demo/evals run offline
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Awaitable, Callable

from scout.harness.tools import TOOLS, Connector
from scout.harness.trace import Trace

MODEL = os.environ.get("SCOUT_MODEL", "claude-sonnet-5-5")
MAX_STEPS = 12

SYSTEM = """You are the fleet agent for a small robot team. A commander gives orders in natural language.
Translate each order into tool calls. Rules:
- Check list_robots first; never assume robot state.
- Never send a robot into a no-go zone or out of bounds. If a tool rejects an action, adapt (reassign
  another robot or pick a nearby safe point) and say so in the report.
- If an order is ambiguous or you cannot make it safe, call request_clarification instead of guessing.
- A robot with lost comms cannot be commanded; reassign its job if possible.
- Finish by calling report with a short, factual summary of what you actually did and what failed.
Coordinates: x/y in -50..50. Home base is (-40,-40). The ridge is around (25, 30)."""

Emit = Callable[[dict], Awaitable[None]]


class ClaudePlanner:
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.messages: list[dict] = []

    async def start(self, order: str):
        self.messages = [{"role": "user", "content": order}]

    async def next_calls(self) -> tuple[str, list[dict]]:
        resp = await self.client.messages.create(
            model=MODEL, max_tokens=1024, system=SYSTEM, tools=TOOLS, messages=self.messages)
        self.messages.append({"role": "assistant", "content": resp.content})
        text = " ".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "input": b.input} for b in resp.content if b.type == "tool_use"]
        return text, calls

    async def give_results(self, results: list[tuple[str, dict]]):
        import json
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": cid, "content": json.dumps(res)} for cid, res in results]})


class ScriptedPlanner:
    """Very small intent parser: 'send <robot|N scouts> to <x,y|the ridge>', 'hold', 'return'."""

    PLACES = {"ridge": (25, 30), "home": (-40, -40), "east": (40, 0), "north": (0, 40), "west": (-40, 0)}

    def __init__(self):
        self.queue: list[dict] = []
        self.stage = 0
        self.order = ""

    async def start(self, order: str):
        self.order, self.stage, self.queue = order.lower(), 0, []

    def _target(self) -> tuple[float, float] | None:
        m = re.search(r"(-?\d+)\s*,\s*(-?\d+)", self.order)
        if m:
            return float(m.group(1)), float(m.group(2))
        for k, v in self.PLACES.items():
            if k in self.order:
                return v
        return None

    async def next_calls(self) -> tuple[str, list[dict]]:
        self.stage += 1
        if self.stage == 1:
            return "Checking fleet state.", [{"id": "s1", "name": "list_robots", "input": {}}]
        if self.stage == 2:
            tgt = self._target()
            if tgt is None:
                return "Order has no location I can resolve.", [
                    {"id": "s2", "name": "request_clarification", "input": {"question": "Where should I send them?"}}]
            n = 2 if "two" in self.order or "2" in self.order.split() else 1
            if "all" in self.order or "everyone" in self.order:
                n = 3
            calls = []
            for i, rid in enumerate(["alpha", "bravo", "charlie"][:n]):
                calls.append({"id": f"m{i}", "name": "move_to",
                              "input": {"robot_id": rid, "x": tgt[0] + i * 4, "y": tgt[1] - i * 4}})
            return f"Dispatching {n} robot(s) to {tgt}.", calls
        return "Done.", [{"id": "r", "name": "report", "input": {"summary": "Order dispatched; see tool results."}}]

    async def give_results(self, results):
        pass


def make_planner():
    if os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("SCOUT_PLANNER", "claude") == "claude":
        return ClaudePlanner(), "claude"
    return ScriptedPlanner(), "scripted"


async def run_order(order: str, connector: Connector, trace: Trace, emit: Emit, planner=None) -> dict:
    """Drive one order to completion. Returns a summary used by the eval runner."""
    planner = planner or make_planner()[0]
    await planner.start(order)
    trace.log("order", text=order)
    await emit({"type": "order", "text": order})
    summary: dict = {"calls": [], "report": None, "clarification": None}

    for step in range(MAX_STEPS):
        text, calls = await planner.next_calls()
        if text:
            trace.log("thought", step=step, text=text)
            await emit({"type": "thought", "text": text})
        if not calls:
            break
        results = []
        for c in calls:
            res = connector.execute(c["name"], c["input"])
            trace.log("tool", step=step, name=c["name"], input=c["input"], result=res)
            await emit({"type": "tool", "name": c["name"], "input": c["input"], "result": res})
            summary["calls"].append({"name": c["name"], "input": c["input"], "result": res})
            results.append((c["id"], res))
            if c["name"] == "report":
                summary["report"] = c["input"].get("summary")
            if c["name"] == "request_clarification":
                summary["clarification"] = c["input"].get("question")
        await planner.give_results(results)
        if summary["report"] is not None or summary["clarification"] is not None:
            break
        await asyncio.sleep(0)  # yield so the sim keeps ticking while we "think"
    await emit({"type": "done", "report": summary["report"], "clarification": summary["clarification"]})
    return summary
