"""Model-backed planners (Anthropic, OpenAI) + the factory the evals and the UI both use.

Both planners keep an append-only conversation (cache-friendly), force one tool call per turn, and
accumulate token usage so every run can be costed and capped.
"""
from __future__ import annotations

import json

from scout.warehouse.agent import SYSTEM, BaselinePlanner, Call
from scout.warehouse.embodiment import TOOLS
from scout.warehouse.models import ModelSpec, Usage, get_spec
from scout.warehouse.scene_graph import SceneGraph

MAX_OUTPUT_TOKENS = 4096  # thinking counts against this; low enough to bound a runaway turn


class BudgetExceeded(RuntimeError):
    pass


class _Metered:
    """Shared usage accounting + hard spend cap."""
    usage: Usage
    spec: ModelSpec
    budget_usd: float | None = None

    def _record(self, u: Usage) -> None:
        self.usage.add(u)
        if self.budget_usd is not None:
            spent = self.usage.cost_usd(self.spec)
            if spent is not None and spent > self.budget_usd:
                raise BudgetExceeded(f"{self.spec.id}: ${spent:.3f} exceeds per-run budget ${self.budget_usd:.2f}")


class ClaudePlanner(_Metered):
    name = "claude"

    def __init__(self, spec: ModelSpec, effort: str | None = None, budget_usd: float | None = None, client=None):
        import anthropic
        self.spec, self.effort, self.budget_usd = spec, effort, budget_usd
        self.client = client or anthropic.AsyncAnthropic()
        self.usage = Usage()
        self.messages: list[dict] = []
        self._pending_ids: list[str] = []

    async def start(self, instruction: str, graph: SceneGraph):
        self.instruction = instruction
        self.usage = Usage()

    async def decide(self, ctx: dict):
        if ctx["turn"] == 0:
            self.messages = [{"role": "user", "content": f"Instruction: {self.instruction}\n\n{ctx['brief']}"}]
        else:
            _call, result = ctx["last"]
            blocks = []
            for i, cid in enumerate(self._pending_ids):
                body = ({"result": result, "scene_graph": ctx["brief"]} if i == 0
                        else {"skipped": "one tool call per turn"})
                blocks.append({"type": "tool_result", "tool_use_id": cid, "content": json.dumps(body)})
            self.messages.append({"role": "user", "content": blocks})
        kw = {}
        if self.spec.effort:
            kw["output_config"] = {"effort": self.effort or self.spec.default_effort}
        resp = await self.client.messages.create(
            model=self.spec.id, max_tokens=MAX_OUTPUT_TOKENS, system=SYSTEM, tools=TOOLS,
            messages=self.messages, cache_control={"type": "ephemeral"},
            tool_choice={"type": "auto", "disable_parallel_tool_use": True}, **kw)
        u = resp.usage
        self._record(Usage(input=u.input_tokens, output=u.output_tokens,
                           cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                           cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0, calls=1))
        self.messages.append({"role": "assistant", "content": resp.content})
        text = " ".join(b.text for b in resp.content if b.type == "text").strip()
        uses = [b for b in resp.content if b.type == "tool_use"]
        self._pending_ids = [x.id for x in uses]
        return text, (Call(uses[0].id, uses[0].name, dict(uses[0].input)) if uses else None)


# Responses API tool format (flat; Chat Completions rejects function tools combined with reasoning_effort)
OPENAI_TOOLS = [{"type": "function", "name": t["name"], "description": t["description"],
                 "parameters": t["input_schema"]} for t in TOOLS]


class OpenAIPlanner(_Metered):
    """OpenAI Responses API. State is chained with previous_response_id, so reasoning items persist
    across turns server-side and each turn only sends the new tool output."""
    name = "openai"

    def __init__(self, spec: ModelSpec, effort: str | None = None, budget_usd: float | None = None, client=None):
        from openai import AsyncOpenAI
        self.spec, self.effort, self.budget_usd = spec, effort, budget_usd
        self.client = client or AsyncOpenAI()
        self.usage = Usage()
        self._prev_id: str | None = None
        self._pending_ids: list[str] = []

    async def start(self, instruction: str, graph: SceneGraph):
        self.instruction = instruction
        self.usage = Usage()
        self._prev_id = None

    async def decide(self, ctx: dict):
        if ctx["turn"] == 0:
            inp = [{"role": "user", "content": f"Instruction: {self.instruction}\n\n{ctx['brief']}"}]
        else:
            _call, result = ctx["last"]
            inp = []
            for i, cid in enumerate(self._pending_ids):
                body = ({"result": result, "scene_graph": ctx["brief"]} if i == 0
                        else {"skipped": "one tool call per turn"})
                inp.append({"type": "function_call_output", "call_id": cid, "output": json.dumps(body)})
        kw = {}
        if self.spec.effort and (self.effort or self.spec.default_effort):
            kw["reasoning"] = {"effort": self.effort or self.spec.default_effort}
        if self._prev_id:
            kw["previous_response_id"] = self._prev_id
        resp = await self.client.responses.create(
            model=self.spec.id, instructions=SYSTEM, input=inp, tools=OPENAI_TOOLS,
            parallel_tool_calls=False, max_output_tokens=MAX_OUTPUT_TOKENS * 2, **kw)
        self._prev_id = resp.id
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        self._record(Usage(input=u.input_tokens - cached, output=u.output_tokens, cache_read=cached, calls=1))
        calls = [it for it in resp.output if it.type == "function_call"]
        self._pending_ids = [c.call_id for c in calls]
        text = (resp.output_text or "").strip()
        if not calls:
            return text, None
        try:
            args = json.loads(calls[0].arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        return text, Call(calls[0].call_id, calls[0].name, args)


def make_planner(model_id: str, *, goals=None, effort: str | None = None, budget_usd: float | None = None):
    spec = get_spec(model_id)
    if spec.provider == "baseline":
        return BaselinePlanner(goals or [])
    if spec.provider == "openai":
        return OpenAIPlanner(spec, effort, budget_usd)
    return ClaudePlanner(spec, effort, budget_usd)
