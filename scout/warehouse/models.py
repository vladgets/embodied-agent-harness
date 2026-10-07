"""Model registry: what the config UI offers, what each costs, and how to build its planner.

Claude entries come from Anthropic's published model table. OpenAI entries are marked
`verified=False`: IDs and prices were NOT looked up, so treat them as placeholders. Override or add
entries in `models.local.json` (a JSON list of objects with the same fields) without touching code.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelSpec:
    id: str
    provider: str  # anthropic | openai | baseline
    label: str
    in_per_mtok: float | None = None
    out_per_mtok: float | None = None
    cache_read_per_mtok: float | None = None
    effort: bool = False  # whether to pass an effort / reasoning_effort setting
    efforts: list[str] = field(default_factory=list)
    default_effort: str | None = None
    verified: bool = True
    note: str = ""

    def available(self) -> bool:
        if self.provider == "baseline":
            return True
        key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[self.provider]
        return bool(os.environ.get(key))


REGISTRY = [
    ModelSpec("baseline", "baseline", "Scripted baseline (offline)", note="rule-based, no API calls"),
    ModelSpec("claude-sonnet-5-5", "anthropic", "Claude Sonnet 5.5", 2.0, 10.0, 0.20, True,
              ["low", "medium", "high", "xhigh", "max"], "high"),
    ModelSpec("claude-opus-5-5", "anthropic", "Claude Opus 5.5", 4.0, 20.0, 0.20, True,
              ["low", "medium", "high", "xhigh", "max"], "medium"),
    ModelSpec("claude-fable-5-1", "anthropic", "Claude Fable 5.1", 10.0, 50.0, 0.25, True,
              ["low", "medium", "high", "xhigh", "max"], "high"),
    ModelSpec("claude-haiku-4-5", "anthropic", "Claude Haiku 4.5", 1.0, 5.0, 0.10, False,
              note="cheapest; no effort setting"),
    # OpenAI IDs below were confirmed against models.list() on the user's account (2026-10-07).
    # Prices come from third-party aggregators (OpenAI's pricing page returned 403), so verified=False:
    # good enough for estimates, check platform.openai.com/docs/pricing before quoting them.
    ModelSpec("gpt-5.5", "openai", "OpenAI GPT-5.5", 5.0, 30.0, 0.50, True,
              ["low", "medium", "high"], "high", verified=False,
              note="the model the Thea paper used (at high reasoning effort); price unverified"),
    ModelSpec("gpt-5.4-mini", "openai", "OpenAI GPT-5.4 mini", 0.75, 4.50, 0.075, True,
              ["low", "medium", "high"], "medium", verified=False,
              note="cheap dev option on the OpenAI side; price unverified, effort levels assumed"),
    ModelSpec("gpt-5.4-nano", "openai", "OpenAI GPT-5.4 nano", None, None, None, True,
              ["low", "medium", "high"], "medium", verified=False, note="price unknown"),
]


def load_registry() -> list[ModelSpec]:
    models = {m.id: m for m in REGISTRY}
    path = Path(__file__).parent.parent.parent / "models.local.json"
    if path.exists():
        for entry in json.loads(path.read_text()):
            models[entry["id"]] = ModelSpec(**entry)
    return list(models.values())


def get_spec(model_id: str) -> ModelSpec:
    for m in load_registry():
        if m.id == model_id:
            return m
    # Unknown id: assume OpenAI-style if it looks like gpt/o-series, else Anthropic. Price unknown.
    provider = "openai" if model_id.startswith(("gpt", "o1", "o3", "o4")) else "anthropic"
    return ModelSpec(model_id, provider, model_id, verified=False, note="custom model id")


def public_registry() -> list[dict]:
    return [asdict(m) | {"available": m.available()} for m in load_registry()]


@dataclass
class Usage:
    input: int = 0  # uncached input tokens
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        for k in ("input", "output", "cache_read", "cache_write", "calls"):
            setattr(self, k, getattr(self, k) + getattr(other, k))

    def cost_usd(self, spec: ModelSpec) -> float | None:
        if spec.in_per_mtok is None or spec.out_per_mtok is None:
            return None
        cr = spec.cache_read_per_mtok if spec.cache_read_per_mtok is not None else spec.in_per_mtok
        return (self.input * spec.in_per_mtok + self.cache_write * spec.in_per_mtok * 1.25
                + self.cache_read * cr + self.output * spec.out_per_mtok) / 1e6
