"""Warehouse demo server: runs one agent task at a time and streams everything the viewer needs.

    uvicorn scout.warehouse.server:app --port 8001

Playback is paced on the server: after each physical action it sleeps for the (scaled) simulated
duration, so the 3D animation keeps up even with the instant offline baseline.
"""
from __future__ import annotations

import asyncio
import hmac
import json
import os
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

from scout.harness.trace import Trace
from scout.warehouse.agent import run_task
from scout.warehouse.embodiment import Embodiment
from scout.warehouse.evaluator import Evaluator
from scout.warehouse.evals import estimate_run_cost
from scout.warehouse.models import get_spec, public_registry
from scout.warehouse.planners import BudgetExceeded, make_planner
from scout.warehouse.tasks import TASKS
from scout.warehouse.world import CHARGER, ROOMS, Warehouse

WEB = Path(__file__).parent.parent.parent / "web"
PLAYBACK = 2.0  # default simulated seconds per wall-clock second (the UI can override per run)
PHYSICAL = {"navigate_to", "open", "pick", "place", "charge"}


# ---- deployment guards (all optional; unset = open, for local development) ----
def token_ok(provided: str | None) -> bool:
    """If DEMO_TOKEN is set, every page and socket must present it (?token=...)."""
    expected = os.environ.get("DEMO_TOKEN")
    return not expected or (provided is not None and hmac.compare_digest(provided, expected))


def run_budget(requested) -> float:
    """Per-run spend cap: what the UI asks for, never above MAX_RUN_BUDGET_USD (default $0.75)."""
    ceiling = float(os.environ.get("MAX_RUN_BUDGET_USD", "0.75"))
    return max(0.01, min(float(requested or ceiling), ceiling))


def total_cap() -> float | None:
    v = os.environ.get("MAX_TOTAL_SPEND_USD")
    return float(v) if v else None


def snapshot(w: Warehouse) -> dict:
    v = w.eval_view()
    return {"t": v["t"], "robot": v["robot"], "objects": list(v["objects"].values()),
            "containers": list(v["containers"].values()), "doors": v["doors"]}


class Hub:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.task: asyncio.Task | None = None
        self.world = Warehouse()
        self.planner = None
        self.run_info: dict = {}
        self.spent = 0.0  # paid spend by this server process, for MAX_TOTAL_SPEND_USD

    async def broadcast(self, msg: dict) -> None:
        text = json.dumps(msg, default=str)
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)

    def hello(self) -> dict:
        models = []
        for m in public_registry():
            est = estimate_run_cost(m["id"])
            models.append(m | {"est_run_usd": est})
        running = bool(self.task and not self.task.done())
        limits = {"max_run_usd": run_budget(None), "max_total_usd": total_cap(), "spent_usd": round(self.spent, 4)}
        return {"type": "hello", "limits": limits, "running": running, "run": self.run_info if running else None, "models": models,
                "tasks": [{"id": t.id, "level": t.level, "instruction": t.instruction, "has_goals": bool(t.goals)}
                          for t in TASKS],
                "static": {"rooms": ROOMS, "charger": CHARGER}, "snapshot": snapshot(self.world)}

    def usage_msg(self) -> dict:
        p = self.planner
        u = getattr(p, "usage", None)
        if u is None:
            return {"tokens_in": 0, "tokens_out": 0, "cache_read": 0, "cost_usd": 0.0}
        return {"tokens_in": u.input, "tokens_out": u.output, "cache_read": u.cache_read,
                "cost_usd": u.cost_usd(p.spec) if hasattr(p, "spec") else 0.0}

    async def reset(self, seed: int = 0) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.world = Warehouse(seed=seed)
        await self.broadcast(self.hello())

    async def start(self, msg: dict) -> None:
        if self.task and not self.task.done():
            await self.broadcast({"type": "error", "message": "A task is already running."})
            return
        await self.reset(int(msg.get("seed", 0)))
        self.task = asyncio.create_task(self._run(msg))

    async def _run(self, msg: dict) -> None:
        task = next((t for t in TASKS if t.id == msg.get("task_id")), None)
        instruction = (msg.get("instruction") or (task.instruction if task else "")).strip()
        model = msg.get("model") or "baseline"
        w = self.world
        if task and task.setup and instruction == task.instruction:
            task.setup(w)
        else:
            task = task if task and instruction == task.instruction else None
        spec = get_spec(model)
        self.planner = None
        try:
            cap = total_cap()
            if spec.provider != "baseline" and cap is not None and self.spent >= cap:
                raise ValueError(f"This server's spend cap (${cap:.2f}) is used up. Use the free baseline, or restart the service.")
            if not instruction:
                raise ValueError("Enter an instruction or pick a task.")
            if spec.provider == "baseline" and not (task and task.goals):
                raise ValueError("The scripted baseline only runs the preset tasks that have goals (L1–L6).")
            if not spec.available():
                raise ValueError(f"No API key set for {spec.provider}.")
            emb = Embodiment(w, Evaluator(false_success_rate=float(msg.get("false_success", 0)), seed=int(msg.get("seed", 0))),
                             use_evaluator=bool(msg.get("use_evaluator", True)),
                             user_reply=lambda q: "Operator: no preference. Do not move heavy items; use your best judgment.")
            self.planner = make_planner(model, goals=task.goals if task else None,
                                        effort=msg.get("effort") or None,
                                        budget_usd=run_budget(msg.get("budget_usd")) if spec.provider != "baseline" else None)
            playback = max(0.5, min(float(msg.get("speed") or PLAYBACK), 16.0))
            state = {"log": 0, "t": w.t}
            self.run_info = {"model": model, "instruction": instruction}
            await self.broadcast({"type": "started", "instruction": instruction, "model": model,
                                  "snapshot": snapshot(w), "usage": self.usage_msg()})

            async def emit(m: dict) -> None:
                if m["type"] == "tool":
                    anim = w.log[state["log"]:]
                    state["log"] = len(w.log)
                    dt, state["t"] = w.t - state["t"], w.t
                    physical = m["name"] in PHYSICAL
                    duration = min(max(dt / playback, 0.8), 20.0) if physical and anim else (0.8 if physical else 0.3)
                    await self.broadcast(m | {"anim": anim, "duration": duration, "snapshot": snapshot(w),
                                              "usage": self.usage_msg(), "physical": physical})
                    await asyncio.sleep(duration)
                else:
                    await self.broadcast(m)

            summ = await run_task(instruction, emb, self.planner, Trace("runs/ui"), emit,
                                  max_turns=task.max_turns if task else 60)
            result = {"type": "result", "turns": summ["turns"], "usage": self.usage_msg(),
                      "stats": emb.stats, "final": summ["final"]}
            if task:
                result["ok"] = bool(task.success(w)) and (summ["asked_user"] if task.expects_user_query else True)
            await self.broadcast(result)
        except asyncio.CancelledError:
            await self.broadcast({"type": "stopped", "usage": self.usage_msg()})
            raise
        except BudgetExceeded as e:
            await self.broadcast({"type": "error", "message": f"Stopped: {e}", "usage": self.usage_msg()})
        except Exception as e:  # surface API/auth/validation errors to the UI instead of dying silently
            await self.broadcast({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            u = getattr(self.planner, "usage", None)
            if u is not None and hasattr(self.planner, "spec"):
                self.spent += u.cost_usd(self.planner.spec) or 0.0


hub = Hub()
app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/")
async def index(token: str | None = None):
    if not token_ok(token):
        return PlainTextResponse("Unauthorized: open the link with ?token=...", status_code=401)
    return FileResponse(WEB / "warehouse.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if not token_ok(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    hub.clients.add(ws)
    await ws.send_text(json.dumps(hub.hello(), default=str))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["type"] == "start":
                await hub.start(msg)
            elif msg["type"] == "stop" and hub.task and not hub.task.done():
                hub.task.cancel()
            elif msg["type"] == "reset":
                await hub.reset(int(msg.get("seed", 0)))
    except WebSocketDisconnect:
        hub.clients.discard(ws)
