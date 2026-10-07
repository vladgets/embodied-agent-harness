"""FastAPI server: ticks the sim at 10 Hz, streams state, accepts orders and fault injections."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from scout.harness.agent import make_planner, run_order
from scout.harness.tools import Connector
from scout.harness.trace import Trace
from scout.sim.world import World

TICK = 0.1
WEB = Path(__file__).parent.parent / "web"


class Hub:
    def __init__(self):
        self.world = World()
        self.connector = Connector(self.world)
        self.trace = Trace()
        self.clients: set[WebSocket] = set()
        self.busy = False

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(json.dumps(msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def sim_loop(self):
        while True:
            self.world.step(TICK)
            for ev in self.world.drain_events():
                self.trace.log("world_event", **ev)
                await self.broadcast({"type": "event", **ev})
            await self.broadcast({"type": "state", **self.world.snapshot()})
            await asyncio.sleep(TICK)

    async def handle_order(self, text: str):
        if self.busy:
            await self.broadcast({"type": "thought", "text": "Still working on the previous order."})
            return
        self.busy = True
        try:
            planner, kind = make_planner()
            await self.broadcast({"type": "planner", "kind": kind})
            await run_order(text, self.connector, self.trace, self.broadcast, planner)
        finally:
            self.busy = False


hub = Hub()


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(hub.sim_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


@app.get("/terrain")
async def terrain():
    from scout.sim.world import WORLD_SIZE, terrain_height
    n, half = 100, WORLD_SIZE / 2
    return {"n": n, "size": WORLD_SIZE, "heights": [
        [terrain_height(-half + WORLD_SIZE * i / n, -half + WORLD_SIZE * j / n) for i in range(n + 1)]
        for j in range(n + 1)]}


@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    hub.clients.add(ws)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg["type"] == "order":
                asyncio.create_task(hub.handle_order(msg["text"]))
            elif msg["type"] == "fault":
                hub.world.inject_fault(msg["robot"], msg["kind"])
            elif msg["type"] == "reset":
                hub.world.__init__()  # in place, so the Connector keeps its reference
                await hub.broadcast({"type": "reset"})
    except WebSocketDisconnect:
        hub.clients.discard(ws)
