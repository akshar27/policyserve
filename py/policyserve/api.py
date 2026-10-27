"""M6: an HTTP API and live viewer around the closed-loop host.

    POST /tasks        start a task (one at a time: the system drives one arm)        GET /tasks/{id}  status, subgoal, result
    GET  /metrics      System 1 / System 2 timing over a recent window               GET  /healthz    is System 1 still ticking?
    GET  /stream       Server-Sent Events: frame + goal + timing, ~10 Hz              GET  /           the viewer

Every task is a real run: the Rust System 1 process, the Python System 2 thread and MuJoCo, exactly as in the tests. The timing numbers
come from the independent action tap added in M5 (it polls the action channel far faster than 100 Hz and stamps every distinct record),
not from anything System 1 reports about itself, except ``compute_us``, which is System 1's own measurement of its controller step.

Nothing here can slow System 1 down: the API only reads snapshots and logs the host already keeps, and JPEG encoding happens on the
request thread. There is no authentication, so the server binds to localhost by default.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .faults import HangingReasoner, once_holding
from .host import INSTRUCTION, PickPlaceHost
from .reasoner import parse_instruction

VIEWER = Path(__file__).parent / "viewer" / "index.html"
STALL_AFTER_S = 0.5          # System 1 counts as stalled if a running task's action channel has not advanced for this long
MAX_STREAM_CLIENTS = 8
HISTORY = 50


class Busy(Exception):
    pass


# ------------------------------------------------------------------------------------------------------------- requests
class Faults(BaseModel):
    """Optional fault injection (M5), so the resilience claim can be watched live. Bounded: this is a demo knob, not an exploit surface."""
    model_config = ConfigDict(extra="forbid")
    system2_hang_s: float | None = Field(None, gt=0, le=30, description="block System 2 this long, once, while the block is held")
    system1_slow_every: int | None = Field(None, ge=2, le=1000, description="make every Nth System 1 controller step slow")
    system1_slow_us: int = Field(8000, ge=100, le=50_000, description="how much slower, in microseconds (budget is 5000)")


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction: str = Field(min_length=1, max_length=200)
    faults: Faults | None = None


# ------------------------------------------------------------------------------------------------------------- metrics
def _pcts(values) -> dict:
    if len(values) == 0:
        return {"p50": None, "p95": None, "p99": None, "max": None, "n": 0}
    a = np.asarray(values, dtype=float)
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"p50": round(float(p50), 3), "p95": round(float(p95), 3), "p99": round(float(p99), 3), "max": round(float(a.max()), 3), "n": int(a.size)}


def tick_intervals_ms(records: list[tuple]) -> list[float]:
    """Wall-clock milliseconds per control tick, one value for each pair of consecutive distinct action records (if the tap missed
    some ticks in between, the gap is divided by the number of ticks it spans). A genuine stall therefore shows up as one large value.
    Resolution is bounded by the tap's polling period (about 1 ms)."""
    out = []
    for (w0, t0, *_), (w1, t1, *_) in zip(records, records[1:]):
        if t1 > t0:
            out.append((w1 - w0) * 1000.0 / (t1 - t0))
    return out


def compute_metrics(action_log: list[tuple], system2, window_s: float | None, now: float | None = None) -> dict:
    """Timing over the last ``window_s`` seconds of the run (None = the whole run). Records are
    ``(wall, tick, phase, deadline_misses, compute_ns, vel)`` as logged by the host's action tap."""
    now = time.perf_counter() if now is None else now
    snap = list(action_log)            # a copy: the tap thread keeps appending
    if window_s is not None:
        snap = [r for r in snap if r[0] >= now - window_s]
    s1: dict = {"tick_interval_ms": _pcts(tick_intervals_ms(snap)), "compute_us": _pcts([r[4] / 1000.0 for r in snap]),
                "observed_hz": None, "deadline_misses_total": 0, "deadline_misses_in_window": 0}
    if len(snap) >= 2 and snap[-1][0] > snap[0][0]:
        s1["observed_hz"] = round((snap[-1][1] - snap[0][1]) / (snap[-1][0] - snap[0][0]), 2)
    if snap:
        s1["deadline_misses_total"] = int(snap[-1][3])
        s1["deadline_misses_in_window"] = int(snap[-1][3] - snap[0][3])
    s2: dict = {"cycles": 0, "late_cycles": 0, "failures": 0, "cycle_duration_ms": _pcts([])}
    if system2 is not None:
        recs = list(system2.records)
        s2 = {"cycles": system2.cycle, "late_cycles": system2.late_cycles, "failures": system2.total_failures,
              "cycle_duration_ms": _pcts([r.duration_s * 1000.0 for r in recs])}
    return {"window_s": window_s, "system1": s1, "system2": s2}


# ------------------------------------------------------------------------------------------------------------- tasks
@dataclass
class Task:
    id: str
    instruction: str
    faults: dict | None
    status: str = "queued"                     # queued | running | succeeded | failed | aborted | error
    created: float = field(default_factory=time.time)
    started_wall: float | None = None          # perf_counter
    finished_wall: float | None = None
    host: PickPlaceHost | None = None
    result: dict | None = None
    final_metrics: dict | None = None
    error: str | None = None

    @property
    def active(self) -> bool:
        return self.status in ("queued", "running")


class TaskManager:
    """Runs at most one task at a time, each on its own thread. The host is built and run on that thread because MuJoCo's offscreen
    renderer must stay on the thread that created it."""

    def __init__(self, host_factory=PickPlaceHost):
        # Always paced to wall-clock: System 2 plans on a wall-clock cadence, so a simulation running faster than real time would
        # outrun it and System 1 would time out waiting for a goal.
        self._factory = host_factory
        self._lock = threading.Lock()
        self._tasks: dict[str, Task] = {}
        self._order: list[str] = []
        self._current: Task | None = None
        self._last: Task | None = None
        self._snap: dict | None = None
        self._jpeg: tuple[int, bytes] | None = None
        self._jpeg_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ---- starting / running
    def start(self, instruction: str, faults: Faults | None) -> Task:
        with self._lock:
            if self._current is not None and self._current.active:
                raise Busy(self._current.id)
            task = Task(uuid.uuid4().hex[:12], instruction, faults.model_dump(exclude_none=True) if faults else None)
            self._tasks[task.id] = task
            self._order.append(task.id)
            while len(self._order) > HISTORY:
                self._tasks.pop(self._order.pop(0), None)
            self._current = task
            self._snap, self._jpeg = None, None
        self._thread = threading.Thread(target=self._run, args=(task,), name=f"task-{task.id}", daemon=True)
        self._thread.start()
        return task

    def _on_snapshot(self, snap: dict) -> None:
        self._snap = snap          # one reference assignment: readers always see a whole snapshot

    def _run(self, task: Task) -> None:
        try:
            f = task.faults or {}
            args: tuple[str, ...] = ()
            if "system1_slow_every" in f:
                args = ("--inject-slow-every", str(f["system1_slow_every"]), "--inject-slow-us", str(f["system1_slow_us"]))
            host = self._factory(instruction=task.instruction, system1_args=args, on_snapshot=self._on_snapshot)
            if "system2_hang_s" in f:
                host.reasoner = HangingReasoner(host.reasoner, f["system2_hang_s"], once_holding)
            task.host = host
            task.started_wall = time.perf_counter()
            task.status = "running"
            res = host.run()
            task.result = {
                "success": res.success, "sim_time_s": round(res.sim_time, 3), "wall_time_s": round(res.wall_time, 3),
                "block_final_xz": [round(res.red_block[0], 4), round(res.red_block[1], 4)], "phases": res.phases(),
                "system1": res.system1,
                "system2": {"status": res.system2_status, "failure": res.system2_failure, "cycles": res.system2_cycles, "late_cycles": res.system2_late_cycles},
            }
            task.final_metrics = compute_metrics(res.action_log, host.system2, None)
            task.status = "aborted" if res.aborted else ("succeeded" if res.success else "failed")
        except Exception as e:  # noqa: BLE001 - a crashed run must be reported, never take the server down
            task.error = f"{type(e).__name__}: {e}"
            task.status = "error"
        finally:
            task.finished_wall = time.perf_counter()
            with self._lock:
                self._last = task

    def shutdown(self) -> None:
        t = self._current
        if t is not None and t.active and t.host is not None:
            t.host.abort()
        if self._thread is not None:
            self._thread.join(15)

    # ---- reading
    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def public(self, t: Task) -> dict:
        host, snap = t.host, self._snap if t is self._current else None
        s2 = host.system2 if host else None
        end = t.finished_wall or time.perf_counter()
        out = {
            "id": t.id, "instruction": t.instruction, "status": t.status, "faults": t.faults,
            "elapsed_s": round(end - t.started_wall, 3) if t.started_wall else 0.0,
            "subgoal": s2.last_goal if s2 else None,
            "phase": (snap or {}).get("phase") if t.active else (t.result or {}).get("phases", [None])[-1],
            "sim_time_s": round(snap["sim_time"], 3) if snap else (t.result or {}).get("sim_time_s"),
            "success": (t.result or {}).get("success"), "failure": (t.result or {}).get("system2", {}).get("failure") or t.error,
        }
        if t.result is not None:
            out["result"] = t.result
        return out

    def current_or_last(self) -> Task | None:
        return self._current

    def metrics(self, window_s: float) -> dict:
        t = self._current
        if t is None:
            return {"task_id": None, "scope": "none", **compute_metrics([], None, window_s)}
        if t.active and t.host is not None:
            return {"task_id": t.id, "scope": "live", **compute_metrics(t.host.action_log, t.host.system2, window_s)}
        if t.final_metrics is not None:
            return {"task_id": t.id, "scope": "whole last run", **t.final_metrics}
        return {"task_id": t.id, "scope": "none", **compute_metrics([], None, window_s)}

    def health(self) -> dict:
        t = self._current
        if t is None or not t.active:
            return {"ok": True, "status": "idle", "task_id": None, "system1": None}
        log = t.host.action_log if t.host else []
        if not log:
            return {"ok": True, "status": "starting", "task_id": t.id, "system1": None}
        age = time.perf_counter() - log[-1][0]
        recent = compute_metrics(log, None, 2.0)["system1"]
        stalled = age > STALL_AFTER_S
        return {"ok": not stalled, "status": "stalled" if stalled else "ticking", "task_id": t.id,
                "system1": {"last_tick_age_s": round(age, 3), "observed_hz": recent["observed_hz"], "tick": log[-1][1]}}

    def latest_jpeg(self) -> tuple[int, bytes] | None:
        """The newest rendered frame as JPEG, encoded at most once per frame however many viewers are watching."""
        snap = self._snap
        if snap is None:
            return None
        with self._jpeg_lock:
            if self._jpeg is None or self._jpeg[0] != snap["step"]:
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(snap["frame"]).save(buf, "JPEG", quality=72)
                self._jpeg = (snap["step"], buf.getvalue())
            return self._jpeg

    def stream_state(self) -> dict:
        t = self._current
        snap = self._snap
        state: dict = {"task": self.public(t) if t else None, "server_time": time.time()}
        if t is not None and t.host is not None and snap is not None and t.active:
            log = t.host.action_log
            recent = [r for r in list(log[-600:]) if r[0] >= time.perf_counter() - 5.0]
            state.update({
                "ee": [round(snap["ee"][0], 4), round(snap["ee"][1], 4)], "block": [round(snap["block"][0], 4), round(snap["block"][1], 4)],
                "holding": snap["holding"], "gap": round(snap["gap"], 4), "tick_dt_ms": [round(x, 2) for x in tick_intervals_ms(recent)[-150:]],
                "metrics": compute_metrics(log, t.host.system2, 5.0),
            })
        elif t is not None and t.final_metrics is not None:
            state["metrics"] = {**t.final_metrics, "scope": "whole last run"}
        return state


# ------------------------------------------------------------------------------------------------------------- app
def create_app(manager: TaskManager | None = None) -> FastAPI:
    mgr = manager or TaskManager()
    clients = {"n": 0}

    @asynccontextmanager
    async def lifespan(_app):
        yield
        await asyncio.get_running_loop().run_in_executor(None, mgr.shutdown)

    app = FastAPI(title="policyserve", version="0.6.0", lifespan=lifespan)
    app.state.manager = mgr

    @app.get("/", include_in_schema=False)
    def viewer():
        return FileResponse(VIEWER, media_type="text/html")

    @app.get("/healthz")
    def healthz():
        h = mgr.health()
        return JSONResponse(h, status_code=200 if h["ok"] else 503)

    @app.post("/tasks", status_code=202)
    def post_task(req: TaskRequest):
        instruction = req.instruction.strip()
        parsed = parse_instruction(instruction)
        if parsed is None:
            raise HTTPException(400, f"Could not understand the instruction. Try: {INSTRUCTION!r}")
        if parsed != ("red", "green"):
            raise HTTPException(400, "This scene only supports moving the red block onto the green platform.")
        try:
            task = mgr.start(instruction, req.faults)
        except Busy as e:
            raise HTTPException(409, f"A task is already running ({e.args[0]}). Only one task runs at a time.")
        return {"task_id": task.id, "status": task.status}

    @app.get("/tasks/{task_id}")
    def get_task(task_id: str):
        t = mgr.get(task_id)
        if t is None:
            raise HTTPException(404, "No such task")
        return mgr.public(t)

    @app.get("/metrics")
    def get_metrics(window: float = 10.0):
        if not 0.5 <= window <= 120:
            raise HTTPException(422, "window must be between 0.5 and 120 seconds")
        return mgr.metrics(window)

    @app.get("/stream")
    async def stream(request: Request, hz: float = 10.0):
        if not 1 <= hz <= 20:
            raise HTTPException(422, "hz must be between 1 and 20")
        if clients["n"] >= MAX_STREAM_CLIENTS:
            raise HTTPException(503, "Too many viewers")
        loop = asyncio.get_running_loop()

        async def events():
            clients["n"] += 1
            last_step = None
            try:
                while not await request.is_disconnected():
                    state = mgr.stream_state()
                    jpeg = await loop.run_in_executor(None, mgr.latest_jpeg)
                    if jpeg is not None and jpeg[0] != last_step:
                        last_step = jpeg[0]
                        state["frame"] = "data:image/jpeg;base64," + base64.b64encode(jpeg[1]).decode()
                    yield f"event: state\ndata: {json.dumps(state)}\n\n"
                    await asyncio.sleep(1.0 / hz)
            finally:
                clients["n"] -= 1

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(description="policyserve HTTP API and live viewer")
    ap.add_argument("--host", default="127.0.0.1", help="there is no authentication: keep this on localhost unless you know why not")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
