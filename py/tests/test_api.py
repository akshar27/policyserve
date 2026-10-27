"""M6: the HTTP API and the numbers it serves. Real runs (Rust System 1 + Python System 2 + MuJoCo) behind a real uvicorn server."""

import base64
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from policyserve.api import Task, TaskManager, compute_metrics, create_app, tick_intervals_ms
from policyserve.faults import HangingReasoner
from policyserve.host import INSTRUCTION

GOOD = {"instruction": INSTRUCTION}


class Server:
    def __init__(self, manager):
        self.manager = manager
        self.server = uvicorn.Server(uvicorn.Config(create_app(manager), host="127.0.0.1", port=0, log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        while not self.server.started:
            time.sleep(0.02)
        self.url = f"http://127.0.0.1:{self.server.servers[0].sockets[0].getsockname()[1]}"

    def stop(self):
        self.manager.shutdown()
        self.server.should_exit = True
        self.thread.join(10)


@pytest.fixture
def live():
    """A server that runs real closed-loop tasks (about 10 s each, paced to wall-clock: it cannot be sped up)."""
    s = Server(TaskManager())
    yield s
    s.stop()


@pytest.fixture
def fake():
    """A server whose tasks are instant fakes, for everything that is not about the real run."""
    s = Server(TaskManager(host_factory=fake_host_factory([])))
    yield s
    s.stop()


def wait_done(client, task_id, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        t = client.get(f"/tasks/{task_id}").json()
        if t["status"] not in ("queued", "running"):
            return t
        time.sleep(0.2)
    raise AssertionError("task did not finish")


# ------------------------------------------------------------------------------------------------ validation
def test_bad_requests_are_rejected_before_anything_starts(fake):
    with httpx.Client(base_url=fake.url) as c:
        for body in ({}, {"instruction": ""}, {"instruction": "x" * 201}, {**GOOD, "extra": 1},
                     {**GOOD, "faults": {"system2_hang_s": 999}}, {**GOOD, "faults": {"system2_hang_s": -1}},
                     {**GOOD, "faults": {"system1_slow_every": 1}}, {**GOOD, "faults": {"rm_rf": True}}):
            assert c.post("/tasks", json=body).status_code == 422, body
        assert c.post("/tasks", json={"instruction": "make me a sandwich"}).status_code == 400
        assert c.post("/tasks", json={"instruction": "pick up the blue block and place it on the green platform"}).status_code == 400
        assert c.get("/tasks/nope").status_code == 404
        assert c.get("/metrics", params={"window": 9999}).status_code == 422
        assert c.get("/stream", params={"hz": 500}).status_code == 422
        assert c.get("/healthz").json()["status"] == "idle"
        assert "<title>policyserve</title>" in c.get("/").text
    assert fake.manager.current_or_last() is None, "no rejected request may have started a task"


# ------------------------------------------------------------------------------------------------ lifecycle
def test_task_lifecycle_one_at_a_time_and_metrics_afterwards(live):
    with httpx.Client(base_url=live.url) as c:
        r = c.post("/tasks", json=GOOD)
        assert r.status_code == 202
        tid = r.json()["task_id"]
        busy = c.post("/tasks", json=GOOD)
        assert busy.status_code == 409 and tid in busy.json()["detail"]

        t = wait_done(c, tid)
        assert t["status"] == "succeeded" and t["success"] is True
        assert {"approach", "grasp", "transport", "release", "finished"} <= set(t["result"]["phases"])
        assert t["result"]["system1"]["deadline_misses"] == 0

        m = c.get("/metrics").json()
        assert m["task_id"] == tid and m["scope"] == "whole last run"
        assert m["system1"]["compute_us"]["n"] > 100 and m["system1"]["deadline_misses_total"] == 0
        assert m["system2"]["cycles"] >= 3

        assert c.post("/tasks", json=GOOD).status_code == 202, "a new task may start once the last has finished"


def test_injected_slow_steps_show_up_as_deadline_misses_and_the_server_survives(live):
    with httpx.Client(base_url=live.url) as c:
        tid = c.post("/tasks", json={**GOOD, "faults": {"system1_slow_every": 10, "system1_slow_us": 8000}}).json()["task_id"]
        t = wait_done(c, tid)
        assert t["status"] != "error", t.get("failure")
        m = c.get("/metrics").json()
        assert m["system1"]["deadline_misses_total"] >= 10
        assert c.get("/healthz").status_code == 200


# ------------------------------------------------------------------------------------------------ unit: the manager and the maths
def fake_host_factory(record, boom=False):
    def factory(**kw):
        record.append(kw)
        if boom:
            raise RuntimeError("mujoco exploded")
        host = SimpleNamespace(reasoner=SimpleNamespace(plan=lambda *a: None), system2=None, action_log=[])
        res = SimpleNamespace(success=True, sim_time=1.0, wall_time=1.0, red_block=(0.1, 0.04), phases=lambda: ["home", "finished"], system1={},
                              system2_status="done", system2_failure=None, system2_cycles=1, system2_late_cycles=0, action_log=[], aborted=False)
        host.run = lambda: res
        return host
    return factory


def test_fault_options_are_translated_into_real_fault_injection():
    seen = []
    mgr = TaskManager(host_factory=fake_host_factory(seen))
    app = TestClient(create_app(mgr))
    r = app.post("/tasks", json={**GOOD, "faults": {"system2_hang_s": 2.5, "system1_slow_every": 7}})
    assert r.status_code == 202
    mgr._thread.join(5)
    kw = seen[0]
    assert kw["system1_args"] == ("--inject-slow-every", "7", "--inject-slow-us", "8000")
    assert callable(kw["on_snapshot"])
    t = mgr.get(r.json()["task_id"])
    assert t.status == "succeeded" and isinstance(t.host.reasoner, HangingReasoner) and t.host.reasoner.hang_s == 2.5


def test_a_crashing_run_is_reported_and_the_server_keeps_working():
    mgr = TaskManager(host_factory=fake_host_factory([], boom=True))
    app = TestClient(create_app(mgr))
    tid = app.post("/tasks", json=GOOD).json()["task_id"]
    mgr._thread.join(5)
    t = app.get(f"/tasks/{tid}").json()
    assert t["status"] == "error" and "mujoco exploded" in t["failure"]
    assert app.get("/healthz").status_code == 200
    assert app.post("/tasks", json=GOOD).status_code == 202, "a crashed task must not wedge the server"


def test_healthz_goes_503_when_system_1_stops_ticking_during_a_task():
    mgr = TaskManager()
    stale = [(time.perf_counter() - 5.0, 100, 2, 0, 1000, (0, 0, 0))]
    t = Task("t1", GOOD["instruction"], None, status="running", host=SimpleNamespace(action_log=stale, system2=None))
    mgr._current = t
    app = TestClient(create_app(mgr))
    r = app.get("/healthz")
    assert r.status_code == 503 and r.json()["status"] == "stalled" and r.json()["system1"]["last_tick_age_s"] > 4
    t.host.action_log.append((time.perf_counter(), 101, 2, 0, 1000, (0, 0, 0)))
    assert app.get("/healthz").status_code == 200


def test_tick_interval_and_percentile_maths():
    recs = [(0.00, 0, 0, 0, 1000, ()), (0.01, 1, 0, 0, 2000, ()), (0.03, 3, 0, 1, 3000, ()), (0.04, 4, 0, 1, 4000, ())]
    assert tick_intervals_ms(recs) == pytest.approx([10.0, 10.0, 10.0]), "a 2-tick gap of 20 ms is 10 ms per tick"
    assert tick_intervals_ms([(0.0, 0, 0, 0, 0, ()), (0.5, 1, 0, 0, 0, ())]) == pytest.approx([500.0]), "a stall shows as one big value"
    m = compute_metrics(recs, None, None)["system1"]
    assert m["observed_hz"] == pytest.approx(100.0)
    assert m["deadline_misses_total"] == 1 and m["deadline_misses_in_window"] == 1
    assert m["compute_us"]["max"] == 4.0 and m["compute_us"]["n"] == 4
    empty = compute_metrics([], None, 5.0)["system1"]
    assert empty["observed_hz"] is None and empty["compute_us"]["n"] == 0
    # a window only looks at recent records
    assert compute_metrics(recs, None, 0.015, now=0.04)["system1"]["compute_us"]["n"] == 2


# ------------------------------------------------------------------------------------------------ end to end, in real time
def test_live_stream_during_a_system_2_hang_shows_system_1_unaffected(live):
    """The M5 claim, observed through the API a viewer would use: System 2 hangs 3 s mid-carry, the stream keeps delivering frames
    and phases, System 1 stays at ~100 Hz with no deadline misses, and the task still succeeds."""
    frames, phases, healthz_during = [], set(), []
    with httpx.Client(base_url=live.url, timeout=30) as c:
        tid = c.post("/tasks", json={**GOOD, "faults": {"system2_hang_s": 3}}).json()["task_id"]
        t0 = time.time()
        with c.stream("GET", "/stream", params={"hz": 10}) as stream:
            for line in stream.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("frame"):
                    frames.append(ev["frame"])
                task = ev.get("task") or {}
                if task.get("phase"):
                    phases.add(task["phase"])
                if task.get("status") == "running" and task.get("phase") == "transport" and not healthz_during:
                    h = c.get("/healthz")
                    healthz_during.append((h.status_code, h.json()["status"]))
                if task.get("status") in ("succeeded", "failed", "error") or time.time() - t0 > 60:
                    break
        final = wait_done(c, tid)
        m = c.get("/metrics").json()

    assert final["status"] == "succeeded", final
    assert len(frames) >= 20, "the viewer must keep receiving new frames"
    jpeg = base64.b64decode(frames[-1].split(",", 1)[1])
    assert jpeg[:2] == b"\xff\xd8", "frames are JPEG"
    assert {"approach", "transport"} <= phases
    assert healthz_during and healthz_during[0] == (200, "ticking")
    s1, s2 = m["system1"], m["system2"]
    assert s2["cycle_duration_ms"]["max"] >= 2900, "System 2's hang must be visible in its own numbers"
    assert s1["deadline_misses_total"] == 0
    assert s1["observed_hz"] >= 95
    assert s1["tick_interval_ms"]["max"] < 60, "no stall anywhere in the run"


def test_a_run_frees_its_gl_context_instead_of_leaving_it_to_the_garbage_collector():
    """Regression (found in the Docker image): a render context destroyed by the GC in the middle of the next run corrupted software-GL
    frames (wrapped, colour channels swapped) and made about one run in five fail. Hosts are single-use and must close it themselves."""
    from policyserve.host import PickPlaceHost

    host = PickPlaceHost()
    host.abort()                                   # end the run at once: this test is about cleanup, not the task
    host.run()
    with pytest.raises(Exception):
        host.renderer.render()                     # a closed renderer cannot render
    host.close()                                   # and closing again is harmless
