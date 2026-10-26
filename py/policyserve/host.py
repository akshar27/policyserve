"""The M4 sim host: owns the MuJoCo world and wires the whole system together.

    camera frame ──► System 2 (Python thread, ~2 Hz, ScriptedReasoner) ──goal──► shared memory ──► System 1 (Rust process, 100 Hz)
    MuJoCo (this thread, paced to wall-clock) ◄──action── shared memory ◄── System 1;   proprioception ──► System 1

This thread is the only one that touches MuJoCo (including the offscreen renderer, which on macOS must stay on one thread):
it renders a frame every ``render_every`` control ticks into a snapshot, and System 2's thread only ever reads that snapshot.
System 1 gets its time from the ``sim_time`` field of each proprioception record, not from a clock, so a slow host tick makes
the simulation lag in wall-clock terms without changing what the controller computes.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .goal import GOAL_PAYLOAD_SIZE, GoalKind, GoalState
from .reasoner import OrthoCamera, Proprioception, ScriptedReasoner
from .shm import SeqLockChannel
from .system2 import System2Loop
from .wire import ACTION_SIZE, FLAG_STOP, PHASES, PROPRIO_SIZE, Action, Proprio

MODEL_PATH = Path(__file__).parent / "models" / "pick_place_gripper.xml"
REPO_ROOT = Path(__file__).resolve().parents[2]
INSTRUCTION = "pick up the red block and place it on the green platform"

GRIP_OPEN_CTRL, GRIP_CLOSED_CTRL = 0.02, -0.015
HOLD_BAND = (0.030, 0.046)      # pad gap that means "closed on a block-sized object"; same numbers as src/task.rs
PLATFORM_HALF_X, PLATFORM_TOP = 0.07, 0.02
BLOCK_HALF = 0.02


def system1_binary() -> Path:
    for profile in ("release", "debug"):
        p = REPO_ROOT / "target" / profile / "system1"
        if p.exists():
            return p
    raise RuntimeError("system1 binary not found -- run `cargo build --release` first")


@dataclass
class TaskResult:
    success: bool
    sim_time: float
    wall_time: float
    system2_status: str
    system2_failure: str | None
    red_block: tuple[float, float]
    system1: dict                       # the JSON summary System 1 printed on exit
    system2_cycles: int
    system2_late_cycles: int
    trace: list[tuple] = field(default_factory=list)   # (sim_time, phase, ee_x, ee_z, gap, block_x, block_z, wall)
    # One entry per distinct action record the host read, in wall-clock terms (perf_counter), for judging System 1's behaviour
    # from the outside: (wall, tick, phase code, deadline_misses, compute_ns, (v1, v2, v3))
    action_log: list[tuple] = field(default_factory=list)

    def phases(self) -> list[str]:
        return [name for _, name in self.system1.get("transitions", [])]


class _ActionTap:
    """An independent observer of the action channel (M5): polls it much faster than the 100 Hz control loop and logs every
    distinct action System 1 publishes. The host's own 100 Hz read is phase-locked to System 1's tick and systematically misses
    some records (it never saw the late-written, deadline-missing ones), so judging System 1 from the outside needs its own reader.
    Reads are stateless, so sharing the channel object with the host thread is safe."""

    def __init__(self, channel, log: list, period_s: float = 0.0005):
        self._ch, self._log, self._period = channel, log, period_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="action-tap", daemon=True)

    def _run(self) -> None:
        last = -1
        while not self._stop.is_set():
            raw = self._ch.read(5)
            if raw is not None:
                a = Action.decode(raw)
                if a.tick != last:
                    last = a.tick
                    self._log.append((time.perf_counter(), a.tick, a.phase, a.deadline_misses, a.compute_ns, tuple(a.vel)))
            time.sleep(self._period)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(2.0)


class PickPlaceHost:
    def __init__(self, *, red_x: float = 0.45, blue_x: float = 0.32, platform_x: float = 0.15,
                 start_q: tuple[float, float, float] = (-1.0, 1.9, 0.3), rate_hz: int = 100, render_every: int = 10,
                 s2_period: float = 0.5, instruction: str = INSTRUCTION, reasoner=None, frame_size: tuple[int, int] = (640, 480),
                 deadline_us: int = 5000, system1_args: tuple[str, ...] = ()):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.rate_hz, self.render_every, self.s2_period = rate_hz, render_every, s2_period
        self.instruction, self.deadline_us = instruction, deadline_us
        self.system1_args = tuple(system1_args)     # extra flags for the System 1 binary (M5: --inject-slow-*)
        self.platform_x = platform_x
        self.n_sub = max(1, round(1.0 / rate_hz / self.model.opt.timestep))

        m, d = self.model, self.data
        self._ids = {
            "red": m.body("red_block").id, "grip": m.site("grip").id,
            "fl": m.jnt_qposadr[m.joint("finger_l").id], "fr": m.jnt_qposadr[m.joint("finger_r").id],
            "red_adr": m.jnt_qposadr[m.joint("red_free").id], "blue_adr": m.jnt_qposadr[m.joint("blue_free").id],
        }
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = start_q
        for adr, x in ((self._ids["red_adr"], red_x), (self._ids["blue_adr"], blue_x)):
            d.qpos[adr:adr + 3] = [x, 0.0, BLOCK_HALF]
        m.geom_pos[m.geom("green_platform").id][0] = platform_x
        d.ctrl[3:5] = GRIP_OPEN_CTRL
        d.qpos[self._ids["fl"]] = d.qpos[self._ids["fr"]] = GRIP_OPEN_CTRL
        mujoco.mj_forward(m, d)

        w, h = frame_size
        self.renderer = mujoco.Renderer(m, h, w)
        self.cam = OrthoCamera.from_model(m, "front", w, h)
        self.reasoner = reasoner or ScriptedReasoner(self.cam)

        self._lock = threading.Lock()
        self._snap: dict = {}
        self._grip_open_cmd = 1.0

    # ------------------------------------------------------------------ measurements (host thread only)
    def gap(self) -> float:
        d, i = self.data, self._ids
        return 0.042 + float(d.qpos[i["fl"]]) + float(d.qpos[i["fr"]])

    def ee(self) -> tuple[float, float]:
        p = self.data.site_xpos[self._ids["grip"]]
        return float(p[0]), float(p[2])

    def red_block(self) -> tuple[float, float]:
        p = self.data.xpos[self._ids["red"]]
        return float(p[0]), float(p[2])

    def holding(self) -> bool:
        return HOLD_BAND[0] <= self.gap() <= HOLD_BAND[1] and self._grip_open_cmd < 0.5

    def step_count(self) -> int:
        return int(round(self.data.time / self.model.opt.timestep))

    def _snapshot(self) -> None:
        self.renderer.update_scene(self.data, camera="front")
        frame = self.renderer.render().copy()
        ee = self.ee()
        with self._lock:
            self._snap = {"frame": frame, "step": self.step_count(), "ee": ee, "holding": self.holding()}

    # ------------------------------------------------------------------ System 2's view of the world (other thread)
    def _frame(self) -> np.ndarray:
        with self._lock:
            return self._snap["frame"]

    def _frame_step(self) -> int:
        with self._lock:
            return self._snap["step"]

    def _proprio(self) -> Proprioception:
        with self._lock:
            s = self._snap
            return Proprioception(s["ee"][0], s["ee"][1], s["holding"])

    # ------------------------------------------------------------------ the run
    def run(self, timeout_sim_s: float = 45.0, realtime: bool = True) -> TaskResult:
        tag = uuid.uuid4().hex[:8]
        goal_ch = SeqLockChannel(f"ps_goal_{tag}", GOAL_PAYLOAD_SIZE, create=True)
        prop_ch = SeqLockChannel(f"ps_prop_{tag}", PROPRIO_SIZE, create=True)
        act_ch = SeqLockChannel(f"ps_act_{tag}", ACTION_SIZE, create=True)
        channels = (goal_ch, prop_ch, act_ch)
        proc = None
        s2 = None
        trace: list[tuple] = []
        action_log: list[tuple] = []
        tap = _ActionTap(act_ch, action_log)
        try:
            goal_ch.write(GoalState(0, GoalKind.NONE).encode())
            act_ch.write(Action(0, (0.0, 0.0, 0.0), 1.0, 0, 0, 0, 0).encode())
            self._snapshot()
            prop_ch.write(self._proprio_record(0))

            proc = subprocess.Popen(
                [str(system1_binary()), "--mode", "control", "--goal-channel", goal_ch._shm.name,
                 "--proprio-channel", prop_ch._shm.name, "--action-channel", act_ch._shm.name,
                 "--rate-hz", str(self.rate_hz), "--duration-secs", str(timeout_sim_s * 2 + 30),
                 "--deadline-us", str(self.deadline_us), *self.system1_args],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            s2 = System2Loop(self.reasoner, self._frame, self._proprio, self.instruction, goal_ch,
                             period_s=self.s2_period, step_fn=self._frame_step, republish_terminal=True)
            s2.start()
            tap.start()

            period = 1.0 / self.rate_hz
            wall0 = time.perf_counter()
            tick = 0
            # The run ends when System 1 itself reports finished/failed. If System 2 gave up (FAILED) System 1 should say so within a
            # moment; if it does not, stop anyway after a short grace rather than wait for the timeout. A DONE from System 2 is not an
            # end condition on its own: System 1 only accepts it once its last sequence (the retreat) has ended.
            settle_until = None
            a = None
            while self.data.time < timeout_sim_s:
                if a is not None and a.phase_name in ("finished", "failed"):
                    break
                if s2.status == "failed":
                    settle_until = settle_until if settle_until is not None else self.data.time + 0.5
                    if self.data.time >= settle_until:
                        break
                raw = act_ch.read()
                if raw is not None:
                    a = Action.decode(raw)
                    self.data.ctrl[:3] = a.vel
                    self._grip_open_cmd = a.grip_open
                    g = GRIP_OPEN_CTRL if a.grip_open >= 0.5 else GRIP_CLOSED_CTRL
                    self.data.ctrl[3:5] = g
                else:
                    a = None
                for _ in range(self.n_sub):
                    mujoco.mj_step(self.model, self.data)
                tick += 1
                if tick % self.render_every == 0:
                    self._snapshot()
                    bx, bz = self.red_block()
                    ee = self.ee()
                    trace.append((self.data.time, PHASES[a.phase] if a else "-", ee[0], ee[1], self.gap(), bx, bz, time.perf_counter()))
                prop_ch.write(self._proprio_record(0))
                if realtime:
                    delay = wall0 + tick * period - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
            sim_time, wall = float(self.data.time), time.perf_counter() - wall0
        finally:
            tap.stop()
            if s2 is not None:
                s2.stop()
            summary: dict = {}
            if proc is not None:
                try:
                    prop_ch.write(self._proprio_record(FLAG_STOP))
                    out, err = proc.communicate(timeout=10)
                    if proc.returncode != 0:
                        raise RuntimeError(f"system1 exited {proc.returncode}: {err}")
                    summary = json.loads(out.strip().splitlines()[-1])
                except subprocess.TimeoutExpired:
                    proc.kill()
                    raise
            for ch in channels:
                ch.close(unlink=True)

        bx, bz = self.red_block()
        on_platform = abs(bx - self.platform_x) <= PLATFORM_HALF_X - 0.02 and bz >= PLATFORM_TOP + BLOCK_HALF - 0.005
        return TaskResult(
            success=bool(on_platform and not self.holding() and s2.status == "done"),
            sim_time=sim_time, wall_time=wall, system2_status=s2.status, system2_failure=s2.last_failure,
            red_block=(bx, bz), system1=summary, system2_cycles=s2.cycle, system2_late_cycles=s2.late_cycles, trace=trace,
            action_log=action_log)

    def _proprio_record(self, flags: int) -> bytes:
        d = self.data
        return Proprio(self.step_count(), float(d.time), tuple(map(float, d.qpos[:3])), tuple(map(float, d.qvel[:3])),
                       self.gap(), flags).encode()
