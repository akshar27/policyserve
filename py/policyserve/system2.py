"""System 2's own fixed-cadence loop. It shares data with System 1 (through the goal channel) but never control flow:
whatever happens here - a slow cycle, an exception, garbage output - can only ever leave the published goal unchanged.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .goal import GoalKind, GoalState
from .reasoner import Done, Failed, Proprioception, Reasoner, Subgoal


@dataclass(frozen=True)
class CycleRecord:
    cycle: int
    started_at: float      # monotonic
    duration_s: float
    late: bool             # the previous cycle overran the period, so this one started late
    outcome: str           # "subgoal" | "done" | "failed" | "error" | "malformed"


class System2Loop:
    def __init__(self, reasoner: Reasoner, frame_fn: Callable[[], np.ndarray], proprio_fn: Callable[[], Proprioception],
                 instruction: str, goal_channel, period_s: float = 0.5, max_consecutive_failures: int = 3,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 history: int = 1024, step_fn: Callable[[], int] = lambda: 0,
                 republish_terminal: bool = False):
        self.reasoner, self.frame_fn, self.proprio_fn, self.step_fn = reasoner, frame_fn, proprio_fn, step_fn
        self.instruction, self.channel, self.period_s = instruction, goal_channel, period_s
        self.max_consecutive_failures = max_consecutive_failures
        # False (M3 default): the loop ends at the first DONE/FAILED. True (M4 closed loop): it keeps cycling and republishing, because
        # System 1 only believes goals computed from frames captured after its last sequence ended -- a single DONE published a
        # moment too early would be (correctly) ignored forever. The host stops the loop once System 1 has acted on it.
        self.republish_terminal = republish_terminal
        self._clock, self._sleep = clock, sleep
        self.records: deque[CycleRecord] = deque(maxlen=history)
        self.cycle = 0
        self.consecutive_failures = 0
        self.total_failures = 0
        self.late_cycles = 0
        self.status = "idle"   # idle | running | done | failed | stopped
        self.last_failure: str | None = None
        self.last_goal: dict | None = None   # what was last published, for observers (the HTTP API / viewer)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- one cycle (public so tests can drive it deterministically) ---------------------------------------------
    def run_cycle(self, late: bool = False) -> CycleRecord:
        t0 = self._clock()
        self.cycle += 1
        outcome = "subgoal"
        self._frame_step = 0
        try:
            frame = self.frame_fn()
            self._frame_step = self.step_fn()   # read right after the frame so the two stay paired
            result = self.reasoner.plan(frame, self.instruction, self.proprio_fn())
            outcome = self._publish(result, t0)
        except Exception as e:  # noqa: BLE001 - a reasoner may fail in any way; none of them may touch System 1
            outcome = "error"
            self._fail(f"{type(e).__name__}: {e}", t0)
        rec = CycleRecord(self.cycle, t0, self._clock() - t0, late, outcome)
        self.records.append(rec)
        return rec

    def _publish(self, result, now: float) -> str:
        if isinstance(result, Subgoal) and result.kind in ("reach", "place") and math.isfinite(result.x) and math.isfinite(result.z):
            kind = GoalKind.REACH if result.kind == "reach" else GoalKind.PLACE
            self.channel.write(GoalState(self.cycle, kind, result.x, result.z, now, self._frame_step).encode())
            self.last_goal = {"kind": result.kind, "x": result.x, "z": result.z}
            self.consecutive_failures = 0
            if self.republish_terminal:
                self.status = "running"
            return "subgoal"
        if isinstance(result, Done):
            self.channel.write(GoalState(self.cycle, GoalKind.DONE, updated_at=now, frame_step=self._frame_step).encode())
            self.consecutive_failures = 0
            self.status = "done"
            self.last_goal = {"kind": "done"}
            return "done"
        if isinstance(result, Failed):
            self.channel.write(GoalState(self.cycle, GoalKind.FAILED, updated_at=now, frame_step=self._frame_step).encode())
            self.last_failure = result.reason
            self.status = "failed"
            self.last_goal = {"kind": "failed", "reason": result.reason}
            return "failed"
        self._fail(f"malformed reasoner output: {result!r}", now)
        return "malformed"

    def _fail(self, reason: str, now: float) -> None:
        """Reasoner crashed or returned garbage: leave the published goal exactly as it was, count it, and give up on the
        task after N in a row (System 1 keeps driving toward the last good goal regardless)."""
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure = reason
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.channel.write(GoalState(self.cycle, GoalKind.FAILED, updated_at=now, frame_step=self._frame_step).encode())
            self.status = "failed"

    # ---- fixed-cadence loop ------------------------------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> None:
        """Blocking loop on a fixed schedule. A cycle that overruns the period makes the NEXT one start late (no catch-up burst)."""
        self.status = "running"
        next_start = self._clock()
        late = False
        while not self._stop.is_set() and (self.status == "running" or self.republish_terminal):
            self.run_cycle(late)
            if max_cycles is not None and self.cycle >= max_cycles:
                break
            next_start += self.period_s
            now = self._clock()
            if now < next_start:
                self._stop.wait(next_start - now) if self._sleep is time.sleep else self._sleep(next_start - now)
                late = False
            else:
                late = True
                self.late_cycles += 1
                next_start = now
        if self.status == "running":
            self.status = "stopped"

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="system2", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
