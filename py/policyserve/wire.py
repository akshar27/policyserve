"""Python side of the two records the sim host exchanges with System 1 in the M4 control mode (the goal record is in goal.py).
The authoritative layout comments are in ``src/wire.rs``; ``tests/test_closed_loop.py`` pushes real records through the compiled
binary so the two definitions cannot drift apart silently.

Proprioception (host -> System 1), 80 bytes:  sim_step u64 | sim_time f64 | q[3] f64 | qd[3] f64 | gap f64 | flags u32 | 4 pad
Action (System 1 -> host), 64 bytes:          tick u64 | vel[3] f64 | grip_open f64 | phase u32 | deadline_misses u32 |
                                              compute_ns u64 | goal_cycle u64
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

_PROPRIO = struct.Struct("<Qd3d3ddI4x")
_ACTION = struct.Struct("<Q3ddIIQQ")
PROPRIO_SIZE = _PROPRIO.size   # 80
ACTION_SIZE = _ACTION.size     # 64
FLAG_STOP = 1

# Index = the u32 code System 1 publishes (src/task.rs `Phase`, declaration order).
PHASES = ("home", "wait_goal", "approach", "descend", "grasp", "lift", "transport", "lower", "release", "retreat",
          "recover", "finished", "failed")


@dataclass(frozen=True)
class Proprio:
    sim_step: int
    sim_time: float
    q: tuple[float, float, float]
    qd: tuple[float, float, float]
    gap: float
    flags: int = 0

    def encode(self) -> bytes:
        return _PROPRIO.pack(self.sim_step, self.sim_time, *self.q, *self.qd, self.gap, self.flags)

    @classmethod
    def decode(cls, data: bytes) -> "Proprio":
        step, t, q0, q1, q2, d0, d1, d2, gap, flags = _PROPRIO.unpack(data)
        return cls(step, t, (q0, q1, q2), (d0, d1, d2), gap, flags)


@dataclass(frozen=True)
class Action:
    tick: int
    vel: tuple[float, float, float]
    grip_open: float
    phase: int
    deadline_misses: int
    compute_ns: int
    goal_cycle: int

    def encode(self) -> bytes:
        return _ACTION.pack(self.tick, *self.vel, self.grip_open, self.phase, self.deadline_misses, self.compute_ns,
                            self.goal_cycle)

    @classmethod
    def decode(cls, data: bytes) -> "Action":
        tick, v0, v1, v2, grip, phase, misses, ns, cycle = _ACTION.unpack(data)
        return cls(tick, (v0, v1, v2), grip, phase, misses, ns, cycle)

    @property
    def phase_name(self) -> str:
        return PHASES[self.phase] if 0 <= self.phase < len(PHASES) else f"?{self.phase}"
