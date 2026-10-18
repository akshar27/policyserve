"""The goal that System 2 publishes and System 1 reads: one fixed-size record through the seqlock channel.

Layout (little-endian, 40 bytes):  source_cycle u64 | kind u8 | 3 pad | frame_step u32 | x f64 | z f64 | updated_at f64
``frame_step`` is the sim step at which the camera frame this goal was computed from was captured. System 1 uses it to
refuse goals computed from a frame older than the moment it finished its last sequence (see src/task.rs), which is a causal
ordering that needs no shared wall clock. ``updated_at`` is System 2's monotonic publish time (informational only: the Rust
side measures staleness itself, by when it last saw ``source_cycle`` change). The matching Rust decoder is src/wire.rs.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

_GOAL = struct.Struct("<QB3xIddd")
GOAL_PAYLOAD_SIZE = _GOAL.size  # 40


class GoalKind(IntEnum):
    NONE = 0     # nothing published yet
    REACH = 1    # move the end effector to (x, z) to grasp
    PLACE = 2    # move the end effector to (x, z) to release
    DONE = 3     # task complete
    FAILED = 4   # task failed; System 1 holds position


@dataclass(frozen=True)
class GoalState:
    source_cycle: int
    kind: GoalKind
    x: float = 0.0
    z: float = 0.0
    updated_at: float = 0.0
    frame_step: int = 0

    def encode(self) -> bytes:
        return _GOAL.pack(self.source_cycle, int(self.kind), self.frame_step & 0xFFFFFFFF, self.x, self.z, self.updated_at)

    @classmethod
    def decode(cls, data: bytes) -> "GoalState":
        cycle, kind, step, x, z, ts = _GOAL.unpack(data)
        return cls(cycle, GoalKind(kind), x, z, ts, step)

    def age(self, now: float) -> float:
        return now - self.updated_at
