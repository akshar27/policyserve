"""The goal that System 2 publishes and System 1 reads: one fixed-size record through the seqlock channel.

Layout (little-endian, 40 bytes):  source_cycle u64 | kind u8 | 7 pad | x f64 | z f64 | updated_at f64
``updated_at`` is a monotonic timestamp (seconds) so System 1 can tell how stale the goal is. M4 adds the matching Rust
decoder; until then this Python encoding is the single definition of the format.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

_GOAL = struct.Struct("<QB7xddd")
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

    def encode(self) -> bytes:
        return _GOAL.pack(self.source_cycle, int(self.kind), self.x, self.z, self.updated_at)

    @classmethod
    def decode(cls, data: bytes) -> "GoalState":
        cycle, kind, x, z, ts = _GOAL.unpack(data)
        return cls(cycle, GoalKind(kind), x, z, ts)

    def age(self, now: float) -> float:
        return now - self.updated_at
