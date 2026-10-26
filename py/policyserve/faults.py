"""M5 fault injection for System 2. Wraps a real reasoner so the failure happens in the real System 2 thread of the real closed
loop, not in a mock: the point is to show that System 1 does not notice.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

from .reasoner import Proprioception


class HangingReasoner:
    """Delegates to ``inner`` until ``trigger(proprio, call_index)`` is first true, then blocks for ``hang_s`` real seconds inside
    ``plan`` (a stuck model server, a GPU stall, a swapped-out process) and afterwards answers normally -- from the frame it was
    handed *before* the hang, so the answer is as stale as a real one would be.

    ``hang_started`` / ``hang_ended`` are ``time.perf_counter()`` stamps (the host's clock), so tests can look at exactly what
    System 1 did inside that window. ``release`` ends the hang early (test teardown).
    """

    def __init__(self, inner, hang_s: float, trigger: Callable[[Proprioception, int], bool]):
        self.inner, self.hang_s, self.trigger = inner, hang_s, trigger
        self.calls = 0
        self.hang_started: float | None = None
        self.hang_ended: float | None = None
        self.release = threading.Event()

    def plan(self, frame, instruction, proprio):
        self.calls += 1
        if self.hang_started is None and self.trigger(proprio, self.calls):
            self.hang_started = time.perf_counter()
            self.release.wait(self.hang_s)
            self.hang_ended = time.perf_counter()
        return self.inner.plan(frame, instruction, proprio)


def at_call(n: int) -> Callable[[Proprioception, int], bool]:
    """Hang on the n-th reasoner call (1-based)."""
    return lambda proprio, call: call >= n


def once_holding(proprio: Proprioception, call: int) -> bool:
    """Hang on the first call made while the gripper is closed on the block, i.e. in the middle of the carry."""
    return proprio.holding
