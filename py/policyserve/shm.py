"""The Python side of the seqlock channel in ``src/shm.rs`` -- read that
module's docstring first for the full rationale. Layout and protocol
are identical: ``[seq: u64 little-endian][payload: N bytes]``, a writer
brackets its write with an odd/even sequence number, a reader retries
(bounded) on catching one odd or on the before/after sequence not
matching, and gives up (returns ``None``) rather than ever blocking.

One honest, documented gap versus the Rust side: ``AtomicU64`` gives
Rust a real hardware memory fence around the sequence number. Python has
no equivalent primitive to reach for here. This relies instead on
x86-64/ARM64's own guarantee that an aligned 8-byte load/store doesn't
tear -- true on the hardware this project runs and tests on, but a
genuine simplification worth naming rather than hiding: a production
system mixing languages this way would want that guarantee checked
against its actual target hardware, not assumed from one machine.
"""

from __future__ import annotations

import struct
from multiprocessing import shared_memory

_SEQ_SIZE = 8
_SEQ_STRUCT = struct.Struct("<Q")


class SeqLockChannel:
    """One fixed-size "latest value wins" channel. ``create=True`` makes
    this process the owner -- by this project's convention (see
    ``main.rs``'s module docstring), Python always creates, Rust only
    ever opens; whichever side creates is the only side that should ever
    ``close(unlink=True)``.
    """

    def __init__(self, name: str, payload_size: int, create: bool):
        self.payload_size = payload_size
        total_size = _SEQ_SIZE + payload_size
        self._shm = shared_memory.SharedMemory(name=name, create=create, size=total_size)
        if create:
            # A freshly created segment isn't guaranteed zeroed on every
            # platform -- start from a known-even sequence explicitly
            # rather than relying on that.
            _SEQ_STRUCT.pack_into(self._shm.buf, 0, 0)

    def write(self, data: bytes) -> None:
        if len(data) != self.payload_size:
            raise ValueError(f"expected {self.payload_size} bytes, got {len(data)}")
        (cur,) = _SEQ_STRUCT.unpack_from(self._shm.buf, 0)
        _SEQ_STRUCT.pack_into(self._shm.buf, 0, (cur + 1) & 0xFFFFFFFFFFFFFFFF)
        self._shm.buf[_SEQ_SIZE : _SEQ_SIZE + self.payload_size] = data
        _SEQ_STRUCT.pack_into(self._shm.buf, 0, (cur + 2) & 0xFFFFFFFFFFFFFFFF)

    def read(self, max_retries: int = 50) -> bytes | None:
        for _ in range(max_retries):
            (s1,) = _SEQ_STRUCT.unpack_from(self._shm.buf, 0)
            if s1 % 2 != 0:
                continue  # writer mid-write
            data = bytes(self._shm.buf[_SEQ_SIZE : _SEQ_SIZE + self.payload_size])
            (s2,) = _SEQ_STRUCT.unpack_from(self._shm.buf, 0)
            if s1 == s2:
                return data
            # s1 != s2: a write straddled this read -- retry.
        return None

    def close(self, unlink: bool = False) -> None:
        self._shm.close()
        if unlink:
            self._shm.unlink()
