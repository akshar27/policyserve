"""M1's actual proof: Rust and Python handing data to each other through
the seqlock channel under real concurrent load, in both directions,
with no corruption and no stalling. This is the cross-language half of
`src/shm.rs`'s own Rust-only concurrency test -- that one proves the
protocol is correct within one process; this one proves it's correct
across the process/language boundary this whole project is built on.
"""

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from policyserve.shm import SeqLockChannel

REPO_ROOT = Path(__file__).resolve().parents[2]
PAYLOAD_SIZE = 16  # must match main.rs's PAYLOAD_SIZE
RATE_HZ = 100
DURATION_SECS = 2.0


def _system1_binary() -> Path:
    release = REPO_ROOT / "target" / "release" / "system1"
    debug = REPO_ROOT / "target" / "debug" / "system1"
    if release.exists():
        return release
    if debug.exists():
        return debug
    raise RuntimeError("system1 binary not found -- run `cargo build` first")


def _encode(counter: int) -> bytes:
    return counter.to_bytes(8, "little") * 2


def _decode(data: bytes) -> int | None:
    a = int.from_bytes(data[0:8], "little")
    b = int.from_bytes(data[8:16], "little")
    return a if a == b else None


def test_rust_and_python_exchange_data_in_both_directions_without_corruption():
    suffix = uuid.uuid4().hex[:8]
    py_to_rust_name = f"policyserve_test_p2r_{suffix}"
    rust_to_py_name = f"policyserve_test_r2p_{suffix}"

    py_to_rust = SeqLockChannel(py_to_rust_name, PAYLOAD_SIZE, create=True)
    rust_to_py = SeqLockChannel(rust_to_py_name, PAYLOAD_SIZE, create=True)

    proc = subprocess.Popen(
        [
            str(_system1_binary()),
            "--read-channel", py_to_rust_name,
            "--write-channel", rust_to_py_name,
            "--rate-hz", str(RATE_HZ),
            "--duration-secs", str(DURATION_SECS),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={"RUST_LOG": "error"},
    )

    stop = threading.Event()

    def write_loop():
        counter = 0
        # Deliberately faster than Rust's own 100Hz tick rate, so the
        # reader is guaranteed to see genuinely fresh values every tick
        # rather than coincidentally always catching the same one.
        while not stop.is_set():
            py_to_rust.write(_encode(counter))
            counter += 1
            time.sleep(0.002)

    writer_thread = threading.Thread(target=write_loop, daemon=True)
    writer_thread.start()

    read_values = []
    torn_or_missing = 0
    deadline = time.monotonic() + DURATION_SECS + 1.0
    while time.monotonic() < deadline and proc.poll() is None:
        data = rust_to_py.read()
        if data is None:
            torn_or_missing += 1
            continue
        value = _decode(data)
        if value is None:
            raise AssertionError("torn read from Rust: mirrored halves disagreed")
        read_values.append(value)
        time.sleep(0.005)

    stop.set()
    writer_thread.join(timeout=2)

    stdout, stderr = proc.communicate(timeout=10)
    assert proc.returncode == 0, f"system1 exited non-zero: {stderr}"

    py_to_rust.close(unlink=True)
    rust_to_py.close(unlink=True)

    summary = json.loads(stdout.strip().splitlines()[-1])
    print("system1 summary:", summary)

    # The real assertions: System 1 actually ran at roughly its target
    # rate for the full duration (not stalled, not short-circuited)...
    expected_ticks = RATE_HZ * DURATION_SECS
    assert summary["ticks"] >= expected_ticks * 0.9, (
        f"expected ~{expected_ticks} ticks, got {summary['ticks']}"
    )
    # ...it actually saw Python's writes change over time, not just the
    # first one forever (which would mean the channel wasn't live)...
    assert summary["distinct_read_values"] > 10, (
        f"system1 only saw {summary['distinct_read_values']} distinct values -- "
        "looks stale, not live"
    )
    # ...and Python's own read of Rust's output was predominantly
    # successful and strictly increasing (the actual "latest value wins,
    # no corruption" property this test exists to prove).
    assert len(read_values) > 50, f"too few successful reads: {len(read_values)}"
    assert read_values == sorted(read_values), "read values were not monotonically increasing"
    assert torn_or_missing < len(read_values) * 0.1, (
        f"too many torn/missing reads: {torn_or_missing} out of {len(read_values) + torn_or_missing}"
    )
