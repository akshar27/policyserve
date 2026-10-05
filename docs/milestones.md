# PolicyServe — Milestones

## M1 — Skeletons + the cross-language non-blocking handoff ✅ (2026-10-04)

- [x] Rust binary skeleton: a loop that ticks at a configurable target
      rate and logs its own timing.
- [x] The actual riskiest unknown, proven first and in isolation: a
      shared-memory "latest value wins" channel (seqlock pattern) that
      Rust and Python can both read/write without blocking or
      corruption, verified bidirectionally under concurrent read/write
      at target rates.

`src/shm.rs`'s `SeqLockChannel` is a classic kernel-style seqlock
(odd/even sequence counter bracketing each write, a bounded-retry
reader that gives up and keeps its last value rather than ever
blocking) implemented over raw POSIX `shm_open`/`mmap`, matched byte-
for-byte by `py/policyserve/shm.py` on the Python side. The core claim
this milestone exists to prove -- that Rust and Python can hand a value
back and forth through shared memory with no IPC round trip, no lock
held across either side's own work, and no corruption -- is verified
three ways: a pure-Rust concurrent torture test (two threads, 200,000
writes, every read's mirrored halves checked for agreement); a
cross-language integration test (`py/tests/test_shm_interop.py`) that
spawns the real compiled `system1` binary as a subprocess, drives it
from a live Python writer thread while reading its own output channel,
and asserts on the real numbers (tick count, distinct values observed,
monotonicity) rather than just "it didn't crash"; and the fact that
rerunning both suites five times back to back never flakes.

**Two real environment/tooling bugs found while building this, not
assumed working:**

1. PyBullet (the sim originally specified in the design doc) fails to
   compile on this machine's Xcode/macOS SDK -- a vendored, outdated
   zlib colliding with a newer `stdio.h`'s own `NULL` macro, confirmed
   by actually running the install, not guessed from PyBullet's
   reputation. Swapped for MuJoCo (real prebuilt wheels, no local
   compile step) everywhere in the design doc; see §4's own note there.
2. macOS's POSIX shared memory has no equivalent of Python's
   `multiprocessing.resource_tracker` -- a segment from a process that
   panics before reaching its own cleanup call just stays around
   forever, and macOS's `ftruncate` additionally refuses to resize an
   already-sized segment a second time (`EINVAL`), so a leaked segment
   from one failed test run made the *next* run's unrelated-looking
   `shm_open`/`ftruncate` call fail with a confusing error far from the
   real cause. Fixed by giving every test a name unique to its own
   process (so no run's segments can ever collide with a previous
   run's leftovers) and moving cleanup onto an `UnlinkGuard`'s `Drop`
   impl, so it still runs even when an `assert!` earlier in the same
   test panics -- caught by the bug actually recurring across repeated
   `cargo test` runs, not anticipated in advance.

3 Rust tests + 1 Python integration test, `fmt`/`clippy --all-targets
-- -D warnings` clean on the Rust side.

## M2 — System 1: closed-form controller + hard deadline enforcement

- [ ] Python skeleton: a MuJoCo launcher that loads a plane + a simple
      arm, steps physics, no control logic yet (carried over from M1,
      deferred so M1 could prove the shared-memory handoff in true
      isolation first).
- [ ] A closed-form controller (joint-space proportional control toward
      a target configuration — no full inverse kinematics) in Rust.
- [ ] Hard per-step deadline enforcement with a safe fallback action
      (hold position) when exceeded; every step's duration recorded.
- [ ] Tests: the controller actually converges toward a target on a
      known arm configuration; an artificially slowed-down compute path
      actually triggers the fallback and gets logged, not just assumed.

## M3 — System 2: pluggable reasoner + the scripted deterministic planner

- [ ] The `Reasoner` interface; a `ScriptedReasoner` that locates a
      named-color block via a simple color-blob heuristic on the sim's
      camera frame and produces subgoal poses for a pick/place sequence.
- [ ] System 2's own fixed-cadence loop, writing into the shared goal
      channel from M1.
- [ ] Tests: correct subgoals for known scene configurations; cadence
      timing respected.

## M4 — Full loop integration: one pick-and-place task, end to end

- [ ] Wire System 1 (Rust) + System 2 (Python, scripted) + the MuJoCo
      sim together into one running task with a real state machine
      (approach → grasp → lift → move → place → release → done).
- [ ] Tests: the task actually completes in sim from a few different
      starting arm/block configurations, not just one lucky run.

## M5 — Fault injection + the deadline/observability proof

- [ ] The single most important test in this project: System 2 is
      artificially hung for several real seconds mid-task; System 1
      must keep ticking at its target rate throughout, acting on its
      last good goal, with no degradation.
- [ ] A deadline-miss injection test for System 1 itself (artificially
      slow a tick's compute) confirming the fallback path and the
      metrics it should produce.

## M6 — HTTP API + a minimal live viewer

- [ ] `POST /tasks`, `GET /tasks/{id}`, `GET /metrics`
      (p50/p95/p99/max + deadline-miss counts per loop), `GET /stream`
      (SSE), `GET /healthz`.
- [ ] A minimal live viewer: current sim frame, current goal, rolling
      latency/deadline-miss readout, updating in real time.

## M7 — The real LLMReasoner + polish

- [ ] `LLMReasoner`: a real multimodal LLM call (frame + instruction +
      structured-output prompt) implementing the same `Reasoner`
      interface as the scripted one, opt-in via config.
- [ ] Docker Compose demo, CI, README (architecture, quickstart, a real
      screenshot/gif of a completed task, design decisions & tradeoffs,
      what's out of scope), `docs/resume-bullets.md`.
