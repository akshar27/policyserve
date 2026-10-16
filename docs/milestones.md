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

## M2 — System 1: closed-form controller + hard deadline enforcement ✅ (2026-10-05)

- [x] Python skeleton: a MuJoCo launcher that loads a plane + a simple
      arm, steps physics, no control logic yet (carried over from M1,
      deferred so M1 could prove the shared-memory handoff in true
      isolation first).
- [x] A closed-form controller (joint-space proportional control toward
      a target configuration — no full inverse kinematics) in Rust.
- [x] Hard per-step deadline enforcement with a safe fallback action
      (hold position) when exceeded; every step's duration recorded.
- [x] Tests: the controller actually converges toward a target on a
      known arm configuration; an artificially slowed-down compute path
      actually triggers the fallback and gets logged, not just assumed.

`py/policyserve/models/simple_arm.xml`: a 3-joint planar arm, one
velocity actuator per joint (so the control signal this scene expects
is exactly the joint-velocity command System 1's controller already
computes — no unit conversion needed when the two get wired together
in M4). `src/controller.rs`: `ProportionalController` computes
`clamp(kp * (target - current), ±max_velocity)` per joint independently
(no cross-joint coupling, matching "joint-space, not task-space"
control); `SafeController<C>` wraps *any* `JointController` with the
hard deadline from design doc §8.2 -- a step whose compute exceeds its
budget is logged as a miss and its action discarded for a safe
fallback, never applied even if the compute technically finished by
the time `step()` returns.

**A real numerical-instability bug found by actually stepping the
sim, not assumed stable from the MJCF**: the first version of
`simple_arm.xml` (explicit per-geom `mass` overrides, the default
semi-implicit Euler integrator) diverged within ~20 simulation steps
under a sustained velocity command — `qpos`/`qvel` went from sane
values to the thousands in well under half a second, confirmed by
actually tracing `qpos`/`qvel`/`qacc` step by step rather than only
checking the final state. The velocity actuators combined with joint
damping form a stiff system the default integrator can't handle at
this timestep; switching to MuJoCo's `implicitfast` integrator and
letting MuJoCo auto-compute geom mass/inertia from density (removing
the manual overrides) fixed it, verified by the same step-by-step
trace and a dedicated regression test (`test_zero_velocity_command_
does_not_blow_up_under_gravity`) that asserts sanity at every step
across a full second of sim time, not just the end.

**One real test-expectation miss, corrected after measuring rather than
assuming**: an early version of the sim test expected a sustained
velocity command to keep accumulating joint displacement linearly over
300 steps. The actual (correct, physically real) behavior of this
scene's damping/actuator combination is to settle into a new
*equilibrium position* within well under a second rather than sustain
indefinite velocity — confirmed by tracing `qvel` decaying back toward
zero around step 150. The test was rewritten to check for real,
measurable movement within the active transient window instead of a
specific displacement after the system has already settled.

The deadline-enforcement test (`a_slow_compute_triggers_the_safe_
fallback_and_reports_its_real_duration`) wraps the real controller in a
`SlowController` that sleeps a genuine, measured 20ms before computing,
against a 5ms budget — proving the fallback path with an actual timing
violation, not a mocked one, and asserting the reported `duration`
reflects that real delay rather than trusting the branch read correctly.

7 Rust tests + 5 Python tests (12 total), `fmt`/`clippy --all-targets
-- -D warnings` clean.

## M3 — System 2: pluggable reasoner + the scripted deterministic planner ✅ (2026-10-16)

- [x] The `Reasoner` interface; a `ScriptedReasoner` that locates a
      named-color block via a simple color-blob heuristic on the sim's
      camera frame and produces subgoal poses for a pick/place sequence.
- [x] System 2's own fixed-cadence loop, writing into the shared goal
      channel from M1.
- [x] Tests: correct subgoals for known scene configurations; cadence
      timing respected.

`py/policyserve/reasoner.py`: `Reasoner.plan(frame, instruction, proprio) -> Subgoal | Done | Failed`, an
HSV colour-blob finder, an instruction parser ("pick up the red block and place it on the green platform"), and
`ScriptedReasoner` with four ordered rules (cannot parse / object not visible -> `Failed`; block resting on the
platform -> `Done`; end effector already at the block -> `place` subgoal above the platform; otherwise `reach` the
block). `py/policyserve/goal.py` defines the 40-byte goal record that goes through the M1 seqlock channel
(cycle, kind, x, z, monotonic timestamp). `py/policyserve/system2.py`'s `System2Loop` runs on a fixed schedule: a cycle
that overruns the period makes the *next* one start late, with no catch-up burst and no overlap; an exception,
malformed output or NaN leaves the published goal untouched and is counted, and N consecutive failures publish a
`FAILED` sentinel and end the task (System 1 never depends on any of it).

`py/policyserve/models/pick_place_arm.xml`: the M2 arm in neutral gray (the M2 model's red/green/blue links would be
mistaken for blocks by a colour heuristic), a red and a blue block, a green platform, and a fixed **orthographic**
side camera, so pixel -> world (x, z) is an exact linear map and the tests assert recovered positions against known
truth (centroids within 1 cm; checked to <0.5 px against the camera maths).

**Things found by running it, not assumed:**
1. MuJoCo 3.14 rejects `orthographic="true"` on `<camera>`; the attribute is `projection="orthographic"`.
2. A flaky test of my own (4 of 5 runs): the "reader thread sees goals while the loop runs" test raced the reader's last
   poll against the loop's last write (saw cycles 0-14, expected 15). Test bug, not product bug; the final state is now
   asserted directly from the channel, and the suite ran 8/8 clean afterwards.

**Known limits, stated plainly:** colour-blob detection only works in this controlled scene (it is a determinism
fixture, not a detector); the Rust side does not decode the goal record yet - that lands with the M4 integration;
"carrying" is inferred from the end effector being at the block, because the 3-joint arm has no gripper until M4's state
machine models the grasp.

12 new Python tests (24 total) + 7 Rust tests; Rust `fmt`/`clippy -D warnings` unchanged and clean.

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
