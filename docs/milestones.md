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

## M4 — Full loop integration: one pick-and-place task, end to end ✅ (2026-10-18)

- [x] Wire System 1 (Rust) + System 2 (Python, scripted) + the MuJoCo sim together into one running task with a real state
      machine (approach -> descend -> grasp -> lift -> transport -> lower -> release -> retreat -> done).
- [x] Tests: the task actually completes in sim from several different starting arm/block configurations, not one lucky run.

**What exists.** Three processes/threads, three shared-memory channels (all the M1 seqlock):
`py/policyserve/host.py` (the MuJoCo host: paced to wall-clock at 100 Hz, publishes proprioception, applies System 1's
action, renders the camera for System 2) -> `src/control.rs` + `src/task.rs` (System 1: reads the latest proprioception
and goal, advances the state machine, closed-form IK in `src/kinematics.rs`, per-joint P control under the M2 deadline
wrapper, publishes the action) <- `py/policyserve/system2.py` (System 2 thread at 2 Hz: camera frame -> `ScriptedReasoner`
-> goal). Wire formats: `src/wire.rs` / `py/policyserve/wire.py` / `goal.py`. New scene `models/pick_place_gripper.xml`:
gravity-compensated 3-joint arm with a two-finger parallel gripper (the M2/M3 scenes are untouched).

**Division of labour.** System 2 says *where* (reach (x, z) / place (x, z) / done / failed); System 1 owns *how and when*:
the contact-sensitive sequencing (close the gripper, judge the grasp from the pad gap, notice a drop) happens inside the
10 ms tick and never waits on System 2. Rules that keep a slow or stale System 2 harmless: a sequence is committed once
the arm starts descending; after a sequence ends System 1 accepts only goals computed from a camera frame captured
*after* that moment (`frame_step` in the goal record, compared against the sim step - a causal ordering that needs no shared
clock); every phase has a timeout; a missed grasp retries twice then fails; losing the grip mid-carry aborts to a recover
phase; the end-effector setpoint follows a speed- and acceleration-limited profile.

**Results** (real, from `pytest tests/test_closed_loop.py`, 5 consecutive full runs all green):
4 start-pose/block/platform configurations complete a clean pick-and-place (phase order asserted), in about 7-11 s of sim
time, with 0 deadline misses, 0 grasp retries, System 1 tick compute p99 well under 2 ms of a 10 ms period; a System 2
at 0.7 Hz instead of 2 Hz completes the same task later while System 1 keeps ticking at 100 Hz; an unparseable instruction
makes System 2 publish `FAILED` and System 1 hold position with nothing disturbed. 20 Rust tests, 35 Python tests.

**Real problems found by running it (not assumed):**
1. **The M1 seqlock was wrong on ARM64.** The M1 torn-read test started failing intermittently once M4 added load
   (7 of 15 runs on the old code). A Release *store* does not order the payload writes after it, and an Acquire *load*
   does not order the payload reads before it, so a reader could accept a half-written value. Fixed with a Release fence
   after the odd marker and an Acquire fence before the second sequence read, and volatile byte copies: 0 failures in 30 runs.
   (The M1 write-up's "never flakes" was true only for the lighter load it was measured under.)
2. MuJoCo's default angle unit is degrees, so `range="-3.14159 3.14159"` meant +-3 DEGREES and the arm stalled against an
   invisible joint limit at 0.07 rad. `<compiler angle="radian"/>`.
3. A held block slid ~3.5 cm out of the pads during a gentle transport and was dropped. Friction coefficient, torsional
   friction and grip force made no difference; contact softness did (default `solref` 0.02 s is far too soft for grasping).
   Stiffer contacts (0.005 s) cut the slip to ~8 mm, and 7 cm-wide pads make the remainder harmless.
4. A velocity-only setpoint limit still jerked the block (it pitched ~18 degrees in the pads at the start of a transport);
   added an acceleration limit (trapezoidal profile).
5. System 2 stopped at its first `DONE`, which could be published a moment before System 1 finished its retreat and was
   therefore (correctly) rejected as stale - System 1 never finished. System 2 now keeps republishing a terminal result
   (`republish_terminal`) until the host stops it.

**Known limits, stated plainly:** the scene is deliberately simple (one block, a colour-blob "perception", a gripper that is
a pair of sliding pads, a 3-joint planar arm solved in closed form with the gripper fixed pointing down); 100 Hz real-time
pacing means each run takes as long as it simulates; the Python side of the seqlock still relies on aligned 8-byte
accesses not tearing (documented since M1); fault injection (a hung System 2) is exercised only lightly here and is the
whole point of M5.

## M5 — Fault injection + the deadline/observability proof ✅ (2026-10-26)

- [x] The single most important test in this project: System 2 is artificially hung for several real seconds mid-task;
      System 1 must keep ticking at its target rate throughout, acting on its last good goal, with no degradation.
- [x] A deadline-miss injection test for System 1 itself (artificially slow a tick's compute) confirming the fallback path
      and the metrics it should produce.

**What exists.** `py/policyserve/faults.py`: `HangingReasoner` wraps the real reasoner and blocks for N real seconds inside
`plan` when a trigger fires (while carrying the block, or on the Nth call), then answers from the frame it was handed
*before* the hang, as stale as a real recovery would be. `src/controller.rs`: `SlowEvery` wraps the controller and makes
every Nth call take an extra real delay (`--inject-slow-every` / `--inject-slow-us` on the binary). New observability:
System 1's summary now has the loop's own interval stats (`loop_dt_ns_p50/p99/max`) and `injected_slow_ticks`; the host
has an independent action-tap thread (~2 kHz) so System 1 can be judged from outside its own process.
`py/tests/test_faults.py`, 3 tests, ~40 s, wall-clock.

**Results** (3 consecutive full runs green after the tap fix; numbers from one run):

| Run | Ticks during the fault | Loop interval p50 / p99 / max | Deadline misses | Outcome |
|---|---|---|---|---|
| System 2 hung 6.0 s while carrying the block | 599 in 6.01 s (100.0/s) | 9.99 / 12.46 / 14.08 ms | 0 | block never dropped; task finished (14.6 s sim) |
| System 2 hung 5.0 s mid-approach | 499 in 5.01 s (100.0/s) | 10.00 / 12.29 / 12.51 ms | 0 | committed motion finished, task finished (15.1 s sim) |
| every 25th compute +8 ms, 5 ms budget | 1007 ticks, 40 injected | max 12.5 ms | 40 = 40 injected | each replaced by zero velocity; task finished |

Normal tick compute is a few microseconds (p99 8 us), so the 5 ms budget is never near; the injected overruns report 8.2-10 ms.
`max_goal_age_s` was 6.49 s and 5.51 s: System 1 really did sit on a goal that stopped changing for that long.

**Assertions that carry the claim:** tick counter advanced 100/s +-3% across the hang window, measured from outside; loop
interval p99 under 15 ms and max under 60 ms; 0 deadline misses; end effector moved under 4 mm over the last 3 s of the
hang and stayed in `wait_goal`; while carrying, the pad gap stayed in the hold band and the block stayed off the table; after
recovery the full phase order completes with 0 grasp retries. The stale post-hang answer is ignored by the freshness rule
(`frame_step` older than the end of the last sequence) instead of being obeyed.

**Checks that the tests can fail:** with the budget raised to 20 ms and the same 8 ms injection, 40 overruns are injected
and 0 are counted as misses, so the `misses == injected` assertion is not a tautology.

**Findings (real, from building this):**
1. **My first host-side check of the fallback was wrong, not the fallback.** The sim host reads the action channel at 100 Hz,
   phase-locked to System 1's 100 Hz writes, and it systematically missed the late-written slow actions (it saw 2 of 40, always
   500 ticks apart), so an assertion on what it read failed most runs. A separate ~2 kHz tap thread sees them; it catches
   about a quarter of the slow actions because the next tick overwrites a slow one quickly, which is why the test needs only 3.
2. **`wait_goal` has no timeout, by design.** A hung System 2 leaves System 1 holding position indefinitely; only the other
   phases fail after 12 s. In simulation that is the safe choice. On hardware it should become a watchdog that parks or
   opens the gripper after a bound, which this project does not do.
3. A hang that crosses a sequence boundary is the interesting case: the answer System 2 returns afterwards was computed from
   a pre-hang frame, and it is rejected by the freshness rule. That rule was written in M4; M5 is the first test that
   exercises it against a multi-second stall.

**Intentionally not covered:** a hung System 1 (the sim host would keep applying its last action, which is a different
failure with a different answer), shared-memory corruption beyond the M1 torn-read test, and real-robot timing.

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
