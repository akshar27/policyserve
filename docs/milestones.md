# PolicyServe — Milestones

## M1 — Skeletons + the cross-language non-blocking handoff

- [ ] Rust binary skeleton: a loop that ticks at a configurable target
      rate and logs its own timing.
- [ ] Python skeleton: a PyBullet launcher that loads a plane + a simple
      arm, steps physics, no control logic yet.
- [ ] The actual riskiest unknown, proven first and in isolation: a
      shared-memory "latest value wins" channel (seqlock pattern) that
      Rust and Python can both read/write without blocking or
      corruption, verified bidirectionally under concurrent read/write
      at target rates.

## M2 — System 1: closed-form controller + hard deadline enforcement

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

- [ ] Wire System 1 (Rust) + System 2 (Python, scripted) + the PyBullet
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
