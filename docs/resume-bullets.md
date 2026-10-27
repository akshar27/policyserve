# Resume bullets (policyserve)

Every claim below traces to code and a measurement in this repo.

- Built a dual-rate robot-policy serving layer: a 100 Hz Rust controller (hard 5 ms per-tick budget, zero-velocity fallback on overrun)
  and a ~2 Hz Python reasoner linked by a lock-free seqlock over POSIX shared memory. In a MuJoCo pick-and-place, hanging the reasoner
  for 6 s mid-carry cost zero deadline misses (599 ticks in 6.01 s) and the task still finished; 40 of 40 injected overruns were caught.
- Found and fixed two concurrency bugs that only showed up under load or on another OS: an ARM64 memory-ordering bug in the seqlock (torn
  reads in 7 of 15 runs, 0 in 30 after explicit fences) and a garbage-collected OpenGL context that corrupted the next run's camera
  frames under software rendering in Docker (about 1 run in 5 failing, 0 in 25 after closing it deterministically).
- Shipped a FastAPI server with a Server-Sent-Events live viewer (frame, goal, p50/p99 tick timing, deadline misses) whose latency
  numbers are observed independently of the controller, plus an opt-in multimodal-LLM planner behind the same interface (forced tool
  call, validated outputs, no model-produced coordinates, call budget), Docker Compose and CI; 92 automated tests (70 Python, 22 Rust).
