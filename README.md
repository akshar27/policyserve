# PolicyServe

A real-time serving layer for robot control policies implementing the
dual-rate "System 2 / System 1" architecture used in production
vision-language-action (VLA) robotics stacks: a slow multimodal
reasoner decides *what to do next*, a fast local controller decides
*how to move right now*, and the two run decoupled so the fast loop
never stalls waiting on the slow one.

**Status:** milestones 1-4 done: a Rust fast loop (System 1) and a Python slow reasoner (System 2) complete a simulated
pick-and-place together through shared memory (see `docs/milestones.md`); fault injection, an HTTP API and a live viewer
come next.

```bash
cargo build --release
cd py && python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/python -m pytest tests -q
.venv/bin/python -c "from policyserve.host import PickPlaceHost; r = PickPlaceHost().run(); print(r.success, r.sim_time, r.system1['transitions'])"
```
