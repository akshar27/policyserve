"""M5: fault injection against the whole closed loop. These are the tests the architecture exists for -- System 2 can stall for
seconds, System 1 can blow its own compute budget, and neither is allowed to disturb what the arm is doing.

All three run the real Rust binary against the real MuJoCo scene in wall-clock time, so they are slow (~25-40 s each).
"""

import pytest

from policyserve.faults import HangingReasoner, at_call, once_holding
from policyserve.host import PickPlaceHost
from policyserve.wire import PHASES

from .test_closed_loop import EXPECTED_ORDER

RATE_HZ = 100


def _window(log, t0, t1, col=0):
    """Rows of a host log whose wall-clock stamp (column ``col``; the trace keeps it last) falls in [t0, t1]."""
    return [e for e in log if t0 <= e[col] <= t1]


def _hang_run(trigger, hang_s, timeout_sim_s=60):
    host = PickPlaceHost()
    host.reasoner = HangingReasoner(host.reasoner, hang_s, trigger)
    return host.reasoner, host.run(timeout_sim_s=timeout_sim_s)


@pytest.mark.parametrize("name, trigger, hang_s", [
    pytest.param("carrying-the-block", once_holding, 6.0, id="hang-while-carrying"),
    pytest.param("during-approach", at_call(2), 5.0, id="hang-mid-motion"),
])
def test_system_1_keeps_its_rate_and_its_grip_while_system_2_is_hung(name, trigger, hang_s):
    reasoner, r = _hang_run(trigger, hang_s)
    s1 = r.system1
    hs, he = reasoner.hang_started, reasoner.hang_ended
    assert hs is not None and he is not None and he - hs == pytest.approx(hang_s, abs=0.3), "the hang did not happen as injected"

    # 1. System 1 ticked at its own rate for the whole hang. Counted from the outside: how far its tick counter advanced.
    acts = _window(r.action_log, hs, he)
    ticks = acts[-1][1] - acts[0][1]
    expected = (acts[-1][0] - acts[0][0]) * RATE_HZ
    assert ticks == pytest.approx(expected, rel=0.03), f"{ticks} ticks in {acts[-1][0] - acts[0][0]:.2f}s"
    # ... and from the inside: the loop never stalled (target interval 10 ms).
    assert s1["loop_dt_ns_p99"] < 15_000_000 and s1["loop_dt_ns_max"] < 60_000_000, s1
    assert s1["deadline_misses"] == 0 and s1["tick_ns_p99"] < 2_000_000

    # 2. It really did sit on a goal that stopped changing for about that long.
    assert s1["max_goal_age_s"] >= hang_s - 0.6, s1["max_goal_age_s"]

    # 3. The arm did not wander and did not let go: during the quiet tail of the hang the end effector is still, and a carried
    #    block stays in the gripper, off the table.
    tail = _window(r.trace, he - 3.0, he, col=-1)
    assert tail, "no trace samples in the tail of the hang"
    xs, zs = [t[2] for t in tail], [t[3] for t in tail]
    assert max(xs) - min(xs) < 0.004 and max(zs) - min(zs) < 0.004, (min(xs), max(xs), min(zs), max(zs))
    assert {PHASES[a[2]] for a in _window(r.action_log, he - 3.0, he)} == {"wait_goal"}
    if name == "carrying-the-block":
        assert all(0.030 <= t[4] <= 0.046 for t in tail), "gripper opened while System 2 was hung"
        assert all(t[6] > 0.05 for t in tail), "the block was dropped while System 2 was hung"

    # 4. When System 2 comes back the task completes, in the usual order, with nothing re-done: the stale answer it returns
    #    (computed from a pre-hang frame) is rejected by the freshness rule instead of being obeyed.
    assert r.success, f"{r.system2_status} {r.system2_failure} block={r.red_block} {s1.get('final_phase')} {s1.get('failure')}"
    assert r.phases() == EXPECTED_ORDER and s1["grasp_retries"] == 0


def test_blowing_the_tick_budget_trips_the_fallback_and_is_counted():
    """System 1's own fault: every 25th controller call is made 8 ms slower against a 5 ms budget. Each one must be discarded
    for the zero-velocity fallback, counted, and reported with its real duration; ordinary ticks stay fast; the task survives."""
    r = PickPlaceHost(system1_args=("--inject-slow-every", "25", "--inject-slow-us", "8000")).run(timeout_sim_s=45)
    s1 = r.system1
    assert s1["injected_slow_ticks"] >= 20
    assert s1["deadline_misses"] == s1["injected_slow_ticks"], "every injected overrun must be caught, and nothing else"
    assert s1["tick_ns_max"] >= 8_000_000          # the real duration of the slow ticks is what gets reported
    assert s1["tick_ns_p50"] < 200_000             # ordinary ticks are unaffected

    # Seen from the host (its reads are not synchronised with System 1's ticks, so it only catches some of them): every action it
    # read that reports a compute time over the 5 ms budget carries the safe fallback -- zero velocity -- not the computed command.
    slow = [e for e in r.action_log if e[4] > 5_000_000]
    assert len(slow) >= 3, "host saw too few over-budget ticks to judge the fallback"
    assert all(e[5] == (0.0, 0.0, 0.0) for e in slow), [e[5] for e in slow if e[5] != (0.0, 0.0, 0.0)]
    assert all(e[4] >= 8_000_000 for e in slow)     # and the recorded duration is the real, injected one
    assert r.action_log[-1][3] <= s1["deadline_misses"]

    # A one-tick hold every quarter second slows the arm slightly but does not derail the task.
    assert r.success, f"{r.system2_status} {r.system2_failure} {s1.get('final_phase')} {s1.get('failure')}"
    assert r.phases() == EXPECTED_ORDER
