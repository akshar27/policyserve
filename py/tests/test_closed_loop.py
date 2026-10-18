"""M4: the whole system in one process tree -- MuJoCo host + Python System 2 + the compiled Rust System 1 -- and the pieces of
it that have to agree across the language boundary (wire formats, kinematics)."""

import json
import math
import subprocess

import mujoco
import numpy as np
import pytest

from policyserve.host import MODEL_PATH, PickPlaceHost, system1_binary
from policyserve.wire import ACTION_SIZE, PHASES, PROPRIO_SIZE, Action, Proprio


def _s1(*args: str) -> dict:
    out = subprocess.run([str(system1_binary()), *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def mj():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    return model, mujoco.MjData(model), model.site("grip").id


# ------------------------------------------------------------------------------------------------ wire + kinematics
def test_wire_records_roundtrip_and_have_the_documented_sizes():
    p = Proprio(99, 1.5, (0.1, -0.2, 0.3), (1.0, 2.0, 3.0), 0.04, 1)
    assert len(p.encode()) == PROPRIO_SIZE == 80 and Proprio.decode(p.encode()) == p
    a = Action(5, (0.5, -0.5, 0.25), 1.0, 4, 2, 1234, 8)
    assert len(a.encode()) == ACTION_SIZE == 64 and Action.decode(a.encode()) == a
    assert a.phase_name == "grasp"


def test_rust_fk_matches_mujoco(mj):
    model, data, site = mj
    rng = np.random.default_rng(0)
    for _ in range(12):
        q = rng.uniform(-1.5, 1.5, 3)
        data.qpos[:3] = q
        mujoco.mj_forward(model, data)
        got = _s1("--mode", "fk", f"--joints={','.join(map(str, q))}")
        assert got["x"] == pytest.approx(data.site_xpos[site][0], abs=1e-9)
        assert got["z"] == pytest.approx(data.site_xpos[site][2], abs=1e-9)


def test_rust_ik_puts_the_mujoco_gripper_where_asked_pointing_down(mj):
    model, data, site = mj
    for x, z in [(0.15, 0.044), (0.3, 0.2), (0.45, 0.024), (0.5, 0.15)]:
        res = _s1("--mode", "ik", f"--target={x},{z}")
        assert res["reachable"]
        data.qpos[:3] = res["q"]
        mujoco.mj_forward(model, data)
        assert data.site_xpos[site][0] == pytest.approx(x, abs=1e-9) and data.site_xpos[site][2] == pytest.approx(z, abs=1e-9)
        # the pad axis (link3's local x) must point straight down in the world
        link3 = data.xmat[model.body("link3").id].reshape(3, 3)
        assert link3[:, 0] == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


# ------------------------------------------------------------------------------------------------ the closed loop
EXPECTED_ORDER = ["home", "wait_goal", "approach", "descend", "grasp", "lift", "wait_goal", "transport", "lower", "release",
                  "retreat", "wait_goal", "finished"]

CONFIGS = [
    pytest.param(dict(red_x=0.45, platform_x=0.15, start_q=(-1.0, 1.9, 0.3)), id="default"),
    pytest.param(dict(red_x=0.50, platform_x=0.13, start_q=(-0.8, 1.0, 0.8)), id="far-block-other-start"),
    pytest.param(dict(red_x=0.38, platform_x=0.17, start_q=(-1.1, 1.5, 0.9)), id="block-next-to-the-distractor"),
    pytest.param(dict(red_x=0.42, blue_x=0.30, platform_x=0.14, start_q=(-1.3, 2.2, -0.3)), id="third-start-pose"),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_pick_and_place_completes_in_sim(cfg):
    r = PickPlaceHost(**cfg).run(timeout_sim_s=40)
    s1 = r.system1
    assert r.success, f"{r.system2_status} {r.system2_failure} block={r.red_block} s1={s1.get('final_phase')} {s1.get('failure')}"
    assert r.phases() == EXPECTED_ORDER
    assert abs(r.red_block[0] - cfg["platform_x"]) < 0.05 and r.red_block[1] > 0.035     # resting on the platform
    assert s1["deadline_misses"] == 0 and s1["grasp_retries"] == 0
    assert s1["tick_ns_p99"] < 2_000_000                                                  # far inside the 10 ms tick
    assert s1["ticks"] > 300 and s1["distinct_goals_seen"] >= 6                           # it really was reading System 2


def test_a_slower_system_2_only_slows_the_task_down():
    """System 2 at 0.7 Hz instead of 2 Hz: System 1 still ticks at its own rate and the task still completes, later."""
    fast = PickPlaceHost(s2_period=0.5).run(timeout_sim_s=40)
    slow = PickPlaceHost(s2_period=1.5).run(timeout_sim_s=60)
    assert fast.success and slow.success, (fast.system2_failure, slow.system2_failure)
    assert slow.sim_time > fast.sim_time
    # System 1's tick rate is independent of how often System 2 publishes: ticks per sim-second stay ~100 in both runs
    for r in (fast, slow):
        assert r.system1["ticks"] / r.sim_time == pytest.approx(100, rel=0.1)


def test_system_2_failure_stops_the_arm_without_dropping_to_garbage():
    """An instruction System 2 cannot parse: it publishes FAILED immediately and System 1 holds position at home."""
    r = PickPlaceHost(instruction="juggle the blocks").run(timeout_sim_s=10)
    assert not r.success and r.system2_status == "failed" and "parse" in r.system2_failure
    assert r.system1["final_phase"] in ("failed", "wait_goal", "home")
    assert r.red_block[0] == pytest.approx(0.45, abs=0.01)                                # nothing was disturbed


def test_phase_names_cover_every_code_the_rust_side_can_publish():
    assert len(PHASES) == 13 and PHASES[0] == "home" and PHASES[-1] == "failed" and not math.isnan(len(PHASES))
