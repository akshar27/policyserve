import threading
import time
import uuid

import mujoco
import numpy as np
import pytest

from policyserve.goal import GOAL_PAYLOAD_SIZE, GoalKind, GoalState
from policyserve.reasoner import (Done, Failed, OrthoCamera, Proprioception, ScriptedReasoner, Subgoal, find_blob,
                                  parse_instruction)
from policyserve.shm import SeqLockChannel
from policyserve.sim import MODEL_PATH as _ARM_ONLY  # noqa: F401  (M2 model stays untouched)
from policyserve.system2 import System2Loop

SCENE = _ARM_ONLY.parent / "pick_place_arm.xml"
W, H = 320, 240
INSTR = "pick up the red block and place it on the green platform"


class Scene:
    """The M3 MuJoCo scene with helpers to place the blocks and the arm and render the side camera."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, H, W)
        self.cam = OrthoCamera.from_model(self.model, "front", W, H)

    def place(self, name: str, x: float, z: float) -> None:
        body = self.model.body(name)
        adr = self.model.jnt_qposadr[body.jntadr[0]]
        self.data.qpos[adr:adr + 3] = [x, 0.0, z]

    def frame(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="front")
        return self.renderer.render().copy()

    def ee(self) -> tuple[float, float]:
        p = self.data.site_xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")]
        return float(p[0]), float(p[2])


@pytest.fixture(scope="module")
def scene():
    return Scene()


def _fresh(scene, red=(0.55, 0.02), blue=(0.3, 0.02)):
    mujoco.mj_resetData(scene.model, scene.data)
    scene.place("red_block", *red)
    scene.place("blue_block", *blue)
    return scene


# ------------------------------------------------------------------------------------------------ perception
def test_blob_centroids_match_known_world_positions(scene):
    for red in [(0.55, 0.02), (0.7, 0.02), (0.1, 0.02), (0.45, 0.30)]:
        f = _fresh(scene, red=red, blue=(0.9, 0.02)).frame()
        b = find_blob(f, "red", scene.cam)
        assert b is not None
        assert abs(b.x - red[0]) < 0.01 and abs(b.z - red[1]) < 0.01, (red, b)


def test_colours_are_not_confused_and_arm_is_not_a_blob(scene):
    f = _fresh(scene).frame()
    red, blue, green = (find_blob(f, c, scene.cam) for c in ("red", "blue", "green"))
    assert abs(red.x - 0.55) < 0.01 and abs(blue.x - 0.3) < 0.01
    assert abs(green.x - 0.2) < 0.01 and abs(green.z - 0.01) < 0.01          # platform centre
    assert abs(green.z_max - 0.02) < 0.006                                   # platform top surface


def test_absent_colour_returns_none(scene):
    f = _fresh(scene, red=(0.55, 0.02), blue=(0.9, 0.02)).frame()
    f[:] = 128                                                               # gray frame: nothing coloured
    assert find_blob(f, "red", scene.cam) is None


def test_instruction_parsing():
    assert parse_instruction(INSTR) == ("red", "green")
    assert parse_instruction("Grab the blue cube and put it onto the green plate") == ("blue", "green")
    assert parse_instruction("pick up the purple block and place it on the green platform") is None
    assert parse_instruction("dance") is None


# ------------------------------------------------------------------------------------------------ reasoner rules
def test_reach_when_block_is_elsewhere(scene):
    f = _fresh(scene, red=(0.55, 0.02)).frame()
    out = ScriptedReasoner(scene.cam).plan(f, INSTR, Proprioception(0.0, 0.5))
    assert isinstance(out, Subgoal) and out.kind == "reach"
    assert abs(out.x - 0.55) < 0.01 and abs(out.z - 0.02) < 0.01


def test_place_once_the_end_effector_is_at_the_block(scene):
    f = _fresh(scene, red=(0.55, 0.02)).frame()
    out = ScriptedReasoner(scene.cam).plan(f, INSTR, Proprioception(0.56, 0.03))
    assert isinstance(out, Subgoal) and out.kind == "place"
    assert abs(out.x - 0.2) < 0.01 and abs(out.z - (0.02 + 0.02)) < 0.01      # platform top + block half-height


def test_done_when_block_rests_on_platform(scene):
    f = _fresh(scene, red=(0.2, 0.04)).frame()                               # on top of the platform (top at z=0.02)
    out = ScriptedReasoner(scene.cam).plan(f, INSTR, Proprioception(0.6, 0.4))
    assert isinstance(out, Done)


def test_block_beside_platform_is_not_done(scene):
    f = _fresh(scene, red=(0.45, 0.02)).frame()
    assert isinstance(ScriptedReasoner(scene.cam).plan(f, INSTR, Proprioception(0.6, 0.4)), Subgoal)


def test_failed_when_object_missing_or_instruction_unparseable(scene):
    f = _fresh(scene).frame()
    f[:] = 128
    r = ScriptedReasoner(scene.cam)
    assert isinstance(r.plan(f, INSTR, Proprioception(0, 0)), Failed)
    assert isinstance(r.plan(_fresh(scene).frame(), "wave hello", Proprioception(0, 0)), Failed)


def test_same_scene_same_plan(scene):
    f = _fresh(scene).frame()
    r = ScriptedReasoner(scene.cam)
    assert r.plan(f, INSTR, Proprioception(0, 0.5)) == r.plan(f.copy(), INSTR, Proprioception(0, 0.5))


# ------------------------------------------------------------------------------------------------ goal format + loop
def test_goal_roundtrip_and_size():
    g = GoalState(7, GoalKind.PLACE, 0.25, 0.04, 123.5)
    assert len(g.encode()) == GOAL_PAYLOAD_SIZE == 40
    assert GoalState.decode(g.encode()) == g
    assert g.age(124.0) == pytest.approx(0.5)


@pytest.fixture
def channel():
    ch = SeqLockChannel(f"ps_goal_{uuid.uuid4().hex[:8]}", GOAL_PAYLOAD_SIZE, create=True)
    ch.write(GoalState(0, GoalKind.NONE).encode())
    yield ch
    ch.close(unlink=True)


def _loop(channel, reasoner, period=0.05, **kw):
    return System2Loop(reasoner, lambda: np.zeros((4, 4, 3), np.uint8), lambda: Proprioception(0, 0), INSTR, channel,
                       period_s=period, **kw)


class Scripted:
    def __init__(self, outputs):
        self.outputs, self.i = outputs, 0

    def plan(self, frame, instruction, proprio):
        out = self.outputs[min(self.i, len(self.outputs) - 1)]
        self.i += 1
        if isinstance(out, Exception):
            raise out
        return out


def test_loop_publishes_subgoal_to_the_shared_channel(channel):
    lp = _loop(channel, Scripted([Subgoal("reach", 0.5, 0.02)]))
    lp.run(max_cycles=1)
    g = GoalState.decode(channel.read())
    assert g.kind == GoalKind.REACH and (g.x, g.z) == (0.5, 0.02) and g.source_cycle == 1


def test_cadence_is_respected():
    ch_name = f"ps_goal_{uuid.uuid4().hex[:8]}"
    ch = SeqLockChannel(ch_name, GOAL_PAYLOAD_SIZE, create=True)
    try:
        lp = _loop(ch, Scripted([Subgoal("reach", 0.5, 0.02)]), period=0.05)
        lp.start()
        time.sleep(1.0)
        lp.stop()
        starts = [r.started_at for r in lp.records]
        gaps = np.diff(starts)
        assert 17 <= len(starts) <= 22                       # ~20 cycles in 1 s at 20 Hz
        assert abs(np.median(gaps) - 0.05) < 0.01
        assert lp.late_cycles == 0
    finally:
        ch.close(unlink=True)


def test_overrunning_cycle_delays_the_next_one_without_catch_up_burst(channel):
    class Slow(Scripted):
        def plan(self, *a):
            time.sleep(0.12)
            return super().plan(*a)

    lp = _loop(channel, Slow([Subgoal("reach", 0.5, 0.02)]), period=0.05)
    lp.run(max_cycles=5)
    gaps = np.diff([r.started_at for r in lp.records])
    assert all(g >= 0.115 for g in gaps)                      # cycles never overlap and are never squeezed together
    assert lp.late_cycles >= 3 and any(r.late for r in lp.records)


def test_exception_leaves_goal_unchanged_and_is_counted(channel):
    lp = _loop(channel, Scripted([Subgoal("reach", 0.5, 0.02), RuntimeError("boom")]))
    lp.run_cycle()
    before = channel.read()
    rec = lp.run_cycle()
    assert rec.outcome == "error" and channel.read() == before
    assert lp.total_failures == 1 and "boom" in lp.last_failure and lp.status == "idle"


def test_malformed_output_is_a_failure_and_n_in_a_row_fails_the_task(channel):
    lp = _loop(channel, Scripted([Subgoal("reach", 0.5, 0.02), "garbage", Subgoal("fly", 1, 1), Subgoal("reach", float("nan"), 0)]),
               max_consecutive_failures=3)
    lp.run_cycle()
    good = channel.read()
    for _ in range(2):
        assert lp.run_cycle().outcome == "malformed"
    assert channel.read() == good and lp.status != "failed"
    lp.run_cycle()                                           # third consecutive failure
    assert lp.status == "failed" and GoalState.decode(channel.read()).kind == GoalKind.FAILED


def test_a_good_cycle_resets_the_consecutive_failure_count(channel):
    lp = _loop(channel, Scripted([RuntimeError("x"), RuntimeError("x"), Subgoal("reach", 0.5, 0.02), RuntimeError("x")]),
               max_consecutive_failures=3)
    for _ in range(4):
        lp.run_cycle()
    assert lp.consecutive_failures == 1 and lp.status != "failed"


def test_done_and_failed_results_publish_sentinels_and_stop_the_loop(channel):
    lp = _loop(channel, Scripted([Subgoal("reach", 0.5, 0.02), Done()]))
    lp.run()
    assert lp.status == "done" and GoalState.decode(channel.read()).kind == GoalKind.DONE
    lp2 = _loop(channel, Scripted([Failed("no block")]))
    lp2.run()
    assert lp2.status == "failed" and lp2.last_failure == "no block"


def test_reader_in_another_thread_sees_goals_while_loop_runs(channel):
    seen, stop = set(), threading.Event()

    def reader():
        while not stop.is_set():
            data = channel.read()
            if data:
                seen.add(GoalState.decode(data).source_cycle)

    t = threading.Thread(target=reader)
    t.start()
    lp = _loop(channel, Scripted([Subgoal("reach", 0.5, 0.02)]), period=0.02)
    lp.run(max_cycles=15)
    stop.set()
    t.join()
    # the reader polls concurrently, so it sees many distinct goals while the loop runs (it may stop before polling the very
    # last write - that final state is asserted directly instead of racing the reader's last poll)
    assert len(seen) >= 5
    assert GoalState.decode(channel.read()).source_cycle == 15
