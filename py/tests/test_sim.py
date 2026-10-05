"""M2: the MuJoCo arm scene loads and steps on its own, no control
logic yet (that's the Rust controller, tested separately, and the two
aren't wired together until M4).
"""

from policyserve.sim import NUM_JOINTS, ArmSim


def test_arm_loads_and_steps_without_error():
    sim = ArmSim()
    sim.reset()
    for _ in range(200):
        sim.step()
    positions = sim.joint_positions()
    assert len(positions) == NUM_JOINTS
    assert all(abs(p) < 100 for p in positions), f"unstable: {positions}"


def test_a_velocity_command_moves_the_commanded_joint():
    # This arm's damping settles each joint into a new equilibrium
    # *position* within well under a second rather than sustaining
    # indefinite velocity (confirmed empirically -- qvel decays back
    # toward 0 by ~step 150 for this scene's damping/actuator gains,
    # not assumed from the actuator type alone) -- so this checks the
    # commanded joint actually moved from its start, not a specific
    # displacement at a specific time.
    sim = ArmSim()
    sim.reset()
    start = sim.joint_positions()[2]
    for _ in range(100):
        sim.step(velocity_command=[0.0, 0.0, 1.0])
    end = sim.joint_positions()[2]
    assert end > start + 0.05, f"joint3 barely moved: {start} -> {end}"


def test_zero_velocity_command_does_not_blow_up_under_gravity():
    # The real bug this test exists to catch: an earlier version of
    # this scene's velocity actuators + joint damping formed a stiff
    # system the default integrator diverged on within ~20 steps
    # (qpos/qvel into the thousands). This drives it for a full second
    # of sim time and asserts it stays physically sane throughout, not
    # just at the end.
    sim = ArmSim()
    sim.reset()
    for _ in range(500):
        sim.step(velocity_command=[0.0, 0.0, 0.0])
        positions = sim.joint_positions()
        velocities = sim.joint_velocities()
        assert all(abs(p) < 10 for p in positions), f"position blew up: {positions}"
        assert all(abs(v) < 50 for v in velocities), f"velocity blew up: {velocities}"


def test_rejects_the_wrong_number_of_joint_velocities():
    sim = ArmSim()
    sim.reset()
    try:
        sim.step(velocity_command=[1.0, 0.0])
        assert False, "should have raised ValueError"
    except ValueError:
        pass
