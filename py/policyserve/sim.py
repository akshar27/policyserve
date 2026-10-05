"""M2: the MuJoCo side of the sim, in isolation -- loads a 3-joint
planar arm and steps physics. No control logic here and no connection
to Rust yet: that wiring (reading proprioception out, writing a
velocity command in, both through `shm.SeqLockChannel`) is M4's job,
once System 1's controller (this milestone) and System 2's reasoner
(M3) both exist to wire together.
"""

from __future__ import annotations

from pathlib import Path

import mujoco

MODEL_PATH = Path(__file__).parent / "models" / "simple_arm.xml"
NUM_JOINTS = 3  # must match src/controller.rs's NUM_JOINTS


class ArmSim:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

    def step(self, velocity_command: list[float] | None = None) -> None:
        """Advances the sim by one physics timestep. `velocity_command`
        (one entry per joint) is applied to this scene's velocity
        actuators before stepping; omit it to step with whatever the
        last commanded velocity still was (MuJoCo holds `ctrl` between
        calls)."""
        if velocity_command is not None:
            if len(velocity_command) != NUM_JOINTS:
                raise ValueError(f"expected {NUM_JOINTS} joint velocities, got {len(velocity_command)}")
            self.data.ctrl[:] = velocity_command
        mujoco.mj_step(self.model, self.data)

    def joint_positions(self) -> list[float]:
        return list(self.data.qpos)

    def joint_velocities(self) -> list[float]:
        return list(self.data.qvel)
