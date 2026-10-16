"""System 2's plug point: ``Reasoner.plan(frame, instruction, proprioception) -> Subgoal | Done | Failed``.

``ScriptedReasoner`` is the deterministic implementation used by tests and the default demo: a colour-blob heuristic
(deliberately not a learned detector - the point is determinism, not detection quality) locates the named blocks in the sim's
camera frame, and a few explicit rules turn that into a subgoal. A real multimodal LLM implements the same interface in M7.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np

COLORS = ("red", "green", "blue")


# ----------------------------------------------------------------------------------------------- reasoner interface
@dataclass(frozen=True)
class Subgoal:
    kind: str    # "reach" | "place"
    x: float
    z: float


@dataclass(frozen=True)
class Done:
    pass


@dataclass(frozen=True)
class Failed:
    reason: str


@dataclass(frozen=True)
class Proprioception:
    ee_x: float
    ee_z: float


class Reasoner(Protocol):
    def plan(self, frame: np.ndarray, instruction: str, proprio: Proprioception) -> Subgoal | Done | Failed: ...


# ----------------------------------------------------------------------------------------------- camera + blob finding
@dataclass(frozen=True)
class OrthoCamera:
    """Fixed orthographic side camera: pixel (u, v) <-> world (x, z) is an exact linear map."""

    width: int
    height: int
    center_x: float
    center_z: float
    extent: float  # vertical extent in metres (MuJoCo's fovy for an orthographic camera)

    @classmethod
    def from_model(cls, model: mujoco.MjModel, name: str, width: int, height: int) -> "OrthoCamera":
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if cid < 0:
            raise ValueError(f"camera '{name}' not in model")
        pos = model.cam_pos[cid]
        return cls(width, height, float(pos[0]), float(pos[2]), float(model.cam_fovy[cid]))

    def to_world(self, u: float, v: float) -> tuple[float, float]:
        s = self.extent / self.height
        return self.center_x + (u - self.width / 2) * s, self.center_z - (v - self.height / 2) * s


@dataclass(frozen=True)
class Blob:
    x: float          # world centroid
    z: float
    x_min: float
    x_max: float
    z_min: float      # world bottom edge
    z_max: float      # world top edge
    pixels: int


def _hsv(img: np.ndarray):
    f = img.astype(np.float32) / 255.0
    mx, mn = f.max(axis=-1), f.min(axis=-1)
    d = mx - mn
    sat = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    h = np.zeros_like(mx)
    nz = d > 1e-6
    h = np.where(nz & (mx == r), ((g - b) / np.maximum(d, 1e-6)) % 6, h)
    h = np.where(nz & (mx == g), (b - r) / np.maximum(d, 1e-6) + 2, h)
    h = np.where(nz & (mx == b), (r - g) / np.maximum(d, 1e-6) + 4, h)
    return h * 60.0, sat, mx


def find_blob(frame: np.ndarray, color: str, cam: OrthoCamera, min_pixels: int = 8) -> Blob | None:
    """Largest-evidence blob of the named colour (hue range + saturation + value), or None if fewer than min_pixels match."""
    h, s, v = _hsv(frame)
    if color == "red":
        hue = (h < 20) | (h > 340)
    elif color == "green":
        hue = (h > 90) & (h < 150)
    elif color == "blue":
        hue = (h > 200) & (h < 260)
    else:
        raise ValueError(f"unknown colour '{color}'")
    mask = hue & (s > 0.5) & (v > 0.25)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    cx, cz = cam.to_world(xs.mean() + 0.5, ys.mean() + 0.5)
    x0, z1 = cam.to_world(xs.min(), ys.min())           # left / top edge
    x1, z0 = cam.to_world(xs.max() + 1, ys.max() + 1)   # right / bottom edge
    return Blob(cx, cz, x0, x1, z0, z1, int(len(xs)))


# ----------------------------------------------------------------------------------------------- instruction parsing
_PICK = re.compile(r"\b(?:pick(?: up)?|grab|take)\s+the\s+(\w+)\s+(?:block|cube)\b", re.I)
_PLACE = re.compile(r"\b(?:on|onto|in|to|at)\s+the\s+(\w+)\s+(?:platform|pad|plate)\b", re.I)


def parse_instruction(instruction: str) -> tuple[str, str] | None:
    """('red', 'green') for 'pick up the red block and place it on the green platform'; None if not understood."""
    a, b = _PICK.search(instruction), _PLACE.search(instruction)
    if not (a and b):
        return None
    block, plat = a.group(1).lower(), b.group(1).lower()
    return (block, plat) if block in COLORS and plat in COLORS else None


# ----------------------------------------------------------------------------------------------- scripted reasoner
class ScriptedReasoner:
    """Deterministic rules, in order:
      1. cannot parse the instruction, or a named object is not visible        -> Failed(reason)
      2. block rests on the platform (centre within its x-extent, above its top) -> Done
      3. end effector is already at the block (within grasp_tol)                -> Subgoal(place) above the platform
      4. otherwise                                                              -> Subgoal(reach) at the block
    """

    def __init__(self, camera: OrthoCamera, grasp_tol: float = 0.05, block_half: float = 0.02):
        self.camera, self.grasp_tol, self.block_half = camera, grasp_tol, block_half

    def plan(self, frame: np.ndarray, instruction: str, proprio: Proprioception) -> Subgoal | Done | Failed:
        parsed = parse_instruction(instruction)
        if parsed is None:
            return Failed(f"could not parse instruction: {instruction!r}")
        block_color, plat_color = parsed
        block = find_blob(frame, block_color, self.camera)
        if block is None:
            return Failed(f"{block_color} block not visible")
        plat = find_blob(frame, plat_color, self.camera)
        if plat is None:
            return Failed(f"{plat_color} platform not visible")

        on_platform = plat.x_min <= block.x <= plat.x_max and block.z >= plat.z_max + 0.5 * self.block_half
        if on_platform and np.hypot(proprio.ee_x - block.x, proprio.ee_z - block.z) > self.grasp_tol:
            return Done()
        if np.hypot(proprio.ee_x - block.x, proprio.ee_z - block.z) <= self.grasp_tol:
            return Subgoal("place", plat.x, plat.z_max + self.block_half)
        return Subgoal("reach", block.x, block.z)
