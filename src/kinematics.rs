//! Closed-form forward and inverse kinematics for the M4 arm: three hinges about +y in the x-z plane, with a gripper that
//! always points straight down. Because the gripper orientation is fixed, a Cartesian target (x, z) for the grip point fixes the
//! wrist position, and the whole problem collapses to the textbook two-link solution -- no iteration, no Jacobian, bounded time.
//! That is what lets System 1 meet a hard per-tick deadline while still taking *Cartesian* goals from System 2.
//!
//! Must match `py/policyserve/models/pick_place_gripper.xml`: base at height `BASE_Z`, link lengths `L1`, `L2`, and `L3` from joint3
//! to the grip site. Direction convention: for accumulated angle `a`, a link points along `(cos a, -sin a)` in (x, z).
//! The gripper points down when `q1 + q2 + q3 = pi/2`. The Python test `test_closed_loop.py::test_rust_fk_matches_mujoco`
//! compares this FK against MuJoCo's own site position, so the two cannot drift apart unnoticed.

use crate::controller::JointConfig;

pub const BASE_Z: f64 = 0.30;
pub const L1: f64 = 0.30;
pub const L2: f64 = 0.25;
pub const L3: f64 = 0.125;

fn dir(a: f64) -> (f64, f64) {
    (a.cos(), -a.sin())
}

/// Grip-point position (x, z) for joint angles `q`.
pub fn fk(q: &JointConfig) -> (f64, f64) {
    let a1 = q[0];
    let a2 = a1 + q[1];
    let a3 = a2 + q[2];
    let (d1, d2, d3) = (dir(a1), dir(a2), dir(a3));
    (
        L1 * d1.0 + L2 * d2.0 + L3 * d3.0,
        BASE_Z + L1 * d1.1 + L2 * d2.1 + L3 * d3.1,
    )
}

/// Joint angles that put the grip point at (x, z) with the gripper pointing down, elbow up. If the wrist target is outside the
/// annulus the two links can reach, it is pulled to the nearest reachable radius and `reachable` is false: the arm goes as close
/// as it can rather than the solver returning NaN into the control loop.
pub fn ik(x: f64, z: f64) -> (JointConfig, bool) {
    let (wx, wz) = (x, z + L3 - BASE_Z);
    let reach_min = (L1 - L2).abs() + 1e-6;
    let reach_max = L1 + L2 - 1e-6;
    let dist = wx.hypot(wz);
    let reachable = (reach_min..=reach_max).contains(&dist);
    let d = dist.clamp(reach_min, reach_max);
    let c = ((d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)).clamp(-1.0, 1.0);
    let q2 = -c.acos(); // negative = elbow above the shoulder-wrist line
    let al1 = wz.atan2(wx) - (L2 * q2.sin()).atan2(L1 + L2 * q2.cos());
    let q3 = -std::f64::consts::FRAC_PI_2 - al1 - q2;
    ([-al1, -q2, -q3], reachable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fk_of_ik_returns_the_target_across_the_workspace() {
        let mut n = 0;
        for xi in 0..=20 {
            for zi in 0..=12 {
                let (x, z) = (0.12 + 0.02 * xi as f64, 0.02 + 0.02 * zi as f64);
                let (q, reachable) = ik(x, z);
                if !reachable {
                    continue;
                }
                let (fx, fz) = fk(&q);
                assert!(
                    (fx - x).abs() < 1e-9 && (fz - z).abs() < 1e-9,
                    "({x}, {z}) -> {q:?} -> ({fx}, {fz})"
                );
                // and the gripper really does point down
                let sum: f64 = q.iter().sum();
                assert!((sum - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
                n += 1;
            }
        }
        assert!(
            n > 100,
            "the sweep should cover a real workspace, got {n} points"
        );
    }

    #[test]
    fn unreachable_targets_degrade_to_the_boundary_instead_of_nan() {
        for (x, z) in [(2.0, 0.1), (-1.0, 0.5), (0.0, 3.0)] {
            let (q, reachable) = ik(x, z);
            assert!(!reachable);
            assert!(q.iter().all(|a| a.is_finite()), "{q:?}");
        }
    }

    #[test]
    fn the_task_workspace_is_reachable() {
        // block pick-up / platform positions the demo scenes use, at hover and at grasp height
        for x in [0.12, 0.15, 0.3, 0.45, 0.5] {
            for z in [0.024, 0.044, 0.15, 0.17] {
                assert!(ik(x, z).1, "({x}, {z}) should be reachable");
            }
        }
    }
}
