//! System 1's actual control logic (design doc §8.2) -- a closed-form,
//! joint-space proportional controller, plus the hard per-step deadline
//! enforcement wrapped around *any* controller. Deliberately no inverse
//! kinematics here (§4's own stated scope boundary): the target is a
//! joint configuration, not an end-effector pose, so "control" is just
//! "drive each joint's error toward zero," which is enough to prove the
//! architecture this project exists to prove without needing a real
//! IK solver.
//!
//! Not wired into `main`'s CLI yet -- that happens in M4, once System 2
//! (M3) exists to actually drive `target` with something other than a
//! value a test wrote directly. Everything here is exercised by its own
//! tests in the meantime, hence the module-level `allow` below rather
//! than `main` pretending to use it early.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// Must match `py/policyserve/sim.py`'s `NUM_JOINTS` -- the two sides
/// agree on this scene's joint count the same way they agree on shared-
/// memory payload sizes, by a shared constant documented in both places
/// rather than inferred at runtime.
pub const NUM_JOINTS: usize = 3;
pub type JointConfig = [f64; NUM_JOINTS];

/// Anything that turns (target, current) joint configs into a velocity
/// command. The seam `SafeController` wraps with deadline enforcement,
/// and the seam the deadline-miss test below swaps in an artificially
/// slow implementation of to prove the enforcement is real.
pub trait JointController {
    fn compute(&self, target: &JointConfig, current: &JointConfig) -> JointConfig;
}

/// `v_i = clamp(kp * (target_i - current_i), -max_velocity, max_velocity)`
/// per joint, independently -- no coupling between joints, which is
/// exactly what "joint-space" (as opposed to task-space/Cartesian)
/// control means here.
pub struct ProportionalController {
    pub kp: f64,
    pub max_velocity: f64,
}

impl JointController for ProportionalController {
    fn compute(&self, target: &JointConfig, current: &JointConfig) -> JointConfig {
        let mut out = [0.0; NUM_JOINTS];
        for ((o, t), c) in out.iter_mut().zip(target.iter()).zip(current.iter()) {
            let error = t - c;
            *o = (self.kp * error).clamp(-self.max_velocity, self.max_velocity);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepResult {
    pub action: JointConfig,
    pub duration: Duration,
    pub missed_deadline: bool,
}

/// Wraps any `JointController` with design doc §8.2's hard deadline:
/// a step whose compute exceeds `budget` is logged as a miss and its
/// (possibly half-computed, possibly stale-by-the-time-it-finished)
/// action is discarded in favor of `safe_fallback` -- never applying a
/// command computed under a budget violation, even if it technically
/// finished by the time `step()` returns.
pub struct SafeController<C: JointController> {
    inner: C,
    budget: Duration,
    safe_fallback: JointConfig,
}

impl<C: JointController> SafeController<C> {
    pub fn new(inner: C, budget: Duration, safe_fallback: JointConfig) -> Self {
        Self {
            inner,
            budget,
            safe_fallback,
        }
    }

    pub fn step(&self, target: &JointConfig, current: &JointConfig) -> StepResult {
        let start = Instant::now();
        let action = self.inner.compute(target, current);
        let duration = start.elapsed();
        if duration > self.budget {
            StepResult {
                action: self.safe_fallback,
                duration,
                missed_deadline: true,
            }
        } else {
            StepResult {
                action,
                duration,
                missed_deadline: false,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wraps any controller with a real, measured artificial delay --
    /// the only way to prove deadline enforcement actually triggers
    /// under a genuine budget violation rather than assuming the branch
    /// works because it reads correctly.
    struct SlowController<C: JointController> {
        inner: C,
        extra_delay: Duration,
    }

    impl<C: JointController> JointController for SlowController<C> {
        fn compute(&self, target: &JointConfig, current: &JointConfig) -> JointConfig {
            std::thread::sleep(self.extra_delay);
            self.inner.compute(target, current)
        }
    }

    #[test]
    fn controller_converges_to_target_joint_config() {
        let safe = SafeController::new(
            ProportionalController {
                kp: 2.0,
                max_velocity: 5.0,
            },
            Duration::from_millis(8),
            [0.0; NUM_JOINTS],
        );
        let target = [1.0, -0.5, 0.3];
        let mut current = [0.0; NUM_JOINTS];
        let dt = 0.01;

        for _ in 0..500 {
            let result = safe.step(&target, &current);
            assert!(
                !result.missed_deadline,
                "a fast controller should never miss"
            );
            for (c, a) in current.iter_mut().zip(result.action.iter()) {
                *c += a * dt;
            }
        }

        for (i, (c, t)) in current.iter().zip(target.iter()).enumerate() {
            assert!(
                (c - t).abs() < 0.01,
                "joint {i} did not converge: {c} vs target {t}"
            );
        }
    }

    #[test]
    fn a_fast_compute_never_triggers_the_fallback() {
        let safe = SafeController::new(
            ProportionalController {
                kp: 1.0,
                max_velocity: 1.0,
            },
            Duration::from_millis(8),
            [0.0; NUM_JOINTS],
        );
        let result = safe.step(&[1.0, 0.0, 0.0], &[0.0; NUM_JOINTS]);
        assert!(!result.missed_deadline);
        assert_eq!(result.action, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn a_slow_compute_triggers_the_safe_fallback_and_reports_its_real_duration() {
        let budget = Duration::from_millis(5);
        let fallback = [-1.0, -1.0, -1.0]; // deliberately not all-zero, so the
        // test can't pass by coincidence if the fallback path silently
        // no-ops instead of actually substituting it.
        let slow = SlowController {
            inner: ProportionalController {
                kp: 1.0,
                max_velocity: 1.0,
            },
            extra_delay: Duration::from_millis(20),
        };
        let safe = SafeController::new(slow, budget, fallback);

        let result = safe.step(&[1.0, 1.0, 1.0], &[0.0; NUM_JOINTS]);

        assert!(result.missed_deadline);
        assert_eq!(result.action, fallback);
        assert!(
            result.duration >= Duration::from_millis(20),
            "reported duration ({:?}) should reflect the real injected delay",
            result.duration
        );
    }

    #[test]
    fn a_compute_just_under_budget_does_not_miss() {
        let budget = Duration::from_millis(20);
        let slow = SlowController {
            inner: ProportionalController {
                kp: 1.0,
                max_velocity: 1.0,
            },
            extra_delay: Duration::from_millis(5),
        };
        let safe = SafeController::new(slow, budget, [0.0; NUM_JOINTS]);
        let result = safe.step(&[1.0, 0.0, 0.0], &[0.0; NUM_JOINTS]);
        assert!(!result.missed_deadline, "{:?}", result);
    }
}
