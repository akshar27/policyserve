//! Byte layouts of the three records that cross the Rust/Python boundary in the M4 control mode. Each is one fixed-size payload
//! in its own seqlock channel (see shm.rs). Python's definitions live in `py/policyserve/wire.py` (proprioception + action)
//! and `py/policyserve/goal.py` (goal); a cross-language test (`py/tests/test_closed_loop.py`) round-trips real records
//! through the compiled binary, so a drift between the two sides fails loudly instead of silently mis-reading fields.
//!
//! All little-endian. Offsets are written out in each `decode`/`encode` rather than derived, so the layout is greppable.

use crate::controller::{JointConfig, NUM_JOINTS};

pub const GOAL_SIZE: usize = 40;
pub const PROPRIO_SIZE: usize = 80;
pub const ACTION_SIZE: usize = 64;

/// Bit 0 of `Proprio::flags`: the sim host asks System 1 to stop ticking and print its summary.
pub const FLAG_STOP: u32 = 1;

fn f64_at(b: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoalKind {
    None,
    Reach,
    Place,
    Done,
    Failed,
}

/// System 2's goal: `source_cycle u64 | kind u8 | 3 pad | frame_step u32 | x f64 | z f64 | updated_at f64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Goal {
    pub cycle: u64,
    pub kind: GoalKind,
    /// Sim step at which the camera frame behind this goal was captured.
    pub frame_step: u32,
    pub x: f64,
    pub z: f64,
}

impl Goal {
    /// `None` if the kind byte is not one this build knows (a newer/older Python): the caller keeps its previous goal.
    pub fn decode(b: &[u8; GOAL_SIZE]) -> Option<Goal> {
        let kind = match b[8] {
            0 => GoalKind::None,
            1 => GoalKind::Reach,
            2 => GoalKind::Place,
            3 => GoalKind::Done,
            4 => GoalKind::Failed,
            _ => return None,
        };
        let (x, z) = (f64_at(b, 16), f64_at(b, 24));
        if !(x.is_finite() && z.is_finite()) {
            return None;
        }
        Some(Goal {
            cycle: u64_at(b, 0),
            kind,
            frame_step: u32_at(b, 12),
            x,
            z,
        })
    }

    #[cfg(test)] // Python is the only writer of goals; Rust only decodes (the encoder exists for the round-trip tests)
    pub fn encode(&self) -> [u8; GOAL_SIZE] {
        let mut b = [0u8; GOAL_SIZE];
        b[0..8].copy_from_slice(&self.cycle.to_le_bytes());
        b[8] = match self.kind {
            GoalKind::None => 0,
            GoalKind::Reach => 1,
            GoalKind::Place => 2,
            GoalKind::Done => 3,
            GoalKind::Failed => 4,
        };
        b[12..16].copy_from_slice(&self.frame_step.to_le_bytes());
        b[16..24].copy_from_slice(&self.x.to_le_bytes());
        b[24..32].copy_from_slice(&self.z.to_le_bytes());
        b
    }
}

/// What the sim host publishes every control tick:
/// `sim_step u64 | sim_time f64 | q[3] f64 | qd[3] f64 | gap f64 | flags u32 | 4 pad`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Proprio {
    pub sim_step: u64,
    pub sim_time: f64,
    pub q: JointConfig,
    pub qd: JointConfig,
    /// Clear distance between the gripper pads, metres.
    pub gap: f64,
    pub flags: u32,
}

impl Proprio {
    pub fn decode(b: &[u8; PROPRIO_SIZE]) -> Proprio {
        let mut q = [0.0; NUM_JOINTS];
        let mut qd = [0.0; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            q[i] = f64_at(b, 16 + 8 * i);
            qd[i] = f64_at(b, 40 + 8 * i);
        }
        Proprio {
            sim_step: u64_at(b, 0),
            sim_time: f64_at(b, 8),
            q,
            qd,
            gap: f64_at(b, 64),
            flags: u32_at(b, 72),
        }
    }

    #[cfg(test)] // Python is the only writer of proprioception
    pub fn encode(&self) -> [u8; PROPRIO_SIZE] {
        let mut b = [0u8; PROPRIO_SIZE];
        b[0..8].copy_from_slice(&self.sim_step.to_le_bytes());
        b[8..16].copy_from_slice(&self.sim_time.to_le_bytes());
        for i in 0..NUM_JOINTS {
            b[16 + 8 * i..24 + 8 * i].copy_from_slice(&self.q[i].to_le_bytes());
            b[40 + 8 * i..48 + 8 * i].copy_from_slice(&self.qd[i].to_le_bytes());
        }
        b[64..72].copy_from_slice(&self.gap.to_le_bytes());
        b[72..76].copy_from_slice(&self.flags.to_le_bytes());
        b
    }
}

/// What System 1 publishes every tick:
/// `tick u64 | vel[3] f64 | grip_open f64 | phase u32 | deadline_misses u32 | compute_ns u64 | goal_cycle u64`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Action {
    pub tick: u64,
    pub vel: JointConfig,
    /// 1.0 = open the gripper, 0.0 = close it.
    pub grip_open: f64,
    pub phase: u32,
    pub deadline_misses: u32,
    pub compute_ns: u64,
    pub goal_cycle: u64,
}

impl Action {
    pub fn encode(&self) -> [u8; ACTION_SIZE] {
        let mut b = [0u8; ACTION_SIZE];
        b[0..8].copy_from_slice(&self.tick.to_le_bytes());
        for i in 0..NUM_JOINTS {
            b[8 + 8 * i..16 + 8 * i].copy_from_slice(&self.vel[i].to_le_bytes());
        }
        b[32..40].copy_from_slice(&self.grip_open.to_le_bytes());
        b[40..44].copy_from_slice(&self.phase.to_le_bytes());
        b[44..48].copy_from_slice(&self.deadline_misses.to_le_bytes());
        b[48..56].copy_from_slice(&self.compute_ns.to_le_bytes());
        b[56..64].copy_from_slice(&self.goal_cycle.to_le_bytes());
        b
    }

    #[cfg(test)] // Python is the only reader of actions
    pub fn decode(b: &[u8; ACTION_SIZE]) -> Action {
        let mut vel = [0.0; NUM_JOINTS];
        for (i, v) in vel.iter_mut().enumerate() {
            *v = f64_at(b, 8 + 8 * i);
        }
        Action {
            tick: u64_at(b, 0),
            vel,
            grip_open: f64_at(b, 32),
            phase: u32_at(b, 40),
            deadline_misses: u32_at(b, 44),
            compute_ns: u64_at(b, 48),
            goal_cycle: u64_at(b, 56),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_roundtrip_and_known_layout() {
        let g = Goal {
            cycle: 7,
            kind: GoalKind::Place,
            frame_step: 4321,
            x: 0.25,
            z: 0.04,
        };
        let bytes = g.encode();
        assert_eq!(Goal::decode(&bytes), Some(g));
        // spot-check against the documented offsets, not just encode->decode symmetry
        assert_eq!(bytes[8], 2);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 4321);
    }

    #[test]
    fn goal_with_unknown_kind_or_nan_is_rejected_not_misread() {
        let mut bytes = Goal {
            cycle: 1,
            kind: GoalKind::Reach,
            frame_step: 0,
            x: 0.3,
            z: 0.02,
        }
        .encode();
        bytes[8] = 99;
        assert_eq!(Goal::decode(&bytes), None);
        bytes[8] = 1;
        bytes[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(Goal::decode(&bytes), None);
    }

    #[test]
    fn proprio_and_action_roundtrip() {
        let p = Proprio {
            sim_step: 99,
            sim_time: 1.5,
            q: [0.1, -0.2, 0.3],
            qd: [1.0, 2.0, 3.0],
            gap: 0.04,
            flags: FLAG_STOP,
        };
        assert_eq!(Proprio::decode(&p.encode()), p);
        let a = Action {
            tick: 5,
            vel: [0.5, -0.5, 0.25],
            grip_open: 1.0,
            phase: 4,
            deadline_misses: 2,
            compute_ns: 1234,
            goal_cycle: 8,
        };
        assert_eq!(Action::decode(&a.encode()), a);
    }
}
