//! System 1's task state machine: approach -> descend -> grasp -> lift (then, on a PLACE goal) transport -> lower -> release ->
//! retreat -> wait for the next goal. It runs every control tick, does only O(1) work, and never waits on System 2.
//!
//! Division of labour (the point of the dual-rate design): System 2 says WHERE ("reach (x, z)", "place at (x, z)", "done"),
//! slowly and possibly late; this module owns HOW and WHEN -- the contact-sensitive sequencing that must react within a tick
//! (close the gripper, check the grasp, notice a drop).
//!
//! Rules that keep a slow or stale System 2 from doing damage:
//!   * A sequence is *committed* once the arm starts descending: later goals can still refine the target while the arm is
//!     free-moving (approach / transport), but cannot interrupt a grasp or release half way.
//!   * After finishing a sequence the arm waits for a goal whose camera frame was captured AFTER the sequence ended
//!     (`Goal::frame_step >= fresh_after_step`). A goal computed from a pre-release frame can never trigger a second pick.
//!   * Every phase has a timeout; a phase that cannot finish ends in `Failed` (hold position), never an endless loop.
//!   * The end-effector setpoint is speed- AND acceleration-limited (trapezoidal velocity profile), so the arm follows straight
//!     Cartesian lines even though the joint controller underneath is plain per-joint proportional control, and a held block is
//!     never jerked: with a speed limit alone the block pitched ~18 degrees in the pads at the start of a transport and fell.

use crate::controller::{JointConfig, NUM_JOINTS};
use crate::kinematics::{fk, ik};
use crate::wire::{Goal, GoalKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Home,
    WaitGoal,
    Approach,
    Descend,
    Grasp,
    Lift,
    Transport,
    Lower,
    Release,
    Retreat,
    Recover,
    Finished,
    Failed,
}

impl Phase {
    pub fn code(self) -> u32 {
        self as u32
    }
    pub fn name(self) -> &'static str {
        match self {
            Phase::Home => "home",
            Phase::WaitGoal => "wait_goal",
            Phase::Approach => "approach",
            Phase::Descend => "descend",
            Phase::Grasp => "grasp",
            Phase::Lift => "lift",
            Phase::Transport => "transport",
            Phase::Lower => "lower",
            Phase::Release => "release",
            Phase::Retreat => "retreat",
            Phase::Recover => "recover",
            Phase::Finished => "finished",
            Phase::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
pub struct TaskConfig {
    /// End-effector setpoint speed, m/s.
    pub speed: f64,
    /// End-effector setpoint acceleration (and deceleration) limit, m/s^2.
    pub accel: f64,
    /// Height above the grasp / release point of the approach and retreat waypoints, m.
    pub hover_clearance: f64,
    /// Position tolerance for "arrived at a waypoint", m.
    pub tol: f64,
    /// The grip point is held this far above the block's centre, so the pads straddle the block and clear the floor, m.
    pub grasp_dz: f64,
    /// Time the gripper is given to close / open before its state is judged, s.
    pub settle: f64,
    /// Pad gap range that means "closed on a block-sized object", m.
    pub hold_band: (f64, f64),
    /// Pad gap above which the gripper counts as open, m.
    pub open_gap: f64,
    pub max_grasp_retries: u32,
    pub phase_timeout: f64,
    /// How long the grip may be lost during lift/transport/lower before it counts as a drop, s.
    pub drop_grace: f64,
    /// Joint tolerance for "homed", rad.
    pub home_tol: f64,
    pub ready: (f64, f64),
}

impl Default for TaskConfig {
    fn default() -> Self {
        TaskConfig {
            speed: 0.25,
            accel: 0.6,
            hover_clearance: 0.12,
            tol: 0.008,
            grasp_dz: 0.004,
            settle: 0.45,
            hold_band: (0.030, 0.046),
            open_gap: 0.075,
            max_grasp_retries: 2,
            phase_timeout: 12.0,
            drop_grace: 0.15,
            home_tol: 0.03,
            ready: (0.30, 0.22),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub sim_step: u64,
    pub sim_time: f64,
    pub q: JointConfig,
    pub gap: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    /// Where the controller should drive the grip point right now (already rate-limited).
    pub ee_target: (f64, f64),
    pub grip_open: bool,
}

pub struct Task {
    cfg: TaskConfig,
    phase: Phase,
    ee_cmd: (f64, f64),
    ee_vel: (f64, f64),
    target: (f64, f64),
    hover_z: f64,
    phase_since: f64,
    last_time: Option<f64>,
    last_gap: f64,
    gap_rate: f64,
    retries: u32,
    fresh_after_step: u64,
    grip_open: bool,
    lost_since: Option<f64>,
    failure: Option<String>,
    /// (sim_time, phase entered): the run's history, for the summary and for tests.
    pub transitions: Vec<(f64, Phase)>,
}

fn dist(a: (f64, f64), b: (f64, f64)) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

impl Task {
    pub fn new(cfg: TaskConfig) -> Self {
        let ready = cfg.ready;
        Task {
            cfg,
            phase: Phase::Home,
            ee_cmd: ready,
            ee_vel: (0.0, 0.0),
            target: ready,
            hover_z: 0.0,
            phase_since: 0.0,
            last_time: None,
            last_gap: 0.0,
            gap_rate: 0.0,
            retries: 0,
            fresh_after_step: 0,
            grip_open: true,
            lost_since: None,
            failure: None,
            transitions: vec![(0.0, Phase::Home)],
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
    pub fn grasp_retries(&self) -> u32 {
        self.retries
    }

    fn holding(&self, gap: f64) -> bool {
        (self.cfg.hold_band.0..=self.cfg.hold_band.1).contains(&gap)
    }

    fn enter(&mut self, phase: Phase, now: f64) {
        self.ee_vel = (0.0, 0.0); // every phase starts from rest
        self.phase = phase;
        self.phase_since = now;
        self.transitions.push((now, phase));
    }

    fn fail(&mut self, why: String, now: f64) {
        self.failure = Some(why);
        self.enter(Phase::Failed, now);
    }

    /// Move the setpoint toward `wp` along a trapezoidal velocity profile (accelerate at `accel`, cruise at `speed`, brake so as to
    /// stop exactly on the waypoint); true once both the setpoint and the measured grip point have arrived.
    fn advance(&mut self, wp: (f64, f64), dt: f64, ee: (f64, f64)) -> bool {
        let (dx, dz) = (wp.0 - self.ee_cmd.0, wp.1 - self.ee_cmd.1);
        let d = dx.hypot(dz);
        if d < 1e-4 {
            self.ee_cmd = wp;
            self.ee_vel = (0.0, 0.0);
        } else {
            let v_des = self.cfg.speed.min((2.0 * self.cfg.accel * d).sqrt());
            let want = (dx / d * v_des, dz / d * v_des);
            let (mut ax, mut az) = (want.0 - self.ee_vel.0, want.1 - self.ee_vel.1);
            let max_dv = self.cfg.accel * dt;
            let n = ax.hypot(az);
            if n > max_dv {
                (ax, az) = (ax / n * max_dv, az / n * max_dv);
            }
            self.ee_vel = (self.ee_vel.0 + ax, self.ee_vel.1 + az);
            let (sx, sz) = (self.ee_vel.0 * dt, self.ee_vel.1 * dt);
            if sx.hypot(sz) >= d {
                self.ee_cmd = wp;
                self.ee_vel = (0.0, 0.0);
            } else {
                self.ee_cmd = (self.ee_cmd.0 + sx, self.ee_cmd.1 + sz);
            }
        }
        dist(self.ee_cmd, wp) < 1e-9 && dist(ee, wp) < self.cfg.tol
    }

    /// A sequence just ended: from now on only goals computed from a frame captured after this moment are believed.
    fn sequence_done(&mut self, obs: &Observation) {
        self.fresh_after_step = obs.sim_step + 1;
        self.lost_since = None;
        self.enter(Phase::WaitGoal, obs.sim_time);
    }

    pub fn step(&mut self, obs: &Observation, goal: Option<&Goal>) -> Command {
        let now = obs.sim_time;
        let ee = fk(&obs.q);
        let dt = match self.last_time {
            Some(t) => (now - t).clamp(0.0, 0.05),
            None => 0.0,
        };
        self.gap_rate = if dt > 0.0 {
            (obs.gap - self.last_gap) / dt
        } else {
            self.gap_rate
        };
        self.last_time = Some(now);
        self.last_gap = obs.gap;

        let fresh = goal.filter(|g| {
            g.kind != GoalKind::None && u64::from(g.frame_step) >= self.fresh_after_step
        });

        // System 2 giving up is honoured immediately, whatever System 1 was doing.
        if let Some(g) = fresh
            && g.kind == GoalKind::Failed
            && !matches!(self.phase, Phase::Failed | Phase::Home)
        {
            self.fail("System 2 reported failure".into(), now);
        }
        if !matches!(
            self.phase,
            Phase::Failed | Phase::Finished | Phase::WaitGoal | Phase::Home
        ) && now - self.phase_since > self.cfg.phase_timeout
        {
            self.fail(format!("timeout in phase {}", self.phase.name()), now);
        }

        let hover = |t: &Task| (t.target.0, t.hover_z);
        match self.phase {
            Phase::Home => {
                self.grip_open = true;
                self.ee_cmd = self.cfg.ready;
                let (q_ready, _) = ik(self.cfg.ready.0, self.cfg.ready.1);
                let err = (0..NUM_JOINTS)
                    .map(|i| (q_ready[i] - obs.q[i]).abs())
                    .fold(0.0, f64::max);
                if err < self.cfg.home_tol {
                    self.ee_cmd = ee;
                    self.sequence_done(obs);
                } else if now - self.phase_since > self.cfg.phase_timeout {
                    self.fail("could not reach the home pose".into(), now);
                }
            }
            Phase::WaitGoal => {
                if let Some(g) = fresh {
                    match g.kind {
                        GoalKind::Reach if !self.holding(obs.gap) => {
                            self.target = (g.x, g.z + self.cfg.grasp_dz);
                            self.hover_z = self.target.1 + self.cfg.hover_clearance;
                            self.retries = 0;
                            self.grip_open = true;
                            self.enter(Phase::Approach, now);
                        }
                        GoalKind::Place if self.holding(obs.gap) => {
                            self.target = (g.x, g.z + self.cfg.grasp_dz);
                            self.hover_z = self.target.1 + self.cfg.hover_clearance;
                            self.enter(Phase::Transport, now);
                        }
                        GoalKind::Done => self.enter(Phase::Finished, now),
                        _ => {} // a goal that does not fit the gripper's state (e.g. PLACE with nothing held) is ignored
                    }
                }
            }
            Phase::Approach => {
                if let Some(g) = fresh.filter(|g| g.kind == GoalKind::Reach) {
                    self.target = (g.x, g.z + self.cfg.grasp_dz); // still free-moving: allowed to refine
                }
                self.grip_open = true;
                if self.advance(hover(self), dt, ee) {
                    self.enter(Phase::Descend, now);
                }
            }
            Phase::Descend => {
                self.grip_open = true;
                if self.advance(self.target, dt, ee) {
                    self.grip_open = false;
                    self.enter(Phase::Grasp, now);
                }
            }
            Phase::Grasp => {
                self.grip_open = false;
                self.advance(self.target, dt, ee);
                if now - self.phase_since >= self.cfg.settle && self.gap_rate.abs() < 0.02 {
                    if self.holding(obs.gap) {
                        self.enter(Phase::Lift, now);
                    } else {
                        // closed on nothing (or on something the wrong size): open, back up, look again
                        self.retries += 1;
                        if self.retries > self.cfg.max_grasp_retries {
                            self.fail(
                                format!(
                                    "grasp failed {} times (gap {:.3} m)",
                                    self.retries, obs.gap
                                ),
                                now,
                            );
                        } else {
                            self.grip_open = true;
                            self.enter(Phase::Approach, now);
                        }
                    }
                }
            }
            Phase::Lift => {
                self.grip_open = false;
                self.check_drop(obs);
                if self.phase == Phase::Lift && self.advance(hover(self), dt, ee) {
                    self.sequence_done(obs);
                }
            }
            Phase::Transport => {
                if let Some(g) = fresh.filter(|g| g.kind == GoalKind::Place) {
                    self.target = (g.x, g.z + self.cfg.grasp_dz);
                }
                self.grip_open = false;
                self.check_drop(obs);
                if self.phase == Phase::Transport && self.advance(hover(self), dt, ee) {
                    self.enter(Phase::Lower, now);
                }
            }
            Phase::Lower => {
                self.grip_open = false;
                self.check_drop(obs);
                if self.phase == Phase::Lower && self.advance(self.target, dt, ee) {
                    self.grip_open = true;
                    self.enter(Phase::Release, now);
                }
            }
            Phase::Release => {
                self.grip_open = true;
                self.advance(self.target, dt, ee);
                if now - self.phase_since >= self.cfg.settle && obs.gap >= self.cfg.open_gap {
                    self.enter(Phase::Retreat, now);
                }
            }
            Phase::Retreat => {
                self.grip_open = true;
                if self.advance(hover(self), dt, ee) {
                    self.sequence_done(obs);
                }
            }
            Phase::Recover => {
                self.grip_open = true;
                if self.advance(hover(self), dt, ee) {
                    self.sequence_done(obs);
                }
            }
            Phase::Finished | Phase::Failed => {} // hold: the setpoint stops where it is
        }
        Command {
            ee_target: self.ee_cmd,
            grip_open: self.grip_open,
        }
    }

    /// During lift/transport/lower the grip must stay inside the hold band; losing it for longer than `drop_grace` means the
    /// block fell. Open the gripper and back up to hover, then wait for a fresh goal (System 2 will see the block on the table
    /// again and ask for a new reach).
    fn check_drop(&mut self, obs: &Observation) {
        if self.holding(obs.gap) {
            self.lost_since = None;
            return;
        }
        let since = *self.lost_since.get_or_insert(obs.sim_time);
        if obs.sim_time - since > self.cfg.drop_grace {
            self.target.1 = self.target.1.max(self.ee_cmd.1);
            self.hover_z = self.hover_z.max(self.ee_cmd.1);
            self.grip_open = true;
            self.lost_since = None;
            self.enter(Phase::Recover, obs.sim_time);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake plant: the joints track the IK target perfectly and instantly, and the gripper is a first-order mass that closes to
    /// `block` if a block is between the pads (`block_at` within reach of the grip point) and to ~0 otherwise.
    struct FakePlant {
        q: JointConfig,
        gap: f64,
        time: f64,
        step: u64,
        block: Option<(f64, f64)>,
        held: bool,
        /// If set, the held block is dropped once it has been held this many seconds in total.
        drop_after_held: Option<f64>,
        held_time: f64,
    }

    impl FakePlant {
        fn new(block: Option<(f64, f64)>) -> Self {
            FakePlant {
                q: [0.2, 0.3, -0.1],
                gap: 0.082,
                time: 0.0,
                step: 0,
                block,
                held: false,
                drop_after_held: None,
                held_time: 0.0,
            }
        }
        fn obs(&self) -> Observation {
            Observation {
                sim_step: self.step,
                sim_time: self.time,
                q: self.q,
                gap: self.gap,
            }
        }
        fn advance(&mut self, cmd: &Command, dt: f64) {
            // joints: slew toward the IK solution at a finite rate (not instant, so waypoint arrival is genuinely tested)
            let (qt, _) = ik(cmd.ee_target.0, cmd.ee_target.1);
            for (q, t) in self.q.iter_mut().zip(qt.iter()) {
                *q += (t - *q).clamp(-3.0 * dt, 3.0 * dt);
            }
            let ee = fk(&self.q);
            if cmd.grip_open {
                self.gap = (self.gap + 0.3 * dt).min(0.082);
                if self.held {
                    self.held = false;
                }
            } else {
                let floor = match self.block {
                    Some(b) if dist(b, ee) < 0.02 || self.held => 0.035,
                    _ => 0.002,
                };
                self.gap = (self.gap - 0.3 * dt).max(floor);
                if floor > 0.01 && self.gap <= 0.036 {
                    self.held = true;
                }
            }
            if self.held {
                self.block = Some((ee.0, ee.1 - 0.004));
                self.held_time += dt;
                if self.drop_after_held.is_some_and(|t| self.held_time > t) {
                    self.held = false;
                    self.block = Some((ee.0, 0.02));
                    self.gap = 0.002;
                    self.drop_after_held = None;
                }
            }
            self.time += dt;
            self.step += 1;
        }
    }

    fn goal(cycle: u64, kind: GoalKind, step: u64, x: f64, z: f64) -> Goal {
        Goal {
            cycle,
            kind,
            frame_step: step as u32,
            x,
            z,
        }
    }

    /// Minimal System 2 stand-in for the state machine's own tests: reads the fake plant and publishes the same goals the
    /// scripted reasoner would (reach until holding, then place, then done once the block is on the platform).
    fn system2(plant: &FakePlant, cycle: u64, platform: (f64, f64)) -> Goal {
        let ee = fk(&plant.q);
        let holding = (0.030..=0.046).contains(&plant.gap);
        let block = plant.block.unwrap();
        let on_platform = (block.0 - platform.0).abs() < 0.07 && block.1 > 0.03;
        if on_platform && !holding && dist(ee, block) > 0.05 {
            goal(cycle, GoalKind::Done, plant.step, 0.0, 0.0)
        } else if holding {
            goal(cycle, GoalKind::Place, plant.step, platform.0, platform.1)
        } else {
            goal(cycle, GoalKind::Reach, plant.step, block.0, block.1)
        }
    }

    fn run(plant: &mut FakePlant, task: &mut Task, seconds: f64, platform: (f64, f64)) {
        let dt = 0.01;
        let mut cycle = 0;
        let mut g = goal(0, GoalKind::None, 0, 0.0, 0.0);
        let mut next_s2 = 0.0;
        while plant.time < seconds && !matches!(task.phase(), Phase::Finished | Phase::Failed) {
            if plant.time >= next_s2 {
                cycle += 1;
                g = system2(plant, cycle, platform);
                next_s2 += 0.5;
            }
            let cmd = task.step(&plant.obs(), Some(&g));
            plant.advance(&cmd, dt);
        }
    }

    #[test]
    fn completes_pick_and_place_through_every_phase_in_order() {
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(TaskConfig::default());
        run(&mut plant, &mut task, 60.0, (0.15, 0.04));
        assert_eq!(
            task.phase(),
            Phase::Finished,
            "{:?} {:?}",
            task.failure(),
            task.transitions
        );

        let order: Vec<Phase> = task.transitions.iter().map(|t| t.1).collect();
        let expect = [
            Phase::Home,
            Phase::WaitGoal,
            Phase::Approach,
            Phase::Descend,
            Phase::Grasp,
            Phase::Lift,
            Phase::WaitGoal,
            Phase::Transport,
            Phase::Lower,
            Phase::Release,
            Phase::Retreat,
            Phase::WaitGoal,
            Phase::Finished,
        ];
        assert_eq!(order, expect);
        let b = plant.block.unwrap();
        assert!(
            (b.0 - 0.15).abs() < 0.02 && b.1 > 0.03,
            "block ended at {b:?}"
        );
    }

    #[test]
    fn a_stale_goal_cannot_trigger_a_second_pick_after_the_sequence_ends() {
        // System 2 keeps republishing the pre-grasp REACH goal (frame_step 0) forever, as a hung/lagging System 2 would.
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(TaskConfig::default());
        let stale = goal(1, GoalKind::Reach, 0, 0.45, 0.02);
        // let the first pick-up run with fresh goals so the arm really holds the block ...
        let mut cycle = 0;
        let mut next_s2 = 0.0;
        let mut g = stale;
        while plant.time < 20.0 && task.phase() != Phase::Lift {
            if plant.time >= next_s2 {
                cycle += 1;
                g = system2(&plant, cycle, (0.15, 0.04));
                next_s2 += 0.5;
            }
            let cmd = task.step(&plant.obs(), Some(&g));
            plant.advance(&cmd, 0.01);
        }
        assert_eq!(task.phase(), Phase::Lift);
        // ... then freeze System 2 on a goal from before the lift finished
        let frozen = goal(
            cycle,
            GoalKind::Reach,
            plant.step.saturating_sub(10),
            0.45,
            0.02,
        );
        for _ in 0..3000 {
            let cmd = task.step(&plant.obs(), Some(&frozen));
            plant.advance(&cmd, 0.01);
        }
        assert_eq!(
            task.phase(),
            Phase::WaitGoal,
            "must idle, not restart: {:?}",
            task.transitions
        );
        assert_eq!(
            task.transitions
                .iter()
                .filter(|t| t.1 == Phase::Approach)
                .count(),
            1
        );
    }

    #[test]
    fn missing_the_block_retries_then_fails_cleanly() {
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(TaskConfig::default());
        // System 2 insists the block is 8 cm from where it really is, so every grasp closes on nothing
        let wrong = (0.37, 0.02);
        let dt = 0.01;
        let mut cycle = 0;
        while plant.time < 90.0 && task.phase() != Phase::Failed {
            cycle += 1;
            let g = goal(cycle, GoalKind::Reach, plant.step, wrong.0, wrong.1);
            let cmd = task.step(&plant.obs(), Some(&g));
            plant.advance(&cmd, dt);
        }
        assert_eq!(task.phase(), Phase::Failed);
        assert!(
            task.failure().unwrap().contains("grasp failed"),
            "{:?}",
            task.failure()
        );
        assert_eq!(
            task.grasp_retries(),
            TaskConfig::default().max_grasp_retries + 1
        );
    }

    #[test]
    fn a_dropped_block_aborts_the_carry_and_waits_for_a_fresh_goal() {
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        plant.drop_after_held = Some(2.5);
        let mut task = Task::new(TaskConfig::default());
        run(&mut plant, &mut task, 40.0, (0.15, 0.04));
        assert!(
            task.transitions.iter().any(|t| t.1 == Phase::Recover),
            "drop not detected: {:?}",
            task.transitions
        );
        // after recovering, System 2 sees the block on the table and the whole pick-and-place is retried to completion
        assert_eq!(
            task.phase(),
            Phase::Finished,
            "{:?} {:?}",
            task.failure(),
            task.transitions
        );
    }

    #[test]
    fn system_2_failure_goal_stops_the_arm_in_place() {
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(TaskConfig::default());
        for _ in 0..400 {
            let g = goal(1, GoalKind::Reach, plant.step, 0.45, 0.02);
            let cmd = task.step(&plant.obs(), Some(&g));
            plant.advance(&cmd, 0.01);
        }
        let g = goal(2, GoalKind::Failed, plant.step, 0.0, 0.0);
        let before = task.step(&plant.obs(), Some(&g)).ee_target;
        plant.advance(
            &Command {
                ee_target: before,
                grip_open: true,
            },
            0.01,
        );
        let after = task.step(&plant.obs(), Some(&g)).ee_target;
        assert_eq!(task.phase(), Phase::Failed);
        assert_eq!(before, after, "the setpoint must stop moving");
    }

    #[test]
    fn the_setpoint_never_exceeds_the_acceleration_limit() {
        let cfg = TaskConfig::default();
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(cfg.clone());
        let mut hist: Vec<(Phase, (f64, f64))> = Vec::new();
        let mut worst: f64 = 0.0;
        for _ in 0..3000 {
            let g = goal(1, GoalKind::Reach, plant.step, 0.45, 0.02);
            let cmd = task.step(&plant.obs(), Some(&g));
            hist.push((task.phase(), cmd.ee_target));
            if let [.., (p0, a), (p1, b), (p2, c)] = hist[..]
                && p0 == p1
                && p1 == p2
                && p1 != Phase::Home
            {
                // velocities over the two intervals, inside one phase (a phase change deliberately restarts from rest)
                let (v1, v2) = (
                    ((b.0 - a.0) / 0.01, (b.1 - a.1) / 0.01),
                    ((c.0 - b.0) / 0.01, (c.1 - b.1) / 0.01),
                );
                worst = worst.max(dist(v1, v2) / 0.01);
            }
            plant.advance(&cmd, 0.01);
        }
        assert!(worst > 0.1, "the test never saw the setpoint accelerate");
        assert!(
            worst <= cfg.accel * 1.001 + 1e-9,
            "setpoint acceleration {worst} m/s^2 exceeds {}",
            cfg.accel
        );
    }

    #[test]
    fn the_setpoint_never_outruns_the_speed_limit() {
        let mut plant = FakePlant::new(Some((0.45, 0.02)));
        let mut task = Task::new(TaskConfig::default());
        let mut prev: Option<(f64, f64)> = None;
        let mut worst: f64 = 0.0;
        for _ in 0..3000 {
            let g = goal(1, GoalKind::Reach, plant.step, 0.45, 0.02);
            let cmd = task.step(&plant.obs(), Some(&g));
            if task.phase() != Phase::Home {
                if let Some(p) = prev {
                    worst = worst.max(dist(p, cmd.ee_target) / 0.01);
                }
                prev = Some(cmd.ee_target);
            }
            plant.advance(&cmd, 0.01);
        }
        assert!(
            worst <= TaskConfig::default().speed + 1e-6,
            "setpoint speed {worst}"
        );
    }
}
