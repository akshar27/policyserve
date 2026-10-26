//! M4's System 1 loop: every tick, read the latest proprioception and goal from shared memory (never blocking), advance the task
//! state machine, turn its Cartesian setpoint into joint velocities with the closed-form IK + proportional controller under the
//! hard deadline, and publish the action. The sim host (Python) applies the newest action every control period.
//!
//! The loop is deliberately ignorant of System 2: it sees a goal record and how long ago its `source_cycle` last changed. If
//! System 2 hangs, the record just stops changing, the age grows, and this loop keeps ticking at its own rate on the last good
//! goal (`max_goal_age_s` in the summary shows how long that went on; M5 turns that into the headline test).
//! M5 also adds the loop's own steadiness (`loop_dt_ns_*`) and the `--inject-slow-*` hook that deliberately blows the tick budget.

use std::time::{Duration, Instant};

use crate::controller::{
    JointConfig, NUM_JOINTS, ProportionalController, SafeController, SlowEvery,
};
use crate::kinematics::ik;
use crate::shm::SeqLockChannel;
use crate::task::{Observation, Phase, Task, TaskConfig};
use crate::wire::{ACTION_SIZE, Action, FLAG_STOP, GOAL_SIZE, Goal, PROPRIO_SIZE, Proprio};

pub struct ControlArgs {
    pub goal_channel: String,
    pub proprio_channel: String,
    pub action_channel: String,
    pub rate_hz: u64,
    pub duration_secs: f64,
    pub deadline_us: u64,
    /// Fault injection: slow every Nth controller call by `inject_slow_us` (0 = off). See `SlowEvery`.
    pub inject_slow_every: u64,
    pub inject_slow_us: u64,
}

pub struct Summary {
    pub ticks: u64,
    pub elapsed_secs: f64,
    pub deadline_misses: u32,
    pub tick_ns_p50: u64,
    pub tick_ns_p99: u64,
    pub tick_ns_max: u64,
    /// Time between the starts of consecutive loop iterations: how steadily the loop actually ticked (target = 1/rate).
    pub loop_dt_ns_p50: u64,
    pub loop_dt_ns_p99: u64,
    pub loop_dt_ns_max: u64,
    pub injected_slow_ticks: u64,
    pub max_goal_age_s: f64,
    pub distinct_goals_seen: u64,
    pub final_phase: Phase,
    pub failure: Option<String>,
    pub grasp_retries: u32,
    pub transitions: Vec<(f64, Phase)>,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

pub fn run(args: &ControlArgs) -> Summary {
    let open = |name: &str, size: usize| {
        SeqLockChannel::open_existing(name, size, Duration::from_secs(5)).unwrap_or_else(|e| {
            panic!("failed to open channel {name}: {e} (did Python create it first?)")
        })
    };
    let goal_chan = open(&args.goal_channel, GOAL_SIZE);
    let proprio_chan = open(&args.proprio_channel, PROPRIO_SIZE);
    let action_chan = open(&args.action_channel, ACTION_SIZE);

    let period = Duration::from_secs_f64(1.0 / args.rate_hz as f64);
    let run_for = Duration::from_secs_f64(args.duration_secs);
    let controller = SafeController::new(
        SlowEvery::new(
            ProportionalController {
                kp: 8.0,
                max_velocity: 2.5,
            },
            args.inject_slow_every,
            Duration::from_micros(args.inject_slow_us),
        ),
        Duration::from_micros(args.deadline_us),
        [0.0; NUM_JOINTS], // safe fallback: hold still
    );
    let mut task = Task::new(TaskConfig::default());

    let mut proprio: Option<Proprio> = None;
    let mut goal: Option<Goal> = None;
    let mut goal_cycle_seen: Option<u64> = None;
    let mut goal_changed_at = Instant::now();
    let mut max_goal_age = 0.0f64;
    let mut distinct_goals = 0u64;
    let mut tick_ns: Vec<u64> = Vec::new();
    let mut loop_dt_ns: Vec<u64> = Vec::new();
    let mut last_loop_start: Option<Instant> = None;
    let mut misses = 0u32;
    let mut ticks = 0u64;
    let mut last_q_target: JointConfig = [0.0; NUM_JOINTS];

    let start = Instant::now();
    let mut next_tick = start + period;
    while start.elapsed() < run_for {
        let loop_start = Instant::now();
        if let Some(prev) = last_loop_start {
            loop_dt_ns.push(loop_start.duration_since(prev).as_nanos() as u64);
        }
        last_loop_start = Some(loop_start);
        // ---- inputs: latest value wins, never blocks; a failed read keeps the previous value ----
        let mut pbuf = [0u8; PROPRIO_SIZE];
        if proprio_chan.read(&mut pbuf, 50) {
            proprio = Some(Proprio::decode(&pbuf));
        }
        let mut gbuf = [0u8; GOAL_SIZE];
        if goal_chan.read(&mut gbuf, 50)
            && let Some(g) = Goal::decode(&gbuf)
        {
            if goal_cycle_seen != Some(g.cycle) {
                goal_cycle_seen = Some(g.cycle);
                goal_changed_at = Instant::now();
                distinct_goals += 1;
            }
            goal = Some(g);
        }
        if goal_cycle_seen.is_some() {
            max_goal_age = max_goal_age.max(goal_changed_at.elapsed().as_secs_f64());
        }

        if let Some(p) = proprio {
            if p.flags & FLAG_STOP != 0 {
                break;
            }
            // ---- one tick of work (this whole block is what the tick-time histogram measures) ----
            let t0 = Instant::now();
            let cmd = task.step(
                &Observation {
                    sim_step: p.sim_step,
                    sim_time: p.sim_time,
                    q: p.q,
                    gap: p.gap,
                },
                goal.as_ref(),
            );
            let (q_target, _reachable) = ik(cmd.ee_target.0, cmd.ee_target.1);
            last_q_target = q_target;
            let result = controller.step(&q_target, &p.q);
            let compute_ns = t0.elapsed().as_nanos() as u64;
            if result.missed_deadline {
                misses += 1;
            }
            tick_ns.push(compute_ns);

            action_chan.write(
                &Action {
                    tick: ticks,
                    vel: result.action,
                    grip_open: if cmd.grip_open { 1.0 } else { 0.0 },
                    phase: task.phase().code(),
                    deadline_misses: misses,
                    compute_ns,
                    goal_cycle: goal_cycle_seen.unwrap_or(0),
                }
                .encode(),
            );
        }
        ticks += 1;

        let now = Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        }
        next_tick += period;
    }
    let _ = last_q_target;

    tick_ns.sort_unstable();
    loop_dt_ns.sort_unstable();
    Summary {
        ticks,
        elapsed_secs: start.elapsed().as_secs_f64(),
        deadline_misses: misses,
        tick_ns_p50: percentile(&tick_ns, 0.50),
        tick_ns_p99: percentile(&tick_ns, 0.99),
        tick_ns_max: tick_ns.last().copied().unwrap_or(0),
        loop_dt_ns_p50: percentile(&loop_dt_ns, 0.50),
        loop_dt_ns_p99: percentile(&loop_dt_ns, 0.99),
        loop_dt_ns_max: loop_dt_ns.last().copied().unwrap_or(0),
        injected_slow_ticks: controller.inner().injected(),
        max_goal_age_s: max_goal_age,
        distinct_goals_seen: distinct_goals,
        final_phase: task.phase(),
        failure: task.failure().map(str::to_owned),
        grasp_retries: task.grasp_retries(),
        transitions: task.transitions.clone(),
    }
}

impl Summary {
    /// One JSON line on stdout; hand-rolled for the same reason as the M1 harness (no other use for a JSON library here).
    pub fn to_json(&self) -> String {
        let transitions = self
            .transitions
            .iter()
            .map(|(t, p)| format!("[{t:.3},\"{}\"]", p.name()))
            .collect::<Vec<_>>()
            .join(",");
        let failure = match &self.failure {
            Some(f) => format!("\"{}\"", f.replace(['"', '\\'], "'")),
            None => "null".to_string(),
        };
        format!(
            "{{\"ticks\":{},\"elapsed_secs\":{:.3},\"deadline_misses\":{},\"tick_ns_p50\":{},\"tick_ns_p99\":{},\"tick_ns_max\":{},\"loop_dt_ns_p50\":{},\"loop_dt_ns_p99\":{},\"loop_dt_ns_max\":{},\"injected_slow_ticks\":{},\"max_goal_age_s\":{:.3},\"distinct_goals_seen\":{},\"final_phase\":\"{}\",\"failure\":{},\"grasp_retries\":{},\"transitions\":[{}]}}",
            self.ticks,
            self.elapsed_secs,
            self.deadline_misses,
            self.tick_ns_p50,
            self.tick_ns_p99,
            self.tick_ns_max,
            self.loop_dt_ns_p50,
            self.loop_dt_ns_p99,
            self.loop_dt_ns_max,
            self.injected_slow_ticks,
            self.max_goal_age_s,
            self.distinct_goals_seen,
            self.final_phase.name(),
            failure,
            self.grasp_retries,
            transitions
        )
    }
}
