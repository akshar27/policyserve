//! M1: a standalone tick loop that reads the latest value from one
//! shared-memory channel and writes an incrementing one to another, at
//! a configured rate, for a configured duration. This is the whole
//! System 1 side of the cross-language handoff in isolation -- no
//! controller, no sim, no deadline enforcement yet (that's M2) -- so the
//! riskiest unknown (can Rust and Python actually hand data to each
//! other this way, under real concurrent load, without corruption or
//! either side blocking) gets proven on its own before anything else is
//! built on top of it.
//!
//! On exit, prints one JSON line to stdout summarizing what happened
//! (ticks run, read-channel samples observed, timing), so the Python
//! side of the M1 test can assert on it without scraping log output.

mod control;
mod controller;
mod kinematics;
mod shm;
mod task;
mod wire;

use std::time::{Duration, Instant};

use clap::Parser;
use shm::SeqLockChannel;

/// Payload layout for this M1 harness: an 8-byte counter mirrored into
/// two 8-byte halves, so a reader can detect a torn read (the two
/// halves disagreeing) independently of the seqlock's own protection --
/// belt and suspenders for the one test (M1) where catching a real
/// corruption bug matters more than anywhere else in this project.
const PAYLOAD_SIZE: usize = 16;

fn encode(counter: u64) -> [u8; PAYLOAD_SIZE] {
    let mut buf = [0u8; PAYLOAD_SIZE];
    buf[0..8].copy_from_slice(&counter.to_le_bytes());
    buf[8..16].copy_from_slice(&counter.to_le_bytes());
    buf
}

fn decode(buf: &[u8; PAYLOAD_SIZE]) -> Option<u64> {
    let a = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let b = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    if a == b { Some(a) } else { None }
}

#[derive(Parser, Debug)]
struct Args {
    /// `harness` = the M1 shared-memory counter exchange; `control` = the M4 System 1 control loop.
    #[arg(long, default_value = "harness")]
    mode: String,
    /// (harness) channel Python writes to and this process reads from.
    #[arg(long)]
    read_channel: Option<String>,
    /// (harness) channel this process writes to and Python reads from.
    #[arg(long)]
    write_channel: Option<String>,
    /// (control) System 2 -> System 1 goal record.
    #[arg(long)]
    goal_channel: Option<String>,
    /// (control) sim host -> System 1 proprioception record.
    #[arg(long)]
    proprio_channel: Option<String>,
    /// (control) System 1 -> sim host action record.
    #[arg(long)]
    action_channel: Option<String>,
    /// (fk) comma-separated joint angles, radians.
    #[arg(long)]
    joints: Option<String>,
    /// (ik) comma-separated grip-point target "x,z", metres.
    #[arg(long)]
    target: Option<String>,
    /// (control) per-tick compute budget; a controller step over this is discarded for the safe fallback.
    #[arg(long, default_value_t = 5000)]
    deadline_us: u64,
    /// (control, fault injection) make every Nth controller call take `--inject-slow-us` extra; 0 = off.
    #[arg(long, default_value_t = 0)]
    inject_slow_every: u64,
    /// (control, fault injection) extra delay per injected slow call, microseconds.
    #[arg(long, default_value_t = 0)]
    inject_slow_us: u64,
    /// Tick rate in Hz.
    #[arg(long, default_value_t = 100)]
    rate_hz: u64,
    /// How long to run before exiting.
    #[arg(long, default_value_t = 2.0)]
    duration_secs: f64,
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    match args.mode.as_str() {
        "harness" => run_harness(&args),
        "control" => {
            let need = |v: &Option<String>, name: &str| {
                v.clone()
                    .unwrap_or_else(|| panic!("--{name} is required in control mode"))
            };
            let summary = control::run(&control::ControlArgs {
                goal_channel: need(&args.goal_channel, "goal-channel"),
                proprio_channel: need(&args.proprio_channel, "proprio-channel"),
                action_channel: need(&args.action_channel, "action-channel"),
                rate_hz: args.rate_hz,
                duration_secs: args.duration_secs,
                deadline_us: args.deadline_us,
                inject_slow_every: args.inject_slow_every,
                inject_slow_us: args.inject_slow_us,
            });
            println!("{}", summary.to_json());
        }
        // Kinematics probes: let the Python tests compare this binary's FK/IK against MuJoCo's own geometry.
        "fk" => {
            let q = parse_floats(
                args.joints
                    .as_deref()
                    .expect("--joints required in fk mode"),
            );
            let (x, z) = kinematics::fk(&[q[0], q[1], q[2]]);
            println!("{{\"x\":{x},\"z\":{z}}}");
        }
        "ik" => {
            let t = parse_floats(
                args.target
                    .as_deref()
                    .expect("--target required in ik mode"),
            );
            let (q, reachable) = kinematics::ik(t[0], t[1]);
            println!(
                "{{\"q\":[{},{},{}],\"reachable\":{reachable}}}",
                q[0], q[1], q[2]
            );
        }
        other => panic!("unknown --mode {other:?} (expected harness, control, fk or ik)"),
    }
}

fn parse_floats(s: &str) -> Vec<f64> {
    s.split(',')
        .map(|v| v.trim().parse().expect("not a number"))
        .collect()
}

fn run_harness(args: &Args) {
    let read_channel = args
        .read_channel
        .clone()
        .expect("--read-channel is required in harness mode");
    let write_channel = args
        .write_channel
        .clone()
        .expect("--write-channel is required in harness mode");
    let period = Duration::from_secs_f64(1.0 / args.rate_hz as f64);
    let run_for = Duration::from_secs_f64(args.duration_secs);

    tracing::info!(
        rate_hz = args.rate_hz,
        read_channel = %read_channel,
        write_channel = %write_channel,
        "system1 opening channels"
    );
    let read_chan =
        SeqLockChannel::open_existing(&read_channel, PAYLOAD_SIZE, Duration::from_secs(5))
            .expect("failed to open read channel (did Python create it first?)");
    let write_chan =
        SeqLockChannel::open_existing(&write_channel, PAYLOAD_SIZE, Duration::from_secs(5))
            .expect("failed to open write channel (did Python create it first?)");

    let mut ticks: u64 = 0;
    let mut read_samples_seen: u64 = 0;
    let mut read_torn_or_missed: u64 = 0;
    let mut last_read_value: Option<u64> = None;
    let mut distinct_read_values: u64 = 0;

    let start = Instant::now();
    let mut next_tick = start + period;
    while start.elapsed() < run_for {
        let mut buf = [0u8; PAYLOAD_SIZE];
        if read_chan.read(&mut buf, 50) {
            match decode(&buf) {
                Some(v) => {
                    read_samples_seen += 1;
                    if last_read_value != Some(v) {
                        distinct_read_values += 1;
                        last_read_value = Some(v);
                    }
                }
                None => read_torn_or_missed += 1,
            }
        } else {
            read_torn_or_missed += 1;
        }

        write_chan.write(&encode(ticks));
        ticks += 1;

        let now = Instant::now();
        if now < next_tick {
            std::thread::sleep(next_tick - now);
        }
        next_tick += period;
    }

    let summary = serde_like_json(
        ticks,
        read_samples_seen,
        read_torn_or_missed,
        distinct_read_values,
        start.elapsed().as_secs_f64(),
    );
    println!("{summary}");
}

/// Hand-rolled instead of pulling in `serde_json` for one output line --
/// this binary has no other use for a JSON library.
fn serde_like_json(
    ticks: u64,
    read_samples_seen: u64,
    read_torn_or_missed: u64,
    distinct_read_values: u64,
    elapsed_secs: f64,
) -> String {
    format!(
        "{{\"ticks\":{ticks},\"read_samples_seen\":{read_samples_seen},\"read_torn_or_missed\":{read_torn_or_missed},\"distinct_read_values\":{distinct_read_values},\"elapsed_secs\":{elapsed_secs:.3}}}"
    )
}
