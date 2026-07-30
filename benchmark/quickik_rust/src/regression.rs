//! Compares fresh `Solver::solve` latency measurements against baseline
//! numbers recorded before the solver API redesign (config flattened into
//! constructors, `with_grad`/`with_fk` flags), flagging any measurement more
//! than [`REGRESSION_THRESHOLD`] slower than its baseline. Not a strict
//! pass/fail gate, since machine load varies -- a flag is a prompt to
//! re-check under controlled conditions, not proof of an actual regression.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;
use quickik::forward::{ForwardKinematicsWorkspace, evaluate_fwdkin};
use quickik::observation::KeypointObservation;
use quickik::state::State;

use crate::perf::BenchConfig;

const WARMUP_ITERS: usize = 500;
const TIMED_ITERS: usize = 20_000;
/// Flag a measurement once it's slower than its baseline by more than this
/// fraction.
const REGRESSION_THRESHOLD: f64 = 0.05;

/// Baseline single-frame `solve()` latencies (microseconds) and the `with_fk`
/// overhead fraction, recorded on this session's own reference machine prior
/// to the solver API redesign.
struct Baseline {
    cold_start_us: f64,
    warm_started_us: f64,
    warm_started_fk_overhead_fraction: f64,
}

const NEUROMECHFLY_BASELINE: Baseline = Baseline {
    cold_start_us: 40.0,
    warm_started_us: 14.0,
    warm_started_fk_overhead_fraction: 0.10,
};
const G1_BASELINE: Baseline = Baseline {
    cold_start_us: 35.0,
    warm_started_us: 12.0,
    warm_started_fk_overhead_fraction: 0.10,
};

/// A synthetic target close to (but not exactly at) the neutral pose:
/// small enough that a fresh solve needs only a handful of Gauss-Newton
/// iterations, matching the difficulty of the scenario these baselines were
/// recorded against. A fixture-derived target can require many more
/// iterations to converge, which would make cold-start latency reflect
/// target difficulty rather than any change in per-iteration cost.
fn small_perturbation_target(tree: &Arc<KinematicTree>) -> Vec<KeypointObservation> {
    let mut perturbed = State::neutral_pose(tree.clone());
    for (i, angle) in perturbed.dof_angles.iter_mut().enumerate() {
        *angle += 0.1 * (i as f32 * 1.37).sin();
    }
    let mut workspace = ForwardKinematicsWorkspace::new(tree);
    evaluate_fwdkin(&mut workspace, &perturbed);

    workspace
        .kpt_positions
        .iter()
        .map(|&obs_pos: &Vector3<f32>| KeypointObservation::Position3D {
            obs_pos,
            weight: 1.0,
        })
        .collect()
}

/// Mean `solve()` latency in microseconds, over `TIMED_ITERS` timed calls
/// (after `WARMUP_ITERS` warmup calls). `cold_start`: a fresh
/// `State::neutral_pose()` every call. Otherwise: one shared `State`,
/// converged once up front then reused for every call, so each timed call
/// only takes the ~1 iteration a warm-started sequence would see once caught
/// up to a stationary target.
fn mean_solve_us(
    tree: &Arc<KinematicTree>,
    target_obs: &[KeypointObservation],
    with_fk: bool,
    cold_start: bool,
) -> f64 {
    let mut solver = BenchConfig::default_3d().new_solver(tree);

    let elapsed = if cold_start {
        for _ in 0..WARMUP_ITERS {
            let mut state = State::neutral_pose(tree.clone());
            solver.solve(&mut state, black_box(target_obs), false, with_fk);
        }
        let t0 = Instant::now();
        for _ in 0..TIMED_ITERS {
            let mut state = State::neutral_pose(tree.clone());
            solver.solve(&mut state, black_box(target_obs), false, with_fk);
        }
        t0.elapsed()
    } else {
        let mut state = State::neutral_pose(tree.clone());
        solver.solve(&mut state, target_obs, false, false);
        for _ in 0..WARMUP_ITERS {
            solver.solve(&mut state, black_box(target_obs), false, with_fk);
        }
        let t0 = Instant::now();
        for _ in 0..TIMED_ITERS {
            solver.solve(&mut state, black_box(target_obs), false, with_fk);
        }
        t0.elapsed()
    };

    elapsed.as_secs_f64() * 1e6 / TIMED_ITERS as f64
}

/// Prints a before/after comparison against `body`'s recorded baseline (see
/// [`NEUROMECHFLY_BASELINE`]/[`G1_BASELINE`]), for cold-start latency,
/// warm-started latency, and the `with_fk` overhead fraction. Skipped for any
/// `body` without a recorded baseline.
pub fn run(tree: &Arc<KinematicTree>, body: &str) {
    let baseline = match body {
        "neuromechfly" => &NEUROMECHFLY_BASELINE,
        "g1" => &G1_BASELINE,
        _ => {
            println!("-- regression check: no baseline recorded for body '{body}', skipped --\n");
            return;
        }
    };

    let target_obs = small_perturbation_target(tree);

    let cold_start_us = mean_solve_us(tree, &target_obs, false, true);
    let warm_started_us = mean_solve_us(tree, &target_obs, false, false);
    let warm_started_fk_us = mean_solve_us(tree, &target_obs, true, false);
    let warm_started_fk_overhead_fraction = warm_started_fk_us / warm_started_us - 1.0;

    println!("-- regression check (vs. baseline recorded before the solver API redesign) --");
    print_comparison("cold start", cold_start_us, baseline.cold_start_us, "us");
    print_comparison(
        "warm started",
        warm_started_us,
        baseline.warm_started_us,
        "us",
    );
    print_comparison(
        "warm started, with_fk overhead",
        warm_started_fk_overhead_fraction * 100.0,
        baseline.warm_started_fk_overhead_fraction * 100.0,
        "%",
    );
    println!();
}

fn print_comparison(label: &str, current: f64, baseline: f64, unit: &str) {
    let delta_fraction = current / baseline - 1.0;
    let flag = if delta_fraction > REGRESSION_THRESHOLD {
        "  <-- REGRESSION?"
    } else {
        ""
    };
    println!(
        "{label:<32} current={current:>8.3}{unit}  baseline={baseline:>8.3}{unit}  \
         delta={:>+7.1}%{flag}",
        delta_fraction * 100.0,
    );
}
