//! High-level APIs for a single tracked body over consecutive frames
//! ([`SequenceSolver`]), and for one long sequence solved in parallel via
//! overlapping segments ([`solve_sequence_segmented_parallel`]). For
//! single-frame or other specialized use cases, use [`Solver`] directly.
//!
//! [`Solver`]: crate::solver::Solver

use std::sync::Arc;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, Mapper3Dto2D, NoMapper};
use crate::solver::{Solver, SolverConfig};
use crate::state::State;

/// Solves a continuous sequence of frames for a single tracked body, warm
/// starting each frame from the previous frame's converged pose.
pub struct SequenceSolver<M: Mapper3Dto2D = NoMapper> {
    pub solver: Solver<M>,
    pub state: State,
}

impl<M: Mapper3Dto2D> SequenceSolver<M> {
    /// Starts a new sequence at the neutral pose.
    pub fn new(kinematic_tree: Arc<KinematicTree>, config: SolverConfig<M>) -> Self {
        let solver = Solver::new(&kinematic_tree, config);
        let state = State::neutral_pose(kinematic_tree);
        Self { solver, state }
    }

    /// Solves the next frame in place, warm-started from the current pose, and
    /// returns the converged state.
    pub fn solve_frame(&mut self, observations: &[KeypointObservation]) -> &State {
        self.solver.solve(&mut self.state, observations);
        &self.state
    }

    /// Solves every frame in `sequence` in order, each warm-started from the
    /// previous one, returning the converged pose after each frame.
    pub fn solve_sequence(&mut self, sequence: &[Vec<KeypointObservation>]) -> Vec<State> {
        sequence
            .iter()
            .map(|observations| self.solve_frame(observations).clone())
            .collect()
    }
}

/// Configuration for [`solve_sequence_segmented_parallel`].
#[derive(Clone, Copy, Debug)]
pub struct ParallelSolveConfig {
    /// Frames per segment, including overlap with neighbors. Must be
    /// greater than `overlap_len`.
    pub segment_len: usize,
    /// Frames shared with the next segment: gives it a warm-started running
    /// start, and a window to cross-check consistency against afterward.
    pub overlap_len: usize,
    /// Maximum per-DOF angle disagreement (radians) allowed between
    /// neighboring segments' overlapping frames before logging a warning.
    pub overlap_tolerance: f32,
    /// Number of worker threads. A positive value is used directly, unless it
    /// exceeds the number of available cores: it's then clipped to that
    /// count and a warning is logged. A negative value counts backward
    /// from all available cores: `-1` uses all, `-2` uses all but one, etc.
    /// `0` is invalid.
    pub n_workers: isize,
}

/// [`ParallelSolveConfig::for_recording`]'s default `overlap_len`.
const DEFAULT_OVERLAP_LEN: usize = 10;
/// [`ParallelSolveConfig::for_recording`]'s default `overlap_tolerance`.
const DEFAULT_OVERLAP_TOLERANCE: f32 = 0.05;

impl ParallelSolveConfig {
    /// A `ParallelSolveConfig` that spreads `total_len` frames evenly across
    /// every available core: one segment per core, `total_len / n_workers`
    /// frames each (plus a fixed default overlap of 10 frames on top). For
    /// finer control over cold-start frequency (how often a segment
    /// restarts from the neutral pose, trading accuracy for finer-grained
    /// parallelism), build a `ParallelSolveConfig` directly instead.
    pub fn for_recording(total_len: usize) -> Self {
        let n_workers = resolve_n_workers(-1);
        let overlap_len = DEFAULT_OVERLAP_LEN;
        let stride = total_len.max(1).div_ceil(n_workers);
        ParallelSolveConfig {
            segment_len: stride + overlap_len,
            overlap_len,
            overlap_tolerance: DEFAULT_OVERLAP_TOLERANCE,
            n_workers: -1,
        }
    }
}

/// Resolves [`ParallelSolveConfig::n_workers`] into an actual thread count –
/// see its docs for the exact convention.
fn resolve_n_workers(n_workers: isize) -> usize {
    assert!(
        n_workers != 0,
        "ParallelSolveConfig::n_workers must not be 0"
    );
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    if n_workers > 0 {
        let n_workers = n_workers as usize;
        if n_workers > available {
            log::warn!(
                "ParallelSolveConfig::n_workers ({n_workers}) exceeds available cores \
                 ({available}); clipping to {available}"
            );
            return available;
        }
        return n_workers;
    }
    (available as isize + 1 + n_workers).max(1) as usize
}

/// Solves a single long sequence in parallel by splitting it into slightly
/// overlapping segments (see [`ParallelSolveConfig`]), each solved on its
/// own thread via [`solve_sequence`](SequenceSolver::solve_sequence).
/// Since segments are solved independently, their overlapping frames can
/// converge to slightly different poses; any disagreement in `dof_angles`
/// beyond `parallel_config.overlap_tolerance` generates a warning and the
/// earlier segment's version is kept.
pub fn solve_sequence_segmented_parallel<M: Mapper3Dto2D + Sync>(
    kinematic_tree: &Arc<KinematicTree>,
    config: SolverConfig<M>,
    sequence: &[Vec<KeypointObservation>],
    parallel_config: ParallelSolveConfig,
) -> Vec<State> {
    let bounds = segment_bounds(sequence.len(), parallel_config);
    let n_workers = resolve_n_workers(parallel_config.n_workers);
    let segment_states = solve_in_parallel(&bounds, n_workers, |&(start, end)| {
        SequenceSolver::new(Arc::clone(kinematic_tree), config)
            .solve_sequence(&sequence[start..end])
    });
    stitch_overlapping_segments(
        segment_states,
        parallel_config.overlap_len,
        parallel_config.overlap_tolerance,
    )
}

/// Applies `solve_one` to every item in `items`, spread across at most
/// `n_workers` threads, preserving order.
fn solve_in_parallel<T: Sync, R: Send>(
    items: &[T],
    n_workers: usize,
    solve_one: impl Fn(&T) -> R + Sync,
) -> Vec<R> {
    if items.is_empty() {
        return Vec::new();
    }
    let n_threads = n_workers.min(items.len());
    let chunk_size = items.len().div_ceil(n_threads);

    std::thread::scope(|scope| {
        // Spawn every chunk's thread before joining any of them, so the
        // chunks actually run concurrently rather than one at a time.
        items
            .chunks(chunk_size)
            .map(|chunk| scope.spawn(|| chunk.iter().map(&solve_one).collect::<Vec<_>>()))
            .collect::<Vec<_>>()
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    })
}

/// Splits `total_len` frames into overlapping `(start, end)` (end-exclusive)
/// segment bounds per `config.segment_len`/`config.overlap_len`.
fn segment_bounds(total_len: usize, config: ParallelSolveConfig) -> Vec<(usize, usize)> {
    assert!(
        config.overlap_len < config.segment_len,
        "overlap_len must be smaller than segment_len"
    );
    if total_len == 0 {
        return Vec::new();
    }
    if total_len <= config.segment_len {
        return vec![(0, total_len)];
    }

    let stride = config.segment_len - config.overlap_len;
    let mut bounds = Vec::new();
    let mut start = 0;
    loop {
        let end = (start + config.segment_len).min(total_len);
        bounds.push((start, end));
        if end == total_len {
            break;
        }
        start += stride;
    }
    bounds
}

/// Concatenates overlapping `segment_states` into one sequence, dropping each
/// segment's overlap with the previous one (i.e. results from the previous
/// segment are used). Logs a warning if any overlapping frame disagrees between
/// segments by more than `overlap_tolerance`.
fn stitch_overlapping_segments(
    segment_states: Vec<Vec<State>>,
    overlap_len: usize,
    overlap_tolerance: f32,
) -> Vec<State> {
    let mut result: Vec<State> = Vec::new();
    for segment in segment_states {
        if result.is_empty() {
            result.extend(segment);
            continue;
        }

        let overlap_start = result.len() - overlap_len;
        for i in 0..overlap_len {
            let max_angle_diff = result[overlap_start + i]
                .dof_angles
                .iter()
                .zip(&segment[i].dof_angles)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if max_angle_diff > overlap_tolerance {
                log::warn!(
                    "solve_sequence_segmented_parallel: overlapping frame {} disagrees between \
                     neighboring segments by {max_angle_diff:.4} rad (tolerance {overlap_tolerance:.4})",
                    overlap_start + i,
                );
            }
        }
        result.extend(segment.into_iter().skip(overlap_len));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_plan::{Dof, DofType, Joint};
    use nalgebra::{UnitQuaternion, Vector3};

    fn minimal_tree() -> Arc<KinematicTree> {
        Arc::new(KinematicTree::new(
            vec![Joint {
                name: "root".to_string(),
                offset_pos: Vector3::zeros(),
                offset_quat: UnitQuaternion::identity(),
                dofs: vec![Dof {
                    axis: Vector3::z(),
                    dof_type: DofType::Hinge,
                    neutral: 0.0,
                    limits: None,
                    weight_scaler: 1.0,
                }],
                parent: None,
                children: Vec::new(),
                dof_offset: 0,
                weight_scaler: 1.0,
            }],
            0,
        ))
    }

    fn state_with_angle(tree: &Arc<KinematicTree>, angle: f32) -> State {
        let mut state = State::neutral_pose(tree.clone());
        state.dof_angles[0] = angle;
        state
    }

    fn parallel_config(segment_len: usize, overlap_len: usize) -> ParallelSolveConfig {
        ParallelSolveConfig {
            segment_len,
            overlap_len,
            overlap_tolerance: 0.01,
            n_workers: -1,
        }
    }

    #[test]
    fn segment_bounds_covers_whole_sequence_with_expected_overlap() {
        let config = parallel_config(10, 3);
        let bounds = segment_bounds(25, config);

        assert_eq!(bounds.first().unwrap().0, 0);
        assert_eq!(bounds.last().unwrap().1, 25);
        for window in bounds.windows(2) {
            let (_, prev_end) = window[0];
            let (next_start, _) = window[1];
            assert_eq!(
                prev_end - next_start,
                config.overlap_len,
                "consecutive segments should overlap by exactly overlap_len"
            );
        }
        for &(start, end) in &bounds {
            assert!(
                end - start > config.overlap_len,
                "every segment should contain at least one non-overlapping frame"
            );
        }
    }

    #[test]
    fn segment_bounds_handles_short_and_empty_sequences() {
        let config = parallel_config(10, 3);
        assert_eq!(segment_bounds(7, config), vec![(0, 7)]);
        assert_eq!(segment_bounds(0, config), Vec::<(usize, usize)>::new());
    }

    #[test]
    #[should_panic(expected = "overlap_len must be smaller than segment_len")]
    fn segment_bounds_rejects_overlap_not_smaller_than_segment_len() {
        segment_bounds(20, parallel_config(5, 5));
    }

    #[test]
    fn stitch_keeps_earlier_segments_overlap_values_and_drops_duplicates() {
        let tree = minimal_tree();
        let segment_a = vec![
            state_with_angle(&tree, 0.0),
            state_with_angle(&tree, 0.1),
            state_with_angle(&tree, 0.2),
        ];
        // segment_b's first frame overlaps segment_a's last frame, and
        // (as independently-solved segments can) disagrees slightly.
        let segment_b = vec![
            state_with_angle(&tree, 0.2 + 1e-4),
            state_with_angle(&tree, 0.3),
        ];

        let stitched = stitch_overlapping_segments(vec![segment_a, segment_b], 1, 0.01);

        let angles: Vec<f32> = stitched.iter().map(|s| s.dof_angles[0]).collect();
        assert_eq!(angles, vec![0.0, 0.1, 0.2, 0.3]);
    }

    #[test]
    fn resolve_n_workers_passes_small_positive_values_through_unchanged() {
        assert_eq!(resolve_n_workers(1), 1);
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if available > 1 {
            // Not clipped, since it's within the available count.
            assert_eq!(resolve_n_workers((available - 1) as isize), available - 1);
        }
    }

    #[test]
    fn resolve_n_workers_clips_positive_values_past_available_cores() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(resolve_n_workers(10_000), available);
    }

    #[test]
    fn resolve_n_workers_follows_joblib_convention_for_negative_values() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(resolve_n_workers(-1), available);
        assert_eq!(resolve_n_workers(-2), (available - 1).max(1));
        // Never resolves below 1, however far negative.
        assert_eq!(resolve_n_workers(-(available as isize) - 100), 1);
    }

    #[test]
    #[should_panic(expected = "n_workers must not be 0")]
    fn resolve_n_workers_rejects_zero() {
        resolve_n_workers(0);
    }

    #[test]
    fn for_recording_uses_all_workers_and_the_default_overlap() {
        let config = ParallelSolveConfig::for_recording(1000);
        assert_eq!(config.n_workers, -1);
        assert_eq!(config.overlap_len, DEFAULT_OVERLAP_LEN);
        assert_eq!(config.overlap_tolerance, DEFAULT_OVERLAP_TOLERANCE);
        assert!(config.segment_len > config.overlap_len);
    }

    #[test]
    fn for_recording_never_produces_an_invalid_segment_len() {
        for total_len in [0, 1, 2] {
            let config = ParallelSolveConfig::for_recording(total_len);
            assert!(config.segment_len > config.overlap_len);
            // Doesn't panic building segment bounds from it either.
            segment_bounds(total_len, config);
        }
    }
}
