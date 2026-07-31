//! High-level APIs for a single tracked body over consecutive frames
//! ([`SequenceSolver`]), for one long sequence solved in parallel via
//! overlapping segments ([`solve_sequence_segmented_parallel`]), and for a
//! batch of independent frames solved in parallel with their solve-time
//! linearization retained for gradient tracking
//! ([`solve_batch_with_grad`]). For single-frame or other specialized use
//! cases, use [`Solver`] directly.
//!
//! [`Solver`]: crate::solver::Solver

use std::sync::Arc;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, UnitQuaternion, Vector3};
use rayon::prelude::*;

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

    /// World-space keypoint positions at the most recently converged pose
    /// (see [`Solver::last_fk_positions`]), indexed by joint/keypoint index
    /// in `KinematicTree` joint order.
    pub fn last_fk_positions(&self) -> &[Vector3<f32>] {
        self.solver.last_fk_positions()
    }

    /// Same as [`solve_sequence`](Self::solve_sequence), but additionally
    /// returns each frame's forward-kinematics keypoint positions (world
    /// space, in `KinematicTree` joint order) alongside its converged state.
    pub fn solve_sequence_with_fk(
        &mut self,
        sequence: &[Vec<KeypointObservation>],
    ) -> Vec<(State, Vec<Vector3<f32>>)> {
        sequence
            .iter()
            .map(|observations| {
                let state = self.solve_frame(observations).clone();
                let fk = self.solver.last_fk_positions().to_vec();
                (state, fk)
            })
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
/// converge to slightly different poses; any disagreement in `dof_angles` or
/// root rotation beyond `parallel_config.overlap_tolerance` generates a
/// warning and the earlier segment's version is kept. Root *position*
/// disagreement isn't checked (there's no comparable position tolerance in
/// [`ParallelSolveConfig`]).
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
/// segment are used). Logs a warning if any overlapping frame's `dof_angles`
/// or root rotation disagree between segments by more than
/// `overlap_tolerance` (radians for both, so they're compared on the same
/// scale). Root *position* disagreement isn't covered by this check: there's
/// no comparable position tolerance in [`ParallelSolveConfig`].
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
            let earlier = &result[overlap_start + i];
            let later = &segment[i];
            let max_dof_angle_diff = earlier
                .dof_angles
                .iter()
                .zip(&later.dof_angles)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let root_rot_diff = earlier.root_rot.angle_to(&later.root_rot);
            let max_angle_diff = max_dof_angle_diff.max(root_rot_diff);
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

/// Every [`solve_batch_with_grad`] item's converged pose and linearization,
/// as a struct of batched arrays (rather than a `Vec` of per-item structs) to
/// match how a batch is naturally represented on the PyTorch side.
pub struct BatchedResultWithGrad {
    /// `(batch_size)`, each of length `n_dofs`. DOF order matches the
    /// `KinematicTree`'s own (`State::dof_angles`'s order): unlike keypoints
    /// (whose observations typically come from an external, already-fixed-order
    /// source, e.g. a pretrained detector, that has no reason to match the
    /// tree's own joint order), DOF order is already fully caller-controlled
    /// -- it's exactly the order joints and their DOFs were listed in when
    /// the [`KinematicTree`] was built -- so there's no equivalent
    /// DOF-ordering parameter to reorder this by.
    pub joint_angles: Vec<Vec<f32>>,
    /// `(batch_size)` free-floating root positions.
    pub base_pos: Vec<Vector3<f32>>,
    /// `(batch_size)` free-floating root orientations.
    pub base_quat: Vec<UnitQuaternion<f32>>,
    /// `(batch_size)`, each item's keypoint-position Jacobian (see
    /// [`Solver::last_jacobian`]). Rows/columns are in the `KinematicTree`'s
    /// internal keypoint/state order, *not* `keypoints_order`: nothing
    /// outside this crate reads these entries directly, so only
    /// `observations_array` (in) and `joint_angles` (out) need reordering;
    /// see [`solve_batch_with_grad`]'s docs.
    pub jacobian: Vec<DMatrix<f32>>,
    /// `(batch_size)`, each item's Cholesky factor L (see
    /// [`Solver::last_cholesky`]). `None` if that item's last iteration
    /// wasn't positive-definite (gradients can't be computed for it).
    pub cholesky_l: Vec<Option<Cholesky<f32, nalgebra::Dyn>>>,
}

/// Resolves `keypoints_order` (external keypoint axis order, given by joint
/// name) into `keypoint_to_joint_idx[i]` = the internal joint/keypoint index
/// that `observations_array`'s keypoint axis position `i` corresponds to.
/// Panics unless `keypoints_order` names every joint in `kinematic_tree`
/// exactly once.
fn resolve_keypoint_order(
    kinematic_tree: &KinematicTree,
    keypoints_order: &[String],
) -> Vec<usize> {
    let n_joints = kinematic_tree.n_joints();
    assert_eq!(
        keypoints_order.len(),
        n_joints,
        "keypoints_order.len() must equal kinematic_tree.n_joints()"
    );

    let name_to_idx: std::collections::HashMap<&str, usize> = kinematic_tree
        .joints
        .iter()
        .enumerate()
        .map(|(idx, joint)| (joint.name.as_str(), idx))
        .collect();
    assert_eq!(
        name_to_idx.len(),
        n_joints,
        "kinematic_tree has duplicate joint names, so keypoints_order can't \
         unambiguously refer to them by name"
    );

    let mut joint_seen = vec![false; n_joints];
    keypoints_order
        .iter()
        .map(|name| {
            let joint_idx = *name_to_idx
                .get(name.as_str())
                .unwrap_or_else(|| panic!("keypoints_order: unknown joint name '{name}'"));
            assert!(
                !joint_seen[joint_idx],
                "keypoints_order: joint '{name}' listed more than once"
            );
            joint_seen[joint_idx] = true;
            joint_idx
        })
        .collect()
}

/// Solves a batch of fully independent sets of keypoint observations in
/// parallel via rayon, each starting from `kinematic_tree`'s neutral pose
/// with its own freshly constructed [`Solver`] (so batch items never contend
/// over solver-internal buffers), and returns every item's converged pose and
/// linearization for later implicit differentiation of the solve. Unlike
/// [`solve_sequence_segmented_parallel`], batch items don't warm-start or
/// stitch against each other; use [`SequenceSolver`] instead if consecutive
/// frames should share a running pose.
///
/// `kinematic_tree` must be free-floating (not
/// [`fixed_base`](KinematicTree::fixed_base)), since [`BatchedResultWithGrad`]
/// always reports `base_pos`/`base_quat`.
///
/// `keypoints_order[i]` is the joint name (matching [`Joint::name`]) that
/// `observations_array`'s keypoint axis position `i` corresponds to; every
/// joint in `kinematic_tree` must appear in it exactly once.
/// `observations_array` is `(batch_size)`, each a `Vec<KeypointObservation>`
/// of length `n_joints` given in that same order (*not* the `KinematicTree`'s
/// internal joint order).
///
/// [`Joint::name`]: crate::body_plan::Joint::name
pub fn solve_batch_with_grad<M: Mapper3Dto2D + Sync>(
    kinematic_tree: &Arc<KinematicTree>,
    solver_config: SolverConfig<M>,
    keypoints_order: &[String],
    observations_array: &[Vec<KeypointObservation>],
) -> BatchedResultWithGrad {
    assert!(
        !kinematic_tree.fixed_base,
        "solve_batch_with_grad requires a free-floating-base tree: base_pos/base_quat \
         have no meaning for a fixed-base tree"
    );
    let n_joints = kinematic_tree.n_joints();
    let keypoint_to_joint_idx = resolve_keypoint_order(kinematic_tree, keypoints_order);

    type PerItemResult = (State, DMatrix<f32>, Option<Cholesky<f32, nalgebra::Dyn>>);
    let per_item_results: Vec<PerItemResult> = observations_array
        .par_iter()
        .map(|external_order_observations| {
            assert_eq!(
                external_order_observations.len(),
                n_joints,
                "every observations_array item must have length kinematic_tree.n_joints()"
            );
            // Remap from the external keypoints_order into the tree's
            // internal joint order: cheap (O(n_joints)) since it's just this
            // one item's small observation list, not the Jacobian/Cholesky
            // matrices below.
            let mut internal_order_observations = vec![KeypointObservation::Missing; n_joints];
            for (external_idx, &joint_idx) in keypoint_to_joint_idx.iter().enumerate() {
                internal_order_observations[joint_idx] = external_order_observations[external_idx];
            }

            let mut solver = Solver::new(kinematic_tree, solver_config);
            let mut state = State::neutral_pose(Arc::clone(kinematic_tree));
            solver.solve_with_grad(&mut state, &internal_order_observations);
            (
                state,
                solver.last_jacobian().clone(),
                solver.last_cholesky(),
            )
        })
        .collect();

    let batch_size = per_item_results.len();
    let mut result = BatchedResultWithGrad {
        joint_angles: Vec::with_capacity(batch_size),
        base_pos: Vec::with_capacity(batch_size),
        base_quat: Vec::with_capacity(batch_size),
        jacobian: Vec::with_capacity(batch_size),
        cholesky_l: Vec::with_capacity(batch_size),
    };
    for (state, jacobian, cholesky_l) in per_item_results {
        result.joint_angles.push(state.dof_angles);
        result.base_pos.push(state.root_pos);
        result.base_quat.push(state.root_rot);
        result.jacobian.push(jacobian);
        result.cholesky_l.push(cholesky_l);
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
