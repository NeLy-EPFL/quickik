//! [`BatchedSolver`]: solving a batch of fully independent (never
//! warm-started) sets of keypoint observations in parallel, for training/
//! inference with an autodiff framework. See [`BatchedSolver::solve`].

use std::collections::HashMap;
use std::sync::Arc;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, Dyn, UnitQuaternion, Vector3};
use rayon::prelude::*;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, Mapper3Dto2D, NoMapper};
use crate::solver::Solver;
use crate::state::State;

/// Every [`BatchedSolver::solve`] item's converged pose and linearization, as
/// a struct of batched arrays (rather than a `Vec` of per-item structs) to
/// match how a batch is naturally represented on the PyTorch side.
pub struct BatchedSolverResult {
    /// `(batch_size)`, each of length `n_dofs`. DOF order matches the
    /// `KinematicTree`'s own (`State::dof_angles`'s order): unlike keypoints
    /// (whose observations typically come from an external, already-fixed-order
    /// source, e.g. a pretrained detector, that has no reason to match the
    /// tree's own joint order), DOF order is already fully caller-controlled:
    /// it's exactly the order joints and their DOFs were listed in when
    /// the [`KinematicTree`] was built, so there's no equivalent
    /// DOF-ordering parameter to reorder this by.
    pub joint_angles: Vec<Vec<f32>>,
    /// `(batch_size)` free-floating root positions.
    pub base_pos: Vec<Vector3<f32>>,
    /// `(batch_size)` free-floating root orientations.
    pub base_quat: Vec<UnitQuaternion<f32>>,
    /// `(batch_size)`, each item's world-space keypoint positions, in
    /// `KinematicTree`'s internal joint order (*not* `keypoints_order`).
    /// `Some` iff `solve` was called with `with_fk: true`.
    pub keypoint_pos: Option<Vec<Vec<Vector3<f32>>>>,
    /// `(batch_size)`, each item's keypoint-position Jacobian. Rows/columns
    /// are in `KinematicTree`'s internal keypoint/state order, *not*
    /// `keypoints_order`: nothing outside this crate reads these entries
    /// directly, so only the input observations and `joint_angles` need
    /// reordering by `keypoints_order`. `Some` iff `solve` was called with
    /// `with_grad: true`.
    pub jacobian: Option<Vec<DMatrix<f32>>>,
    /// `(batch_size)`, each item's Cholesky factor L. The inner `Option` is
    /// `None` if that item's last iteration wasn't positive-definite
    /// (gradients can't be computed for it). The outer `Option` is `Some`
    /// iff `solve` was called with `with_grad: true`.
    pub cholesky_l: Option<Vec<Option<Cholesky<f32, Dyn>>>>,
}

/// Solves a batch of fully independent sets of keypoint observations, for
/// training/inference with an autodiff framework (e.g. the Python bindings'
/// PyTorch integration).
///
/// Every [`solve`](Self::solve) call is completely independent: each item
/// always starts from `kinematic_tree`'s neutral pose (no warm-starting,
/// unlike [`SequenceSolver`](crate::sequential_solver::SequenceSolver)),
/// since batch composition typically changes every call (e.g. every
/// minibatch in a training loop). Constructing one `BatchedSolver` and
/// calling `solve` on it repeatedly (rather than reconstructing one per
/// call) still pays off: `keypoints_order` is resolved into internal joint
/// indices once, at construction, not re-resolved every call.
pub struct BatchedSolver<M: Mapper3Dto2D = NoMapper> {
    kinematic_tree: Arc<KinematicTree>,
    mapper: M,
    n_iterations: usize,
    neutral_weight: f32,
    position_tolerance: f32,
    angle_tolerance: f32,
    damping: f32,
    /// `keypoint_to_joint_idx[i]` is the internal joint/keypoint index that
    /// `solve`'s `observations_array` keypoint axis position `i` corresponds
    /// to, resolved once here from the external, by-name `keypoints_order`.
    keypoint_to_joint_idx: Vec<usize>,
}

impl<M: Mapper3Dto2D + Sync> BatchedSolver<M> {
    /// `kinematic_tree` must be free-floating (not
    /// [`fixed_base`](KinematicTree::fixed_base)), since
    /// [`BatchedSolverResult`] always reports `base_pos`/`base_quat`.
    ///
    /// `keypoints_order[i]` is the joint name (matching [`Joint::name`]) that
    /// `solve`'s `observations_array` keypoint axis position `i` corresponds
    /// to; every joint in `kinematic_tree` must appear in it exactly once.
    ///
    /// [`Joint::name`]: crate::body_plan::Joint::name
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kinematic_tree: &Arc<KinematicTree>,
        mapper: M,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
        keypoints_order: Vec<String>,
    ) -> Self {
        assert!(
            !kinematic_tree.fixed_base,
            "BatchedSolver requires a free-floating-base tree: base_pos/base_quat have no \
             meaning for a fixed-base tree"
        );
        let keypoint_to_joint_idx = resolve_keypoint_order(kinematic_tree, &keypoints_order);
        Self {
            kinematic_tree: Arc::clone(kinematic_tree),
            mapper,
            n_iterations,
            neutral_weight,
            position_tolerance,
            angle_tolerance,
            damping,
            keypoint_to_joint_idx,
        }
    }

    /// `keypoint_to_joint_idx()[i]` is the internal joint/keypoint index that
    /// `solve`'s `observations_array` keypoint axis position `i` corresponds
    /// to (the resolved inverse of the by-name `keypoints_order` this solver
    /// was constructed with).
    pub fn keypoint_to_joint_idx(&self) -> &[usize] {
        &self.keypoint_to_joint_idx
    }

    /// Solves every item in `observations_array` in parallel via rayon, each
    /// starting from `kinematic_tree`'s neutral pose with its own freshly
    /// constructed [`Solver`] (so items never contend over solver-internal
    /// buffers). `observations_array` is `(batch_size)`, each a
    /// `Vec<KeypointObservation>` of length `n_joints`, in the order given by
    /// this `BatchedSolver`'s `keypoints_order` (*not* the `KinematicTree`'s
    /// internal joint order).
    pub fn solve(
        &self,
        observations_array: &[Vec<KeypointObservation>],
        with_grad: bool,
        with_fk: bool,
    ) -> BatchedSolverResult {
        let n_joints = self.kinematic_tree.n_joints();

        let per_item_results: Vec<crate::solver::SolverResult> = observations_array
            .par_iter()
            .map(|external_order_observations| {
                assert_eq!(
                    external_order_observations.len(),
                    n_joints,
                    "every observations_array item must have length kinematic_tree.n_joints()"
                );
                // Remap from the external keypoints_order into the tree's
                // internal joint order: cheap (O(n_joints)) since it's just
                // this one item's small observation list, not the
                // Jacobian/Cholesky matrices below.
                let mut internal_order_observations = vec![KeypointObservation::Missing; n_joints];
                for (external_idx, &joint_idx) in self.keypoint_to_joint_idx.iter().enumerate() {
                    internal_order_observations[joint_idx] =
                        external_order_observations[external_idx];
                }

                let mut solver = Solver::new(
                    &self.kinematic_tree,
                    self.mapper,
                    self.n_iterations,
                    self.neutral_weight,
                    self.position_tolerance,
                    self.angle_tolerance,
                    self.damping,
                );
                let mut state = State::neutral_pose(Arc::clone(&self.kinematic_tree));
                solver.solve(&mut state, &internal_order_observations, with_grad, with_fk)
            })
            .collect();

        let batch_size = per_item_results.len();
        let mut joint_angles = Vec::with_capacity(batch_size);
        let mut base_pos = Vec::with_capacity(batch_size);
        let mut base_quat = Vec::with_capacity(batch_size);
        let mut keypoint_pos = with_fk.then(|| Vec::with_capacity(batch_size));
        let mut jacobian = with_grad.then(|| Vec::with_capacity(batch_size));
        let mut cholesky_l = with_grad.then(|| Vec::with_capacity(batch_size));
        for result in per_item_results {
            joint_angles.push(result.state.dof_angles);
            base_pos.push(result.state.root_pos);
            base_quat.push(result.state.root_rot);
            if let Some(keypoint_pos) = &mut keypoint_pos {
                keypoint_pos.push(result.keypoint_pos.unwrap());
            }
            if let Some(jacobian) = &mut jacobian {
                jacobian.push(result.jacobian.unwrap());
            }
            if let Some(cholesky_l) = &mut cholesky_l {
                cholesky_l.push(result.cholesky_l);
            }
        }
        BatchedSolverResult {
            joint_angles,
            base_pos,
            base_quat,
            keypoint_pos,
            jacobian,
            cholesky_l,
        }
    }
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

    let name_to_idx: HashMap<&str, usize> = kinematic_tree
        .joints
        .iter()
        .enumerate()
        .map(|(idx, joint)| (joint.name.as_str(), idx))
        .collect();
    assert_eq!(
        name_to_idx.len(),
        n_joints,
        "kinematic_tree has duplicate joint names, so keypoints_order can't unambiguously refer \
         to them by name"
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
