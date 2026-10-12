//! [`BatchedSolver`]: solving a batch of fully independent items in parallel,
//! for training/inference with an autodiff framework.

use std::sync::Arc;

use nalgebra::linalg::Cholesky;
use nalgebra::{DMatrix, Dyn, UnitQuaternion, Vector3};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuilder};

use crate::body_plan::KinematicTree;
use crate::error::{Error, Result, bail};
use crate::observation::ObservationSequence;
use crate::sequential_solver::resolve_n_workers;
use crate::solver::{SolverConfig, SolverCore, SolverResult};
use crate::state::State;

/// Every [`BatchedSolver::solve`] item's result, as a struct of per-item
/// arrays (rather than a `Vec` of per-item structs) to match how a batch is
/// represented on the PyTorch side.
#[derive(Debug)]
pub struct BatchedSolverResult {
    /// Per item, one value per DOF, in the tree's DOF order.
    pub dof_values: Vec<Vec<f32>>,
    pub root_pos: Vec<Vector3<f32>>,
    pub root_quat: Vec<UnitQuaternion<f32>>,
    /// Per item, world-frame keypoint positions in joint order. `Some` if and
    /// only if `solve` had `with_fk: true`.
    pub keypoint_pos: Option<Vec<Vec<Vector3<f32>>>>,
    /// Per item, the residual Jacobian (see
    /// [`SolverResult::jacobian`]). `Some` if and only if `solve` had
    /// `with_grad: true`.
    pub jacobian: Option<Vec<DMatrix<f32>>>,
    /// Per item, the Cholesky factor, `None` for items whose last iteration
    /// wasn't positive-definite (see [`SolverResult::cholesky_l`]). `Some` if
    /// and only if `solve` had `with_grad: true`.
    pub cholesky_l: Option<Vec<Option<Cholesky<f32, Dyn>>>>,
}

/// Solves a batch of fully independent items in parallel, each starting from
/// the neutral pose (no warm start, since batch composition typically changes
/// every call, e.g. every minibatch).
pub struct BatchedSolver {
    core: SolverCore,
    /// Dedicated pool, so `solve` neither contends with nor gets starved by
    /// rayon's global pool or other `BatchedSolver`s.
    thread_pool: ThreadPool,
}

impl BatchedSolver {
    /// Returns an error if the tree is fixed-base (results always report the
    /// root pose), if `config` has smoothness priors (items are independent),
    /// or if `n_workers` is invalid. `n_workers` follows the convention of
    /// [`SequentialSolver::solve_parallel`](crate::sequential_solver::SequentialSolver::solve_parallel).
    pub fn new(
        kinematic_tree: &Arc<KinematicTree>,
        config: &SolverConfig,
        n_workers: isize,
    ) -> Result<Self> {
        if kinematic_tree.fixed_base() {
            bail!("BatchedSolver requires a free-floating tree");
        }
        if config.priors.has_temporal() {
            bail!("BatchedSolver doesn't support smoothness priors (items are independent)");
        }
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(resolve_n_workers(n_workers)?)
            .build()
            .map_err(|e| Error::InvalidInput(format!("failed to build thread pool: {e}")))?;
        Ok(Self {
            core: SolverCore::new(kinematic_tree, config)?,
            thread_pool,
        })
    }

    pub fn kinematic_tree(&self) -> &Arc<KinematicTree> {
        self.core.tree()
    }

    pub fn config(&self) -> &SolverConfig {
        self.core.config()
    }

    /// Solves every frame of `batch` as an independent item.
    pub fn solve(
        &self,
        batch: &ObservationSequence,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<BatchedSolverResult> {
        self.core.check_sequence(batch)?;
        let neutral = State::neutral_pose(Arc::clone(self.core.tree()));
        let results: Vec<SolverResult> = self.thread_pool.install(|| {
            (0..batch.len())
                .into_par_iter()
                // One core (and its buffers) per rayon work split rather than
                // per item.
                .map_init(
                    || self.core.clone(),
                    |core, i| {
                        let mut state = neutral.clone();
                        core.solve_frame(&mut state, None, &batch.frames()[i], with_grad, with_fk)
                    },
                )
                .collect()
        });

        let n = results.len();
        let mut output = BatchedSolverResult {
            dof_values: Vec::with_capacity(n),
            root_pos: Vec::with_capacity(n),
            root_quat: Vec::with_capacity(n),
            keypoint_pos: with_fk.then(|| Vec::with_capacity(n)),
            jacobian: with_grad.then(|| Vec::with_capacity(n)),
            cholesky_l: with_grad.then(|| Vec::with_capacity(n)),
        };
        for result in results {
            output.dof_values.push(result.state.dof_values);
            output.root_pos.push(result.state.root_pos);
            output.root_quat.push(result.state.root_quat);
            if let (Some(all), Some(item)) = (&mut output.keypoint_pos, result.keypoint_pos) {
                all.push(item);
            }
            if let (Some(all), Some(item)) = (&mut output.jacobian, result.jacobian) {
                all.push(item);
            }
            if let Some(all) = &mut output.cholesky_l {
                all.push(result.cholesky_l);
            }
        }
        Ok(output)
    }
}
