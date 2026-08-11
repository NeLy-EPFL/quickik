use std::sync::Arc;

use numpy::{AllowTypeChange, PyArrayLike2, PyArrayLike3};
use pyo3::prelude::*;

use crate::body_plan::KinematicTree;
use crate::catch_panic;
use crate::observation::{Projection, observations_from_arrays, validate_position_weight_shapes};
use crate::solver::SolverResult;

/// Warm-started solving for a continuous sequence of frames; see
/// `quickik_core::sequential_solver::SequenceSolver`'s docs for exactly what
/// "warm-started" means and how `solve_segments_parallel` relates to it.
///
/// `projection` is fixed for this object's lifetime, mirroring `Solver`'s own
/// `projection` property (no setter). Unlike `Solver`, the other tuning
/// parameters (`n_iterations`, `neutral_weight`, ...) aren't exposed as
/// retunable attributes here.
#[pyclass(module = "quickik")]
pub(crate) struct SequenceSolver {
    inner: quickik_core::sequential_solver::SequenceSolver,
    kinematic_tree: Arc<quickik_core::body_plan::KinematicTree>,
}

#[pymethods]
impl SequenceSolver {
    /// Starts a new continuous sequence at the neutral pose.
    #[new]
    #[pyo3(signature = (
        kinematic_tree, projection=Projection::default(), n_iterations=10, neutral_weight=1e-3,
        position_tolerance=1e-3, angle_tolerance=1e-3, damping=1e-6,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinematic_tree: KinematicTree,
        projection: Projection,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> PyResult<Self> {
        Ok(SequenceSolver {
            inner: quickik_core::sequential_solver::SequenceSolver::new(
                &kinematic_tree.inner,
                projection.inner,
                n_iterations,
                neutral_weight,
                position_tolerance,
                angle_tolerance,
                damping,
            ),
            kinematic_tree: Arc::clone(&kinematic_tree.inner),
        })
    }

    /// Solves every frame in order, each warm-started from wherever this
    /// object's last `solve`/`solve_segments_parallel` call left off (see the
    /// class docstring). Returns one `SolverResult` per frame.
    ///
    /// `weights` is `(n_frames, n_joints)`; `positions` is `(n_frames,
    /// n_joints, 3)` for a 3D projection, or `(n_frames, n_joints, 2)`
    /// otherwise (2D observations in that projection's space). Both
    /// are in `kinematic_tree.joints` order; a keypoint with `weight <= 0`
    /// (or NaN) is treated as missing. Given as raw arrays rather than a list
    /// of per-frame `KeypointObservation` lists so this never constructs one
    /// Python object per keypoint per frame, which otherwise dominates call
    /// overhead for long recordings. Any dtype is accepted and cast to
    /// `float32`, following NumPy's own casting rules.
    #[pyo3(signature = (positions, weights, with_grad=false, with_fk=false))]
    fn solve(
        &mut self,
        positions: PyArrayLike3<'_, f32, AllowTypeChange>,
        weights: PyArrayLike2<'_, f32, AllowTypeChange>,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<Vec<SolverResult>> {
        let positions_arr = positions.as_array();
        let weights_arr = weights.as_array();
        validate_position_weight_shapes(
            &positions_arr,
            &weights_arr,
            self.kinematic_tree.n_joints(),
            !self.inner.projection().is_3d(),
        )?;
        let sequence = observations_from_arrays(positions_arr, weights_arr);
        let inner = &mut self.inner;
        catch_panic(move || {
            inner
                .solve(&sequence, with_grad, with_fk)
                .into_iter()
                .map(SolverResult::new)
                .collect()
        })
    }

    /// Solves `positions`/`weights` in parallel by splitting them into
    /// exactly `n_workers` contiguous, non-overlapping segments, each
    /// cold-started at the neutral pose and then warm-started within itself.
    /// See `quickik_core::sequential_solver::SequenceSolver::solve_segments_parallel`
    /// for the exact `n_workers` convention. Never reads or writes this
    /// object's own `solve` state.
    ///
    /// `positions`/`weights` follow the same convention as `solve`.
    #[pyo3(signature = (positions, weights, n_workers, with_grad=false, with_fk=false))]
    #[allow(clippy::too_many_arguments)]
    fn solve_segments_parallel(
        &self,
        py: Python<'_>,
        positions: PyArrayLike3<'_, f32, AllowTypeChange>,
        weights: PyArrayLike2<'_, f32, AllowTypeChange>,
        n_workers: isize,
        with_grad: bool,
        with_fk: bool,
    ) -> PyResult<Vec<SolverResult>> {
        let positions_arr = positions.as_array();
        let weights_arr = weights.as_array();
        validate_position_weight_shapes(
            &positions_arr,
            &weights_arr,
            self.kinematic_tree.n_joints(),
            !self.inner.projection().is_3d(),
        )?;
        let inner = &self.inner;
        py.detach(|| {
            catch_panic(|| {
                let sequence = observations_from_arrays(positions_arr, weights_arr);
                inner
                    .solve_segments_parallel(&sequence, n_workers, with_grad, with_fk)
                    .into_iter()
                    .map(SolverResult::new)
                    .collect()
            })
        })
    }

    /// Fixed at construction (read-only).
    #[getter]
    fn projection(&self) -> Projection {
        Projection {
            inner: self.inner.projection(),
        }
    }
}
