#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use quickik_core::sequential_solver;

    use crate::body_plan::ffi::KinematicTree;
    use crate::error::ffi::Error;
    use crate::observation::ffi::{KeypointObservationList, ObservationSequence};
    use crate::solver::ffi::{SolverConfig, SolverResult, SolverResultList};
    use crate::solver::result_list;
    use crate::state::ffi::State;

    /// Warm-started solving of one continuous stream.
    ///
    /// Holds a single running state that `step` and `solve` advance: each
    /// frame starts from the previous frame's solution, which is also the
    /// smoothness priors' target. Create one per independent stream (e.g. per
    /// tracked subject).
    #[diplomat::opaque_mut]
    pub struct SequentialSolver(pub(crate) sequential_solver::SequentialSolver);

    impl SequentialSolver {
        /// Starts a stream at a copy of `initial_state`, or the neutral pose if
        /// null. Returns an error if `config` has root priors and the tree is
        /// fixed-base, or if `initial_state` doesn't match the tree.
        pub fn create(
            tree: &KinematicTree,
            config: &SolverConfig,
            initial_state: Option<&State>,
        ) -> Result<Box<SequentialSolver>, Box<Error>> {
            let initial_state = initial_state.map(|state| state.0.clone());
            let solver =
                sequential_solver::SequentialSolver::new(&tree.0, &config.0, initial_state)?;
            Ok(Box::new(SequentialSolver(solver)))
        }

        /// A copy of the running state after the last `step`/`solve` call.
        pub fn state(&self) -> Box<State> {
            Box::new(State(self.0.state().clone()))
        }

        /// Starts a new stream: sets the running state to a copy of `state`
        /// (neutral pose if null) and forgets the previous frame, so smoothness
        /// priors are inactive on the next frame.
        pub fn reset(&mut self, state: Option<&State>) -> Result<(), Box<Error>> {
            Ok(self.0.reset(state.map(|state| state.0.clone()))?)
        }

        /// Solves the next frame of the stream.
        pub fn step(
            &mut self,
            observations: &KeypointObservationList,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResult>, Box<Error>> {
            let result = self.0.step(&observations.0, with_grad, with_fk)?;
            Ok(Box::new(result.into()))
        }

        /// `step` for each frame of `sequence` in order.
        pub fn solve(
            &mut self,
            sequence: &ObservationSequence,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResultList>, Box<Error>> {
            Ok(result_list(self.0.solve(
                &sequence.0,
                with_grad,
                with_fk,
            )?))
        }

        /// Offline: splits `sequence` into up to `n_workers` contiguous chunks
        /// of near-equal length, solved in parallel. Never reads or writes the
        /// running state.
        ///
        /// Each chunk starts from the neutral pose. Every chunk but the first
        /// starts `n_overlap` frames early (or at the sequence start if that's
        /// closer) and discards those warm-up results, so its first real frame
        /// is warm-started and has a previous frame for the smoothness priors.
        /// `n_overlap = 0` makes chunks fully independent.
        ///
        /// `n_workers`: a positive value is used directly, unless it exceeds
        /// the number of available cores, in which case it's clipped to that.
        /// A negative value counts back from all available cores: -1 uses all,
        /// -2 all but one, etc. 0 is an error.
        pub fn solve_parallel(
            &self,
            sequence: &ObservationSequence,
            n_workers: isize,
            n_overlap: usize,
            with_grad: bool,
            with_fk: bool,
        ) -> Result<Box<SolverResultList>, Box<Error>> {
            let results =
                self.0
                    .solve_parallel(&sequence.0, n_workers, n_overlap, with_grad, with_fk)?;
            Ok(result_list(results))
        }
    }
}
