//! [`SequentialSolver`]: warm-started solving of one continuous stream of
//! frames, one at a time (real-time use) or a whole recording at once,
//! optionally split into chunks solved in parallel.

use std::sync::Arc;

use crate::body_plan::KinematicTree;
use crate::error::{Result, bail};
use crate::observation::{KeypointObservation, ObservationSequence};
use crate::solver::{SolverConfig, SolverCore, SolverResult};
use crate::state::State;

/// Warm-started solving of one continuous stream.
///
/// Holds a single running state that [`step`](Self::step) and
/// [`solve`](Self::solve) advance: each frame starts from the previous frame's
/// solution, which is also the smoothness priors' target. Construct one per
/// independent stream (e.g. per tracked subject).
pub struct SequentialSolver {
    core: SolverCore,
    state: State,
    /// The previous frame's solution, i.e. the smoothness priors' target.
    /// `None` before the first frame of a stream.
    prev_state: Option<State>,
}

impl SequentialSolver {
    /// Starts a stream at `initial_state`, or the neutral pose if `None`.
    /// Returns an error if `config` has root priors and the tree is fixed-base,
    /// or if `initial_state` doesn't match the tree.
    pub fn new(
        kinematic_tree: &Arc<KinematicTree>,
        config: &SolverConfig,
        initial_state: Option<State>,
    ) -> Result<Self> {
        let core = SolverCore::new(kinematic_tree, config)?;
        let state = match initial_state {
            Some(state) => {
                core.check_state(&state)?;
                state
            }
            None => State::neutral_pose(Arc::clone(kinematic_tree)),
        };
        Ok(Self {
            state,
            core,
            prev_state: None,
        })
    }

    pub fn kinematic_tree(&self) -> &Arc<KinematicTree> {
        self.core.tree()
    }

    pub fn config(&self) -> &SolverConfig {
        self.core.config()
    }

    /// The running state after the last `step`/`solve` call.
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Starts a new stream: sets the running state (neutral pose if `None`) and
    /// forgets the previous frame, so smoothness priors are inactive on the
    /// next frame.
    pub fn reset(&mut self, state: Option<State>) -> Result<()> {
        self.state = match state {
            Some(state) => {
                self.core.check_state(&state)?;
                state
            }
            None => State::neutral_pose(Arc::clone(self.core.tree())),
        };
        self.prev_state = None;
        Ok(())
    }

    /// Solves the next frame of the stream.
    pub fn step(
        &mut self,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> Result<SolverResult> {
        self.core.check_observations(observations)?;
        Ok(self.solve_next(observations, with_grad, with_fk))
    }

    /// [`step`](Self::step) for each frame of `sequence` in order.
    pub fn solve(
        &mut self,
        sequence: &ObservationSequence,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<Vec<SolverResult>> {
        self.core.check_sequence(sequence)?;
        let frames = sequence.frames().iter();
        Ok(frames
            .map(|frame| self.solve_next(frame, with_grad, with_fk))
            .collect())
    }

    /// Offline: splits `sequence` into up to `n_workers` contiguous chunks of
    /// near-equal length, solved in parallel. Never reads or writes the running
    /// state.
    ///
    /// Each chunk starts from the neutral pose. Every chunk but the first
    /// starts `n_overlap` frames early (or at the sequence start if that's
    /// closer) and discards those warm-up results, so its first real frame is
    /// warm-started and has a previous frame for the smoothness priors.
    /// `n_overlap = 0` makes chunks fully independent.
    ///
    /// `n_workers`: a positive value is used directly, unless it exceeds the
    /// number of available cores, in which case it's clipped to that and a
    /// warning is logged. A negative value counts back from all available
    /// cores: `-1` uses all, `-2` all but one, etc. `0` is an error.
    pub fn solve_parallel(
        &self,
        sequence: &ObservationSequence,
        n_workers: isize,
        n_overlap: usize,
        with_grad: bool,
        with_fk: bool,
    ) -> Result<Vec<SolverResult>> {
        self.core.check_sequence(sequence)?;
        let n_workers = resolve_n_workers(n_workers)?;
        let bounds = even_chunk_bounds(sequence.len(), n_workers);

        let chunk_results: Vec<Vec<SolverResult>> = std::thread::scope(|scope| {
            let handles: Vec<_> = bounds
                .iter()
                .map(|&(start, end)| {
                    let mut core = self.core.clone();
                    scope.spawn(move || {
                        let warmup_start = start.saturating_sub(n_overlap);
                        let tree = Arc::clone(core.tree());
                        let mut state = State::neutral_pose(Arc::clone(&tree));
                        let mut prev_state: Option<State> = None;
                        let mut results = Vec::with_capacity(end - start);
                        for i in warmup_start..end {
                            let is_warmup = i < start;
                            let result = core.solve_frame(
                                &mut state,
                                prev_state.as_ref(),
                                &sequence.frames()[i],
                                with_grad && !is_warmup,
                                with_fk && !is_warmup,
                            );
                            if !is_warmup {
                                results.push(result);
                            }
                            prev_state = Some(state.clone());
                        }
                        results
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        Ok(chunk_results.into_iter().flatten().collect())
    }

    /// Solves the next frame from the running state, then records the result
    /// as the previous frame for the one after.
    fn solve_next(
        &mut self,
        observations: &[KeypointObservation],
        with_grad: bool,
        with_fk: bool,
    ) -> SolverResult {
        let result = self.core.solve_frame(
            &mut self.state,
            self.prev_state.as_ref(),
            observations,
            with_grad,
            with_fk,
        );
        self.prev_state = Some(self.state.clone());
        result
    }
}

/// Resolves an `n_workers`-style value into a thread count; see
/// [`SequentialSolver::solve_parallel`] for the convention.
pub(crate) fn resolve_n_workers(n_workers: isize) -> Result<usize> {
    if n_workers == 0 {
        bail!("n_workers must not be 0");
    }
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    if n_workers > 0 {
        let n_workers = n_workers as usize;
        if n_workers > available {
            log::warn!(
                "n_workers ({n_workers}) exceeds available cores ({available}); clipping to \
                 {available}"
            );
            return Ok(available);
        }
        return Ok(n_workers);
    }
    Ok((available as isize + 1 + n_workers).max(1) as usize)
}

/// Splits `total_len` frames into `n_chunks` contiguous `(start, end)`
/// (end-exclusive) bounds, as evenly sized as possible (the first
/// `total_len % n_chunks` get one extra frame). Uses fewer chunks if
/// `total_len < n_chunks`, so none is empty.
fn even_chunk_bounds(total_len: usize, n_chunks: usize) -> Vec<(usize, usize)> {
    if total_len == 0 {
        return Vec::new();
    }
    let n_chunks = n_chunks.min(total_len);
    let base_len = total_len / n_chunks;
    let remainder = total_len % n_chunks;
    let mut bounds = Vec::with_capacity(n_chunks);
    let mut start = 0;
    for i in 0..n_chunks {
        let end = start + base_len + usize::from(i < remainder);
        bounds.push((start, end));
        start = end;
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_chunk_bounds_covers_sequence_contiguously() {
        let bounds = even_chunk_bounds(25, 4);
        assert_eq!(bounds, vec![(0, 7), (7, 13), (13, 19), (19, 25)]);
        assert_eq!(even_chunk_bounds(2, 4), vec![(0, 1), (1, 2)]);
        assert!(even_chunk_bounds(0, 4).is_empty());
    }

    #[test]
    fn resolve_n_workers_follows_joblib_convention() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        assert_eq!(resolve_n_workers(1).unwrap(), 1);
        assert_eq!(resolve_n_workers(10_000).unwrap(), available);
        assert_eq!(resolve_n_workers(-1).unwrap(), available);
        assert_eq!(resolve_n_workers(-(available as isize) - 100).unwrap(), 1);
        assert!(resolve_n_workers(0).is_err());
    }
}
