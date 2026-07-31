//! [`SequenceSolver`]: warm-started solving for a continuous sequence of
//! frames, either one at a time (or in chunks) as they arrive, or all at
//! once via [`SequenceSolver::solve_segments_parallel`] for a whole
//! already-available recording.

use std::sync::Arc;

use crate::body_plan::KinematicTree;
use crate::observation::{KeypointObservation, Mapper3Dto2D, NoMapper};
use crate::solver::{Solver, SolverResult};
use crate::state::State;

/// Warm-started solving for a continuous sequence of frames.
///
/// [`solve`](Self::solve) always continues from wherever the previous call
/// (if any) left off, for this object's whole lifetime; construct one per
/// independent continuous stream (e.g. one per tracked subject, or one per
/// recording), not one for your whole program. Frames can be fed in one at a
/// time as they arrive, or as a whole pre-recorded sequence in one call: both
/// are the same thing to `solve`, just a different slice length.
///
/// [`solve_segments_parallel`](Self::solve_segments_parallel) is unrelated to
/// this continuity: it's a self-contained bulk operation over whatever
/// sequence you pass it, split across worker threads, and never reads or
/// writes the object's own running state.
pub struct SequenceSolver<M: Mapper3Dto2D = NoMapper> {
    solver: Solver<M>,
    state: State,
    kinematic_tree: Arc<KinematicTree>,
}

impl<M: Mapper3Dto2D + Sync + Send> SequenceSolver<M> {
    /// Starts a new continuous sequence at the neutral pose.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kinematic_tree: &Arc<KinematicTree>,
        mapper: M,
        n_iterations: usize,
        neutral_weight: f32,
        position_tolerance: f32,
        angle_tolerance: f32,
        damping: f32,
    ) -> Self {
        let solver = Solver::new(
            kinematic_tree,
            mapper,
            n_iterations,
            neutral_weight,
            position_tolerance,
            angle_tolerance,
            damping,
        );
        let state = State::neutral_pose(Arc::clone(kinematic_tree));
        Self {
            solver,
            state,
            kinematic_tree: Arc::clone(kinematic_tree),
        }
    }

    /// Solves every frame in `sequence` in order, continuing to warm-start
    /// from wherever this object's last `solve` call (on any previous
    /// `sequence`) left off; see the struct docs. Returns one
    /// [`SolverResult`] per frame.
    pub fn solve(
        &mut self,
        sequence: &[Vec<KeypointObservation>],
        with_grad: bool,
        with_fk: bool,
    ) -> Vec<SolverResult> {
        sequence
            .iter()
            .map(|observations| {
                self.solver
                    .solve(&mut self.state, observations, with_grad, with_fk)
            })
            .collect()
    }

    /// Solves `sequence` in parallel by splitting it into exactly
    /// `n_workers` contiguous, non-overlapping segments (one per worker,
    /// as evenly sized as possible): each segment cold-starts at the neutral
    /// pose on its own thread, then warm-starts frame-to-frame within itself,
    /// same as [`solve`](Self::solve). Segments don't overlap and aren't
    /// cross-checked against each other: better load distribution is worth
    /// more than that consistency check, since segments are independent
    /// either way. This never reads or writes this object's own running
    /// `solve` state (see the struct docs).
    ///
    /// `n_workers`: a positive value is used directly, unless it exceeds the
    /// number of available cores: it's then clipped to that count and a
    /// warning is logged. A negative value counts backward from all
    /// available cores: `-1` uses all, `-2` uses all but one, etc. `0` is
    /// invalid.
    pub fn solve_segments_parallel(
        &self,
        sequence: &[Vec<KeypointObservation>],
        n_workers: isize,
        with_grad: bool,
        with_fk: bool,
    ) -> Vec<SolverResult> {
        let n_workers = resolve_n_workers(n_workers);
        let bounds = even_segment_bounds(sequence.len(), n_workers);
        let mapper = self.solver.mapper();
        let n_iterations = self.solver.n_iterations;
        let neutral_weight = self.solver.neutral_weight;
        let position_tolerance = self.solver.position_tolerance;
        let angle_tolerance = self.solver.angle_tolerance;
        let damping = self.solver.damping;

        std::thread::scope(|scope| {
            bounds
                .iter()
                .map(|&(start, end)| {
                    scope.spawn(move || {
                        let mut solver = Solver::new(
                            &self.kinematic_tree,
                            mapper,
                            n_iterations,
                            neutral_weight,
                            position_tolerance,
                            angle_tolerance,
                            damping,
                        );
                        let mut state = State::neutral_pose(Arc::clone(&self.kinematic_tree));
                        sequence[start..end]
                            .iter()
                            .map(|observations| {
                                solver.solve(&mut state, observations, with_grad, with_fk)
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        })
    }
}

/// Resolves a `solve_segments_parallel`-style `n_workers` value into an
/// actual thread count; see [`SequenceSolver::solve_segments_parallel`]'s
/// docs for the exact convention.
fn resolve_n_workers(n_workers: isize) -> usize {
    assert!(n_workers != 0, "n_workers must not be 0");
    let available = std::thread::available_parallelism().map_or(1, |n| n.get());
    if n_workers > 0 {
        let n_workers = n_workers as usize;
        if n_workers > available {
            log::warn!(
                "n_workers ({n_workers}) exceeds available cores ({available}); clipping to \
                 {available}"
            );
            return available;
        }
        return n_workers;
    }
    (available as isize + 1 + n_workers).max(1) as usize
}

/// Splits `total_len` frames into exactly `n_workers` contiguous
/// `(start, end)` (end-exclusive) bounds, as evenly sized as possible (the
/// first `total_len % n_workers` segments get one extra frame). Uses fewer
/// than `n_workers` segments if `total_len < n_workers`, so every segment
/// still gets at least one frame. Empty if `total_len == 0`.
fn even_segment_bounds(total_len: usize, n_workers: usize) -> Vec<(usize, usize)> {
    if total_len == 0 {
        return Vec::new();
    }
    let n_segments = n_workers.min(total_len);
    let base_len = total_len / n_segments;
    let remainder = total_len % n_segments;

    let mut bounds = Vec::with_capacity(n_segments);
    let mut start = 0;
    for i in 0..n_segments {
        let len = base_len + usize::from(i < remainder);
        let end = start + len;
        bounds.push((start, end));
        start = end;
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_segment_bounds_covers_whole_sequence_in_exactly_n_workers_segments() {
        let bounds = even_segment_bounds(25, 4);
        assert_eq!(bounds.len(), 4);
        assert_eq!(bounds.first().unwrap().0, 0);
        assert_eq!(bounds.last().unwrap().1, 25);
        for window in bounds.windows(2) {
            assert_eq!(
                window[0].1, window[1].0,
                "segments should be contiguous, not overlapping"
            );
        }
        // 25 / 4 = 6 remainder 1: one segment of 7, three of 6.
        let lens: Vec<usize> = bounds.iter().map(|&(s, e)| e - s).collect();
        assert_eq!(lens, vec![7, 6, 6, 6]);
    }

    #[test]
    fn even_segment_bounds_handles_short_and_empty_sequences() {
        assert_eq!(even_segment_bounds(0, 4), Vec::<(usize, usize)>::new());
        // Fewer frames than workers: one segment per frame, not empty ones.
        assert_eq!(even_segment_bounds(2, 4), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn resolve_n_workers_passes_small_positive_values_through_unchanged() {
        assert_eq!(resolve_n_workers(1), 1);
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        if available > 1 {
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
        assert_eq!(resolve_n_workers(-(available as isize) - 100), 1);
    }

    #[test]
    #[should_panic(expected = "n_workers must not be 0")]
    fn resolve_n_workers_rejects_zero() {
        resolve_n_workers(0);
    }
}
