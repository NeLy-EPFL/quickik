# Python API reference

`quickik`'s Python bindings mirror the Rust crate's layout: a `KinematicTree`
loaded once, `State`/`KeypointObservation`/`Projection` values fed in per
frame, and one of three solvers that tie them together: `Solver` for a single
frame, `SequenceSolver` for a warm-started sequence, or `BatchedSolver` for a
batch of independent frames (e.g. training with `quickik.torch`).

::: quickik
    options:
      show_root_heading: false
      # `quickik/__init__.py` re-exports the compiled extension module via
      # `from .quickik import *`; static analysis can't see through that
      # wildcard import, so force runtime introspection instead.
      force_inspection: true
      members:
        - KinematicTree
        - State
        - KeypointObservation
        - Projection
        - SolverResult
        - Solver
        - SequenceSolver
        - BatchedSolverResult
        - BatchedSolver
