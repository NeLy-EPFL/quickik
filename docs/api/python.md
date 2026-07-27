# Python API reference

`quickik`'s Python bindings mirror the Rust crate's layout: a `KinematicTree`
loaded once, `State`/`KeypointObservation` values fed in per frame, and a
`Solver` (or `SequenceSolver`/`solve_sequence_segmented_parallel` for whole
sequences) that ties them together.

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
        - Camera
        - XYView
        - SolverConfig
        - Solver
        - SequenceSolver
        - ParallelSolveConfig
        - solve_sequence_segmented_parallel
