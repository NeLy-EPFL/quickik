# Python API reference

`quickik`'s Python bindings mirror the Rust crate's API: a `KinematicTree` loaded once, `KeypointObservation`s that carry a `Projection` (a `PinholeCamera` or `OrthographicViewXY`) when they are 2D, a `SolverConfig` with optional priors, and one of three solvers: `SingleFrameSolver` for one frame at a time, `SequentialSolver` for a warm-started stream (frame by frame, or a whole `ObservationSequence` at once, optionally in parallel), or `BatchedSolver` for a batch of independent items (e.g. training with `quickik.torch`).

::: quickik
    options:
      show_root_heading: false
      # `quickik/__init__.py` re-exports the compiled extension module via
      # `from .quickik import *`; static analysis can't see through that
      # wildcard import, so force runtime introspection instead.
      force_inspection: true
      members:
        - DofType
        - Dof
        - Joint
        - KinematicTree
        - State
        - ForwardKinematicsWorkspace
        - forward_kinematics
        - Projection
        - PinholeCamera
        - OrthographicViewXY
        - KeypointObservation
        - ObservationSequence
        - SolverConfig
        - SolverResult
        - SingleFrameSolver
        - SequentialSolver
        - BatchedSolverResult
        - BatchedSolver
