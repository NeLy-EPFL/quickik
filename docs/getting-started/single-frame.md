# Solve pose for single frame

QuickIK solves *whole-tree* IK: one `Solver::solve` call takes one `KeypointObservation` per keypoint – `Missing`, `Position3D`, or `Position2D` (see [From 2D keypoint positions](from-2d-keypoints.md)) – in `kinematic_tree.joints` order, and jointly fits every joint angle at once against all of them – plus the root pose too, unless the body plan's root is [fixed](body-plan.md). This is what makes it different from solving each limb as its own small IK problem: a keypoint on one limb can still help constrain the root pose (and therefore every other limb) even if that other limb's own keypoints are all `Missing` this frame.

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::observation::KeypointObservation;
    use quickik::solver::{Solver, SolverConfig};
    use quickik::state::State;
    use nalgebra::Vector3;

    let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
    let mut state = State::neutral_pose(kinematic_tree.clone());
    let mut solver: Solver = Solver::new(&kinematic_tree, SolverConfig::default());

    let observations = vec![
        KeypointObservation::Position3D { obs_pos: Vector3::new(0.0, 0.0, 0.0), weight: 1.0 },
        KeypointObservation::Position3D { obs_pos: Vector3::new(1.0, 0.0, 0.0), weight: 1.0 },
        KeypointObservation::Position3D { obs_pos: Vector3::new(1.0, 1.0, 0.0), weight: 1.0 },
    ];
    solver.solve(&mut state, &observations);
    println!("{:?}", state.dof_angles);
    ```

=== "Python"

    ```python
    import quickik

    kinematic_tree = quickik.KinematicTree.from_json_file("body_plan.json")
    state = quickik.State.neutral_pose(kinematic_tree)
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig())

    observations = [
        quickik.KeypointObservation.position_3d((0.0, 0.0, 0.0), 1.0),
        quickik.KeypointObservation.position_3d((1.0, 0.0, 0.0), 1.0),
        quickik.KeypointObservation.position_3d((1.0, 1.0, 0.0), 1.0),
    ]
    solver.solve(state, observations)
    print(state.dof_angles)
    ```

=== "C++"

    ```cpp
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto state = quickik::state_neutral_pose(*tree);
    auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::no_mapper());

    std::vector<quickik::KeypointObservation> observations = {
        quickik::keypoint_position_3d({0.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 1.0, 0.0}, 1.0),
    };
    solver->solve(*state, rust::Slice<const quickik::KeypointObservation>(observations.data(), observations.size()));
    for (float angle : state->dof_angles()) { /* ... */ }
    ```

`Missing` keypoints (occluded this frame, or simply not tracked by this body plan's data source) don't just get dropped from the residual – with nothing pulling them away, the solve falls back on `SolverConfig`'s neutral-pose prior for whatever DOFs only those keypoints could otherwise constrain, so a body with everything missing settles at its neutral pose rather than an arbitrary one.

`SolverConfig` bundles the iteration count, damping, regularization weight, and convergence tolerance:

- `n_iterations`: how many Gauss-Newton steps to run per `solve` call, and the cap early stopping can cut short.
- `damping`: Levenberg-Marquardt damping added to the normal equations' diagonal, for numerical stability only – keep it very small (the default is `1e-6`).
- `weight`: how strongly every joint angle is pulled toward its neutral pose, multiplied together with each DOF's own `weight_scaler` from the body plan. This is what keeps `Missing` keypoints (and, more generally, under-constrained DOFs) from drifting to an arbitrary angle instead of a sensible default, at the cost of some bias on frames where that DOF *is* observed.
- `position_tolerance`/`angle_tolerance`: stop iterating early once an update step's largest position and angle components both drop below these; `0` disables early stopping.

It's set via `Solver::new`/`Solver(...)`, though `solver.config` stays mutable for retuning between calls (Python: `solver.config` is the same object every time, so `solver.config.n_iterations = 5` takes effect on the next `solve`, just like Rust). C++'s `SolverConfig` is a plain value struct instead (no shared live handle): mutate a copy and call `solver->set_config(config)` to apply it.
