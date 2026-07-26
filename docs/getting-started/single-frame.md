# Solve pose for a single frame

## Setting up a solver

A solver is built once from a kinematic tree (loaded from a [body plan](body-plan.md)) and a solver configuration object, then reused across every frame you solve.

The solver configuration bundles the iteration count, regularization weight, convergence tolerance, and damping:

- **`n_iterations`:** how many Gauss-Newton steps to run per solve call, and the cap early stopping can cut short.
- **`neutral_weight`:** how strongly every joint angle is pulled toward its neutral pose, multiplied with each DOF's own `weight_scaler` from the body plan. This is what keeps `Missing` keypoints, and under-constrained DOFs generally, from drifting to an arbitrary angle, at the cost of some bias where that DOF *is* observed.
- **`position_tolerance`/`angle_tolerance`:** stop iterating early once an update step's largest position and angle components both drop below these. `0` disables early stopping.
- **`damping`:** Levenberg-Marquardt damping added to the normal equations' diagonal, for numerical stability only. Keep it very small (default `1e-6`).
- **`mapper`:** used for keypoint positions given in 2D projections (see ["From 2D keypoint positions"](2d-keypoints.md)). By default, it is `NoMapper` for 3D keypoint positions (as is the case here).

It stays mutable for retuning between calls: in Rust and Python it's a live handle attached to the solver, so changing a field takes effect on the next solve. C++'s configuration is a plain value struct instead, with no shared live handle: mutate a copy and pass it back to the solver to apply it.

The example below loads a body plan, then creates a solver with the default configuration and a state initialized to the neutral pose:

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::solver::{Solver, SolverConfig};
    use quickik::state::State;
    use quickik::observation::KeypointObservation;
    use nalgebra::Vector3;

    let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
    let mut solver_config = SolverConfig::default();
    let mut solver: Solver = Solver::new(&kinematic_tree, solver_config);
    
    // Construct a mutable state once, reuse across many solves (to be used later)
    let mut state = State::neutral_pose(kinematic_tree.clone());
    ```

=== "Python"

    ```python
    from quickik import KinematicTree, State, Solver, SolverConfig, KeypointObservation

    kinematic_tree = KinematicTree.from_json_file("body_plan.json")
    solver_config = SolverConfig()
    solver = Solver(kinematic_tree, solver_config)

    # Initiate a state object once, reuse across many solves (to be used later)
    state = State.neutral_pose(kinematic_tree)
    ```

=== "C++"

    ```cpp
    #include <iostream>
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto solver_config = quickik::default_solver_config();
    auto solver = quickik::new_solver(
        *tree,
        solver_config,
        quickik::no_mapper(),  // mapper must be written out explicitly in C++
    );

    // Initiate a state object once, reuse across many solves (to be used later)
    auto state = quickik::state_neutral_pose(*tree);
    ```

## Solving a frame

To fit a pose, call the solver's solve method with the state to update and a list of keypoint observations, one per keypoint, in the body plan's joint order. An observation must be given for every keypoint, but some can be of type `Missing`. QuickIK supports keypoint observations in both a 3D and 2D (see ["From 2D keypoint positions"](2d-keypoints.md)). For now, we will use 3D keypoints.

The solve method jointly fits every joint angle, plus the root pose (unless the root is fixed in the body plan), against all of the observations at once. This is what differentiates QuickIK from inverse kinematics in its <abbr title="&quot;The inverse kinematics problem consists of the determination of the joint variables corresponding to a given end-effector position and orientation.&quot; Siciliano, Bruno, et al. Robotics: modelling, planning and control. Springer, 2009.">traditional definition</abbr>: instead of solving for only the end effector (e.g., hand or foot), and doing so independently for each kinematic chain (e.g., limb), we take all keypoints on the whole body into consideration in a single solve.[^1] This way, the keypoints help constrain one another and make QuickIK more robust to missing and 2D observations.

[^1]:
    Though QuickIK is not the only library that does this: for example, see [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) and [RBDL](https://github.com/rbdl/rbdl) in our [benchmark tests](../technical/benchmarks.md).

Continuing the example above with three observed keypoint positions:

=== "Rust"

    ```rust
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
    observations = [
        KeypointObservation.position_3d((0.0, 0.0, 0.0), 1.0),
        KeypointObservation.position_3d((1.0, 0.0, 0.0), 1.0),
        KeypointObservation.position_3d((1.0, 1.0, 0.0), 1.0),
    ]
    solver.solve(state, observations)
    print(state.dof_angles)
    ```

=== "C++"

    ```cpp
    std::vector<quickik::KeypointObservation> observations = {
        quickik::keypoint_position_3d({0.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 0.0, 0.0}, 1.0),
        quickik::keypoint_position_3d({1.0, 1.0, 0.0}, 1.0),
    };
    // Wrap this frame's observations in a Rust Slice view
    auto observations = rust::Slice<const quickik::KeypointObservation>(
        observations.data(), observations.size()
    );
    
    solver->solve(*state, observations);

    for (float angle : state->dof_angles()) {
        std::cout << angle << " ";
    }
    std::cout << std::endl;
    ```

The `solve` method updates the `State` object in place, so the fitted joint angles and root pose are read back off the same state object afterward.

`Missing` keypoints don't just drop out of the fit: with nothing pulling them away, the solve falls back on the solver configuration's neutral-pose prior for any DOF only those keypoints could otherwise constrain. A body with everything missing settles at its neutral pose rather than an arbitrary one.
