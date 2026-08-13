# Solve pose for a single frame

## Setting up a solver

A solver is built once from a kinematic tree (loaded from a [body plan](body-plan.md)) and its tuning parameters, then reused across every frame you solve.

Those tuning parameters are the iteration count, regularization weight, convergence tolerance, and damping:

- **`n_iterations`:** how many Gauss-Newton steps to run per solve call, and the cap early stopping can cut short.
- **`neutral_weight`:** how strongly every joint angle is pulled toward its neutral pose, multiplied with each DOF's own `weight_scaler` from the body plan. This is what keeps `Missing` keypoints, and under-constrained DOFs generally, from drifting to an arbitrary angle, at the cost of some bias where that DOF *is* observed.
- **`position_tolerance`/`angle_tolerance`:** stop iterating early once an update step's largest position and angle components both drop below these. `0` disables early stopping.
- **`damping`:** Levenberg-Marquardt damping added to the normal equations' diagonal, for numerical stability only. Keep it very small (default `1e-6`).
- **`projection`:** used for keypoint positions given in 2D projections (see ["From 2D keypoint positions"](2d-keypoints.md)). By default there is no projection, meaning keypoint positions are given in 3D (as is the case here).

`n_iterations`, `neutral_weight`, `position_tolerance`, `angle_tolerance`, and `damping` are plain, freely retunable fields on the solver (Python: attributes with getters/setters) -- change one and it takes effect on the next `solve` call. `projection` is fixed at construction instead, with no setter: a solver preallocates its buffers against the residual dimension the projection implies, so changing it means building a new solver.

The example below loads a body plan, then creates a solver with the default tuning and a state initialized to the neutral pose:

=== "Rust"

    ```rust
    use std::sync::Arc;
    use quickik::body_plan::KinematicTree;
    use quickik::solver::Solver;
    use quickik::state::State;
    use quickik::observation::{KeypointObservation, Projection};
    use nalgebra::Vector3;

    let kinematic_tree = Arc::new(KinematicTree::from_json_file("body_plan.json"));
    let mut solver = Solver::new(
        &kinematic_tree,
        Projection::new_3d(),
        10,    // n_iterations
        1e-3,  // neutral_weight
        1e-3,  // position_tolerance
        1e-3,  // angle_tolerance
        1e-6,  // damping
    );

    // Construct a mutable state once, reuse across many solves (to be used later)
    let mut state = State::neutral_pose(kinematic_tree.clone());
    ```

=== "Python"

    ```python
    from quickik import KinematicTree, State, Solver, KeypointObservation

    kinematic_tree = KinematicTree.from_json_file("body_plan.json")
    solver = Solver(kinematic_tree)  # every tuning parameter above has a default

    # Initiate a state object once, reuse across many solves (to be used later)
    state = State.neutral_pose(kinematic_tree)
    ```

=== "C++"

    ```cpp
    #include <iostream>
    #include <vector>
    #include "quickik.h"

    auto tree = quickik::kinematic_tree_from_json_file("body_plan.json");
    auto solver_config = quickik::default_solver_config();
    auto solver = quickik::new_solver(
        *tree,
        solver_config,
        quickik::projection_3d()  // projection must be written out explicitly in C++
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
    let result = solver.solve(&mut state, &observations, false, false);
    println!("{:?}", result.state.dof_values);
    ```

=== "Python"

    ```python
    observations = [
        KeypointObservation.position_3d((0.0, 0.0, 0.0), 1.0),
        KeypointObservation.position_3d((1.0, 0.0, 0.0), 1.0),
        KeypointObservation.position_3d((1.0, 1.0, 0.0), 1.0),
    ]
    result = solver.solve(state, observations)
    print(result.dof_angles)
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

`solve` updates the `State` object passed in place, so the fitted joint angles and root pose are read back off the same state object afterward; the returned `SolverResult` carries the same pose too (`dof_angles`/`root_pos`/`root_rot`, or the full `state`), which is often more convenient since it's a self-contained snapshot rather than a handle you have to keep mutating.

`with_grad`/`with_fk` (both `false` above) each gate extra output that costs a little more work per solve, so only turn on what you'll actually use:

- **`with_fk`:** returns the converged pose's world-space keypoint positions; see ["Checking fit quality"](#checking-fit-quality) below.
- **`with_grad`:** returns the residual Jacobian and its Cholesky factor at (approximately) the converged pose, as `SolverResult`'s `jacobian`/`cholesky_l`. Most one-off solves don't need this; it exists for differentiating through the solve, as [`BatchedSolver`](../api/python.md) does for training with an autodiff framework.

`Missing` keypoints don't just drop out of the fit: with nothing pulling them away, the solve falls back on the neutral-pose prior for any DOF only those keypoints could otherwise constrain. A body with everything missing settles at its neutral pose rather than an arbitrary one.

## Checking fit quality

Forward kinematics itself isn't exposed as a standalone call, but `solve` can return the world-space keypoint positions it converged to, in the same joint order as the observations you passed in, by passing `with_fk`. This is the easiest way to check fit quality (e.g. residual error against your original observations) without recomputing forward kinematics yourself.

=== "Rust"

    ```rust
    let result = solver.solve(&mut state, &observations, false, true);
    for pos in result.keypoint_pos.unwrap() {
        println!("{:?}", pos);
    }
    ```

=== "Python"

    ```python
    result = solver.solve(state, observations, with_fk=True)
    print(result.keypoint_pos)  # (n_joints, 3) NumPy array
    ```

=== "C++"

    ```cpp
    auto fk_positions = solver->last_fk_positions();  // flat, n_joints * 3 long
    for (size_t k = 0; k < tree->n_joints(); k++) {
        std::cout << fk_positions[k * 3] << " " << fk_positions[k * 3 + 1] << " " << fk_positions[k * 3 + 2] << std::endl;
    }
    ```
