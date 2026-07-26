# From 2D keypoint positions

Not every tracking source gives 3D positions directly: a single camera view only gives 2D pixel coordinates, and recovering the underlying 3D pose is itself part of what the solve needs to do. `Position2D` observations cover this:

- **`Camera`:** a pinhole projection model (focal lengths, principal point, and the camera's own pose relative to the body plan's world frame). QuickIK projects each candidate 3D keypoint position through it before comparing to the observed pixel coordinates, rather than comparing 3D positions directly.
- **`XYView`:** keypoints already reprojected onto a physical X-Y plane (e.g., an overhead tracking setup). No camera intrinsics/extrinsics involved, just the identity projection dropping Z.

Set `SolverConfig`'s mapper to either one to switch a solver from 3D to 2D observations:

=== "Rust"

    ```rust
    use quickik::observation::XYView;

    let ortho_xy = Some(XYView);
    let config = SolverConfig { mapper: ortho_xy, ..SolverConfig::default() };
    let mut solver: Solver<XYView> = Solver::new(&kinematic_tree, config);
    ```

=== "Python"

    ```python
    ortho_xy = quickik.XYView()
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig(), mapper=ortho_xy)
    ```

    !!! note "Handling of `mapper` in Python"
        Rust's `Solver<M>` is generic over the mapper type. Python doesn't have an equivalent. Instead, the solver config instance holds every attribute that the Rust `SolverConfig` has except the mapper. The `__init__` method of the `Solver` receives the kinematic tree, the solver config, _plus a mapper_, as arguments. The optional `mapper` argument can be `None` (for 3D keypoints, default), a `Camera` object, or an empty `XYView` object.

=== "C++"

    ```cpp
    auto solver_config = quickik::default_solver_config();
    auto ortho_xy = quickik::xyview_mapper();
    auto solver = quickik::new_solver(*tree, solver_config, ortho_xy);
    ```

    !!! note "Handling of `mapper` in C++"
        Rust's `Solver<M>` is generic over the mapper type. C++ doesn't have an equivalent. Instead, the solver config instance holds every attribute that the Rust `SolverConfig` has except the mapper. The `new_solver` function receives the kinematic tree, the solver config, _plus a mapper_, as arguments. The `mapper` argument can be built via `no_mapper()` (for 3D keypoints), `camera_mapper(camera)`, or `xyview_mapper()`.

!!! warning "Inverse kinematics from 2D keypoint positions is fundamentally underconstrained"
    Fitting 3D kinematic states from only 2D keypoints is fundamentally a degenerate, underconstrained problem. QuickIK's 2D capability doesn't magically solve that. The user must take the inverse kinematics from 2D poses with a grain of salt and validate it more rigorously.

    !!! tip
        Increasing the weight for the neutral pose prior generally improves the quality of inverse kinematics from 2D data. Better camera angles also make a huge difference.


## 2D vs. 3D fit

In the following video, the blue lines show the result of inverse kinematics based on 3D keypoint positions, shown as gray dots in 3D.

On the floor "X-Y projection" pane, the same observed keypoint positions are projected to 2D. QuickIK attempts another inverse kinematics reconstruction, this time given only these 2D observations. From these reconstructed joint angles, albeit based on 2D data, one can nevertheless recover a 3D pose, shown in green.

Of course, limitations exist: The green fit matches the observations about as well as blue on the X-Y plane (see projections on the floor), but the green keypoints deviate from their ground truth more noticeably in 3D. This comes from a fundamental limitation in data.

<video style="width: 70%" autoplay loop muted controls>
  <source src="https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/example_clip_2d_xyview.mp4" type="video/mp4">
</video>