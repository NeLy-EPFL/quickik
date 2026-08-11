# From 2D keypoint positions

Not every tracking source gives 3D positions directly: a single camera view only gives 2D pixel coordinates, and recovering the underlying 3D pose is itself part of what the solve needs to do. `Position2D` observations cover this:

- **`Camera`:** a pinhole projection model (focal lengths, principal point, and the camera's own pose relative to the body plan's world frame). QuickIK projects each candidate 3D keypoint position through it before comparing to the observed pixel coordinates, rather than comparing 3D positions directly.
- **`ortho-XY projection`:** keypoints already reprojected onto a physical X-Y plane (e.g., an overhead tracking setup). No camera intrinsics/extrinsics involved, just the identity projection dropping Z.

Set `SolverConfig`'s projection to either one to switch a solver from 3D to 2D observations:

=== "Rust"

    ```rust
    use quickik::observation::ortho-XY projection;

    let ortho_xy = Some(ortho-XY projection);
    let config = SolverConfig { projection: ortho_xy, ..SolverConfig::default() };
    let mut solver: Solver<ortho-XY projection> = Solver::new(&kinematic_tree, config);
    ```

=== "Python"

    ```python
    ortho_xy = quickik.ortho-XY projection()
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig(), projection=ortho_xy)
    ```

    !!! note "Handling of `projection` in Python"
        The `projection` argument is built with `quickik.Projection.new_3d()` (the default, for 3D keypoints), `quickik.Projection.new_ortho_xy()`, or `quickik.Projection.new_pinhole_camera(fx, fy, cx, cy, world2cam_pos, world2cam_rot_mat)` -- the same three constructors as Rust's `Projection`. It's fixed at construction and exposed as a read-only property.

=== "C++"

    ```cpp
    auto solver_config = quickik::default_solver_config();
    auto ortho_xy = quickik::projection_ortho_xy();
    auto solver = quickik::new_solver(*tree, solver_config, ortho_xy);
    ```

    !!! note "Handling of `projection` in C++"
        The `projection` argument is built via `projection_3d()` (for 3D keypoints), `projection_pinhole_camera(camera)`, or `projection_ortho_xy()`, and is fixed at construction. Rust's `Projection` holds its calibration inline, which can't cross the cxx bridge, so C++ names the projection it wants with a small tagged struct instead.

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

The figure below shows a comparison of the throughput and error (from ground-truth 3D keypoint positions) of the 2D and 3D inverse kinematics solutions.

![2D vs. 3D solutions](../assets/benchmarks/comparison-2d.svg)
