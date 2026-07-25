# From 2D keypoint positions

Not every tracking source gives 3D positions directly. A single camera view only ever gives 2D pixel coordinates, and recovering the underlying 3D pose is itself part of what the solve needs to do. `Position2D` observations cover this: set `SolverConfig`'s mapper to a `Camera` (a pinhole projection model – focal lengths, principal point, and the camera's own pose relative to the body plan's world frame) and QuickIK projects each candidate 3D keypoint position through it before comparing to the observed pixel coordinates, rather than comparing 3D positions directly. `XYView` covers the simpler case of keypoints already reprojected onto a physical X-Y plane (e.g. a top-down tracking setup) – no camera intrinsics/extrinsics involved, just the identity projection dropping Z.

=== "Rust"

    ```rust
    use quickik::observation::Camera;
    use nalgebra::Matrix3;

    let camera = Camera {
        fx: 800.0,
        fy: 800.0,
        cx: 320.0,
        cy: 240.0,
        world2cam_pos: Vector3::new(0.0, 0.0, 5.0),
        world2cam_rot_mat: Matrix3::identity(),
    };
    let config = SolverConfig { mapper: Some(camera), ..SolverConfig::default() };
    let mut solver: Solver<Camera> = Solver::new(&kinematic_tree, config);
    ```

=== "Python"

    ```python
    camera = quickik.Camera(
        fx=800.0, fy=800.0, cx=320.0, cy=240.0,
        world2cam_pos=(0.0, 0.0, 5.0),
        world2cam_rot_mat=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),  # row-major 3x3
    )
    solver = quickik.Solver(kinematic_tree, quickik.SolverConfig(), mapper=camera)
    ```

=== "C++"

    ```cpp
    quickik::Camera camera{};
    camera.fx = 800.0f;
    camera.fy = 800.0f;
    camera.cx = 320.0f;
    camera.cy = 240.0f;
    camera.world2cam_pos = {0.0f, 0.0f, 5.0f};
    camera.world2cam_rot_mat = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};  // row-major 3x3
    auto solver = quickik::new_solver(*tree, quickik::default_solver_config(), quickik::camera_mapper(camera));
    ```

Rust's `Solver<M>` is generic over the mapper type at compile time; neither Python nor C++ has an equivalent, so every `Solver`/`SequenceSolver` is backed by a single mapper value chosen at runtime instead – Python takes `mapper=None` (the default), a `Camera`, or an `XYView()` as a keyword argument; C++ takes a `quickik::Mapper` built via `no_mapper()`/`camera_mapper(camera)`/`xyview_mapper()` as an ordinary constructor argument in the same spots. It's deliberately not part of `SolverConfig` in either binding: like Rust's `M`, it's fixed for the solver's lifetime (read-only `mapper`/`solver.mapper` accessor, no setter), whereas `SolverConfig`'s other fields stay freely mutable.

Errors from malformed input (bad JSON, wrong-sized vectors, a `Position2D` observation with no mapper set) don't crash either binding – both raise a catchable error instead. In C++, that's a `rust::Error` (a normal `std::exception`). In Python, it's `pyo3_runtime.PanicException`, which – deliberately, on PyO3's part – subclasses `BaseException`, not `Exception`; a bare `except Exception:` won't catch it, so code that needs to handle these needs `except BaseException:` (or the specific `PanicException` type) instead.

## 2D vs. 3D fit

The same NeuroMechFly recording solved twice with a `SequenceSolver` – once from full 3D keypoint observations, once from `XYView`-only (x/y) observations – overlaid to show what the missing depth information costs. Blue is the 3D fit, green is the `XYView` fit, gray dots are the raw MoCap keypoints; the floor-projected copy of each skeleton is the actual view the 2D fit was observed from.

<video controls style="width: 100%">
  <source src="https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/example_clip_2d_xyview.mp4" type="video/mp4">
</video>
