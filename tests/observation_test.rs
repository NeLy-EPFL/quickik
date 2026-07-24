use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};
use quickik::observation::{Camera, Mapper3Dto2D, XYView};

#[test]
fn xyview_drops_z_and_passes_through_xy() {
    let mapper = XYView;
    let jacobian_world3d = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let mut jac2d = DMatrix::<f32>::zeros(2, 2);
    let pos2d = mapper.project_3d_to_2d(&Vector3::new(1.0, 2.0, 3.0), &jacobian_world3d, &mut jac2d);

    assert_eq!(pos2d, Vector2::new(1.0, 2.0));
    assert_eq!(jac2d, jacobian_world3d.rows(0, 2).into_owned());
}

#[test]
fn camera_projects_point_directly_in_front_to_principal_point() {
    let camera = Camera {
        fx: 500.0,
        fy: 500.0,
        cx: 320.0,
        cy: 240.0,
        world2cam_pos: Vector3::new(0.0, 0.0, 5.0),
        world2cam_rot_mat: Matrix3::identity(),
    };
    let jacobian_world3d = DMatrix::<f32>::identity(3, 3);
    let mut jac2d = DMatrix::<f32>::zeros(2, 3);
    let pos2d = camera.project_3d_to_2d(&Vector3::new(0.0, 0.0, 0.0), &jacobian_world3d, &mut jac2d);

    assert!((pos2d - Vector2::new(320.0, 240.0)).norm() < 1e-4);
}

#[test]
fn camera_jacobian_matches_finite_differences() {
    let camera = Camera {
        fx: 500.0,
        fy: 480.0,
        cx: 320.0,
        cy: 240.0,
        world2cam_pos: Vector3::new(0.1, -0.2, 4.0),
        world2cam_rot_mat: Matrix3::identity(),
    };
    let jacobian_world3d = DMatrix::<f32>::identity(3, 3);
    let pos_world3d = Vector3::new(0.3, -0.1, 0.5);
    let mut analytical_jac = DMatrix::<f32>::zeros(2, 3);
    let baseline_2d = camera.project_3d_to_2d(&pos_world3d, &jacobian_world3d, &mut analytical_jac);

    // eps is not too small: f32 rounding in the projected pixel coordinates
    // (which are ~O(1e2)) would otherwise swamp the finite-difference ratio.
    let eps = 1e-2;
    for axis in 0..3 {
        let mut perturbed = pos_world3d;
        perturbed[axis] += eps;
        let mut perturbed_jac = DMatrix::<f32>::zeros(2, 3);
        let perturbed_2d =
            camera.project_3d_to_2d(&perturbed, &jacobian_world3d, &mut perturbed_jac);
        let numerical_d = (perturbed_2d - baseline_2d) / eps;
        let analytical_d = analytical_jac.column(axis).into_owned();
        assert!(
            (numerical_d - &analytical_d).norm() < 5e-2,
            "axis {axis}: analytical {analytical_d:?} vs numerical {numerical_d:?}"
        );
    }
}
