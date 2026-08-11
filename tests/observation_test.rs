use nalgebra::{DMatrix, Matrix3, Vector2, Vector3};
use quickik::observation::Projection;

fn test_camera() -> Projection {
    Projection::new_pinhole_camera(
        500.0,
        480.0,
        320.0,
        240.0,
        Vector3::new(0.1, -0.2, 4.0),
        // A generic (not identity) rotation, so a bug that drops or
        // transposes the world-to-camera rotation can't pass unnoticed.
        *nalgebra::Rotation3::from_euler_angles(0.2, -0.3, 0.15).matrix(),
    )
}

/// Calls `project_to_2d` with a fresh `3 x n` input Jacobian, returning the
/// projected position and the `2 x n` output. `project_to_2d` writes its
/// Jacobian into a caller-owned view, which is what the solver wants but is
/// noisy to set up in every test.
fn project(
    projection: &Projection,
    pos_3d: &Vector3<f32>,
    jacobian_3d: &DMatrix<f32>,
) -> (Vector2<f32>, DMatrix<f32>) {
    let mut jacobian_2d = DMatrix::<f32>::zeros(2, jacobian_3d.ncols());
    let pos_2d = projection.project_to_2d(
        pos_3d,
        &jacobian_3d.as_view(),
        &mut jacobian_2d.as_view_mut(),
    );
    (pos_2d, jacobian_2d)
}

/// Projects a position only, with a placeholder Jacobian: for tests that don't
/// care about the derivative.
fn project_pos(projection: &Projection, pos_3d: &Vector3<f32>) -> Vector2<f32> {
    project(projection, pos_3d, &DMatrix::<f32>::identity(3, 3)).0
}

#[test]
fn new_3d_reports_itself_as_3d_and_the_others_dont() {
    assert!(Projection::new_3d().is_3d());
    assert!(Projection::default().is_3d());
    assert!(!Projection::new_ortho_xy().is_3d());
    assert!(!test_camera().is_3d());
}

#[test]
#[should_panic(expected = "projects nothing")]
fn new_3d_panics_if_asked_to_project() {
    project_pos(&Projection::new_3d(), &Vector3::new(1.0, 2.0, 3.0));
}

#[test]
fn ortho_xy_drops_z_and_passes_through_xy() {
    let jacobian_3d = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let (pos_2d, jacobian_2d) = project(
        &Projection::new_ortho_xy(),
        &Vector3::new(1.0, 2.0, 3.0),
        &jacobian_3d,
    );

    assert_eq!(pos_2d, Vector2::new(1.0, 2.0));
    // Dropping Z means the projected Jacobian is the input's first two rows.
    assert_eq!(jacobian_2d, jacobian_3d.rows(0, 2).into_owned());
}

#[test]
fn pinhole_camera_projects_point_directly_in_front_to_principal_point() {
    let camera = Projection::new_pinhole_camera(
        500.0,
        500.0,
        320.0,
        240.0,
        Vector3::new(0.0, 0.0, 5.0),
        Matrix3::identity(),
    );

    let pos_2d = project_pos(&camera, &Vector3::new(0.0, 0.0, 0.0));

    assert!((pos_2d - Vector2::new(320.0, 240.0)).norm() < 1e-4);
}

#[test]
fn pinhole_camera_principal_point_shifts_the_projection_one_for_one() {
    let pos_3d = Vector3::new(0.3, -0.1, 0.5);
    let baseline_2d = project_pos(&test_camera(), &pos_3d);

    let shifted = Projection::new_pinhole_camera(
        500.0,
        480.0,
        320.0 + 7.0,
        240.0 - 3.0,
        Vector3::new(0.1, -0.2, 4.0),
        *nalgebra::Rotation3::from_euler_angles(0.2, -0.3, 0.15).matrix(),
    );

    let delta = project_pos(&shifted, &pos_3d) - baseline_2d;
    assert!((delta - Vector2::new(7.0, -3.0)).norm() < 1e-3);
}

#[test]
fn pinhole_camera_jacobian_matches_finite_differences() {
    let camera = test_camera();
    let pos_3d = Vector3::new(0.3, -0.1, 0.5);
    // An identity input Jacobian makes the output exactly d(pos_2d)/d(pos_3d),
    // which is what the finite differences below approximate.
    let (baseline_2d, jacobian_2d) = project(&camera, &pos_3d, &DMatrix::identity(3, 3));

    // The derivatives here are ~O(1e2), so the tolerance is relative. eps
    // balances two error sources: perspective division makes the projection
    // nonlinear in every world axis once the rotation isn't axis-aligned (so
    // a large eps carries real truncation error), while f32 rounding in the
    // ~O(1e2) pixel coordinates swamps the difference if eps is too small.
    let eps = 1e-3;
    for axis in 0..3 {
        let mut perturbed = pos_3d;
        perturbed[axis] += eps;
        let numerical_d = (project_pos(&camera, &perturbed) - baseline_2d) / eps;
        let analytical_d = jacobian_2d.column(axis);
        assert!(
            (numerical_d - analytical_d).norm() < 1e-3 * analytical_d.norm().max(1.0),
            "axis {axis}: analytical {:?} vs numerical {:?}",
            analytical_d.as_slice(),
            numerical_d.as_slice()
        );
    }
}

#[test]
fn projection_chains_onto_the_input_jacobian_rather_than_replacing_it() {
    // Two different input Jacobians through the same camera: the second output
    // must be the first left-multiplied onto it. Catches an implementation that
    // ignores `jacobian_3d` and writes d(pos_2d)/d(pos_3d) straight out.
    let camera = test_camera();
    let pos_3d = Vector3::new(0.3, -0.1, 0.5);
    let (_, d_pos2d_d_pos3d) = project(&camera, &pos_3d, &DMatrix::identity(3, 3));

    let jacobian_3d = DMatrix::from_row_slice(3, 2, &[0.5, -1.0, 2.0, 0.25, -0.75, 1.5]);
    let (_, jacobian_2d) = project(&camera, &pos_3d, &jacobian_3d);

    let expected = &d_pos2d_d_pos3d * &jacobian_3d;
    let max_abs_diff = (jacobian_2d - expected)
        .iter()
        .fold(0.0f32, |acc, &x| acc.max(x.abs()));
    assert!(max_abs_diff < 1e-4, "max abs diff = {max_abs_diff}");
}
