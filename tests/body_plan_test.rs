mod common;

use nalgebra::Vector3;
use quickik::body_plan::KinematicTree;

use common::{body_plan_path, hinge, joint};

#[test]
fn parses_example_body_plans() {
    let fly = KinematicTree::from_json_file(body_plan_path("neuromechfly")).unwrap();
    assert_eq!(
        (fly.n_joints(), fly.n_dofs(), fly.state_dim()),
        (31, 42, 48)
    );
    let coxa = fly.joint_index("lf_thorax_coxa").unwrap();
    let tibia = fly.joint_index("lf_trochanterfemur_tibia").unwrap();
    assert_eq!(fly.joints()[coxa].weight_multiplier, 0.5);
    assert_eq!(fly.joints()[tibia].weight_multiplier, 1.0);

    let g1 = KinematicTree::from_json_file(body_plan_path("g1")).unwrap();
    assert_eq!(
        (g1.n_joints(), g1.n_dofs(), g1.fixed_base()),
        (33, 29, false)
    );
}

#[test]
fn rejects_invalid_trees() {
    let z = Vector3::z();
    let cases = [
        (
            "no root",
            vec![
                joint("a", Some(1), [0.0; 3], vec![]),
                joint("b", Some(0), [0.0; 3], vec![]),
            ],
        ),
        (
            "two roots",
            vec![
                joint("a", None, [0.0; 3], vec![]),
                joint("b", None, [0.0; 3], vec![]),
            ],
        ),
        (
            "cycle",
            vec![
                joint("root", None, [0.0; 3], vec![]),
                joint("a", Some(2), [0.0; 3], vec![]),
                joint("b", Some(1), [0.0; 3], vec![]),
            ],
        ),
        (
            "bad parent",
            vec![
                joint("a", None, [0.0; 3], vec![]),
                joint("b", Some(7), [0.0; 3], vec![]),
            ],
        ),
        (
            "duplicate name",
            vec![
                joint("a", None, [0.0; 3], vec![]),
                joint("a", Some(0), [0.0; 3], vec![]),
            ],
        ),
        (
            "zero axis",
            vec![joint("a", None, [0.0; 3], vec![hinge(z * 0.0)])],
        ),
    ];
    for (name, joints) in cases {
        assert!(
            KinematicTree::new(joints, false).is_err(),
            "{name} was accepted"
        );
    }
}

#[test]
fn rejects_unknown_parent_name_in_json() {
    let json = r#"{"joints": [
        {"name": "root", "parent": null, "offset_pos": [0, 0, 0],
         "offset_quat": [1, 0, 0, 0], "dofs": []},
        {"name": "a", "parent": "nope", "offset_pos": [0, 0, 0],
         "offset_quat": [1, 0, 0, 0], "dofs": []}
    ]}"#;
    assert!(KinematicTree::from_json_str(json).is_err());
}
