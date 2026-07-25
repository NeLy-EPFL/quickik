use quickik::body_plan::{DofType, KinematicTree};

fn valid_body_json() -> &'static str {
    r#"{
        "joints": [
            {
                "name": "root",
                "parent": null,
                "offset_pos": [0.0, 0.0, 0.0],
                "offset_quat": [1.0, 0.0, 0.0, 0.0],
                "dofs": []
            },
            {
                "name": "joint1",
                "parent": "root",
                "offset_pos": [1.0, 0.0, 0.0],
                "offset_quat": [1.0, 0.0, 0.0, 0.0],
                "weight_scaler": 0.5,
                "dofs": [
                    {"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.0, "limits": null}
                ]
            },
            {
                "name": "joint2",
                "parent": "joint1",
                "offset_pos": [1.0, 0.0, 0.0],
                "offset_quat": [1.0, 0.0, 0.0, 0.0],
                "dofs": [
                    {"axis": [0.0, 0.0, 1.0], "type": "hinge", "neutral": 0.1,
                     "weight_scaler": 0.25, "limits": [-0.5, 0.5]}
                ]
            }
        ]
    }"#
}

#[test]
fn parses_joints_parents_and_dof_offsets() {
    let tree = KinematicTree::from_json_str(valid_body_json());

    assert_eq!(tree.n_joints(), 3);
    assert_eq!(tree.n_dofs(), 2);
    assert_eq!(tree.root_idx, 0);

    assert_eq!(tree.joints[0].parent, None);
    assert_eq!(tree.joints[1].parent, Some(0));
    assert_eq!(tree.joints[2].parent, Some(1));

    assert_eq!(tree.joints[1].dof_offset, 0);
    assert_eq!(tree.joints[2].dof_offset, 1);
    assert_eq!(tree.joints[2].dofs[0].neutral, 0.1);
    assert_eq!(tree.joints[2].dofs[0].limits, Some([-0.5, 0.5]));

    // weight_scaler: explicit values parse through, and omitted ones default
    // to 1.0.
    assert_eq!(tree.joints[0].weight_scaler, 1.0);
    assert_eq!(tree.joints[1].weight_scaler, 0.5);
    assert_eq!(tree.joints[2].dofs[0].weight_scaler, 0.25);

    assert_eq!(tree.children_indices(0), &[1]);
    assert_eq!(tree.children_indices(1), &[2]);
    assert_eq!(tree.children_indices(2), &[] as &[usize]);
}

#[test]
#[should_panic(expected = "Duplicate joint name")]
fn rejects_duplicate_joint_names() {
    let json = r#"{
        "joints": [
            {"name": "root", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "root", "parent": "root", "offset_pos": [1,0,0], "offset_quat": [1,0,0,0], "dofs": []}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}

#[test]
#[should_panic(expected = "No root joint found")]
fn rejects_missing_root() {
    let json = r#"{
        "joints": [
            {"name": "a", "parent": "b", "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "b", "parent": "a", "offset_pos": [1,0,0], "offset_quat": [1,0,0,0], "dofs": []}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}

#[test]
#[should_panic(expected = "Multiple root joints found")]
fn rejects_multiple_roots() {
    let json = r#"{
        "joints": [
            {"name": "a", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "b", "parent": null, "offset_pos": [1,0,0], "offset_quat": [1,0,0,0], "dofs": []}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}

#[test]
#[should_panic(expected = "Parent joint 'missing' not found")]
fn rejects_unknown_parent_name() {
    let json = r#"{
        "joints": [
            {"name": "root", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "a", "parent": "missing", "offset_pos": [1,0,0], "offset_quat": [1,0,0,0], "dofs": []}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}

#[test]
#[should_panic(expected = "Failed to parse body plan JSON")]
fn rejects_dof_missing_type() {
    let json = r#"{
        "joints": [
            {"name": "root", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "a", "parent": "root", "offset_pos": [1,0,0], "offset_quat": [1,0,0,0],
             "dofs": [{"axis": [0,0,1], "neutral": 0.0, "limits": null}]}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}

#[test]
fn parses_slide_dofs() {
    let json = r#"{
        "joints": [
            {"name": "root", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0], "dofs": []},
            {"name": "a", "parent": "root", "offset_pos": [1,0,0], "offset_quat": [1,0,0,0],
             "dofs": [{"axis": [0,0,2], "type": "slide", "neutral": 0.5, "limits": [-1.0, 1.0]}]}
        ]
    }"#;
    let tree = KinematicTree::from_json_str(json);

    let dof = &tree.joints[1].dofs[0];
    assert_eq!(dof.dof_type, DofType::Slide);
    assert_eq!(dof.neutral, 0.5);
    assert_eq!(dof.limits, Some([-1.0, 1.0]));
    // Non-unit axes are normalized once at parse time, same as hinge DOFs.
    assert!((dof.axis - nalgebra::Vector3::new(0.0, 0.0, 1.0)).norm() < 1e-6);
}

#[test]
#[should_panic(expected = "invalid type")]
fn rejects_explicit_null_weight_scaler() {
    let json = r#"{
        "joints": [
            {"name": "root", "parent": null, "offset_pos": [0,0,0], "offset_quat": [1,0,0,0],
             "weight_scaler": null, "dofs": []}
        ]
    }"#;
    KinematicTree::from_json_str(json);
}
