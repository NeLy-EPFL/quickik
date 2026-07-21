use fastik::body_plan::KinematicTree;

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
                "dofs": [
                    {"axis": [0.0, 0.0, 1.0], "neutral_angle": 0.0, "limits": null}
                ]
            },
            {
                "name": "joint2",
                "parent": "joint1",
                "offset_pos": [1.0, 0.0, 0.0],
                "offset_quat": [1.0, 0.0, 0.0, 0.0],
                "dofs": [
                    {"axis": [0.0, 0.0, 1.0], "neutral_angle": 0.1, "limits": [-0.5, 0.5]}
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
    assert_eq!(tree.joints[2].dofs[0].neutral_angle, 0.1);
    assert_eq!(tree.joints[2].dofs[0].limits, Some([-0.5, 0.5]));

    assert_eq!(tree.children_indices(0), vec![1]);
    assert_eq!(tree.children_indices(1), vec![2]);
    assert_eq!(tree.children_indices(2), Vec::<usize>::new());
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
