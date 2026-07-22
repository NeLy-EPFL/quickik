//! Correctness cross-check and throughput/latency benchmark for fastik on the
//! NeuroMechFly model, against flygym.ik. Rust API only (no fastik Python
//! bindings exercised here). See `README.md` for background and how to
//! regenerate `assets/fixtures.json`.

use std::sync::Arc;

use fastik::body_plan::KinematicTree;
use fastik_benchmark::{correctness, fixtures, perf};

fn main() {
    let assets_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets");
    let tree = Arc::new(KinematicTree::from_json_file(
        assets_dir.join("neuromechfly_ypr_legs.json"),
    ));
    let fixtures = fixtures::load(assets_dir.join("fixtures.json"));

    println!(
        "Loaded body plan: {} joints, {} dofs, state_dim={}\n",
        tree.n_joints(),
        tree.n_dofs(),
        tree.state_dim()
    );
    let leg_joint_names: Vec<&str> = tree.joints[1..].iter().map(|j| j.name.as_str()).collect();
    assert_eq!(
        leg_joint_names, fixtures.leg_joint_names,
        "fixtures.json's leg_joint_names order doesn't match the loaded body plan's joints[1..] \
         -- regenerate fixtures with scripts/generate_fixtures.py"
    );

    correctness::run_all(&tree, &fixtures);
    perf::run_all(&tree, &fixtures);
}
