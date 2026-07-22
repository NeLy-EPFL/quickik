//! Correctness cross-check and throughput/latency benchmark for fastik,
//! against reference fixtures. Rust API only (no fastik Python bindings
//! exercised here). Runs the same harness once per body in [`BODIES`]. See
//! `README.md` for background and how to regenerate the fixtures.

use std::sync::Arc;

use fastik::body_plan::KinematicTree;
use fastik_benchmark::{correctness, fixtures, perf};

/// One body to benchmark: its body plan and matching fixtures file.
struct BodyConfig {
    name: &'static str,
    body_plan: &'static str,
    fixtures: &'static str,
}

const BODIES: &[BodyConfig] = &[
    BodyConfig {
        name: "neuromechfly",
        body_plan: "neuromechfly_ypr_legs.json",
        fixtures: "fixtures.json",
    },
    BodyConfig {
        name: "g1",
        body_plan: "g1_body_plan.json",
        fixtures: "fixtures_g1.json",
    },
];

fn main() {
    let assets_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets");

    for body in BODIES {
        println!("\n########## body: {} ##########\n", body.name);

        let tree = Arc::new(KinematicTree::from_json_file(
            assets_dir.join(body.body_plan),
        ));
        let fixtures = fixtures::load(assets_dir.join(body.fixtures));

        println!(
            "Loaded body plan: {} joints, {} dofs, state_dim={}\n",
            tree.n_joints(),
            tree.n_dofs(),
            tree.state_dim()
        );
        let leg_joint_names: Vec<&str> =
            tree.joints[1..].iter().map(|j| j.name.as_str()).collect();
        assert_eq!(
            leg_joint_names, fixtures.leg_joint_names,
            "{}'s leg_joint_names order doesn't match the loaded body plan's joints[1..] -- \
             regenerate fixtures with scripts/generate_fixtures.py",
            body.fixtures
        );

        correctness::run_all(&tree, &fixtures);
        perf::run_all(&tree, &fixtures, body.name);
    }
}
