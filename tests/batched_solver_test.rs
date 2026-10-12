mod common;

use std::sync::Arc;

use quickik::batched_solver::BatchedSolver;
use quickik::observation::ObservationSequence;
use quickik::solver::{SingleFrameSolver, SolverConfig};
use quickik::state::State;

use common::{branching_tree, observations_3d, some_pose};

#[test]
fn matches_independent_single_frame_solves() {
    let tree = branching_tree(false);
    let items: Vec<_> = (0..20)
        .map(|i| {
            let mut state = some_pose(&tree);
            state
                .dof_values
                .iter_mut()
                .for_each(|v| *v *= 0.1 * i as f32);
            observations_3d(&state)
        })
        .collect();
    let mut config = SolverConfig::default();
    config.add_dof_neutral_prior(1e-3).unwrap();

    let batched = BatchedSolver::new(&tree, &config, 4).unwrap();
    let result = batched
        .solve(&ObservationSequence::from_lists(items.clone()), true, true)
        .unwrap();
    assert_eq!(result.dof_values.len(), items.len());
    assert_eq!(result.jacobian.as_ref().unwrap().len(), items.len());

    let mut single = SingleFrameSolver::new(&tree, &config).unwrap();
    for (i, observations) in items.iter().enumerate() {
        let mut state = State::neutral_pose(Arc::clone(&tree));
        let expected = single.solve(&mut state, observations, true, true).unwrap();
        assert_eq!(result.dof_values[i], expected.state.dof_values);
        assert_eq!(result.root_quat[i], expected.state.root_quat);
        assert_eq!(
            &result.jacobian.as_ref().unwrap()[i],
            expected.jacobian.as_ref().unwrap()
        );
    }
}

#[test]
fn rejects_fixed_base_and_smoothness_priors() {
    let config = SolverConfig::default();
    assert!(BatchedSolver::new(&branching_tree(true), &config, 1).is_err());
    let mut smoothness = SolverConfig::default();
    smoothness.add_dof_smoothness_prior(1.0).unwrap();
    assert!(BatchedSolver::new(&branching_tree(false), &smoothness, 1).is_err());
    assert!(BatchedSolver::new(&branching_tree(false), &config, 0).is_err());
}
