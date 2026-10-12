#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use std::sync::Arc;

    use diplomat_runtime::DiplomatStr;

    use crate::error::ffi::Error;
    use crate::error::invalid_input;

    /// A body plan: joints (each with a keypoint) connected in a tree. Loaded
    /// from JSON (see `docs/getting-started/body-plan.md` for the format) and
    /// immutable afterwards. Solvers and states share it rather than copy it.
    #[diplomat::opaque]
    pub struct KinematicTree(pub(crate) Arc<quickik_core::body_plan::KinematicTree>);

    impl KinematicTree {
        /// Parses a body plan from a JSON string.
        pub fn from_json_str(json_str: &DiplomatStr) -> Result<Box<KinematicTree>, Box<Error>> {
            let json_str = std::str::from_utf8(json_str)
                .map_err(|e| invalid_input(format!("body plan is not valid UTF-8: {e}")))?;
            let tree = quickik_core::body_plan::KinematicTree::from_json_str(json_str)?;
            Ok(Box::new(KinematicTree(Arc::new(tree))))
        }

        /// Parses a body plan from a JSON file.
        pub fn from_json_file(path: &DiplomatStr) -> Result<Box<KinematicTree>, Box<Error>> {
            let path = std::str::from_utf8(path)
                .map_err(|e| invalid_input(format!("path is not valid UTF-8: {e}")))?;
            let tree = quickik_core::body_plan::KinematicTree::from_json_file(path)?;
            Ok(Box::new(KinematicTree(Arc::new(tree))))
        }

        pub fn n_joints(&self) -> usize {
            self.0.n_joints()
        }

        /// Total number of DOFs over all joints, excluding the root pose.
        pub fn n_dofs(&self) -> usize {
            self.0.n_dofs()
        }

        /// Number of state variables, i.e. columns of a Jacobian: `n_dofs()`,
        /// plus 6 (root position and rotation) for a free-floating tree.
        pub fn state_dim(&self) -> usize {
            self.0.state_dim()
        }

        /// Index of the joint named `name`, or none if there is no such joint.
        pub fn joint_index(&self, name: &DiplomatStr) -> Option<usize> {
            self.0.joint_index(std::str::from_utf8(name).ok()?)
        }
    }
}
