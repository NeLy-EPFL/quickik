#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use std::sync::Arc;

    use nalgebra::{Quaternion, UnitQuaternion};

    use crate::body_plan::ffi::KinematicTree;
    use crate::error::ffi::Error;
    use crate::error::invalid_input;
    use crate::geometry::ffi::{Quat, Vec3};

    /// The pose being solved for: root position and rotation, plus the value
    /// of every DOF.
    #[diplomat::opaque_mut]
    pub struct State(pub(crate) quickik_core::state::State);

    impl State {
        /// Every DOF at its neutral value, root at the origin with identity
        /// rotation.
        pub fn neutral_pose(tree: &KinematicTree) -> Box<State> {
            Box::new(State(quickik_core::state::State::neutral_pose(Arc::clone(
                &tree.0,
            ))))
        }

        #[diplomat::attr(cpp, rename = "clone")]
        pub fn clone_state(&self) -> Box<State> {
            Box::new(State(self.0.clone()))
        }

        /// Position of the root joint in world coordinates.
        pub fn root_pos(&self) -> Vec3 {
            self.0.root_pos.into()
        }

        pub fn set_root_pos(&mut self, root_pos: Vec3) {
            self.0.root_pos = root_pos.into();
        }

        /// Rotation of the root joint in world coordinates.
        pub fn root_quat(&self) -> Quat {
            self.0.root_quat.into()
        }

        /// Normalizes `root_quat`.
        pub fn set_root_quat(&mut self, root_quat: Quat) {
            let Quat { w, x, y, z } = root_quat;
            self.0.root_quat = UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
        }

        /// Values of all DOFs (angles in radians for hinges, positions for
        /// slides), in the tree's joint order, then each joint's DOF order.
        pub fn dof_values<'a>(&'a self) -> &'a [f32] {
            &self.0.dof_values
        }

        /// Returns an error if `dof_values` doesn't have one value per DOF.
        pub fn set_dof_values(&mut self, dof_values: &[f32]) -> Result<(), Box<Error>> {
            let n_dofs = self.0.dof_values.len();
            if dof_values.len() != n_dofs {
                return Err(invalid_input(format!(
                    "got {} DOF values, but the tree has {n_dofs} DOFs",
                    dof_values.len()
                )));
            }
            self.0.dof_values.copy_from_slice(dof_values);
            Ok(())
        }
    }
}
