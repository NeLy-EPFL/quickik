#[diplomat::bridge]
#[diplomat::abi_rename = "quickik_{0}"]
pub mod ffi {
    use crate::body_plan::ffi::KinematicTree;
    use crate::error::ffi::Error;
    use crate::geometry::flatten;
    use crate::state::ffi::State;

    /// Buffers for `forward_kinematics`, which writes its results here.
    #[diplomat::opaque_mut]
    pub struct ForwardKinematicsWorkspace(
        pub(crate) quickik_core::forward::ForwardKinematicsWorkspace,
    );

    impl ForwardKinematicsWorkspace {
        pub fn create(tree: &KinematicTree) -> Box<ForwardKinematicsWorkspace> {
            let workspace = quickik_core::forward::ForwardKinematicsWorkspace::new(&tree.0);
            Box::new(ForwardKinematicsWorkspace(workspace))
        }

        /// `n_joints * 3`: world-frame keypoint positions from the last
        /// `forward_kinematics` call, in joint order.
        pub fn keypoint_positions<'a>(&'a self) -> &'a [f32] {
            flatten(self.0.keypoint_positions())
        }
    }

    /// Computes every keypoint's world position at `state` into `workspace`.
    /// Returns an error if `state` doesn't match the workspace's tree.
    pub fn forward_kinematics(
        workspace: &mut ForwardKinematicsWorkspace,
        state: &State,
    ) -> Result<(), Box<Error>> {
        Ok(quickik_core::forward::forward_kinematics(
            &mut workspace.0,
            &state.0,
        )?)
    }
}
