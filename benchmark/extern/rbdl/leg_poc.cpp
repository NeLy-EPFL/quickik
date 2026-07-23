// Minimal proof-of-concept: build the "lf" (left-front) leg chain from
// neuromechfly_ypr_legs.json using RBDL's native C++ API, run forward
// kinematics at the neutral pose, and then run RBDL's own built-in
// InverseKinematics() solver to reach a nearby target.
//
// Chain (parent -> child), matching the JSON body plan:
//   thorax (root, fixed at world origin)
//     -> lf_thorax_coxa       (3 dof: yaw=X, pitch=Y, roll=Z)
//       -> lf_coxa_trochanterfemur (2 dof: pitch=Y, roll=Z)
//         -> lf_trochanterfemur_tibia (1 dof: pitch=Y)
//           -> lf_tibia_tarsus (1 dof: pitch=Y)
//             -> lf_claw (fixed point, no dof; tracked as a body-fixed
//                point on the last moving body)
//
// Each named joint with N dofs is expanded into N chained 1-dof RBDL
// revolute joints: the first carries the joint's offset translation, the
// rest use an identity SpatialTransform (zero offset) since they act at
// the same point. This reproduces QuickIK's intrinsic-axis composition
// (see forward.rs: `rotation *= from_axis_angle(axis_local, angle)`
// applied sequentially), because each subsequent RBDL joint's axis is
// expressed in the frame already rotated by the previous dofs.

#include <cstdio>

#include <rbdl/rbdl.h>

using namespace RigidBodyDynamics;
using namespace RigidBodyDynamics::Math;

namespace {

// Adds a 1-dof revolute joint as a child of `parent_id`, with zero-mass
// dummy body, returning the new body id.
unsigned int AddRevoluteDof(Model& model, unsigned int parent_id,
                             const Vector3d& offset, const Vector3d& axis) {
  Joint joint(JointTypeRevolute, axis);
  Body body(0.0, Vector3d(0., 0., 0.), Vector3d(0., 0., 0.));
  SpatialTransform joint_frame(Matrix3d::Identity(), offset);
  return model.AddBody(parent_id, joint_frame, joint, body);
}

} // namespace

int main() {
  Model model;
  model.gravity = Vector3d(0., 0., 0.);

  // lf_thorax_coxa: offset from thorax (root, body id 0), 3 dofs.
  unsigned int b = 0;
  b = AddRevoluteDof(model, b, Vector3d(-0.161, 0.172, -0.23),
                     Vector3d(1., 0., 0.)); // yaw
  b = AddRevoluteDof(model, b, Vector3d(0., 0., 0.), Vector3d(0., 1., 0.)); // pitch
  b = AddRevoluteDof(model, b, Vector3d(0., 0., 0.), Vector3d(0., 0., 1.)); // roll

  // lf_coxa_trochanterfemur: 2 dofs.
  b = AddRevoluteDof(model, b, Vector3d(-0.00181, -0.00172, -0.365),
                     Vector3d(0., 1., 0.)); // pitch
  b = AddRevoluteDof(model, b, Vector3d(0., 0., 0.), Vector3d(0., 0., 1.)); // roll

  // lf_trochanterfemur_tibia: 1 dof.
  b = AddRevoluteDof(model, b, Vector3d(-0.00465, -0.00149, -0.705),
                     Vector3d(0., 1., 0.)); // pitch

  // lf_tibia_tarsus: 1 dof.
  b = AddRevoluteDof(model, b, Vector3d(-0.000144, -0.000952, -0.518),
                     Vector3d(0., 1., 0.)); // pitch

  const unsigned int tip_body_id = b;
  const Vector3d tip_local_offset(0.09203502219396992, 0.010583907142921345,
                                   -0.6712937582145876);

  printf("Model built: %d dofs, %d bodies (incl. root).\n", model.dof_count,
         model.mBodies.size());

  // --- Forward kinematics at the neutral (all-zero) pose ---
  VectorNd q_neutral = VectorNd::Zero(model.q_size);
  UpdateKinematicsCustom(model, &q_neutral, NULL, NULL);
  Vector3d tip_pos =
      CalcBodyToBaseCoordinates(model, q_neutral, tip_body_id, tip_local_offset, false);
  printf("lf_claw world position at neutral pose: [%.6f, %.6f, %.6f]\n",
         tip_pos[0], tip_pos[1], tip_pos[2]);

  // --- Inverse kinematics using RBDL's own built-in solver ---
  // Target: nudge the neutral-pose tip position by a few mm.
  Vector3d target = tip_pos + Vector3d(0.05, -0.02, 0.03);

  std::vector<unsigned int> body_ids{tip_body_id};
  std::vector<Vector3d> body_points{tip_local_offset};
  std::vector<Vector3d> target_positions{target};

  VectorNd q_init = VectorNd::Zero(model.q_size);
  VectorNd q_result;
  bool ok = InverseKinematics(model, q_init, body_ids, body_points,
                               target_positions, q_result,
                               /*step_tol=*/1.0e-10, /*lambda=*/0.01,
                               /*max_iter=*/100);

  printf("\nInverseKinematics() converged: %s\n", ok ? "true" : "false");
  printf("Target position:               [%.6f, %.6f, %.6f]\n", target[0],
         target[1], target[2]);

  UpdateKinematicsCustom(model, &q_result, NULL, NULL);
  Vector3d achieved =
      CalcBodyToBaseCoordinates(model, q_result, tip_body_id, tip_local_offset, false);
  printf("Achieved tip position:         [%.6f, %.6f, %.6f]\n", achieved[0],
         achieved[1], achieved[2]);
  printf("Position error norm: %.3e\n", (achieved - target).norm());

  printf("\nResulting joint angles (rad): ");
  for (unsigned int i = 0; i < q_result.size(); i++) {
    printf("%.4f ", q_result[i]);
  }
  printf("\n");

  return 0;
}
