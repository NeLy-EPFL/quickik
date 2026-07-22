// Research proof-of-concept: build the LF (left-front) leg chain of the
// NeuroMechFly body plan (see benchmark/assets/neuromechfly_ypr_legs.json)
// as a raw KDL::Chain -- no URDF, no ROS -- and solve it with TRAC-IK.
//
// Chain: thorax -> lf_thorax_coxa (yaw, pitch, roll)
//               -> lf_coxa_trochanterfemur (pitch, roll)
//               -> lf_trochanterfemur_tibia (pitch)
//               -> lf_tibia_tarsus (pitch)
//               -> lf_claw (fixed tip, no DOF)
// Total actuated DOFs: 3 + 2 + 1 + 1 = 7.
//
// Each body-plan "joint" node becomes: one fixed (Joint::None) segment
// carrying its offset_pos translation, followed by one RotAxis segment per
// DOF (zero further translation between them, since they act at the same
// point). The per-DOF "neutral_angle" from the JSON is encoded as the KDL
// Joint offset, so actual angle = neutral_angle + q.

#include <cmath>
#include <iomanip>
#include <iostream>

#include "kdl/chain.hpp"
#include "kdl/chainfksolverpos_recursive.hpp"
#include "kdl/frames.hpp"
#include "kdl/jntarray.hpp"
#include "trac_ik/trac_ik.hpp"

namespace {

KDL::Segment FixedTranslate(const KDL::Vector & offset_pos)
{
  return KDL::Segment(KDL::Joint(KDL::Joint::None), KDL::Frame(offset_pos));
}

KDL::Segment RotDof(const KDL::Vector & axis, double neutral_angle)
{
  return KDL::Segment(
    KDL::Joint(KDL::Vector(0, 0, 0), axis, KDL::Joint::RotAxis, /*scale=*/1.0, neutral_angle),
    KDL::Frame::Identity());
}

}  // namespace

int main()
{
  KDL::Chain chain;

  // lf_thorax_coxa: parent thorax, offset_pos = [-0.161, 0.172, -0.23]
  chain.addSegment(FixedTranslate(KDL::Vector(-0.161, 0.172, -0.23)));
  chain.addSegment(RotDof(KDL::Vector(1, 0, 0), -0.29670597283903605));  // yaw
  chain.addSegment(RotDof(KDL::Vector(0, 1, 0), 0.33161255787892263));   // pitch
  chain.addSegment(RotDof(KDL::Vector(0, 0, 1), 0.5759586531581288));    // roll

  // lf_coxa_trochanterfemur: offset_pos = [-0.00181, -0.00172, -0.365]
  chain.addSegment(FixedTranslate(KDL::Vector(-0.00181, -0.00172, -0.365)));
  chain.addSegment(RotDof(KDL::Vector(0, 1, 0), -2.426007660272118));   // pitch
  chain.addSegment(RotDof(KDL::Vector(0, 0, 1), -0.2617993877991494));  // roll

  // lf_trochanterfemur_tibia: offset_pos = [-0.00465, -0.00149, -0.705]
  chain.addSegment(FixedTranslate(KDL::Vector(-0.00465, -0.00149, -0.705)));
  chain.addSegment(RotDof(KDL::Vector(0, 1, 0), 1.361356816555577));  // pitch

  // lf_tibia_tarsus: offset_pos = [-0.000144, -0.000952, -0.518]
  chain.addSegment(FixedTranslate(KDL::Vector(-0.000144, -0.000952, -0.518)));
  chain.addSegment(RotDof(KDL::Vector(0, 1, 0), -0.17453292519943295));  // pitch

  // lf_claw: fixed tip, offset_pos = [0.09203502219396992, 0.010583907142921345,
  // -0.6712937582145876]
  chain.addSegment(
    FixedTranslate(KDL::Vector(0.09203502219396992, 0.010583907142921345, -0.6712937582145876)));

  const unsigned int n_dof = chain.getNrOfJoints();
  std::cout << "Chain has " << chain.getNrOfSegments() << " segments, " << n_dof
            << " actuated DOFs\n";

  // Joint limits on the free variable q (before the neutral-angle offset).
  KDL::JntArray q_min(n_dof), q_max(n_dof);
  for (unsigned int i = 0; i < n_dof; ++i) {
    q_min(i) = -M_PI;
    q_max(i) = M_PI;
  }

  // Ground-truth joint configuration: pick a non-trivial pose and compute its
  // end-effector (claw) position via forward kinematics -- this becomes the
  // IK target, so we know a solution exists and can check recovery.
  KDL::JntArray q_true(n_dof);
  const double true_vals[7] = {0.1, -0.15, 0.2, -1.9, 0.05, 1.2, -0.1};
  for (unsigned int i = 0; i < n_dof; ++i) {
    q_true(i) = true_vals[i];
  }

  KDL::ChainFkSolverPos_recursive fk_solver(chain);
  KDL::Frame target_pose;
  if (fk_solver.JntToCart(q_true, target_pose) < 0) {
    std::cerr << "FK failed on ground-truth configuration\n";
    return 1;
  }

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Target claw position: [" << target_pose.p.x() << ", " << target_pose.p.y()
            << ", " << target_pose.p.z() << "]\n";

  // Seed IK from a different configuration (not the ground truth).
  KDL::JntArray q_seed(n_dof);
  for (unsigned int i = 0; i < n_dof; ++i) {
    q_seed(i) = 0.0;
  }

  TRAC_IK::TRAC_IK solver(chain, q_min, q_max, /*maxtime=*/0.01, /*eps=*/1e-5, TRAC_IK::Speed);

  KDL::JntArray q_result(n_dof);
  // Only constrain position (x, y, z); leave orientation unconstrained by
  // setting large rotational tolerance, since only the claw tip position is
  // a tracked keypoint in the body plan.
  KDL::Twist tol_bounds(KDL::Vector(0, 0, 0), KDL::Vector(1e6, 1e6, 1e6));
  int rc = solver.CartToJnt(q_seed, target_pose, q_result, tol_bounds);

  std::cout << "\nTRAC_IK::CartToJnt return code: " << rc << " (>=0 means success)\n\n";

  std::cout << "Recovered joint angles (q, i.e. before neutral-angle offset):\n";
  for (unsigned int i = 0; i < n_dof; ++i) {
    std::cout << "  q[" << i << "] = " << q_result(i) << "  (ground truth " << q_true(i) << ")\n";
  }

  KDL::Frame result_pose;
  fk_solver.JntToCart(q_result, result_pose);
  KDL::Vector err = target_pose.p - result_pose.p;
  std::cout << "\nRecovered claw position: [" << result_pose.p.x() << ", " << result_pose.p.y()
            << ", " << result_pose.p.z() << "]\n";
  std::cout << "Position error norm: " << err.Norm() << " m\n";

  return 0;
}
