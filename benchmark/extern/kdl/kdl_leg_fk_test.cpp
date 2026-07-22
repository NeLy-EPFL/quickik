// Minimal proof-of-concept: build the left-front (lf) leg chain of the
// neuromechfly_ypr_legs body plan (thorax -> lf_thorax_coxa (3 dof) ->
// lf_coxa_trochanterfemur (2 dof) -> lf_trochanterfemur_tibia (1 dof) ->
// lf_tibia_tarsus (1 dof) -> lf_claw (fixed tip)) as a KDL::Chain and run
// ChainFkSolverPos_recursive at all-zero joint angles.
//
// fastik's own FK convention (see fastik/src/forward.rs::evaluate_frame_at_joint):
//   origin   = parent.origin + parent.rotation * offset_pos
//   rotation = parent.rotation * offset_quat
//   for each dof in order: rotation *= AxisAngle(dof.axis_local, angle)
// i.e. translate once by offset_pos, then apply each dof as a rotation about
// its own axis expressed in the frame as rotated by all previous dofs of the
// SAME joint. This maps directly onto a sequence of KDL Segments:
//   Segment(Fixed,   f_tip = Frame(Rotation::Quaternion(offset_quat), offset_pos))
//   Segment(RotAxis(axis_1), f_tip = Identity)
//   Segment(RotAxis(axis_2), f_tip = Identity)
//   ...
// Because Segment::pose(q) = joint.pose(q) * f_tip, and every RotAxis segment
// here has an Identity f_tip and a joint origin at the local (0,0,0), the
// *position* of a joint's keypoint is unaffected by that joint's own dofs --
// exactly matching fastik's semantics where a joint's own rotation only
// re-orients the frame for its children, never displaces the joint itself.

#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>

#include <iostream>

using KDL::Chain;
using KDL::Joint;
using KDL::Segment;
using KDL::Frame;
using KDL::Rotation;
using KDL::Vector;

// Helper: append one JSON "joint" (a translation offset + a sequence of
// single-axis rotational dofs) as one Fixed segment + N RotAxis segments.
static void addJsonJoint(Chain& chain, const std::string& name_prefix,
                          const Vector& offset_pos,
                          const std::vector<Vector>& dof_axes) {
  chain.addSegment(
      Segment(name_prefix + "_offset", Joint(Joint::Fixed), Frame(offset_pos)));
  for (size_t i = 0; i < dof_axes.size(); ++i) {
    chain.addSegment(Segment(
        name_prefix + "_dof" + std::to_string(i),
        Joint(name_prefix + "_dof" + std::to_string(i), Vector::Zero(),
              dof_axes[i], Joint::RotAxis)));
  }
}

int main() {
  Chain chain;

  // lf_thorax_coxa: offset_pos [-0.161, 0.172, -0.23], dofs yaw(X) pitch(Y) roll(Z)
  addJsonJoint(chain, "lf_thorax_coxa", Vector(-0.161, 0.172, -0.23),
               {Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)});

  // lf_coxa_trochanterfemur: offset_pos [-0.00181, -0.00172, -0.365], dofs pitch(Y) roll(Z)
  addJsonJoint(chain, "lf_coxa_trochanterfemur",
               Vector(-0.00181, -0.00172, -0.365),
               {Vector(0, 1, 0), Vector(0, 0, 1)});

  // lf_trochanterfemur_tibia: offset_pos [-0.00465, -0.00149, -0.705], dof pitch(Y)
  addJsonJoint(chain, "lf_trochanterfemur_tibia",
               Vector(-0.00465, -0.00149, -0.705), {Vector(0, 1, 0)});

  // lf_tibia_tarsus: offset_pos [-0.000144, -0.000952, -0.518], dof pitch(Y)
  addJsonJoint(chain, "lf_tibia_tarsus", Vector(-0.000144, -0.000952, -0.518),
               {Vector(0, 1, 0)});

  // lf_claw: fixed tip, offset_pos [0.09203502219396992, 0.010583907142921345, -0.6712937582145876]
  chain.addSegment(Segment("lf_claw", Joint(Joint::Fixed),
                            Frame(Vector(0.09203502219396992,
                                          0.010583907142921345,
                                          -0.6712937582145876))));

  std::cout << "Chain has " << chain.getNrOfSegments() << " segments, "
            << chain.getNrOfJoints() << " movable joints (dofs).\n";

  KDL::ChainFkSolverPos_recursive fk_solver(chain);
  KDL::JntArray q(chain.getNrOfJoints());
  for (unsigned int i = 0; i < q.rows(); ++i) q(i) = 0.0;

  Frame tip_frame;
  int status = fk_solver.JntToCart(q, tip_frame);
  if (status < 0) {
    std::cerr << "FK failed with status " << status << "\n";
    return 1;
  }

  std::cout << "lf_claw tip position at q=0 (all dof angles zero): ("
            << tip_frame.p.x() << ", " << tip_frame.p.y() << ", "
            << tip_frame.p.z() << ")\n";
  std::cout << "Expected (hand-summed offset_pos from JSON): "
            << "(-0.07556897780603009, 0.17842190714292133, "
               "-2.4892937582145875)\n";
  return 0;
}
