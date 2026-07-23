"""Proof-of-concept: build the thorax + one leg chain from
neuromechfly_ypr_legs.json in Pinocchio, using a free-flyer root joint plus a
chain of single-DOF revolute joints (mirroring QuickIK's per-joint DOF list),
and compute forward kinematics at zero joint angles.

This is a feasibility check only, not part of the final benchmark.
"""

import json

import numpy as np
import pinocchio as pin

MODEL_JSON = (
    "/home/sibwang/Projects/quickik/benchmark/assets/neuromechfly_ypr_legs.json"
)

AXIS_MAP = {
    (1.0, 0.0, 0.0): pin.JointModelRX(),
    (-1.0, 0.0, 0.0): pin.JointModelRX(),  # sign handled via axis flip below
    (0.0, 1.0, 0.0): pin.JointModelRY(),
    (0.0, 0.0, 1.0): pin.JointModelRZ(),
}


def build_leg_model(joints: list[dict], leg_prefix: str) -> tuple[pin.Model, int]:
    """Build a Pinocchio model with a free-flyer thorax and one leg chain.

    Returns the model and the frame id of the leg's tip ("claw") keypoint.
    """
    model = pin.Model()

    # thorax: free-flyer root, attached to the universe (joint 0).
    thorax_id = model.addJoint(
        0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "thorax"
    )
    model.appendBodyToJoint(thorax_id, pin.Inertia.Zero(), pin.SE3.Identity())

    joints_by_name = {j["name"]: j for j in joints}
    chain_names = [
        j["name"]
        for j in joints
        if j["name"] == "thorax" or j["name"].startswith(leg_prefix)
    ]

    parent_joint_id = {"thorax": thorax_id}
    tip_frame_id = None

    for name in chain_names:
        if name == "thorax":
            continue
        node = joints_by_name[name]
        parent_id = parent_joint_id[node["parent"]]
        offset = pin.SE3(np.eye(3), np.array(node["offset_pos"], dtype=float))

        if not node["dofs"]:
            # Leaf keypoint with no DOFs (e.g. the claw tip): add as a fixed
            # operational frame rather than a joint.
            frame = pin.Frame(name, parent_id, 0, offset, pin.FrameType.OP_FRAME)
            tip_frame_id = model.addFrame(frame)
            continue

        # Chain one single-DOF revolute joint per scalar DOF. Only the first
        # joint in the group carries the translational offset from the
        # parent keypoint; subsequent joints in the same group are collocated
        # (identity placement), as they represent a composite rotation at the
        # same physical point.
        current_parent = parent_id
        placement = offset
        last_joint_id = None
        for dof in node["dofs"]:
            axis = np.array(dof["axis"], dtype=float)
            # Pinocchio's RX/RY/RZ are all +axis; QuickIK allows signed axes
            # for mirrored left/right legs, so each sign gets its own case.
            if tuple(axis) == (-1.0, 0.0, 0.0):
                joint_model = pin.JointModelRX()
                flip = -1.0
            elif tuple(axis) == (1.0, 0.0, 0.0):
                joint_model = pin.JointModelRX()
                flip = 1.0
            elif tuple(axis) == (0.0, 1.0, 0.0):
                joint_model = pin.JointModelRY()
                flip = 1.0
            elif tuple(axis) == (0.0, 0.0, -1.0):
                joint_model = pin.JointModelRZ()
                flip = -1.0
            elif tuple(axis) == (0.0, 0.0, 1.0):
                joint_model = pin.JointModelRZ()
                flip = 1.0
            else:
                raise ValueError(f"Unsupported axis {axis} for dof {dof['name']}")
            del flip  # not needed at zero angle for this POC; see report note.

            joint_id = model.addJoint(
                current_parent, joint_model, placement, dof["name"]
            )
            model.appendBodyToJoint(joint_id, pin.Inertia.Zero(), pin.SE3.Identity())
            current_parent = joint_id
            placement = pin.SE3.Identity()
            last_joint_id = joint_id

        parent_joint_id[name] = last_joint_id

    assert tip_frame_id is not None, f"no claw frame found for leg {leg_prefix}"
    return model, tip_frame_id


def main() -> None:
    with open(MODEL_JSON) as f:
        body_plan = json.load(f)
    joints = body_plan["joints"]

    model, tip_frame_id = build_leg_model(joints, leg_prefix="lf_")
    data = model.createData()

    print(f"Model nq={model.nq}, nv={model.nv}, njoints={model.njoints}")
    print("Joint names:", list(model.names))

    q = pin.neutral(model)  # zero joint angles, identity floating base pose
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    tip_pos = data.oMf[tip_frame_id].translation
    print(f"lf_claw tip position at q=0: {tip_pos}")


if __name__ == "__main__":
    main()
