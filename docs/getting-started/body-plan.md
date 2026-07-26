# Body plan

A body plan describes the kinematic tree QuickIK solves against – a robot's joints, or an animal's skeleton.

In QuickIK, the kinematic tree is defined using **joints**. Each joint can contain multiple **degrees of freedom (DOFs)**. There are two types of DOFs: *hinges* that rotate, and *slides* that move translationally. For example, a 3-axis ball joint would have three hinge DOFs and no slide DOF.

All joints trace back to a single parent – the **root** of the kinematic tree. In principle, the definition of the root is arbitrary (you can say the whole human body stems from the left index fingertip if you wish), but practically it should be defined as a central body part like the pelvis or thorax. The root can be *freely floating* – useful when the body is a robot or an animal that can move around, or *fixed* – useful for fixed-base robotic arms.

In QuickIK, joints double as **keypoints**[^1] – the points on the body whose positions are recorded and used to constrain the state. If you need a keypoint in the middle of a body segment, add a "pseudo joint" where they keypoint is supposed to be with no associated DOFs.

[^1]:
    This is merely a practical choice, as articulated joints are usually easier to track than arbitrary points on the body and indeed they are what's typically available in MoCap/pose estimation data.

**Each joint has the following properties:**

- A **name**.
- A **parent joint** (the parent of the root joint is `null`).
- A **position offset** and a **rotation offset** from its parent joint, representing the properties of the rigid-body link connecting them. Rotation offsets are specified in quaternions in wxyz format.
- A **weight scaler** that controls the _scale_[^2] of how hard the solver should try to minimize the mismatch of this joint's position. If some joints are intrinsically harder to measure (e.g., if they are usually occluded or embedded in soft tissues), it's useful to lower this number.
- A list of **degrees of freedom** (DOFs). The order of the DOFs is important, as 3D rotations do not commute.

[^2]:
    During inverse kinematics, the user can also supply a weight for each keypoint on a frame-to-frame basis (e.g., using the uncertainty measure of the pose estimation model on that particular frame). The final weight is the product of the weight supplied at runtime and this scaler.

**Each DOF has the following properties:**

- A **type**: can be `hinge` or `slide`.
- An **axis**: this is the rotational axis for hinge joints or translational axis for slide joints.
- A **neutral** value: the solver favors poses that are closer to this is the "natural" rotation angle or slide position. For slide joints, the value is in radians; for hinge DOFs, the value is given in or whatever unit the joint's position offset is given in.
- A **weight scaler** controlling the _scale_[^3] of how strongly the solver favors the neutral state defined above.
- Optionally, the **limits** for the value of this DOF (angle for hinge joints, positions for slide joints). Same unit as the neutral value. Set to `null` if unbounded.

[^3]:
    Upon initiating the inverse kinematics solver, the user can define a weight for the pull toward the neutral pose. Like the weight scaler for the joint's weight, the final weight toward the neutral value for each DOF is the product of the weight supplied at runtime and the scaler specified here in the body plan.


## JSON body plan format

In QuickIK, the body plan is specified in JSON. An example JSON file is as follows.

!!! info "JSON schema"
    A formal schema of the JSON format is [available here](https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/bodyplan_20260726.schema.json). You can use it for formal [syntax check with you IDE](https://code.visualstudio.com/docs/languages/json).

```json
{
  "fixed_base": false,
  "joints": [
    {
      "name": "root",
      "parent": null,
      "offset_pos": [0.0, 0.0, 0.0],
      "offset_quat": [1.0, 0.0, 0.0, 0.0],
      "weight_scaler": 1.0,
      "dofs": []
    },
    {
      "name": "elbow",
      "parent": "root",
      "offset_pos": [1.0, 0.0, 0.0],
      "offset_quat": [1.0, 0.0, 0.0, 0.0],
      "weight_scaler": 1.0,
      "dofs": [
        {
          "type": "hinge",
          "axis": [0.0, 0.0, 1.0],
          "neutral": 0.0,
          "weight_scaler": 1.0,
          "limits": [-3.0, 3.0]
        }
      ]
    },
    {
      "name": "wrist",
      "parent": "elbow",
      "offset_pos": [1.0, 0.0, 0.0],
      "offset_quat": [1.0, 0.0, 0.0, 0.0],
      "weight_scaler": 1.0,
      "dofs": []
    }
  ],
  "x-anything": [
    "Keys starting with 'x-' are allowed at any level and are ignored.",
    "They can be of any type and are useful for custom metadata/documentation."
  ]
}
```