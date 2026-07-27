---
icon: lucide/house
---

# QuickIK

QuickIK is a fast <abbr title="Computing XYZ world coordinates of joint keypoints given joint angles. Can be solved analytically.">forward</abbr> and <abbr title="Fitting joint angles and base state needed to place joint keypoints at their observed place. Generally requires iterative optimization.">inverse</abbr> kinematics library. It provides high-level APIs for processing consecutive frames with warm starts and multi-threaded batch processing, as well as a low-level API for more specific use cases (e.g., real-time applications). QuickIK is written in Rust but comes with Python and C++ bindings.

## Feature highlights

- **Whole-body kinematics:** Traditionally, inverse kinematics solves for joint angles to match the position of only the end effector (e.g., foot). QuickIK finds the joint angles and <abbr title="The position and orientation of the root body link. Freely moving animals and robots have &quot;freely floating&quot; bases, while fixed robotic arms have fixed bases.">base state</abbr> that best matches many tracked keypoints, possibly spread across multiple kinematic chains (e.g., limbs), in a single solve.
- **Favoring "natural" poses:** QuickIK can be configured with a bias to favor more neutral joint angles when the problem is underconstrained.
- **Incomplete observation:** QuickIK allows some keypoints to be missing on some of the frames and does its best using only the available ones.
- **From 2D keypoint positions:** QuickIK can accept keypoint positions that are only in 2D projections. The problem is intrinsically underconstrained, but it can still work if the camera angle is reasonable and the pull toward "natural" states is properly tuned.
- **Differential weights:** When not all keypoint positions are equally reliable in the upstream MoCap data, QuickIK can consider them with different weights.
- **Very fast:** QuickIK is about [5 or more times faster](technical/benchmarks.md) than RBDL/Pinocchio.

## Example

The following video shows QuickIK's solution to two inverse kinematics tasks used in the [benchmarks](technical/benchmarks.md):

- **Biomechanics:** [Behavior recording](https://nely-epfl.github.io/spotlight-poseforge-paper/) of a fruit fly retargeted to the [NeuroMechFly](https://neuromechfly.org/) model
- **Robotics:** [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) walking kinematics retargeted to a [Unitree G1](https://www.unitree.com/g1) humanoid robot (greater mismatch expected due to larger difference between robotic and human bodies)

<video style="width: 100%" autoplay loop muted controls>
  <source src="https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/example_clips.mp4" type="video/mp4">
</video>

## What QuickIK does not do

- In some specific cases (typically robotic arms with fixed bases), inverse kinematics can be solved analytically. There are libraries that can solve these problems in constant time (given fixed body configuration), including [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) and [IKFast](https://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/ikfast/ikfast_tutorial.html).
- QuickIK does not solve _inverse dynamics_, the process of solving for forces and torques that need to be applied to achieve observed kinematics. For inverse dynamics, see libraries like [Pinocchio](https://stack-of-tasks.github.io/pinocchio/) and [RBDL](https://github.com/rbdl/rbdl). In the author's own pipeline, _imitation learning_ is used downstream of inverse kinematics instead of inverse dynamics.
