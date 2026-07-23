---
icon: lucide/house
---

# QuickIK

Fast inverse kinematics library, aimed for both high throughput and low latency.

QuickIK solves *whole-tree* inverse kinematics: given a robot or animal with a free-floating base (e.g. a pelvis or thorax not bolted to anything) and many tracked keypoints spread across multiple limbs, it finds the one root pose + joint-angle vector that best matches every keypoint's target position, jointly, in a single solve.

It provides high-level APIs for processing consecutive frames with warm starts and multi-threaded batch processing, as well as a low-level API for more specific use cases (e.g. real-time applications) – in Rust, with Python and C++ bindings.

## Example

The following video shows QuickIK's solution to two inverse kinematics tasks used in the [benchmarks](benchmarks.md):

- **Biomechanics:** [Behavior recording](https://nely-epfl.github.io/spotlight-poseforge-paper/) of a fruit fly retargeted to [NeuroMechFly](https://neuromechfly.org/)
- **Robotics:** [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) walking kinematics retargeted to a [Unitree G1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) humanoid robot (larger keypoint mismatch is expected due to greater difference between robotic and human bodies)

<video controls style="width: 100%">
  <source src="https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/example_clips.mp4" type="video/mp4">
</video>
