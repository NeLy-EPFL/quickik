---
icon: lucide/rocket
---

# QuickIK

Fast inverse kinematics library, aimed for both high throughput and low latency.

QuickIK solves *whole-tree* inverse kinematics: given a robot or animal with a free-floating base (e.g. a pelvis or thorax not bolted to anything) and many tracked keypoints spread across multiple limbs, it finds the one root pose + joint-angle vector that best matches every keypoint's target position, jointly, in a single solve.

It provides high-level APIs for processing consecutive frames with warm starts and multi-threaded batch processing, as well as a low-level API for more specific use cases (e.g. real-time applications) – in Rust, with Python and C++ bindings.

## Example

QuickIK fitting real motion-capture recordings for the two bodies in the [benchmarks](benchmarks.md): NeuroMechFly (a fly) and a Unitree G1 humanoid.

<video controls style="width: 100%">
  <source src="https://datasets.epfl.ch/nely-public-share/quickik_assets/docs/example_clips.mp4" type="video/mp4">
</video>
