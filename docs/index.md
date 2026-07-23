---
icon: lucide/rocket
---

# FastIK

Fast inverse kinematics library, aimed for both high throughput and low latency.

FastIK solves *whole-tree* inverse kinematics: given a robot or animal with a free-floating base (e.g. a pelvis or thorax not bolted to anything) and many tracked keypoints spread across multiple limbs, it finds the one root pose + joint-angle vector that best matches every keypoint's target position, jointly, in a single solve.

It provides high-level APIs for processing consecutive frames with warm starts and multi-threaded batch processing, as well as a low-level API for more specific use cases (e.g. real-time applications) – in Rust, with Python and C++ bindings.