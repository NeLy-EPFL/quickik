# Benchmarks

The following results are obtained usings an 8-core (16-thread) Intel® Core™ i9-11900K processor. See [`benchmark/README.md`](https://github.com/NeLy-EPFL/quickik/blob/main/benchmark/README.md) to reproduce these numbers.

## Comparison with other libraries

QuickIK is compared against three other whole-tree IK libraries – [KDL](https://github.com/orocos/orocos_kinematics_dynamics), [Pinocchio](https://github.com/stack-of-tasks/pinocchio), and [RBDL](https://github.com/rbdl/rbdl) – on the same task, across QuickIK's Rust API, Python bindings, and C++ bindings. Two inverse kinematics tasks are benchmarked: [**NeuroMechFly**](https://neuromechfly.org/) (biomechanical model of a fruit fly with 42 leg DOFs with real behavior recording data), and **G1** (a Unitree humanoid with 29 DOFs, driven by raw human walking data from the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) dataset, rescaled onto G1's proportions – retargeting a human skeleton onto a robot is itself an IK problem, so each compared solver does that retarget too, rather than starting from a pre-retargeted dataset; expect a real, nonzero fit residual, same as the fly's own mocap).

??? note "Other inverse kinematics libraries and why they are not included"
    Those libraries solve a categorically easier problem: a single fixed-base kinematic chain reaching one end-effector target, not a free-floating base fit jointly against many keypoints across multiple limbs. There's no way to run them "the same task, just slower" without either bolting the base down and solving one limb at a time (losing the floating base and the joint multi-limb fit) or writing a different, non-standard algorithm. Benchmarking them would measure a different, strictly smaller problem, flattering every excluded library rather than QuickIK. KDL, Pinocchio, and RBDL are included because each one can fit an arbitrary floating base against multiple simultaneous keypoints.

!!! info "Definition of metrics"
    - **Latency**: how long one cold-start `solve()` call takes against a fixed target. Reported both as the *mean* under each library's own early-stopping behavior, and as a *worst case* with early stopping disabled, so every solve runs the same fixed number of iterations regardless of how easy the target is.
    - **Throughput (single thread)**: frames per second solving a long recording one frame at a time, each warm-started from the previous frame's solution.
    - **Throughput (multi-thread, 8 threads)**: the same recording split across 8 threads and solved in parallel, for libraries/bindings that support it.

![Benchmark comparison across both the NeuroMechFly fly body and the Unitree G1 humanoid body](assets/benchmarks/comparison.svg)


??? Note "Implementation notes on external benchmarked libraries"
    === "RBDL"
        RBDL's native floating-base joint crashes its `InverseKinematicsConstraintSet` solver (an upstream dimension-mismatch bug), so QuickIK's floating base is represented as a translation plus Euler-angle joint in series instead, matching KDL's workaround for the same underlying reason. Its solver is a joint-space damped Levenberg-Marquardt normal-equations solve, tuned to match QuickIK's own iteration count and tolerance for a fair comparison. Python bindings wrap the same native C++ solver (via Cython), so Python and C++ perform almost identically on latency and single-thread throughput; multi-thread throughput uses `multiprocessing` in Python versus in-process threads in C++, which accounts for the modest gap there.

    === "Pinocchio"
        Pinocchio provides forward kinematics, Jacobians, and configuration integration as building blocks, not a ready-made multi-keypoint solve() call – its own official IK tutorials are a hand-written Newton loop on top of those primitives (turnkey IK solvers like TSID are built on top of Pinocchio, not part of it). This benchmark's Gauss-Newton/Levenberg-Marquardt loop is hand-written the same way, matching QuickIK's solver as closely as possible (same residual formulation, damping, tolerance, and iteration cap). The Python benchmark pays Python-level overhead on every solver iteration (unlike RBDL's single-call-per-frame Cython wrapper), which is why a native C++ port exists and measures meaningfully faster – though still slower than RBDL's C++ numbers, mostly due to Pinocchio's per-keypoint Jacobian extraction API.

    === "KDL"
        KDL has no native 6-DOF floating joint or position-only solver, so QuickIK's floating base is represented as 6 scalar joints in series, and the 3 rotational rows of every endpoint's task-space weight matrix are zeroed so orientation error never drives the solve. KDL remains the slowest solver here even once its early-stopping is fixed to match QuickIK's own tolerance: its `TreeIkSolverVel_wdls` computes a dense SVD of the full weighted task-space Jacobian every iteration, where RBDL and QuickIK instead form and solve the much cheaper normal-equations matrix – an algorithmic difference between the libraries' building blocks, not a stopping-criterion artifact.

## Scaling

A separate [weak-scaling test](https://hpc-wiki.info/hpc/Scaling#Weak_Scaling) (QuickIK's Rust API, using NeuroMechFly) measures how throughput grows as both thread count and total workload grow together.

![Speedup vs. worker threads](assets/benchmarks/scaling.svg)
