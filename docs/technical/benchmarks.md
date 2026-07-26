# Benchmarks

The following results are obtained using an 8-core (16-thread) Intel® Core™ i9-11900K processor. See [`benchmark/README.md`](https://github.com/NeLy-EPFL/quickik/blob/main/benchmark/README.md) to reproduce these results.

## Comparison with other libraries

The following plots show a comparison of QuickIK against three other whole-tree IK libraries – [KDL](https://github.com/orocos/orocos_kinematics_dynamics), [Pinocchio](https://github.com/stack-of-tasks/pinocchio), and [RBDL](https://github.com/rbdl/rbdl) – on two tasks:

- **[NeuroMechFly](https://neuromechfly.org/):** a biomechanical model of a fruit fly with 42 leg DOFs, driven by [experimental behavior recordings](https://go.epfl.ch/spotlight-poseforge).
- **[Unitree G1](https://www.unitree.com/g1):** a humanoid robot with 29 DOFs, driven by raw human walking data from the [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) dataset, rescaled onto G1's proportions. Essentially, we are using inverse kinematics as a means of solving the <abbr title="&quot;Transferring motion from a source body to a target robot with different physical dimensions, enabling teleoperation and human-to-robot skill transfer.&quot; Open Source Robotics, https://robotics.growbotics.ai/glossary/terms/motion-retargeting">motion retargeting</abbr> problem.

??? note "Scope of the comparison"
    Many popular inverse kinematics libraries solve a different problem: a single fixed-base kinematic chain reaching one end-effector target, rather than a free-floating base fit jointly against many keypoints across multiple limbs. Benchmarking them on this task would require either fixing the base and solving one limb at a time, or writing a custom, non-standard algorithm – either way, the result would no longer reflect the same problem. KDL, Pinocchio, and RBDL are included here because each can fit an arbitrary floating base against multiple simultaneous keypoints.

!!! info "Definition of metrics"
    - **Latency:** how long one cold-start `solve()` call takes against a fixed target. Reported both as the *mean* under each library's own early-stopping behavior (capped nonetheless at a fixed max iterations count), and as a *worst case* with early stopping disabled, so every solve runs the same fixed number of iterations regardless of how easy the target is.
    - **Throughput (single thread):** frames per second solving a long recording, each warm-started from the previous frame's solution.
    - **Throughput (multi-thread, 8 threads):** the same recording split across 8 threads and solved in parallel.

![Benchmark comparison across both the NeuroMechFly fly body and the Unitree G1 humanoid body](assets/benchmarks/comparison.svg){ width="480" }

??? note "Implementation notes"
    === "QuickIK (Python/C++/Rust)"
        All three of QuickIK's own bindings run the identical compiled Rust solver (the core `quickik` crate, same release build) – Python and C++ are thin FFI wrappers around it, not separate implementations. Python's throughput metrics batch a whole recording into one call via a numpy-array API instead of one Python object per keypoint per frame, which is what keeps it within a few percent of Rust and C++ here; an earlier, naive per-frame-object-list version of the same binding was roughly 2x slower on these metrics, since Python/PyO3 call and object-marshaling overhead then dominated the actual (fast) per-frame solve. Any remaining C++-vs-Rust gap in the chart is measurement noise from CPU scheduling on a shared machine (see `benchmark/README.md`'s "Reducing measurement noise" section), not a real difference, since both run the same code.

    === "RBDL"
        RBDL's native floating-base joint crashes its `InverseKinematicsConstraintSet` solver (an upstream dimension-mismatch bug), so QuickIK's floating base is represented as a translation plus Euler-angle joint in series instead, matching KDL's workaround for the same underlying reason. Its solver is a joint-space damped Levenberg-Marquardt normal-equations solve, tuned to match QuickIK's own iteration count and tolerance for a fair comparison. The Python bindings wrap the same native C++ solver (via Cython), so Python and C++ perform almost identically on latency and single-thread throughput; multi-thread throughput uses `multiprocessing` in Python versus in-process threads in C++, which accounts for the modest gap there.

    === "Pinocchio"
        Pinocchio provides forward kinematics, Jacobians, and configuration integration as building blocks, not a ready-made multi-keypoint `solve()` call – its own official IK tutorials are a hand-written Newton loop on top of those primitives (turnkey solvers like TSID are built on top of Pinocchio, not part of it). This benchmark's Gauss-Newton/Levenberg-Marquardt loop is hand-written the same way, matching QuickIK's solver as closely as possible (same residual formulation, damping, tolerance, and iteration cap). The Python benchmark pays Python-level overhead on every solver iteration (unlike RBDL's single-call-per-frame Cython wrapper), which is why a native C++ port exists and measures meaningfully faster – though still slower than RBDL's C++ numbers, mostly due to Pinocchio's per-keypoint Jacobian extraction API.

    === "KDL"
        KDL has no native 6-DOF floating joint or position-only solver, so QuickIK's floating base is represented as 6 scalar joints in series, and the 3 rotational rows of every endpoint's task-space weight matrix are zeroed so orientation error never drives the solve. KDL remains the slowest solver here even once its early-stopping is fixed to match QuickIK's own tolerance: its `TreeIkSolverVel_wdls` computes a dense SVD of the full weighted task-space Jacobian every iteration, where RBDL and QuickIK instead form and solve the much cheaper normal-equations matrix – an algorithmic difference between the libraries' building blocks, not a stopping-criterion artifact.

## Scaling

A separate <abbr title="Scaling the total workload proportionally to the number of workers to measure parallel computing performance. Under ideal scaling, the relative speed-up is the same as the number of workers. As we reach hardware limits, overhead dominates and the speed-up drops. By contrast, a &quot;strong scaling test&quot; keeps the total workload constant.">weak scaling test</abbr> (QuickIK's Rust API, using NeuroMechFly) measures how throughput grows as both thread count and total workload grow together.

![Speedup vs. worker threads](assets/benchmarks/scaling.svg)
