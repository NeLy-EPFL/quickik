# Preprocessing

Generates the G1 body plan and fixtures (`../assets/g1_body_plan.json`, `../assets/fixtures_g1.json`), in the same JSON schema as the fly's own assets (see `../scripts/generate_fixtures.py`'s docstring), so `plot_comparison.py` can benchmark against a second body: a Unitree G1 humanoid (29 DOF), driven by real human motion capture ([LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)) rescaled onto G1 -- see `g1_fixtures.py`'s module docstring for why this benchmark does that rescale itself rather than starting from a pre-retargeted dataset.

## Fetch the raw inputs

Not committed (see `../../.gitignore`) -- fetch once into `../assets/g1_raw/`:

```sh
mkdir -p benchmark/assets/g1_raw
curl -sL -o benchmark/assets/g1_raw/g1_29dof.urdf \
    "https://datasets.epfl.ch/nely-public-share/quickik_assets/benchmark_data/unitree/g1_29dof.urdf"
curl -sL -o benchmark/assets/g1_raw/walk1_subject1.bvh \
    "https://datasets.epfl.ch/nely-public-share/quickik_assets/benchmark_data/lafan1/walk1_subject1.bvh"
```

## Generate

```sh
cd benchmark/preprocessing
python g1_body_plan.py     # -> ../assets/g1_body_plan.json
uv run --with numpy --with scipy python g1_fixtures.py   # -> ../assets/fixtures_g1.json
```

`g1_body_plan.py` converts the URDF's 29 revolute joints into QuickIK's body-plan schema (one single-DOF node per URDF joint) plus 3 zero-DOF leaf keypoints (`head`, `left_hand`, `right_hand`, mirroring the fly's leaf "claw" nodes) taken from the URDF's own fixed joints. `g1_kinematics.py` is a from-scratch forward-kinematics reimplementation matching `src/forward.rs`'s exact algorithm (cross-checked against Pinocchio's own FK on the same URDF+angles), used to generate `g1_fixtures.py`'s synthetic exact-fit frames. `lafan1_bvh.py` is a from-scratch BVH parser and FK evaluator for the raw motion capture skeleton, used by `g1_fixtures.py` to turn it into `target_ego` keypoint positions.

The pelvis root always gets a `Missing` observation at solve time (same harness-side convention as the fly's `thorax` -- see `build_observations` in each of the 6 benchmark harnesses): root pose is inferred purely from the other 32 keypoints' targets, keeping G1's code path identical to the fly's rather than adding a new one.
