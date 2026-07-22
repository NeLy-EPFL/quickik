# Preprocessing

Generates the G1 body plan and fixtures (in the same JSON schema as the fly's `../assets/neuromechfly_ypr_legs.json` / `../assets/fixtures.json` -- see `../scripts/generate_fixtures.py`'s docstring for that schema), so `plot_comparison.py` can compare all 6 whole-tree IK implementations against a second body: a Unitree G1 humanoid (29 DOF), driven by real motion capture retargeted from [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) onto G1 by the [LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) (numerical-optimization retargeting with foot-slip correction -- not a from-scratch geometric retarget, since every one of G1's 29 DOFs already has a real angle in that dataset).

## Fetch the raw inputs

Not committed (see `../../.gitignore`) -- fetch once into `../assets/g1_raw/`:

```sh
mkdir -p benchmark/assets/g1_raw
curl -sL -o benchmark/assets/g1_raw/g1_29dof.urdf \
    "https://datasets.epfl.ch/nely-public-share/quickik_assets/benchmark_data/unitree/g1_29dof.urdf"
curl -sL -o benchmark/assets/g1_raw/walk1_subject1.csv \
    "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset/resolve/main/g1/walk1_subject1.csv"
```

The CSV is `root_joint(x,y,z,qx,qy,qz,qw)` + 29 joint angles per row, 30 fps, in the joint order documented in that dataset's own README.md (which matches this URDF's own kinematic order exactly).

## Generate

```sh
cd benchmark/preprocessing
python g1_body_plan.py     # -> ../assets/g1_body_plan.json
uv run --with numpy --with scipy python g1_fixtures.py   # -> ../assets/fixtures_g1.json
```

`g1_body_plan.py` converts the URDF's 29 revolute joints into fastik's body-plan schema (one single-DOF node per URDF joint -- every one has a nonzero offset from its parent, unlike the fly's collocated multi-axis joints, so there's no grouping to do) plus 3 zero-DOF leaf keypoints (`head`, `left_hand`, `right_hand`, mirroring the fly's leaf "claw" nodes) taken from the URDF's own fixed joints. `g1_kinematics.py` is a from-scratch forward-kinematics reimplementation matching `src/forward.rs`'s exact algorithm (cross-checked against Pinocchio's own FK on the same URDF+angles), used both to generate `g1_fixtures.py`'s synthetic exact-fit frames and to turn the retargeted CSV's (root pose, joint angles) rows into `target_ego` keypoint positions.

The pelvis root always gets a `Missing` observation at solve time (same harness-side convention as the fly's `thorax` -- see `build_observations` in each of the 6 benchmark harnesses), even though the retargeted data has a real root track: root pose is inferred purely from the other 32 keypoints' targets, not given directly, keeping G1's code path identical to the fly's rather than adding a new one.

