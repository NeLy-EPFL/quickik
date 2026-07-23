"""Renders a single side-by-side comparison video (NeuroMechFly left, G1
right), overlaying real motion-capture keypoints against QuickIK's inverse-
kinematics-fitted skeleton, frame by frame over `native_rate_frames` (see
`../scripts/generate_fixtures.py` / `../preprocessing/g1_fixtures.py`'s
module docstrings -- a contiguous run of consecutive recorded frames, the
same warm-start sequence the throughput benchmark solves, so the fit shown
here is a genuine continuous-tracking result, not a slideshow of
independently solved poses).

Solves each body's whole sequence once with `quickik.SequenceSolver` (warm-
started frame to frame, exactly like `../quickik_python/bench.py`'s
`bench_solve_sequence`), then re-derives every joint's world position from
the solved state via an independent from-JSON forward-kinematics replica
(same technique as that script's own `forward_kinematics`, extended here to
return every node -- including the root -- rather than just the tracked
keypoints, since the video needs the whole skeleton's bones, not just the
points fed to the solver).

Each panel plays at its own `playback_speed` (an on-screen label shows
which) -- the fly's is slowed to 0.1x since its legs move too fast to follow
at native rate; G1 plays at native (1x) speed. The two panels share one
output timeline; since they cover different real durations at these speeds,
the video ends as soon as either panel's own sequence runs out.

Writes `results/example_clips.mp4` (requires ffmpeg on PATH).

Usage (with devtools-pyenv/'s shared venv active, plus QuickIK's Python
extension built for that same interpreter; see `../quickik_python/bench.py`'s
own docstring):

    python render_video.py
"""

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import quickik
from matplotlib import animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.spatial.transform import Rotation as R

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = BENCHMARK_DIR / "assets"
OUT_DIR = Path(__file__).resolve().parent / "results"
FONTS_DIR = Path(__file__).resolve().parent / "fonts"

MOCAP_COLOR = "gray"
FIT_COLOR = "#4051b5"

DPI = 112.5

# Per-body asset paths, display title, native mocap frame rate (for real-time
# playback -- see each fixture generator's own NATIVE_RATE_LENGTH comment), a
# 3D view angle chosen to show the body's motion clearly, a playback speed
# (multiple of real time), and an axis-limit padding factor (how much slack
# past the tightest bounding box that still holds every frame).
BODIES = {
    "neuromechfly": {
        "body_plan": "neuromechfly_ypr_legs.json",
        "fixtures": "fixtures.json",
        "title": "NeuroMechFly",
        "fps": 330.0,
        "view": {"elev": 20, "azim": -60},
        "playback_speed": 0.1,
        "padding": 1.02,
        "up_reference": None,
        "missing_keypoints": [],
        "weight": None,  # library default (SolverConfig's own)
    },
    "g1": {
        "body_plan": "g1_body_plan.json",
        "fixtures": "fixtures_g1.json",
        "title": "G1 humanoid robot",
        "fps": 30.0,
        "view": {"elev": 12, "azim": -60},
        "playback_speed": 1.0,
        "padding": 1.1,
        # The raw LAFAN1 BVH's "Hips" joint's rest-pose local axes aren't
        # aligned with world axes the way `g1_fixtures.py`'s `ego()` implicitly
        # assumes (it treats the Hips rotation channel as a pure heading/yaw,
        # which this rig's rest pose isn't) -- so the fixtures' "up" ends up
        # along a tilted direction rather than the Z axis. This is a
        # coordinate-frame labeling quirk of how the fixtures were generated,
        # not a solve-correctness issue (whole-tree IK is rotation-invariant),
        # so it's corrected here for display only: `up_reference` names the
        # keypoints used to empirically measure "up" (head above the ankles'
        # midpoint), see `up_alignment_rotation`.
        "up_reference": {
            "head": "head",
            "ankles": ["left_ankle_roll", "right_ankle_roll"],
        },
        # LAFAN1 has one raw mocap landmark per hand ("LeftHand"/"RightHand"),
        # but G1 splits the wrist into 3 single-axis DOFs plus the hand
        # keypoint itself -- all 4 get that same single target position (see
        # g1_fixtures.py's BVH_TO_G1), even though they're a few cm apart in
        # the body plan's own real geometry. Solving all 4 as independent
        # observations fights itself two ways: the 3 wrist DOFs can't
        # actually be distinguished by one shared point (many equally
        # "valid" angle decompositions, so the fit jumps between them frame
        # to frame), and every one of them competes with the hand -- the
        # true end effector -- for the exact same target, degrading its own
        # fit for no benefit (wrist_roll/pitch/yaw have no keypoint of their
        # own worth matching; only the hand's position is real signal).
        # Marking the wrist DOFs' own keypoints Missing leaves the hand as
        # the sole target for this whole sub-chain, letting it actually
        # reach that target instead of splitting the difference, while the
        # neutral-pose prior picks a single stable wrist angle instead of
        # chasing noise.
        "missing_keypoints": [
            "left_wrist_roll",
            "left_wrist_pitch",
            "left_wrist_yaw",
            "right_wrist_roll",
            "right_wrist_pitch",
            "right_wrist_yaw",
        ],
        # With the wrist DOFs unobserved, they're still indirectly driven by
        # the hand's own position residual (they're its ancestors in the
        # chain), and the whole shoulder+elbow+wrist assembly is a redundant
        # manipulator for that single 3D target -- with only
        # SolverConfig::default()'s weak 1e-3 weight pulling toward neutral,
        # the solver was consistently using the wrist as free self-motion to
        # (partly) compensate for the arm's coarse rescale mismatch, landing
        # on a stable but visually wrong ~50-degree wrist bend every frame.
        # 10x the default weight was enough to mostly stop that trade --
        # empirically it also *improves* every other keypoint's fit
        # (including two 30+cm outlier residuals on the unweighted default),
        # so this isn't accuracy traded away for looks.
        "weight": 0.01,
    },
}


def register_fonts():
    """Registers the locally-fetched Open Sans TTFs with matplotlib's font
    manager, if present (same fonts/ directory and fetch steps as
    `plot_comparison.py`'s own `register_fonts`), falling back to matplotlib's
    bundled DejaVu Sans otherwise."""
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    ttfs = list(FONTS_DIR.glob("OpenSans-*.ttf"))
    if not ttfs:
        return ["DejaVu Sans"]
    for ttf in ttfs:
        fm.fontManager.addfont(str(ttf))
    return ["Open Sans"]


def load_body(name):
    cfg = BODIES[name]
    body_plan = json.loads((ASSETS_DIR / cfg["body_plan"]).read_text())
    fixtures = json.loads((ASSETS_DIR / cfg["fixtures"]).read_text())
    joints = body_plan["joints"]
    leg_joint_names = fixtures["leg_joint_names"]
    assert [j["name"] for j in joints][1:] == leg_joint_names

    dof_offsets = {}
    cursor = 0
    for j in joints:
        dof_offsets[j["name"]] = cursor
        cursor += len(j["dofs"])

    tree = quickik.KinematicTree.from_json_file(str(ASSETS_DIR / cfg["body_plan"]))
    # (parent, child) node-name pairs to draw as bones -- every joint except
    # the root, which has no parent of its own.
    edges = [(j["parent"], j["name"]) for j in joints if j["parent"] is not None]
    return cfg, joints, dof_offsets, tree, fixtures, edges


def forward_kinematics_full(joints, dof_offsets, dof_angles, root_pos, root_rot_wxyz):
    """Every body-plan node's world position (including the root), from a
    solved state -- an independent from-JSON FK replica, does not call into
    QuickIK at all (same role as `../quickik_python/bench.py`'s own
    `forward_kinematics`, extended to return the whole tree rather than just
    the non-root keypoints, since the video needs every bone)."""
    w, x, y, z = root_rot_wxyz
    world_pos, world_rot = {}, {}
    for j in joints:
        name, parent = j["name"], j["parent"]
        if parent is None:
            origin, rot = np.array(root_pos, dtype=float), R.from_quat([x, y, z, w])
        else:
            p_origin, p_rot = world_pos[parent], world_rot[parent]
            origin = p_origin + p_rot.apply(j["offset_pos"])
            qw, qx, qy, qz = j["offset_quat"]
            rot = p_rot * R.from_quat([qx, qy, qz, qw])
        dof_start = dof_offsets[name]
        for i, dof in enumerate(j["dofs"]):
            axis = np.array(dof["axis"])
            rot = rot * R.from_rotvec(axis * dof_angles[dof_start + i])
        world_pos[name], world_rot[name] = origin, rot
    return world_pos


def build_observations(target_ego, missing_indices=frozenset()):
    obs = [quickik.KeypointObservation.missing()]
    for i, p in enumerate(target_ego):
        obs.append(
            quickik.KeypointObservation.missing()
            if i in missing_indices
            else quickik.KeypointObservation.position_3d(list(p), 1.0)
        )
    return obs


def solve_sequence(tree, fixtures, missing_keypoints=(), weight=None):
    """Warm-started solve over `native_rate_frames`, exactly like the
    throughput benchmark (see `../quickik_python/bench.py`'s
    `bench_solve_sequence`) -- returns one solved `State` per frame.
    `missing_keypoints` (body-plan joint names) are given a `Missing`
    observation instead of their fixture target every frame; see `BODIES`'
    own `missing_keypoints` comment for why G1 needs this. `weight`
    overrides `SolverConfig`'s own default when given; see `BODIES`' own
    comment for why G1 needs a stronger one."""
    leg_joint_names = fixtures["leg_joint_names"]
    missing_indices = {leg_joint_names.index(name) for name in missing_keypoints}
    config = (
        quickik.SolverConfig()
        if weight is None
        else quickik.SolverConfig(weight=weight)
    )
    seq = quickik.SequenceSolver(tree, config)
    return [
        seq.solve_frame(build_observations(f["target_ego"], missing_indices))
        for f in fixtures["native_rate_frames"]
    ]


def up_alignment_rotation(fitted_frames, full_name_idx, up_reference):
    """A fixed rotation that maps this body's empirical "up" direction --
    averaged, over every frame, head position minus the ankles' midpoint --
    onto the plot's Z axis. See `up_reference`'s definition in `BODIES` for
    why this is needed for G1."""
    head_idx = full_name_idx[up_reference["head"]]
    ankle_idxs = [full_name_idx[n] for n in up_reference["ankles"]]
    up_vecs = [
        frame[head_idx] - frame[ankle_idxs].mean(axis=0) for frame in fitted_frames
    ]
    up_dir = np.mean(up_vecs, axis=0)
    up_dir /= np.linalg.norm(up_dir)
    rotation, _rmsd = R.align_vectors([[0.0, 0.0, 1.0]], [up_dir])
    return rotation


def axis_limits(*point_sets, padding=1.1):
    """A single fixed, equal-aspect bounding box covering every frame in
    every point set, so the camera never zooms or pans mid-video."""
    all_pts = np.concatenate(
        [np.concatenate(pts, axis=0) for pts in point_sets], axis=0
    )
    center = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    half_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * padding
    return center - half_range, center + half_range


def prepare_body(name):
    """Loads, solves, and derives every per-frame array this body's panel
    needs to plot -- keypoints and bones, already chase-cammed (recentered on
    the fitted root every frame) and, for G1, up-realigned (see
    `up_alignment_rotation`)."""
    cfg, joints, dof_offsets, tree, fixtures, edges = load_body(name)
    states = solve_sequence(tree, fixtures, cfg["missing_keypoints"], cfg["weight"])
    native_frames = fixtures["native_rate_frames"]
    full_names = [j["name"] for j in joints]
    full_name_idx = {n: i for i, n in enumerate(full_names)}

    fitted_positions = [
        forward_kinematics_full(
            joints, dof_offsets, s.dof_angles, s.root_pos, s.root_rot
        )
        for s in states
    ]
    # Chase-cam: recenter every frame on the fitted root position. G1's root
    # genuinely translates meters across the whole clip (a walking human), so
    # a fixed-camera box wide enough to hold the whole path would shrink the
    # body to a speck; subtracting the same per-frame offset from both point
    # sets keeps their relative alignment (the actual thing being compared)
    # untouched.
    root_positions = [np.array(s.root_pos) for s in states]
    mocap_frames = [
        np.array(f["target_ego"]) - root_pos
        for f, root_pos in zip(native_frames, root_positions, strict=True)
    ]
    fitted_frames = [
        np.array(list(p.values())) - root_pos
        for p, root_pos in zip(fitted_positions, root_positions, strict=True)
    ]
    fitted_bones = [
        np.array([[pos[parent], pos[child]] for parent, child in edges]) - root_pos
        for pos, root_pos in zip(fitted_positions, root_positions, strict=True)
    ]

    if cfg["up_reference"] is not None:
        align = up_alignment_rotation(fitted_frames, full_name_idx, cfg["up_reference"])
        mocap_frames = [align.apply(f) for f in mocap_frames]
        fitted_frames = [align.apply(f) for f in fitted_frames]
        fitted_bones = [
            align.apply(b.reshape(-1, 3)).reshape(b.shape) for b in fitted_bones
        ]

    lo, hi = axis_limits(mocap_frames, fitted_frames, padding=cfg["padding"])

    # Don't render mocap markers for keypoints the solver was told to
    # ignore (see `missing_keypoints`): their target is a redundant
    # duplicate of another keypoint's, not a real observation, so plotting
    # it next to a QuickIK fit that correctly ignores it just looks like an
    # unexplained extra bend sprouting from a single mocap dot.
    if cfg["missing_keypoints"]:
        missing_idx = [
            fixtures["leg_joint_names"].index(name) for name in cfg["missing_keypoints"]
        ]
        for frame in mocap_frames:
            frame[missing_idx] = np.nan

    display_fps = cfg["fps"] * cfg["playback_speed"]
    return {
        "cfg": cfg,
        "mocap_frames": mocap_frames,
        "fitted_frames": fitted_frames,
        "fitted_bones": fitted_bones,
        "lo": lo,
        "hi": hi,
        "display_fps": display_fps,
        "n_frames": len(states),
    }


def setup_panel(ax, body, show_legend):
    cfg = body["cfg"]
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(body["lo"][0], body["hi"][0])
    ax.set_ylim(body["lo"][1], body["hi"][1])
    ax.set_zlim(body["lo"][2], body["hi"][2])
    ax.view_init(**cfg["view"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(cfg["title"])

    mocap_scatter = ax.scatter([], [], [], c=MOCAP_COLOR, s=25, depthshade=False)
    fit_scatter = ax.scatter([], [], [], c=FIT_COLOR, s=15, depthshade=False)
    # add_collection3d computes its own axis bounds from the initial segments,
    # so it must be seeded with real data (frame 0), not an empty list.
    fit_bones = Line3DCollection(
        body["fitted_bones"][0], colors=FIT_COLOR, linewidths=1.5
    )
    ax.add_collection3d(fit_bones)
    if show_legend:
        # Proxy handles, not the scatters' own auto-generated legend entries:
        # a scatter-only handle would draw "QuickIK fit" as a bare dot, hiding
        # that it's also a connected skeleton (the bones have no label of
        # their own to contribute a line to the legend).
        mocap_handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=MOCAP_COLOR,
            label="MoCap keypoints",
        )
        fit_handle = Line2D(
            [0], [0], marker="o", linestyle="-", color=FIT_COLOR, label="QuickIK fit"
        )
        ax.legend(handles=[mocap_handle, fit_handle], loc="upper left", frameon=False)
    # text2D pins this to the axes' own 2D display space (top right corner),
    # unaffected by the 3D view/rotation -- unlike ax.text, which would place
    # it at a fixed *data* point instead.
    ax.text2D(
        0.98,
        0.98,
        f"{cfg['playback_speed']:g}x speed",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    return mocap_scatter, fit_scatter, fit_bones


def render_comparison():
    bodies = [prepare_body(name) for name in ("neuromechfly", "g1")]

    # Shared output timeline: sampled at the faster of the two panels' own
    # display rates, so neither panel loses temporal resolution; the slower
    # panel just occasionally holds a frame for two output ticks. Stops as
    # soon as either panel's own sequence runs out, rather than looping or
    # freezing the finished one -- with different speeds and native lengths,
    # that's normally the fly's 0.1x-slowed sequence.
    output_fps = max(b["display_fps"] for b in bodies)
    duration = min(b["n_frames"] / b["display_fps"] for b in bodies)
    total_output_frames = int(duration * output_fps)

    plt.rcParams["font.family"] = register_fonts()

    fig = plt.figure(figsize=(9.6, 5.2), frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02, wspace=0.02)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(len(bodies))]
    panels = [
        setup_panel(ax, body, show_legend=(i == 0))
        for i, (ax, body) in enumerate(zip(axes, bodies, strict=True))
    ]

    def update(k):
        t = k / output_fps
        artists = []
        for (mocap_scatter, fit_scatter, fit_bones), body in zip(
            panels, bodies, strict=True
        ):
            idx = min(int(t * body["display_fps"]), body["n_frames"] - 1)
            mocap_scatter._offsets3d = tuple(body["mocap_frames"][idx].T)
            fit_scatter._offsets3d = tuple(body["fitted_frames"][idx].T)
            fit_bones.set_segments(body["fitted_bones"][idx])
            artists += [mocap_scatter, fit_scatter, fit_bones]
        return artists

    anim = animation.FuncAnimation(fig, update, frames=total_output_frames, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "example_clips.mp4"
    # matplotlib's default FFMpegWriter settings pick a fairly high (lossy)
    # compression ratio, which shows up as blur/ringing around sharp edges
    # like text -- an encoding artifact, not a font or rendering issue (a
    # static savefig() at the same dpi comes out crisp). A low CRF fixes it.
    writer = animation.FFMpegWriter(
        fps=output_fps, extra_args=["-crf", "18", "-preset", "slow"]
    )
    anim.save(out_path, writer=writer, dpi=DPI)
    plt.close(fig)
    print(
        f"Wrote {total_output_frames} frames ({duration:.2f}s @ {output_fps:.1f}fps) to {out_path}"
    )


if __name__ == "__main__":
    render_comparison()
