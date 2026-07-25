"""Renders a NeuroMechFly-only comparison video for the 2D-observation
benchmark (`quickik.XYView`, see `../quickik_rust/src/twod.rs` and
`plot_2d_comparison.py`): the same mocap sequence solved twice with
`quickik.SequenceSolver`, once from the usual 3D keypoint observations and
once from only their x/y coordinates (as if seen by a camera looking straight
down/up the Z axis -- "bottom ViewXY"), overlaid to show how much pose
accuracy that missing depth information costs.

Reuses `render_video.py`'s NeuroMechFly setup (body plan, fixtures, warm-
started sequence solving, from-JSON forward-kinematics replica, chase-cam
recentering, hidden root-to-coxa bones) rather than duplicating it -- see
that module's docstring for those details. The only new piece is the second,
XYView-mapped solve.

One 3D panel, drawn twice per frame: once at each keypoint's real solved
position, and once flattened onto the axes' own floor grid (z set to
`ax.get_zlim()[0]`, not literally 0) -- the actual view the 2D fit was
observed from -- faded so it reads as a shadow rather than a second skeleton.
The 3D-observation fit is blue
(`render_video.FIT_COLOR`/`plot_comparison.NEUROMECHFLY_COLOR`, thin bones,
small dots), the 2D/XYView-observation fit is green
(`plot_comparison.G1_COLOR`, thick bones) -- the same two colors
`plot_2d_comparison.py` uses for "3d" vs. "xyview" -- and raw MoCap keypoints
are gray dots.

Writes `results/example_clip_2d_xyview.mp4` (requires ffmpeg on PATH).

Usage (same environment as `render_video.py`):

    python render_video_2d.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import quickik
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.text import TextPath
from mpl_toolkits.mplot3d.art3d import Line3DCollection, pathpatch_2d_to_3d
from plot_comparison import G1_COLOR
from render_video import (
    DPI,
    FIT_COLOR,
    MOCAP_COLOR,
    axis_limits,
    forward_kinematics_full,
    load_body,
    register_fonts,
    solve_sequence,
)

OUT_DIR = Path(__file__).resolve().parent / "results"

FIT_3D_COLOR = FIT_COLOR
FIT_2D_COLOR = G1_COLOR

# Tighter than render_video.BODIES["neuromechfly"]["padding"] (1.02): that
# padding is tuned for the original video's own single skeleton, but here two
# overlaid fits plus the floor projection already fill the box, so a little
# less slack still holds every frame comfortably.
PADDING = 1.01

# 10x SolverConfig's own default (1e-3) -- same fix, and same magnitude, as
# render_video.BODIES["g1"]["weight"]'s: XYView leaves every keypoint's depth
# only weakly constrained (through the kinematic chain, not observed
# directly), so like G1's redundant wrist sub-chain, without a stronger pull
# toward the neutral pose the solver is free to use that depth as arbitrary
# self-motion, picking a different-looking (but equally XY-consistent) pose
# from frame to frame.
WEIGHT_2D_XYVIEW = 0.01

# The flattened (z=0) copies are a shadow/projection, not a second skeleton --
# faded out relative to the real, z-varying one.
FLAT_ALPHA = 0.35


def build_observations_2d_xyview(target_ego):
    """Same convention as `render_video.build_observations`, but reprojecting
    each target via XYView (x/y unchanged, z dropped) into `Position2D`
    observations -- the Python mirror of `twod.rs`'s
    `observations_2d_xyview`."""
    obs = [quickik.KeypointObservation.missing()]
    for x, y, _z in target_ego:
        obs.append(quickik.KeypointObservation.position_2d([x, y], 1.0))
    return obs


def solve_sequence_xyview(tree, fixtures):
    """Warm-started XYView solve, mirroring `render_video.solve_sequence` but
    with a `quickik.XYView()` mapper, 2D observations, and a stronger
    neutral-pose prior -- see `WEIGHT_2D_XYVIEW`."""
    config = quickik.SolverConfig(weight=WEIGHT_2D_XYVIEW)
    seq = quickik.SequenceSolver(tree, config, mapper=quickik.XYView())
    return [
        seq.solve_frame(build_observations_2d_xyview(f["target_ego"]))
        for f in fixtures["native_rate_frames"]
    ]


def add_floor_label(ax, x, y, z, s, size, color):
    """Draws `s` as a filled glyph-outline patch embedded in the z=`z` plane
    (via `pathpatch_2d_to_3d`), anchored at `(x, y)`, rather than `ax.text`'s
    single rigid rotation of the whole string. `ax.text`'s `zdir` only
    orients the text block as a flat rotated rectangle -- every glyph stays
    undistorted, so it still reads as a sign propped up at an angle. Here
    each glyph outline is its own set of points, which the real 3D-to-2D
    projection then skews individually (the same perspective the floor grid
    itself has), so the string reads as actually painted on the floor."""
    patch = PathPatch(TextPath((x, y), s, size=size), facecolor=color, edgecolor="none")
    ax.add_patch(patch)
    pathpatch_2d_to_3d(patch, z=z, zdir="z")
    return patch


def flatten_z(frames, z_floor):
    """Copies of `frames` (each an (N, 3) array) with z set to `z_floor` --
    the XY-plane projection, drawn in the same 3D axes (on its own floor
    grid) rather than a separate 2D panel."""
    flat = [f.copy() for f in frames]
    for f in flat:
        f[:, 2] = z_floor
    return flat


def flatten_bones_z(bones, z_floor):
    """Same idea as `flatten_z`, for (E, 2, 3) bone-segment arrays."""
    flat = [b.copy() for b in bones]
    for b in flat:
        b[:, :, 2] = z_floor
    return flat


def prepare_neuromechfly():
    """Loads NeuroMechFly once and solves it twice (3D observations, then
    XYView), chase-camming each fit on its own solved root every frame so the
    two overlaid skeletons compare pose (joint angles), not each solve's own
    independent root-position estimate. MoCap keypoints are recentered on the
    3D fit's root, same reference `render_video.prepare_body` would use for
    this body alone. Also derives each set's counterpart flattened onto the
    axes' own floor -- see `flatten_z`/`flatten_bones_z`."""
    cfg, joints, dof_offsets, tree, fixtures, edges = load_body("neuromechfly")
    states_3d = solve_sequence(tree, fixtures, cfg["missing_keypoints"], cfg["weight"])
    states_2d = solve_sequence_xyview(tree, fixtures)
    native_frames = fixtures["native_rate_frames"]

    positions_3d = [
        forward_kinematics_full(
            joints, dof_offsets, s.dof_angles, s.root_pos, s.root_rot
        )
        for s in states_3d
    ]
    positions_2d = [
        forward_kinematics_full(
            joints, dof_offsets, s.dof_angles, s.root_pos, s.root_rot
        )
        for s in states_2d
    ]

    roots_3d = [np.array(s.root_pos) for s in states_3d]
    roots_2d = [np.array(s.root_pos) for s in states_2d]

    mocap_frames = [
        np.array(f["target_ego"]) - root
        for f, root in zip(native_frames, roots_3d, strict=True)
    ]
    fitted_frames_3d = [
        np.array(list(p.values())) - root
        for p, root in zip(positions_3d, roots_3d, strict=True)
    ]
    fitted_frames_2d = [
        np.array(list(p.values())) - root
        for p, root in zip(positions_2d, roots_2d, strict=True)
    ]
    fitted_bones_3d = [
        np.array([[pos[parent], pos[child]] for parent, child in edges]) - root
        for pos, root in zip(positions_3d, roots_3d, strict=True)
    ]
    fitted_bones_2d = [
        np.array([[pos[parent], pos[child]] for parent, child in edges]) - root
        for pos, root in zip(positions_2d, roots_2d, strict=True)
    ]

    lo, hi = axis_limits(
        mocap_frames, fitted_frames_3d, fitted_frames_2d, padding=PADDING
    )
    # This is exactly the value `ax.set_zlim(lo[2], hi[2])` (see `setup_panel`)
    # gives `ax.get_zlim()[0]` -- the floor grid the axes draw at the bottom
    # of the z range, computed here since it's needed before the axes exist.
    z_floor = lo[2]

    display_fps = cfg["fps"] * cfg["playback_speed"]
    return {
        "cfg": cfg,
        "mocap_frames": mocap_frames,
        "fitted_frames_3d": fitted_frames_3d,
        "fitted_frames_2d": fitted_frames_2d,
        "fitted_bones_3d": fitted_bones_3d,
        "fitted_bones_2d": fitted_bones_2d,
        "mocap_frames_flat": flatten_z(mocap_frames, z_floor),
        "fitted_frames_3d_flat": flatten_z(fitted_frames_3d, z_floor),
        "fitted_frames_2d_flat": flatten_z(fitted_frames_2d, z_floor),
        "fitted_bones_3d_flat": flatten_bones_z(fitted_bones_3d, z_floor),
        "fitted_bones_2d_flat": flatten_bones_z(fitted_bones_2d, z_floor),
        "lo": lo,
        "hi": hi,
        "display_fps": display_fps,
        "n_frames": len(states_3d),
    }


def legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color=MOCAP_COLOR,
            label="MoCap keypoints",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=FIT_3D_COLOR,
            label="3D-observation fit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=FIT_2D_COLOR,
            label="2D-observation fit",
        ),
    ]


def setup_panel(ax, body):
    cfg = body["cfg"]
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(body["lo"][0], body["hi"][0])
    ax.set_ylim(body["lo"][1], body["hi"][1])
    ax.set_zlim(body["lo"][2], body["hi"][2])
    ax.view_init(**cfg["view"])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("X", labelpad=-10)
    ax.set_ylabel("Y", labelpad=-10)
    ax.set_zlabel("Z", labelpad=-10)
    ax.set_title(cfg["title"], pad=28)

    # edgecolors="none": scatter's default edge otherwise stays fully opaque
    # even when `alpha` fades the face, which on the flattened (FLAT_ALPHA)
    # copies reads as a dark ring around a faint dot instead of one uniformly
    # faded marker.
    def scatter(color, size, alpha):
        return ax.scatter(
            [],
            [],
            [],
            c=color,
            s=size,
            alpha=alpha,
            depthshade=False,
            edgecolors="none",
        )

    # add_collection3d computes its own axis bounds from the initial segments,
    # so every Line3DCollection must be seeded with real data (frame 0), not
    # an empty list.
    def bones(key, color, width, alpha):
        collection = Line3DCollection(
            body[key][0], colors=color, linewidths=width, alpha=alpha
        )
        ax.add_collection3d(collection)
        return collection

    artists = {
        "mocap": scatter(MOCAP_COLOR, 25, 1.0),
        "fit_3d": scatter(FIT_3D_COLOR, 4, 1.0),
        "fit_2d": scatter(FIT_2D_COLOR, 15, 1.0),
        "bones_3d": bones("fitted_bones_3d", FIT_3D_COLOR, 0.8, 1.0),
        "bones_2d": bones("fitted_bones_2d", FIT_2D_COLOR, 2.5, 1.0),
        "mocap_flat": scatter(MOCAP_COLOR, 25, FLAT_ALPHA),
        "fit_3d_flat": scatter(FIT_3D_COLOR, 4, FLAT_ALPHA),
        "fit_2d_flat": scatter(FIT_2D_COLOR, 15, FLAT_ALPHA),
        "bones_3d_flat": bones("fitted_bones_3d_flat", FIT_3D_COLOR, 0.8, FLAT_ALPHA),
        "bones_2d_flat": bones("fitted_bones_2d_flat", FIT_2D_COLOR, 2.5, FLAT_ALPHA),
    }

    ax.legend(
        handles=legend_handles(),
        loc="upper left",
        bbox_to_anchor=(0.0, 1.06),
        frameon=False,
    )
    # text2D pins this to the axes' own 2D display space, unaffected by the 3D
    # view/rotation -- unlike ax.text, which would place it at a fixed *data*
    # point instead. Kept in `artists` (as "speed_text", not part of the
    # suffix-indexed set `update` touches) so `render` can hide it just for
    # the static frame export.
    artists["speed_text"] = ax.text2D(
        0.98,
        0.98,
        f"{cfg['playback_speed']:g}x speed",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    x_range = body["hi"][0] - body["lo"][0]
    y_range = body["hi"][1] - body["lo"][1]
    add_floor_label(
        ax,
        body["lo"][0] + x_range * 0.05,
        body["lo"][1] + y_range * 0.05,
        body["lo"][2],
        "X-Y projection",
        size=x_range * 0.06,
        color="0.4",
    )
    return artists


def render():
    body = prepare_neuromechfly()
    output_fps = body["display_fps"]
    total_output_frames = body["n_frames"]

    plt.rcParams["font.family"] = register_fonts()

    fig = plt.figure(figsize=(6.4, 5.6), frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    artists = setup_panel(ax, body)

    def update(idx):
        for suffix in ("", "_flat"):
            artists[f"mocap{suffix}"]._offsets3d = tuple(
                body[f"mocap_frames{suffix}"][idx].T
            )
            artists[f"fit_3d{suffix}"]._offsets3d = tuple(
                body[f"fitted_frames_3d{suffix}"][idx].T
            )
            artists[f"fit_2d{suffix}"]._offsets3d = tuple(
                body[f"fitted_frames_2d{suffix}"][idx].T
            )
            artists[f"bones_3d{suffix}"].set_segments(
                body[f"fitted_bones_3d{suffix}"][idx]
            )
            artists[f"bones_2d{suffix}"].set_segments(
                body[f"fitted_bones_2d{suffix}"][idx]
            )
        return list(artists.values())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # A static SVG of frame 0 -- scatters start out empty (see setup_panel),
    # so this needs one update() call before saving, unlike the bones (already
    # seeded with frame 0 to satisfy add_collection3d; see its own comment).
    # The speed label is hidden just for this export, then restored so the
    # mp4 animation (built from the same figure/artists below) still shows it.
    update(0)
    artists["speed_text"].set_visible(False)
    fig.savefig(OUT_DIR / "example_clip_2d_xyview_frame0.svg")
    artists["speed_text"].set_visible(True)

    anim = animation.FuncAnimation(fig, update, frames=total_output_frames, blit=False)
    out_path = OUT_DIR / "example_clip_2d_xyview.mp4"
    # Same low-CRF encoding as render_video.py's own writer -- matplotlib's
    # FFMpegWriter defaults otherwise blur/ring around sharp edges like text.
    writer = animation.FFMpegWriter(
        fps=output_fps, extra_args=["-crf", "18", "-preset", "slow"]
    )
    anim.save(out_path, writer=writer, dpi=DPI)
    plt.close(fig)
    print(
        f"Wrote {total_output_frames} frames ({total_output_frames / output_fps:.2f}s "
        f"@ {output_fps:.1f}fps) to {out_path}"
    )


if __name__ == "__main__":
    render()
