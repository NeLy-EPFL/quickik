# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Plots the output of the examples as videos, and checks parity between
languages.

Every example writes `examples/<language>/output/<example>_<body>.json`, in the
same format for every language. Usage (needs ffmpeg on PATH; uv installs the Python
dependencies declared above, or install them yourself and use `python`):

    uv run examples/plot_results.py                   # every */output/*.json
    uv run examples/plot_results.py <file.json> ...   # specific files

For each file, this writes a video (`.mp4`) and a still of its first frame
(`_frame0.png`) next to it: the recorded keypoints and the fitted skeleton in
3D, plus each camera's view for the 2D examples. The 2D examples also show the
fit from 3D observations (`from_3d`, which must be run first) for reference.
Recordings faster than 33 fps are slowed down.

If the same example and body were run in several languages, this also prints
the largest difference between their fitted DOF values.
"""

import itertools
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

EXAMPLES_DIR = Path(__file__).resolve().parent
MAX_VIDEO_FPS = 33
FIT_COLOR = "#4051b5"
FIT_WIDTH = plt.rcParams["lines.linewidth"]
LABEL_PAD = 3 * plt.rcParams["axes.labelpad"]
REF_COLOR = "green"


def project(points, projection):
    """Projects `(..., 3)` world positions into a view, using the same formulas
    as QuickIK's projections."""
    if projection["type"] == "orthographic_view_xy":
        return projection["scale"] * points[..., :2] + projection["offset"]
    rotation = np.array(projection["world2cam_rot_mat"])
    p = points @ rotation.T + projection["world2cam_pos"]
    focal = [projection["fx"], projection["fy"]]
    return focal * p[..., :2] / p[..., 2:] + [projection["cx"], projection["cy"]]


def plot(path):
    data = json.loads(path.read_text())
    recorded = np.array(data["observations"]["positions"])  # (n_frames, n_tracked, 3)
    fit = np.array(data["fit"]["keypoint_positions"])  # (n_frames, n_joints, 3)
    reference = None
    if data["example"] != "from_3d":
        reference_path = path.with_name(f"from_3d_{data['body']}.json")
        reference_data = json.loads(reference_path.read_text())
        reference = np.array(reference_data["fit"]["keypoint_positions"])

    # One bone per joint with a parent. The fly's thorax (root) has no visible
    # extent in its body plan, so its bones to each leg's base are skipped.
    parents = data["parents"]
    hidden = parents.index(None) if data["body"] == "neuromechfly" else None
    bones = [(p, c) for c, p in enumerate(parents) if p not in (None, hidden)]
    bones = np.array(bones)

    views = data["views"]
    # G1's walk spans a long, flat box, so its 3D panel gets more room. 3D axes
    # occupy a square region, so the box is zoomed to fill the panel and its
    # contents aren't clipped to that square.
    is_g1 = data["body"] == "g1"
    width_3d, zoom = (12, 1.6) if is_g1 else (6, 0.9)
    # Camera views are stacked in one column, as wide as needed for each to be
    # about 5 inches tall in total, whatever its image aspect ratio.
    aspects = [width / height for width, height in (v["image_size"] for v in views)]
    width_views = 5 / len(views) * max(aspects) if views else 0
    fig = plt.figure(figsize=(width_3d + width_views, 6))
    grid = fig.add_gridspec(
        max(len(views), 1),
        2 if views else 1,
        width_ratios=[width_3d, width_views] if views else None,
        wspace=0.5,
        hspace=0.3,
    )
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d", computed_zorder=False)

    # 3D panel with equal scale on all axes. G1 stands on the ground, so the
    # floor of the plot is at z = 0; the fly's recording is centered on its
    # thorax instead.
    points = np.concatenate([recorded.reshape(-1, 3), fit.reshape(-1, 3)])
    lower, upper = points.min(axis=0), points.max(axis=0)
    if is_g1:
        lower[2] = 0
    ax_3d.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]))
    ax_3d.set(zlim=(lower[2], upper[2]))
    ax_3d.set_box_aspect(upper - lower, zoom=zoom)
    for axis in "xyz":
        set_label = getattr(ax_3d, f"set_{axis}label")
        set_label(f"{axis} ({data['units']})", labelpad=LABEL_PAD)

    # Each panel: its axes, how to map 3D positions into it, and the keypoints
    # it observed.
    panels = [(ax_3d, lambda x: x, recorded)]
    for i, view in enumerate(views):
        ax = fig.add_subplot(grid[i, 1])
        width, height = view["image_size"]
        ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal", title=view["name"])
        projection = view["projection"]
        panels.append((ax, lambda x, p=projection: project(x, p), view["observed"]))

    artists = []
    for ax, _, observed in panels:
        dots = ax.plot(*np.transpose(observed[0]), "o", color="gray", ms=3)[0]
        lines = Line3DCollection if ax.name == "3d" else LineCollection
        fit_lines = lines([], colors=FIT_COLOR, linewidths=FIT_WIDTH, zorder=2)
        ref_lines = lines([], colors=REF_COLOR, linewidths=FIT_WIDTH / 2, zorder=1)
        ax.add_collection(fit_lines)
        ax.add_collection(ref_lines)
        for artist in (dots, fit_lines, ref_lines):
            artist.set_clip_on(ax.name != "3d")  # see `zoom` above
        artists.append((dots, fit_lines, ref_lines))

    legend = {
        "observations": Line2D([], [], color="gray", marker="o", ms=3, ls=""),
        "fit": Line2D([], [], color=FIT_COLOR, lw=FIT_WIDTH),
    }
    if reference is not None:
        legend["ref. fit from 3D"] = Line2D([], [], color=REF_COLOR, lw=FIT_WIDTH / 2)
    fig.legend(legend.values(), legend.keys(), loc="upper left")

    def draw(t):
        for (ax, to_panel, observed), (dots, fit_lines, ref_lines) in zip(
            panels, artists, strict=True
        ):
            if ax.name == "3d":
                dots.set_data_3d(*np.transpose(observed[t]))
            else:
                dots.set_data(*np.transpose(observed[t]))
            fit_lines.set_segments(to_panel(fit[t])[bones])
            if reference is not None:
                ref_lines.set_segments(to_panel(reference[t])[bones])

    speed = min(1.0, MAX_VIDEO_FPS / data["fps"])
    anim = animation.FuncAnimation(fig, draw, frames=len(recorded))
    video_path = path.with_suffix(".mp4")
    anim.save(video_path, writer=animation.FFMpegWriter(fps=data["fps"] * speed))
    draw(0)
    fig.savefig(path.with_name(f"{path.stem}_frame0.png"))
    plt.close(fig)
    print(f"Saved {video_path}")


def check_parity(paths):
    """Prints the largest DOF value difference between languages, per example
    and body."""
    runs = [json.loads(path.read_text()) for path in paths]
    key = lambda run: (run["example"], run["body"])  # noqa: E731
    for (example, body), group in itertools.groupby(sorted(runs, key=key), key=key):
        for a, b in itertools.combinations(group, 2):
            diff = np.subtract(a["fit"]["dof_values"], b["fit"]["dof_values"])
            print(
                f"{example}, {body}: {a['language']} vs {b['language']}: "
                f"max DOF difference {np.abs(diff).max():.2e}"
            )


def main():
    paths = [Path(p) for p in sys.argv[1:]]
    paths = paths or sorted(EXAMPLES_DIR.glob("*/output/*.json"))
    for path in paths:
        plot(path)
    check_parity(paths)


if __name__ == "__main__":
    main()
