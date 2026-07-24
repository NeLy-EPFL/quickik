"""Compares QuickIK's 2D-observation IK (a synthetic bottom-view pinhole
Camera, plus the trivial XYView) against its own 3D-observation baseline,
across all three bindings (Rust/Python/C++), for NeuroMechFly only. 2D
observations are a QuickIK-only feature -- none of KDL/Pinocchio/RBDL support
camera-projected keypoints (see quickik_rust/src/twod.rs) -- so unlike
plot_comparison.py, there's nothing external to compare against here, and no
G1 data (2D fit quality is still being validated for that body; see twod.rs).

Reads, all under results/:
  - quickik-<rust|python|cpp>-neuromechfly.json (3D baseline; written by the
    existing 3D benchmarks)
  - quickik-<rust|python|cpp>-2d-<camera|xyview>-neuromechfly.json (written by
    this repo's three 2D benchmark additions)
  - errors-neuromechfly.json (per-keypoint fit-residual distributions for
    3D/XYView/Camera, computed once from the Rust API -- see errors.rs; every
    binding runs the identical compiled solver, so this doesn't depend on
    which one produced it)

Writes results/comparison-2d.svg. Usage (with devtools-pyenv/'s shared venv
active):

    python plot_2d_comparison.py
"""

import json
from pathlib import Path

from plot_comparison import despine, format_value, register_fonts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BODY = "neuromechfly"

# Same 4 performance metrics as plot_comparison.py, same units/scaling
# choices, so the numbers read the same way in both charts.
METRICS = [
    {
        "key": "single_frame_latency_us",
        "title": "Mean single-frame latency (with early stopping)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
    },
    {
        "key": "single_frame_latency_max_us",
        "title": "Single-frame latency (fixed 10 iterations)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
    },
    {
        "key": "single_thread_throughput_fps",
        "title": "Throughput (sequential frames, single thread)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
    },
    {
        "key": "multi_thread_throughput_fps",
        "title": "Throughput (sequential frames, 8 threads)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
    },
]

# Fixed display/color order throughout -- both the bar groups and the error
# panel use this same order and colors, so a reader can track one
# observation type down the whole figure. Colors are the reference palette's
# first three categorical slots (blue/orange/aqua): the only ordering of that
# palette validated for *all-pairs* comparisons (scatter/small-multiples-like
# reading, not just adjacent bars) -- see dataviz skill's palette.md.
OBSERVATIONS = ["3d", "xyview", "camera"]
OBSERVATION_LABELS = {"3d": "3D", "xyview": "XYView", "camera": "Camera"}
OBSERVATION_COLORS = {
    "3d": "#2a78d6",
    "xyview": "#eb6834",
    "camera": "#1baf7a",
}

BINDINGS = ["rust", "cpp", "python"]
BINDING_LABELS = {"rust": "Rust", "cpp": "C++", "python": "Python"}

TEXT_NUMBER_FONTSIZE = 8
# 3 touching sub-bars per group (vs. plot_comparison.py's 2): height chosen
# so BAR_HEIGHT * 3 matches that chart's total per-group height (0.8),
# leaving the same proportional gap between groups.
BAR_HEIGHT = 0.8 / 3
BAR_OFFSETS = [-BAR_HEIGHT, 0.0, BAR_HEIGHT]


def load_perf_results():
    """(binding, observation) -> result dict, skipping any file that isn't
    present (e.g. a binding whose 2D benchmark hasn't been run yet)."""
    results = {}
    for binding in BINDINGS:
        path_3d = RESULTS_DIR / f"quickik-{binding}-{BODY}.json"
        if path_3d.exists():
            results[(binding, "3d")] = json.loads(path_3d.read_text())
        for obs in ("camera", "xyview"):
            path = RESULTS_DIR / f"quickik-{binding}-2d-{obs}-{BODY}.json"
            if path.exists():
                results[(binding, obs)] = json.loads(path.read_text())
    return results


def load_errors():
    path = RESULTS_DIR / f"errors-{BODY}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_table(results):
    header = ["binding", "observation"] + [m["title"] for m in METRICS]
    rows = []
    for binding in BINDINGS:
        for obs in OBSERVATIONS:
            r = results.get((binding, obs))
            if r is None:
                continue
            row = [BINDING_LABELS[binding], OBSERVATION_LABELS[obs]]
            row += [f"{r[m['key']]:,.1f}" for m in METRICS]
            rows.append(row)

    if not rows:
        print(f"No 2D results found in {RESULTS_DIR}. Run the 2D benchmarks first.")
        return

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def draw_metric_panel(ax, results, metric):
    key, unit, small_unit, scale = (
        metric["key"],
        metric["unit"],
        metric["small_unit"],
        metric["scale"],
    )
    group_ys = list(range(len(BINDINGS)))
    widest = 0.0

    for gi, binding in enumerate(reversed(BINDINGS)):
        for obs, offset in zip(OBSERVATIONS, BAR_OFFSETS, strict=True):
            r = results.get((binding, obs))
            if r is None or r.get(key) is None:
                continue
            value = r[key] * scale
            widest = max(widest, value)
            y = gi + offset
            bar = ax.barh(
                y, value, height=BAR_HEIGHT, color=OBSERVATION_COLORS[obs], zorder=3
            )[0]
            # format_value's small_unit fallback expects its value already in
            # `unit`-scale (so it knows whether to convert back down to
            # small_unit) -- pass the scaled `value`, not the raw r[key].
            ax.annotate(
                format_value(value, unit, small_unit),
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(4, 0),
                fontsize=TEXT_NUMBER_FONTSIZE,
                textcoords="offset points",
                ha="left",
                va="center",
                zorder=4,
            )

    ax.set_title(metric["title"])
    ax.set_xlabel(metric["xlabel"])
    ax.set_yticks(
        group_ys, labels=[BINDING_LABELS[b] for b in reversed(BINDINGS)], va="center"
    )
    ax.set_ylim(-0.5, len(BINDINGS) - 1 + 0.5)
    despine(ax)
    if widest > 0:
        ax.set_xlim(0, widest * 1.2)
    return widest


def draw_error_panel(ax, errors):
    # Reverse so the reading order (top-to-bottom) matches the bar panels
    # above (3D on top): boxplot's positions grow upward.
    order = list(reversed(OBSERVATIONS))
    data = [errors[obs] for obs in order]
    bp = ax.boxplot(
        data,
        orientation="horizontal",
        positions=range(len(order)),
        widths=0.6,
        patch_artist=True,
        showfliers=True,
        flierprops={
            "marker": "o",
            "markersize": 2,
            "alpha": 0.35,
            "markeredgewidth": 0,
        },
        medianprops={"color": "black"},
    )
    for patch, obs in zip(bp["boxes"], order, strict=True):
        patch.set_facecolor(OBSERVATION_COLORS[obs])
        patch.set_alpha(0.85)
        patch.set_edgecolor("none")

    n = len(errors[order[0]])
    ax.set_title(
        f"Fit-residual distribution (real mocap frames, n={n} keypoints pooled)"
    )
    ax.set_xlabel("3D distance from solved pose to target (model units)")
    ax.set_yticks(
        range(len(order)), labels=[OBSERVATION_LABELS[o] for o in order], va="center"
    )
    despine(ax)


def plot_chart(results, errors):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print(
            "\n(matplotlib not installed -- skipping chart; the table above is still complete)"
        )
        return

    plt.rcParams["font.family"] = register_fonts()
    plt.rcParams["svg.fonttype"] = "none"

    n_panels = len(METRICS) + (1 if errors else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(8, 4.2 * n_panels))
    fig.subplots_adjust(hspace=0.5, left=0.16, right=0.97, top=0.93, bottom=0.06)

    for ax, metric in zip(axes[: len(METRICS)], METRICS, strict=True):
        draw_metric_panel(ax, results, metric)

    if errors:
        draw_error_panel(axes[len(METRICS)], errors)

    legend_handles = [
        Patch(facecolor=OBSERVATION_COLORS[obs], label=OBSERVATION_LABELS[obs])
        for obs in OBSERVATIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    out_path = RESULTS_DIR / "comparison-2d.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    results = load_perf_results()
    errors = load_errors()
    print("=== QuickIK 2D vs. 3D observations -- NeuroMechFly ===")
    print_table(results)
    if errors is None:
        print(
            f"\n(no {RESULTS_DIR / f'errors-{BODY}.json'} found -- skipping error-distribution panel)"
        )
    plot_chart(results, errors)
