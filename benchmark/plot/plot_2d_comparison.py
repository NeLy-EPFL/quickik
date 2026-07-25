"""Compares QuickIK's 2D-observation IK (via XYView) against its own
3D-observation baseline, Rust only. 2D observations are a QuickIK-only
feature -- none of KDL/Pinocchio/RBDL support this (see
quickik_rust/src/twod.rs) -- and Python/C++ only carry a lightweight perf
sanity test for this (see python/tests/test_bindings.py,
cpp/tests/test_main.cpp), not a full benchmark, so there's nothing to compare
across bindings here. No G1 data either (2D fit quality is still being
validated for that body; see twod.rs).

Reads, both under results/:
  - quickik-rust-neuromechfly.json (3D baseline; written by the existing 3D
    benchmark)
  - quickik-rust-2d-xyview-neuromechfly.json (written by this repo's Rust 2D
    benchmark)
  - errors-neuromechfly.json (per-frame average fit-residual distributions
    for 3D/XYView, computed once from the Rust API -- see errors.rs)

Writes results/comparison-2d.svg. Usage (with devtools-pyenv/'s shared venv
active):

    python plot_2d_comparison.py
"""

import json
from pathlib import Path

from plot_comparison import BAR_HEIGHT, despine, format_value, register_fonts

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

# Fixed display/color order throughout (bars and KDE alike). Colors are the
# dataviz reference palette's first two categorical slots (blue/orange),
# validated for all-pairs comparisons -- see scripts/validate_palette.js.
OBSERVATIONS = ["3d", "xyview"]
OBSERVATION_LABELS = {"3d": "3D", "xyview": "XYView"}
OBSERVATION_COLORS = {"3d": "#2a78d6", "xyview": "#eb6834"}

TEXT_NUMBER_FONTSIZE = 8
# 2 touching bars, 3D on top -- same BAR_HEIGHT as plot_comparison.py's own
# bars, so the two charts' bars read at the same visual thickness.
BAR_Y = {"3d": BAR_HEIGHT / 2, "xyview": -BAR_HEIGHT / 2}


def load_perf_results():
    """observation -> result dict ("3d"/"xyview"), skipping either that isn't
    present."""
    results = {}
    path_3d = RESULTS_DIR / f"quickik-rust-{BODY}.json"
    if path_3d.exists():
        results["3d"] = json.loads(path_3d.read_text())
    path_xyview = RESULTS_DIR / f"quickik-rust-2d-xyview-{BODY}.json"
    if path_xyview.exists():
        results["xyview"] = json.loads(path_xyview.read_text())
    return results


def load_errors():
    path = RESULTS_DIR / f"errors-{BODY}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def print_table(results):
    header = ["observation"] + [m["title"] for m in METRICS]
    rows = []
    for obs in OBSERVATIONS:
        r = results.get(obs)
        if r is None:
            continue
        row = [OBSERVATION_LABELS[obs]] + [f"{r[m['key']]:,.1f}" for m in METRICS]
        rows.append(row)

    if not rows:
        print(f"No 2D results found in {RESULTS_DIR}. Run the 2D benchmark first.")
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
    widest = 0.0

    for obs in OBSERVATIONS:
        r = results.get(obs)
        if r is None or r.get(key) is None:
            continue
        value = r[key] * scale
        widest = max(widest, value)
        bar = ax.barh(
            BAR_Y[obs],
            value,
            height=BAR_HEIGHT,
            color=OBSERVATION_COLORS[obs],
            zorder=3,
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
        [BAR_Y[obs] for obs in OBSERVATIONS],
        labels=[OBSERVATION_LABELS[obs] for obs in OBSERVATIONS],
        va="center",
    )
    ax.set_ylim(-0.5, 0.5)
    despine(ax)
    if widest > 0:
        ax.set_xlim(0, widest * 1.2)
    return widest


def draw_error_kde(ax, errors):
    import numpy as np
    from scipy.stats import gaussian_kde

    all_values = [v for obs in OBSERVATIONS for v in errors[obs]]
    x = np.linspace(0, max(all_values) * 1.15, 400)

    for obs in OBSERVATIONS:
        values = np.asarray(errors[obs])
        density = gaussian_kde(values)(x)
        color = OBSERVATION_COLORS[obs]
        ax.plot(x, density, color=color, linewidth=2, zorder=3)
        ax.fill_between(x, density, color=color, alpha=0.2, zorder=2)
        # Rug ticks for the actual per-frame values underneath the curves --
        # honest about n being small (one point per real mocap frame).
        ax.plot(
            values,
            np.full_like(values, -0.02 * density.max()),
            "|",
            color=color,
            markersize=8,
            markeredgewidth=1.5,
            zorder=4,
            clip_on=False,
        )

    n = len(errors[OBSERVATIONS[0]])
    ax.set_title(
        f"Fit-residual distribution (per-frame average, n={n} real mocap frames)"
    )
    ax.set_xlabel("3D distance from solved pose to target (model units)")
    ax.set_ylabel("density")
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
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
    # Perf panels are very short (2 touching bars each); the KDE panel needs
    # real height for its curves to read clearly.
    height_ratios = [1] * len(METRICS) + ([3] if errors else [])
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(6, 1.35 * len(METRICS) + (3.2 if errors else 0)),
        gridspec_kw={"height_ratios": height_ratios},
    )
    fig.subplots_adjust(hspace=1.6, left=0.16, right=0.97, top=0.90, bottom=0.08)

    for ax, metric in zip(axes[: len(METRICS)], METRICS, strict=True):
        draw_metric_panel(ax, results, metric)

    if errors:
        draw_error_kde(axes[len(METRICS)], errors)

    legend_handles = [
        Patch(facecolor=OBSERVATION_COLORS[obs], label=OBSERVATION_LABELS[obs])
        for obs in OBSERVATIONS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    out_path = RESULTS_DIR / "comparison-2d.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    results = load_perf_results()
    errors = load_errors()
    print("=== QuickIK 2D (XYView) vs. 3D -- NeuroMechFly, Rust ===")
    print_table(results)
    if errors is None:
        print(
            f"\n(no {RESULTS_DIR / f'errors-{BODY}.json'} found -- skipping error-distribution panel)"
        )
    plot_chart(results, errors)
