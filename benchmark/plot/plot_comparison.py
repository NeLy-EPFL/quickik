"""Aggregates every benchmark's results/<name>.json (see RESULTS_SCHEMA.md)
into a markdown comparison table (always) and a bar chart (if matplotlib is
available).

Only `formulation: "whole-tree"` results are compared here -- results
solving fastik's actual problem (floating base + all 6 legs + all 30
keypoints, jointly). "fixed-base-per-leg" libraries (currently TRAC-IK and
FABRIK) are excluded entirely, not just visually de-emphasized: neither can
be reformulated to solve the whole-tree problem (TRAC-IK's solver takes
exactly one chain and one end-effector target per call, by its own public
API; FABRIK is defined around a fixed-base single chain reaching one tip --
extending either to a floating base + multi-keypoint branching tree would
mean writing a different, non-standard algorithm). Their benchmarks and
results still exist under ../extern/{trac_ik,fabrik}/ as standalone
reference implementations of that different problem -- just not compared
here. See RESULTS_SCHEMA.md.

Usage:

    python plot_results.py
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# fastik/RBDL/Pinocchio all cap their Gauss-Newton solve at this many
# iterations by default (early-stop tolerances usually trigger sooner); KDL
# uses a higher nominal cap but converges within the same range in practice
# (see extern/kdl/README.md).
MAX_ITERATIONS_PER_SOLVE = 10

METRICS = [
    {
        "key": "single_frame_latency_us",
        "max_key": "single_frame_latency_max_us",
        "title": f"Mean and max single-frame latency (≤{MAX_ITERATIONS_PER_SOLVE} iters/solve)",
        "xlabel": "Latency (μs)",
        "unit": "μs",
    },
    {
        "key": "single_thread_throughput_fps",
        "max_key": None,
        "title": "Throughput (sequential frames, single thread)",
        "xlabel": "Throughput (frames/s)",
        "unit": "fps",
    },
    {
        "key": "multi_thread_throughput_fps",
        "max_key": None,
        "title": "Throughput (sequential frames, 8 threads)",
        "xlabel": "Throughput (frames/s)",
        "unit": "fps",
    },
]

# Explicit display order (not alphabetical): language variants of the same
# library grouped together, fastest-per-library first.
ORDER = [
    "fastik-rust",
    "fastik-cpp",
    "fastik-python",
    "rbdl",
    "rbdl-python",
    "pinocchio-cpp",
    "pinocchio",
    "kdl",
]
DISPLAY_NAMES = {
    "fastik-rust": "FastIK (Rust)",
    "fastik-python": "FastIK (Python)",
    "fastik-cpp": "FastIK (C++)",
    "kdl": "KDL",
    "pinocchio": "Pinocchio (Python)",
    "pinocchio-cpp": "Pinocchio (C++)",
    "rbdl": "RBDL (C++)",
    "rbdl-python": "RBDL (Python)",
}
FASTIK_COLOR = "tab:orange"
OTHER_COLOR = "#888888"
TEXT_NUMBER_FONTSIZE = 8
# DejaVu Sans is the only open-source sans-serif shipped inside matplotlib
# itself (also its default) -- "Open Sans" is not bundled with matplotlib.
FONT_FAMILY = "DejaVu Sans"


def load_results():
    all_results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        # fastik-scaling.json is a list of weak-scaling data points (see
        # plot_scaling.py), not a single-library result -- skip it here.
        if isinstance(data, dict):
            all_results.append(data)
    included = [r for r in all_results if r.get("formulation") == "whole-tree"]
    excluded = [r for r in all_results if r not in included]
    included.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER))
    return included, excluded


def print_table(results, excluded):
    if not results:
        print(f"No whole-tree results found in {RESULTS_DIR}. Run the benchmarks first.")
        return

    header = ["library"] + [m["title"] for m in METRICS]
    rows = []
    for r in results:
        row = [r["name"]]
        for m in METRICS:
            value = r.get(m["key"])
            row.append("--" if value is None else f"{value:,.1f}")
        rows.append(row)

    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths)))

    if excluded:
        names = ", ".join(r["name"] for r in excluded)
        print(
            f"\nExcluded (different, easier problem -- fixed base, per-leg, tip-only, not "
            f"comparable to the whole-tree rows above): {names}. See their own "
            f"../extern/<name>/README.md."
        )

    notes = [(r["name"], r["notes"]) for r in results if r.get("notes")]
    if notes:
        print("\nNotes:")
        for name, note in notes:
            print(f"  {name}: {note}")


def format_value(value, unit=None):
    """Fixed-point, never scientific -- e.g. '6,301 μs' or '147.2 μs', never '1.06e+06'."""
    text = f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"
    return f"{text} {unit}" if unit else text


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_chart(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart; the table above is still complete)")
        return

    plt.rcParams["font.family"] = FONT_FAMILY

    # results is already in ORDER; reversed for barh so it reads top-to-bottom.
    ordered = list(reversed(results))
    names = [DISPLAY_NAMES.get(r["name"], r["name"]) for r in ordered]
    colors = [FASTIK_COLOR if r["name"].startswith("fastik-") else OTHER_COLOR for r in ordered]

    # Figure is 50% larger than a "natural" size, saved at a correspondingly
    # lower dpi -- output pixel dimensions stay about the same, but every
    # font (all at their absolute point-size defaults) occupies a smaller
    # fraction of the figure, so the rendered image reads less oversized.
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(7, 9))

    for ax, metric in zip(axes, METRICS):
        key, max_key, title, xlabel, unit = metric["key"], metric["max_key"], metric["title"], metric["xlabel"], metric["unit"]
        values = [r.get(key) for r in ordered]
        max_values = [r.get(max_key) if max_key else None for r in ordered]
        bar_names = [n for n, v in zip(names, values) if v is not None]
        bar_values = [v for v in values if v is not None]
        bar_max_values = [mv for v, mv in zip(values, max_values) if v is not None]
        bar_colors = [c for c, v in zip(colors, values) if v is not None]
        if not bar_values:
            ax.set_title(f"{title}\n(no data)")
            continue

        bars = ax.barh(bar_names, bar_values, color=bar_colors, height=0.55)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        despine(ax)
        widest = max(mv if mv is not None else v for v, mv in zip(bar_values, bar_max_values))
        ax.set_xlim(0, widest * 1.3)
        for bar, value, max_value, color in zip(bars, bar_values, bar_max_values, bar_colors):
            y = bar.get_y() + bar.get_height() / 2
            if max_value is None:
                ax.annotate(
                    format_value(value, unit),
                    (bar.get_width(), y),
                    xytext=(4, 0),
                    fontsize=TEXT_NUMBER_FONTSIZE,
                    textcoords="offset points",
                    ha="left",
                    va="center",
                )
                continue
            ax.plot([bar.get_width(), max_value], [y, y], color=color, linewidth=1, zorder=2)
            ax.plot(max_value, y, marker="o", color=color, markersize=3, zorder=3)
            ax.annotate(
                f"{format_value(value, unit)} (max {format_value(max_value, unit)})",
                (max_value, y),
                xytext=(4, 0),
                fontsize=TEXT_NUMBER_FONTSIZE,
                textcoords="offset points",
                ha="left",
                va="center",
            )

    fig.suptitle("NeuroMechFly (42 limb DOFs)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=2.5)
    out_path = RESULTS_DIR / "comparison.png"
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    results, excluded = load_results()
    print_table(results, excluded)
    plot_chart(results)
