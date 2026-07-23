"""Plots quickik-scaling's sweep (results/quickik-scaling.json -- a JSON array
of {n_threads, total_frames, elapsed_s, throughput_fps} points, one per
thread count -- written by benchmark/quickik_scaling, see its own README and
run_sweep.sh) as a speedup-vs-workers line chart: speedup(n) =
throughput(n)/throughput(1), against a dashed 1:1 ideal-scaling reference
line, log2 on both axes (so ideal scaling is a straight diagonal and equal
ratios -- e.g. 1->2 threads, 8->16 threads -- take up equal horizontal
space).

Note: the underlying quickik_scaling test is a *weak*-scaling design (work
per thread is fixed at one 200-frame segment; total work grows with thread
count -- see quickik_scaling's own module docs), not strong scaling (fixed
total work, more threads). Plotting it as speedup-vs-workers with a 1:1
ideal line is the classic strong-scaling visualization, reused here for the
weak-scaling data: since ideal weak scaling keeps elapsed time flat as both
threads and work grow together, throughput(n)/throughput(1) ideally equals
exactly n under that same ideal, so the same "how far below the diagonal"
reading applies.

Usage (with devtools-pyenv/'s shared venv active):

    python plot_scaling.py
"""

import json
from pathlib import Path

from plot_comparison import register_fonts

RESULTS_DIR = Path(__file__).resolve().parent / "results"
COLOR = "#4051b5"


def load_points():
    path = RESULTS_DIR / "quickik-scaling.json"
    if not path.exists():
        return []
    return sorted(json.loads(path.read_text()), key=lambda p: p["n_threads"])


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def print_table(points):
    if not points:
        print(
            f"No weak-scaling data found in {RESULTS_DIR}/quickik-scaling.json. Run quickik_scaling/run_sweep.sh first."
        )
        return
    baseline = next(p["throughput_fps"] for p in points if p["n_threads"] == 1)
    header = ["threads", "total frames", "elapsed (ms)", "throughput (fps)", "speedup"]
    rows = [
        [
            str(p["n_threads"]),
            str(p["total_frames"]),
            f"{p['elapsed_s'] * 1e3:,.1f}",
            f"{p['throughput_fps']:,.1f}",
            f"{p['throughput_fps'] / baseline:.2f}x",
        ]
        for p in points
    ]
    widths = [max(len(row[i]) for row in [header] + rows) for i in range(len(header))]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))


def plot_chart(points):
    if not points:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "\n(matplotlib not installed -- skipping chart; the table above is still complete)"
        )
        return

    plt.rcParams["font.family"] = register_fonts()
    # SVG, not PNG -- smaller file size, crisp at any zoom; fonttype="none"
    # keeps text as real <text> elements (see plot_comparison.py's own
    # savefig comment for why).
    plt.rcParams["svg.fonttype"] = "none"

    n_threads = [p["n_threads"] for p in points]
    baseline = next(p["throughput_fps"] for p in points if p["n_threads"] == 1)
    speedup = [p["throughput_fps"] / baseline for p in points]

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(n_threads, n_threads, linestyle="--", color="gray", label="Ideal scaling")
    ax.plot(n_threads, speedup, marker="o", color=COLOR, label="QuickIK (Rust)")
    for x, y in zip(n_threads, speedup, strict=True):
        ax.annotate(
            f"{y:.2f}x",
            (x, y),
            xytext=(9, -4),
            textcoords="offset points",
            ha="left",
            va="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(n_threads)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Workers (threads)")
    ax.set_ylabel("Speedup (v.s. 1 thread)")
    ax.set_title("Speedup v.s. workers")
    ax.legend(frameon=False)
    despine(ax)

    fig.tight_layout()
    out_path = RESULTS_DIR / "scaling.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    points = load_points()
    print_table(points)
    plot_chart(points)
