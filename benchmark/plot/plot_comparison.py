"""Aggregates every benchmark's results/<name>-<body>.json into a markdown
comparison table (always) and a bar chart (if matplotlib is available), one
of each per body -- currently `neuromechfly` (a fly) and `g1` (a Unitree G1
humanoid; see ../preprocessing/README.md for how its assets are generated).
Every result JSON carries a `"body"` field; this script groups by it and
writes `comparison_<body>.svg` for each body found.

Only `formulation: "whole-tree"` results are compared here -- results
solving the body's actual problem (floating base + every limb/keypoint,
jointly). Any result with a different `formulation` (e.g. a fixed-base,
single-chain-only library that can't be reformulated to solve the whole-tree
problem) is excluded entirely, not just visually de-emphasized, and listed
separately in the printed table.

Usage:

    python plot_comparison.py
"""

import json
import math
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# QuickIK/RBDL/Pinocchio all cap their Gauss-Newton solve at this many
# iterations by default (early-stop tolerances usually trigger sooner); KDL
# uses a higher nominal cap but converges within the same range in practice
# (see extern/kdl/README.md).
MAX_ITERATIONS_PER_SOLVE = 10

# Padding past the widest bar's tip, as a fraction of that bar's value.
XLIM_PAD = 1.15

# 2x2 grid: latency in column 0 (mean-with-early-stop above, fixed-iteration
# worst-case below), throughput in column 1 (single-thread above, multi-
# thread below). "scale" converts the raw JSON units (us, fps) to the
# panel's own axis unit (ms, kFPS); a bar under 1 (in that unit) instead
# displays its label in "small_unit" (us, FPS) -- see format_value --
# without changing the bar's own length, which always stays in the panel's
# one fixed axis unit.
METRICS = [
    {
        "key": "single_frame_latency_us",
        "title": "Mean single-frame latency (with early stopping)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
        "row": 0,
        "col": 0,
    },
    {
        "key": "single_thread_throughput_fps",
        "title": "Throughput (sequential frames, single thread)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
        "row": 0,
        "col": 1,
    },
    {
        "key": "single_frame_latency_max_us",
        "title": f"Single-frame latency (fixed {MAX_ITERATIONS_PER_SOLVE} iters)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
        "row": 1,
        "col": 0,
    },
    {
        "key": "multi_thread_throughput_fps",
        "title": "Throughput (sequential frames, 8 threads)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
        "row": 1,
        "col": 1,
    },
]

# (body, metric key) -> the bar's *display* name (matched against
# `bar_names`, not the JSON "name" field -- see DISPLAY_NAMES) that should be
# drawn capped at CAP_MULTIPLE-x the next-highest bar's value instead of
# running out to its true value, which would flatten every other bar down
# near zero on a linear axis -- see `draw_bars`. Only KDL's g1 mean-latency
# result is extreme enough (~60x the next-highest bar) to need this;
# everywhere else a plain, uncapped bar is clearer.
CAPPED_BARS = {("g1", "single_frame_latency_us"): "KDL"}
CAP_MULTIPLE = 3

# Explicit display order (not alphabetical): language variants of the same
# library grouped together, fastest-per-library first.
ORDER = [
    "quickik-rust",
    "quickik-cpp",
    "quickik-python",
    "rbdl",
    "rbdl-python",
    "pinocchio-cpp",
    "pinocchio",
    "kdl",
]
DISPLAY_NAMES = {
    "quickik-rust": "QuickIK (Rust)",
    "quickik-python": "QuickIK (Python)",
    "quickik-cpp": "QuickIK (C++)",
    "kdl": "KDL",
    "pinocchio": "Pinocchio (Python)",
    "pinocchio-cpp": "Pinocchio (C++)",
    "rbdl": "RBDL (C++)",
    "rbdl-python": "RBDL (Python)",
}
QUICKIK_COLOR = "#4051b5"
OTHER_COLOR = "#888888"
TEXT_NUMBER_FONTSIZE = 8
# Open Sans isn't bundled with matplotlib (unlike DejaVu Sans, its default);
# fetched on demand into fonts/ (gitignored, not vendored) rather than
# assumed installed system-wide -- see register_fonts().
FONT_FAMILY = "Open Sans"
FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# Every result JSON has a "body" field (see ../preprocessing/README.md for
# g1, ../scripts/generate_fixtures.py for neuromechfly); this is just the
# figure suptitle for each.
BODY_TITLES = {
    "neuromechfly": "NeuroMechFly (42 limb DOFs)",
    "g1": "Unitree G1 (29-DOF humanoid)",
}


def load_results():
    all_results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        # quickik-scaling.json is a list of weak-scaling data points (see
        # plot_scaling.py), not a single-library result -- skip it here.
        if isinstance(data, dict):
            all_results.append(data)
    included = [r for r in all_results if r.get("formulation") == "whole-tree"]
    excluded = [r for r in all_results if r not in included]
    included.sort(key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER))
    return included, excluded


def group_by_body(results):
    groups = {}
    for r in results:
        groups.setdefault(r.get("body", "neuromechfly"), []).append(r)
    return groups


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


def round_sig(value, sig=3):
    """Rounds to `sig` significant digits -- e.g. round_sig(6405, 3) == 6410,
    round_sig(0.1498, 3) == 0.15."""
    if value == 0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    decimals = sig - 1 - exponent
    factor = 10**decimals
    return round(value * factor) / factor


def format_value(value, unit=None, small_unit=None, sig=3):
    """Fixed-point, never scientific, no thousand separator, no parens,
    rounded to `sig` significant digits -- e.g. '6.41 ms' or '32.9 kFPS',
    never '1.06e+04' or '6,405.2'.

    If `small_unit` is given and `value` is under 1 (in `unit`), displays in
    `small_unit` instead (x1000) -- e.g. '110 μs' rather than '0.110 ms' --
    purely a label readability choice; the bar's own length stays in `unit`
    regardless."""
    if small_unit and unit and abs(value) < 1:
        value, unit = value * 1000, small_unit
    rounded = round_sig(value, sig)
    if rounded == 0:
        text = "0"
    else:
        decimals = max(0, sig - 1 - math.floor(math.log10(abs(rounded))))
        text = f"{rounded:.{decimals}f}"
    return f"{text} {unit}" if unit else text


def despine(ax, hidden=("top", "right")):
    for side in hidden:
        ax.spines[side].set_visible(False)


# Charts are saved as SVG with fonttype="none" (see plot_chart()), so this
# family list -- not just its first entry -- ends up verbatim in the SVG's
# font-family CSS: matplotlib writes the whole list, and its own generic
# "sans-serif" keyword expands to a further list of real font names. That
# matters because the SVG is displayed in the *viewer's* browser, which may
# not have Open Sans available (blocked/failed webfont fetch, no matching
# local install) -- with only "Open Sans" and no fallback, a failed match
# falls through to the browser's own document default, which is commonly a
# *serif* font, not a generic sans one.
FONT_FALLBACK_CHAIN = ["-apple-system", "BlinkMacSystemFont", "Helvetica", "Arial", "sans-serif"]


def register_fonts():
    """Registers the locally-fetched Open Sans TTFs with matplotlib's font
    manager, if present, and returns the font family list (Open Sans first,
    if found) to assign to `rcParams["font.family"]`. Not vendored -- fetch
    once with:

        mkdir -p fonts
        curl -sL -o fonts/OpenSans-Regular.ttf \\
            "https://fonts.gstatic.com/s/opensans/v44/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsjZ0C4n.ttf"
        curl -sL -o fonts/OpenSans-Bold.ttf \\
            "https://fonts.gstatic.com/s/opensans/v44/memSYaGs126MiZpBA-UvWbX2vVnXBbObj2OVZyOOSr4dVJWUgsg-1y4n.ttf"
    """
    import logging

    import matplotlib.font_manager as fm

    # Every fallback name past the first is expected to be missing on
    # whatever machine renders this chart -- they're for the SVG's viewer,
    # not this process -- so the resulting "findfont: Font family ... not
    # found" spam is noise, not something to fix.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    ttfs = list(FONTS_DIR.glob("OpenSans-*.ttf"))
    if not ttfs:
        print(f"({FONT_FAMILY} not found under {FONTS_DIR} -- falling back to DejaVu Sans; see register_fonts()'s docstring to fetch it)")
        return ["DejaVu Sans"] + FONT_FALLBACK_CHAIN
    for ttf in ttfs:
        fm.fontManager.addfont(str(ttf))
    return [FONT_FAMILY] + FONT_FALLBACK_CHAIN


def draw_bars(ax, bar_names, bar_values, bar_colors, unit, small_unit=None, cap_name=None):
    """Draws one horizontal-bar panel on `ax` and annotates each bar with its
    value (see `format_value`) just past its tip -- plain "xxx unit", no
    parens.

    If `cap_name` names one of `bar_names`, that bar is instead drawn at a
    shortened length -- `CAP_MULTIPLE`x the next-highest bar's value -- since
    running it out to its true value would flatten every other bar down near
    zero on a linear axis. Only that one bar's label is different: its true
    value followed by "(capped in chart)", e.g. "66.7 ms (capped in
    chart)" -- every other bar keeps the plain format. The caller is
    expected to set this panel's x-axis limit to exactly this bar's
    (truncated) length, so the bar runs flush to the plot's right edge
    rather than stopping short of it -- the visual cue that it's cut off,
    not actually that short.

    Returns `(bars, widest)`, where `widest` is the widest x-extent actually
    drawn (unpadded, so the caller decides how much room to leave past it).
    """
    display_values = list(bar_values)
    cap_idx = bar_names.index(cap_name) if cap_name in bar_names else None
    if cap_idx is not None:
        display_values[cap_idx] = max(v for i, v in enumerate(bar_values) if i != cap_idx) * CAP_MULTIPLE

    bars = ax.barh(bar_names, display_values, color=bar_colors, height=0.55)
    for i, bar in enumerate(bars):
        text = format_value(bar_values[i], unit, small_unit)
        if i == cap_idx:
            # Inside the bar, right-justified against its (truncated) tip,
            # in white for contrast against the fill -- not past the tip
            # like every other bar's label, since the bar itself is already
            # marked as truncated via "(capped in chart)".
            ax.annotate(
                f"{text} (capped in chart)",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(-4, 0),
                fontsize=TEXT_NUMBER_FONTSIZE,
                textcoords="offset points",
                color="white",
                ha="right",
                va="center",
            )
            continue
        ax.annotate(
            text,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            fontsize=TEXT_NUMBER_FONTSIZE,
            textcoords="offset points",
            ha="left",
            va="center",
        )
    return bars, max(display_values)


def plot_chart(results, body):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed -- skipping chart; the table above is still complete)")
        return

    plt.rcParams["font.family"] = register_fonts()
    # SVG output, not PNG: a vector chart is a fraction of the file size and
    # stays crisp at any zoom. fonttype="none" keeps text as real <text>
    # elements (referencing the Open Sans family by name) instead of
    # converting every glyph to its own vector path, which would bloat a
    # text-heavy chart like this one past the PNG's size -- the docs site
    # loads Open Sans itself (see zensical.toml) so the family resolves
    # correctly in the browser.
    plt.rcParams["svg.fonttype"] = "none"

    # results is already in ORDER; reversed for barh so it reads top-to-bottom.
    ordered = list(reversed(results))
    names = [DISPLAY_NAMES.get(r["name"], r["name"]) for r in ordered]
    colors = [QUICKIK_COLOR if r["name"].startswith("quickik-") else OTHER_COLOR for r in ordered]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    # Manually tuned for this 2x2 layout: top=0.84 leaves room for the
    # suptitle; hspace=0.49 and wspace=0.35 keep panel titles/labels from
    # overlapping their neighbors.
    fig.subplots_adjust(hspace=0.49, wspace=0.35, left=0.09, right=0.97, top=0.84, bottom=0.08)

    for metric in METRICS:
        ax = axes[metric["row"], metric["col"]]
        key, title, xlabel, unit, small_unit = metric["key"], metric["title"], metric["xlabel"], metric["unit"], metric["small_unit"]
        values = [r.get(key) for r in ordered]
        bar_names = [n for n, v in zip(names, values) if v is not None]
        bar_values = [v * metric["scale"] for v in values if v is not None]
        bar_colors = [c for c, v in zip(colors, values) if v is not None]

        if not bar_values:
            ax.set_title(f"{title}\n(no data)")
            continue

        cap_name = CAPPED_BARS.get((body, key))
        _, widest = draw_bars(ax, bar_names, bar_values, bar_colors, unit, small_unit, cap_name=cap_name)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        despine(ax)
        # No padding past a capped bar's tip: running the axis flush to its
        # (truncated) length is what visually shows it's cut off rather than
        # actually that short, instead of adding a triangle or other marker.
        is_capped = cap_name in bar_names
        ax.set_xlim(0, widest if is_capped else widest * XLIM_PAD)

    fig.suptitle(BODY_TITLES.get(body, body), fontsize=14, fontweight="bold")
    out_path = RESULTS_DIR / f"comparison_{body}.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    results, excluded = load_results()
    results_by_body = group_by_body(results)
    excluded_by_body = group_by_body(excluded)
    for body, body_results in results_by_body.items():
        print(f"\n=== {BODY_TITLES.get(body, body)} ===")
        print_table(body_results, excluded_by_body.get(body, []))
        plot_chart(body_results, body)
