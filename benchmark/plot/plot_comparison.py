"""Aggregates every benchmark's results/<name>-<body>.json into a markdown
comparison table (one per body, always) and a single bar chart (if
matplotlib is available) comparing both bodies at once -- currently
`neuromechfly` (a fly) and `g1` (a Unitree G1 humanoid; see
../preprocessing/README.md for how its assets are generated). Every result
JSON carries a `"body"` field; this script groups by it and writes
`comparison.svg` with each library's bar replaced by a NeuroMechFly/G1 pair.

Only `formulation: "whole-tree"` results are compared here -- results
solving the body's actual problem (floating base + every limb/keypoint,
jointly). Any result with a different `formulation` (e.g. a fixed-base,
single-chain-only library that can't be reformulated to solve the whole-tree
problem) is excluded entirely, not just visually de-emphasized, and listed
separately in the printed table.

Usage (with devtools-pyenv/'s shared venv active):

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

# Padding past the widest bar's tip, as a fraction of that bar's value --
# leaves room for that bar's value label.
XLIM_PAD = 1.15

# One column, one row per metric: mean latency, then fixed-iteration
# latency, then single-thread throughput, then multi-thread throughput.
# "scale" converts the raw JSON units (us, fps) to the panel's own axis unit
# (ms, kFPS); a bar under 1 (in that unit) instead displays its label in
# "small_unit" (us, FPS) -- see format_value -- without changing the bar's
# own length, which always stays in the panel's one fixed axis unit.
METRICS = [
    {
        "key": "single_frame_latency_us",
        "title": "Mean single-frame latency (with early stopping)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
        "row": 0,
    },
    {
        "key": "single_frame_latency_max_us",
        "title": f"Single-frame latency (fixed {MAX_ITERATIONS_PER_SOLVE} iterations)",
        "xlabel": "Latency (ms)",
        "unit": "ms",
        "small_unit": "μs",
        "scale": 1e-3,
        "row": 1,
    },
    {
        "key": "single_thread_throughput_fps",
        "title": "Throughput (sequential frames, single thread)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
        "row": 2,
    },
    {
        "key": "multi_thread_throughput_fps",
        "title": "Throughput (sequential frames, 8 threads)",
        "xlabel": "Throughput (kFPS)",
        "unit": "kFPS",
        "small_unit": "FPS",
        "scale": 1e-3,
        "row": 3,
    },
]

# Metric key -> the library whose bar, in *both* bodies, should be drawn
# capped at CAP_MULTIPLE-x the next-highest bar's value (the max across
# every other library's bar in either body) instead of running out to its
# true value, which would flatten every other bar down near zero on a
# linear axis -- see `draw_body_bars`. Only KDL's mean-latency result is
# extreme enough (up to ~60x the next-highest bar, for g1) to need this;
# everywhere else a plain, uncapped bar is clearer.
CAPPED_METRIC_BARS = {"single_frame_latency_us": "kdl"}
CAP_MULTIPLE = 2

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
# Bar color now encodes which body a bar belongs to (every library's bar
# pair uses the same two colors); QuickIK is instead called out via
# HIGHLIGHT_COLOR, QUICKIK_ALPHA/OTHER_ALPHA, and bold text below, since it
# no longer has a color of its own.
NEUROMECHFLY_COLOR = "#4051b5"
G1_COLOR = "#2fb170"
# Background band behind QuickIK's bars (its 3 language variants, 2 bars --
# one per body -- each), since color no longer distinguishes it. zorder=0,
# entirely behind the (zorder=3) bars, so it only ever shows through the
# gaps between and around them.
HIGHLIGHT_COLOR = "#ffec3d"
HIGHLIGHT_ALPHA = 0.3
# QuickIK's bars are fully opaque; every other library's are dimmed, one
# more way (besides HIGHLIGHT_COLOR and bold text) that QuickIK stands out.
QUICKIK_ALPHA = 1.0
OTHER_ALPHA = 0.5
TEXT_NUMBER_FONTSIZE = 8
# Open Sans isn't bundled with matplotlib (unlike DejaVu Sans, its default);
# fetched on demand into fonts/ (gitignored, not vendored) rather than
# assumed installed system-wide -- see register_fonts().
FONT_FAMILY = "Open Sans"
FONTS_DIR = Path(__file__).resolve().parent / "fonts"

# Every result JSON has a "body" field (see ../preprocessing/README.md for
# g1, ../scripts/generate_fixtures.py for neuromechfly); used as each
# printed table's section header.
BODY_TITLES = {
    "neuromechfly": "NeuroMechFly (42 limb DOFs)",
    "g1": "Unitree G1 (29-DOF humanoid)",
}
# Shorter labels for the chart legend's two entries (title-appropriate
# detail belongs in BODY_TITLES above instead).
LEGEND_LABELS = {
    "neuromechfly": "NeuroMechFly (42 DOFs)",
    "g1": "G1 humanoid (29 DOFs)",
}


def load_results():
    all_results = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(path.read_text())
        # quickik-scaling.json is a list of weak-scaling data points (see
        # plot_scaling.py), not a single-library result -- skip it here.
        # errors-<body>.json is plot_2d_comparison.py's own per-frame
        # fit-residual distribution data (no "name"/"formulation" keys),
        # not a per-library result either -- skip that too. So is
        # quickik-rust-2d-xyview-<body>.json (an "observation" override of
        # quickik-rust's own 3D result, also only used by
        # plot_2d_comparison.py) -- without this it'd show up here as a
        # second, indistinguishable "quickik-rust" row.
        if isinstance(data, dict) and "name" in data and "observation" not in data:
            all_results.append(data)
    included = [r for r in all_results if r.get("formulation") == "whole-tree"]
    excluded = [r for r in all_results if r not in included]
    included.sort(
        key=lambda r: ORDER.index(r["name"]) if r["name"] in ORDER else len(ORDER)
    )
    return included, excluded


def group_by_body(results):
    groups = {}
    for r in results:
        groups.setdefault(r.get("body", "neuromechfly"), []).append(r)
    return groups


def print_table(results, excluded):
    if not results:
        print(
            f"No whole-tree results found in {RESULTS_DIR}. Run the benchmarks first."
        )
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
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths, strict=True)))
    print("-|-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(c.ljust(w) for c, w in zip(row, widths, strict=True)))

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
FONT_FALLBACK_CHAIN = [
    "-apple-system",
    "BlinkMacSystemFont",
    "Helvetica",
    "Arial",
    "sans-serif",
]


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
        print(
            f"({FONT_FAMILY} not found under {FONTS_DIR} -- falling back to DejaVu Sans; see register_fonts()'s docstring to fetch it)"
        )
        return ["DejaVu Sans"] + FONT_FALLBACK_CHAIN
    for ttf in ttfs:
        fm.fontManager.addfont(str(ttf))
    return [FONT_FAMILY] + FONT_FALLBACK_CHAIN


# Within each library's group, the NeuroMechFly bar sits directly above the
# G1 bar with no gap between them (BAR_HEIGHT must equal 2*BAR_OFFSET for
# their touching edges to land exactly on the group's integer y-position),
# leaving a visible gap only between groups.
BAR_OFFSET = 0.20
BAR_HEIGHT = 0.40

# A capped bar's "keeps going" dashes past its tip (see draw_body_bars):
# DASH_COUNT blocks, each DASH_WIDTH*cap_length wide, separated by
# DASH_GAP*cap_length (also used as the gap right after the bar's tip).
# Sized so the whole run -- DASH_GAP + DASH_COUNT*DASH_WIDTH +
# (DASH_COUNT-1)*DASH_GAP, as a fraction of cap_length -- lands comfortably
# inside XLIM_PAD's headroom past the widest bar.
DASH_COUNT = 5
DASH_WIDTH = 0.012
DASH_GAP = 0.01


def dash_positions(cap_length):
    """Left-edge x-positions for DASH_COUNT dashes continuing a capped bar
    of length `cap_length` past its own tip."""
    step = (DASH_WIDTH + DASH_GAP) * cap_length
    first = cap_length + DASH_GAP * cap_length
    return [first + i * step for i in range(DASH_COUNT)]


def draw_body_bars(
    ax,
    lib_names,
    y_positions,
    results_by_name,
    key,
    scale,
    unit,
    small_unit,
    color,
    cap_lib=None,
    cap_length=None,
):
    """Draws one body's bars (one per library in `lib_names`, at the matching
    `y_positions`) on `ax`, skipping libraries with no value for `key`, and
    annotates each with its value (see `format_value`) just past its tip --
    plain "xxx unit", no parens.

    If `cap_lib` names one of `lib_names` and `cap_length` is given, that
    bar is instead drawn at the (already-scaled) `cap_length` rather than
    its true length -- since running it out to its true value would flatten
    every other bar down near zero on a linear axis. The caller computes
    `cap_length` once per metric (shared across both bodies -- see
    CAPPED_METRIC_BARS), so both bodies' bars for `cap_lib` end up the same
    length even though their true values differ. That one bar's true value
    plus "(capped in chart)" is written inside it, and a few small
    same-color, same-height blocks continue past its tip (see
    dash_positions()) -- a dashed-line cue that the bar keeps going -- in
    place of the plain label every other bar gets there. Those blocks are
    deliberately excluded from this function's returned widest extent (they
    size themselves off of it, via dash_positions(cap_length)), so they land
    inside the caller's XLIM_PAD headroom rather than pushing it out further.

    Every QuickIK bar is drawn at QUICKIK_ALPHA with its value label bold;
    every other library's bar gets OTHER_ALPHA and a plain-weight label.

    Returns the widest x-extent actually drawn (unpadded, so the caller
    decides how much room to leave past it), or 0.0 if no library had data.
    """
    from matplotlib.patches import Rectangle

    present = [
        (y, lib, results_by_name[lib][key] * scale)
        for y, lib in zip(y_positions, lib_names, strict=True)
        if lib in results_by_name and results_by_name[lib].get(key) is not None
    ]
    if not present:
        return 0.0

    ys = [y for y, _, _ in present]
    values = [v for _, _, v in present]
    display_values = list(values)
    cap_idx = next((i for i, (_, lib, _) in enumerate(present) if lib == cap_lib), None)
    if cap_idx is not None and cap_length is not None:
        display_values[cap_idx] = cap_length

    bars = ax.barh(ys, display_values, height=BAR_HEIGHT, color=color, zorder=3)
    for i, (bar, (_, lib, _)) in enumerate(zip(bars, present, strict=True)):
        is_quickik = lib.startswith("quickik-")
        bar.set_alpha(QUICKIK_ALPHA if is_quickik else OTHER_ALPHA)
        weight = "bold" if is_quickik else "normal"
        text = format_value(values[i], unit, small_unit)
        if i == cap_idx and cap_length is not None:
            ax.annotate(
                f"{text} (capped in chart)",
                (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(-4, 0),
                fontsize=TEXT_NUMBER_FONTSIZE,
                fontweight=weight,
                textcoords="offset points",
                ha="right",
                va="center",
                zorder=4,
            )
            for dash_x in dash_positions(cap_length):
                ax.add_patch(
                    Rectangle(
                        (dash_x, bar.get_y()),
                        DASH_WIDTH * cap_length,
                        bar.get_height(),
                        facecolor=color,
                        alpha=bar.get_alpha(),
                        edgecolor="none",
                        zorder=3,
                    )
                )
            continue
        ax.annotate(
            text,
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0),
            fontsize=TEXT_NUMBER_FONTSIZE,
            fontweight=weight,
            textcoords="offset points",
            ha="left",
            va="center",
            zorder=4,
        )
    return max(display_values)


def plot_chart(results_by_body):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print(
            "\n(matplotlib not installed -- skipping chart; the table above is still complete)"
        )
        return

    nmf_by_name = {r["name"]: r for r in results_by_body.get("neuromechfly", [])}
    g1_by_name = {r["name"]: r for r in results_by_body.get("g1", [])}
    # ORDER, filtered to libraries actually present in either body; reversed
    # for barh so it reads top-to-bottom in ORDER (fastest-per-library
    # first).
    lib_names = [n for n in ORDER if n in nmf_by_name or n in g1_by_name]
    if not lib_names:
        print("\n(no whole-tree results found -- skipping chart)")
        return
    ordered_libs = list(reversed(lib_names))
    display_names = [DISPLAY_NAMES.get(n, n) for n in ordered_libs]
    group_ys = list(range(len(ordered_libs)))

    plt.rcParams["font.family"] = register_fonts()
    # SVG output, not PNG: a vector chart is a fraction of the file size and
    # stays crisp at any zoom. fonttype="none" keeps text as real <text>
    # elements (referencing the Open Sans family by name) instead of
    # converting every glyph to its own vector path, which would bloat a
    # text-heavy chart like this one past the PNG's size -- the docs site
    # loads Open Sans itself (see zensical.toml) so the family resolves
    # correctly in the browser.
    plt.rcParams["svg.fonttype"] = "none"

    # Width matches plot_scaling.py's chart (figsize=(8, 4)); height is
    # taller, one row per metric instead of 2x2.
    fig, axes = plt.subplots(4, 1, figsize=(8, 17))
    # Manually tuned for this 4x1 layout: top=0.94 leaves room for the
    # legend (no suptitle); hspace=0.4 keeps panel titles/labels from
    # overlapping their neighbors; left=0.24 fits the longest library label
    # ("Pinocchio (Python)") in this figure.
    fig.subplots_adjust(
        hspace=0.4, wspace=0.35, left=0.24, right=0.97, top=0.94, bottom=0.045
    )

    # QuickIK's 3 language variants (6 bars: NeuroMechFly + G1 each) sit last
    # in ordered_libs (ORDER's first 3, after the top-to-bottom reversal),
    # i.e. the topmost groups in every panel.
    quickik_count = sum(1 for n in ordered_libs if n.startswith("quickik-"))

    for metric in METRICS:
        ax = axes[metric["row"]]
        key, title, xlabel, unit, small_unit, scale = (
            metric["key"],
            metric["title"],
            metric["xlabel"],
            metric["unit"],
            metric["small_unit"],
            metric["scale"],
        )

        cap_lib = CAPPED_METRIC_BARS.get(key)
        cap_length = None
        if cap_lib:
            other_values = [
                r[key]
                for by_name in (nmf_by_name, g1_by_name)
                for name, r in by_name.items()
                if name != cap_lib and r.get(key) is not None
            ]
            if other_values:
                cap_length = CAP_MULTIPLE * max(other_values) * scale

        nmf_ys = [y + BAR_OFFSET for y in group_ys]
        g1_ys = [y - BAR_OFFSET for y in group_ys]
        widest_nmf = draw_body_bars(
            ax,
            ordered_libs,
            nmf_ys,
            nmf_by_name,
            key,
            scale,
            unit,
            small_unit,
            NEUROMECHFLY_COLOR,
            cap_lib,
            cap_length,
        )
        widest_g1 = draw_body_bars(
            ax,
            ordered_libs,
            g1_ys,
            g1_by_name,
            key,
            scale,
            unit,
            small_unit,
            G1_COLOR,
            cap_lib,
            cap_length,
        )
        widest = max(widest_nmf, widest_g1)
        if widest == 0.0:
            ax.set_title(f"{title}\n(no data)")
            continue

        if quickik_count:
            ax.axhspan(
                len(ordered_libs) - quickik_count - 0.5,
                len(ordered_libs) - 1 + 0.5,
                color=HIGHLIGHT_COLOR,
                alpha=HIGHLIGHT_ALPHA,
                zorder=0,
            )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_yticks(group_ys, labels=display_names, va="center")
        # QuickIK's groups sit last in ordered_libs (see quickik_count
        # above); bold their labels the same way their bar values are bold.
        for label in ax.get_yticklabels()[len(ordered_libs) - quickik_count :]:
            label.set_fontweight("bold")
        ax.set_ylim(-0.5, len(ordered_libs) - 1 + 0.5)
        despine(ax)
        # Even a capped bar's tip gets the usual padding now: its "..."
        # label past the tip (see draw_body_bars) needs the room, and the
        # in-bar "(capped in chart)" text already marks it as truncated, so
        # a flush-to-the-edge tip is no longer the only cue for that.
        ax.set_xlim(0, widest * XLIM_PAD)

    legend_handles = [
        Patch(facecolor=NEUROMECHFLY_COLOR, label=LEGEND_LABELS["neuromechfly"]),
        Patch(facecolor=G1_COLOR, label=LEGEND_LABELS["g1"]),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=2,
        frameon=False,
        fontsize=11,
    )
    out_path = RESULTS_DIR / "comparison.svg"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"\nWrote chart to {out_path}")


if __name__ == "__main__":
    results, excluded = load_results()
    results_by_body = group_by_body(results)
    excluded_by_body = group_by_body(excluded)
    for body, body_results in results_by_body.items():
        print(f"\n=== {BODY_TITLES.get(body, body)} ===")
        print_table(body_results, excluded_by_body.get(body, []))
    plot_chart(results_by_body)
