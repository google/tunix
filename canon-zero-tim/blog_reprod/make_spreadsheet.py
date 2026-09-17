#!/usr/bin/env python3
"""Build figure4.xlsx: one raw W&B tab per arm, plus a README tab with charts.

Each arm tab is that arm's own W&B export, unabridged: every column of
runs/<run_id>/history.csv, on the 200 rows data/manifest.json says Figure 4
plotted, written as the export's own strings so a reader sees the digits that
were drawn. Only the column order is ours - identity, then the five Figure 4
columns, then the arm's training-dynamics namespace, then everything else
grouped by prefix - and only four columns are derived: display_step, numeric
twins of the two plotted string columns so the embedded charts have something to
plot, and the ten-step trailing mean, which reuses moving_average() from the
figure builder so its last row is the endpoint label the figure prints.

Run from this directory (needs openpyxl and PyYAML):
    python make_spreadsheet.py            # writes ./figure4.xlsx
    python make_spreadsheet.py --output /tmp/other.xlsx
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, Series
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

import build_end_to_end_results_measured as figure
# Same trailing-mean function the figure's foreground curve and endpoint labels
# use (build_end_to_end_results_measured re-exports it from this module).
from build_end_to_end_results_provisional import moving_average

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
WANDB_PROJECT = "zero-tim-p57-frozenlake-tim"
ARMS = (("standard", "Standard"),
        ("importance_sampling", "TIS"),
        ("zero_tim", "Zero-TIM"))
SHEETS = ("README",) + tuple(label for _, label in ARMS)

# Column order on an arm tab. IDENTITY and FIGURE_COLUMNS are the frozen block
# the freeze pane keeps on screen; everything after it follows the arm's own
# order inside each namespace.
DISPLAY = "display_step"
TRAIL = "derived:solve_ratio_trail10"
# Numeric twins of the two plotted string columns. The strings stay
# authoritative; a chart cannot plot text, so it reads these instead.
SOLVE_NUM = "derived:solve_ratio_num"
MEAN_NUM = "derived:logp_diff_mean_num"
NUMERIC_TWINS = {SOLVE_NUM: figure.REWARD, MEAN_NUM: figure.MEAN}
DERIVED = (DISPLAY, SOLVE_NUM, TRAIL, MEAN_NUM)
IDENTITY = ("_step", "_runtime", "_timestamp")
FIGURE_COLUMNS = (figure.REWARD, SOLVE_NUM, TRAIL,
                  figure.MEAN, MEAN_NUM, figure.MAXIMUM, figure.BYTES)
DYNAMICS_PREFIXES = ("actor/train/", "canonical/train/")
ARM_SPECIFIC_PREFIX = "sampler_is/"
FIRST_DATA_ROW = 2
LAST_DATA_ROW = FIRST_DATA_ROW + figure.STEPS - 1

GROUP_LEGEND = (
    (DISPLAY + ", " + ", ".join(IDENTITY),
     "identity: display step 1-200, W&B's own step counter, seconds since the run "
     "started, and the Unix timestamp of the log call."),
    (figure.REWARD,
     "Figure 4, right panel, faint line: fraction of the update's 256 trajectories "
     "that reached the goal."),
    (SOLVE_NUM,
     "the same solve rate as a number rather than text, derived here so a chart can "
     "read it."),
    (TRAIL,
     "Figure 4, right panel, bold line: ten-step trailing mean of the solve rate. "
     "Derived here, not exported by W&B."),
    (figure.MEAN,
     "Figure 4, left panel: mean absolute sampler-trainer logprob difference over the "
     "sampled action tokens of that update."),
    (MEAN_NUM,
     "the same mean difference as a number rather than text, derived here so the left "
     "chart on the README tab can read it."),
    (figure.MAXIMUM,
     "Figure 4 companion: the worst single-token logprob difference in the same "
     "update. Not drawn, but it bounds the mean."),
    (figure.BYTES,
     "Figure 4 companion, Zero-TIM only: bytes differing between the sampler's and "
     "the trainer's logprob tensors. Zero on every plotted step."),
    ("actor/train/*",
     "training dynamics for Standard and TIS: loss, gradient norm, clipping, KL and "
     "the zero_tim censuses. Logged one step after the rewards of the same update."),
    ("canonical/train/*",
     "training dynamics for Zero-TIM, which logs this namespace instead of "
     "actor/train/*: commit gradient norm, parameter delta, alignment census."),
    ("sampler_is/*",
     "token-level sampler importance weights and clipping fraction. Only the TIS arm "
     "logs them; the other two tabs have no such columns."),
    ("frozenlake_eval/*",
     "held-out FrozenLake evaluation. Standard and TIS evaluate every 50 steps, so "
     "these cells are empty on most training rows; Zero-TIM never evaluates."),
    ("generation/*", "prompt and completion lengths and the generation clip ratio."),
    ("perf/*", "wall-clock seconds per phase of the update."),
    ("rewards/*",
     "solve counts, solve fractions and advantage statistics, on train and - where "
     "the arm evaluates - eval."),
    ("sampler_trainer/*",
     "sampler-versus-trainer agreement: logprob and probability differences and their "
     "Pearson correlation."),
    ("trajectory/*", "environment and reward-function latencies per update."),
    ("trajectory_rewards/*", "per-trajectory reward statistics for the update."),
)

SEMANTIC_NOTES = (
    ("_step",
     "W&B step 0-199 on these rows is display step 1-200; the export itself runs to "
     "_step 300 (Standard, TIS) or 215 (Zero-TIM), outside the plotted window."),
    ("actor/train/* offset",
     "these are logged one step after the rewards/* of the same update, so 199 of the "
     "200 rows carry them - the first row is empty - and the export's _step=300 row "
     "carries only actor metrics."),
    ("Zero-TIM namespace",
     "the Zero-TIM arm has no actor/train/* at all; canonical/train/* is its "
     "training-dynamics namespace, and it is populated on all 200 rows."),
    ("trailing mean",
     "derived:solve_ratio_trail10 averages the available prefix for the first nine "
     "steps, then a full ten-step window - exactly what the figure draws."),
    ("empty cells",
     "an empty cell is empty in the W&B export too: an evaluation row the arm did not "
     "write, or a metric that arm never logged. Nothing was filled in."),
    ("number strings",
     "W&B cells hold the export's own text, so a spreadsheet flags them as numbers "
     "stored as text and will not plot them. Reformatting them here would change the "
     "digits on display, so they stay as exported and the two *_num columns carry the "
     "same values as numbers for charting."),
    ("numeric precision",
     "the four derived columns are numbers, written at openpyxl's 16-significant-digit "
     "precision, so a *_num cell can differ from its string in the 17th digit: TIS "
     "display step 7 logs 0.016703682020306587 and stores 0.01670368202030659. Read "
     "the string columns for the exported digits, the numeric ones to plot."),
    ("regeneration",
     "re-running make_spreadsheet.py changes the file's bytes - openpyxl stamps the "
     "wall clock into docProps and the zip member timestamps - but not the contents "
     "of any sheet."),
)


def cell(value):
    """Render a manifest scalar for a spreadsheet cell without inventing values."""
    return "null" if value is None else value


def read_history(entry: dict) -> tuple[list, list]:
    """Return the arm's full W&B header and the 200 rows the manifest plotted."""
    path = ROOT / "runs" / entry["run_id"] / "history.csv"
    blob = path.read_bytes()
    figure.require(figure.digest(blob) == entry["source_history_sha256"],
                   f"{path.name}: history hash does not match the manifest")
    lines = list(csv.reader(io.StringIO(blob.decode("utf-8"))))
    header = lines[0]
    records = []
    for number in entry["source_csv_lines"]:
        row = lines[number - 1]
        figure.require(len(row) == len(header), f"{path.name}: ragged line {number}")
        records.append(dict(zip(header, row)))
    figure.require(len(records) == figure.STEPS, f"{path.name}: wrong plotted row count")
    return header, records


def order_columns(header: list) -> tuple[list, list]:
    """Split an arm's columns into the frozen leading block and the rest."""
    leading = [DISPLAY, *IDENTITY]
    leading += [name for name in FIGURE_COLUMNS if name in DERIVED or name in header]
    used = set(leading)
    prefix = next(candidate for candidate in DYNAMICS_PREFIXES
                  if any(name.startswith(candidate) for name in header))
    dynamics = [name for name in header if name.startswith(prefix) and name not in used]
    used.update(dynamics)
    specific = [name for name in header
                if name.startswith(ARM_SPECIFIC_PREFIX) and name not in used]
    used.update(specific)
    groups = {}
    for name in header:
        if name not in used:
            groups.setdefault(name.split("/")[0], []).append(name)
    rest = [name for group in sorted(groups) for name in groups[group]]
    trailing = dynamics + specific + rest
    figure.require(set(leading + trailing) == set(header) | set(DERIVED),
                   "column order dropped or invented a column")
    figure.require(len(leading) + len(trailing) == len(header) + len(DERIVED),
                   "column order duplicated a column")
    return leading, trailing


def write_arm(book, label: str, header: list, records: list, plotted: list, trail) -> list:
    """Write one arm's raw W&B rows; only display_step and the trail are derived."""
    leading, trailing = order_columns(header)
    columns = leading + trailing
    sheet = book.create_sheet(label)
    sheet.append(columns)
    for heading in sheet[1]:
        heading.font = Font(bold=True)
    for index, record in enumerate(records):
        step = index + 1
        figure.require(int(float(record["_step"])) + 1 == step, f"{label}: steps out of step")
        figure.require(trail[index].update == step, f"{label}: trailing mean out of step")
        # The raw rows must be the very rows the figure plotted.
        for name in figure.FIELDS:
            if name in header:
                figure.require(record[name] == plotted[index].get(name, ""),
                               f"{label}: {name} differs from the plotted CSV at step {step}")
        row = []
        for name in columns:
            if name == DISPLAY:
                row.append(step)
            elif name == TRAIL:
                row.append(trail[index].value)
            elif name in NUMERIC_TWINS:
                row.append(float(record[NUMERIC_TWINS[name]]))
            else:
                row.append(record[name] or None)
        sheet.append(row)
    figure.require(sheet.max_row == LAST_DATA_ROW, f"{label}: wrong row count")
    sheet.freeze_panes = sheet.cell(row=FIRST_DATA_ROW, column=len(leading) + 1)
    for position, name in enumerate(columns[:9], 1):
        sheet.column_dimensions[get_column_letter(position)].width = min(34, max(12, len(name) + 2))
    return columns


def readme_blocks(manifest: dict, trail: dict) -> list:
    """The README tab as titled blocks: (title, optional header row, rows)."""
    labels = " / ".join(f"{label} {trail[arm][-1].value * 100:.1f}%" for arm, label in ARMS)
    how = [
        ("x axis", f"{DISPLAY} (column A of the {', '.join(SHEETS[1:])} tabs), 1 to 200."),
        ("left panel, y", f"{figure.MEAN}, one line per arm, unsmoothed."),
        ("right panel, faint y", f"{figure.REWARD}, the raw training solve rate."),
        ("right panel, bold y", f"{TRAIL}, its ten-step trailing mean."),
        ("which columns to plot",
         f"plot {MEAN_NUM}, {SOLVE_NUM} and {TRAIL}: a spreadsheet cannot plot the W&B "
         "string columns. The strings hold the exported digits and stay authoritative; "
         "a *_num cell is that string at 16 significant digits, so the two can differ "
         "in the 17th - TIS display step 7 logs 0.016703682020306587 and stores "
         "0.01670368202030659."),
        ("endpoint labels", f"{labels} - the bold line's value on row {LAST_DATA_ROW}."),
        ("charts on this sheet",
         "the two line charts on the right are those two panels, drawn by the "
         "spreadsheet straight from the arm tabs."),
        ("published figure",
         "build_end_to_end_results_measured.py --data-dir data draws figures/figure4.svg "
         "from the same rows."),
    ]
    provenance_header = ["arm", "run_id", "wandb_project", "executed_source_sha",
                         "source_history_sha256", "source_config_sha256",
                         "source_csv_line_first", "source_csv_line_last",
                         "archive_commit", "evidence_grade"]
    provenance = []
    for arm, label in ARMS:
        entry = manifest["runs"][arm]
        lines = entry["source_csv_lines"]
        provenance.append([label, entry["run_id"], WANDB_PROJECT,
                           entry["run_id"].rsplit("-", 1)[-1],
                           entry["source_history_sha256"], entry["source_config_sha256"],
                           lines[0], lines[-1], manifest["source_commit"],
                           manifest["evidence_grade"]])
    recipe_header = ["config_key"] + [label for _, label in ARMS] + ["differs"]
    configs = [manifest["runs"][arm]["config"] for arm, _ in ARMS]
    for other in configs[1:]:
        figure.require(set(other) == set(configs[0]), "arms disagree on config keys")
    recipe = []
    for key in configs[0]:
        values = [config[key] for config in configs]
        recipe.append([key] + [cell(value) for value in values]
                      + [not all(value == values[0] for value in values)])
    return [
        ("Figure 4 - how to plot it from these tabs", None, how),
        ("Column groups, in the order each arm tab uses", None, list(GROUP_LEGEND)),
        ("Provenance", provenance_header, provenance),
        ("Recipe - the 24 recorded config keys", recipe_header, recipe),
        ("Semantic notes", None, list(SEMANTIC_NOTES)),
    ]


def write_readme(sheet, blocks) -> None:
    for index, (title, header, rows) in enumerate(blocks):
        if index:
            sheet.append([])
        sheet.append([title])
        sheet.cell(row=sheet.max_row, column=1).font = Font(bold=True)
        if header is not None:
            sheet.append(header)
            for heading in sheet[sheet.max_row]:
                heading.font = Font(bold=True)
        for row in rows:
            sheet.append(list(row))
    sheet.column_dimensions["A"].width = 44
    sheet.column_dimensions["B"].width = 66


def add_chart(book, sheet, title, column, y_title, anchor, limits=None) -> None:
    """One native line chart: the same column from each arm tab, rows 2-201."""
    chart = LineChart()
    chart.title = title
    chart.style = 2
    chart.height, chart.width = 9, 18
    chart.x_axis.title = DISPLAY
    chart.y_axis.title = y_title
    if limits is not None:
        chart.y_axis.scaling.min, chart.y_axis.scaling.max = limits
    for _, label in ARMS:
        tab = book[label]
        position = [heading.value for heading in tab[1]].index(column) + 1
        values = Reference(tab, min_col=position,
                           min_row=FIRST_DATA_ROW, max_row=LAST_DATA_ROW)
        chart.append(Series(values, title=label))
    chart.set_categories(Reference(book[SHEETS[1]], min_col=1,
                                   min_row=FIRST_DATA_ROW, max_row=LAST_DATA_ROW))
    sheet.add_chart(chart, anchor)


def build(output: Path) -> Path:
    selected, manifest = figure.load_exported_data(DATA)
    run = figure.make_workload(selected)
    trail = {arm: moving_average(run.solve_rate[figure.DRAWING_KEYS[arm]])
             for arm, _ in ARMS}

    book = Workbook()
    readme = book.active
    readme.title = "README"
    for arm, label in ARMS:
        header, records = read_history(manifest["runs"][arm])
        write_arm(book, label, header, records, selected[arm], trail[arm])
    write_readme(readme, readme_blocks(manifest, trail))
    add_chart(book, readme, "Sampler-trainer mean |Δlogp|", MEAN_NUM,
              "mean |Δ logprob|", "M2")
    add_chart(book, readme, "Training solve rate, trailing-10 mean", TRAIL,
              "solve rate", "M22", limits=(0, 1))

    figure.require(book.sheetnames == list(SHEETS), f"unexpected sheets {book.sheetnames}")
    book.save(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figure4.xlsx")
    args = parser.parse_args()
    path = build(args.output)
    print(f"PASS: wrote {path} ({path.stat().st_size} bytes) from {DATA}")


if __name__ == "__main__":
    main()
