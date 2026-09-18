# results — the raw W&B exports, one home for all of them

Every run this branch keeps evidence for is exported here in full, so any of them can be found,
checked and plotted without going back to W&B. `canon-zero-tim/blog_reprod/` is the frozen
Figure 4 package: it reads its three short-horizon runs from here, and nothing in this directory
is specific to it.

## Layout

```
canon-zero-tim/results/
├── README.md  RUNS.tsv  check_results.py  plot_curves.py  (+ one test each)
└── <workload>/                      frozenlake-short-horizon, …
    ├── curves.svg  summary.csv      written by plot_curves.py
    └── <arm>/<run-id>/              standard, tis, zero_tim, …
        ├── history.csv              one row per W&B step, _step contiguous from 0
        ├── config.yaml              run.config without the _wandb block
        └── summary.json             the run's final summary
```

Those three files are exactly what `canon-zero-tim/blog_reprod/export_wandb.py` writes, and
exactly what a run directory may hold: `canon-zero-tim/results/check_results.py` fails on a
missing file and on an extra one.

## RUNS.tsv

One tab-separated row per run directory, checked both ways — a directory with no row and a row
with no directory are both failures.

| column | meaning |
|---|---|
| `workload` `arm` `run_id` | the three path segments, in that order |
| `wandb` | `<entity>/<project>/<run id>`, where the run still is |
| `executed_sha` | the commit the pods executed, from the tail of the run name |
| `steps` | rows in `history.csv`, which the checker recounts |
| `grade` | see below |
| `date` | UTC date of the run's first `_timestamp` |
| `note` | one clause saying what the run is for |

The four grades: `complete` — the run reached its configured last update; `incomplete` — it was
exported early or it died; `legacy-code` — real data, from code this branch no longer carries;
`receipts-only` — console receipts survive, the W&B history does not.

## Adding a workload

```bash
python3 canon-zero-tim/blog_reprod/export_wandb.py --entity yuxzhang-google \
  --project <project> --run <run-id> --out canon-zero-tim/results/<workload>/<arm>/<run-id>
python3 canon-zero-tim/results/check_results.py
python3 canon-zero-tim/results/plot_curves.py --workload <workload>
```

1. Export every run with the first command, once each, with `WANDB_API_KEY` in the environment.
2. Append one `RUNS.tsv` row per run, filling the columns above.
3. Run the checker. expected: `RESULTS_CHECK PASS runs=<n>`.
4. Run the plotter. expected: `PLOT_CURVES PASS workload=<workload> runs=<n> …`; it writes that
   workload's `curves.svg` and `summary.csv`.
5. Commit the exports, the rows and those two outputs in one commit that touches no file under
   `canon-zero-tim/blog_reprod/`.
