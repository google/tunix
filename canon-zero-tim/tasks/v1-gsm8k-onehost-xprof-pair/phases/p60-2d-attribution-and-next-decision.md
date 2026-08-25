# P60-2D — Turn the hierarchy into an attribution decision

- Status: pending

## Finding

- Confirmed: XProf is causal/shape attribution evidence, not an A/B stopwatch.
- Confirmed: the historical Native/Zero pair consumed different completion,
  mask, and advantage arrays, so no timing ratio may be inferred from it.
- Hypothesis: once the Zero-HP update has stable parent/group spans, an offline
  summary can distinguish real model/reducer work from report/evidence/tree
  glue and identify the next single-variable performance experiment.

## Execution

1. Add a streaming/all-plane summary tool that consumes the full XPlane and
   hierarchy census, not only the bounded trace JSON. Follow the read-xprof
   tool-routing rules: inspect each script's Usage/docstring, reject hard-coded
   run paths, and record the exact script hash used for any scratch adaptation.
2. Emit deterministic JSON and text with:
   - host span counts and durations;
   - all-plane module counts and summed device durations by forward,
     model-backward, report-adjoint, fixed-reducer, replica-compare,
     accumulator, optimizer, and uncategorized families;
   - per-group distribution and first/last timestamps;
   - dropped-event and orphan-module checks.
3. Clearly label overlapping/summed device durations as attribution totals, not
   wall time. Use unprofiled `[PERF]` and global-step logs for speed decisions.
   `XLA Ops` is the directly sampled device line; host annotations, Modules,
   Steps, and semantic Perfetto are separate overlapping views and must not be
   added together.
4. Produce one decision record:

| Result | Next action |
|---|---|
| Report/evidence/tree glue dominates | propose one profile-backed dispatch-consolidation phase with full Zero-TIM recertification |
| Fixed reducer/replica compare dominates | profile the fixed-reduction transaction; do not change rank order |
| Model backward dominates | keep P59 structure; optimize only a measured kernel family under a separate flag |
| Attribution remains ambiguous | improve the offline join; do not change numerical code |

5. Keep the frozen-train-batch Native/Zero causal A/B as a separate deferred
   task. Do not fold it into this readability phase.
6. Use `xprof-trace-analysis` only after the read-xprof all-plane ledger is
   complete, and retain `INCONCLUSIVE_INPUT_MISMATCH` for the historical pair.

## Exit gate

- Command: run the deterministic summary twice on the same P60-2C artifact and
  compare output SHA-256, then run its synthetic orphan/drop negative controls.
- Pass: repeated summaries are byte-identical, all categories reconcile to the
  full all-plane module ledger, and exactly one next experiment is selected or
  the evidence explicitly says no optimization is justified.
- Fail: preserve raw profiles and stop; never force unclassified events into a
  convenient performance story.

## Result

Pending P60-2C.
