# Plan

## Outcome

Preserve the existing complete Native/Zero-HP one-host captures, then make the
decomposed Zero-HP P59 update readable in XProf. The change is observational:
it may add JAX host annotations under the existing `CANON_XPROF_LABELS=1`
contract, but it must not merge JITs, change shard maps, modify fixed reduction,
add `block_until_ready`, alter semantic Perfetto vocabulary, or claim causal
Native-vs-Zero timing from the existing input-mismatched pair.

## Prior completed carrier

- Native and Zero-HP each completed 3/3 optimizer updates on Qwen3-1.7B
  DP4xTP1.
- Native has 16 monolithic `jit__train_step` modules per TPU plane.
- Zero-HP has layer/head/norm/embed/adjoint P59 backward families on every TPU
  plane and passed 51/51 strict alignment records.
- The pair remains `INCONCLUSIVE_INPUT_MISMATCH`; readability work does not
  change that verdict or authorize a timing ratio.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| [P60-2A](phases/p60-2a-readability-baseline.md) | Frozen readability diagnosis from the existing full XPlanes | Native/Zero plane and host-annotation counts reproduced from immutable artifacts | passed |
| [P60-2B](phases/p60-2b-hierarchy-instrumentation.md) | Profile-only `train -> zero_tim_update -> phase -> group -> transaction` annotations and mechanical tests | Host/static, flag audit, P59 numerical controls, exact-image, and no-new-sync review pass; the census requires the complete hierarchy on `/host:CPU` `python3`, the exact host `train(step_num=1)` API, and 8/8 non-empty device `Steps` rows | passed (local) |
| [P60-2C](phases/p60-2c-onehost-visual-certification.md) | One fresh Zero-HP one-host capture with hierarchy census and UI checklist | 3/3 commits, 51/51 alignment PASS, 8/8 backward planes, exact hierarchy counts, decode absent; optimizer-tail drop was not yet a mechanical gate | passed (dirty-tree analysis grade) |
| [P60-2D](phases/p60-2d-attribution-and-next-decision.md) | Deterministic hierarchy summary and performance follow-up decision | All-plane stage report produced; any optimization is split into a new single-variable phase | pending |
| [P60-2E](phases/p60-2e-microstep-readability.md) | Truthful accumulator microstep and optimizer-update metadata without pretending Zero-HP is Native's monolithic graph | CPU/negative controls, exact-image 16-span receipt, old-artifact fail-closed hierarchy probe, 8/8 device planes with scaled-step×16 + commit×1, flag/diff/no-sync gates | core target gates pass; evidence packaging red |
| [P60-2F](phases/p60-2f-evidence-ledger-finalization.md) | Fail-closed terminal marker and immutable evidence-ledger finalization | GREEN/RED/tamper CPU controls, immediate `sha256sum -c`, no post-manifest hashed writes, exact-image and static gates | historical clean-SHA target pass (`5549b5b6`); latest-tip integration exact-image admitted (`c87838d8`), target not rerun |

## Decisions

- Confirmed: this is a readability defect, not a missing-backward capture. The
  current all-plane census proves P59 backward is present.
- Confirmed: stock Native enters `PeftTrainer.train`, which supplies
  `StepTraceAnnotation("train")`; the G6 Zero-HP path bypasses that loop and
  currently emits only a separate semantic-Perfetto `peft_train` span.
- Confirmed: `CANON_XPROF_LABELS=1` currently names individual JIT modules but
  does not provide parent update/group host spans.
- Decision: extend the existing observational flag instead of adding another
  flag. Absent/empty/`0` must remain an exact no-op.
- Decision: match only Native's host annotation API,
  `StepTraceAnnotation("train", step_num=<global step>)`. This does not match
  Native's microstep cadence, cardinality, or monolithic program shape. Require
  the complete hierarchy on one `/host:CPU` `python3` track and a non-empty
  `Steps` line on every one of the eight TPU device planes. Numeric `Steps`
  events are not semantic stage labels; the bounded host hierarchy supplies
  those labels.
- Decision: keep semantic Perfetto's official flat vocabulary unchanged. The
  new hierarchy belongs in JAX/XProf host annotations only; the reverted P55
  custom semantic spans must not return.
- Decision: no layer-per-span explosion. One model-backward span per group is
  sufficient because existing module names already identify layers 00..27.
- Decision: do not add synchronization to make spans visually line up. If
  asynchronous issue intervals prove insufficient, fail P60-2C and build a
  derived view from timestamps rather than perturb the measured schedule.
- Decision: do not fabricate Native's 16 monolithic `train` microsteps. Keep
  one whole-update `train(step_num=global_step)` and label the real Zero-HP
  accumulator sinks with `micro_step` plus a unique last-accumulator bit; keep
  optimizer commit separate and label it with `update_step`.
- Decision: make optimizer-tail capture completeness mechanical. Every one of
  the exact eight TPU planes must contain the five backward families,
  `jit__precomputed_gradient_scaled_step` exactly 16 times, and
  `jit__precomputed_gradient_commit` exactly once, with decode absent.
- Decision: a terminal GREEN marker is an execution/classifier result, not by
  itself an acceptance receipt. Acceptance also requires runner exit 0,
  `SHA_LEDGER_PASS`, and an independently verifiable root `SHA256SUMS` created
  only after the unique terminal marker is frozen in `driver.log`.
- Hypothesis: one top-level `StepTraceAnnotation` plus bounded group/stage
  `TraceAnnotation` scopes will restore navigation while adding only tens of
  host events to a capture that currently contains 59,028 TPU module events.

## Analysis authority

- Primary operating method:
  `/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md`.
  Its rules are binding for this phase: use `phase=update` for backward,
  inspect the complete XPlane and all eight TensorCore planes, use the
  semantic Perfetto only for the official high-level timeline, and use
  unprofiled `[PERF]` records rather than profile-wall time for performance
  decisions.
- Secondary paired-analysis method: the installed `xprof-trace-analysis`
  skill. It may summarize Native versus Zero-HP program shape, but it cannot
  turn the existing input-mismatched pair into a causal timing result.
- A host annotation is an issue/schedule envelope. It is not device busy time.
  Device attribution remains grounded in TPU `XLA Ops` and `XLA Modules`,
  joined to the host intervals without summing overlapping views.
