# P60-2C — Certify one readable Zero-HP update on one host

- Status: PASS on second Zero-HP canary (dirty-tree analysis grade)

## Finding

- Confirmed: the existing one-host carrier captures only update 1 of three real
  optimizer commits, with rollout outside the XProf device window.
- Hypothesis: the P60-2B scopes will add a small bounded host hierarchy while
  preserving the complete eight-plane P59 backward capture.

## Execution

1. Do not rerun Native. Its stock hierarchy and complete backward capture are
   already established; this phase tests only the repaired Zero-HP view.
2. Obtain explicit user approval for a direct four-chip v5p run.
3. From a clean executor-owned worktree and fresh label, run:

   ```bash
   export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
   bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
     '<fresh-p60-readable-zero-label>'
   ```

4. Require the old arm classifier, all-plane XProf census, semantic census, and
   new hierarchy census to pass before opening the UI.
5. Apply `/home/yuxuan/code_rl_repro/.claude/skills/read-xprof/SKILL.md` to the
   new run: verify one non-empty XPlane, record its byte size and SHA-256,
   inspect all eight TensorCore planes, and run the update-window backward
   census. Do not use binary `grep` or the bounded trace JSON as a completeness
   gate.
6. Inspect the full XPlane/trace in Trace Viewer. Starting at `train`, verify
   that `zero_tim_update` contains the forward, loss, 16 reverse-group, reducer,
   accumulator, and optimizer regions. Selecting one reverse group must expose
   the existing head/norm/layer/embed/adjoint device modules in the same time
   neighborhood.
7. Preserve the full run root, including XPlane, trace JSON, semantic Perfetto,
   raw log, all censuses, classification, and SHA ledger. Screenshots are
   optional side evidence and never replace the raw capture.

## Exact gates

| Gate | Pass criterion |
|---|---|
| Training | 3/3 optimizer updates |
| Zero-TIM | 51/51 alignment PASS; zero FAIL |
| Device completeness | five P59 backward families on 8/8 TensorCore planes |
| Window | one update-only XProf; decode absent; dropped traces/events zero |
| Parent API | one Native API-compatible `train(step_num=1)` and one `zero_tim_update` |
| Host track | every hierarchy span on the same `/host:CPU` `python3` line |
| Device step rows | non-empty `Steps` line on 8/8 TPU planes |
| Groups | 16 forward groups and 16 reverse groups; all inside their parents |
| Transactions | 16 report adjoints, 16 fixed reductions, 16 accumulator sinks, one optimizer commit |
| Semantic Perfetto | existing official event counts unchanged; no custom span leftovers |
| Human check | update is navigable without using numeric `Steps` events as semantic labels |

Profiled wall time is not a speed verdict. Any later performance claim must
come from comparable unprofiled `[PERF]` steps and matched work; this phase
certifies readability and numerical neutrality only.

## Exit gate

- Command: the one-host Zero-HP launcher above followed by the task classifiers
  and hierarchy census embedded in its postflight.
- Pass: every exact gate in the table passes and the complete immutable run root
  is readable.
- Fail branches:
  - Numerical/alignment red: stop; instrumentation is not neutral.
  - Missing hierarchy with complete backward: instrumentation defect; fix
    locally, do not rerun Native.
  - Span/device timing misalignment: do not add a barrier. Proceed only to a
    derived timestamp view proposal.
  - Dropped trace or incomplete artifacts: infrastructure INCONCLUSIVE; preserve
    the run and repair the capture contract before retrying.

## Result

The first dirty development canary
`p60_readable_zero_dev1_20260825` completed 3/3 updates and 51/51 strict
alignment PASS. Its full XPlane is 778,688,563 bytes with SHA-256
`06b0c43c34361eab3a976d5870bd5b3b49a898500f741ab3676d06adc1da12a2`.
All eight TPU planes passed the P59 backward/decode census and had non-empty
Steps rows; every bounded child hierarchy count was exact. The hierarchy gate
failed because `train` and `zero_tim_update` both had count zero.

Confirmed cause: both JAX TraceMe parent objects were constructed immediately
before `_canon_xprof_update_entry()` opened the profiler. Their intervals
therefore began outside the capture, while all children constructed after the
start were retained. The local fix moves only those two object constructions
after the existing start call and adds a source-order regression gate. No
numerical expression, sync, window boundary, or child scope changed.

The failed root is preserved at
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_dev1_20260825`.
The fix passes the task suite 8/8, document set, 371/371 flag audit, branch
preflight, syntax, diff check, and the full pinned exact-image ladder including
labels-off/on, one-ULP negative, P59/V1 regressions, and the annotation API
probe. A fresh target retry requires separate approval. Native, Kubernetes,
commit, push, and image publication remain out of scope.

The user authorized exactly one fresh retry on 2026-08-25. Its bound label was
`p60_readable_zero_dev2_20260825`; the failed dev1 root remains immutable.

The second canary passed. It completed all 3/3 optimizer updates, emitted 3
pre-alignment plus 48 update-alignment PASS records with zero FAIL, and ended
with `TRAINING_DONE max_steps=3`, `docker_exit=0`, and wrapper GREEN. The
update-only capture was armed at step 1, started at update entry step 1, and
stopped at step-completed step 2.

The immutable PASS root is
`/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p60_readable_zero_dev2_20260825`.
Its full XPlane is 778,720,935 bytes with SHA-256
`4ee534ed81ff4e721a5482ac42057048382161d569f6c608c9c56f29e7aa38fd`;
the trace JSON is 33,809,800 bytes with SHA-256
`dc68ba730cb54108d6be27147cdf315871819ec24193880473ed61bbcfb240a2`;
and the official semantic Perfetto is 12,436 bytes with SHA-256
`c6f76f5845b988923aeea85dbf5723f76d91081aa77b84e8c1108b2a23aa38f8`.

All exact gates pass: one `train(step_num=1, _r=1)` contains one
`zero_tim_update`; it contains 16 forward groups, one loss pullback, 16
reverse groups, 16 replay/model-backward/report-adjoint/fixed-reduce/
accumulator transactions, and one optimizer commit in the required order.
All 8/8 TPU planes have non-empty `Steps` rows, all five P59 backward module
families are present, and decode is absent. The official semantic Perfetto
retains one profiled update and no custom hierarchy vocabulary.

A direct full-XPlane navigation check selected reverse group 0. Its time
window overlaps the existing head, norm, layer, embed, and adjoint backward
module families on every one of the eight TPU planes. This establishes the
requested Native API-compatible step annotation and navigable host-to-device
hierarchy without adding synchronization. It does not claim Native-compatible
cadence, cardinality, or monolithic program shape. Profiled wall time remains
non-causal and is not a Native/Zero performance result.

This pass is analysis-grade because the explicitly approved dirty-tree
override was used. It is not a signed clean-SHA receipt. The single retry
authorization is consumed; no additional TPU run, Native rerun, Kubernetes
action, commit, push, or image publication is authorized.
