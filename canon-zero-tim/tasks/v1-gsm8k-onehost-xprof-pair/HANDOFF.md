# Execution handoff: P74 one-host XProf and GSM8K DP16xTP4 full train

## Current v2 one-host default delivery (2026-09-08, local candidate)

This section supersedes the historical P74/P60 launch and mode guidance
below for normal v2 training. Final-source hardware certification is pending;
CPU/process tests are not a TPU or full-training certificate. Do not push or
launch a target campaign based on this note.

Use the common `scripts/run_onehost_gsm8k_xprof_zero_hp.sh` route (or
`scripts/run_onehost_xprof_backward_zero.sh` for its backward label wrapper).
For registered GSM8K DP4xTP1 and DP2xTP2 short/long/long8k, BOTH absent
`CANON_P32_KEEP_TAPE` and `CANON_DP_REDUCE_ONCE` now resolve to stream/1
before Docker forwarding and module/hierarchy census arguments. The shared
owner is `cluster/v1_full_system_optimization.py`; the inner profiles and
runtime parsers are unchanged. P71 remains the normal carrier's fwd_block
default and chunk batch remains2; do not copy the full-target fwd override.

Any explicit presence of either name preserves BOTH raw values, including
empty/0 and partial overrides. Native, signed capture/negative controls and
the P45-shaped GSM8K diagnostic do not acquire these defaults. FrozenLake
r0/r1/r2 are separate vehicles and are not selected by this launcher.

Certification must use the canonical onehost-geometry-certify sequence and
judge, continuous non-system idle120, one TPU container, a fresh label and
a clean physical worktree. Never stop someone else's container. Sequence
fields have the following meanings; unset the reducer in the parent for a
default test, not merely the keep field:

| Intent | keep field | extra field |
|---|---|---|
| Normal default, verify against explicit bundle | empty (unset) | empty, with parent reducer unset |
| Explicit optimized control | stream | CANON_DP_REDUCE_ONCE=1 |
| Explicit legacy/off control | 0 | CANON_DP_REDUCE_ONCE=0 |

A blank historical sequence field meant unset, not a permanent off value;
do not replay old control lists blindly or rewrite their historical evidence.
The P74-specialized wrapper deliberately retains P71=fwd and pins keep/reduce
off when both were absent. It is NOT the latest v2 optimized-default wrapper.

For DP2xTP2's current reverse-chunk program, use the registered
reduce-once+chunk anchor and mode-aware censuses, not the old P74 64-window
contract below. The three registered norms are
1.6838672161102295 / 3.302103281021118 / 1.8242988586425781.
Require strict alignment, checked-VMA receipts, actual program fingerprints,
terminal classification and SHA verification before considering HBM/timing.
Capture truncation never proves whole-update bubble/D2H absence.

DP1xTP4 GSM8K is not registered in this branch; do not substitute P66 or
another worktree. DP4's current chunk-mode anchor coverage remains pending.
The inherited DP4 profile now accepts its already-declared V1 six-update
horizon, retaining wrong-kind/tail/P66/no-commit refusals; six-update TPU
execution itself remains unverified. Three-update is the certification start.

## Historical P74 instructions and evidence

> **Current as of 2026-08-28:** use the P74 commands in the next two sections.
> The Native/Zero pair and P60 procedures later in this file are historical.
> P74 is accepted on the immutable matched r3/r4 DP2xTP2 captures, but a run
> from the newly published source SHA is still required before calling that
> exact SHA one-host-certified. Target DP16xTP4 performance remains TARGET NOT
> RUN for P74.

## Current A — one-host DP2xTP2 backward XProf with P74

Use a clean checkout of the approved pushed SHA whenever producing acceptance
evidence. The development-only dirty override is shown separately and must not
be used for a clean-SHA claim. Work from the physical worktree path, never a
symlink, and never append a pipe to the launch command.

First verify the one-host lane is idle. Do not stop or remove another user's
container:

```bash
sudo docker ps --format '{{.Names}}|{{.Image}}|{{.Status}}'
```

Then launch one fresh immutable label:

```bash
cd <physical-clean-worktree>
export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"

bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_xprof_backward_p74_dp2tp2.sh \
  "<fresh-host-date-label>"
```

For an intentionally dirty development checkout only:

```bash
V1_GSM8K_XPROF_ALLOW_DIRTY=1 \
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_xprof_backward_p74_dp2tp2.sh \
  "<fresh-development-label>"
```

The wrapper pins DP2xTP2, `fingerprint-hybrid`, `first-group-warmup`,
`batched-commit`, and `CANON_P71_SCAN=fwd`. It does not set or weaken checked
VMA: the signed DP2xTP2 profile requires `CANON_P66_P59_CHECK_VMA=1` and fails
closed on drift. `num_chunks` remains data-derived as
`ceil(max_real_tokens / local_M)` with `local_M=256`; P74 does not hard-code 2
or fuse chunks.

With the default evidence root, the run directory is:

```text
/mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_dp2tp2-ba_<label>
```

Return and inspect all of these:

```text
train/p74_gap_census.txt
train/p74_gap_receipt.json
train/classification.json
train/raw.log
train/xprof/
train/perf/
SHA256SUMS
```

The P74 receipt is fail-closed: exactly 64 seed-to-head windows, mean gap
`<=70ms`, one `jit__p74_identity_head_cotangent_partition` per window, and all
seven old D2H/H2D victim overlaps equal zero. It separately records the final
`[PERF] stage=p32_vag_reverse` row; do not add XProf gap and host wall columns.
The three commit-gradient norms must be bitwise
`1.6838101148605347 / 3.3025829792022705 / 1.8203867673873901`, strict
alignment must be 96/96, and `[P66.VMA] outer_check_enabled` must remain in
production logs. A capped trace JSON may keep the outer arm classifier RED at
`trace_census_rc=1`; that convention does not waive any P74, numerical,
full-XPlane hierarchy, or SHA-ledger red.

Historical matched truth for comparison: r3 mean gap 150.746ms/chunk,
726ms/group, reverse 24.458s; r4 mean 0.063ms/chunk, 459ms/group, reverse
15.844s. The two exact chunk boundaries recovered 301.366ms/group, while the
independent group wall recovered 267ms.

## Current B — render one optimized GSM8K DP16xTP4 full train

This command renders exactly one 64-chip, 200-update GSM8K JobSet. It never
launches Kubernetes. Run it only after the P74 commit is pushed, the remote SHA
is read back, and a clean checkout is at that exact 40-character SHA:

```bash
cd <physical-clean-worktree>
approved_sha="$(git rev-parse HEAD)"

bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_gsm8k_full_dp16tp4_p74.sh \
  "$approved_sha" \
  "/tmp/v1-gsm8k-p74-<fresh-wave-id>" \
  "<fresh-gsm8k-run-id>"
```

The script refuses a dirty tree, SHA/HEAD mismatch, reused output directory,
or an empty ID. It writes one manifest plus `manifest-index.json`, hashes both,
prints `V1_HP_GSM8K_P74_WAVE_READY ... launch=not-executed`, and prints one
unpiped `kubectl apply` command for later review. It does not execute that
command.

The rendered contract is DP16xTP4 strict Zero-TIM full training with resident
optimizer, checked VMA, fixed head, P63 overflow-safe clip, P74's flagless
device partition, the three receipt-lightening selectors, and
`CANON_P71_SCAN=fwd`. `CANON_DP_COLLECTIVE_REDUCE` is deliberately absent
because the DP16 FP64-oracle gate is not complete; `CANON_P71_SCAN=bwd` is
forbidden on TP4. The profile/`00_env.sh` chain maps
`CANON_P59_CHECKED_VMA=1` to the exact compatibility alias consumed by P74.

Before any separately approved launch, inspect the manifest and index, confirm
the source SHA, one JobSet, mesh `16,4`, `--max_steps=200`, all selectors above,
and absence of collective-reduce. At runtime, numerics and strict alignment
come before timing. Parse the real `[PERF] p32_vag_reverse` and official step
walls; do not mistake `grad_accumulate` for model backward. The one-host P74
census is shape/count-specific and must not be used to certify the target
XPlane. Target P74 performance stays unverified until a target-aware warm
reverse receipt is captured.

## Historical P60 procedure (do not use for current P74)

P60-2F remains a historical clean-SHA TARGET PASS for its old whole-update
contract, while its UI trace fails the newer navigation gate. The historical
Native/Zero pair remains `INCONCLUSIVE_INPUT_MISMATCH` for timing.

## Historical two-arm procedure (do not use for P60-2G)

This task is GSM8K, not DeepSWE. Do not run anything under
`tasks/p58-deepswe-native-zero-comparison/`. The two authoritative launchers
are in this task directory and must run sequentially on one direct-attached
four-chip v5p host.

## What the two arms mean

| Arm | Model/workload | Numerical and backward program | What its XProf must show |
|---|---|---|---|
| Native | Qwen3-1.7B GSM8K, DP4×TP1 | Stock inference, `CANON_GSM8K_VANILLA=1`, ordinary monolithic trainer backward; no canonical overlay, P32, P59, or G6 | Every TensorCore plane has exactly 16 `jit__train_step` modules and no decode module |
| Zero-HP | The same Qwen3-1.7B GSM8K shape, DP4×TP1 | Strict Zero-TIM V1 overlay plus P59 rank-parallel backward and fixed DP reduction | Every TensorCore plane has layer/head/norm/embed/adjoint backward families and no decode module; 51/51 alignment PASS |

Both arms execute three real optimizer updates with resident optimizer state.
Only update 1 is captured: XProf starts at its update entry and stops when the
step completes (`phase=update`, `start=1`, `stop=2`). Rollout is intentionally
outside the device capture, so the buffer can hold the complete backward.
Both runners set `CANON_P60_DETERMINISTIC_AB=1`, which pins seed 42, serial
rollout scheduling, concurrency one, and the 1024/256 padded work shape. It is
an input-control flag and does not turn on canonical kernels in Native. Equal
seed and shape do not guarantee equal sampled completions across numerically
different inference programs; the pair classifier checks the actual arrays.

## Preconditions

Use this exact worktree until the changes are committed and integrated:

```bash
cd /home/yuxuan/code_rl_repro/worktrees/p60_gsm8k_native_zero_xprof_0824
```

The runner itself fail-closes on all of these:

- hostname matches `V1_GSM8K_XPROF_EXPECT_HOSTNAME`;
- four direct TPU devices form the registered DP4×TP1 mesh `[0,2,1,3]`;
- pinned image `tunix_frozenlake_image:vllm-tpu0.25.0` exists;
- Qwen3-1.7B model snapshot and local GSM8K data exist;
- W&B credentials are present without printing them;
- no other P51/P59/GSM8K-XProf carrier owns the TPU lane;
- each artifact root is an absolute, fresh path.

For a clean committed-tree acceptance run, leave the dirty override unset:

```bash
unset V1_GSM8K_XPROF_ALLOW_DIRTY
```

Only while validating an uncommitted development tree may the operator use:

```bash
export V1_GSM8K_XPROF_ALLOW_DIRTY=1
```

Such a run is analysis-grade, not a signed clean-SHA release receipt.

## Step 1 — run the Native arm

Choose one timestamp and fresh labels. Do not launch Zero-HP concurrently:

```bash
p60_stamp="$(date -u +%Y%m%d_%H%M%S)"
p60_native_label="p60n_${p60_stamp}"
p60_zero_label="p60z_${p60_stamp}"
p60_evidence_root=/mnt/disks/tunix-data/gsm8k-onehost-xprof

export V1_GSM8K_XPROF_EXPECT_HOSTNAME="$(hostname)"
export V1_GSM8K_XPROF_EVIDENCE_ROOT="$p60_evidence_root"

bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_native.sh \
  "$p60_native_label"
```

The wrapper calls the shared runner with `arm=native`. It does not install any
canonical inference shim. It runs three stock GSM8K updates and captures the
second update's ordinary backward.

The final stdout lines must be:

```text
[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=<count> root=<absolute-root>
[V1.GSM8K.XPROF] GREEN arm=native backward_xprof=1 root=<absolute-root>
```

Require wrapper exit 0 and independent
`sha256sum -c "$p60_native_root/SHA256SUMS"` success after assigning the root
below. A GREEN marker without `SHA_LEDGER_PASS` is not acceptance evidence.

With the default evidence root, record and inspect:

```bash
p60_native_root="$p60_evidence_root/v1_native_${p60_native_label}"
cat "$p60_native_root/train/xprof_census.txt"
cat "$p60_native_root/train/semantic_census.txt"
python3 -m json.tool "$p60_native_root/train/classification.json"
```

Required Native markers:

```text
V1_GSM8K_XPROF_CENSUS_GREEN arm=native planes=8 backward=present decode=absent
V1_GSM8K_SEMANTIC_CENSUS_GREEN arm=native single_profiled_update=present
"verdict": "PASS"
```

Native normally has no `[CANON_ALIGN]` rows because it deliberately bypasses
the Zero-TIM observer. That absence is expected, not a missing-evidence bug.

## Step 2 — run the Zero-HP arm

Only after Native exits and its three required files above are readable:

```bash
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_zero_hp.sh \
  "$p60_zero_label"
```

This wrapper installs the Qwen3-1.7B TP1 canonical overlay, sources the strict
V1 profile, enables P59 rank-parallel backward, and retains fixed-order DP
reduction. It uses the same topology, prompt/response tensor shapes, seed,
number of trajectory groups, optimizer placement, and XProf window as Native.

The final stdout lines must be:

```text
[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=<count> root=<absolute-root>
[V1.GSM8K.XPROF] GREEN arm=zero-hp backward_xprof=1 root=<absolute-root>
```

Also require wrapper exit 0 and an independent
`sha256sum -c "$p60_zero_root/SHA256SUMS"` success. The driver must contain
exactly one terminal marker. A GREEN marker without `SHA_LEDGER_PASS` is an
evidence-packaging failure, not acceptance.

Inspect the Zero-HP result:

```bash
p60_zero_root="$p60_evidence_root/v1_zero-hp_${p60_zero_label}"
cat "$p60_zero_root/train/xprof_census.txt"
cat "$p60_zero_root/train/semantic_census.txt"
python3 -m json.tool "$p60_zero_root/train/classification.json"
```

Required Zero-HP markers:

```text
V1_GSM8K_XPROF_CENSUS_GREEN arm=zero-hp planes=8 backward=present decode=absent
V1_GSM8K_SEMANTIC_CENSUS_GREEN arm=zero-hp single_profiled_update=present
"verdict": "PASS"
```

The arm classifier additionally requires exactly 51 alignment PASS records,
zero alignment FAIL, and all five P59 backward families. The certified
development XPlane also contains the fixed reducer, replica comparison and
optimizer transaction; those are useful attribution evidence but are not
separate arm-verdict predicates in the current classifier.

## Step 3 — analyze and package the pair

Run the task's postflight; do not manually glob trace paths:

```bash
p60_pair_root="$p60_evidence_root/pair_${p60_stamp}"
set +e
bash canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair/scripts/analyze_gsm8k_xprof_pair.sh \
  "$p60_native_root" \
  "$p60_zero_root" \
  "$p60_pair_root"
p60_pair_rc=$?
set -e

case "$p60_pair_rc" in
  0) echo "matched profiled work: causal operation-attribution pair" ;;
  3) echo "both arm captures valid, but profiled training arrays differ" ;;
  *) echo "pair tooling/capture failure: rc=$p60_pair_rc"; exit "$p60_pair_rc" ;;
esac
```

The postflight resolves exactly one non-empty trace per arm, compares the
source/image/model/topology/capture/work hashes, runs the installed
`xprof-trace-analysis` summary, and writes:

```text
pair_classification.json
pair_classifier.txt
xprof_trace_summary.json
xprof_trace_summary.txt
SHA256SUMS
```

Exit 0 means both arms consumed identical profiled work. Exit 3 with
`INCONCLUSIVE_INPUT_MISMATCH` is not an arm failure: both backward captures
remain valid, but their durations cannot be treated as a causal performance
A/B. The 2026-08-24 development pair took this exit because prompt/source/
shape matched while completion/mask/advantage arrays differed. Do not rerun
the same seed-only pair expecting that to disappear; exact timing attribution
would require a future frozen-train-batch replay carrier.

## Which artifact proves what

1. `train/xprof_census.txt` plus the full `.xplane.pb` is the authoritative
   proof that all eight TensorCore planes contain complete backward work.
2. `train/semantic_census.txt` plus `train/perf/*.pb` proves the update window
   opened and closed around the intended transaction.
3. `train/xprof/**/*.trace.json.gz` is a convenient visualization/attribution
   view. Its bounded buffer may omit events; never use its raw module count as
   the completeness gate.
4. `classification.json` is the arm verdict. A zero shell exit without this
   PASS record is not acceptance evidence.
5. root `SHA256SUMS` is the immutable evidence ledger. It is written only
   after the terminal marker is frozen and must pass `sha256sum -c`; runner
   output must include `SHA_LEDGER_PASS`.

This distinction matters in the certified development run: Native's full
XPlane contains 16/16 `jit__train_step` modules on every plane, while its
compressed trace JSON exposes only 11 on the selected plane.

## What to return to the analyzing agent

For each arm, return the complete immutable run root, including:

1. `driver.log` and `train/raw.log`;
2. `train/classification.json`;
3. `train/xprof_census.txt` and `train/semantic_census.txt`;
4. complete `train/xprof/` and `train/perf/` trees;
5. root `SHA256SUMS`;
6. the complete pair output directory from Step 3.

Do not return only screenshots, selected log lines, or the trace JSON. Do not
delete RED or partial roots. If a wrapper does not print its GREEN marker,
return the whole root and let the classifier identify the first failed gate.

Known harmless tail noise: the pinned vLLM image can print a weakref cleanup
`AttributeError` after `TRAINING_DONE`. The classifier accepts the run only when
the actual process exit, three updates, capture markers, censuses, and artifacts
are all correct; never ignore an earlier traceback on the basis of this note.
