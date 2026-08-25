# Execution handoff: produce Native and Zero-HP backward XProf

> **Active follow-up (P60-2):** the original pair below is complete, but the
> Zero-HP view lacks a navigable host hierarchy. Do not rerun this two-arm
> recipe as a readability fix. A new executor must first follow
> [`HANDOFF_P60_2.md`](HANDOFF_P60_2.md). Its P60-2B local gates and the
> separately approved P60-2C Zero-HP development canary now pass. Do not rerun
> either arm without new approval. The historical pair remains
> `INCONCLUSIVE_INPUT_MISMATCH` for timing.

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
