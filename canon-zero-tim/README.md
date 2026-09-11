# canon-zero-tim

> **New here? Read `START_HERE.md`.** This file explains the mechanism; that one tells you
> what to do. For the blog experiment, start with
> [Reproduce the blog training curves](#reproduce-the-blog-training-curves).

Make the rollout engine, its re-scorer, and the training forward produce **bit-identical**
logprobs — then keep them that way while the model trains.

```
A = engine decode logprobs      the behaviour policy the tokens were actually sampled from
B = engine prefill re-score     the same engine, same tokens, scored in one pass
C = differentiable training forward
goal: A = B = C, bitwise, over the full distribution
```

---

## Reproduce the blog training curves

The current blog figure compares **three treatments on P45 FrozenLake**, with
two panels: sampler–trainer logprob difference and **training** solve rate.
M15 is a separate workload, not another curve in this figure. No training-code
or YAML edits are needed to select the existing treatments.

### Reference runs and recipe

The archives below contain `config.yaml`, `history.csv`, and raw console exports.
They are available at archive commit
`a7255cfc4b15a29ed9cabcd67a54cac416cd8b74`; this is the **archive revision**, not
the execution revision of all three runs.

| Blog label / archived run | Executed source prefix | Frozen policy-ratio denominator | Extra TIS weight | Tokens / evaluation |
|---|---|---|---|---|
| [Standard — jff877lt](../wandb_exports/jff877lt_canon-p57-fl-stan-r01-567c96d5/) | `567c96d5` | Trainer-old | None | Legacy / every 50 updates |
| [Token-level TIS — 8zjz4li7](../wandb_exports/8zjz4li7_canon-p57-fl-is-i45g-ccbcf572/) | `ccbcf572` | Trainer-old | Truncated trainer-old / rollout probability ratio, cap 2 | Legacy / every 50 updates |
| [Zero-TIM — tybj4xr0](../wandb_exports/tybj4xr0_canon-p57-fl-zero-r10a-06a0fdb9/) | `06a0fdb9` | Rollout-old | None | Exact TiTO, record-full / off |

**Use `standard`, not `native`, for the gray curve.** The historical
`native`/`mismatch` arm uses rollout-old as its denominator and is not plotted.
“No TIS” does not disable GSPO-token's policy-ratio clipping. Trainer-old means
trainer recomputation frozen before the update; rollout-old means the logprobs
recorded when sampling. See [the Standard handoff](tasks/p57-frozenlake-tim-causal-study/STANDARD64.md).

Shared recipe: Qwen3-8B, **64 v5p chips / DP8×TP8**, 32 prompts × 8 generations
= 256 trajectories per update, GSPO-token + RLOO, one optimizer iteration per
fresh batch, seed 42, learning rate 1e-6, clipping 0.003/0.005, temperature 0.7,
top-p 1, top-k 0. P45 uses grid sides 2–9, at most 5 turns, prompt limit 4096
and generation limit 2048. Each recipe specifies **300 updates**; the figure
uses only the first **200** training observations. Preserve the archived model,
tokenizer, dataset, optimizer and batch settings as well as these headline values.
The 32-chip / 128-trajectory option is a different experiment.

### Render the existing recipes

Run from the **repository root**, one directory above `canon-zero-tim/`.
First select a clean checkout of an approved, published **full source SHA**;
pin the registry image digest and model/checkpoint, tokenizer and dataset
identities using the linked handoffs. Supply your cluster/evidence access
outside Git. Do not hand-edit generated YAML or its autoscale/exclusive-topology
placement settings.

These commands render the current registered recipes, not three historical
source trees. Compare the rendered configuration with each archived config and
record any drift before describing a rerun as a historical reproduction.
Replace every placeholder; use fresh run IDs and fresh output directories
**outside the source worktree**. Rendering does not launch jobs.

```bash
SOURCE="<approved-published-40-character-sha>"
OUT="<absolute-fresh-output-parent-outside-worktree>"
P57="canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts"
V1="canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"

bash "$P57/render_three_arm_wave.sh" standard "$SOURCE" "$OUT/standard" \
  "<fresh-standard-p45-id>" "<unused-standard-m15-id>" "<fresh-standard-campaign>"
bash "$P57/render_three_arm_wave.sh" is "$SOURCE" "$OUT/is" \
  "<fresh-is-p45-id>" "<unused-is-m15-id>" "<fresh-is-campaign>"
bash "$V1/prepare_p67_frozenlake_two_full_wave.sh" \
  "$SOURCE" "$OUT/zero" "<fresh-zero-campaign>" \
  "<fresh-zero-p45-id>" "<unused-zero-m15-id>" \
  --token-continuity both-exact --token-continuity-debug-mode record-full \
  --train-geometry dp8-tp8-b256
```

The wrappers reject dirty or mismatched source and reused output directories.
Require `P57_THREE_ARM_WAVE_PASS` for Standard/IS and
`V1_P67_FROZENLAKE_WAVE_READY` with `token_continuity=both-exact` and
`token_continuity_debug=record-full` for Zero; retain the generated manifests
and SHA receipts. These are preparation receipts, not target certification.

Both wrappers also generate M15 manifests: **do not apply the whole output
directory**. The three P45 manifests for this figure are:

```text
<OUT>/standard/p45/jobset-p57-frozenlake-standard-300.yaml
<OUT>/is/p45/jobset-p57-frozenlake-is-300.yaml
<OUT>/zero/frozenlake-p45/jobset-p57-frozenlake-zero-300.yaml
```

Before a separately approved launch, start one persistent worker-log collector
per JobSet (16 workers per run), following
[RUNBOOK: External worker-log collection](tasks/p57-frozenlake-tim-causal-study/RUNBOOK.md#external-worker-log-collection).
Follow the [TiTO handoff](tasks/multiturn-tito-cross-workload/HANDOFF.md) for
record-full witness, trajectory/sidecar and evidence-upload checks. Launch only
the intended P45 files, without a trailing shell pipeline. Three simultaneous
runs need 192 chips; they may instead run sequentially. Preserve failed runs.

### Check the result, not just the launch

Keep source/image/config identities, raw worker logs, runtime admission and
optimizer receipts, and the W&B history/config export for each run. Standard
must report `old_logps=trainer tis_weights=absent`; IS must retain the registered
token-level correction; Zero must report exact TiTO and its selected numerical
profile. Check finite gradient/update diagnostics independently of forward
alignment. A complete 300-update run and a 200-observation plot are distinct checks.

To compare with the figure, select W&B `_step=0..199` from training rows, reject
missing or duplicate steps, and display them as steps 1–200. Plot
`sampler_trainer/train/logp_diff_mean` directly (mean **absolute** difference on
action tokens), and `rewards/train/solve_ratio` as raw observations with a trailing
10-observation average, using the available prefix at the start. Do not substitute
evaluation reward or fill missing values with zero. The archived final-ten means
are **62.7% / 66.9% / 88.2%** for Standard / TIS / Zero, not promised rerun targets.
The blog's figure builder is maintained separately; it is not bundled by this README.

Evidence boundary: all 200 exported Zero mean-difference observations are zero,
but sampled raw receipts use warning-only admission and do not cover every step
continuously. This is measured telemetry, **not a signed strict full-run certificate**.
Strict Zero-TIM requires both decode A–independent prefill B and B–trainer C to
have zero differing bytes on the sampled action mask; missing receipts are not
zeros. The historical arms differ in TiTO, evaluation, execution revision and
backward implementation, so this is a comparison of complete configurations,
not an isolated causal ablation. Equal seeds do not guarantee identical sampled
trajectories, gradient bits, or training curves.

---

## Why bitwise, and not "close enough"

Two amplifiers make any nonzero seed useless to shrink.

**Depth.** A ULP-scale difference entering the layer stack is amplified fast and saturates
within 3–4 layers: measured 1 layer `3.4e-07` → 2 layers `1.5e-03` → end-to-end logprob
difference `0.015–0.024`. Shrinking the seed from `1e-2` to `1e-8` lands in the same place.
Only exactly zero is different in kind.

**Time.** A trajectory's log-probability is a sum, so per-token differences accumulate over
its length, and Gibbs' inequality fixes the sign — self-scored tokens drift systematically
high, they do not cancel. Long-horizon agentic RL feels this first.

## Why it happens

Not because anyone's operator is wrong. Measured with both engines' real modules in one
process, fed identical inputs: RMSNorm `0`, matmul `0`, RoPE `0`, attention kernel `0` —
and the end-to-end difference was still there.

The cause is **program identity**. An XLA executable's numerics are fixed by more than the
operators:

```
executable = operators + SHAPE + JIT BOUNDARY + REDUCTION ORDER
```

Change the token bucket, wrap a call in a nested `jit`, or ask for a gradient, and you get a
different program — and bf16 addition is not associative, so a different program is a
different number. The package pins every one of those degrees of freedom.

A useful corollary: two *different implementations* of the same maths differ by ~1% in bf16
no matter how carefully written (measured: swapping only the attention implementation moved
logprobs by `0.0115`, the same order as the whole A/B/C gap). You cannot align two
implementations below that floor. You can only run **the same code**. That is why `C` calls
the engine's own attention kernel rather than a training-side equivalent.

---

## What is changed

Nine ordered patches against six engine files. Every switch defaults off; with
the environment unset the patched files retain the stock execution branches.

| # | Patch | Root cause it addresses | Switch |
|---|---|---|---|
| 01 | `attention-interface` | RPA block sizes are chosen by a shape- *and* concurrency-dependent lookup, so decode and prefill silently accumulate at different granularity; and the kernel is not differentiable | `CANON_RPA_D/P/M`, `CANON_RPA_VJP2` |
| 02 | `embed` | The vocab-sharded embedding gather is the last 4-way reduction in the forward, and 4-way reductions lower differently in a forward-plus-backward program | `CANON_FIXED_AR_EMBED` |
| 03 | `linear` | The bf16 all-reduce closing a contract-sharded projection sums in **ring-position order**, so the same token at a different padded row gets different bits | `CANON_FIXED_AR` |
| 04 | `qwen3` | **none — diagnostic instrumentation only** (bisection cut points, depth reduction) | `CANON_CUT`, `P16_NUM_LAYERS` |
| 05 | `qwen2` | **none — diagnostic instrumentation only** (tail cut, optimization barriers) | `CANON_TAIL`, `CANON_BARRIER_ALL` |
| 06 | `tpu-runner` | Decode and prefill land in different token buckets; the vocab reduction splits by `M` **and** by nested-jit caller; prompt scoring does not inherit decode's processed-logprob semantics | `MIN_TOKEN_BUCKET`, `CANON_LOGPROB_M`, `CANON_PROMPT_PROCESSED_LOGPROBS` |
| 07 | `tpu-runner` | **none — P35 diagnostic metadata only** | `CANON_P35_METADATA_DIR` |
| 08 | `attention-interface` | **none by default — P38 target-only combined cache-write/all-cache-read causal arm** | `CANON_KV_UNIFIED` |
| 09 | `tpu-runner` | **none — bounded capture around the real donated-cache `continue_decode` program** | `CANON_P38_SERVING_CAPTURE_DIR` and bounds |

Patches 04 and 05 carry no fix. They are here because the shim chain bottoms out in those two
files, so removing their instrumentation would break byte-identity with the sources that carry
the signed evidence. The diagnostic *switches* are excluded from the canonical profile
(`cluster/profiles/`); the diagnostic *code* is not separable from the chain.

Plus a **shim chain** (`src/engine_shims/`) carrying the promoted Pallas canonical ops and the
differentiable attention wrapper, and one XLA flag,
`--xla_allow_excess_precision=false`, which removes a layer of forward-vs-forward+backward
lowering difference.

### The forward/backward split

The engine's attention kernel is fast, bit-exact, and **not differentiable** — JAX has no
transpose rule for it. Training needs a gradient, so the package supplies one:

```
forward   = the real kernel, verbatim          <- must be bitwise; this is what A, B and C compare
backward  = the Jacobian of a line-by-line
            pure-JAX transcription             <- must be correct; SGD does not care about 1 ULP
```

They meet only at the inputs: `custom_vjp` saves the arguments, and the transcription's own
output value is discarded. The transcription is faithful to `1.49e-08` in fp32 (a from-scratch
reimplementation is `7.06e-04`), and an fp64 oracle shows the chunked-with-cache and
full-prefill forms are the same mathematical function — so its VJP is the kernel's VJP, not a
surrogate.

---

## Running the tests

### T0 — pure CPU, seconds, no TPU

```bash
tests/t0_cpu/run.sh
tests/t0_cpu/negative_control.sh     # proves the gate rejects bad runs
```

Proves the differentiable contract is mathematically sound: value residual exactly `0`,
gradient agreement `~5e-16`, finite-difference cross-check `~1.1e-08`. Does **not** touch the
kernel — that is T1's job.

### T1 — needs ≥2 TPU chips, no model, no image build

```bash
tests/t1_tpu/run.sh                       # host: re-enters the pinned image
CANON_IN_CONTAINER=1 tests/t1_tpu/run.sh  # already inside a TPU container
```

Topology diagnostics, the Mosaic compatibility P1a gate, the production-operator P1b hard gate,
plus the historical minimal reproducers. **Run this first on any new cluster.** Generic P1 scans
TP widths `2,4,8` and detects platform drift; only P1b
decides whether the installed P22.XK Qwen operator chain is admitted. See
`CLUSTER_ADMISSION.md`.

### T2 / T3 — need the pinned image, a checkpoint, and a 4-chip host

Not runnable from this package alone. `recipes/` records the exact commands, the expected
output lines, and the artifact SHA-256s so a run elsewhere can be checked against the signed
ones.

---

## Installing the chain

```bash
./install.sh /somewhere --from-image tunix_frozenlake_image:vllm-tpu0.25.0 --model qwen1p7b
./install.sh /somewhere --from-path  /path/to/site-packages/tpu_inference --model qwen1p7b
```

`--from-path` needs no docker, no image on disk, and no network — that is the mode a pod uses.
Both verify every produced file against `MANIFEST.sha256` and **fail** on a mismatch, because
the chain is loaded by path: a stale member does not raise, it silently reverts you to stock.

For Kubernetes see `cluster/README.md`.

---

## The one thing to remember

> A run with the switches on and no `[PATHTRACE]` lines did not do what you think it did.

The chain is resolved by module name and by absolute sibling path. Neither raises when a
member is missing — the engine falls back to its stock module, every switch still reads "on",
and the run goes green having computed nothing canonical. The `[PATHTRACE]` tally is the only
evidence that the intervention executed. The exit code is not evidence.

---

## Layout

```
START_HERE.md           zero-context entry point: status, what your machine can run
RUNBOOK.md              step-by-step procedures for each runnable task
README.md               this file -- what the mechanism is and why
ANCHORS.md              the two versions everything is pinned to, and how to reach them
EVIDENCE.md             claim -> artifact -> SHA-256, including where the chain is weak
KNOWN_FOOTGUNS.md       traps that produce green runs
CLUSTER_ADMISSION.md    what to measure before trusting a new topology
install.sh              assemble the chain (two source modes)
MANIFEST.sha256         expected SHA of every installed file
STOCK_MANIFEST.sha256   expected SHA of the six upstream files the patches are cut against
patches/tpu_inference/  nine ordered diffs across six pinned engine files
src/engine_shims/       the shim chain + promoted Pallas ops + model-specific modules
tests/t0_cpu/           CPU gates
tests/t1_tpu/           topology admission probes
tests/t2_dp/            DP reduction/update admission + negative control
cluster/                Pathways/GKE entry point, steps, profiles, manifest
recipes/                T2/T3 reproduction recipes and expected outputs
docs/                   the phase-by-phase record of how this package was built
```
