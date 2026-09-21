# Sequence packing on Qwen3.5-35B-A3B: validation, performance, and test inventory

**Author:** Anisha Mazumder
**Last Modified:** 2026-09-21

Packing is confirmed active and correct on two completed GRPO runs:
`maz-q35-10` (100 steps) and `maz-q35-11` (20 steps), both 2026-09-17.
A matched unpacked control, `maz-q35-13` (20 steps, 2026-09-18), supplies a
prompt-paired measurement of its effect on generation quality.

This document answers five questions:

1. Is sequence packing actually on in these runs, and how would you check that yourself?
2. What did it cost or save?
3. Does it change what the model generates?
4. Do the two assemblers produce the same loss and the same gradient when handed
   the same trajectories?
5. Which tests establish that the packed arithmetic equals the unpacked arithmetic?

Everything below is measured, not estimated. Every number was re-derived from
Cloud Logging while writing this document; the log links reproduce the raw
records.

---

## 1. Configuration under test

Both runs used the same configuration, in
[docker/maz-q35/submit.sh](docker/maz-q35/submit.sh).

| Setting | Value | submit.sh line |
| --- | --- | --- |
| Model | `Qwen/Qwen3.5-35B-A3B` | [59-60](docker/maz-q35/submit.sh#L59-L60) |
| Image | `gcr.io/cloud-tpu-multipod-dev/mazumdera-runner@sha256:428ef303…` (tag `q35-0916-v3`) | [48](docker/maz-q35/submit.sh#L48) |
| Trainer | `tpuv5p:2x2x2`, FSDP 8, TP 1, expert 1 | [66-69](docker/maz-q35/submit.sh#L66-L69) |
| Rollout | 1 × `tpuv5p:2x2x1`, DP 2, TP 2, `SAMPLER=vllm` | [108-128](docker/maz-q35/submit.sh#L108-L128) |
| Prompts × generations | 16 × 16 = 256 rollouts per step | [149-150](docker/maz-q35/submit.sh#L149-L150) |
| `MINI_BATCH_SIZE` | 256 — one optimizer step per rollout batch | [151](docker/maz-q35/submit.sh#L151) |
| `TRAIN_MICRO_BATCH_SIZE` | 8 — one packed row per FSDP shard | [152](docker/maz-q35/submit.sh#L152) |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | 512 / 1024 (unpacked row = 1536) | [147-148](docker/maz-q35/submit.sh#L147-L148) |
| **`MAX_SEQ_TOKEN_PER_TPU`** | **4096** — the packing budget; empty in the `maz-q35-13` control | [157](docker/maz-q35/submit.sh#L157) |
| `BETA` (KL coefficient) | 0 (launcher default, not overridden) | [k8s_launcher.sh:64](tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh#L64) |
| `loss_agg_mode` | `sequence-mean-token-mean` (GRPOConfig default) | [algorithm_config.py:165](tunix/rl/algorithm_config.py#L165) |
| Checkpointing | disabled (`CHECKPOINT_SAVE_INTERVAL_STEPS=0`) | [174](docker/maz-q35/submit.sh#L174) |
| Priority | `medium` | [37](docker/maz-q35/submit.sh#L37) |
| `DEBUG` | 1 | [208](docker/maz-q35/submit.sh#L208) |

Both runs set `TRAJECTORY_LOG_DIR` to a GCS prefix
([236](docker/maz-q35/submit.sh#L236)), asking for every rollout's response,
gold answer and reward to be written to CSV. Only `maz-q35-11` produced one:
run 10's logger thread raised an assertion on every flush and wrote nothing.
The two runs differ in exactly two ways — step count, and the placeholder object
that makes that assertion hold. See §7 for the assertion and §8 for the
workaround.

Cluster: `bodaborg-v5p-nap`, `europe-west4`, namespace `trellis`, project
**`cloud-tpu-shared-capacity`**. Note that the project holding the image and the
output bucket (`cloud-tpu-multipod-dev`) is *not* the project holding the logs;
querying the wrong one returns nothing.

---

## 2. Is packing on? Four independent checks

Each check is stronger than the last. The first establishes that the code path
was selected; the last establishes that the trainer consumed packed rows.

### 2.1 The assembler was selected

`tunix/experimental/orchestrator/batch_assembly.py` picks
`SequencePackedBatchAssembler` when `max_seq_token_per_tpu` is set, and
`PaddedBatchAssembler` otherwise
([:1039](tunix/experimental/orchestrator/batch_assembly.py#L1039)). The chosen
branch logs itself at
[:1018](tunix/experimental/orchestrator/batch_assembly.py#L1018):

```
2026-09-17 00:14:20,099 - [Orchestrator] Using SequencePackedBatchAssembler with max_seq_token_per_tpu: 4096, max_segments_per_packed_row: None, pack_size: 8   # maz-q35-10
2026-09-17 20:03:10,838 - [Orchestrator] Using SequencePackedBatchAssembler with max_seq_token_per_tpu: 4096, max_segments_per_packed_row: None, pack_size: 8   # maz-q35-11
```

[Log link — assembler selection, both runs](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D~%22%5Emaz-q35-1%5B01%5D-orch%24%22%0AtextPayload%3A%22Using%20SequencePackedBatchAssembler%22;timeRange=2026-09-17T00%3A00%3A00.000Z%2F2026-09-17T21%3A00%3A00.000Z?project=cloud-tpu-shared-capacity)

`pack_size: 8` is derived, not configured. With `trainer_fsdp` set, it is
`trainer_fsdp * trainer_dp` = 8 × 1
([:994-1005](tunix/experimental/orchestrator/batch_assembly.py#L994-L1005)), so
each microbatch is 8 packed rows of up to 4096 tokens — one row per FSDP shard.
Seeing `pack_size: 8` rather than `pack_size: 1` is itself confirmation that the
trainer mesh was read correctly.

If this line is absent from your run, packing is off. Grep for it first.

### 2.2 Rows really contained multiple sequences

`rl_program.py` logs each assembled microbatch at
[:862](tunix/experimental/orchestrator/rl_program.py#L862):

```
Packed %d trajectories into microbatch: %s
```

Counting these across both runs:

| | maz-q35-10 | maz-q35-11 |
| --- | --- | --- |
| Microbatch records | 516 | 101 |
| Trajectories packed | 25,600 | 5,120 |
| = steps × 256 | 100.00 | 20.00 |
| Trajectories per microbatch (min / median / max / mean) | 2 / 51 / 89 / 49.6 | 3 / 52 / 81 / 50.7 |
| **Segments per row** (÷ pack_size 8) — median / mean / max | 6.38 / 6.20 / 11.12 | 6.50 / 6.34 / 10.12 |

Log links:
[packing density, run 10](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D%22maz-q35-10-orch%22%0AtextPayload%3A%22trajectories%20into%20microbatch%22;timeRange=2026-09-17T00%3A00%3A00.000Z%2F2026-09-17T03%3A30%3A00.000Z?project=cloud-tpu-shared-capacity)
·
[packing density, run 11](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D%22maz-q35-11-orch%22%0AtextPayload%3A%22trajectories%20into%20microbatch%22;timeRange=2026-09-17T19%3A50%3A00.000Z%2F2026-09-17T21%3A00%3A00.000Z?project=cloud-tpu-shared-capacity)

A median of roughly 6.4 sequences per 4096-token row is the expected density:
GSM8K completions are short relative to the 1024-token response budget, so the
FFD bin packer fits six or seven of them where the padded assembler would have
used one row each.

### 2.3 Nothing was lost or duplicated in packing

The trajectory totals above are exact, not approximate:

- run 10: 25,600 trajectories = **exactly** 100 steps × 256
- run 11: 5,120 trajectories = **exactly** 20 steps × 256

Every rollout appears in exactly one packed row. A packer that dropped or
double-counted sequences would change these totals; they are exact.

Work per step:

| | maz-q35-10 | maz-q35-11 |
| --- | --- | --- |
| Microbatches per step, packed | 5.16 | 5.05 |
| Microbatches per step, unpacked (256 ÷ `TRAIN_MICRO_BATCH_SIZE` 8) | 32 | 32 |
| **Forward/backward passes saved** | **6.20×** | **6.34×** |
| Padded token area, packed (rows × 4096) | 16,908,288 | 3,309,568 |
| Padded token area, unpacked (rows × 1536) | 39,321,600 | 7,864,320 |
| **Padding eliminated** | **2.33×** | **2.38×** |

The two ratios differ because they measure different quantities. 6.2× is the
reduction in kernel invocations; 2.33× is the reduction in tokens processed by
those kernels. The second bounds the achievable speedup; the first reduces the
fixed per-microbatch overhead.

### 2.4 The trainer was given the larger budget, not just the orchestrator

`--max_seq_token_per_tpu` is passed to both processes by `k8s_launcher.sh` — to
the orchestrator at
[:242](tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh#L242) and to
the trainer node at
[:335](tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh#L335). On the
trainer side it widens the model's sequence dimension in
[`tunix/utils/maxtext_utils.py:133-134`](tunix/utils/maxtext_utils.py#L133-L134):

```python
max_target_length = max(max_prompt_length + max_response_length, max_seq_token_per_tpu)
# = max(512 + 1024, 4096) = 4096
```

Had only the orchestrator received the flag, the trainer would have been
compiled for 1536-token rows and rejected the 4096-token packed rows. Both runs
completing is therefore evidence that both processes received the same value.

---

## 3. Performance

Step times, all 100 and all 20 steps respectively, read from the orchestrator's
`Train step N … step_time: X.XXs` lines.

| | maz-q35-10 (100 steps) | maz-q35-11 (20 steps) |
| --- | --- | --- |
| Step 0 (includes compilation) | 542.95 s | 539.89 s |
| Steps ≥ 1: median | **97.12 s** | **96.30 s** |
| Steps ≥ 1: mean | 97.74 s | 97.95 s |
| Steps ≥ 1: min / max | 87.37 / 119.42 s | 93.13 / 123.07 s |
| Steps ≥ 1: stdev | 4.48 s | 6.11 s |
| Wall clock | 00:08:35 → 03:07:23 Z | 19:57:45 → 20:46:07 Z |

Log links:
[step times, run 10](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D%22maz-q35-10-orch%22%0AtextPayload%3A%22Train%20step%22;timeRange=2026-09-17T00%3A00%3A00.000Z%2F2026-09-17T03%3A30%3A00.000Z?project=cloud-tpu-shared-capacity)
·
[step times, run 11](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D%22maz-q35-11-orch%22%0AtextPayload%3A%22Train%20step%22;timeRange=2026-09-17T19%3A50%3A00.000Z%2F2026-09-17T21%3A00%3A00.000Z?project=cloud-tpu-shared-capacity)
·
[all orchestrator output, run 11](https://console.cloud.google.com/logs/query;query=resource.type%3D%22k8s_container%22%0Alabels.%22k8s-pod%2Fjobset_sigs_k8s_io%2Fjobset-name%22%3D%22maz-q35-11-orch%22;timeRange=2026-09-17T19%3A50%3A00.000Z%2F2026-09-17T21%3A00%3A00.000Z?project=cloud-tpu-shared-capacity)

**No recompilation after step 1.** This was the specific risk to watch: a packed
batch whose segment count changes forces a retrace, because `num_segments` is a
static field of the batch (see
`PackedBatchStaticsTest::test_num_segments_is_not_a_traced_leaf`). Over 99 warm
steps in run 10 the standard deviation is 4.48 s, 4.6% of the median, and the
maximum of 119.42 s occurs at step 2 rather than at an arbitrary later step.
There is no step at which the time steps up and stays up, which is what a
recompilation would look like. Run 11's 123.07 s at step 1 is the first
trajectory-CSV flush, not a retrace.

**Trajectory logging costs nothing measurable at this scale.** Run 11 wrote every
rollout to GCS and had a lower median step time than run 10 (96.30 s vs
97.12 s) — the difference is within run-to-run noise, so the logging cost is
below the measurement floor. This does not extend beyond 20 steps: `maz-q35-12`
hung at step 22 inside the logger's GCS read and never recovered. See §7.

Weight sync, run 10: 86.90 s for the initial transfer, then 52–53 s per step.
That is over half the 97 s step time, and is the largest single target for
further optimisation — it is unaffected by packing.

### Learning signal was intact

From the run 11 trajectory CSV (5,120 rows, 132 MB), analysed with
[`tunix/experimental/examples/common/analyze_trajectories.py`](tunix/experimental/examples/common/analyze_trajectories.py):

- Overall mean reward **0.9222**
- Reward histogram: 1.0 → 4,701 (91.8%), 0.0 → 272 (5.3%), 0.1 → 132 (2.6%), 0.5 → 15 (0.3%)
- Active gradient groups (non-zero reward variance): 61 / 320 (19.1%)
- Generation status: 4,851 SUCCEEDED, 269 MAX_CONTEXT_LIMIT_REACHED

Cross-validation: the CSV's own per-step mean reward matches the orchestrator's
independently computed metric exactly — step 19, CSV 0.8629 vs log
`reward_mean: 0.8629`; step 17, CSV 0.9352 vs log `reward_mean: 0.9352`. The
rewards attached to packed segments are the rewards the trainer used.

Two findings worth acting on, both unrelated to packing:

- **269 of the 272 zero rewards are truncations**, not wrong answers; only 3
  completed generations scored zero. Of the 61 active groups, 44 contain a
  truncation and **31 are active only because of one**. Raising
  `MAX_RESPONSE_LENGTH` above 1024 would therefore roughly halve the active-group
  rate, from 19.1% to about 9%. Most of the current gradient signal comes from
  truncation, not from incorrect answers.
- At 91.8% reward 1.0, GSM8K is close to saturated for this model. A harder
  dataset would give more signal per step.

### The unpacked control: maz-q35-13

Runs 10 and 11 establish that packing is active and that nothing downstream broke.
Neither run has an unpacked counterpart, so neither measures how much packing
changes the reward or the completion lengths. `maz-q35-13` supplies that
counterpart: 20 steps, identical to `maz-q35-12` down to the image digest, with
`MAX_SEQ_TOKEN_PER_TPU` empty. That drops
`--max_seq_token_per_tpu` from both the orchestrator and the trainer
(`k8s_launcher.sh:242,335`, both guarded with `:+`), selecting
`PaddedBatchAssembler` and leaving `max_target_length` at 1536 — confirmed in the
trainer log.

The comparison is legitimate because both arms see the same data. `--seed`
defaults to 42 and `--shuffle` to True (`run_gsm8k_dist_grpo.py:182,184`) and
neither `submit.sh` nor `k8s_launcher.sh` overrides them. This is verified rather
than assumed: on all 20 steps the two runs' prompt-id sets and question sets are
identical. Generation is still stochastic (`--temperature` defaults to 1.0) and
unseeded, so the comparison is statistical, not exact-match.

| | packed (`maz-q35-12`) | unpacked (`maz-q35-13`) |
| --- | --- | --- |
| Microbatches per step | 5.16 | **32** |
| Trajectories per microbatch | ~51 | **8** |
| `max_target_length` | 4096 | 1536 |
| Step 0 | 442.98 s | 512.88 s |
| Steps ≥ 1: mean | **98.48 s** | **171.74 s** |
| Steps ≥ 1: min / max | 89.21 / 112.60 s | 162.21 / 179.70 s |
| Steps ≥ 1: stdev | 5.65 s | 3.89 s |
| Mean reward, steps 0–19 | 0.9208 | 0.9232 |
| Fraction at reward 1.0 | 0.9172 | 0.9193 |
| Generated chars, p50 / p90 / p99 | 1156 / 2017 / 3757 | 1149 / 2028 / 3775 |
| Whitespace words, mean / median | 228.2 / 196 | 228.4 / 195 |

**Packing reduces wall-clock step time by 1.74×.** That is well below the 6.2×
reduction in forward/backward passes, because generation runs in vLLM and is
unaffected by packing; only the trainer's portion of the step is reduced.

### Does packed training produce a different policy?

Reward is a weak instrument for this question. With 91.8% of rollouts at reward
1.0 and 5.4% truncated, both channels sit near their ceiling, so a defect in
packed training could leave the mean reward unmoved. Completion length is not
saturated and is the channel a corrupted gradient would move, so the tests below
lead with it.

All 320 prompts — 20 steps × 16 prompts — match across the two arms, so every
comparison below is paired on the same question at the same step:

| paired per-prompt, control − packed, n = 320 | difference | 95% CI | p |
| --- | --- | --- | --- |
| Generated length, chars | −1.22 | [−12.1, +9.7] | 0.825 |
| Reward | +0.0024 | [−0.0038, +0.0086] | 0.444 |
| Truncation rate | −0.0010 | [−0.0063, +0.0043] | 0.717 |

**The test resolves an effect roughly twelve times smaller than the one training
itself produces.** Over the same 20 steps mean generated length falls by 132 chars
in the packed arm and 127 in the control (KS D = 0.110 and 0.119, p = 3.5e-7 and
2.8e-8), against a paired CI half-width of 10.9 chars.

A defect in packed training has to appear as divergence that grows with step
index, because generation precedes the first gradient update: at step 0 both arms
hold the identical base policy, so step 0 measures sampling noise alone.

```
step 0, identical policy
  reward difference   +0.0059
  generated length    KS D = 0.0625, p = 0.700

slope of the paired gap against step index
  reward          +0.00008/step   p = 0.870
  truncation      +0.00054/step   p = 0.124
  length            +0.70/step    p = 0.384

per-step KS on generated length, steps 1-19
  D mean 0.0518, min 0.0312, max 0.0781   (step 0 floor: D = 0.0625)
  steps with p < 0.05: 0 of 20            (about 1 expected by chance)
```

Divergence after training is at or below the step-0 noise floor, and no metric
trends with step index. The truncation channel agrees independently: non-truncated
accuracy is 0.9698 packed against 0.9711 control, and packed truncation falls
slightly faster over the run (difference of differences −0.0078, 0.62σ).

### Comparing the generated text directly

The tests above compare summary statistics. The completions can also be compared
as text, against the baseline of what "no difference" looks like: two draws from
the *same* arm for the same prompt. Each prompt has 16 generations per arm, giving
a cross-arm class (packed against unpacked) and two within-arm classes, all on the
same prompt at the same step.

| same prompt and step | exact match | shared prefix | 5-gram Jaccard |
| --- | --- | --- | --- |
| cross-arm | 0.0009 | 52.7 chars | 0.1291 |
| within-packed | 0.0004 | 49.5 chars | 0.1242 |
| within-unpacked | 0.0002 | 49.8 chars | 0.1264 |

**A packed completion resembles an unpacked completion slightly more than it
resembles another packed completion.** Paired across the 320 groups, cross-arm
minus within-arm is +0.0038 in Jaccard (95% CI [+0.0016, +0.0060]) and +3.1 chars
of shared prefix (95% CI [+1.5, +4.7]). The gap does not trend with step index
(slope −0.0001, p = 0.777). Whatever separates the two arms is smaller than what
separates two samples drawn from one arm.

The absolute numbers also show why exact matching was never available: within a
single arm, two generations for the same prompt match exactly about once in 2,500
and share only 12% of their 5-grams and 50 characters of prefix.

Generation is unseeded for two independent reasons. `--seed 42` reaches only the
dataset shuffle (`run_gsm8k_dist_grpo.py:296` passes it to
`load_gsm8k_dataset`); nothing routes it to sampling. And under `SAMPLER=vllm`,
`vllm_sampler_v2._build_vllm_params:162` constructs `VllmSamplingParams` from
temperature, top_p, top_k, max_tokens, stop and logprobs, omitting `seed`, so the
`seed` field that `rollout_worker.py:256` populates is dropped before it reaches
vLLM. Pairing on `prompt_id`, with step 0 as the calibration, is the substitute.

Two limits. This compares policies after 19 updates, so a defect that accumulates
slowly needs a longer run. And it detects only a defect whose effect depends on
packing, not one identical in both arms. The same-trajectory result comes from §4.

TODO(tunix): the `seed` field on `rollout/sampler.py:52` is documented as "Random
seed for reproducible generation" and is populated by `rollout_worker.py:256`, but
`vllm_sampler_v2._build_vllm_params:162` never forwards it. `vllm_sampler.py:521`
separately notes that the vLLM Jax backend does not support a per-request seed, so
forwarding it may not be sufficient on its own.

TODO(tunix): cross-arm similarity exceeding within-arm similarity by 3% relative
(p = 0.001) is unexplained. The direction is away from divergence, so it does not
affect the conclusion here.

All lengths above are of the assistant turn alone. The CSV's `completion` column
holds the entire conversation — a Python repr of `[{'role': 'system', ...},
{'role': 'user', ...}, {'role': 'assistant', ...}]` — in all 5,120 rows of both
runs, so measuring that column directly adds roughly 650 characters of prompt to
every row. `qwen35_paired_ab.py` parses out the assistant turn. Paired differences
are unaffected either way, since the prompt is identical within a group and
cancels, but absolute lengths and the KS distributions are not.

TODO(tunix): length is measured in characters because `completion_tokens` is null
in all 5,120 rows of both runs. `rl_program.py:414` reads it with
`getattr(src_item, "completion_tokens", None)` and the attribute is absent on the
sampler's item, so the column is written empty.

---

## 4. Direct measurement: one trajectory set through both assemblers

§3 compares two runs that never see the same data — generation is unseeded, so
even a defect-free packer produces two different policies there. That bounds a
packing defect but cannot exclude one.
[`qwen35_packing_equivalence.py`](qwen35_packing_equivalence.py) removes the
limit: one identical 64-trajectory set through `SequencePackedBatchAssembler` and
`PaddedBatchAssembler` against identical weights, with
`MaxTextTrainingEngine.fwd_bwd` called on every microbatch of each arm. The arms
differ only in layout — packed is 3 microbatches of `(4, 4096)` rows carrying
`segment_ids`, 5.3 sequences per row; unpacked is 16 microbatches of `(4, 512)`
prompt plus `(4, 1024)` completion, one sequence per row. Both reach a
denominator of 64, and both are checked to consume the same trajectory **ids**,
not merely the same count.

### 4.1 The null control

Neither arm reproduces the other bitwise, and not because of packing: packed
accumulates the 64 trajectories over 3 microbatches (19, 26 and 19 segments),
unpacked over 16 of 4, and float32 addition is not associative. A verdict
therefore needs a tolerance, and one picked by eye establishes nothing.

`--null_control N` measures it. It replaces the packed arm with a second
**unpacked** arm at microbatch size `N`, so both arms see byte-identical rows, no
packing is involved anywhere, and every difference it reports is reassociation
alone. Read **observed ÷ null**: a ratio at or below 1 means packing disagrees no
more than repartitioning the same unpacked rows does. This self-calibrates across
dtype and checkpoint, where a fixed `--rtol` does not. The runs below use
`--null_control 8` against an unpacked microbatch size of 4.

### 4.2 Result

Qwen3-0.6B on a 4-chip v5p, `trainer_fsdp` 4, `MAX_SEQ_TOKEN_PER_TPU` 4096,
`epsilon` 0.2, `beta` 0, `loss_agg_mode` `sequence-mean-token-mean`. The 64
completion lengths are sampled from the `maz-q35-12` trajectory CSV (median 312
tokens, max 1024), converted from characters at 3.6 chars per token because
`completion_tokens` is empty there for the reason in §3. Three regimes — random
init, the trained checkpoint
`gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items`, and that
checkpoint in bfloat16 — each against its own null. Six runs.

| | random init, fp32 | checkpoint, fp32 | checkpoint, bf16 |
| --- | --- | --- | --- |
| Logp scale, mean | -247.82 | -13.57 | -13.57 |
| Per-token logp, mean abs diff | 4.986e-04 | 1.878e-05 | 3.120e-02 |
| &nbsp;&nbsp;*same, null mb8* | **0.0 exactly** | **0.0 exactly** | **0.0 exactly** |
| Pooled loss, rel diff | 6.599e-04 | 8.939e-06 | 1.164e-02 |
| &nbsp;&nbsp;*same, null mb8* | 2.027e-04 | 1.068e-05 | 3.572e-02 |
| Gradient, whole-tree rel L2 | 1.179e-03 | 3.118e-04 | 1.287e-01 |
| &nbsp;&nbsp;*same, null mb8* | 1.207e-03 | 3.221e-04 | 2.054e-01 |
| &nbsp;&nbsp;**ratio observed ÷ null mb8** | **0.977** | **0.968** | **0.626** |
| Gradient, worst per-param rel L2 | 1.846e-03 | 1.286e-03 | 2.211e-01 |
| &nbsp;&nbsp;*same, null mb8* | 1.825e-03 | 1.106e-03 | 3.168e-01 |
| &nbsp;&nbsp;**ratio observed ÷ null mb8** | **1.012** | **1.163** | **0.698** |
| Gradient, worst per-param cosine | 0.999998329 | 0.999999618 | 0.975404 |
| &nbsp;&nbsp;*same, null mb8* | 0.999998411 | 0.999999665 | 0.948764 |

A comparison with no packing in it reproduces the packed-unpacked difference in
every regime, and in bfloat16 exceeds it.

The floor is not a single number, so read the observation as sitting inside it
rather than below it. At the trained checkpoint in float32, against two different
unpacked partitions:

| null partition | whole-tree rel L2 | observed ÷ null |
| --- | --- | --- |
| 8 microbatches against 16 | 3.221e-04 | 0.968 |
| 4 microbatches against 16 | 3.004e-04 | 1.038 |

The observed 3.118e-04 falls between them. Which side of 1 the ratio lands on is
set by the partition chosen for the null, which is a property of the yardstick
and not of packing. This bounds packing's effect at the floor; it does not show
the effect is zero.

Per parameter at the checkpoint in float32, ten of the thirteen paths disagree
less under packing than under both nulls. Three do not:

| observed | null mb8 | null mb16 | path |
| --- | --- | --- | --- |
| 1.286e-03 | 1.163 | 1.094 | `token_embedder/embedding` |
| 1.834e-04 | 1.113 | 1.074 | `decoder/layers/self_attention/query_norm/scale` |
| 1.425e-04 | 1.591 | 1.591 | `decoder/decoder_norm/scale` |

All three are at or under 1.3e-03 absolute, with cosine above 0.9999996.

TODO(packing): no unpacked null reaches the packed arm's 4096-token row, so none
of them accumulates `token_embedder/embedding` over as many positions. Padding an
unpacked arm to 4096 would isolate row length from packing and settle whether
these three are a row-length effect. Untested.

### 4.3 Where the logp difference comes from

At `--null_control 8` the per-token logp difference is **exactly 0.0** in all
three regimes. At `--null_control 16` it is not: mean 1.654e-05 and max
4.7588e-03, against the packed arm's 1.878e-05 and 4.7588e-03, the two maxima
agreeing to the bit. The variable is rows per shard — 2 at microbatch 8 against 1
at microbatch 4 leaves the forward pass bitwise unchanged, 4 at microbatch 16
does not. A comparison containing no packing therefore reproduces 88% of the
packed arm's mean logp difference and matches its maximum exactly.

Within the packed comparison itself both arms hold 1 row per shard, so what
differs there is the row: 4096 tokens with `segment_ids` against 1536 with
padding changes the length of attention's reduction over the key axis and the
shapes the matmul kernels are tiled for. Masked positions contribute zero either
way, but they are summed over a longer vector.

The difference falls 27× (4.986e-04 → 1.878e-05) between random init and the
trained checkpoint, as the logp scale falls from -247.8 to -13.57.

TODO(packing): two regimes is not enough to establish that the logp difference
scales with logit magnitude rather than with something else that differs between
a random and a trained model. The nulls in §4.2, not this trend, are what rule
out a packing defect.

### 4.4 bfloat16 cannot answer this question

The bfloat16 column FAILs at `--rtol 2e-2`, and the FAIL carries no information:
its null is *larger* than its observation — 2.054e-01 against 1.287e-01
whole-tree, worst cosine 0.948764 against 0.975404. With 8 significand bits the
reassociation floor sits above whatever either comparison is trying to resolve,
so neither a PASS nor a FAIL there means anything. Use `--float32`, which injects
`dtype=float32 weight_dtype=float32 matmul_precision=highest` through
`pyconfig.initialize`. This constrains the measurement only; production training
in bfloat16 never computes the same quantity twice and compares.

### 4.5 Scope

Covered at the floor of float32 arithmetic, on real length statistics, at both
random initialization and a trained checkpoint: the assembler, the segment-id
plumbing, the attention mask, the loss reduction and the backward pass. Not
covered: **MoE expert routing**, since Qwen3-0.6B is dense; **cross-step
optimizer state**, since the harness stops at `fwd_bwd` (§3 covers that range,
with the weaker guarantee described there); and the production **scale and mesh**
— 0.6B against 35B, `pack_size` 4 against 8, `trainer_fsdp` 4 against 8.

Reproducing:

```bash
# Requires a local TPU; each run took about 7 minutes on a 4-chip v5p.
# Do not run from $HOME, which shadows installed packages.
# /tmp/r12.csv is the maz-q35-12 trajectory CSV, at
# gs://mazumdera-bucket-cloud-tpu-multipod-dev/maz-q35/maz-q35-12/trajectories
cd ~/git/tunix
CKPT=gs://maxtext-model-checkpoints/qwen3-0.6b/2025-10-27/scanned/0/items
COMMON="--maxtext_model_name qwen3-0.6b --mesh_fsdp 4 --mesh_tp 1
        --max_prompt_length 512 --max_response_length 1024
        --max_seq_token_per_tpu 4096 --trainer_fsdp 4 --trainer_dp 1
        --num_generations 4 --rollouts_per_update 64
        --train_micro_batch_size 4 --length_csv /tmp/r12.csv"

# observed: packed against unpacked, trained checkpoint, float32
~/maxtext_venv/bin/python qwen35_packing_equivalence.py $COMMON --float32 \
  --maxtext_ckpt_path $CKPT --out /tmp/eq_ckpt_f32.json

# null: unpacked mb8 against unpacked mb4, same checkpoint, same dtype
~/maxtext_venv/bin/python qwen35_packing_equivalence.py $COMMON --float32 \
  --maxtext_ckpt_path $CKPT --null_control 8 --out /tmp/eq_ckpt_null8.json
```

Drop `--maxtext_ckpt_path` for the random-init regime and `--float32` for the
bfloat16 one. `--length_csv` draws completion lengths from a trajectory CSV
produced by a real run; omit it for the built-in lognormal.

Never read an observed run alone, and prefer more than one null: `--null_control`
accepts 8 and 16 here, giving 8-against-16 and 4-against-16 microbatches, and the
two disagree by 7%. `--null_control 32` and `64` extend the range further.

Five ways this comparison returns a verdict that carries no information are in
the appendix, for anyone reimplementing it.

---

## 5. Tests establishing that packed arithmetic equals unpacked arithmetic

All of these are on `main` in both repositories — no branch needed. Counts below
are from runs on a CPU box on 2026-09-18; all exit 0.

### 5.1 The core equivalence tests (MaxText)

`tests/post_training/unit/maxtext_engine_packing_test.py` — **17 tests,
6 subtests, 126.83 s, exit 0**

```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/post_training/unit/maxtext_engine_packing_test.py -q
```

| Class | Test | What it establishes |
| --- | --- | --- |
| `PackedVersusUnpackedLogpsTest` | `test_packed_logps_match_unpacked_per_segment` | Per-token log-probs of a packed row equal those of the same sequences run separately. Every other test in this table depends on this one holding. |
| | `test_segment_positions_matter_only_up_to_a_per_segment_offset` | A sequence's result does not depend on where in the row it landed. Establishes segment isolation in attention and position encoding. |
| `PackedVersusUnpackedGradientsTest` | `test_packed_gradients_match_unpacked_over_the_whole_tree` | Gradients match across the entire parameter pytree, not just the loss scalar. |
| | `test_packed_denominator_counts_segments_not_rows` | The loss denominator is the segment count, not the row count. Getting this wrong silently scales the learning rate by roughly 6× at our density. |
| | `test_packed_accumulation_over_uneven_micro_batches` | Microbatches holding different numbers of segments still accumulate to the correct total — which is exactly our situation (2 to 89 trajectories per microbatch). |
| `PackedDenominatorPartitionsTest` | `test_any_partition_into_packed_micro_batches_gives_the_full_batch_gradient` | *How* the packer splits a step into microbatches cannot change the gradient. Since FFD's split depends on generated lengths and so varies step to step, this is the property that makes our runs reproducible in expectation. |
| | `test_uniform_micro_batches_agree_with_the_mean_of_their_means` | The equal-size case reduces to the simple mean. |
| `PackedBatchStaticsTest` | `test_num_segments_is_not_a_traced_leaf` | `num_segments` is `pytree_node=False`, so it participates in the jit cache key rather than being traced. |
| | `test_changing_num_segments_changes_the_batch_signature` | The corollary: a different segment count is a different signature. This is the mechanism behind the recompilation risk in §3. |
| `PackedUpdateCadenceTest` | `test_packed_micro_steps_accumulate_and_only_the_update_applies` | Optimizer state advances once per step, not once per microbatch. |
| | `test_update_without_accumulated_grads_does_not_advance_the_step` | No silent empty updates. |
| `PackedCompiledPathTest` | `test_lazily_compiled_packed_micro_steps_match_the_eager_branch` | The compiled and eager paths agree numerically. |
| | `test_ahead_of_time_kernels_built_on_a_packed_batch_are_reused` | AOT kernels are reused across steps — the positive case for §3's no-recompilation claim. |
| | `test_a_changed_segment_count_recompiles_rather_than_reusing_the_kernel` | The negative case: a changed count forces a recompile instead of silently reusing a wrong kernel. |
| `ClosedFormPackedLossTest` | `test_packed_loss_equals_the_hand_computed_value` | The packed loss equals a value computed by hand, independent of the implementation. |
| | `test_the_wrong_aggregation_is_a_value_this_test_can_see` | A mutation test: it asserts that substituting the wrong aggregation mode produces a *different* number, so the test above cannot pass vacuously. |
| `AdapterSegmentIdsGateTest` | `test_adapter_passes_tunix_segment_ids_gate` | The MaxText adapter satisfies the signature Tunix inspects before it will send segment IDs at all. |

The last two rows deserve emphasis. `ClosedFormPackedLossTest` is the only test
here that does not compare one implementation against another — it compares
against arithmetic done on paper. Its companion mutation test establishes that
the comparison is sensitive enough to detect a wrong aggregation mode.

### 5.2 Segment IDs reach the model correctly (MaxText)

`tests/post_training/unit/tunix_adapter_test.py` — **18 tests**, classes
`TunixAdapterSegmentIdsTest` (9) and `TunixAdapterAttentionMaskTest` (9).

These cover the handoff where a packing error would be least likely to raise:
whether `segment_ids` is synthesised, passed through, or overridden, and
precedence between an explicit `segment_ids`, an attention mask, and `pad_id`.
Notable:
`test_segment_ids_is_named_so_the_tunix_gate_passes` (the parameter name is part
of the contract), `test_recovery_is_exact_for_every_query_row` (mask →
segment-ID conversion is exact, not approximate), and `test_mask_survives_jit`.

### 5.3 Packing mechanics (Tunix)

`tests/rl/packing_test.py` — **21 tests**, classes `PackItemInvariantTest` (6),
`PackCarriedFieldsTest` (2), `PackCoreTest` (13).

The bin packer itself. `test_every_item_is_packed_only_once` is the unit-level
version of the conservation check in §2.3. `test_pack_bin_exceeds_budget_raises`
and `test_oversized_sequence_errors` establish that a budget violation raises
rather than truncating. `test_row_layout`, `test_reserve_non_action_mask_zeros` and
`test_carried_per_token_fields_in_packed_row` verify that per-token fields —
advantages, completion masks, logprobs — are sliced and placed with the tokens
they belong to, which is where an off-by-one would corrupt training without
crashing.

`tests/rl/common_test.py` — **66 tests**, including:

- `test_packed_logps_match_unpacked_per_segment` — the Tunix-side mirror of §5.1's first test
- `test_aggregate_loss_values` — the aggregation modes against known values
- `test_reduced_equals_unreduced_compute` — the deferred-all-reduce path agrees with the eager one

`tests/rl/algo_core_test.py::AlgoCoreTest::test_grpo_loss_fn_packed_equals_unpacked`
— the GRPO loss itself, packed against unpacked, end to end.

### 5.4 The assembler (Tunix)

`tests/experimental/orchestrator/batch_assembly_test.py` — **41 packing tests**
in `SequencePackedBatchAssemblerTest` (22) and `SequencePackedConversionTest` (19).

`SequencePackedBatchAssemblerTest` covers the streaming behaviour our runs
depend on: packing across input-batch and prompt-group boundaries
(`test_feed_emits_packed_sequence_across_input_batch_boundaries_with_segment_verification`,
`test_feed_cross_group_packing_and_auto_flush`), opening a new bin on overflow,
early emission when a bin fills mid-step, and trajectory-ID tracking — the
bookkeeping that makes §2.3's exact totals possible.

`SequencePackedConversionTest` covers trajectory → pack-item conversion, where
whole-sequence advantages and masks are sliced to the completion
(`test_to_pack_item_whole_sequence_advantages_sliced`,
`test_to_pack_item_whole_sequence_completion_mask_sliced`) and partially
populated optional fields are rejected rather than silently zero-filled.

### 5.5 Reproducing the suites

```bash
# MaxText (AI-Hypercomputer/maxtext, main)
cd maxtext
JAX_PLATFORMS=cpu python3 -m pytest \
  tests/post_training/unit/maxtext_engine_packing_test.py \
  tests/post_training/unit/tunix_adapter_test.py \
  tests/post_training/unit/router_replay_engine_test.py -q
# observed: 17 passed + 6 subtests (126.83s); 28 passed (29.48s)

# Tunix (google/tunix, main)
cd tunix
JAX_PLATFORMS=cpu python3 -m pytest \
  tests/rl/packing_test.py \
  tests/rl/algo_core_test.py \
  tests/rl/common_test.py \
  tests/experimental/orchestrator/batch_assembly_test.py::SequencePackedBatchAssemblerTest \
  tests/experimental/orchestrator/batch_assembly_test.py::SequencePackedConversionTest -q
# observed: 25 passed + 2 subtests (8.34s); 107 passed (19.02s)
```

Do not point pytest at `tests/post_training/unit/` as a directory: collection
imports `vllm`/`tpu_inference` and aborts. Name the files.

A larger end-to-end comparison harness exists at
`tests/end_to_end/tpu/compare_tunix_trainer.py` on the MaxText branch
`packing-compare-harness`; it compares `MaxTextTrainingEngine` against Tunix
`PeftTrainer` on real TPUs. It is not part of the CPU suites above.

---

## 6. What these runs do *not* exercise

Three code paths related to packing are inactive here. Stating this explicitly
matters, because open issues against them do not apply to these results.

**Router replay is not invoked.** `router_replay_gen_model_input_fn` is defined
at `src/maxtext/training_engine/maxtext_engine.py:414` and has no production
caller — the only other occurrence in the repository is a docstring mention at
:480, plus `tests/post_training/unit/router_replay_engine_test.py`. The
distributed path installs `GRPOAdapter`'s `_algo_model_input` instead
(`tunix/experimental/orchestrator/algorithm_adapter.py:279`, wired at
`distributed_rl_engine.py:563-571`). Zero matches in either run's logs.

**Explicit sharding / deferred all-reduce is not reached.** Its gates
(`maxtext_engine.py:232-245`) require `shard_mode == EXPLICIT`, a `data` axis
greater than 1, and every non-data axis equal to 1. Our trainer mesh is data 1 /
FSDP 8, failing two of the three.

**The fix for aggregation over unequal segment lengths is present in this
image.** `sequence-mean-token-mean` and `token-mean` differ only when segments
have unequal lengths, which packing guarantees. The segmented implementation
dispatches automatically whenever `segment_ids` is present
(`tunix/rl/common.py:877`), the production markers from the fix are all present
in the image, and `loss_agg_mode` resolves to `sequence-mean-token-mean` at
runtime. There is no live caveat here.

Relatedly, the step-1 loss spike associated with ragged segments is a pure
KL-term effect, and `BETA=0` in these runs. Neither run shows it: step 1 loss was
0.0144 (run 10) and 0.0394 (run 11), the same order as their neighbours.

---

## 7. Open observations

```
TODO(maz-q35): run 10 shows three isolated single-step loss spikes -- step 23
(1.0258, perplexity 2.79), step 32 (0.8979), step 83 (0.3355) -- each recovering
to the ~0.005 baseline on the next step. step_time at those steps is nominal
(94.80s, 87.37s, 97.43s) and reward_mean is normal, so this is not compilation
and not a packing shape change. Run 11 (20 steps) shows nothing above 0.0539.
Cause not established.
```

```
TODO(tunix): trajectory_logger.py:105 calls mkdir() then asserts is_dir(), above
the gs:// branch written to handle this case. GCS has no directories, so on a
bucket prefix mkdir is a no-op and is_dir is False until some object exists --
the logger can never write its own first file. maz-q35-10 raised this on all 100
steps and produced no CSV. submit.sh works around it by creating one placeholder
object under the prefix before launch; the fix belongs in the logger.
```

```
TODO(tunix): trajectory_logger.py hangs the job. maz-q35-12 stopped at step 22
inside pd.read_csv on the GCS handle at offset 105226240 of a 138462753-byte
object, and never returned or raised; the orchestrator sat at 5 millicores for
105 minutes. Two defects, both from the code alone. (1) The read has no deadline:
the `except Exception` at :122-131 covers a read that fails, not one that never
returns. (2) stop() calls queue.join() with no timeout, and the blocked worker
never reaches its `finally: task_done()`, so the join never returns and the
timeout=10 on the next line is never evaluated -- and since atexit and
_handle_signal both call stop(), SIGTERM hangs too. Separately the write path is
quadratic: each flush re-reads, concatenates and re-uploads the whole CSV. That
was not the cause here (the logger was caught up, at exactly 5,376 rows = 21.00
steps, and parsing 132 MiB takes 1.3 s against a 98 s step), but maz-q35-13's
final stop() still took 68 s to drain. Raised separately with the logger's
authors.
```

---

## 8. Reproducing the runs

```bash
export WANDB_API_KEY=<your key>          # submit.sh asserts this; it is not stored in the file
bash docker/maz-q35/submit.sh 12 100 start
bash docker/maz-q35/submit.sh 12 100 stop   # teardown
```

`submit.sh` takes a run number, a step count, and `start` or `stop`, and is the
single carrier of every deviation from the stock launcher — each one is
commented in place. Set `DRY_RUN=true` to render the manifests without
submitting.

To run the unpacked control arm, pass an empty packing budget:

```bash
MAX_SEQ_TOKEN_PER_TPU= bash docker/maz-q35/submit.sh 13 20 start
```

Note the `-` rather than `:-` in `${MAX_SEQ_TOKEN_PER_TPU-4096}` at
[157](docker/maz-q35/submit.sh#L157): with `:-`, an explicitly empty value is
replaced by the default; with `-`, it is preserved. Confirm the arm rendered
correctly by
checking that `--max_seq_token_per_tpu` appears zero times under `DRY_RUN=true`
(it appears twice in the packed arm) and that the trainer logs
`max_target_length: 1536`.

Rendered manifests under `/tmp` contain the live `WANDB_API_KEY`. Do not share
them.

To check packing on a new run, in order of decreasing cost:

```bash
J=maz-q35-12-orch
P=cloud-tpu-shared-capacity
F='resource.type="k8s_container" AND labels."k8s-pod/jobset_sigs_k8s_io/jobset-name"='

# 1. Was the packed assembler selected, and with what pack_size?
gcloud logging read "$F\"$J\" AND textPayload:\"Using SequencePackedBatchAssembler\"" \
  --project=$P --freshness=1d --format='value(textPayload)'

# 2. How dense are the rows, and do the totals conserve?
gcloud logging read "$F\"$J\" AND textPayload:\"trajectories into microbatch\"" \
  --project=$P --freshness=1d --limit=2000 --format='value(textPayload)'
# sum the counts: it must equal steps x (BATCH_SIZE x NUM_GENERATIONS), exactly.
# divide each by pack_size for segments per row.

# 3. Any recompilation after step 1?
gcloud logging read "$F\"$J\" AND textPayload:\"Train step\"" \
  --project=$P --freshness=1d --limit=2000 --format='value(textPayload)'
# look for a step time that steps up and stays up, not for isolated outliers.
```

The jobset name carries an `-orch` suffix; filtering on the bare job name
returns nothing. Under `DEBUG=1` the container log ring buffer rotates and drops
these lines, so `kubectl logs` is unreliable for this. Cloud Logging retains
them; use it.

With `TRAJECTORY_LOG_DIR` set, analyse the resulting CSV with:

```bash
gcloud storage cp gs://<bucket>/<prefix>/*.csv /tmp/traj.csv
python3 tunix/experimental/examples/common/analyze_trajectories.py /tmp/traj.csv \
  --orchestrator_log /tmp/orchestrator.log --max_response_length 1024
```

The script is pure standard library — no pandas — but it reads local paths only,
so the CSV must be copied down first.

To reproduce the paired packed-against-unpacked comparison in §3, with both CSVs
copied down:

```bash
python3 qwen35_paired_ab.py /tmp/packed.csv /tmp/unpacked.csv 20
```

This one needs `pandas` and `scipy`. It inner-joins on `(global_step, prompt_id)`,
so runs that saw different data silently produce fewer pairs rather than an error;
check that the printed pair count equals `steps × BATCH_SIZE`, which is 320 here.
Output is the paired difference table, the trend of that difference against step
index, the per-step KS distances with step 0 as the noise floor, the power check
against training's own effect, and the cross-arm against within-arm text
comparison.

---

## 9. Summary

| Claim | Evidence |
| --- | --- |
| Packing is active | `Using SequencePackedBatchAssembler … pack_size: 8` in both runs |
| Rows hold ~6.4 sequences | 516 and 101 microbatch records; median 6.38 and 6.50 segments per row |
| Nothing is lost | 25,600 = 100 × 256 and 5,120 = 20 × 256, exactly |
| 6.2× fewer forward/backward passes | 5.16 microbatches per step against 32 measured in the unpacked control |
| 2.33× less padding | 16.9M padded token slots against 39.3M |
| 1.74× faster steps | 98.48 s packed against 171.74 s unpacked, same data, same image |
| No recompilation after step 1 | 99 warm steps, median 97.12 s, stdev 4.48 s, no sustained step-up |
| Packed arithmetic equals unpacked, measured directly | one 64-trajectory set through both assemblers, identical weights: whole-tree gradient relative L2 3.12e-04 at a trained checkpoint, bracketed by two null controls containing no packing at 3.00e-04 and 3.22e-04 (§4) |
| Packed arithmetic equals unpacked, by unit test | 17 + 28 MaxText tests and 132 Tunix tests, all passing |
| Generation quality is unchanged | paired on all 320 prompts: generated length −1.22 chars, 95% CI [−12.1, +9.7]; reward +0.0024, 95% CI [−0.0038, +0.0086] |
| The comparison has power | resolves an effect 12× smaller than training's own 132-char shift over the same 20 steps |
| The difference does not grow with training | post-step-0 divergence at or below the step-0 sampling-noise floor; all trend slopes p ≥ 0.124 |
| The generated text is indistinguishable | cross-arm 5-gram Jaccard 0.1291 against a within-arm same-policy baseline of 0.1242 and 0.1264 |
| Rewards are the ones the trainer used | CSV per-step means match orchestrator `reward_mean` exactly |

---

## Appendix. Five ways to get a meaningless PASS

Recorded for anyone reimplementing the §4 harness. Each was hit while building
it, and each yields a verdict that carries no information.

| Trap | Why the verdict is empty | Guard |
| --- | --- | --- |
| Old logps from a guessed distribution | `r = exp(logp - old_logp)` lands outside the GRPO clip band for every token, and only the unclipped branch of `max(-A·r, -A·clip(r, 1-ε, 1+ε_high))` carries a gradient ([`algo_core.py:489-492`](tunix/rl/algo_core.py#L489-L492)), so both arms return exactly zero and agree vacuously. No constant works either, with logps spanning 400 nats. | `measure_old_logps()` runs the policy forward and uses its own logps; an all-zero tree returns `INVALID` |
| Mean of per-microbatch means | 3 microbatches against 16 weights the same trajectories differently | pool as `Σ unreduced_sum / Σ denominator`, which is what the optimizer sees |
| Per-tensor gradient sums | Cancellation, not disagreement: `post_self_attention_layer_norm/scale` summed to -0.662 against -0.617 on 179.3 of absolute mass, reading as 6.8e-02 where the exact figure is 1.6e-03 | exact per-parameter relative L2 and cosine, at the cost of one extra parameter-sized buffer |
| A `nan` in the tree | Every comparison against `nan` is False, so `sort` places it anywhere and `rows[0]` misses it; one run printed PASS off a worst case of 4.766e-07 | non-finite rows partition out before the sort and return `INVALID` naming the paths |
| `--null_control` below the shard count | A microbatch narrower than `trainer_fsdp × trainer_dp` leaves shards with no rows, which produced the `nan` above; a broken null is worse than none, since it supplies a number that looks like calibration | `parse_args` rejects a value that is not a multiple of the shard count |
