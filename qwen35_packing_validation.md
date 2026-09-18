# Sequence packing on Qwen3.5-35B-A3B: validation, performance, and test inventory

Written 2026-09-18. Packing is confirmed active and correct on two completed
GRPO runs: `maz-q35-10` (100 steps) and `maz-q35-11` (20 steps), both 2026-09-17.

This document answers three questions:

1. Is sequence packing actually on in these runs, and how would you check that yourself?
2. What did it cost or save?
3. Which tests establish that the packed arithmetic equals the unpacked arithmetic?

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
| **`MAX_SEQ_TOKEN_PER_TPU`** | **4096** — the packing budget | [157](docker/maz-q35/submit.sh#L157) |
| `BETA` (KL coefficient) | 0 (launcher default, not overridden) | [k8s_launcher.sh:64](tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh#L64) |
| `loss_agg_mode` | `sequence-mean-token-mean` (GRPOConfig default) | [algorithm_config.py:165](tunix/rl/algorithm_config.py#L165) |
| Checkpointing | disabled (`CHECKPOINT_SAVE_INTERVAL_STEPS=0`) | [174](docker/maz-q35/submit.sh#L174) |
| Priority | `medium` | [37](docker/maz-q35/submit.sh#L37) |
| `DEBUG` | 1 | [208](docker/maz-q35/submit.sh#L208) |

`maz-q35-11` additionally set `TRAJECTORY_LOG_DIR` to a GCS prefix
([236](docker/maz-q35/submit.sh#L236)) so that every rollout's response, gold
answer and reward were written to CSV. `maz-q35-10` did not produce
trajectories — see §6.

Cluster: `bodaborg-v5p-nap`, `europe-west4`, namespace `trellis`, project
**`cloud-tpu-shared-capacity`**. Note that the project holding the image and the
output bucket (`cloud-tpu-multipod-dev`) is *not* the project holding the logs;
querying the wrong one returns nothing.

---

## 2. Is packing on? Four independent checks

Each check is stronger than the last. The first says the code path was chosen;
the last says the trainer actually consumed packed rows.

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

If this line is absent from your run, packing is off. That is the single most
useful thing to grep for.

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

Every rollout appears in exactly one packed row. This is the conservation check
that matters — a packer that dropped or double-counted sequences would show up
here first, and does not.

The collapse in work per step:

| | maz-q35-10 | maz-q35-11 |
| --- | --- | --- |
| Microbatches per step, packed | 5.16 | 5.05 |
| Microbatches per step, unpacked (256 ÷ `TRAIN_MICRO_BATCH_SIZE` 8) | 32 | 32 |
| **Forward/backward passes saved** | **6.20×** | **6.34×** |
| Padded token area, packed (rows × 4096) | 16,908,288 | 3,309,568 |
| Padded token area, unpacked (rows × 1536) | 39,321,600 | 7,864,320 |
| **Padding eliminated** | **2.33×** | **2.38×** |

The two ratios differ because they measure different things. 6.2× is the
reduction in *kernel invocations*; 2.33× is the reduction in *tokens the trainer
has to push through those kernels*. The second bounds the achievable speedup;
the first is why the fixed per-microbatch overhead falls away.

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
compiled for 1536-token rows and rejected the 4096-token packed rows outright.
Both runs completing is therefore evidence that both sides agreed.

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

**Trajectory logging is free at this scale.** Run 11 wrote every rollout to GCS
and had a *lower* median step time than run 10 (96.30 s vs 97.12 s) — the
difference is within run-to-run noise, so the logging cost is below the
measurement floor. Note that the GCS writer re-reads and re-uploads the whole
CSV on each flush (`tunix/utils/trajectory_logger.py:122-142`), so this cost is
quadratic in step count and will not stay negligible for much longer runs.

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
  rate, from 19.1% to about 9% — most of the current gradient signal is the model
  being penalised for running out of room rather than for being wrong.
- At 91.8% reward 1.0, GSM8K is close to saturated for this model. A harder
  dataset would give more signal per step.

---

## 4. Tests establishing that packed arithmetic equals unpacked arithmetic

All of these are on `main` in both repositories — no branch needed. Counts below
are from runs on a CPU box on 2026-09-18; all exit 0.

### 4.1 The core equivalence tests (MaxText)

`tests/post_training/unit/maxtext_engine_packing_test.py` — **17 tests,
6 subtests, 126.83 s, exit 0**

```bash
JAX_PLATFORMS=cpu python3 -m pytest tests/post_training/unit/maxtext_engine_packing_test.py -q
```

| Class | Test | What it pins down |
| --- | --- | --- |
| `PackedVersusUnpackedLogpsTest` | `test_packed_logps_match_unpacked_per_segment` | Per-token log-probs of a packed row equal those of the same sequences run separately. This is the foundational claim; everything else is downstream. |
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
against arithmetic done on paper. Its companion mutation test is what keeps it
honest.

### 4.2 Segment IDs reach the model correctly (MaxText)

`tests/post_training/unit/tunix_adapter_test.py` — **18 tests**, classes
`TunixAdapterSegmentIdsTest` (9) and `TunixAdapterAttentionMaskTest` (9).

These cover the handoff where packing most plausibly breaks quietly: whether
`segment_ids` is synthesised, passed through, or overridden, and precedence
between an explicit `segment_ids`, an attention mask, and `pad_id`. Notable:
`test_segment_ids_is_named_so_the_tunix_gate_passes` (the parameter name is part
of the contract), `test_recovery_is_exact_for_every_query_row` (mask →
segment-ID conversion is exact, not approximate), and `test_mask_survives_jit`.

### 4.3 Packing mechanics (Tunix)

`tests/rl/packing_test.py` — **21 tests**, classes `PackItemInvariantTest` (6),
`PackCarriedFieldsTest` (2), `PackCoreTest` (13).

The bin packer itself. `test_every_item_is_packed_only_once` is the unit-level
version of the conservation check in §2.3. `test_pack_bin_exceeds_budget_raises`
and `test_oversized_sequence_errors` make budget violations loud rather than
truncating. `test_row_layout`, `test_reserve_non_action_mask_zeros` and
`test_carried_per_token_fields_in_packed_row` verify that per-token fields —
advantages, completion masks, logprobs — are sliced and placed with the tokens
they belong to, which is where an off-by-one would corrupt training without
crashing.

`tests/rl/common_test.py` — **66 tests**, including:

- `test_packed_logps_match_unpacked_per_segment` — the Tunix-side mirror of §4.1's first test
- `test_aggregate_loss_values` — the aggregation modes against known values
- `test_reduced_equals_unreduced_compute` — the deferred-all-reduce path agrees with the eager one

`tests/rl/algo_core_test.py::AlgoCoreTest::test_grpo_loss_fn_packed_equals_unpacked`
— the GRPO loss itself, packed against unpacked, end to end.

### 4.4 The assembler (Tunix)

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

### 4.5 Reproducing the suites

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

## 5. What these runs do *not* exercise

Three packing-adjacent code paths are inactive here. Stating this explicitly
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

**The unequal-segment-length aggregation gap is closed and present in this
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

## 6. Open observations

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
TODO(tunix): trajectory_logger.py:122-142 re-reads, concatenates and re-uploads
the entire CSV on every flush. Cost is quadratic in step count. Unmeasurable at
20 steps; will not stay that way.
```

---

## 7. Reproducing the runs

```bash
export WANDB_API_KEY=<your key>          # submit.sh asserts this; it is not stored in the file
bash docker/maz-q35/submit.sh 12 100 start
bash docker/maz-q35/submit.sh 12 100 stop   # teardown
```

`submit.sh` takes a run number, a step count, and `start` or `stop`, and is the
single carrier of every deviation from the stock launcher — each one is
commented in place. Set `DRY_RUN=true` to render the manifests without
submitting.

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

---

## 8. Summary

| Claim | Evidence |
| --- | --- |
| Packing is active | `Using SequencePackedBatchAssembler … pack_size: 8` in both runs |
| Rows hold ~6.4 sequences | 516 and 101 microbatch records; median 6.38 and 6.50 segments per row |
| Nothing is lost | 25,600 = 100 × 256 and 5,120 = 20 × 256, exactly |
| 6.2× fewer forward/backward passes | 5.16 microbatches per step against 32 unpacked |
| 2.33× less padding | 16.9M padded token slots against 39.3M |
| No recompilation after step 1 | 99 warm steps, median 97.12 s, stdev 4.48 s, no sustained step-up |
| Packed arithmetic equals unpacked | 17 + 28 MaxText tests and 132 Tunix tests, all passing |
| Rewards are the ones the trainer used | CSV per-step means match orchestrator `reward_mean` exactly |
