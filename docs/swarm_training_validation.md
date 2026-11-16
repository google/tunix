# Shared-policy swarm validation

Date: November 11, 2026. Scope: the first local-transport slice of the
[system design](swarm_training_design.md). Entry point and folder rationale:
[shared math recipe](../examples/swarm_math/README.md).

Initial implementation acceptance: **292 tests passed**, including **43 swarm tests**, with
**58 additional subtests passed**. The real-tokenizer check was enabled. The
three existing HTTP OPD tests required a separate run with localhost binding
permitted; they passed there. The other 289 tests passed in the sandbox.

After merging `origin/main` at `73b21537a` (merge `9e3547c29`), all **768 selected
regression tests** passed across the suite and targeted reruns. The local native
packing extension was rebuilt from the merged source; its **157 batch-assembly
tests** and **46 swarm/OPD training tests** passed after rebuilding. The swarm
path now uses `GRPOConfig`, shared loss/input registration, and plain dataclass
payloads. The two-round CPU recipe retained a parameter L2 change of `0.0648973`
and maximum sampler/actor logprob error of `3.21e-7`.

## Evidence and boundaries

| Layer | Executed check | Result |
| --- | --- | --- |
| Reference control flow | Execute the pinned upstream orchestration and verdict functions with scripted calls; compare exact role order, prompts, final answer and validity | Approval, rejection, whitespace, missing/unknown verdict, both tags, reversed tags and Chinese-character validity agree |
| Credit | Native NumPy role/episode credit against extracted upstream PyTorch advantage function | Agrees within `2e-6` for mixed rewards, unequal call counts, invalid-call penalties, missing roles, multiple tasks and singletons |
| Zero variance | Constant decimal scores and unequal call counts | Explicit zero advantage; avoids an upstream float32 rounding artifact |
| Loss and gradients | Native JAX GRPO versus original PyTorch dual-clip loss with independent autoregressive scoring | Loss and all parameter gradients agree within `2e-6`; finite-difference checks also pass |
| Masks | Active EOS equal to pad, padding rows, loss-masked observations, separate attention masks | Correct gradients and preserved context; padded assembler checked in both compiled and Python paths |
| Categorical training | Two real native optimizer updates under role normalization and episode baseline | Each updated parameter tensor agrees with the upstream full-batch gradient oracle within `2e-6`; recorded updates reduce that batch's surrogate loss |
| Qwen3 integration | Random 1-layer Qwen3, 16 hidden dimensions, 8-token fixture vocabulary, native KV sampler and native trainer | Two successful updates; sampler/actor logprob maximum error `4.77e-7`; parameter L2 changes approximately `0.315` and `0.111` |
| Real Qwen tokenizer | Official pinned Qwen3-4B tokenizer/config, unchanged Solver prompt and golden token IDs | Exact 101-token prompt; non-thinking suffix and middle truncation verified |
| Artifacts | CPU CLI output, exact trace JSONL and actual native checkpoint writer | Config, report, parameters and model/optimizer checkpoint emitted |
| Input boundaries | Duplicate IDs, mixed/stale policy versions, partial groups, bad sampler responses, logprob mismatch and all-zero signal | Fail before update, or skip optimizer/publication for zero signal |

These are correctness and integration checks. The pretrained Qwen3-4B weights
were not loaded or trained on this CPU host. Random tiny-Qwen and categorical
accuracy are not benchmark results or evidence of collaboration gains. No remote
episode RPC, Raiden, accelerator throughput, convergence or recovery result is
claimed. The single-Solver evaluation is a diagnostic, not a matched-budget
trained baseline.

## Independent reference

DrMAS commit: `c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9`.
[Original repository](https://github.com/langfengQ/DrMAS/tree/c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9).
The test fixture contains exact extracted functions and source-file hashes;
tests execute those functions instead of reimplementing a second NumPy formula.
[Attribution and copied files](../examples/swarm_math/THIRD_PARTY.md).

The port preserves `ddof=1`, epsilon `1e-6`, singleton mean/std `0/1`, and the
source's distinction between per-role call normalization and per-task episode
normalization. The shared variant is explicitly enabled; the original launcher
defaults to separate models. The native loss supports both `global_token_mean`
(one accumulated numerator divided by one global active-token denominator) and
`microbatch_token_mean` (`loss_microbatch_rows=4` in the Qwen3-4B profile,
matching upstream `verl`'s `ppo_micro_batch_size_per_gpu=4` equal-microbatch
weighting via per-call advantage weights and verified against the upstream
PyTorch per-microbatch loss and gradient oracle).

One explicit numerical correction is applied: repeated equal float32 decimal
scores can have a rounded mean and nonzero computed standard deviation. For
example, the pinned PyTorch function on this host produces an advantage around
`0.00739` for eight identical `-0.1` rewards. Equal-score multi-call groups now
return zero, and episode means accumulate in float64 before conversion to
float32. The source singleton convention remains unchanged. Zero-signal rounds
also skip AdamW, preventing weight decay from creating an unintended update.

## Fixes found by integration

1. `PaddedBatchAssembler` discarded `completion_attention_mask`. It now pads
   and preserves that field independently of `completion_mask` through both
   implementations.
2. Native sampling omitted the sampled EOS. An opt-in `include_eos` flag
   preserves termination tokens and aligned scores, including immediate EOS;
   existing sampler callers keep their previous default behavior.
3. `nnx.clone` shares array buffers. Local serving snapshots now copy storage
   and retain each destination leaf's sharding, preventing donated optimizer
   buffers from invalidating inference state.
4. Constant decimal rewards could create artificial advantages. Explicit
   equal-score handling and wider episode-mean accumulation remove this signal.

## Reproduce

The isolated local environment used Python 3.13.1, JAX/JAXlib 0.11.0, Flax
0.12.8, Optax 0.2.8, PyTorch 2.15.0, Transformers 4.57.1 and safetensors 0.7.0
on one Apple Silicon CPU device. No TPU/GPU was available.

For the distributed-engine regression tests, generate RPC stubs using the
[distributed setup instructions](../tunix/dist_rl/distributed/examples/README.md)
and rebuild native extensions after pulling changes to their C++ sources.

```bash
JAX_PLATFORMS=cpu python -m pytest tests/dist_rl/swarm -q

JAX_PLATFORMS=cpu python -m examples.swarm_math.train \
  --backend cpu --output /tmp/new-swarm-cpu-run

# Optional: point at the downloaded tokenizer-only snapshot or full snapshot.
TUNIX_SWARM_QWEN_TOKENIZER=/path/to/pinned/tokenizer \
  JAX_PLATFORMS=cpu python -m pytest tests/dist_rl/swarm/recipe_test.py -q
```

The real tokenizer check skips without that environment variable; all core
numerical and training tests are offline. Existing sampler, batch assembly,
payload builder and OPD gradient/training regressions are included in the
acceptance run because the mask and EOS changes touch shared infrastructure.

```bash
TUNIX_SWARM_QWEN_TOKENIZER=/path/to/pinned/tokenizer JAX_PLATFORMS=cpu \
  python -m pytest tests/dist_rl/swarm \
  tests/dist_rl/rollout/vanilla_sampler_test.py \
  tests/dist_rl/orchestrator/batch_assembly_test.py \
  tests/dist_rl/orchestrator/payload_builder_test.py \
  tests/dist_rl/distillation/on_policy_test.py \
  tests/dist_rl/distillation/opd_gradient_test.py \
  tests/dist_rl/distillation/opd_training_test.py -q
```

The reference training corpus was also prepared successfully from
`aaabiao/dapo_filter` revision `0b10fea35812902fe4f6985f2064424ec460f9c0`:
13,044 raw rows, 12,613 retained unique problems, 10 conflicting problem groups
excluded, and a deterministic split of 12,357 train / 256 dev. The data manifest
records excluded problem hashes and SHA-256 hashes for both split files.
This development split is an explicit adaptation; it does not consume the
original recipe's published benchmark validation questions.

## TPU v5p-256 bring-up validation

On November 11, 2026 (`hy-swarm-4b-1111k`), the two-step `examples/swarm_math`
recipe ran end-to-end on one TPU `v5p-256` slice (128 chips under Pathways on
GKE) with pinned `Qwen/Qwen3-4B` (`1cfa9a7208912126459214e8b04321603b3df60c`)
and `aaabiao/dapo_filter` (`0b10fea35812902fe4f6985f2064424ec460f9c0`):

- **Resource layout**: 64 rollout chips (`dp=8, tp=8`, `InprocessVllmSampler`,
  `bfloat16`, `hbm_utilization=0.7`, `max_num_seqs=64`,
  `max_num_batched_tokens=16384`, `reshard_chunk_size=1`) + 64 training chips
  (`fsdp=8, tp=8`, `bfloat16` compute with `float32` master parameters,
  `use_flash_attention=True` (later found to give wrong gradients; see
  "Gradient correctness"), `flash_attention_block_size=1024`,
  `logps_chunk_size=1024`, `remat_config=decoder`, `micro_batch_size=16`).
- **In-container CPU regression suite before TPU execution**:
  `123 passed` across `tests/dist_rl/swarm`,
  `tests/dist_rl/rollout/vanilla_sampler_test.py`,
  `tests/models/safetensors_loader_test.py`, and
  `tests/sft/sharding_utils_test.py` with `TUNIX_SWARM_QWEN_TOKENIZER` set to
  the pinned `Qwen/Qwen3-4B` snapshot.
- **End-to-end 2-step metrics** (`train.py` completed in `3672.5s`, exit `0`):
  - `initial_team_eval`: accuracy `0.515625` (256 attempts, `2.2109375` mean
    calls, `2443.13671875` mean generated tokens)
  - Step 1 (`policy_version=1`, `optimizer_applied=true`): `256` episodes, `556`
    calls, `576,722` active tokens, `reward_mean=0.53125`,
    `max_logprob_error=1.42315`, `mean_logprob_error=0.00887857`, training loss
    `0.088701`
  - Step 2 (`policy_version=2`, `optimizer_applied=true`): `256` episodes, `546`
    calls, `536,813` active tokens, `reward_mean=0.5703125`,
    `max_logprob_error=1.29769`, `mean_logprob_error=0.00974538`, training loss
    `0.002437`
  - `final_team_eval`: accuracy `0.51171875` (256 attempts, `2.1875` mean
    calls, `2362.47265625` mean generated tokens)
  - `final_single_solver_eval`: accuracy `0.55859375` (256 attempts, `1.0` mean
    call, `1592.6953125` mean generated tokens)
  - Artifacts: `512` recorded training episodes in `train_traces.jsonl`
    (verified trailing `<|im_end|>` token `151645` kept in `completion_ids` and
    stripped from decoded prompt text) and Orbax checkpoint committed at step
    `2` on GCS.

### Fixes from TPU integration

1. `workflow.chat_encoder` passed `return_dict=False` to
   `apply_chat_template` and sanitized control tokens before encoding, fixing a
   `BatchEncoding` list-concatenation error with `transformers >= 4.57`.
2. `Qwen3._init_sharded_cache` and `safetensors_loader` chunked Pathways tensor
   placement (`_PATHWAYS_PLACEMENT_BATCH_SIZE = 4` + `block_until_ready`)
   fixed unsharded KV cache allocation and Pathways host-buffer thread limits
   when loading `Qwen3-4B` across multi-host meshes.
3. `examples/swarm_math` adopted split rollout/train meshes with
   `InprocessVllmSampler` (`tpu_inference` JAX backend), `VllmMathSampler`
   clearing per-request `seed` and normalizing `top_k=None -> 0`, and
   `bfloat16` SplashAttention training.
4. Diagnostics (`hy-swarm-diag-1111i` vs `hy-swarm-diag-1111j`) showed that
   enabling `prefix_caching` / `async_scheduling` with DP=8 + RPA v2 caused
   rare ~43-nat first-token (`</think>`) logprob spikes on repeated concurrent
   prompts. Setting `enable_prefix_caching=False` and
   `rollout_vllm_async_scheduling=False` eliminated the spikes (`max` dropped
   from `43.63` to `0.907` across `39,046` diagnostic tokens, `c0_mean` from
   `2.50` to `0.068`). Over longer runs (~600k–900k tokens/step), normal per-step
   maxima are 1.5–6.0 nats with rare single-token `bfloat16` TPU vLLM glitches
   of 10–25 nats while the step mean stays around 0.006–0.008.
   `max_logprob_error` in `configs/qwen3_4b.json` is therefore `100.0` (still
   catching systemic alignment failures while `sampler_is: "token"` downweights
   rare sampler-disagreement tokens via `min(exp(logp_trainer - logp_vllm), 2.0)`).

## Next acceptance gate

1. Add worker-side episode collection and the distributed engine adapter; verify
   the same loss/gradient invariants with actual transport and weight sync.
2. Extend the committed round-boundary resume (implemented for the local
   engine and validated on TPU) to failures between optimizer update and policy
   publication in a distributed deployment.
3. Run role/global normalization, untrained-team and matched-budget single-agent
   baselines over multi-epoch budgets on locked tests with multiple seeds before
   drafting a performance claim.

## Gradient correctness

Runs 1–3 of the Qwen3-4B reproduction lost training reward and validation
accuracy within 25–50 steps. A diagnostic JobSet then computed the token-mean
policy gradient of 32 real calls (Run 2's step-0 trace, two 16-row
microbatches padded to 8192 + 4096 tokens) through `PeftTrainer.fwd_bwd` in
four configurations on the production mesh (`fsdp=8, tp=8`):

| Configuration | Cosine vs. float32 reference | Norm |
| --- | --- | --- |
| splash attention, log-prob chunk 1024 (production) | 0.155 | 5.27 |
| splash attention, no chunking | 0.158 | 5.21 |
| XLA reference attention, bfloat16 | 0.971 | 1.11 |
| XLA reference attention, float32, highest matmul precision | 1 | 1.08 |

Per layer, the splash gradient matched the reference in layers 16–35
(cosine 0.93–0.999) and was wrong below layer 15 (cosine mostly between −0.1
and +0.2, norm 2–23 times too large), so the error enters the backward pass in
the middle of the stack and grows toward the input. Forward log-probs were correct: the same job
measured a mean absolute sampler/trainer difference of 0.0095 nats, and the
mean sampled negative log-prob (0.2350) equals the mean policy entropy
(0.2373), so rollouts are sampled from the current policy. `swarm_math` now
uses `dtype: float32` XLA reference attention for training (preventing both
the splash backward corruption and `bfloat16` early-layer `RMSNorm` /
`q_norm` / `k_norm` / RoPE gradient drift on long multi-turn contexts, where
layer-7 `input_layernorm` cosine fell to −0.09 and `k_norm` to +0.13 in
`bfloat16`) together with `sampler_is: token` truncated importance weights on
an exact on-policy PPO ratio baseline (`is_ratio = 1.0`), and rejects
`use_flash_attention: true` or `dtype != "float32"`.

With all recipe fixes enabled on TPU `v5p-256` (`dtype: "float32"`,
`use_flash_attention: false`, `sampler_is: "token"`,
`loss_reduction: "microbatch_token_mean"` with `loss_microbatch_rows: 4`, and
`MATH_MULTIAGENT_TEMPLATE`), the shared `Qwen/Qwen3-4B` Solver-Verifier run
(`hy-swarm-run5-1112a`, 32 tasks x 8 attempts/step) climbs steadily without
entropy or length collapse on `test_sampled` (`110` questions x `4` samples at
`T=0.6, top_p=0.95`):

| Step | AIME'24 avg@4 / pass@4 | AIME'25 avg@4 / pass@4 | MATH500[:50] avg@4 / pass@4 | Overall avg@4 | Macro avg@4 / pass@4 | Mean tokens / episode |
| --- | --- | --- | --- | --- | --- | --- |
| 0 (base) | 20.0 / 26.7 | 18.3 / 26.7 | 83.0 / 90.0 | 48.2 | 40.4 / 47.8 | 2,176 |
| 75 | 22.5 / 36.7 | 21.7 / 36.7 | 85.5 / 96.0 | 50.9 | 43.2 / 56.4 | 2,764 |
| 125 | 29.2 / 46.7 | 21.7 / 36.7 | 83.5 / 90.0 | 51.8 | 44.8 / 57.8 | 2,765 |
| 200 | 38.3 / 60.0 | 29.2 / 46.7 | 86.5 / 90.0 | 57.7 | 51.3 / 65.6 | 3,592 |
| 250 | 38.3 / 63.3 | 31.7 / 46.7 | 87.0 / 92.0 | 58.6 | 52.3 / 67.3 | 3,906 |
| 275 | 44.2 / 63.3 | 32.5 / 53.3 | 88.0 / 94.0 | 60.9 | 54.9 / 70.2 | 4,393 |
| 350 | 47.5 / 63.3 | 34.2 / 50.0 | 87.5 / 94.0 | 62.0 | 56.4 / 69.1 | 3,235 |
| 400 | 42.5 / 66.7 | 37.5 / 56.7 | 89.5 / 96.0 | 62.5 | 56.5 / 73.1 | 4,594 |
| 475 | 48.3 / 66.7 | 39.2 / 60.0 | 90.0 / 92.0 | 64.8 | 59.2 / 72.9 | 4,672 |
| 500 | 42.5 / 60.0 | 41.7 / 63.3 | 89.5 / 96.0 | 63.6 | 57.9 / 73.1 | 5,139 |

### Full Paper Table 1 Evaluation (`1,547` questions × `16` samples = `24,752` episodes)

Evaluated at `T=0.6, top_p=0.95` across all six published benchmarks (`AIME'24`, `AIME'25`, `AMC'23`, `MATH500`, `Minerva`, `OlympiadBench`):

| Benchmark (`n`) | Untrained Base | Paper MA-GRPO (Sharing) | Paper Dr. MAS (Sharing) | **Tunix Step 475** | **Tunix Step 500** | **Step 500 Decontam.** |
| --- | --- | --- | --- | --- | --- | --- |
| AIME'24 (30) | 19.6 / 43.3 | 39.3 / 64.0 | 39.3 / 63.3 | **46.2 / 80.0** | **45.0 / 80.0** | 45.0 / 80.0 |
| AIME'25 (30) | 18.8 / 43.3 | 31.4 / 53.3 | 38.1 / 63.3 | **41.5 / 73.3** | **39.8 / 70.0** | 39.8 / 70.0 |
| AMC'23 (40) | 63.7 / 90.0 | 85.6 / 95.0 | 87.3 / 95.0 | **88.9 / 97.5** | **90.3 / 97.5** | 89.8 / 97.1 |
| MATH500 (500) | 81.4 / 95.4 | 89.5 / 96.2 | 90.5 / 96.0 | **92.2 / 97.2** | **92.2 / 97.2** | 92.1 / 97.2 |
| Minerva (272) | 31.8 / 44.1 | 37.5 / 50.0 | 40.9 / 53.3 | 38.4 / 49.3 | 37.8 / 49.3 | 37.8 / 49.3 |
| OlympiadBench (675) | 45.2 / 67.7 | 57.6 / 65.6 | 58.2 / 68.6 | **59.2 / 72.1** | **59.5 / 72.7** | 59.5 / 72.7 |
| **Macro Average (1,547)** | **43.4 / 64.0** | **56.8 / 70.7** | **59.0 / 73.2** | **61.1 / 78.2** | **60.8 / 77.8** | **60.7 / 77.7** |

## Milestone SW3 Validation: Trainable R²Orch Graph Director + Frozen Workers

On November 16, 2026 (`hy-swarm-r2o-run1` + paired `hy-swarm-r2o-ev0`), `examples/swarm_r2orch` validated `PolicyBinding`, `RecordingRuntime.from_bindings`, and frozen-call payload isolation (`trainable=False`, `loss_mask=0`) on a 32-chip TPU `v5p` slice (`16` fp32 trainer chips + `8` trainable Director vLLM chips + `8` frozen `Qwen/Qwen3-4B` Worker vLLM chips) over 100 optimizer steps across the four self-contained `beita6969/R2Orch@12ceb7563` domains (`aime`, `hotpotqa`, `medqa`, `mbppplus`, `3,658` train tasks):

- **Bit-exact Worker isolation:** `frozen_worker_max_norm_drift == 0.0` verified online across all 100 optimizer steps and final evaluations while `director_norm_delta_l2 = 1.04e-3`.
- **Benchmark gains with frozen Workers (`k=2` at `T=0.3, top_p=0.95`):**
  - **Heldout (`126` tasks):** macro `avg@2 / pass@2` rose from `53.96 / 57.55` (`initial_eval`: `54.71 / 59.17`) to **`59.24 / 63.28` at Step 60** and **`58.39 / 64.06` at Step 100** (`AIME` `28.33/33.33 -> 36.67/50.00`, `HotpotQA` `50.00/56.25 -> 68.75/71.88`, `finish_rate` `90.5% -> 98.8%`, `worker_cost` `-13.2%`).
  - **Locked Test IID (`414` tasks):** macro `avg@2 / pass@2` rose from `44.30 / 47.47` to **`47.03 / 48.79`** (micro accuracy `52.42% -> 55.80%`, `HotpotQA` `+8.59 pp`, `MBPP+` `+3.12 pp`, `worker_cost` `-22.5%`).
  - **Locked Test OOD (`256` tasks — `MATH-Hard` + `MuSiQue`):** macro `avg@2 / pass@2` rose from `42.77 / 48.05` to **`44.53 / 50.78`**, with **`MATH-Hard` (`128`) jumping `+6.64 / +10.16` pp (`54.30 / 59.38 -> 60.94 / 69.53`)** and multi-agent graph construction rising `11.8×` (`0.4% -> 4.7%`).
