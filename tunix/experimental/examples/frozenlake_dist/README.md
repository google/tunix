# Distributed FrozenLake GRPO Recipe

This directory ports the Qwen3 and Gemma4 E2B FrozenLake recipes under
`examples/frozenlake/` to the experimental distributed RL stack. The control
plane runs on CPU, while the generic distributed trainer and rollout workers
own model execution.

The distributed module keeps only its in-memory dataset and request wiring. It
directly registers and reuses `examples/frozenlake/agent.py` and
`examples/frozenlake/env.py`, avoiding a second copy of the recipe behavior.

The defaults preserve the reference Qwen3 recipe: Qwen3-8B, 64 prompt groups
per full step, 64 prompt groups per optimizer update, 8 generations, 8 turns,
GSPO-token loss, RLOO advantages, asymmetric clipping (`0.003`/`0.005`),
`sequence-mean-token-mean` aggregation, `low_var_kl`, temperature `0.7`, AdamW
(`1e-6`, `b1=0.9`, `b2=0.95`, no weight decay), and gradient clipping at 100.
Maps use the same seed/size/frozen-probability distribution as the original
dataset recipe, but are generated directly in memory without Grain, pandas, or
Parquet.

Install the FrozenLake and distributed extras, then launch from an 8-chip TPU
host:

```bash
pip install -e '.[frozenlake,experimental]'
cd tunix/experimental/examples/frozenlake_dist
WEIGHT_SYNC_MODE=raiden ./launcher.sh
```

For a small infrastructure smoke test, use a smaller supported model and turn
off weight synchronization:

```bash
MODEL_NAME=Qwen3-1.7B MODEL_ID=Qwen/Qwen3-1.7B \
TRAINER_TPU_CHIPS=0,1 TRAINER_TP=2 \
ROLLOUT_TPU_CHIPS=2,3 ROLLOUT_TP=2 \
BATCH_SIZE=1 MINI_BATCH_SIZE=1 NUM_GENERATIONS=2 \
MAX_STEPS=1 MAX_TURNS=3 WEIGHT_SYNC_MODE=none ./launcher.sh
```

`BATCH_SIZE` is the full/global batch and determines checkpointing, global
step advancement, and weight-sync cadence. `MINI_BATCH_SIZE` determines each
optimizer update, so one full step performs
`BATCH_SIZE / MINI_BATCH_SIZE` updates. Multi-step training should use
`WEIGHT_SYNC_MODE=raiden`; `none` is intended for smoke tests, and the launcher
rejects the protocol-only `fallback` mode through the orchestrator validation.

The launcher starts one actor and one rollout worker, so `BETA` must remain
zero. A nonzero KL coefficient requires adding a reference inference worker.
FrozenLake is deterministic by default, matching the reference recipe's
environment construction; set `IS_SLIPPERY=1` to enable Gymnasium's slippery
transitions.

## Gemma4 E2B

The Gemma4 launcher uses the same model, chat parser, trainer settings, and
text-only vLLM overrides as `examples/frozenlake/train_frozenlake.py`. On the
default 4-chip topology it assigns chips 0-1 to a two-way FSDP trainer and
chips 2-3 to a two-way TP rollout worker:

```bash
cd tunix/experimental/examples/frozenlake_dist
WEIGHT_SYNC_MODE=raiden ./run_gemma4_e2b.sh
```

The wrapper selects `google/gemma-4-E2B-it`, fp32 actor parameter storage,
decoder rematerialization, flash attention with block size 256, the Gemma4
non-thinking chat template, two-trajectory trainer/logp micro-batches, and the
reference logp chunk size and rollout limits (`compute_logps_chunk_size=2048`,
`max_concurrency=512`, `max_num_seqs=32`,
`max_num_batched_tokens=8192`). All variables remain overridable through the
environment before invoking the wrapper.

One reference-only feature is not modeled separately: the original recipe's
`sampler_is` threshold is folded into the distributed path's rollout-logprob
importance ratio because Orchestrator V2 currently exposes
`use_rollout_logps`, but not an independent sampler-IS threshold.

## Distributed vs. agentic benchmark

`../rl_efficiency_benchmark/benchmark.py` runs the Gemma4 E2B recipe on the
distributed stack (`run_gemma4_e2b.sh`) or the agentic stack
(`tunix.cli.grpo_main`), then compares recorded timings.
`frozenlake_benchmark.py` generates matched configs for both stacks, and
`frozenlake_benchmark_test.py` checks them against the launcher defaults. Both
stacks use BF16 compute, FP32 loading, exact token continuity, separate
two-chip actor and rollout meshes, and one prompt group (8 trajectories) per
microbatch. The benchmark injects timing hooks through `TRAINER_PROCESS_MAIN`,
`ROLLOUT_PROCESS_MAIN`, and `ORCHESTRATOR_PROCESS_MAIN`.

Use one dedicated four-chip TPU host with a pre-downloaded model, and a Raiden
build whose `RaidenController.register_work_unit` accepts `host_subgrid`. Keep
code, dependencies, and flags fixed across the runs you compare, and use a new
`--output` for every run. Without `--execute`, `run` only writes the plan.

```bash
BENCH=(python3 tunix/experimental/examples/rl_efficiency_benchmark/benchmark.py)
TIMING=stages  # Module timings; use pipeline for end-to-end step time.
runs=()
for repeat in 1 2 3; do
  order=(agentic dist); (( repeat % 2 )) || order=(dist agentic)
  for stack in "${order[@]}"; do
    out="$PWD/artifacts/rl_efficiency/$TIMING-$repeat-$stack"
    "${BENCH[@]}" run --mode "$stack" --timing-mode "$TIMING" --output "$out" \
      --model-dir /path/to/gemma-4-E2B-it --cache-root "$PWD/artifacts/cache" \
      --hardware v5p-4 --execute
    runs+=("$out")
  done
done
"${BENCH[@]}" report "${runs[@]}" > "$TIMING-comparison.json"
```

Defaults run 150 steps (5 warmup) with 64 prompt groups × 8 generations;
`run --help` lists all flags. `stages` waits for device completion after
training, log-probs, and weight sync (compare
`api_calls.<module>.host_time_per_step_seconds`). `pipeline` keeps intra-step
overlap (compare `step_time_seconds.mean`). Do not mix modes in one report.

The report gives, for every numeric summary field, each stack's median across
runs, `dist_minus_agentic`, and `dist_relative_to_agentic_percent`. Positive
time differences mean distributed is slower. It rejects failed or incomplete
runs, timed-out trajectories, missing worker hooks, and mismatched workload,
hardware, model, dataset, dtype, timing mode, or per-step training work.
