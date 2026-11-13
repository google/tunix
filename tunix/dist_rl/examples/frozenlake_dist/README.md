# Distributed FrozenLake GRPO Recipe

This directory ports the Qwen3 and Gemma4 E2B FrozenLake recipes under
`examples/frozenlake/` to the distributed RL stack. It runs on the shared
launcher and node mains in `tunix/dist_rl/launch/`: the control plane runs on
CPU, while the generic distributed trainer and rollout workers own model
execution.

The distributed module keeps only its in-memory dataset. It directly registers
and reuses `examples/frozenlake/agent.py` and `examples/frozenlake/env.py`,
avoiding a second copy of the recipe behavior, and its `create_dataset` gives
the orchestrator one task record per map. Maps use the same
seed/size/frozen-probability distribution as the original dataset recipe, but
are generated directly in memory without Grain, pandas, or Parquet; they are
shuffled once and repeated in the same order every epoch.

`frozenlake.yaml` (on top of `launch/common/base_dist_config.yaml`) preserves
the reference Qwen3 recipe: Qwen3-8B, 64 prompt groups per full step, 64 prompt
groups per optimizer update, 8 generations, 8 turns, GSPO-token loss, RLOO
advantages, asymmetric clipping (`0.003`/`0.005`), `sequence-mean-token-mean`
aggregation, `low_var_kl`, temperature `0.7`, AdamW (`1e-6`, `b1=0.9`,
`b2=0.95`, no weight decay), and gradient clipping at 100.

Install the FrozenLake and distributed extras, then launch from an 8-chip TPU
host. The tokenizer loads from the Hugging Face hub id, which needs `HF_TOKEN`
only for gated repos; on a host without hub access, also pass
`tokenizer_config.tokenizer_path=<model_config.model_download_path>`:

```bash
pip install -e '.[frozenlake,experimental]'
bash tunix/dist_rl/examples/frozenlake_dist/launcher.sh
```

Change settings with `section.key=value` arguments (the keys of
`frozenlake.yaml` and `base_dist_config.yaml`); `--dry-run` prints the node
commands without starting them. The launcher fails when environment variables
of the old launcher, such as `BATCH_SIZE`, `MAX_STEPS`, or `WEIGHT_SYNC_MODE`,
are set: unset them and pass the settings as arguments. The node logs go to
`artifacts/frozenlake/*.log` (or to `LOG_ROOT`), the checkpoints to
`artifacts/frozenlake/checkpoints`, the metrics to
`artifacts/frozenlake/metrics`, and the per-step trajectories to
`artifacts/frozenlake/metrics/trajectories`.

For a small infrastructure smoke test, use a smaller supported model and turn
off weight synchronization:

```bash
bash tunix/dist_rl/examples/frozenlake_dist/launcher.sh \
  model_config.model_name=Qwen3-1.7B model_config.model_id=Qwen/Qwen3-1.7B \
  tokenizer_config.tokenizer_path=Qwen/Qwen3-1.7B \
  model_config.mesh.shape="(1,2)" \
  cluster_config.local.trainer_devices=0,1 \
  cluster_config.local.rollout_devices=2,3 \
  cluster_config.local.tpu_chips_per_host_bounds=1,2,1 \
  batch_size=1 rl_training_config.mini_batch_size=1 \
  rl_training_config.train_micro_batch_size=1 \
  rl_training_config.compute_logps_micro_batch_size=1 \
  agentic_grpo_config.num_generations=2 rl_training_config.max_steps=1 \
  agentic_grpo_config.max_turns=3 env_kwargs.max_steps=3 \
  weight_sync_config.mode=none vllm_config.init_with_random_weights=false
```

`batch_size` is the full/global batch and determines checkpointing, global step
advancement, and weight-sync cadence. `rl_training_config.mini_batch_size`
determines each optimizer update, so one full step performs
`batch_size / mini_batch_size` updates. Multi-step training should use
`weight_sync_config.mode=raiden`, the default. The rollout starts from random
weights and gets the trainer's through Raiden, so `none`, intended for smoke
tests, also needs `vllm_config.init_with_random_weights=false`. Keep
`env_kwargs.max_steps` equal to `agentic_grpo_config.max_turns`.

`agentic_grpo_config.beta` stays zero: a nonzero KL coefficient needs a
reference inference worker, which needs chips of its own
(`cluster_config.local.inference_devices`). FrozenLake is deterministic by
default, matching the reference recipe's environment construction; set
`env_kwargs.is_slippery=true` to enable Gymnasium's slippery transitions.

## Gemma4 E2B

`../recipes/benchmark_frozenlake_gemma4_e2b.sh` runs the Gemma4 E2B recipe of
`examples/frozenlake/train_frozenlake.py` through `launcher.sh`. On a 4-chip
host it assigns chips 0-1 to a two-way FSDP trainer and chips 2-3 to a two-way
TP rollout worker:

```bash
bash tunix/dist_rl/examples/recipes/benchmark_frozenlake_gemma4_e2b.sh
```

The recipe selects `google/gemma-4-E2B-it` (also as the tokenizer),
two-trajectory trainer/logp micro-batches, `compute_logps_chunk_size=2048`, and
`max_concurrency=512`, and writes its artifacts to
`artifacts/gemma4_e2b_dist_frozenlake`. The rest comes from `frozenlake.yaml`:
fp32 compute and parameters, decoder rematerialization, flash attention with
block size 256, the Gemma4 non-thinking chat template, and the vLLM limits
(`max_num_seqs=512`, `max_num_batched_tokens=32768`). Later
`section.key=value` arguments override any of them.

## Distributed vs. agentic benchmark

`../../benchmarks/rl_efficiency/` runs this recipe on the distributed stack
and the matching `tunix.cli.grpo_main` config on the agentic stack, and
compares their recorded timings; see its `README.md`.
