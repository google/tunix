# Distributed FrozenLake GRPO Recipe

This directory ports `examples/frozenlake/train_frozenlake_qwen3.py` to the
experimental distributed RL stack. The control plane runs on CPU, while the
generic distributed trainer and rollout workers own model execution.

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

## Local 4-chip TPU VM smoke test

The following command splits a single 4-chip TPU VM into two chips for the
trainer and two chips for rollout. It uses Qwen3-1.7B and runs one full step
without weight synchronization to validate the local distributed
infrastructure:

```bash
cd tunix/experimental/examples/frozenlake_dist
MODEL_NAME=Qwen3-1.7B MODEL_ID=Qwen/Qwen3-1.7B \
TRAINER_TPU_CHIPS=0,1 TRAINER_FSDP=1 TRAINER_TP=2 \
ROLLOUT_TPU_CHIPS=2,3 ROLLOUT_FSDP=1 ROLLOUT_TP=2 \
TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 \
BATCH_SIZE=1 MINI_BATCH_SIZE=1 NUM_GENERATIONS=2 \
TRAIN_MICRO_BATCH_SIZE=1 DATASET_SIZE=32 \
MAX_STEPS=1 MAX_TURNS=3 \
MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=1024 \
BETA=0.0 WEIGHT_SYNC_MODE=none ./launcher.sh
```

The launcher downloads the model when `MODEL_DIR` does not already contain
safetensors. Logs are written to `trainer.log`, `rollout.log`, and
`orchestrator.log` in this directory. Use `WEIGHT_SYNC_MODE=raiden` for a
multi-step training test.

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

One reference-only feature is not modeled separately: the original recipe's
`sampler_is` threshold is folded into the distributed path's rollout-logprob
importance ratio because Orchestrator V2 currently exposes
`use_rollout_logps`, but not an independent sampler-IS threshold.
