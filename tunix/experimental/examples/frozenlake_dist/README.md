# Distributed FrozenLake GRPO Recipe

This directory ports `examples/frozenlake/train_frozenlake_qwen3.py` to the
experimental distributed RL stack. The control plane runs on CPU, while the
generic distributed trainer and rollout workers own model execution.

The distributed module keeps only its in-memory dataset and request wiring. It
directly reuses `examples/frozenlake/agent.py`, `examples/frozenlake/env.py`,
and the dataset parameter generator in `examples/frozenlake/data.py`, avoiding
a second copy of the recipe behavior.

The defaults preserve the reference Qwen3 recipe: Qwen3-8B, 64 prompt groups
per full step, 64 prompt groups per optimizer update, 8 generations, 8 turns,
GSPO-token loss, RLOO advantages, asymmetric clipping (`0.003`/`0.005`),
`sequence-mean-token-mean` aggregation, `low_var_kl`, temperature `0.7`, AdamW
(`1e-6`, `b1=0.9`, `b2=0.95`, no weight decay), and gradient clipping at 100.
It also matches the reference's token-level truncated sampler importance
sampling (threshold 2), fp32 actor parameter storage with bf16 compute,
decoder rematerialization, and flash attention. `TRAIN_MICRO_BATCH_SIZE`
counts trajectories: its default is `4 * NUM_GENERATIONS = 32`, matching the
agentic recipe's four prompt groups. Log-prob recomputation scores each of
these micro-batches directly, so the default optimizer update accumulates
16 micro-batches of 32 trajectories. Rollout, loss, and sampler-IS log-prob
recomputation all use the configured temperature (default `0.7`).

Training data is generated with seed 42, shuffled with NumPy PCG64 in exactly
the order used by Hugging Face `Dataset.shuffle(seed=42)`, truncated to
`150 * 64 = 9600` examples, and repeated for three epochs. This gives the same
450 optimizer steps as the agentic recipe without materializing Parquet or
Grain datasets in the distributed control plane.

Install the FrozenLake and distributed extras, then launch from a 4-chip TPU
host:

```bash
pip install -e '.[frozenlake,experimental]'
cd tunix/experimental/examples/frozenlake_dist
./launcher.sh
```

For a small infrastructure smoke test, use a smaller supported model and turn
off weight synchronization:

```bash
MODEL_NAME=Qwen3-1.7B MODEL_ID=Qwen/Qwen3-1.7B \
BATCH_SIZE=1 MINI_BATCH_SIZE=1 NUM_GENERATIONS=2 TRAIN_MICRO_BATCH_SIZE=2 \
MAX_STEPS=1 MAX_TURNS=3 WEIGHT_SYNC_MODE=none ./launcher.sh
```

`BATCH_SIZE` is the full/global batch and determines checkpointing, global
step advancement, and weight-sync cadence. `MINI_BATCH_SIZE` determines each
optimizer update, so one full step performs
`BATCH_SIZE / MINI_BATCH_SIZE` updates. Multi-step training uses
`WEIGHT_SYNC_MODE=raiden` by default; `none` is intended for smoke tests, and
the launcher rejects the protocol-only `fallback` mode through the
orchestrator validation. When synchronization is disabled, the rollout worker
loads checkpoint weights instead of starting from random weights.

The launcher starts one actor and one rollout worker, so `BETA` must remain
zero. A nonzero KL coefficient requires adding a reference inference worker.
FrozenLake is deterministic by default, matching the reference recipe's
environment construction; set `IS_SLIPPERY=1` to enable Gymnasium's slippery
transitions.

The distributed topology uses separate trainer and rollout devices, so exact
bit-for-bit equality with the single-mesh agentic run is not expected: vLLM
sampling and parallel floating-point reductions can differ. The distributed
path also still needs the agentic recipe's exact token continuity and explicit
completion attention masks, as well as its held-out evaluation loop. These
remaining differences prevent claiming full recipe or training-result parity;
no end-to-end TPU comparison has been run yet.
