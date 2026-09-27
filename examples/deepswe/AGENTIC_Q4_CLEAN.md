# Qwen3-4B clean-data agentic GRPO

`run_deepswe_agentic_q4_clean.sh` runs the same Qwen3-4B clean task recipe as
`tunix/experimental/examples/deepswe_dist/launcher.sh` through the in-process
agentic GRPO learner. It does not resume a distributed trainer checkpoint;
the two training engines have different checkpoint formats.

The data module calls the distributed recipe's loader. Both paths pin the
`R2E-Gym/R2E-Gym-Subset` train revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5`, join the canonical
`p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl` selector to
the full task rows, verify its SHA-256 and 1,012 unique images, and shuffle
with seed 42. Two passes provide the 1,600 prompt groups for 200 updates.

| Setting | Agentic Q4 clean recipe |
| --- | --- |
| Model | `Qwen/Qwen3-4B-Instruct-2507`, full parameter bfloat16 |
| Batch | 8 tasks × 16 generations = 128 trajectories/update |
| Context | 4,096 prompt + 16,384 response = 20,480 tokens |
| Trainer | 2 trajectories per microbatch, 64 gradient accumulations/update, 1×2 TPU mesh |
| Logprobs | 1 trajectory per forward pass |
| Rollout | vLLM, 1×2 TPU mesh, 32 maximum sequences, seed 42, temperature 1, top-p 1, top-k 0, prefix caching off |
| GRPO | RLOO, `sequence-mean-token-scale`, beta 0, epsilon 0.2/0.28, rollout logprobs |
| Optimizer | AdamW, LR 1e-6, b1 0.9, b2 0.99, decay 0.01, global norm clip 1 |
| Environment | Docker R2E-Gym, Q4 XML action compatibility, 50 turns, episode 4,800 s, step/reward 1,800 s |

Run on an idle four-chip host (the distributed and agentic jobs cannot use the
same chips simultaneously). On this host, the launcher defaults to the
existing 4B model and Python environment and stores outputs under
`/mnt/disks/persist/haoyu-deepswe-agentic-q4-clean`:

```bash
cd /home/haoyugao_google_com/tunix
./examples/deepswe/run_deepswe_agentic_q4_clean.sh
```

`PYTHON_BIN`, `MODEL_DIR`, `ARTIFACT_ROOT`, `DOCKER_HOST`, `GOLD_WHITELIST`,
`DATASET_CACHE`, `WANDB_PROJECT`, and `WANDB_RUN_NAME` can be overridden. The
launcher writes checkpoints every update and keeps the latest
two. The older `run_deepswe_disagg_v5p_32.sh` remains the separate 32B recipe.
The agentic learner dispatches rollouts one prompt at a time; its rollout
microbatch is therefore 1, while vLLM can process up to 32 sequences at once.

On this host the `deepswe-agentic-q4-clean.service` unit starts after the
persistent disk is mounted. Its output is appended to
`/mnt/disks/persist/haoyu-deepswe-agentic-q4-clean/agentic-training.log`.
The exact-token collector counts chat-template end tokens and stops before an
environment message would exceed the 16,384-token completion budget.
