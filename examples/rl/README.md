# Hardware Resource Requirement Needed for RL

To provide more people with reference points for resource selection when dealing
with different models and tasks, this section is mainly dedicated to
introducing the environmental requirements based on experiments we have conducted
on free-tier TPU.

If you happen to have a configuration that has already been tested, we welcome
you to submit a PR and include a screenshot from Wandb or other verifiable
evidence.

Note, CLI scripts linked are listed for reference, the config in the script
does not match the exact configuration listed in the table.
--- 

## RL

| Algo | Model         | Type  | Min Resources | Max Training Micro Batch Size     | Sharding              | Launch Script                 |
| :--- | :------------ | :---- | :------------ | :-------------------------------- | :-------------------- | :---------------------------- |
| GRPO | **Gemma2-2b** | Full  | v5e-4         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | *[run_gemma2_2b.sh](examples/rl/gsm8k/run_gemma2_2b.sh)* |
| GRPO | **Gemma2-2b** | LoRA  | v5e-4         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | *[run_gemma2_2b.sh](examples/rl/gsm8k/run_gemma2_2b.sh)* |
| GRPO | **Gemma 7b**  | Full  | v5p-2         | Num of generation = 4, batch_size = 8 | Train: fsdp Rollout: tp | *[run_gemma_7b.sh](examples/rl/gsm8k/run_gemma_7b.sh)* |
| GRPO | **Gemma 7b**  | LoRA  | v5p-2         | Num of generation = 4, batch_size = 8 | Train: fsdp Rollout: tp | *[run_gemma_7b.sh](examples/rl/gsm8k/run_gemma_7b.sh)* |
| GRPO | **Llama3.1 8b**| Full  | v5p-2         | Num of generation = 4, batch_size = 8 | Train: fsdp Rollout: tp | *[run_llama3.1_8b.sh](examples/rl/gsm8k/run_llama3.1_8b.sh)* |
| GRPO | **Llama3.1 8b**| LoRA  | v5p-2         | Num of generation = 4, batch_size = 8 | Train: fsdp Rollout: tp | *[run_llama3.1_8b.sh](examples/rl/gsm8k/run_llama3.1_8b.sh)* |
| GRPO | **Llama3.2 1b**| Full  | v5e-4         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | *[run_llama3.2_1b.sh](examples/rl/gsm8k/run_llama3.2_1b.sh)* |
| GRPO | **Llama3.2 1b**| LoRA  | v5e-4         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | *[run_llama3.2_1b.sh](examples/rl/gsm8k/run_llama3.2_1b.sh)* |
| GRPO | **Qwen3 0.6b**| Full  | v5e-1         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | |
| GRPO | **Qwen3 0.6b**| LoRA  | v5e-1         | Num of generation = 4, batch_size = 1 | Train: fsdp Rollout: tp | |
| GRPO | **Qwen3 14b** | Full  | v5p-2         | Num of generation = 4, batch_size = 4 | Train: fsdp Rollout: tp | |
| GRPO | **Qwen3 14b** | LoRA  | v5p-2         | Num of generation = 4, batch_size = 4 | Train: fsdp Rollout: tp | |
