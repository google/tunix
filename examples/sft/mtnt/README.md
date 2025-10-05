# Hardware Resource Requirement Needed for SFT

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

## SFT

| Model         | Type  | Min Resources | Max Batch Size (seq_length=256) | Sharding      | Launch Script                 |
| :------------ | :---- | :------------ | :------------------------------ | :------------ | :---------------------------- |
| **Gemma 2b**  | Full  | v5e-2         | 4                               | N/A           | *[run_gemma_2b.sh](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma_2b.sh)* |
| **Gemma 2b** | LoRA  | v5e-1         | 4                               | N/A           | *[run_gemma_2b.sh](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma_2b.sh)* |
| **Gemma 2b**  | QLoRA | v5e-1         | 4                               | N/A           | *[run_gemma_2b.sh](https://github.com/google/tunix/blob/main/examples/sft/mtnt/run_gemma_2b.sh)* |
| **Gemma 7b**  | Full  | v5p-8         | 128                             | fsdp          | |
| **Gemma 7b** | LoRA  | v5e-4         | 4                               | fsdp           | |
| **Gemma 7b**  | QLoRA | v5e-4         | 4                               | fsdp          | |
| **Llama3.2 1b**| Full  | v5e-2         | 4                               | N/A           | |
| **Llama3.2 1b** | LoRA  | v5e-1         | 4                               | N/A           |  |
| **Llama3.2 1b**  | QLoRA | v5e-1         | 4                               | N/A           | |
| **Llama3.1 8b**| Full  | v5p-8         | 128                             | fsdp          | |
| **Llama3.1 8b** | LoRA  | v5e-4         | 4                               | fsdp          | |
| **Llama3.1 8b** | QLoRA | v5e-4         | 4                               | fsdp          | |
| **Qwen3 0.6b**| Full  | v2-2          | 4                               | fsdp or tp    |  |
| **Qwen3 0.6b** | LoRA  | v2-1          | 4                               | N/A           | |
| **Qwen3 0.6b** | QLoRA | v2-1          | 4                               | N/A           | |
| **Qwen3 14b** | Full  | v5p-8         | 64                              | tp            | |
| **Qwen3 14b** | LoRA  | v5e-8         | 8                               | tp            | |
| **Qwen3 14b** | QLoRA | v5p-8         | 64                              | tp            | |




