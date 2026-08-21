# P46 Qwen3-4B learnable task set

This directory contains the canonical prompt selector for the first
Qwen3-4B-Instruct-2507 DeepSWE training run after the P46 clean-data campaign.

## Training input

```text
p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl
rows=1012
sha256=ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973
```

Every row is one unique `docker_image` task from the reviewed 1,851-task
source whitelist. Qwen3-4B-Instruct-2507 produced exactly 16 valid evaluation
outcomes for each task, with solved count `k` from 1 through 15. These tasks
therefore have within-group reward signal under the measured Q4 policy.

The file is a byte-for-byte promoted copy of:

```text
tasks/p46-deepswe-eval-training-profiles/evidence/p46q4census02/
  p46-campaign.q4_learnable.jsonl
```

The complete campaign evidence reports 29,616 valid trajectories, 1,012
partial tasks, 839 Q4 all-fail tasks and zero all-pass tasks. Training consumes
this 1,012-row task selector; it does not treat the 29,616 evaluation
trajectories as training tasks. The Q4 all-fail rows remain separately
available as Q32 candidates and are not included here.

For the existing P46 `q4-debug` renderer, pass both this path and the recorded
SHA explicitly. The renderer's historical default is still the pre-wash
1,851-row whitelist.
