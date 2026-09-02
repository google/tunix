# Distributed GSM8K GRPO Training on TPU Pods

This directory contains the distributed implementation of **Group Relative Policy Optimization (GRPO)** for mathematical reasoning on the **GSM8K** dataset using Google Cloud TPU slices on GKE.

The architecture decouples the RL training loop into independent Kubernetes JobSets:
- **CPU Orchestrator**: Coordinates the RL training loop, manages the async trajectory queue, calculates GRPO advantages, evaluates GSM8K reward accuracy, and coordinates policy-versioned weight sync.
- **TPU Trainer Worker**: Runs [MaxText](https://github.com/google/maxtext) on a TPU slice (`tpu7x:2x2x1` or `tpuv5:2x2x2`), computing forward/backward gradient updates over streamed microbatches.
- **TPU Rollout Worker**: Runs high-throughput [vLLM](https://github.com/vllm-project/vllm) on a TPU slice (`tpu7x:2x2x1` or `tpuv5:2x2x1`), generating response trajectories asynchronously for prompt problem batches.
- **Weight Synchronization**: Transfers updated model weights across pods using high-performance [Raiden](https://github.com/google/tunix) RDMA/host-staged tensor streaming.

---

## 1. Quickstart

Use `submit_gsm8k.sh` to submit, monitor, and manage training jobs with a single command.

### Run 100-Step Training for Convergence (Real Weight Sync)
```bash
tunix/experimental/examples/math_gsm8k_dist/submit_gsm8k.sh start \
  --steps 100 \
  --weight-sync raiden
```

### Run a Fast 10-Step Dry Run (NoOp Weight Sync)
To test connectivity, scheduling, and pipeline progression without tensor transfers:
```bash
tunix/experimental/examples/math_gsm8k_dist/submit_gsm8k.sh start \
  --steps 10 \
  --weight-sync noop
```

---

## 2. CLI Tool Reference (`submit_gsm8k.sh`)

`submit_gsm8k.sh` provides an ergonomic, unified CLI for cluster context switching, job submission, status monitoring, and log streaming.

### Subcommands

| Subcommand | Description | Example |
| :--- | :--- | :--- |
| `start` | Launches Orchestrator, Trainer, and Rollout JobSets (default) | `./submit_gsm8k.sh start --steps 100` |
| `status` | Displays live status of JobSets, Pods, and Kueue queue quota | `./submit_gsm8k.sh status` |
| `logs <target>` | View logs for `orch`, `train`, or `roll` (`-f` to follow) | `./submit_gsm8k.sh logs orch -f` |
| `stop` | Terminates and cleans up all active JobSets for the current user | `./submit_gsm8k.sh stop` |
| `restart` | Cleanly stops existing jobs and relaunches them | `./submit_gsm8k.sh restart --steps 100` |
| `watch` | Continuously monitors pod states and Kueue reservations | `./submit_gsm8k.sh watch` |

### Key Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--steps <N>` | `100` | Total GRPO training steps. |
| `--weight-sync <MODE>` | `raiden` | Weight synchronization mode: `raiden` (real weight sync for convergence), `noop` (version advance only for control-plane testing), or `none`. |
| `--profile <NAME>` | `bodaborg` | Preconfigured hardware profile: `bodaborg` (TPU v7x) or `v5p` (TPU v5p). |
| `--cluster <NAME>` | `bodaborg-tpu7x-nap` | Target GKE cluster name. |
| `--project <ID>` | `cloud-tpu-shared-capacity` | GCP Project ID. |
| `--region <REGION>` | `us-central1` | GCP Region. |
| `--branch <BRANCH>` | Current git branch | Git branch to fetch and run inside the containers (`atwigg/gsm8k-dist-fixes`). |
| `--ckpt <GCS_PATH>` | Qwen3-1.7B checkpoint | GCS path to Orbax parameters checkpoint for MaxText. |
| `--batch-size <N>` | `2` | Number of distinct prompt problems sampled per step. |
| `--num-generations <N>`| `2` | Number of rollout completions sampled per prompt (total rollouts/step = `batch_size * num_generations`). |
| `--micro-batch <N>` | `1` | Trainer microbatch size for gradient accumulation. |
| `--wandb-project <NAME>`| `trellis-gsm8k` | Weights & Biases project name. |
| `--wandb-run <NAME>` | `<auto>` | Optional custom WandB run name. |
| `--dry-run` | `false` | Prints resolved configuration and commands without submitting. |

---

## 3. Hardware Profiles

### Profile: `bodaborg` (Default - TPU v7x)
- **Cluster**: `bodaborg-tpu7x-nap` in project `cloud-tpu-shared-capacity` (`us-central1-c`).
- **Orchestrator Machine**: `e2-standard-16` (runs on `default-pool` without taints).
- **Trainer TPU Slice**: `tpu7x:2x2x1` (4 TPU chips, FSDP=4).
- **Rollout TPU Slice**: `tpu7x:2x2x1` (4 TPU chips, TP=4).
- **Kueue LocalQueue**: `default` (borrowing from cohort `tpu-shared-cohort`).

### Profile: `v5p` (TPU v5p)
- **Cluster**: `trellis-demo-0810` / `mlperf-v5p` in project `cloud-tpu-multipod-dev` (`europe-west4`).
- **Orchestrator Machine**: `n2-standard-64`.
- **Trainer TPU Slice**: `tpuv5:2x2x2` (8 TPU chips, FSDP=8).
- **Rollout TPU Slice**: `tpuv5:2x2x1` (4 TPU chips, TP=4).

To use the v5p profile:
```bash
./submit_gsm8k.sh start --profile v5p --steps 100 --weight-sync raiden
```

---

## 4. Monitoring & Verifying Convergence (100 Steps)

### A. Checking Overall Job Status
Run `status` to inspect all components at once:
```bash
./submit_gsm8k.sh status
```
This shows:
1. **JobSets**: Terminal and restart status for orchestrator, trainer, and rollout.
2. **Pods**: Host nodes, pod IPs, readiness gates, and container phases (`ContainerCreating`, `Running`, `Completed`).
3. **Kueue Workloads**: Admission conditions and quota reservations.
4. **TPU Cohort Quota**: Real-time allocation across cluster queues.

### B. Streaming Component Logs
- **Orchestrator**:
  ```bash
  ./submit_gsm8k.sh logs orch -f
  ```
- **Trainer**:
  ```bash
  ./submit_gsm8k.sh logs train -f
  ```
- **Rollout Worker**:
  ```bash
  ./submit_gsm8k.sh logs roll -f
  ```

### C. Convergence Metrics to Expect in Logs
During a healthy 100-step convergence run, you will observe the following progression:

1. **Worker Registration & Discovery**:
   ```
   [Orchestrator] Discovered trainer service (atwigg-r28637-train) at atwigg-r28637-train:20002.
   [Orchestrator] Discovered rollout service (atwigg-r28637-roll) at atwigg-r28637-roll:20001.
   [Orchestrator] Cluster workers ready. Starting StandardRLProgram execution...
   ```

2. **Step Progression & Generation Rewards**:
   ```
   >>> Step 1 starting | Policy Version: 1
   [Rollout] Generating 4 trajectories with policy_version=1...
   [Sampler] Response for prompt_0: correct=True, reward=1.00
   [Sampler] Response for prompt_1: correct=False, reward=0.00
   [train] microbatch fwd_bwd #1 done
   ...
   [train] update -> train_step=1 loss=0.4218
   <<< Step 1 finished | Advanced to Policy Version: 2
   ```

3. **Raiden Weight Synchronization**:
   ```
   [WeightSync] Initiating tensor sync from trainer to rollout...
   [WeightSync] Successfully transferred parameters (bytes=3.4GB, time=0.82s).
   [RolloutWorker] Updated policy_version to 2.
   ```

4. **Tracking Convergence over 100 Steps**:
   - **Step 0–10**: Base Qwen3-1.7B math accuracy begins around 25%–35% on GSM8K questions.
   - **Step 10–50**: GRPO advantage weighting reinforces reasoning steps that lead to correct final numerical answers (`#### <answer>`). Reward mean trends upward from ~0.35 toward ~0.65.
   - **Step 50–100**: Responses stabilize, format compliance reaches >95%, and final mean reward converges toward 0.75–0.85+.
   - If configured with WandB (`--wandb-project`), the `reward/mean`, `reward/std`, `loss`, and `kl_divergence` curves are tracked in real-time.

---

## 5. Architecture Details & Pipeline Flow

```
                     +---------------------------------------+
                     |       CPU Orchestrator Pod            |
                     |  (tunix.experimental.runtime.main)    |
                     +-------------------+-------------------+
                                         |
               +-------------------------+-------------------------+
               | GRPO Trajectories & Loss| Control & Weight Sync   |
               v                                                   v
+-------------------------------+                 +-------------------------------+
|       TPU Rollout Worker      |                 |       TPU Trainer Worker      |
|    (vLLM In-Process Engine)   |<================|     (MaxText Engine / JAX)    |
|       tpu7x:2x2x1 (TP=4)      |  Raiden RDMA/   |      tpu7x:2x2x1 (FSDP=4)     |
|   Generates Rollout Batches   |  Host Staged    |   Forward/Backward Gradients  |
+-------------------------------+  Weight Sync    +-------------------------------+
```

### Policy Versioning and Decoupled Rollout
To prevent training stalls:
1. Rollouts are tagged with the active `policy_version`.
2. When the trainer completes step $N$, the weight sync coordinator updates the rollout worker's weights and increments its `policy_version` via RPC (`set_policy_version`).
3. The queue manager accepts trajectories within the configured `--max_staleness` bound (default: 0 on-policy), preventing stale trajectories from entering gradient calculation.

---

## 6. Common Operations & Troubleshooting

### Stopping and Cleaning Up
Always stop jobs when finished or before switching configurations to release TPU quota:
```bash
./submit_gsm8k.sh stop
```

### Insufficient TPU Quota (`QuotaReserved: False`)
If `status` shows workloads pending with:
`insufficient unused quota for google.com/tpu in flavor tpu7x-flavor`
The shared cohort is temporarily full. Once other users' workloads finish or are stopped, Kueue will automatically admit your jobset without requiring resubmission.

### Checking Initial Image Pulling
Large container images (containing JAX, PyTorch, vLLM, and MaxText) can take 5–8 minutes to pull on newly scaled nodes:
```bash
kubectl describe pod -l jobset.sigs.k8s.io/jobset-name=atwigg-r28637-orch | grep -A 5 Events:
```
Once pulled, containers transition to `Running` immediately.
