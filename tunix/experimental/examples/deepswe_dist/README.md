# Distributed DeepSWE with Kubernetes Agent Sandboxes

This package implements distributed reinforcement learning (GRPO) for
**DeepSWE** using Tunix's distributed runtime and Kubernetes-native agent
sandboxes (`agent-sandbox-rl` / `SandboxFleet`).

---

## 1. Architecture Overview

- **`run_deepswe_dist_grpo.py` (Orchestrator)**:
  - Initializes the global `SandboxFleet` targeting the `rl-tunix-swebench` namespace.
  - Prewarms and pipelines sandbox instances across tasks from the Hugging Face `R2E-Gym/R2E-Gym-V1` dataset.
  - Connects to distributed workers via the Tunix discovery service.
  - Contains embedded mock workers (`MockRolloutWorker`, `MockTrainerWorker`) when running with `--test_mode`.

- **`deepswe_rl_program.py` (RL Program)**:
  - Orchestrates multi-turn episode interactions for software engineering tasks.
  - Dynamically claims a pre-warmed sandbox for each task (`self.env = SWEEnv(example, max_steps=self.max_turns)`).
  - Calls `await self.engine.generate([chat_history])` over gRPC to obtain actions from the rollout sampler.
  - Executes tool actions (such as `execute_bash`, `file_editor`, `finish`) in the remote sandbox container.
  - Releases sandbox resources cleanly via `try ... finally: self.env.close()`.
  - Submits assembled trajectory payloads to the distributed trainer via `await self.engine.train_step(...)`.

- **`k8s_launcher.sh` (Launcher)**:
  - Deploys and manages the distributed components as Kubernetes JobSets.
  - Provides a single-command `test_orchestrator` target to reproduce end-to-end multi-turn tests without requiring TPU hardware.

---

## 2. Prerequisites

1. **Kubernetes Cluster**:
   - Access to the GKE cluster with node pool labeled:
     - `cloud.google.com/gke-nodepool: deepswe-cpu-pool`
   - Namespace: `rl-tunix-swebench`

2. **Docker Image**:
   - Standard Tunix image with `agent-sandbox-rl`, `swebench`, and `r2egym`
     installed:
     ```
     gcr.io/cloud-tpu-multipod-dev/wuhao_tunix_deepswe:latest
     ```

---

## 3. How to Trigger the End-to-End Test

### Option A: Using `k8s_launcher.sh`
```bash
bash experimental/examples/deepswe_dist/k8s_launcher.sh \
  --command=test_orchestrator \
  --image=gcr.io/cloud-tpu-multipod-dev/wuhao_tunix_deepswe:latest
```

### Option B: Direct JobSet Submission
```bash
kubectl apply -f experimental/users/wuhao/deepswe/deepswe_test_mode_wuhao.yaml
```

To monitor progress:
```bash
# Watch pod status
kubectl get pods -l jobset.sigs.k8s.io/jobset-name=nt-ds-35b-v5p-256

# Follow orchestrator logs
kubectl logs -l jobset.sigs.k8s.io/jobset-name=nt-ds-35b-v5p-256 -c jax-tpu -f
```

---

## 4. Expected Execution & Log Output

During execution, each step provisions a unique task from the dataset,
claims a pre-warmed sandbox, and runs 5 interaction turns:

```text
INFO:root:Creating SandboxClaim 'sandbox-claim-b63ff2f2' in namespace 'rl-tunix-swebench' using warm pool 'pool-r2e-img-8686d06e2329'...
INFO:root:Resolved sandbox name 'pool-r2e-img-8686d06e2329-g5j6s' from claim status (claim Ready)

INFO:root:Turn 0/5 calling remote Rollout...
INFO:absl:Generating rollouts for 1 prompt(s)/request(s) across 1 worker(s)...
INFO:absl:Completed synchronous generation of 1 trajectory item(s).
INFO:root:Turn 0/5 Rollout completed. Action: <function=execute_bash><parameter=cmd>pwd</parameter></function> Reward: 0

INFO:root:Turn 1/5 calling remote Rollout...
INFO:absl:Generating rollouts for 1 prompt(s)/request(s) across 1 worker(s)...
INFO:absl:Completed synchronous generation of 1 trajectory item(s).
INFO:root:Turn 1/5 Rollout completed. Action: <function=execute_bash><parameter=cmd>pwd</parameter></function> Reward: 0

...

INFO:root:Turn 4/5 Rollout completed. Action: <function=execute_bash><parameter=cmd>pwd</parameter></function> Reward: 0
INFO:absl:Executing train_step on actor worker (accumulate_gradients=False, apply_optimizer=True)...
INFO:root:Trainer processed trajectory of length 1515 with reward 0.0
INFO:root:Step 0 finished. Reward: 0.0
```

Upon completing all configured steps (e.g. 100 steps), the process tears
down sandbox pools and exits cleanly:
```text
XPK End: Tue Sep 1 17:29:40 UTC 2026
EXIT_CODE=0
```

---

## 5. Running Local Unit Tests

To run the standalone unit test suite:
```bash
pytest tunix/experimental/examples/deepswe_dist/deepswe_rl_program_test.py
```
