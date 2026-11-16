# SW3 Recipe: R²Orch Inner-Loop (Trainable Graph Director + Frozen Workers)

Implements milestone **SW3 / Gate G4** (the inner orchestration loop of
[R²Orch](https://github.com/beita6969/R2Orch), arXiv:2610.22740) on top of
`tunix.dist_rl.swarm`:

- **Trainable Graph Director** (`policy_id="qwen3_4b_director"`, `trainable=True`):
  iteratively constructs and edits a layered DAG workflow of Workers via single
  JSON canvas actions (`add_agent`, `set_prompt`, `set_layer`, `add_edge`,
  `delete_agent`, `run_graph`, `finish`), optimizing
  `u = correctness - 0.005 * worker_cost` with DAPO asymmetric clipping
  (`clip = (0.2, 0.28)`) and `sampler_is="token"`.
- **Frozen Workers** (`policy_id="qwen3_4b_frozen_worker"`, `trainable=False`):
  served on an isolated TPU sub-mesh (`worker_mesh`) from the untouched
  `Qwen/Qwen3-4B` checkpoint (`T = 0.3`). Ready nodes in the same layer execute
  concurrently via `asyncio.gather`. Worker calls remain in `SwarmEpisode.calls`
  with `loss_mask = 0` for audit and token-cost accounting, emit zero trainer
  payloads, and are verified at every round to have exact zero weight drift
  (`frozen_worker_max_norm_drift == 0.0`).
- **4 Self-Contained Reasoning & Coding Domains + 2 OOD Suites**:
  pinned SHA-256 splits from `beita6969/R2Orch@12ceb7563`:
  - `train.jsonl` (`3,658` tasks across `aime`, `hotpotqa`, `medqa`, `mbppplus`,
    preserving `curriculum` metadata + `seed_cards.json` for Stage 2 Proposer
    and Stage 3 Skill Withdrawal),
  - `heldout.jsonl` (`126` tasks: `30` AIME + `32`×3),
  - `test_iid.jsonl` (`414` tasks: `30` AIME'26 + `128`×3),
  - `test_ood.jsonl` (`256` tasks: `128` `MATH-Hard` + `128` `MuSiQue`).

## Quick Start

### 1. Prepare Pinned Dataset Splits

```bash
python -m examples.swarm_r2orch.prepare_data --output /tmp/r2orch-data
```

### 2. CPU End-to-End Smoke Verification

```bash
python -m examples.swarm_r2orch.train \
  --backend cpu \
  --config examples/swarm_r2orch/configs/cpu_smoke.json \
  --output /tmp/r2orch-cpu-smoke
```

### 3. TPU v5p Qwen3-4B Training & Evaluation

```bash
python -m examples.swarm_r2orch.train \
  --backend qwen \
  --config examples/swarm_r2orch/configs/qwen3_4b.json \
  --train-data /tmp/r2orch-data/train.jsonl \
  --eval-data /tmp/r2orch-data/heldout.jsonl \
  --test-iid-data /tmp/r2orch-data/test_iid.jsonl \
  --test-ood-data /tmp/r2orch-data/test_ood.jsonl \
  --model-path /path/to/Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --output /tmp/r2orch-run
```

## Verified TPU v5p Qwen3-4B Results (100 Steps, Frozen Workers)

Validated on a 32-chip TPU `v5p` slice (`16` fp32 trainer chips + `8` trainable
Director vLLM chips + `8` frozen Worker vLLM chips; `frozen_worker_max_norm_drift == 0.0`
across all 100 steps, `director_norm_delta_l2 = 1.04e-3`):

| Evaluation Suite | Step 0 (`avg@2 / pass@2`) | Step 60 (`avg@2 / pass@2`) | Step 100 (`avg@2 / pass@2`) | Gain (`Step 100 − Step 0`) | Worker Cost (`Step 0 → 100`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Heldout Macro (`126`)** | `53.96 / 57.55` | **`59.24 / 63.28`** | **`58.39 / 64.06`** | **`+4.43 / +6.51` pp** | `0.174 → 0.151` (`-13.2%`) |
| **Locked Test IID Macro (`414`)** | `44.30 / 47.47` | — | **`47.03 / 48.79`** | **`+2.73 / +1.32` pp** | `0.200 → 0.155` (`-22.5%`) |
| **Locked Test OOD Macro (`256`)** | `42.77 / 48.05` | — | **`44.53 / 50.78`** | **`+1.76 / +2.73` pp** | `0.219 → 0.225` (`multi_agent 4.7%`) |
| — OOD `MATH-Hard` (`128`) | `54.30 / 59.38` | — | **`60.94 / 69.53`** | **`+6.64 / +10.16` pp** | `0.067 → 0.082` |
