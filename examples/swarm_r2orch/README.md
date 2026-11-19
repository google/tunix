# R²Orch Recipe: Trainable Graph Director, Frozen Workers & Two-Timescale Self-Play

Implements both **Milestone SW3 / Gate G4** (inner-loop Graph Director + frozen Workers) and **Full `R²Orch`** (`Recursive Self-Play Orchestration`, [beita6969/R2Orch@12ceb756](https://github.com/beita6969/R2Orch/tree/12ceb7563aa50439687f21ba5da60ad74a75eb89), arXiv:2610.22740) on top of `tunix.dist_rl.swarm`:

- **Trainable Graph Director** (`policy_id="qwen3_4b_director"`, `trainable=True`):
  iteratively constructs and edits a layered DAG + same-layer peer cross-check workflow of Workers via single JSON canvas actions (`add_agent`, `set_prompt`, `set_layer`, `add_edge`, `consider_relation`, `delete_agent`, `run_graph`, `finish`), optimizing `u = correctness - 0.005 * worker_cost` with asymmetric dual-clip GRPO (`clip = (0.2, 0.28)`) and `sampler_is="token"`.
- **Frozen Workers & PATS Reviewer** (`policy_id="qwen3_4b_frozen_worker"`, `trainable=False`):
  served on an isolated TPU sub-mesh (`worker_mesh`) from the untouched `Qwen/Qwen3-4B` checkpoint (`T = 0.3`). Ready nodes in the same layer execute concurrently via `asyncio.gather` (including bidirectional `consider_relation` peer review waves). Worker calls remain in `SwarmEpisode.calls` with `loss_mask = 0` for audit and token-cost accounting, emit zero trainer payloads, and are verified at every round to have exact zero weight drift (`frozen_worker_max_norm_drift == 0.0`).
- **Full Two-Timescale Self-Play (`enable_proposer` + `enable_skill_ledger`):**
  - **Learned Pairwise Proposer (`proposer.py`):** `ADSBoundaryScheduler` + `TSDSRetriever` (`128`-dim pool embeddings) build `8` candidate lists (`A..H`: `4` ADS + `4` TSDS) per cycle; `ProposerRuntime` samples `2` distinct picks per list (`16` tasks × `5` = `80` episodes/cycle), computes Recursive Learnability `arms_bell_reward` (`gamma=0.5`, endpoint penalty `-0.2`, Beta shrinkage `m=2.0`, `(T_d / T_q)^0.5`), and executes a fused `PeftTrainer.train_step` pairwise policy-gradient + reference-KL (`beta=1.0`, `gap_clip=4.0`) update directly on `train_mesh` (`skip_lm_head=True`).
  - **Bayesian Skill Withdrawal & PATS (`skill_ledger.py`, `pats.py`):** `SkillScaffold` loads the 12 domain seed cards (`configs/seed_cards.json`), selects a candidate card `k` via Thompson sampling (or audits a retired card with `audit_share=0.30`), runs `n_h = 3` `hidden` and `n_v = 2` `visible` episodes per task with condition-grouped `(task_id, arm)` GRPO advantages, withdraws internalized skills via `2,000`-draw Rubin Dirichlet bootstrap (`n >= 12, E[Delta+] <= 0.02`), admits positive `trial` cards (`q_0.05(Delta) > 0`), and invokes `PatsOuterReviewer` (`REVISE` + `DISTILL` on scrubbed public traces, zero gold-answer leakage) every `5` cycles.
  - **Strict Zero-Skill Deployment:** All `Heldout`, `Locked Test IID`, and `Locked Test OOD` evaluations run with `skill_context=""`.
- **4 Self-Contained Reasoning & Coding Domains + 2 OOD Suites**:
  pinned SHA-256 splits from `beita6969/R2Orch@12ceb7563`:
  - `train.jsonl` (`3,658` tasks across `aime`, `hotpotqa`, `medqa`, `mbppplus`, preserving `curriculum` 128-dim embeddings + `task_type` + `source_id`),
  - `heldout.jsonl` (`126` tasks: `30` AIME + `32`×3),
  - `test_iid.jsonl` (`414` tasks: `30` AIME'26 + `128`×3),
  - `test_ood.jsonl` (`256` tasks: `128` `MATH-Hard` + `128` `MuSiQue`).

## Quick Start

### 1. Prepare Pinned Dataset Splits

```bash
python -m examples.swarm_r2orch.prepare_data --output /tmp/r2orch-data
```

### 2. CPU End-to-End Smoke Verification (SW3 & Full R²Orch)

```bash
# SW3 inner-loop smoke
python -m examples.swarm_r2orch.train \
  --backend cpu \
  --config examples/swarm_r2orch/configs/cpu_smoke.json \
  --output /tmp/r2orch-cpu-sw3

# Full two-timescale R²Orch smoke (Proposer + SkillScaffold + PATS)
python -m examples.swarm_r2orch.train \
  --backend cpu \
  --config examples/swarm_r2orch/configs/cpu_full_smoke.json \
  --output /tmp/r2orch-cpu-full
```

### 3. TPU v5p Qwen3-4B Training & Zero-Skill Evaluation

```bash
# Full R²Orch (4x4x8 = 128 TPU v5p chips: 64 train_mesh + 32 director_mesh + 32 worker_mesh)
python -m examples.swarm_r2orch.train \
  --backend qwen \
  --config examples/swarm_r2orch/configs/qwen3_4b_full.json \
  --train-data /tmp/r2orch-data/train.jsonl \
  --eval-data /tmp/r2orch-data/heldout.jsonl \
  --test-iid-data /tmp/r2orch-data/test_iid.jsonl \
  --test-ood-data /tmp/r2orch-data/test_ood.jsonl \
  --model-path /path/to/Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --checkpoint-dir /tmp/r2orch-checkpoints \
  --output /tmp/r2orch-full-run
```

## Verified TPU v5p Qwen3-4B Results (Zero-Skill Deployment, Frozen Workers)

Both **SW3** (`hy-swarm-r2o-run1`, 32-chip `v5p`, `32 × 4 = 12,800` episodes) and **Full `R²Orch`** (`hy-swarm-r2o-full1`, 128-chip `4x4x8` `v5p`, `16 × 5 = 8,000` episodes; `director_norm_delta_l2 = 1.44e-3`, `proposer_norm_delta_l2 = 4.28e-3`, `19` Bayesian skill withdrawals / `11` audit reinstatements) completed 100 cycles with **`frozen_worker_max_norm_drift == 0.0`** and strict zero-skill evaluation (`skill_context=""`):

| Evaluation Suite (`k=2`, `T=0.3, top_p=0.95`) | Step 0 Base (`ev0`) | `SW3` Inner-Loop (100 Steps) | **Full `R²Orch` (100 Cycles)** | Full `R²Orch` Peak / Highlights |
| :--- | :---: | :---: | :---: | :--- |
| **Heldout Macro (`126`)** | `53.96 / 57.55` | **`58.39 / 64.06`** *(Step 60: `59.24 / 63.28`)* | **`55.08 / 57.50`** *(Step 20: `53.96 / 61.61`, Step 40: `55.99 / 58.39`)* | `100.0%` finish rate, `-16.1%` cost; Step 60 `HotpotQA` **`70.31 / 71.88`** |
| **Locked Test IID Macro (`414`)** | `44.30 / 47.47` | **`47.03`** / `48.79` | `45.81` / **`48.98`** | **Best `pass@2` (`48.98`)**, **best `MBPP+` (`72.27 / 73.44`)**, **best `MedQA pass@2` (`71.88`)**, `100.0%` finish, `-22.5%` cost |
| **Locked Test OOD Macro (`256`)** | `42.77 / 48.05` | **`44.53 / 50.78`** | `43.16 / 46.88` | **`MATH-Hard`: `60.55 / 66.41` (`+6.25 / +7.03` pp over Base)**, `100.0%` finish, lowest cost (`0.205`) |
