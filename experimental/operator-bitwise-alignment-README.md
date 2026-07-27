# Operator Bitwise Alignment — Probe README (for the remote/TPU agent)

**Branch**: `yuxzhang/logp-diff-probe` (forked from `deepswe-quality-fix`).
**You are picking up a CPU-prepared, TPU-pending experiment.** Everything that can be proven
without a TPU is proven; your job is to wire the `_REMOTE_VERIFY_` hooks, run on TPU, and report back.
Date: 2026-07-23.

## ⚠️⚠️ Heads-up 2026-07-26 — probe is now DISAGGREGATED (read this first)
The probe was reworked from a single 8-chip mesh to the **real two-mesh disaggregated
topology** (author's call: "faithfully simulate training, no other cluster, run on 256").
Why: deepswe RL runs vLLM on `rollout_mesh` (fsdp8×tp8, chips 0-63) and tunix on `train_mesh`
(fsdp8×tp8, chips 64-127) — PHYSICALLY DISJOINT (`train_deepswe_nb.py:648-654,894-897`). The
real diff RL pays (A-vs-C = 0.92) is kernel+mesh+decode STACKED; a single-mesh probe can't see
the mesh term. Now:
- `build_meshes()` builds the two disjoint meshes (mirrors deepswe); vLLM→rollout_mesh, tunix→train_mesh.
- One run measures: **A-vs-C** (real total, reproduce 0.92), **A-vs-B** (decode, clean),
  **B-vs-C** (kernel+mesh), **additivity guard** (per-token A-C≡(A-B)+(B-C)), **C-vs-C2** (`--mesh_sensitivity`,
  tunix fsdp-sharding sensitivity, bounds the mesh term).
- yaml is now `instance_type=tpuv5:4x8x8` (256), `completions/parallelism=64`, args
  `--rollout_mesh_{tp,fsdp}=8 --train_mesh_{tp,fsdp}=8 --mesh_sensitivity`.
- **#1 REMOTE RISK: does vLLM confine to rollout_mesh (64 chips) or grab all 256?** If it grabs
  all, it collides with tunix on train_mesh. deepswe avoids this via rl_cluster `role_to_mesh`;
  standalone we pass `mesh=rollout_mesh` — VERIFY confinement first, else reuse role_to_mesh scaffolding.
- Each source's logp is now evaluated ONCE (cached) — the old `compare()` called vLLM/tunix up to
  3× per pair. Decomposition is in `decompose()`. Full design: `tasks/logprob_diff_operator_alignment/phase3b.md`.

## ⚠️ Heads-up (updates after the initial push — read first)
- **History was rewritten** (Claude attribution trailers removed). Do **`git fetch && git reset --hard
  origin/yuxzhang/logp-diff-probe`** — do NOT `git pull` (it would diverge). Content is unchanged +
  the two fixes below on top; rebase any local unpushed work onto the new tip.
- **Source A is now token-in** (`run_vllm` uses `TokensPrompt(prompt_ids)`, not `decode(prompt_ids)`→
  `input_strings`). Reason: the text path re-tokenizes and the BPE round-trip is not identity, which
  would shift A's prompt vs C's and contaminate A-vs-C. Mirrors source B. **Not CPU-runnable — verify
  the token-in generate + logprob extraction on the real engine.**
- **CPU wiring test passes a local `--out`**: the GCS report-write (`epath.Path(gs://…)`) needs `gcsfs`,
  which is TPU/cluster-only; on CPU pass a local `--out` (the wiring test does this). On the cluster,
  `--out gs://…` writes the report.

## Goal
Eliminate the rollout-vs-train per-token **log-prob diff** on **Qwen3-32B** down to **bitwise**
(operator / kernel alignment). This is the **Training-Inference Mismatch (TIM)** residual
(deepswe measured: mean 0.002 / max 0.92 / pearson 0.998, APC off). **Ultimate goal = 0 logp diff**
by aligning the culprit kernel (batch-invariant, à la Thinking Machines). TIS stays as a safety net,
NOT the solution. (Root cause class: batch-size-dependent reduction order in **RMSNorm / matmul / attention**.)

## What's in this branch
```
scripts/logp_diff/
  logp_diff_probe.py     # THE probe: 3-source A/B/C on Qwen3-32B; disaggregated (256) + --colocated (1-host)
  logp-diff-probe.yaml   # Step 2 launch JobSet (256-chip disaggregated)
  patch_qwen3.py         # tpu_inference RoPE hot-patch as a file (run inside the container; A/B need it)
  harness.py             # framework-agnostic logp-diff + per-op attribution (CPU-proven)
  sources.py             # Source abstraction + compare()
  toy_qwen3.py           # numpy toy qwen3 (mirrors real structure; harness validation only)
  logp_diff_test.py      # CPU gate: 6 known-answer tests (attribution correctness) — ALL PASS
  probe_wiring_test.py   # CPU gate: probe main() with tunix/vLLM mocked, disaggregated + colocated — ALL PASS
experimental/
  probe_v5p_1host_docker.sh   # Step 1 launcher: 1-host 4-chip COLOCATED kernel probe (direct, no Pathways)
```

## The experiment: three sources, one run
Flow = **generate once → A(decode logp); then B(prefill) + C(forward) on the SAME generated tokens.**

| Source | What it is | = which real path |
|---|---|---|
| **A** vllm-decode | per-token logp vLLM returned while sampling each generated token | training's `old_logprobs` (agentic_grpo_learner.py:449) |
| **B** vllm-forward | same tokens re-run through vLLM **prefill** (`prompt_logprobs`) | kernel-isolation diagnostic |
| **C** tunix-forward | `common.compute_per_token_logps(graphdef, state, …)` | the trainer path (`rl_cluster.get_actor_per_token_logps`, rl_cluster.py:1122) |

**Decompose the diff:**
- `A-vs-C` = the REAL diff RL experiences (ground truth; reproduce the deepswe residual).
- `B-vs-C` = **pure kernel** — the FIX TARGET (align → bitwise). **Attack this first.**
- `A-vs-B` = decode-vs-prefill effect (vLLM-internal; inherent to autoregressive decode).
- Roughly: `A-vs-C ≈ (A-vs-B decode) + (B-vs-C kernel)`.

**Attribution (which op-type):** per-layer activation diff (first divergence = culprit op) +
standalone isolation for RMSNorm/matmul. Attention: compare **post-softmax/post-o_proj output**
(pre-softmax logit shifts are shift-invariant → spurious).

## What YOU must wire (grep `_REMOTE_VERIFY_` in logp_diff_probe.py — 9 marks)
1. **[CRITICAL] vLLM must use REAL weights, not dummy.** deepswe's vLLM is `init_with_random_weights=True`
   and gets real weights **synced from the actor by the training loop**. This standalone probe has NO
   training loop → you MUST either load real safetensors into vLLM (`init_with_random_weights=False`) or
   call `sampler.update_params(state)` to sync from the tunix model. **Otherwise A/B run on random weights
   and every diff is meaningless.**
2. `run_vllm`: VllmSampler.generate(prompt, return_logprobs=True) → `(full_tokens, decode_logp)` = source A.
3. `vllm_prefill_logp`: `sampling_params.prompt_logprobs=1` (vllm_sampler.py:470) → per-token logp;
   **slice to the last `n_completion`** to align with A/C; verify numerics (`b/428730696`).
4. `tunix_forward_logp` activation hooks (optional): return ordered `[(name, array)]` matching Phase-2
   naming for per-layer attribution; if not hookable, leave `get_acts=None` (attribution falls back to
   logp-only + standalone op isolation).
5. **[#1 REMOTE RISK] Disaggregated meshes — vLLM confinement.** `build_meshes()` builds two
   DISJOINT meshes (rollout devices[:64], train devices[64:128]). vLLM gets `VllmConfig.mesh=rollout_mesh`
   → `device_indexes = mesh.device_ids` (vllm_sampler.py:295) confines it to chips 0-63; tunix loads on
   train_mesh (chips 64-127). **VERIFY both engines coexist and vLLM does NOT grab all 256** — if it does,
   reuse deepswe's `rl_cluster` `role_to_mesh` scaffolding instead of hand-rolling.

## Pre-checks before trusting any diff (go/no-go)
- **Weight equality**: A/B(vLLM) and C(tunix) bitwise-equal weights (see #1). vLLM uses
  `init_with_random_weights=False` (direct load) since the probe has no training loop to sync. Verify first.
- **Additivity residual ~0**: `report.decomposition.additivity_residual_max` must be ~0; a nonzero value
  means A/B/C are not on the same completion span (alignment bug) — fix before reading any diff.
- **Temperature**: A sampled at T=1.0, C uses `temperature=1.0` (deepswe default). T=1.0 → prompt_logprobs moot.
- **dtype (audited)**: C uses `config.dtype=bfloat16` (COMPUTE — weights downcast to it at matmul,
  qwen3/model.py:328) + `param_dtype=float32` (storage). Matches deepswe (train:588). `call_model_config`
  defaults config.dtype to fp32, so the probe sets it explicitly — don't drop that.
- **No LoRA**: deepswe `--train_with_lora` default False, not passed → plain qwen3 (C is plain too).

## Run — TWO-STEP LADDER (Phase 3c)

Rationale: kernel alignment (B-vs-C) needs only a SAME-mesh comparison, not disaggregation. So
do the cheap kernel step on 1 host first, then the expensive mesh step on 256. (Decisions locked
2026-07-26; see `tasks/logprob_diff_operator_alignment/phase3c.md`.)

**CPU gates first (no TPU — run before either step):**
```bash
git fetch origin yuxzhang/logp-diff-probe && git reset --hard origin/yuxzhang/logp-diff-probe  # NOT git pull
cd scripts/logp_diff
python3 logp_diff_test.py        # ALL 6 PASS  (harness known-answer)
python3 probe_wiring_test.py     # ALL PASS    (disaggregated + COLOCATED wiring, mocked engines)
python3 logp_diff_probe.py --dry_run --colocated --rollout_mesh_tp=4 --rollout_mesh_fsdp=1
```

### Step 1 — 1-host COLOCATED (cheap, pure kernel diff)
4-chip v5p VM, DIRECT (no Pathways). vLLM(A/B) + tunix(C) share ONE mesh (fsdp1×tp4), run
SEQUENTIALLY (vLLM → free HBM via `delete_kv_cache` → tunix). No mesh confound → **B-vs-C = pure
kernel diff** (the Phase 4 fix target), at real Qwen3-32B. Caveat: tp4≠tp8, so the MAGNITUDE may
differ from Step 2 — this validates the method + attribution, not the final number.
```bash
# on a single-host v5p TPU VM (docker preinstalled):
gcloud auth configure-docker europe-west4-docker.pkg.dev          # one-time
HF_TOKEN=hf_xxx bash experimental/probe_v5p_1host_docker.sh       # fetches branch, patches RoPE, runs
# report at /tmp/logp_probe_1host/report.json
```
Prereq: Qwen3-32B safetensors at `/mnt/disks/linchai_data/models/Qwen3-32B` on the VM.
Watch (e2e-only, this is what Step 1 de-risks): `--vllm_server_mode=false` in-process vLLM boots;
`delete_kv_cache` actually frees HBM; colocated vLLM→tunix hand-off doesn't OOM (57GB→48GB/chip).

### Step 2 — 256 DISAGGREGATED (mirrors deepswe: kernel + mesh + decode)
vLLM on rollout_mesh (fsdp8×tp8, chips 0-63), tunix on train_mesh (fsdp8×tp8, chips 64-127),
128 idle — exactly deepswe's 64+64 split. Reproduces the real A-vs-C ~0.92 + the mesh term.
```bash
kubectl apply -f scripts/logp_diff/logp-diff-probe.yaml
# report lands at gs://yuxzhang-tunix-models/logp-diff/${RUN_TAG}/report.json
```

**What the probe args already match to deepswe** (audited 2026-07-26, no need to change):
`--rollout_mesh_{tp,fsdp}=8 --train_mesh_{tp,fsdp}=8` (64+64), `--param_dtype=float32`
`--config_dtype=bfloat16` (compute dtype = the numerics lever), `--vllm_hbm_util=0.6`
`--vllm_max_num_seqs=64 --vllm_max_num_batched_tokens=8192`, `enable_prefix_caching=False`,
`server_mode=True`, `async_scheduling=False`, no LoRA, temperature 1.0.
**Known un-matched (author's open decision):** sequence length (probe 2048+512 vs deepswe
4096+32768) and decode concurrency (probe batch=1 vs deepswe 64) — both affect batch-variance
magnitude; see `phase3b.md`.

## Report back (what Phase 4 needs)
`report.json` has:
- `comparisons{A-vs-C, A-vs-B, B-vs-C}` each `{mean,max,pearson,n}`.
- `decomposition`: `real_total(A-vs-C)` (should reproduce deepswe ~0.92), `decode(A-vs-B)`,
  `kernel+mesh(B-vs-C)`, `additivity_residual_max` (**must be ~0** — else A/B/C misaligned, a
  wiring bug, DON'T trust the numbers), `tunix_sharding_sensitivity(C-vs-C2)` (bounds the mesh term).
- Startup log line `[probe] devices=… rollout_mesh=… train_mesh=… idle=…` (expect 256/…/128 idle).

Read: is the diff dominated by `decode` or by `kernel+mesh`? If `kernel+mesh` dominates, a
follow-up co-located same-mesh run splits kernel from mesh. Then align the culprit kernel → bitwise 0.

## Fidelity — why the measured diff IS the training diff
C **calls the real `compute_per_token_logps`** (not a reimplementation) → same code as the trainer,
by construction. A = the exact `old_logprobs` array training uses. So `A-vs-C` here == the diff the
RL loop actually pays. See `tasks/logprob_diff_operator_alignment/{plan,design,goal}.md` for full rationale.
