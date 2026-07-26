# Operator Bitwise Alignment — Probe README (for the remote/TPU agent)

**Branch**: `yuxzhang/logp-diff-probe` (forked from `deepswe-quality-fix`).
**You are picking up a CPU-prepared, TPU-pending experiment.** Everything that can be proven
without a TPU is proven; your job is to wire the `_REMOTE_VERIFY_` hooks, run on TPU, and report back.
Date: 2026-07-23.

## Goal
Eliminate the rollout-vs-train per-token **log-prob diff** on **Qwen3-32B** down to **bitwise**
(operator / kernel alignment). This is the **Training-Inference Mismatch (TIM)** residual
(deepswe measured: mean 0.002 / max 0.92 / pearson 0.998, APC off). **Ultimate goal = 0 logp diff**
by aligning the culprit kernel (batch-invariant, à la Thinking Machines). TIS stays as a safety net,
NOT the solution. (Root cause class: batch-size-dependent reduction order in **RMSNorm / matmul / attention**.)

## What's in this branch
```
scripts/logp_diff/
  logp_diff_probe.py     # THE probe: real 3-source A/B/C on Qwen3-32B (run this on TPU)
  logp-diff-probe.yaml   # launch JobSet (adapt instance_type/mesh to your cluster)
  harness.py             # framework-agnostic logp-diff + per-op attribution (CPU-proven)
  sources.py             # Source abstraction + compare()
  toy_qwen3.py           # numpy toy qwen3 (mirrors real structure; harness validation only)
  logp_diff_test.py      # CPU gate: 6 known-answer tests (attribution correctness) — ALL PASS
  probe_wiring_test.py   # CPU gate: runs probe main() with tunix/vLLM mocked — ALL PASS
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
5. **Same mesh** for B and C: vLLM manages devices via its sharding config (`device_indexes`), NOT the
   nnx `Mesh` used by C. "Same mesh" = same devices / same tp; wire explicitly.

## Pre-checks before trusting any diff (go/no-go)
- **Weight equality**: B(vLLM) and C(tunix) bitwise-equal weights (see #1). Verify first.
- **Temperature**: A sampled at T, C uses `temperature=T` (= rollout config temperature; deepswe **1.0**);
  ensure B's `prompt_logprobs` use the same convention (raw vs T-scaled). T=1.0 → moot.
- **Config/graphdef**: `call_model_config("Qwen3-32B")`; deepswe uses **no LoRA** → plain qwen3 graphdef.

## Run
```bash
git fetch origin yuxzhang/logp-diff-probe && git checkout yuxzhang/logp-diff-probe
# 1) wire the _REMOTE_VERIFY_ hooks above (esp. #1 real weights)
# 2) sanity on CPU first (proves harness + wiring):
python3 scripts/logp_diff/logp_diff_test.py        # expect ALL 6 PASS
python3 scripts/logp_diff/probe_wiring_test.py     # expect ALL PASS
python3 scripts/logp_diff/logp_diff_probe.py --dry_run
# 3) on TPU: adapt + apply the JobSet
kubectl apply -f scripts/logp_diff/logp-diff-probe.yaml
```

## Report back (what Phase 4 needs)
`report.json` with, for each pair (A-vs-C, B-vs-C, A-vs-B): `logp_diff{mean,max,pearson}` +
`attribution.first_divergence` (op-type) + per-op-type standalone isolation diffs. Plus the startup
log's `*** mesh ***` / `jax.devices()` count (idle-chip check). Then we align the culprit kernel →
verify B-vs-C → bitwise 0.

## Fidelity — why the measured diff IS the training diff
C **calls the real `compute_per_token_logps`** (not a reimplementation) → same code as the trainer,
by construction. A = the exact `old_logprobs` array training uses. So `A-vs-C` here == the diff the
RL loop actually pays. See `tasks/logprob_diff_operator_alignment/{plan,design,goal}.md` for full rationale.
