# GSM8K Zero-TIM Full Step 64 Rescore Alignment Incident Report

## 1. Executive Summary

| Property | Value |
| :--- | :--- |
| **JobSet Name** | `canon-v1hp-gsm8k-gfull1-799a0bd1` |
| **Workload** | GSM8K Zero-TIM Full Training (200 Steps target) |
| **Hardware Topology** | 64 TPU v5p (DP16 x TP4) |
| **Source Commit** | `799a0bd1ed5ecfd7a2f6e42eeaced82886fec76c` |
| **Profile** | `cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env` |
| **W&B Run URL** | [zero-tim-gsm8k-dp16-tp4/runs/fmrug2iu](https://wandb.ai/yuxzhang-google/zero-tim-gsm8k-dp16-tp4/runs/fmrug2iu) |
| **Terminal Step** | Step 64 (32.0% progress) |
| **Solve Ratio** | 77.7% (Reward mean: 0.792, up from initial 34.8%) |
| **Zero-TIM Alignment** | 100% PASS across 64 steps (`alignment_max_differing_bytes=0`) |
| **Failure Mode** | `RuntimeError: row 255: engine returned 1 prompt logprobs for 1130 tokens; cannot align the re-score` |

---

## 2. Root Cause Analysis

### Sequence of Events
1. Training executed stably for 64 full GRPO updates (steps 0–64) at ~46s/step.
2. Monotonic solve rate climbed from 34.8% to **77.7%**, with zero gradient anomalies (`commit_gradient_norm` = 0.596, `alignment_max_differing_bytes` = 0).
3. At step 64 rollout call 65, trajectory clipping triggered:
   `WARNING - [absl] [step_idx=1, pair_index=7, group_id=2077] trajectory clipped: MAX_CONTEXT_LIMIT_REACHED`
   Row 255 accumulated prompt + generated completion tokens to total length $N = 1130$.
4. During `get_prefill_rescore_logps` (`tunix/rl/rollout/vllm_rollout.py:526`), vLLM evaluated the 1130-token request under `prompt_logprobs=0`.
5. Due to context length bounds or prompt logprob handling for long prompt sequences in the rollout engine, vLLM returned only 1 prompt logprob element (`len(plp) = 1`) instead of the required 1130 elements.
6. The strict fail-closed alignment validation check in `vllm_rollout.py:525-529` asserted `len(plp) == len(seq)` and threw:
   ```text
   RuntimeError: row 255: engine returned 1 prompt logprobs for 1130 tokens; cannot align the re-score
   ```

---

## 3. Evidence & Artifacts

All files extracted from head pod `canon-v1hp-gsm8k-gfull1-799a0bd1-pathways-head-0-0-hkc9k`:
* `run.log`: Full execution log (60,072 lines, 6.9 MiB).
* `RAW_ERROR.log`: Tail containing traceback, W&B final summary, and step 64 metrics.
* `pre_alignment.jsonl`: Step 0 pre-alignment bitwise logs (0 differing bytes).
* `updates.jsonl`: Monotonic gradient and parameter update logs for steps 0–64.
* `env.sh`: Resolved cluster runtime environment variables.
* `receipt.json`: Signed run metadata and SHA-256 digests.

---

## 4. Remediation Plan

1. **Prompt/Completion Length Clamping**:
   Ensure `max_prompt_length` + `max_response_length` strictly respects vLLM `max_model_len` across multi-turn trajectories, or configure vLLM prompt logprob generation to never truncate prompt logprob arrays when `prompt_logprobs=0`.
2. **Relaunch**:
   Launch fresh GSM8K Zero-TIM run with the updated length bounds alongside M15 and DeepSWE runs.
