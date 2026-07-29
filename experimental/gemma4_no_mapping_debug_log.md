# Gemma4-E2B `No mapping for source key` & RL Zero-Reward Debug Log

This document records the exact error logs, architectural root causes, and diagnostic proofs for Gemma4-E2B on TPU VMs (`tpu_inference` vLLM + JAX Trainer), intended for cross-agent debugging.

---

## ★★ CORRECTION (2026-07-29) — sections 1-4 below misread the evidence

**Root cause found and fixed: `TO_HF_MAPPINGS` targeted the wrong namespace, so
*zero* weights were ever transferred. vLLM ran on random init weights.**

Fix: `tunix/models/gemma4/mapping_vllm_jax.py`, all 29 target names
`model.*` → `language_model.*`. Ground truth: `tpu_inference` 0.25.0's
`Gemma4ForCausalLM.__init__` assigns the decoder to `self.language_model`
(`models/jax/gemma4.py:1035`), so nnx attribute paths — which is what
`build_flat_dict` matches against — start with `language_model.`, never `model.`.
The `model.` string in checkpoint names is stripped by `load_weights`
(`gemma4.py:1052`, `WeightsMapper(orig_to_new_prefix={"model.": ""})`); it is not
part of the attribute path. `origin/main` already carried this fix
(`71eef52c`); the sync from `origin/gemma4_mapping_update` (`953e3edc`) reverted
it while adding the (correct, keep-them) fused `qkv_fused`/`gate_up_fused`
entries.

Where sections 1-4 went wrong:

- **§1-2 "layers 0-5 map, 6+ fail" is false.** The pasted excerpt is a window
  that happens to start at `layers.6`. In the full logs `layers.0` … `layers.34`
  are *all* present in the failure list. `/tmp/train_frozenlake_logs/fl_mem8k.log`
  contains exactly **495** `No mapping` lines = 35 layers × 14 keys + 5 non-layer
  keys = **one sync in which every single source param failed**;
  `fl_smoke.log` has 3465 = 495 × 7, i.e. seven syncs, each 100% failing.
  Verified independently: instantiating the real 35-layer E2B structure yields
  **495 source params, of which 0 are unmapped after the fix**.
  ⟹ There is no layer-count / shared-layer discrepancy to chase.
- **§3 "100% numerically aligned" is backwards.** `prob_diff≈1e-5` is trivially
  satisfied when *both* engines are near-uniform over a 262144-token vocab; it
  does not show agreement. The load-bearing numbers in the same log line are
  `logp_diff_mean ≈ 9.0` nats (a ~9000× probability ratio) and
  `probs_pearson_corr = 0.00000` — **zero** correlation, the signature of one
  side being noise.
- **§4's response budget is a symptom, not the cause.** A random-weight model
  babbles, so it hits `MAX_CONTEXT_LIMIT_REACHED` at any budget. This is why
  `4096+4096` was *also* all-zero. Re-measure the budget question only after a
  run whose weight sync actually lands.

Also added so this class of bug can never be silent again:
`transfer_state_with_mappings(..., strict_mapping=True)` now raises
`MappingError` naming the unmapped keys, instead of logging one `ERROR` per key
and letting the caller report `"Weights synced"`
(`tunix/generate/utils.py`; gates in `tests/generate/utils_test.py`).

---

## 1. Error Log: `No mapping for source key: layers.6...`

When running state transfer between JAX (`Gemma4`) and vLLM (`Gemma4ForCausalLM`), logs show that layers 0 through 5 map without issues, but starting from layer 6, `No mapping for source key` errors occur:

```text
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.post_attention_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.post_ffw_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.post_per_layer_input_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.pre_attention_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.pre_ffw_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.6.skip_scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.attn._key_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.attn._query_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.attn.attn_vec_einsum.w
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.attn.qkv_fused.w
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.mlp.down_proj.kernel
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.mlp.gate_up_fused.kernel
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.per_layer_input_gate.w
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.per_layer_projection.w
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.post_attention_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.post_ffw_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.post_per_layer_input_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.pre_attention_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.pre_ffw_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.7.skip_scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.8.attn._key_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.8.attn._query_norm.scale
2026-07-29 22:06:26 - ERROR - [absl] No mapping for source key: layers.8.attn.attn_vec_einsum.w
```

---

## 2. Technical Analysis: Why do layers 0-5 match, but layers 6+ say `No mapping`?

### A. How `transfer_state_with_mappings` works (`tunix/generate/utils.py`)
1. **Target-driven mapping table registration**:  
   `transfer_state_with_mappings` calls `build_flat_dict(dst_state.flat_state(), key_mappings)`, which iterates over the **target (vLLM) model's parameter tree (`dst_state`)**. For each target key present in `dst_state`, it matches against `TO_HF_MAPPINGS` regular expressions and inserts the corresponding source key into `src_to_tgt_map`.
2. **Source parameter traversal**:  
   Next, `_unroll_scanned_layers(src_state, src_to_tgt_map)` iterates through the **source JAX trainer model's parameters (`src_state`)**.
3. **Why `No mapping` is logged**:  
   Whenever `_unroll_scanned_layers` encounters a key in `src_state` that was **not registered in `src_to_tgt_map`**, it logs `ERROR - [absl] No mapping for source key: <src_key>` and calls `continue`.

### B. Why layers 0..5 succeed while layers 6+ fail
- Because `layers.0` through `layers.5` match successfully with the exact same parameter names (`post_attention_norm.scale`, `qkv_fused.w`, etc.), **the regex patterns in `TO_HF_MAPPINGS` are verified to be 100% correct**.
- The failure starting at `layers.6` indicates a **layer count / node existence discrepancy between the JAX source `ModelConfig` and the instantiated vLLM target model**:
  - The JAX `Gemma4` config (`num_layers=35`, `frac_shared_layers=20.0/35`) generates 35 source layer nodes.
  - If the instantiated vLLM target model (`dst_state`) only has layers 0 to 5 (e.g. in a micro/test config) or has different shared-layer collapsing / layer-sharding schemes, target nodes for `layers.6` to `layers.34` do not exist in `dst_state`. Consequently, `build_flat_dict` never registers them into `src_to_tgt_map`, causing `_unroll_scanned_layers` to complain when scanning JAX layers 6+.

---

## 3. Status of Gemma4-E2B Numerical Alignment (`prob_diff = 0.00000`)

By syncing Host Linchai's working branch `origin/gemma4_mapping_update` (commit `953e3edc`), we added:
1. **`_reorder_for_tp_sharding(concatenated, split_sizes, tp_size)`**: Interleaves Tensor Parallelism shards (`[part0_shard_0, part1_shard_0, ...]`) so that concatenated QKV and Gate-Up weights align with `tpu_inference` kernel expectations.
2. **Synthetic Fused Mappings**:
   - `'layers.*.attn.qkv_fused.w' -> 'model.layers.*.self_attn.qkv_proj.weight'`
   - `'layers.*.mlp.gate_up_fused.kernel' -> 'model.layers.*.mlp.gate_up_proj.weight'`
3. **Full 29-Entry `TO_HF_MAPPINGS`**: Targets vLLM's `'model.'` namespace.

### Numerical Proof from Real TPU VM Runs
In our parallel TPU VM runs (`fl_pack_gemma_v2` on `maxtext-single-host-1-v5p-8` and `fl_unpack_gemma_v2` on `lancewang-v5p-8`):
```text
2026-07-29 21:56:11 - INFO - [absl] sampler-trainer: logp_diff=(9.12551,46.92870) prob_diff=(0.00001,0.00815) pearson=0.00000
2026-07-29 21:56:31 - INFO - [absl] sampler-trainer: logp_diff=(8.98676,45.53446) prob_diff=(0.00000,0.00351) pearson=0.00000
```
- **`prob_diff_mean = 0.00000` (0.0% mean probability error)**
- **`prob_diff_max = 0.003` (0.3% max token probability error across sequence)**
- **Conclusion**: JAX Trainer and vLLM Rollout engine forward-pass probabilities are **100% numerically aligned**.

---

## 4. Why `solve_ratio=0.000` on FrozenLake (The Response Budget Issue)

Even with 100% probability alignment, `solve_ratio` remains `0.000` when running with default `main` hyperparams (`MAX_RESPONSE=2048`, `ENV_MAX_STEPS=8`):
- **Observation**: Logs show over **2,000+ occurrences** of `trajectory clipped: MAX_CONTEXT_LIMIT_REACHED`.
- **Reason**: In `train_frozenlake.py`, `--max_response_length 2048` is an **episode-level budget across all turns**. With `--env_max_steps 8`, the agent gets only `2048 / 8 = 256` tokens per turn. Gemma4-E2B outputs verbose reasoning and formatting, exceeding 256 tokens and getting truncated by `MAX_CONTEXT_LIMIT_REACHED` before reaching the maze goal.
- **Host Linchai's Solution**: In Host Linchai's working branch `origin/linchai_gemma4` (commit `9a10e029`), she used:
  - `--max_response_length 4096`
  - `max_steps = 5` in `examples/frozenlake/env.py`
  - This provides `4096 / 5 = 819` tokens per turn, allowing Gemma4-E2B to complete episodes without truncation.
