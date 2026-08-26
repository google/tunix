# P58.13 — DeepSWE 4B TP8 Step-0 backward fixed LM head M=2048 diagnosis

## Status

`DIAGNOSIS COMPLETE / ERROR PRESERVED / EVIDENCE RECORDED / TARGET NOT RUN`

## Trigger

Target JobSet `canon-p58-ds4b-zero-hp-full-p58z02` (Qwen3-4B-Instruct DP8xTP8, 128 TPU chips) executed Step 0.
All 128 SWE-bench RepoEnv trajectories completed generation in 1 wave without timeouts, verifying the P58.12 engine-seed repair.

At Step 0 learner backward pass (`_process_results` -> `get_actor_per_token_logps` -> `compute_per_token_logps` -> `compute_logits`), execution failed at `fixed_lm_head`:

```text
[rank0]: Traceback (most recent call last):
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 36, in <module>
[rank0]:     main()
[rank0]:   File "/app/examples/deepswe/canonical_entrypoint.py", line 32, in main
[rank0]:     runpy.run_module("examples.deepswe.train_deepswe_nb", run_name="__main__")
[rank0]:   File "<frozen runpy>", line 229, in run_module
[rank0]:   File "<frozen runpy>", line 88, in _run_code
[rank0]:   File "/app/examples/deepswe/train_deepswe_nb.py", line 1812, in <module>
[rank0]:     agentic_grpo_learner.train(train_dataset=train_dataset)
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 3489, in train
[rank0]:     train_examples = self._batch_to_train_example(
[rank0]:                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_rl_learner.py", line 2866, in _batch_to_train_example
[rank0]:     return self._process_results(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/agentic/agentic_grpo_learner.py", line 862, in _process_results
[rank0]:     trainer_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
[rank0]:                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/rl_cluster.py", line 1335, in get_actor_per_token_logps
[rank0]:     common.compute_per_token_logps(
[rank0]:   File "/app/tunix/rl/common.py", line 394, in compute_per_token_logps
[rank0]:     return canonical_forward.compute_per_token_logps(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_forward.py", line 83, in compute_per_token_logps
[rank0]:     return require_registered().compute_per_token_logps(**kwargs)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 9320, in compute_per_token_logps
[rank0]:     grouped_logps, grouped_entropy = jax.lax.map(
[rank0]:                                      ^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 9310, in grouped_body
[rank0]:     return self._sequence_group(
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 9125, in _sequence_group
[rank0]:     caches, chunk_output = jax.lax.cond(
[rank0]:                            ^^^^^^^^^^^^^
[rank0]:   File "/app/tunix/rl/canonical_qwen3_adapter.py", line 9029, in run_nonempty
[rank0]:     logits = self._runner.compute_logits_fn(
[rank0]:              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/models/common/model_loader.py", line 414, in run_compute_logits
[rank0]:     return model.compute_logits(hidden_state)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z02/canon/qwen3.py", line 672, in compute_logits
[rank0]:     return self.model.embed_tokens.decode(hidden_states)
[rank0]:            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/jax/linear.py", line 130, in _p38_fixed_tied_head_decode
[rank0]:     return _p38_fixed_lm_head(
[rank0]:            ^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z02/canon/p38_fixed_lm_head.py", line 405, in fixed_lm_head
[rank0]:     semantic_m = validate_global_contract(
[rank0]:                  ^^^^^^^^^^^^^^^^^^^^^^^^^
[rank0]:   File "/mnt/disks/linchai_data/deepswe_zero_tim/canon-p58-ds4b-zero-hp-full-p58z02/canon/p38_fixed_lm_head.py", line 237, in validate_global_contract
[rank0]:     raise ValueError(
[rank0]: ValueError: P38 fixed lm_head requires semantic M in (8, 16, 32, 64, 128, 256, 4096), got (2048, 2560)
```

## Root Cause Analysis

1. Geometry: Qwen3-4B has `hidden_size=2560`, `vocab_size=151936`.
   On TP8, `tp_size=8`, so `(hidden_size, tp_size) = (2560, 8)`.
2. Learner Microbatching:
   - Batch size: 128 trajectories across 8 DP ranks = 16 trajectories per DP rank.
   - Microbatch chunk length: 128 tokens.
   - Total batch tokens per microbatch: $16 \times 128 = 2048$ tokens ($M = 2048$).
3. Shape Validation:
   - `validate_global_contract` calls `_semantic_m_for_geometry(geometry)` where `geometry = (hidden_size, tp_size) = (2560, 8)`.
   - `_semantic_m_for_geometry` contained:
     ```python
     if geometry == (4096, 8):
       return QWEN8B_TP8_LEARNER_M  # (2048, 4096)
     return LEARNER_M  # (4096,)
     ```
   - Because `(2560, 8) != (4096, 8)`, it returned `LEARNER_M = (4096,)`, which caused $M=2048$ to be rejected with `ValueError`.

## Planned Resolution (For Implementation Phase)

1. Generalize learner semantic M admission for all TP8 geometries:
   ```python
   TP8_LEARNER_M = (2048, 4096)
   QWEN8B_TP8_LEARNER_M = TP8_LEARNER_M
   ```
   In `_semantic_m_for_geometry(geometry)`:
   ```python
   learner_m = TP8_LEARNER_M if geometry.tp_size == 8 else LEARNER_M
   ```
2. Update tests in `test_fixed_lm_head.py` and `probe_fixed_lm_head_overlay.py` to assert $M \in (2048, 4096)$ for 4B TP8 `(2560, 8)` and 8B TP8 `(4096, 8)`.
3. Add `CANON_P67_P66_VMA_P59_ONLY=1` to `cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env`.

## Preserved Evidence

- Error log: `evidence/p58z02_backward_fixed_lm_head_error/run.log`
  - SHA256: `7349c7965f31e2c84dfd98f8cb7fe175f9b2d4281759d0bb5c07bb336ef8784d`
  - Size: 2.1 MB
