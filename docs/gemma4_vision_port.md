# Porting the real Gemma 4 vision tower (`gemma4_vision`) to Tunix

## Why this exists

The first attempt at "multi-modal (vision) support for Gemma 4" reused Gemma 3's
**SigLIP** encoder (`tunix/models/gemma3/vision.py`, `SigLIPConfig`). That is the
wrong architecture. The released `google/gemma-4-e2b-it` checkpoint declares its
own vision tower:

```
architectures : ['Gemma4ForConditionalGeneration']
vision_config -> model_type: 'gemma4_vision', 16 layers, hidden 768,
                 intermediate 3072, patch 16
audio_config  -> model_type: 'gemma4_audio',  12 layers, hidden 1024
```

`gemma4_vision` is **not** SigLIP: it uses gated MLPs (`gate/up/down`), 2D RoPE,
per-head q/k/v RMSNorm, a 4-norm sandwich block, a factored 2D learned position
table, and a spatial pooler. The checkpoint also has a `gemma4_audio` tower and
single-tensor `embed_vision` / `embed_audio` projectors (there is **no**
`multi_modal_projector` that the SigLIP loader expected).

This document is the reverse-engineered spec (from HF
`transformers.models.gemma4.modeling_gemma4`, v5.x) and the staged plan to do it
correctly.

## Scope

* **In scope:** the vision tower + the `embed_vision` projector + the safetensors
  loader mappings + wiring into `Gemma4.__call__` + parity validation.
* **Out of scope (for now):** the `gemma4_audio` Conformer tower. The loader
  should gracefully **skip** all `model.audio_tower.*` and `model.embed_audio.*`
  keys (the existing loader already logs skipped keys; that is acceptable for a
  vision-only PR).

## Architecture (as ported in `tunix/models/gemma4/vision_real.py`)

`Gemma4VisionConfig` (e2b defaults): hidden 768, intermediate 3072, 16 layers,
12 heads, 12 kv-heads, head_dim 64, eps 1e-6, patch 16, position_embedding_size
10240, pooling_kernel_size 3, rope_theta 100.0, act `gelu_pytorch_tanh`,
`use_clipped_linears=False`, `standardize=False`.

Forward pipeline (`Gemma4VisionModel`):

1. **PatchEmbedder** — `pixel_values` are pre-flattened patches `[B, P, 3*16²]`.
   Compute `2*(px-0.5)`, project with a bias-free `Linear(768→768)`, then add a
   factored 2D position embedding: `one_hot(pos).permute @ table[2,10240,768]`
   summed over the two spatial axes. Padding patches (pos == -1) get zeroed.
2. **Encoder** — 2D RoPE (`rotary_emb`) computes cos/sin from `pixel_position_ids`
   (`inv_freq` over `spatial_dim = head_dim//2 = 32`, reused per axis). 16
   bidirectional sandwich layers:
   `x = x + post_attn_norm(attn(input_norm(x)))`, then
   `x = x + post_ffw_norm(mlp(pre_ffw_norm(x)))`.
   * **Attention** — bias-free q/k/v/o projections; `q_norm`,`k_norm` are scaled
     RMSNorm, `v_norm` is **unscaled** RMSNorm; multidim RoPE on q,k; `scaling =
     1.0` (not `head_dim**-0.5`); non-causal.
   * **MLP** — `down(gelu_tanh(gate(x)) * up(x))`.
3. **Pooler** — zero padding, 2D average-pool patches into
   `output_length = P / pooling_kernel²` soft tokens, scale by `sqrt(768)`.

Projector (`Gemma4MultimodalEmbedder`, a.k.a. `embed_vision`):
`embedding_projection(embedding_pre_projection_norm(soft_tokens))`, where the
pre-projection norm is **unscaled** RMSNorm(768) and the projection is bias-free
`Linear(768 → text_hidden=1536)`.

### Gemma-4-specific gotchas (verified against HF source)

* `Gemma4RMSNorm` scales by `weight` **directly** (not `1 + weight` like Gemma
  1/2/3). Loader must not apply a +1.
* `with_scale=False` norms (`v_norm`, `embedding_pre_projection_norm`) have **no**
  weight in the checkpoint — don't expect one.
* `Gemma4ClippableLinear` nests an inner `linear`, so checkpoint keys are
  `...{proj}.linear.weight` (note the extra `.linear.`). For e2b
  (`use_clipped_linears=False`) the `input_min/max`,`output_min/max` buffers are
  ±inf no-ops and can be ignored.
* Attention `scaling = 1.0`.

## Checkpoint → nnx key mapping (for the loader)

All real keys are prefixed `model.` (the existing vision regexes in
`params_safetensors.py` lack this prefix and use `re.match`, so they would skip
every vision key — that must be fixed, e.g. with an optional `(?:model\.)?`).

| Real checkpoint key | nnx param path | transform |
|---|---|---|
| `model.vision_tower.patch_embedder.input_proj.weight` | `vision_tower.patch_embedder.input_proj.kernel` | transpose (1,0) |
| `model.vision_tower.patch_embedder.position_embedding_table` | `vision_tower.patch_embedder.position_embedding_table` | none |
| `model.vision_tower.encoder.layers.N.self_attn.{q,k,v,o}_proj.linear.weight` | `...self_attn.{q,k,v,o}_proj.linear.kernel` | transpose (1,0) |
| `model.vision_tower.encoder.layers.N.self_attn.{q,k}_norm.weight` | `...self_attn.{q,k}_norm.scale` | none |
| `model.vision_tower.encoder.layers.N.mlp.{gate,up,down}_proj.linear.weight` | `...mlp.{gate,up,down}_proj.linear.kernel` | transpose (1,0) |
| `model.vision_tower.encoder.layers.N.{input_layernorm,post_attention_layernorm,pre_feedforward_layernorm,post_feedforward_layernorm}.weight` | `...{same}.scale` | none |
| `model.embed_vision.embedding_projection.weight` | `embed_vision.embedding_projection.kernel` | transpose (1,0) |
| `model.vision_tower.encoder.layers.N.self_attn.{q,k,v,o}_proj.{input,output}_{min,max}` | — | **skip** (±inf, e2b) |
| `model.audio_tower.*`, `model.embed_audio.*` | — | **skip** (out of scope) |

`tests/models/gemma4/vision_real_test.py::test_module_tree_matches_checkpoint_keys`
pins the nnx side of this table.

## Staged plan & status

* **Stage 1 — vision tower port. DONE (wiring/shapes only).**
  `tunix/models/gemma4/vision_real.py` builds and forward-passes; module tree
  maps 1:1 to checkpoint keys. Tests:
  `tests/models/gemma4/vision_real_test.py` (4 passing). **Numerics unvalidated.**
* **Stage 2 — image processor + loader. DONE (coverage-verified; resize + numerics pending).**
  * `tunix/models/gemma4/image_processing.py` — NumPy port of the HF processor
    (target-size, patchify, `(x,y)` position ids, padding = -1). Resize uses PIL
    (not bit-exact with torchvision — see caveat).
  * `tunix/models/gemma4/vision_params_safetensors.py` — `vision_key_mapping`
    (`model.`-prefix-safe, `$`-anchored) + `create_vision_stack_from_safe_tensors`
    loading a `Gemma4VisionStack`. Audio + clip buffers are skipped.
  * Tests (`vision_params_safetensors_test.py`, `image_processing_test.py`):
    every real vision key maps to a param, linears transpose / norms don't,
    audio+clip buffers skip, no double-matches, **no uninitialised params**, and
    processor output feeds the stack end-to-end. **Numeric parity still pending.**
* **Stage 3 — numeric parity (GATING). PORT MATH VALIDATED; real-checkpoint run pending.**
  Two harnesses:
  * `examples/gemma4/vision_parity_random_weights.py` — **checkpoint-free**.
    Builds a small HF `Gemma4VisionModel`+`Gemma4MultimodalEmbedder` with random
    weights, serializes them under real checkpoint key names, loads them into the
    JAX `Gemma4VisionStack` via the production loader, and compares per-layer in
    fp32. **Run in-sandbox (transformers 5.9.0 + torch 2.12): PARITY PASSED** —
    `after_patch_embed` exact, per-layer max-abs ~1e-6, tower 1.5e-5, projector
    3.6e-7. This validates the port math + key mapping + loader against HF.
  * `examples/gemma4/vision_parity_check.py` — same comparison against a **real**
    `google/gemma-4-*-it` checkpoint (adds real-weight file loading + an actual
    forward). Run:
    ```
    pip install torch 'transformers==5.9.0' safetensors   # 5.10.1 trips a torch._dynamo bug
    python examples/gemma4/vision_parity_check.py --ckpt ~/gemma4-e2b
    ```
  Status: the architecture/math is proven correct against HF. The only thing the
  real-checkpoint run adds is confirming the real `.safetensors` files load and a
  real image captions — do that before Stage 4.
* **Stage 4 — end-to-end. WIRED + merge-validated; full-model parity + caption pending.**
  * `tunix/models/gemma4/model.py` — `Gemma4.__call__` gains a non-breaking
    `input_embeddings` override (legacy SigLIP path untouched).
  * `tunix/models/gemma4/multimodal.py` — `Gemma4Multimodal` composes the text
    `Gemma4` with `Gemma4VisionStack`: embed tokens, run the vision stack, scatter
    soft tokens at `tokens == image_token_id` (HF `masked_scatter` equivalent via
    `merge_embeddings`), run the transformer on merged embeddings. Plus
    `create_multimodal_from_safe_tensors` (loads text + vision from one checkpoint).
  * `examples/gemma4/multimodal_generate.py` — single-image, no-padding, eager
    greedy caption demo.
  * Tests (`multimodal_test.py`, sandbox dry-run): soft tokens land exactly at
    image positions, text positions are untouched, the image changes downstream
    logits, the mask is bidirectional over the image span, and the greedy loop
    runs end-to-end. **26/26 gemma4 tests pass.**

  Open items / honest limits:
  * **Single, non-padded image only** — the merge assumes #valid-soft-tokens ==
    #placeholders; multi-image / padded batches need a valid-token gather first.
  * **Real caption** needs the checkpoint (run `multimodal_generate.py`).

* **Stage 4 follow-up — full-model parity vs HF `Gemma4Model.forward`.**
  `examples/gemma4/multimodal_parity_random_weights.py` builds a tiny random HF
  `Gemma4ForConditionalGeneration`, saves it as safetensors, loads into
  `Gemma4Multimodal`, runs both, and diffs per-position logits + PLE.

  Findings against HF:
  * **PLE token-identity branch: bit-exact (max=0).** HF
    `Gemma4Model.forward` does NOT pass the merged image embeddings to PLE;
    it substitutes `image_token_id → pad_token_id` in the ids handed to
    `embed_tokens_per_layer`, while the context-projection branch sees the
    merged embeddings (with vision at image positions). `Gemma4Multimodal.
    _compute_per_layer_inputs` mirrors this exactly; locked in by
    `tests/models/gemma4/multimodal_test.py::
    test_per_layer_inputs_substitutes_pad_at_image_positions`.
  * **Bidirectional vs causal mask.** HF `Gemma4TextConfig.
    use_bidirectional_attention == "vision"` controls this; smaller models
    default to plain causal. Tunix matches via a `bidirectional_image_span`
    flag on `Gemma4Multimodal` (default `False` = causal, like HF small).
  * **Residual divergence at text positions after the image** (max ~9e-2 in
    the random-weights probe) reproduces in a **pure-text** parity (same HF
    weights, no image, no multimodal wrapper) — so this is a **pre-existing
    Tunix-vs-HF divergence in `tunix.models.gemma4` text-model arithmetic**
    (probably global-layer RoPE / proportional partial-rotary settings), not
    something introduced by the vision port. Out of scope for this PR; worth
    flagging upstream separately.

  Net: the multimodal wrapper itself now matches HF semantically on every
  point we can prove without the real checkpoint. BOS (pure text, no image
  influence) is exact (1.6e-7); image-position logit diff is `~4e-2` and
  bounded by the pre-existing text-model divergence.

## Validation prerequisites (Stage 3)

Neither a clean Tunix venv nor this sandbox ships torch. To produce HF reference
activations you must, on a machine with the checkpoint:

```bash
pip install 'torch' 'transformers==5.10.1'   # or whatever ships Gemma4
```

Then load the same safetensors into both stacks and diff. (A harness skeleton
will live in `examples/gemma4/vision_parity_check.py` once Stage 2 lands.)

## Caveats

* This is a from-source port; bit-exact parity is unproven until Stage 3.
* The official multimodal Gemma 4 implementation is likely in progress upstream
  in Tunix (the text model is Google-authored). Coordinate before investing in
  Stages 2-4 to avoid duplicating in-flight work.
