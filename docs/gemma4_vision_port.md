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
* **Stage 2 — image processor + loader.** TODO. Port
  `image_processing_gemma4.py` (patchification to `[B,P,3*16²]` + `(x,y)`
  `pixel_position_ids`, padding = -1), and add the vision/`embed_vision` mappings
  above to `params_safetensors.py` (with the `model.` prefix fix). Add a loader
  key-coverage check so no vision key is silently skipped.
* **Stage 3 — numeric parity (GATING).** TODO. With torch + the real checkpoint,
  load identical weights into HF `Gemma4VisionModel` and this module, feed the
  same `pixel_values`/`pixel_position_ids`, and assert per-layer max-abs-diff is
  within bf16 tolerance. Until this passes, do **not** claim the port works.
* **Stage 4 — end-to-end.** TODO. Wire into `Gemma4.__call__` (replace the
  SigLIP path), merge soft tokens at image-token positions, and generate a
  caption from the real checkpoint.

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
