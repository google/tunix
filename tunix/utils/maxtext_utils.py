# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MaxText model configuration and runtime utilities."""

from __future__ import annotations

import logging
import os
from typing import Any


def apply_gdn_conv_padding_fix() -> None:
  """Patches GDN conv1d/output/query padding masks and safe l2norm in MaxText."""
  patches = [
      (
          "qkv_mask",
          "qkv = jnp.where((decoder_segment_ids != 0)[..., None], qkv, 0.0)",
          "    qkv = jnp.concatenate([q, k, v], axis=-1)\n    batch, seq_len, _ = qkv.shape",
          (
              "    qkv = jnp.concatenate([q, k, v], axis=-1)\n"
              "    if decoder_segment_ids is not None:\n"
              "      qkv = jnp.where((decoder_segment_ids != 0)[..., None], qkv, 0.0)\n"
              "    batch, seq_len, _ = qkv.shape"
          ),
      ),
      (
          "conv_out_mask",
          "conv_out = jnp.where((decoder_segment_ids != 0)[..., None], conv_out, 0.0)",
          "    conv_out = conv_out[:, -seq_len:, :]\n    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)",
          (
              "    conv_out = conv_out[:, -seq_len:, :]\n"
              "    if decoder_segment_ids is not None:\n"
              "      conv_out = jnp.where((decoder_segment_ids != 0)[..., None], conv_out, 0.0)\n"
              "    qkv_conv = jax.nn.silu(conv_out.astype(jnp.float32)).astype(cfg.dtype)"
          ),
      ),
      (
          "output_mask",
          "output = jnp.where((decoder_segment_ids != 0)[..., None], output, 0.0)",
          "    output = self.out_proj(gated_output, out_sharding=out_sharding)\n\n    return output, active_cache",
          (
              "    output = self.out_proj(gated_output, out_sharding=out_sharding)\n"
              "    if decoder_segment_ids is not None:\n"
              "      output = jnp.where((decoder_segment_ids != 0)[..., None], output, 0.0)\n\n"
              "    return output, active_cache"
          ),
      ),
      (
          "query_mask",
          "query = jnp.where(mask[..., None, None], query, 0.0)",
          "      # Apply mask by broadcasting to respective shapes\n      key = jnp.where(mask[..., None, None], key, 0.0)",
          (
              "      # Apply mask by broadcasting to respective shapes\n"
              "      query = jnp.where(mask[..., None, None], query, 0.0)\n"
              "      key = jnp.where(mask[..., None, None], key, 0.0)"
          ),
      ),
  ]
  candidates = [
      "/app/maxtext/src/maxtext/models/qwen3.py",
      "/opt/venv/lib/python3.12/site-packages/maxtext/models/qwen3.py",
  ]
  for p in candidates:
    if not os.path.exists(p):
      continue
    try:
      with open(p, encoding="utf-8") as f:
        txt = f.read()
      modified = False
      for name, check_str, needle, replacement in patches:
        if check_str in txt:
          print(f"[gdn-fix] {name} already present in {p}", flush=True)
        elif needle in txt:
          txt = txt.replace(needle, replacement, 1)
          modified = True
          print(f"[gdn-fix] PATCHED {name} into {p}", flush=True)
        else:
          print(f"[gdn-fix] WARNING: needle for {name} not found in {p}", flush=True)
      if modified:
        with open(p, "w", encoding="utf-8") as f:
          f.write(txt)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[gdn-fix] WARNING: failed to patch {p}: {e}", flush=True)

  norm_patches = [
      (
          "safe_l2norm",
          "safe_norm_sq = jnp.where(is_zero, 1.0, norm_sq)",
          (
              "  inv_norm = jax.lax.rsqrt((x * x).sum(axis=dim, keepdims=True) + jnp.array(eps, dtype=x.dtype))\n"
              "  return x * inv_norm"
          ),
          (
              "  norm_sq = (x * x).sum(axis=dim, keepdims=True)\n"
              "  is_zero = norm_sq == 0.0\n"
              "  safe_norm_sq = jnp.where(is_zero, 1.0, norm_sq)\n"
              "  inv_norm = jnp.where(is_zero, 0.0, jax.lax.rsqrt(safe_norm_sq + jnp.array(eps, dtype=x.dtype)))\n"
              "  return x * inv_norm"
          ),
      ),
  ]
  norm_candidates = [
      "/app/maxtext/src/maxtext/layers/normalizations.py",
      "/opt/venv/lib/python3.12/site-packages/maxtext/layers/normalizations.py",
  ]
  for p in norm_candidates:
    if not os.path.exists(p):
      continue
    try:
      with open(p, encoding="utf-8") as f:
        txt = f.read()
      modified = False
      for name, check_str, needle, replacement in norm_patches:
        if check_str in txt:
          print(f"[gdn-fix] {name} already present in {p}", flush=True)
        elif needle in txt:
          txt = txt.replace(needle, replacement, 1)
          modified = True
          print(f"[gdn-fix] PATCHED {name} into {p}", flush=True)
        else:
          print(f"[gdn-fix] WARNING: needle for {name} not found in {p}", flush=True)
      if modified:
        with open(p, "w", encoding="utf-8") as f:
          f.write(txt)
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"[gdn-fix] WARNING: failed to patch {p}: {e}", flush=True)


def maxtext_modules():
  """Imports MaxText lazily; some installs nest it under maxtext.src.maxtext."""
  apply_gdn_conv_padding_fix()
  from maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import maxtext_engine  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils  # pylint: disable=g-import-not-at-top
  return pyconfig, maxtext_engine, maxtext_utils


def get_tokenizer_pad_id(
    model_id: str,
    tokenizer_path: str = "",
    model_dir: str = "",
) -> int:
  """Resolves the pad token id the MaxText adapter masks with."""
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  path = tokenizer_path or model_dir or model_id
  tokenizer: Any = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
  if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = getattr(tokenizer, "pad_token_id", None)
  return int(pad_id) if pad_id is not None else 0


# vLLM instantiates a model class by the HF `architectures` entry, so this is
# what selects maxtext_vllm_adapter's `MaxTextForCausalLM` over the stock
# vLLM implementation. Pass it as the engine's `hf_overrides`.
VLLM_MAXTEXT_HF_OVERRIDES = {"architectures": ["MaxTextForCausalLM"]}


def build_vllm_maxtext_additional_config(
    model_name: str,
    *,
    attention: str = "",
    prefuse_moe_weights: bool | None = None,
) -> dict[str, Any]:
  """Builds the vLLM `additional_config` a MaxText rollout model reads.

  `MaxTextForCausalLM` builds its own MaxText config from
  `additional_config["maxtext_config"]`; these are the inference-side overrides
  that make it match the trainer's model. Kept here so the in-process and
  server-mode samplers cannot drift apart.

  Args:
    model_name: MaxText model name, e.g. `qwen3-1.7b`.
    attention: MaxText attention kernel override; MaxText's default is used when
      empty.
    prefuse_moe_weights: Whether the rollout expects w0/w1 pre-fused into the
      TPU GMM layout. None leaves MaxText's default in place.

  Returns:
    The `additional_config` mapping to hand to the vLLM engine.
  """
  apply_gdn_conv_padding_fix()
  overrides: dict[str, Any] = {
      "model_name": model_name,
      "model_call_mode": "inference",
      "enable_dp_attention": False,
      "allow_split_physical_axes": True,
      "log_config": False,
      "weight_dtype": "bfloat16",
      "logits_dot_in_fp32": True,
      "cast_logits_to_fp32": True,
      "float32_logits": True,
      "float32_gate_logits": True,
      "float32_weight_sum": True,
  }
  if prefuse_moe_weights is not None:
    overrides["prefuse_moe_weights"] = prefuse_moe_weights
  if attention:
    overrides["attention"] = attention

  # Load the SAME MaxText/Orbax checkpoint the trainer loads, instead of letting
  # MaxText-in-vLLM convert the HF safetensors on the fly.
  #
  # Why this exists. Production had the trainer reading an Orbax checkpoint and
  # the sampler reading `Qwen/Qwen3.5-35B-A3B` from HF -- two INDEPENDENT
  # conversion paths into the two engines. Conversion is a real transformation
  # (fused QKV packing, expert stacking order, SwiGLU gate/up order, bf16
  # rounding order), so any disagreement shows up as a small persistent policy
  # gap. And the mean log ratio measures exactly that gap: sampling x from the
  # sampler p_s and scoring under the trainer p_t gives
  #     E[log p_t(x) - log p_s(x)] = -KL(p_s || p_t) <= 0,
  # which is why seq_geomean has been below 1.0 in every run and why no sequence
  # ever lands above the band. Production sat at KL ~= 0.07 nats/token at step 0,
  # before any optimizer update; the offline harness, loading ONE checkpoint into
  # both engines, measures 0.003. The band [0.999, 1.002] permits 0.002.
  #
  # Pass the UNSCANNED layout here and the SCANNED layout to the trainer. Both
  # layouts of the golden checkpoint live in the same bucket and come from one
  # conversion (maxtext tests/end_to_end/.../2_test_qwen3.5_35b_a3b.sh:36-37).
  # They are not interchangeable: an unscanned model cannot read a scanned
  # checkpoint, it dies with "Checkpoint structure mismatch: 67 of 70 model
  # parameter paths were not found".
  #
  # The five checkpoint_* / *checkpointing flags are not optional -- without
  # enable_checkpointing the loader ignores load_parameters_path entirely. Values
  # mirror the reference harness (maxtext tools/rl_logprob_parity,
  # compare_trainer_sampler.py:248-269), which is the configuration that measured
  # the 0.003 floor.
  rollout_ckpt = os.environ.get("ROLLOUT_MAXTEXT_CKPT", "")
  if rollout_ckpt:
    overrides.update({
        "load_parameters_path": rollout_ckpt,
        "scan_layers": False,
        "enable_checkpointing": True,
        "async_checkpointing": False,
        "checkpoint_storage_use_ocdbt": True,
        "checkpoint_storage_use_zarr3": True,
        "convert_checkpoint_if_possible": False,
    })
    logging.info(
        "rollout sampler loading MaxText checkpoint %s (unscanned) instead of"
        " converting HF weights; trainer must load the SCANNED layout of the"
        " same checkpoint or the two engines are still different policies.",
        rollout_ckpt,
    )
  return {"maxtext_config": overrides}


def build_maxtext_config(
    model_name: str,
    worker_id: str = "",
    train_micro_batch_size: int = 1,
    mesh_fsdp: int = 1,
    mesh_tp: int = 1,
    mesh_expert: int = 1,
    num_devices: int = 1,
    max_prompt_length: int = 512,
    max_response_length: int = 128,
    learning_rate: float = 1e-5,
    warmup_steps_fraction: float = 0.0,
    load_parameters_path: str = "",
    padded_moe_mlp_dim: int = 0,
    base_output_directory: str = "",
    gradient_accumulation_steps: int = 1,
    checkpointing_options: Any = None,
    *,
    base_num_kv_heads: int = 0,
    kv_tp_size: int = 0,
    moe_mlp_tp_size: int = 0,
    rollout_mesh_tp: int = 0,
    prefuse_moe_weights: bool = False,
    use_weight_converter: bool = True,
) -> Any:
  """Builds the MaxText HyperParameters the training engine runs on."""
  pyconfig, _, _ = maxtext_modules()

  # KV_TP_SIZE env override.
  #
  # This exists because of a packaging constraint, not a design preference.
  # `kv_tp_size` must equal the ROLLOUT's tp * ep: vLLM replicates KV heads up
  # to tp*ep when the model has fewer, and Raiden pairs tensors by name, so a
  # trainer that builds the native head count fails weight-sync preflight with
  #   'decoder.layers.N.attention.attention.key.kernel' global shape differs:
  #   source (2048, 2, 256), destination (2048, 4, 256)
  # The natural channel is `--kv_tp_size` on run_trainer_node.py, but the copy
  # baked into TUNIX_IMAGE predates that flag and run_trainer_node.py cannot be
  # injected (its optimizer flag spellings have diverged from this checkout, and
  # injecting it kills the trainer at argparse). THIS file is injected, so an
  # env var is the only channel that reaches the trainer without an image
  # rebuild.
  #
  # Only needed when the rollout runs ep > 1. Leave unset for ep == 1, where the
  # rollout_mesh_tp fallback below is already correct.
  env_kv_tp_size = os.environ.get("KV_TP_SIZE", "").strip()
  if env_kv_tp_size:
    try:
      parsed_kv_tp_size = int(env_kv_tp_size)
    except ValueError as e:
      # Fail closed. A typo here silently reverts to the rollout_mesh_tp
      # fallback, which is wrong for ep > 1 and surfaces much later as an
      # opaque shape mismatch during weight sync.
      raise ValueError(
          f"KV_TP_SIZE must be an integer, got {env_kv_tp_size!r}"
      ) from e
    if parsed_kv_tp_size < 0:
      raise ValueError(
          f"KV_TP_SIZE must be non-negative, got {parsed_kv_tp_size}"
      )
    logging.info(
        "Overriding kv_tp_size from %d to KV_TP_SIZE=%d (env)",
        kv_tp_size,
        parsed_kv_tp_size,
    )
    kv_tp_size = parsed_kv_tp_size

  # Backward compatibility: if rollout_mesh_tp was provided, default kv_tp_size and moe_mlp_tp_size
  if rollout_mesh_tp > 0:
    if kv_tp_size == 0:
      logging.info(
          "Overriding kv_tp_size from 0 to rollout_mesh_tp=%d", rollout_mesh_tp
      )
      kv_tp_size = rollout_mesh_tp
    if moe_mlp_tp_size == 0:
      logging.info(
          "Overriding moe_mlp_tp_size from 0 to rollout_mesh_tp=%d",
          rollout_mesh_tp,
      )
      moe_mlp_tp_size = rollout_mesh_tp


  if padded_moe_mlp_dim < 0:
    raise ValueError(
        f"padded_moe_mlp_dim must be non-negative, got {padded_moe_mlp_dim}"
    )
  if base_num_kv_heads < 0:
    raise ValueError(
        f"base_num_kv_heads must be non-negative, got {base_num_kv_heads}"
    )
  if kv_tp_size < 0:
    raise ValueError(f"kv_tp_size must be non-negative, got {kv_tp_size}")
  if moe_mlp_tp_size < 0:
    raise ValueError(
        f"moe_mlp_tp_size must be non-negative, got {moe_mlp_tp_size}"
    )
  if rollout_mesh_tp < 0:
    raise ValueError(
        f"rollout_mesh_tp must be non-negative, got {rollout_mesh_tp}"
    )

  if train_micro_batch_size % mesh_fsdp:
    raise ValueError(
        f"train_micro_batch_size={train_micro_batch_size} must be a multiple of "
        f"mesh_fsdp={mesh_fsdp}; MaxText shards the batch dimension across it."
    )
  per_device_batch_size = train_micro_batch_size / num_devices

  base_yml = os.path.join(
      os.path.dirname(os.path.abspath(pyconfig.__file__)), "base.yml"
  )
  if not os.path.exists(base_yml):
    raise FileNotFoundError(f"MaxText base.yml not found at {base_yml}")

  effective_kv_heads = base_num_kv_heads
  effective_padded_moe_mlp_dim = padded_moe_mlp_dim

  # Determine if we need to inspect the model's YAML configuration
  needs_model_yml = (effective_kv_heads <= 0 and kv_tp_size > 0) or (
      not effective_padded_moe_mlp_dim and moe_mlp_tp_size > 0
  )
  model_data = None
  if needs_model_yml:
    models_dir = os.path.join(os.path.dirname(base_yml), "models")
    model_yml = os.path.join(models_dir, f"{model_name}.yml")
    if os.path.exists(model_yml):
      try:
        import yaml

        with open(model_yml, "r") as f:
          data = yaml.safe_load(f)
        if isinstance(data, dict):
          model_data = data
        else:
          logging.warning(
              "Expected dict in model config %s, got %s",
              model_yml,
              type(data).__name__,
          )
      except Exception as e:
        logging.warning("Failed to load model config from %s: %s", model_yml, e)
    else:
      logging.warning("Model config file not found at %s", model_yml)

  # 1. Resolve KV head replication:
  if effective_kv_heads <= 0 and kv_tp_size > 0:
    if model_data:
      effective_kv_heads = int(model_data.get("base_num_kv_heads") or 0)
    if effective_kv_heads <= 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) requires base_num_kv_heads > 0, but could"
          f" not determine base_num_kv_heads from config or {model_name}.yml."
          " Please specify --base_num_kv_heads."
      )

  if effective_kv_heads > 0 and kv_tp_size > effective_kv_heads:
    if kv_tp_size % effective_kv_heads != 0:
      raise ValueError(
          f"kv_tp_size ({kv_tp_size}) must be cleanly divisible by "
          f"base_num_kv_heads ({effective_kv_heads})."
      )
    effective_kv_heads = kv_tp_size

  # 2. Resolve padded MoE MLP dimension before pyconfig initialization:
  if not effective_padded_moe_mlp_dim and moe_mlp_tp_size > 0:
    compute_padded_moe_mlp_dim = None
    try:
      from maxtext.integration.vllm.convert_utils import compute_padded_moe_mlp_dim
    except (ImportError, ModuleNotFoundError) as e:
      logging.warning(
          "Could not import compute_padded_moe_mlp_dim: %s. Skipping automatic"
          " MoE dimension padding.",
          e,
      )

    if compute_padded_moe_mlp_dim is not None and model_data:
      base_dim = model_data.get("base_moe_mlp_dim") or model_data.get(
          "moe_intermediate_size"
      )
      if base_dim:
        try:
          effective_padded_moe_mlp_dim = compute_padded_moe_mlp_dim(
              base_dim, moe_mlp_tp_size
          )
          logging.info(
              "Auto-computed padded_base_moe_mlp_dim=%d for moe_mlp_tp_size=%d",
              effective_padded_moe_mlp_dim,
              moe_mlp_tp_size,
          )
        except Exception as e:
          raise RuntimeError(
              "Failed to auto-compute padded_base_moe_mlp_dim for"
              f" moe_mlp_tp_size={moe_mlp_tp_size}: {e}"
          ) from e

  output_dir = base_output_directory or "/tmp/maxtext"
  argv = [
      "maxtext_trainer",
      base_yml,
      f"model_name={model_name}",
      f"run_name={worker_id or 'tunix_maxtext'}",
      f"base_output_directory={output_dir}",
  ]
  if load_parameters_path:
    argv.append(f"load_parameters_path={load_parameters_path}")
  # Checkpointing configs. `save_interval_steps=0` means "never save"
  save_interval_steps = int(
      getattr(checkpointing_options, "save_interval_steps", 0) or 0
  )
  if checkpointing_options is not None and save_interval_steps < 0:
    raise ValueError(
        "checkpoint save_interval_steps must be non-negative, got"
        f" {save_interval_steps}."
    )
  if checkpointing_options is not None and save_interval_steps > 0:
    argv.extend([
        "enable_checkpointing=True",
        f"checkpoint_period={save_interval_steps}",
        f"max_num_checkpoints_to_keep={checkpointing_options.max_to_keep}",
    ])
  elif checkpointing_options is not None:
    logging.info(
        "checkpoint save_interval_steps=0; disabling checkpoint saving."
    )
    argv.append("enable_checkpointing=False")
  else:
    argv.append("enable_checkpointing=False")
  argv.extend([
      "scan_layers=True",
      "convert_checkpoint_if_possible=False",
      "skip_jax_distributed_system=True",
      f"per_device_batch_size={per_device_batch_size}",
      f"gradient_accumulation_steps={gradient_accumulation_steps}",
      f"max_target_length={max_prompt_length + max_response_length}",
      "attention=dot_product",
      "use_tokamax_gmm=true",
      "use_gmm_v2=true",
      f"ici_fsdp_parallelism={mesh_fsdp}",
      *(
          [f"padded_base_moe_mlp_dim={effective_padded_moe_mlp_dim}"]
          if effective_padded_moe_mlp_dim
          else []
      ),
      # The vLLM rollout replicates KV heads up to kv_tp_size (tp*ep) when the
      # model has fewer -- see maxtext_vllm_adapter. Weight sync pairs by name,
      # so the trainer must build the same shape. Prefer attention DP on the
      # rollout instead, which avoids the replication entirely; this is the
      # fallback when that is not available.
      #
      # base_num_kv_heads is also defined by the model yml, and MaxText aborts
      # when a key is set by both the model config and the CLI with different
      # values (pyconfig.validate_no_keys_overridden_twice):
      #   MAXTEXT CONFIG ERROR: Keys ['base_num_kv_heads'] are overridden by
      #   both model config and CLI/kwargs with different values.
      # override_model_config=true is what lets the CLI value win. With it set,
      # pyconfig applies the model yml for every key NOT passed on the CLI
      # (pyconfig.py:487-488), so nothing else about the model changes.
      #
      # It is emitted under exactly the same condition as base_num_kv_heads so
      # runs that do not replicate KV heads (kv_tp_size <= base_num_kv_heads,
      # e.g. the ep=1 baseline) keep the strict double-override check.
      #
      # Caveat: the check is disabled for the whole run, so a future argv key
      # that also exists in the model yml would silently take the CLI value
      # instead of erroring. Verified 2026-09-18 for qwen3.5-35b-a3b that
      # base_num_kv_heads is the ONLY overlap between this argv and the model
      # yml. Re-check when adding model-level flags here.
      *(
          [
              f"base_num_kv_heads={effective_kv_heads}",
              "override_model_config=true",
          ]
          if effective_kv_heads
          else []
      ),
      f"ici_tensor_parallelism={mesh_tp}",
      f"ici_expert_parallelism={mesh_expert}",
      f"learning_rate={learning_rate}",
      f"warmup_steps_fraction={warmup_steps_fraction}",
      "learning_rate_final_fraction=1.0",
      f"adam_b1={os.environ.get('ADAM_B1', '0.9')}",
      f"adam_b2={os.environ.get('ADAM_B2', '0.999')}",
      "adam_eps=1e-8",
      f"adam_weight_decay={os.environ.get('WEIGHT_DECAY', '0.0')}",
      f"gradient_clipping_threshold={os.environ.get('MAX_GRAD_NORM', '0.125')}",
      'trainable_parameters_mask=["^(?!.*routed_experts/gate/kernel).*"]',
      # Activation dtype and matmul precision. Defaults follow the dense-GRPO
      # bringup (docs/bringup/2026-09-15-dense-grpo on jfacevedo/trellis-mlperf),
      # which established by elimination that the sampler-vs-trainer log-prob
      # divergence is dominated by *bf16 matmul rounding on the MXU*, whose
      # tiling and accumulation order are keyed on matmul shape -- so the
      # trainer's own log-probs move when only the forward micro-batch shape
      # changes (6.2x from micro-batch 1 to 2, identical data and weights).
      #
      # Two traps that cost that investigation real time, both avoided here:
      #
      #  1. `matmul_precision` ALONE is a bit-identical no-op on a bf16 model.
      #     jax.lax.Precision only affects f32 operands, and this path has bf16
      #     operands everywhere. The activations must be widened first; the
      #     precision setting then has something to act on.
      #  2. An fp32 output head alone does not work. It buys 8% and leaves the
      #     hidden-state divergence untouched. We have had logits_dot_in_fp32 /
      #     cast_logits_to_fp32 / float32_logits on below this whole time, which
      #     is exactly that 8% case. The variance is born in the transformer
      #     body.
      #
      # Measured on Qwen3-1.7B dense: |delta| bs=4 vs bs=1 goes 0.03134
      # (bf16 acts) -> 0.0000299 (fp32 acts + HIGH), ~1000x, and GRPO acceptance
      # 18.75% -> 42.71% with the trainer alone changed. fp32 WEIGHTS are not
      # needed -- bf16 weights measured marginally better than fp32 (0.0000299
      # vs 0.0000344) -- so weight memory is unchanged and only activations
      # widen. Forward cost is ~2x, which the bringup found hidden behind
      # rollout generation.
      #
      # HIGHEST is ~21x more accurate again but ~20% slower than HIGH; HIGH
      # already sits ~172x inside the nearest band edge.
      f"dtype={os.environ.get('TRAINER_ACT_DTYPE', 'float32')}",
      "weight_dtype=bfloat16",
      "grad_dtype=float32",
      f"matmul_precision={os.environ.get('TRAINER_MATMUL_PRECISION', 'high')}",
      "logits_dot_in_fp32=true",
      "cast_logits_to_fp32=true",
      "float32_logits=true",
      "float32_gate_logits=true",
      "float32_weight_sum=true",
      "enable_tensorboard=False",
      "record_internal_nn_metrics=False",
      "init_weights_seed=42",
      f"prefuse_moe_weights={prefuse_moe_weights}",
      f"use_weight_converter={use_weight_converter}",
      *(
          [
              f"rollout_tensor_parallelism={rollout_mesh_tp or kv_tp_size or moe_mlp_tp_size}"
          ]
          if (rollout_mesh_tp or kv_tp_size or moe_mlp_tp_size) > 0
          else []
      ),
  ])
  # Pathways persistence: let the TPU workers write the checkpoint themselves
  # The persistence handler rejects the OCDBT/zarr3 layout MaxText writes by
  # default (see maxtext/common/checkpoint_context.py), so both must be off
  if os.environ.get("ENABLE_PATHWAYS_PERSISTENCE", "") == "1":
    logging.info(
        "ENABLE_PATHWAYS_PERSISTENCE=1; disabling OCDBT/zarr3 so the Pathways "
        "persistence handler can save directly from the TPU workers."
    )
    argv.extend([
        "checkpoint_storage_use_ocdbt=false",
        "checkpoint_storage_use_zarr3=false",
    ])

  logging.info("MaxText config argv: %s", argv)
  return pyconfig.initialize(argv)


def create_maxtext_mesh(maxtext_config: Any) -> Any:
  """Builds the JAX device Mesh with axis names from MaxText config."""
  from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top

  _, _, m_utils = maxtext_modules()
  devices = m_utils.create_device_mesh(maxtext_config)
  return Mesh(devices, maxtext_config.mesh_axes)


def log_param_shapes(model: Any) -> None:
  """Logs parameter shapes as a sanity check that weights loaded correctly."""
  from flax import nnx  # pylint: disable=g-import-not-at-top

  flat = nnx.to_pure_dict(nnx.state(model, nnx.Param))

  def walk(node, path=""):
    if isinstance(node, dict):
      for key, value in node.items():
        yield from walk(value, f"{path}.{key}" if path else str(key))
    elif hasattr(node, "shape"):
      yield path, node.shape

  shapes = dict(walk(flat))
  for name, shape in shapes.items():
    if "wi_0" in name or "query" in name:
      logging.info("MaxText param %s shape=%s", name, shape)
  logging.info("MaxText model has %d parameter arrays.", len(shapes))


def create_maxtext_engine(
    maxtext_config: Any,
    mesh: Any,
    tokenizer_pad_id: int = 0,
    wrap_with_tunix_adapter: bool = True,
    log_shapes: bool = True,
) -> Any:
  """Builds and initializes a MaxTextTrainingEngine within the given mesh."""
  _, maxtext_engine, _ = maxtext_modules()

  with mesh:
    engine = maxtext_engine.MaxTextTrainingEngine(
        maxtext_config,
        mesh=mesh,
        wrap_with_tunix_adapter=wrap_with_tunix_adapter,
        tokenizer_pad_id=tokenizer_pad_id,
    )

  model_type = type(engine.model).__name__
  logging.info(
      "MaxText engine model: %s (pad_id=%d)", model_type, tokenizer_pad_id
  )
  if wrap_with_tunix_adapter and model_type != "TunixMaxTextAdapter":
    raise RuntimeError(
        f"Expected the engine's model to be TunixMaxTextAdapter, got {model_type}."
    )
  if log_shapes:
    log_param_shapes(engine.model)

  return engine


_SLOT_GROUP_HELPER = '''def _routed_experts_slot_layout(runner):
    """Returns ``(kv_cache_group_id, block_size)`` keying the routed-experts
    slot buffer.

    This MUST agree with vLLM's ``RoutedExpertsManager``, which keys its buffer
    by the FULL-ATTENTION group (``get_routed_experts_attn_gid``) and by that
    group's ``block_size`` -- not by group 0, and not by the global
    ``cache_config.block_size``.

    A hybrid model has more than one KV-cache group and the full-attention
    group is not necessarily group 0. Qwen3.5-35B-A3B is such a model: its
    Gated Delta Network layers form their own group. If the write side keys off
    group 0 while the manager reads off ``attn_gid``, writes and reads address
    disjoint slot spaces and the read returns zero-initialised memory. That
    never raises -- expert id 0 is in range -- so it decodes as "every token
    routed to expert 0", and it damages only the PROMPT, because the decode
    path reads the step's fresh routing tensor and never consults the buffer.

    Resolution is delegated to vLLM's own helper so the two sides cannot drift.
    """
    global _ROUTED_EXPERTS_SLOT_LAYOUT_LOGGED

    default_block_size = getattr(runner, "block_size", 0)
    kv_cache_config = getattr(runner, "kv_cache_config", None)
    if kv_cache_config is None:
        return 0, default_block_size

    try:
        from vllm.model_executor.layers.fused_moe.routed_experts_capturer import get_routed_experts_attn_gid
        gid = get_routed_experts_attn_gid(kv_cache_config)
    except Exception:
        return 0, default_block_size

    groups = kv_cache_config.kv_cache_groups
    block_size = getattr(groups[gid].kv_cache_spec, "block_size",
                         default_block_size) or default_block_size

    if not _ROUTED_EXPERTS_SLOT_LAYOUT_LOGGED:
        _ROUTED_EXPERTS_SLOT_LAYOUT_LOGGED = True
        logger.info(
            "[routed-experts] slot layout: attn_gid=%d of %d KV group(s), "
            "block_size=%d (global cache block_size=%d). Writes previously "
            "hardcoded group 0; a mismatch here silently zero-fills prompt "
            "routing.", gid, len(groups), block_size, default_block_size)

    return gid, block_size


_ROUTED_EXPERTS_SLOT_LAYOUT_LOGGED = False


'''

_SLOT_GROUP_PATCHES = [
    (
        "slot_layout_helper",
        "_routed_experts_slot_layout",
        (
            "def _reconstruct_slots_for_request(\n"
            "    req_state: CachedRequestState,\n"
            "    num_tokens: int,\n"
            "    block_size: int,\n"
            "    start_pos: int,\n"
            ") -> np.ndarray:"
        ),
        (
            _SLOT_GROUP_HELPER
            + "def _reconstruct_slots_for_request(\n"
            "    req_state: CachedRequestState,\n"
            "    num_tokens: int,\n"
            "    block_size: int,\n"
            "    start_pos: int,\n"
            "    kv_group_id: int = 0,\n"
            ") -> np.ndarray:"
        ),
    ),
    (
        "slot_group_select",
        "kv_group_id < len(req_state.block_ids)",
        "    block_ids = req_state.block_ids[0] if req_state.block_ids else []",
        (
            "    # Index the group the reader reads, not group 0. On a hybrid\n"
            "    # model those differ, and keying off the wrong one points\n"
            "    # writes at a slot space the reader never looks at.\n"
            "    if req_state.block_ids and kv_group_id < len(req_state.block_ids):\n"
            "        block_ids = req_state.block_ids[kv_group_id]\n"
            "    else:\n"
            "        block_ids = []"
        ),
    ),
    (
        "reconstruct_layout",
        "kv_group_id, block_size = _routed_experts_slot_layout(runner)",
        "    block_size = runner.block_size",
        "    kv_group_id, block_size = _routed_experts_slot_layout(runner)",
    ),
    (
        "reconstruct_callsite",
        "start_pos=chunk_start[req_id],",
        "                        start_pos=chunk_start[req_id])",
        (
            "                        start_pos=chunk_start[req_id],\n"
            "                        kv_group_id=kv_group_id)"
        ),
    ),
    (
        # Anchored on the following parameter, because the helper inserted by
        # slot_layout_helper also contains "kv_group_id: int = 0," and a bare
        # sentinel matches it, silently skipping this patch and leaving
        # kv_group_id undefined in this function.
        "decode_param",
        "    kv_group_id: int = 0,\n    scheduler_output:",
        "    block_size: int = 0,\n    scheduler_output:",
        "    block_size: int = 0,\n    kv_group_id: int = 0,\n    scheduler_output:",
    ),
    (
        "decode_callsite",
        "start_pos=req_state.num_computed_tokens,",
        "                    start_pos=req_state.num_computed_tokens)",
        (
            "                    start_pos=req_state.num_computed_tokens,\n"
            "                    kv_group_id=kv_group_id)"
        ),
    ),
    (
        "async_caller",
        "kv_group_id=(_routed_experts_slot_layout(self._runner)[0]",
        '            block_size=getattr(self._runner, "block_size", 0),',
        (
            "            block_size=(_routed_experts_slot_layout(self._runner)[1]\n"
            "                        if self._runner else 0),\n"
            "            kv_group_id=(_routed_experts_slot_layout(self._runner)[0]\n"
            "                         if self._runner else 0),"
        ),
    ),
    (
        "sync_caller",
        "kv_group_id=_routed_experts_slot_layout(self)[0],",
        "            block_size=self.block_size,",
        (
            "            block_size=_routed_experts_slot_layout(self)[1],\n"
            "            kv_group_id=_routed_experts_slot_layout(self)[0],"
        ),
    ),
]


def apply_routed_experts_slot_group_fix(path: str | None = None) -> None:
  """Keys routed-experts slot writes off the full-attention KV cache group.

  WHAT IT FIXES. tpu-inference derives the slot mapping it WRITES from
  ``req_state.block_ids[0]`` and the global ``cache_config.block_size``, while
  vLLM's ``RoutedExpertsManager`` READS that buffer using the full-attention
  group's block IDs and that group's block size. On a hybrid model --
  Qwen3.5-35B-A3B carries a Gated Delta Network group alongside its attention
  group -- those are different groups, so the routing captured for the PROMPT
  is written to slots nobody reads and the prompt reads back as all zeros.
  Expert id 0 is a legal expert (num_experts=256 exactly saturates uint8), so
  nothing downstream rejects it: the trainer is handed "every prompt token
  routed to expert 0 in all 40 layers".

  Only the prompt is affected, because the decode path reads the step's fresh
  routing tensor and never consults the slot buffer.

  Measured cost of not fixing it: replay ON is worse than replay OFF
  (seq_geomean 0.88-0.94 vs 0.997-0.999), and before the routes were screened
  for corruption it collapsed to NaN at step 3.

  WHY A TEXT PATCH RATHER THAN FILE INJECTION. tpu_runner.py is ~160KB; gzipped
  and base64'd it is ~47KB, and the rollout startup command already carries
  ~86KB of injected payloads against the 128KB (MAX_ARG_STRLEN) per-argument
  execve limit. Injecting the whole file would break the launch.

  Raises rather than warns on a missed anchor: a silently unpatched rollout
  still produces a plausible-looking run, and that is precisely the failure
  mode this fix exists to remove.

  Args:
    path: tpu_runner.py to patch. Defaults to the installed tpu_inference.
  """
  if path is None:
    try:
      import tpu_inference  # pylint: disable=g-import-not-at-top

      path = os.path.join(
          os.path.dirname(tpu_inference.__file__), "runner", "tpu_runner.py"
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(
          f"[re-slot-fix] tpu_inference not importable ({e}); skipping."
          " Expected on the trainer; a BUG on the rollout node.",
          flush=True,
      )
      return

  if not os.path.exists(path):
    raise RuntimeError(f"[re-slot-fix] {path} does not exist")

  with open(path, encoding="utf-8") as f:
    txt = f.read()

  modified = False
  problems = []
  for name, check_str, needle, replacement in _SLOT_GROUP_PATCHES:
    if check_str in txt:
      print(f"[re-slot-fix] {name} already present", flush=True)
      continue
    found = txt.count(needle)
    if found != 1:
      problems.append(f"{name} (needle matched {found} times, expected 1)")
      continue
    txt = txt.replace(needle, replacement, 1)
    modified = True
    print(f"[re-slot-fix] PATCHED {name}", flush=True)

  if problems:
    raise RuntimeError(
        "[re-slot-fix] refusing to continue with a partially applied patch;"
        f" {path} does not match the expected source: {problems}. The"
        " routed-experts slot mapping would stay wrong and the run would"
        " silently train on prompt routing that is entirely expert 0."
    )

  if modified:
    with open(path, "w", encoding="utf-8") as f:
      f.write(txt)
    print(f"[re-slot-fix] wrote {path}", flush=True)
