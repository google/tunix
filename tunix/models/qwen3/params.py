# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for loading and converting Qwen3 PT weights."""

from collections.abc import Mapping, Sequence
import time
from typing import Any

import jax
import jax.numpy as jnp
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
from tunix.models.qwen3 import model as model_lib

from_hf_impl = hf_mapping.from_hf_impl
to_hf_impl = hf_mapping.to_hf_impl


_FROM_HF_BASE_JSON = """[
  {
    "source_pattern": ["model.embed_tokens.weight"],
    "target_pattern": ["embedder.input_embedding"],
    "ops": []
  },
  {
    "source_pattern": ["model.norm.weight"],
    "target_pattern": ["final_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.input_layernorm.weight"],
    "target_pattern": ["layers.{i}.input_layernorm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.post_attention_layernorm.weight"],
    "target_pattern": ["layers.{i}.post_attention_layernorm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.q_norm.weight"],
    "target_pattern": ["layers.{i}.attn.q_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.k_norm.weight"],
    "target_pattern": ["layers.{i}.attn.k_norm.w"],
    "ops": []
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.q_proj.weight"],
    "target_pattern": ["layers.{i}.attn.q_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $nh], "dst_shape": [$d, $n, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.k_proj.weight"],
    "target_pattern": ["layers.{i}.attn.k_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.v_proj.weight"],
    "target_pattern": ["layers.{i}.attn.v_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$d, $kh], "dst_shape": [$d, $k, $h]}}
    ]
  },
  {
    "source_pattern": ["model.layers.{i}.self_attn.o_proj.weight"],
    "target_pattern": ["layers.{i}.attn.o_proj.w"],
    "ops": [
      {"transpose": {"axes": [1, 0]}},
      {"reshape": {"src_shape": [$nh, $d], "dst_shape": [$n, $h, $d]}}
    ]
  }
]"""

_FROM_HF_LM_HEAD_JSON = """[
  {
    "source_pattern": ["lm_head.weight"],
    "target_pattern": ["lm_head.w"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

_FROM_HF_DENSE_MLP_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.mlp.gate_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.gate_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.up_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.up_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.down_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.down_proj.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  }
]"""

_FROM_HF_MOE_MLP_JSON = """[
  {
    "source_pattern": ["model.layers.{i}.mlp.gate.weight"],
    "target_pattern": ["layers.{i}.mlp.router.kernel"],
    "ops": [{"transpose": {"axes": [1, 0]}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.experts.{e}.gate_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.gate_proj"],
    "ops": [{"transpose": {"axes": [1, 0]}}, {"stack": {"axis": 0}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.experts.{e}.up_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.up_proj"],
    "ops": [{"transpose": {"axes": [1, 0]}}, {"stack": {"axis": 0}}]
  },
  {
    "source_pattern": ["model.layers.{i}.mlp.experts.{e}.down_proj.weight"],
    "target_pattern": ["layers.{i}.mlp.down_proj"],
    "ops": [{"transpose": {"axes": [1, 0]}}, {"stack": {"axis": 0}}]
  }
]"""


def build_from_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `HF -> Tunix` manifest for `cfg`."""
  d, n, k, h = cfg.embed_dim, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
  num_experts = 0 if cfg.num_experts is None else cfg.num_experts
  templates = hf_mapping.render_json_templates(
      _FROM_HF_BASE_JSON,
      "" if cfg.use_tied_embedding else _FROM_HF_LM_HEAD_JSON,
      _FROM_HF_MOE_MLP_JSON if num_experts > 0 else _FROM_HF_DENSE_MLP_JSON,
      d=d,
      n=n,
      k=k,
      h=h,
      nh=n * h,
      kh=k * h,
  )
  return hf_mapping.generate_manifest(
      model_type="qwen3",
      template_rules=templates,
      axes={"i": cfg.num_layers, "e": num_experts},
      dropped_keys=["lm_head.weight"] if cfg.use_tied_embedding else [],
  )


def build_to_hf_manifest(cfg: model_lib.ModelConfig) -> dict[str, Any]:
  """Builds the declarative PrimOp `Tunix -> HF` manifest for `cfg`."""
  return hf_mapping.invert_manifest(build_from_hf_manifest(cfg))


def export_to_hf_state_dict(
    model: model_lib.Qwen3, cfg: model_lib.ModelConfig
) -> dict[str, jax.Array]:
  """Exports a Tunix `Qwen3` model into a HuggingFace state dict."""
  flat_state = safetensors_loader.flatten_model_state(model)
  return hf_mapping.to_hf_impl(build_to_hf_manifest(cfg), flat_state)


def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
    mode: str = "auto",
) -> model_lib.Qwen3:
  """Load tensors from the safetensors file and create a Qwen3 model."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=model_lib.Qwen3,
      config=config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
      manifest=build_from_hf_manifest(config),
  )


def _make_tensor_spec(
    flat_state: Mapping[str, jax.Array],
    source_key: str,
    hf_key: str,
    *,
    shape_2d: Sequence[int] | None = None,
    transpose_axes: Sequence[int] | None = None,
    unstack_axis: int | None = None,
    lora_scaling: float = 1.0,
    itemsize: int | None = None,
) -> safetensors_saver.TensorExportSpec:
  """Constructs a TensorExportSpec from flat model state."""
  arr = flat_state[source_key]
  lora_a_key, lora_b_key = f"{source_key}_lora_a", f"{source_key}_lora_b"
  has_lora = lora_a_key in flat_state and lora_b_key in flat_state
  return safetensors_saver.TensorExportSpec(
      source_key=source_key,
      hf_key=hf_key,
      shape_2d=tuple(shape_2d) if shape_2d is not None else None,
      transpose_axes=(
          tuple(transpose_axes) if transpose_axes is not None else None
      ),
      unstack_axis=unstack_axis,
      nbytes=int(arr.size) * (
          arr.dtype.itemsize if itemsize is None else itemsize
      ),
      lora_a_key=lora_a_key if has_lora else None,
      lora_b_key=lora_b_key if has_lora else None,
      lora_scaling=lora_scaling,
      is_replicated=safetensors_saver.infer_is_replicated(arr),
  )


_QWEN3_INV_RULES_CACHE: dict[Any, tuple[dict[str, Any], ...]] = {}


_EXPORT_OP_ORDER = ("reshape", "unstack", "transpose")


def _parse_export_rule_ops(
    src_key: str,
    rule_ops: Sequence[Mapping[str, Any]],
) -> tuple[Sequence[int] | None, tuple[int, ...] | None, int | None]:
  """Parses and validates a Qwen3 inverse-manifest op sequence."""
  if any(len(step) != 1 for step in rule_ops):
    raise ValueError(
        f"Expected single-op steps in manifest rule {src_key!r}: {rule_ops!r}."
    )
  op_names = [next(iter(step)) for step in rule_ops]
  unknown = [name for name in op_names if name not in _EXPORT_OP_ORDER]
  if unknown:
    raise ValueError(
        f"Unsupported manifest op {unknown[0]!r} in rule {src_key!r}."
    )
  if len(set(op_names)) != len(op_names) or op_names != sorted(
      op_names, key=_EXPORT_OP_ORDER.index
  ):
    raise ValueError(
        f"Unsupported op order in manifest rule {src_key!r}: {rule_ops!r}."
    )
  ops_by_name = {
      next(iter(step)): next(iter(step.values())) for step in rule_ops
  }
  shape_2d = (
      ops_by_name["reshape"]["dst_shape"] if "reshape" in ops_by_name else None
  )
  unstack_axis: int | None = None
  if "unstack" in ops_by_name:
    if ops_by_name["unstack"]["axis"] != 0:
      raise ValueError(
          f"Unsupported unstack op in manifest rule {src_key!r}: {rule_ops!r}."
      )
    unstack_axis = 0
  axes: tuple[int, ...] | None = None
  if "transpose" in ops_by_name:
    raw_axes = tuple(ops_by_name["transpose"]["axes"])
    if unstack_axis is not None and raw_axes != (1, 0):
      raise ValueError(
          f"Unsupported stacked transpose {raw_axes!r} for {src_key!r}."
      )
    axes = (0, 2, 1) if unstack_axis is not None else raw_axes
  return shape_2d, axes, unstack_axis


def build_qwen3_export_specs(
    model: model_lib.Qwen3,
    *,
    rank: int = 1,
    alpha: float = 1.0,
    output_dtype: jnp.dtype | None = None,
) -> tuple[
    dict[str, jax.Array],
    list[safetensors_saver.TensorExportSpec],
    dict[str, int],
]:
  """Builds flat state and Subtree all-to-all TensorExportSpecs for Qwen3."""
  flat_state = safetensors_saver.extract_flat_state_dict(model)
  itemsize = jnp.dtype(output_dtype).itemsize if output_dtype else None
  lora_scaling = float(alpha) / float(rank)
  cfg_key = repr(model.config)
  if cfg_key not in _QWEN3_INV_RULES_CACHE:
    inv_manifest = build_to_hf_manifest(model.config)
    _QWEN3_INV_RULES_CACHE[cfg_key] = tuple(inv_manifest["rules"])
  specs: list[safetensors_saver.TensorExportSpec] = []
  unstack_counts: dict[str, int] = {}
  for rule in _QWEN3_INV_RULES_CACHE[cfg_key]:
    src_key, tgt_keys = rule["source_keys"][0], rule["target_keys"]
    if src_key not in flat_state:
      raise KeyError(
          f"Missing expected Qwen3 parameter {src_key!r} in model state."
      )
    shape_2d, axes, unstack_axis = _parse_export_rule_ops(src_key, rule["ops"])
    hf_key = tgt_keys[0]
    if unstack_axis is not None:
      hf_key = hf_key.replace(".experts.0.", ".experts.{expert}.")
      unstack_counts[src_key] = len(tgt_keys)
    specs.append(
        _make_tensor_spec(
            flat_state,
            src_key,
            hf_key,
            shape_2d=shape_2d,
            transpose_axes=axes,
            unstack_axis=unstack_axis,
            lora_scaling=lora_scaling,
            itemsize=itemsize,
        )
    )
  return flat_state, specs, unstack_counts


def export_qwen3_as_safetensors(
    model: model_lib.Qwen3,
    output_dir: str,
    *,
    local_model_path: str | None = None,
    rank: int = 1,
    alpha: float = 1.0,
    num_ranks: int = 1,
    max_bucket_bytes: int | None = None,
    output_dtype: jnp.dtype | None = None,
    devices: Sequence[jax.Device] | None = None,
    async_write: bool = False,
    per_wave_shards: bool | None = None,
    write_buffer_bytes: int | None = None,
    max_write_threads: int | None = None,
    defer_contiguous: bool = True,
) -> safetensors_saver.SaveMetrics:
  """Exports Qwen3 (Dense or MoE, base or LoRA) via Subtree all-to-all plan."""
  t_plan = time.perf_counter()
  backend = safetensors_saver.resolve_storage_backend(output_dir)
  use_wave_shards = (
      backend.default_per_wave_shards
      if per_wave_shards is None
      else per_wave_shards
  )
  flat_state, specs, unstack_counts = build_qwen3_export_specs(
      model, rank=rank, alpha=alpha, output_dtype=output_dtype
  )
  plan = safetensors_saver.build_save_plan(
      specs,
      num_ranks=num_ranks,
      max_bucket_bytes=max_bucket_bytes,
      unstack_counts=unstack_counts,
      per_wave_shards=use_wave_shards,
  )
  plan_sec = time.perf_counter() - t_plan
  return safetensors_saver.export_safetensors_with_plan(
      flat_state,
      plan,
      output_dir,
      local_model_path=local_model_path,
      output_dtype=output_dtype,
      devices=devices,
      plan_time_sec=plan_sec,
      async_write=async_write,
      write_buffer_bytes=write_buffer_bytes,
      max_write_threads=max_write_threads,
      defer_contiguous=defer_contiguous,
  )


def save_lora_merged_model_as_safetensors(
    local_model_path: str | None,
    output_dir: str,
    lora_model: model_lib.Qwen3,
    rank: int = 1,
    alpha: float = 1.0,
    *,
    num_ranks: int = 1,
    max_bucket_bytes: int | None = None,
    output_dtype: jnp.dtype | None = None,
    devices: Sequence[jax.Device] | None = None,
    async_write: bool = False,
    per_wave_shards: bool | None = None,
    write_buffer_bytes: int | None = None,
    max_write_threads: int | None = None,
    defer_contiguous: bool = True,
) -> safetensors_saver.SaveMetrics:
  """Saves in-memory Qwen3 weights (with optional LoRA merged) as safetensors.

  Exports `lora_model` directly via `export_qwen3_as_safetensors` in model dtype
  or `output_dtype`; `local_model_path`, when set, is only read on process 0 to
  copy non-safetensors metadata/tokenizer files rather than loading base weights
  from disk.

  Args:
    local_model_path: Optional source checkpoint dir for metadata/tokenizer
      files.
    output_dir: Destination checkpoint directory (local, `gs://`, or remote).
    lora_model: Qwen3 model instance (base or LoRA-adapted).
    rank: LoRA rank used to compute `alpha / rank` scaling.
    alpha: LoRA alpha used to compute `alpha / rank` scaling.
    num_ranks: Number of output writer ranks.
    max_bucket_bytes: Optional per-wave byte cap per rank.
    output_dtype: Optional cast dtype for exported tensors.
    devices: Optional explicit device sequence (defaults to `jax.devices()`).
    async_write: Whether to overlap wave host writes asynchronously.
    per_wave_shards: Whether to emit one shard per `(wave, rank)`.
    write_buffer_bytes: Optional per-writer byte buffer size (`0` unbuffered).
    max_write_threads: Optional cap on concurrent shard writer threads.
    defer_contiguous: Whether to defer contiguous host copies until write time.

  Returns:
    `SaveMetrics` describing exported byte volume, timings, and shard paths.
  """
  return export_qwen3_as_safetensors(
      lora_model,
      output_dir,
      local_model_path=local_model_path,
      rank=rank,
      alpha=alpha,
      num_ranks=num_ranks,
      max_bucket_bytes=max_bucket_bytes,
      output_dtype=output_dtype,
      devices=devices,
      async_write=async_write,
      per_wave_shards=per_wave_shards,
      write_buffer_bytes=write_buffer_bytes,
      max_write_threads=max_write_threads,
      defer_contiguous=defer_contiguous,
  )
