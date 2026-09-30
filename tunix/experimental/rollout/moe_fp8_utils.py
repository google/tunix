# Copyright 2026 Google LLC
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

"""Utilities for running MoE FP8 weights on rollout samplers with a BF16 trainer."""

from __future__ import annotations

import functools
import inspect
import logging
import os
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def is_rollout_moe_fp8_enabled() -> bool:
  """Returns True when `ROLLOUT_MOE_FP8` is enabled in the environment."""
  val = os.environ.get("ROLLOUT_MOE_FP8", "").strip().lower()
  return val in _TRUTHY_ENV_VALUES


@jax.jit
def _quantize_3d_axis1_jit(w: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Per-channel symmetric absmax quantization along reduction axis 1."""
  finfo = jnp.finfo(jnp.float8_e4m3fn)
  dtype_max = jnp.asarray(finfo.max, dtype=jnp.float32)
  w_f32 = w.astype(jnp.float32)
  abs_max = jnp.max(jnp.abs(w_f32), axis=1, keepdims=True)
  scale = abs_max / dtype_max
  inv_scale = jnp.where(scale == 0.0, 0.0, 1.0 / scale)
  q = jnp.clip(w_f32 * inv_scale, -dtype_max, dtype_max).astype(
      jnp.float8_e4m3fn
  )
  return q, scale


def quantize_moe_weight_3d(w: jax.Array) -> tuple[jax.Array, jax.Array]:
  """Quantizes a 3D MoE weight `(E, K, N)` to `(q_w, scale)` along `axis=1`.

  Args:
    w: 3D JAX array of shape `(num_experts, in_features, out_features)` in
      bfloat16/float32.

  Returns:
    Tuple `(q_w, scale)` where `q_w` has dtype `float8_e4m3fn` and shape
    `(E, K, N)` with the same sharding as `w`, and `scale` has dtype `float32`
    and shape `(E, 1, N)` sharded along the expert axis `(spec[0], None, None)`.
  """
  if w.ndim != 3:
    raise ValueError(f"Expected 3D MoE weight (E, K, N), got shape {w.shape}")

  sharding = getattr(w, "sharding", None)
  if isinstance(sharding, jax.sharding.NamedSharding):
    spec = sharding.spec
    expert_axis = spec[0] if len(spec) > 0 else None
    scale_sharding = jax.sharding.NamedSharding(
        sharding.mesh,
        jax.sharding.PartitionSpec(expert_axis, None, None),
    )
    q, scale = _quantize_3d_axis1_jit(w)
    if getattr(q, "sharding", None) != sharding:
      q = jax.lax.with_sharding_constraint(q, sharding)
    if getattr(scale, "sharding", None) != scale_sharding:
      scale = jax.lax.with_sharding_constraint(scale, scale_sharding)
    return q, scale

  return _quantize_3d_axis1_jit(w)


_MOE_BLOCK_KEYS = frozenset({"MoeBlock_0", "moe_block", "routed_experts"})


def _set_param_attr(module: Any, name: str, param: nnx.Param) -> None:
  """Assigns `param` onto `module`, overriding static status in `nnx.Module`."""
  if isinstance(module, nnx.Module):
    setattr(module, name, nnx.data(param))
  else:
    setattr(module, name, param)


def _get_serve_fp8_quant() -> Any:
  """Returns MaxText's `ServeFp8WeightQuantization()` instance (or stub)."""
  try:
    from maxtext.layers import quantizations  # pylint: disable=g-import-not-at-top

    if hasattr(quantizations, "ServeFp8WeightQuantization"):
      return quantizations.ServeFp8WeightQuantization()
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  class _FallbackServeFp8Quant:
    quant_mode = "serve_fp8"
    quant_dg = True

  return _FallbackServeFp8Quant()


def _iter_routed_moe_modules(root: Any):
  """Yields all `RoutedMoE` modules reachable from `root`."""
  visited: set[int] = set()

  def _walk(obj: Any):
    obj_id = id(obj)
    if obj_id in visited:
      return
    visited.add(obj_id)

    if (
        type(obj).__name__ == "RoutedMoE"
        and hasattr(obj, "wi")
        and hasattr(obj, "wo")
    ):
      yield obj
      return

    if isinstance(obj, nnx.Module):
      for _, val in vars(obj).items():
        yield from _walk(val)
    elif isinstance(obj, (list, tuple)):
      for item in obj:
        yield from _walk(item)
    elif isinstance(obj, dict):
      for item in obj.values():
        yield from _walk(item)

  yield from _walk(root)


def _unwrap_array(val: Any) -> Any:
  if isinstance(val, nnx.Variable):
    return val[...]
  return getattr(val, "value", val)


def quantize_routed_moe_modules_in_place(model: Any) -> int:
  """Quantizes all `RoutedMoE` modules in `model` in-place to FP8 + F32 scales.

  Updates each `RoutedMoE` module `m`:
  - `m.wi`: `nnx.Param` of shape `(E, K, N)` `float8_e4m3fn`
  - `m.wo`: `nnx.Param` of shape `(E, K, N)` `float8_e4m3fn`
  - `m.wi_scale`: `nnx.Param` of shape `(E, 1, N)` `float32`
  - `m.wo_scale`: `nnx.Param` of shape `(E, 1, N)` `float32`
  - `m.weight_dtype`: `jnp.float8_e4m3fn`
  - `m.quant`: `ServeFp8WeightQuantization()`

  Args:
    model: Root MaxText `nnx.Module` on the sampler worker.

  Returns:
    Number of `RoutedMoE` modules quantized (or already in FP8).
  """
  serve_fp8_quant = _get_serve_fp8_quant()
  count = 0
  for moe in _iter_routed_moe_modules(model):
    wi_param = getattr(moe, "wi", None)
    wo_param = getattr(moe, "wo", None)
    if wi_param is None or wo_param is None:
      continue
    wi_val = _unwrap_array(wi_param)
    wo_val = _unwrap_array(wo_param)

    if (
        getattr(wi_val, "dtype", None) == jnp.float8_e4m3fn
        and getattr(wo_val, "dtype", None) == jnp.float8_e4m3fn
        and getattr(moe, "wi_scale", None) is not None
        and getattr(moe, "wo_scale", None) is not None
    ):
      moe.weight_dtype = jnp.float8_e4m3fn
      moe.quant = serve_fp8_quant
      count += 1
      continue

    wi_q, wi_scale = quantize_moe_weight_3d(wi_val)
    wo_q, wo_scale = quantize_moe_weight_3d(wo_val)

    wi_axes = getattr(moe, "wi_kernel_axes", (None, None, None))
    wo_axes = getattr(moe, "wo_kernel_axes", (None, None, None))
    wi_scale_axes = (wi_axes[0] if wi_axes else None, None, None)
    wo_scale_axes = (wo_axes[0] if wo_axes else None, None, None)

    _set_param_attr(moe, "wi", nnx.Param(wi_q, out_sharding=wi_axes))
    _set_param_attr(moe, "wo", nnx.Param(wo_q, out_sharding=wo_axes))
    _set_param_attr(
        moe, "wi_scale", nnx.Param(wi_scale, out_sharding=wi_scale_axes)
    )
    _set_param_attr(
        moe, "wo_scale", nnx.Param(wo_scale, out_sharding=wo_scale_axes)
    )
    moe.weight_dtype = jnp.float8_e4m3fn
    moe.quant = serve_fp8_quant
    count += 1

  if count > 0:
    logger.info(
        "[MoE-FP8] Quantized %d RoutedMoE modules in-place to float8_e4m3fn.",
        count,
    )
  return count


def patch_sampler_moe_fp8() -> int:
  """Quantizes active `MaxTextForCausalLM` model and patches `load_weights`.

  When called from `patch_raiden_worker_sync()` (which runs at the end of
  `MaxTextForCausalLM.load_weights` before `nnx.split`), this immediately
  converts the active model's `RoutedMoE` layers to FP8 so that `runner.state`
  and `RaidenSynchronizer.bind` see `(wi, wi_scale, wo, wo_scale)`.

  Returns:
    Number of `RoutedMoE` modules quantized in the active caller frame.
  """
  if not is_rollout_moe_fp8_enabled():
    return 0

  quantized_now = 0
  frame = inspect.currentframe()
  try:
    caller = frame.f_back if frame is not None else None
    while caller is not None:
      self_obj = caller.f_locals.get("self")
      if (
          self_obj is not None
          and getattr(self_obj, "model", None) is not None
          and (
              type(self_obj).__name__ == "MaxTextForCausalLM"
              or isinstance(getattr(self_obj, "model", None), nnx.Module)
          )
      ):
        quantized_now = quantize_routed_moe_modules_in_place(self_obj.model)
        if isinstance(self_obj, nnx.Module):
          self_obj.model = nnx.data(self_obj.model)
        break
      caller = caller.f_back
  finally:
    del frame

  try:
    from maxtext.utils import model_creation_utils  # pylint: disable=g-import-not-at-top

    if hasattr(model_creation_utils, "from_pretrained") and not getattr(
        model_creation_utils.from_pretrained, "_tunix_moe_fp8_patched", False
    ):
      orig_from_pretrained = model_creation_utils.from_pretrained

      @functools.wraps(orig_from_pretrained)
      def _wrapped_from_pretrained(*args, **kwargs):
        model = orig_from_pretrained(*args, **kwargs)
        if is_rollout_moe_fp8_enabled() and model is not None:
          quantize_routed_moe_modules_in_place(model)
        return model

      _wrapped_from_pretrained._tunix_moe_fp8_patched = True  # pylint: disable=protected-access
      model_creation_utils.from_pretrained = _wrapped_from_pretrained
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("[MoE-FP8] Could not patch from_pretrained: %s", e)

  try:
    from maxtext.integration.vllm.maxtext_vllm_adapter import adapter  # pylint: disable=g-import-not-at-top

    cls = getattr(adapter, "MaxTextForCausalLM", None)
    if cls is not None and not getattr(
        cls.load_weights, "_tunix_moe_fp8_patched", False
    ):
      orig_load_weights = cls.load_weights

      @functools.wraps(orig_load_weights)
      def _wrapped_load_weights(self, *args, **kwargs):
        res = orig_load_weights(self, *args, **kwargs)
        if is_rollout_moe_fp8_enabled() and getattr(self, "model", None) is not None:
          quantize_routed_moe_modules_in_place(self.model)
          if isinstance(self, nnx.Module):
            self.model = nnx.data(self.model)
        return res

      _wrapped_load_weights._tunix_moe_fp8_patched = True  # pylint: disable=protected-access
      cls.load_weights = _wrapped_load_weights
  except Exception as e:  # pylint: disable=broad-exception-caught
    logger.debug("[MoE-FP8] Could not patch MaxTextForCausalLM class: %s", e)

  return quantized_now


def quantize_converted_state_moe_fp8(converted_state: Any) -> int:
  """Quantizes MoE `wi` and `wo` tensors in a converted state tree.

  Handles both nested dicts (`state['decoder']['layers_0'][...]['wi']`)
  and flat tuple-keyed dicts (`state[('decoder', 'layers_0', ..., 'wi')]`)
  across `MoeBlock_0`, `moe_block`, and `routed_experts` parent keys.

  Args:
    converted_state: Mutable dict returned by `WeightConverter.convert` or
      populated by `_execute_group_target_free`.

  Returns:
    Number of MoE weight tensors quantized.
  """
  if not isinstance(converted_state, dict):
    return 0

  quantized_tensors = 0

  # Case 1: Flat tuple-keyed dict (used inside _execute_group_target_free).
  tuple_keys = [
      k
      for k in list(converted_state.keys())
      if isinstance(k, tuple)
      and len(k) >= 2
      and k[-2] in _MOE_BLOCK_KEYS
      and k[-1] in ("wi", "wo")
  ]
  for key in tuple_keys:
    val = converted_state[key]
    arr = _unwrap_array(val)
    if (
        not hasattr(arr, "dtype")
        or getattr(arr, "ndim", 0) != 3
        or arr.dtype == jnp.float8_e4m3fn
    ):
      continue
    q_w, scale = quantize_moe_weight_3d(arr)
    if isinstance(val, nnx.Variable):
      converted_state[key] = nnx.Param(q_w)
      scale_val = nnx.Param(scale)
    else:
      converted_state[key] = q_w
      scale_val = scale
    scale_key = key[:-1] + (f"{key[-1]}_scale",)
    converted_state[scale_key] = scale_val
    quantized_tensors += 1

  # Case 2: Nested dict tree (returned by WeightConverter.convert).
  def _walk_dict(d: dict[Any, Any], parent_key: Any = None):
    nonlocal quantized_tensors
    if parent_key in _MOE_BLOCK_KEYS:
      for w_name in ("wi", "wo"):
        if w_name in d:
          val = d[w_name]
          arr = _unwrap_array(val)
          if (
              hasattr(arr, "dtype")
              and getattr(arr, "ndim", 0) == 3
              and arr.dtype != jnp.float8_e4m3fn
          ):
            q_w, scale = quantize_moe_weight_3d(arr)
            if isinstance(val, nnx.Variable):
              d[w_name] = nnx.Param(q_w)
              d[f"{w_name}_scale"] = nnx.Param(scale)
            elif hasattr(val, "value"):
              val.value = q_w
              d[f"{w_name}_scale"] = scale
            else:
              d[w_name] = q_w
              d[f"{w_name}_scale"] = scale
            quantized_tensors += 1
      return

    for k, v in list(d.items()):
      if isinstance(v, dict):
        _walk_dict(v, parent_key=k)

  _walk_dict(converted_state)
  return quantized_tensors


def _quantize_exec_group_outputs(outs: list[Any]) -> list[Any]:
  """Quantizes 3D MoE `(tgt_key, array)` pairs returned by `_execute_group*`."""
  new_outs = []
  for item in outs:
    if (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], tuple)
        and len(item[0]) >= 2
        and item[0][-2] in _MOE_BLOCK_KEYS
        and item[0][-1] in ("wi", "wo")
    ):
      tgt_key, val = item
      arr = _unwrap_array(val)
      if (
          hasattr(arr, "dtype")
          and getattr(arr, "ndim", 0) == 3
          and arr.dtype != jnp.float8_e4m3fn
      ):
        q_w, scale = quantize_moe_weight_3d(arr)
        scale_key = tgt_key[:-1] + (f"{tgt_key[-1]}_scale",)
        new_outs.append((tgt_key, q_w))
        new_outs.append((scale_key, scale))
        continue
    new_outs.append(item)
  return new_outs


def patch_trainer_converter_moe_fp8(weight_converter: Any) -> bool:
  """Patches trainer `WeightConverter` to emit FP8 MoE weights + F32 scales.

  Hooks `_direct._execute_group_target_free` and `_direct._execute_group` (so
  each layer group's `wi` and `wo` is quantized on the fly during streaming
  conversion, reducing peak trainer HBM by ~360 GB) and `weight_converter.convert`
  as a fallback.

  Args:
    weight_converter: `MaxTextTrainingEngine._weight_converter` instance.

  Returns:
    True if patching was applied, False otherwise.
  """
  if weight_converter is None or not is_rollout_moe_fp8_enabled():
    return False

  direct = getattr(weight_converter, "_direct", None)
  for method_name in ("_execute_group_target_free", "_execute_group"):
    if direct is not None and hasattr(direct, method_name):
      orig_method = getattr(direct, method_name)
      if not getattr(orig_method, "_tunix_moe_fp8_patched", False):

        def _make_wrapper(fn):
          @functools.wraps(fn)
          def _wrapped_exec_group(group, source_flat, *args, **kwargs):
            res = fn(group, source_flat, *args, **kwargs)
            if not is_rollout_moe_fp8_enabled():
              return res
            if isinstance(res, list):
              return _quantize_exec_group_outputs(res)
            if args and isinstance(args[0], dict):
              quantize_converted_state_moe_fp8(args[0])
            elif "target_flat" in kwargs and isinstance(
                kwargs["target_flat"], dict
            ):
              quantize_converted_state_moe_fp8(kwargs["target_flat"])
            return res

          _wrapped_exec_group._tunix_moe_fp8_patched = True  # pylint: disable=protected-access
          return _wrapped_exec_group

        setattr(direct, method_name, _make_wrapper(orig_method))

  if hasattr(weight_converter, "convert") and not getattr(
      weight_converter.convert, "_tunix_moe_fp8_patched", False
  ):
    orig_convert = weight_converter.convert

    @functools.wraps(orig_convert)
    def _wrapped_convert(*args, **kwargs):
      converted = orig_convert(*args, **kwargs)
      if is_rollout_moe_fp8_enabled():
        n_q = quantize_converted_state_moe_fp8(converted)
        if n_q > 0:
          logger.info(
              "[MoE-FP8] Quantized %d MoE weight tensors in trainer converter.",
              n_q,
          )
      return converted

    _wrapped_convert._tunix_moe_fp8_patched = True  # pylint: disable=protected-access
    weight_converter.convert = _wrapped_convert

  logger.info("[MoE-FP8] Patched trainer weight converter for MoE FP8 rollout.")
  return True
