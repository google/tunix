"""Qwen3-32B additive all-RMSNorm wrapper for the P22.XK stack."""

from __future__ import annotations

import importlib.util
import os

from p22xf_contract import HIDDEN_SIZE, validate_model_env
from p22xh_contract import preflight


BASE_PATH = __import__("canon_shim_root").resolve("qwen3.py")
value = os.environ.get("CANON_PALLAS_ALL_RMSNORM", "")
if value not in ("", "1"):
  raise RuntimeError(
      f"CANON_PALLAS_ALL_RMSNORM must be unset or 1, got {value!r}"
  )

spec = importlib.util.spec_from_file_location(
    "_canon_qwen3_p22xh_qwen32b_base", BASE_PATH
)
if spec is None or spec.loader is None:
  raise RuntimeError(f"cannot load qwen3 base from {BASE_PATH}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
stock_rmsnorm = base.JaxRmsNorm

if value == "1":
  validate_model_env()
  preflight(require_enabled=True)
  from p22_pallas_rmsnorm import rmsnorm as pallas_rmsnorm


def _site(prefix: str) -> str:
  for suffix, site in (
      (".input_layernorm", "input"),
      (".post_attention_layernorm", "post"),
      (".q_norm", "q"),
      (".k_norm", "k"),
  ):
    if prefix.endswith(suffix):
      return site
  if prefix == "model.norm" or prefix.endswith(".model.norm"):
    return "final"
  raise RuntimeError(
      f"Qwen3-32B P22.XH unregistered RMSNorm prefix={prefix!r}"
  )


class P22XHRmsNorm(stock_rmsnorm):
  """Routes every registered Qwen3-32B RMSNorm through canonical Pallas."""

  def __init__(self, *args, **kwargs):
    self._p22xh_prefix = str(kwargs.get("prefix", ""))
    super().__init__(*args, **kwargs)

  def __call__(self, x, mask=None):
    if value != "1":
      return super().__call__(x, mask=mask)

    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P
    import tpu_inference.layers.jax.linear as linear_module

    site = _site(self._p22xh_prefix)
    if mask is not None:
      raise RuntimeError(
          f"P22.XH does not support RMSNorm mask at {self._p22xh_prefix}"
      )
    if self.quant_method is not None:
      raise RuntimeError(
          f"P22.XH does not support quantized RMSNorm at {self._p22xh_prefix}"
      )
    if self.reduction_axes not in (-1, (-1,)) or self.feature_axes not in (
        -1,
        (-1,),
    ):
      raise RuntimeError(
          f"P22.XH requires last-axis norm at {self._p22xh_prefix}: "
          f"reduction={self.reduction_axes} feature={self.feature_axes}"
      )
    if self.axis_name is not None or self.axis_index_groups is not None:
      raise RuntimeError(
          f"P22.XH does not support axis collectives at {self._p22xh_prefix}"
      )
    if not self.use_scale:
      raise RuntimeError(f"P22.XH requires scale at {self._p22xh_prefix}")
    if x.dtype != jnp.bfloat16 or int(x.shape[-1]) not in (128, HIDDEN_SIZE):
      raise RuntimeError(
          f"Qwen3-32B P22.XH requires bf16 F=128/{HIDDEN_SIZE} at "
          f"{self._p22xh_prefix}, got shape={x.shape} dtype={x.dtype}"
      )
    weight = self.weight[...]
    if weight.dtype != jnp.bfloat16 or tuple(weight.shape) != (
        int(x.shape[-1]),
    ):
      raise RuntimeError(
          f"P22.XH weight contract failed at {self._p22xh_prefix}: "
          f"shape={weight.shape} dtype={weight.dtype}"
      )
    mesh = linear_module._CANON_MESH
    axis = linear_module._CANON_TP_AXIS
    if mesh is None:
      raise RuntimeError("P22.XH canonical mesh is unset")
    if x.ndim == 2:
      x_spec = P(None, None)
    elif x.ndim == 3 and site in ("q", "k"):
      x_spec = P(None, axis, None)
    else:
      raise RuntimeError(
          f"P22.XH unregistered rank/site at {self._p22xh_prefix}: "
          f"site={site} shape={x.shape}"
      )

    epsilon = float(self.epsilon)

    def local(x_local, weight_local):
      local_shape = x_local.shape
      flat = x_local.reshape((-1, local_shape[-1]))
      if flat.shape[0] % 8:
        raise RuntimeError(
            f"P22.XH local rows must divide BM8 at {self._p22xh_prefix}, "
            f"got {flat.shape}"
        )
      out = pallas_rmsnorm(flat, weight_local, epsilon=epsilon)
      print(
          f"[PATHTRACE] CANON_PALLAS_ALL_RMSNORM=1 site={site} "
          f"prefix={self._p22xh_prefix} rows={flat.shape[0]} "
          f"F={flat.shape[1]} model=qwen3-32b",
          flush=True,
      )
      return out.reshape(local_shape)

    kwargs = {
        "mesh": mesh,
        "in_specs": (x_spec, P(None)),
        "out_specs": x_spec,
    }
    try:
      mapped = shard_map(local, check_vma=False, **kwargs)
    except TypeError:
      mapped = shard_map(local, check_rep=False, **kwargs)
    return mapped(x, weight)


if value == "1":
  base.JaxRmsNorm = P22XHRmsNorm

for name, obj in vars(base).items():
  if name not in {"__name__", "__loader__", "__package__", "__spec__"}:
    globals()[name] = obj
