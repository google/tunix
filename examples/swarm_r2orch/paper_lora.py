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
"""Pure-JAX functional LoRA utilities for Qwen3.5-9B Director and Proposer."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

LORA_TARGET_SUFFIXES: tuple[str, ...] = (
    "attn.q_proj.w",
    "attn.k_proj.w",
    "attn.v_proj.w",
    "attn.o_proj.w",
    "mlp.gate_proj.kernel",
    "mlp.up_proj.kernel",
    "mlp.down_proj.kernel",
)

LoraTree = dict[str, dict[str, jax.Array]]


def _flat_key_str(keys: tuple[Any, ...]) -> str:
  return ".".join(str(k) for k in keys)


def _lora_matrix_dims(key_str: str, shape: tuple[int, ...]) -> tuple[int, int]:
  if len(shape) == 2:
    return int(shape[0]), int(shape[1])
  if len(shape) == 3 and key_str.endswith(".attn.o_proj.w"):
    return int(shape[0] * shape[1]), int(shape[2])
  if len(shape) == 3:
    return int(shape[0]), int(shape[1] * shape[2])
  raise ValueError(f"Unsupported LoRA target shape for {key_str}: {shape}")


def matches_lora_target(key_str: str) -> bool:
  return any(key_str.endswith("." + suffix) for suffix in LORA_TARGET_SUFFIXES)


def init_lora_params(
    base_state: nnx.State,
    *,
    rank: int = 64,
    seed: int = 0,
    mesh: jax.sharding.Mesh | None = None,
) -> LoraTree:
  """Initializes Kaiming-uniform `lora_a` and zero `lora_b` in float32."""
  rng = np.random.default_rng(seed)
  replicated = (
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      if mesh is not None
      else None
  )
  params: LoraTree = {}
  for keys, var in nnx.to_flat_state(base_state):
    key_str = _flat_key_str(keys)
    if not matches_lora_target(key_str):
      continue
    arr = var.get_value() if isinstance(var, nnx.Variable) else var
    d_in, d_out = _lora_matrix_dims(key_str, tuple(arr.shape))
    bound = 1.0 / math.sqrt(d_in)
    a_np = rng.uniform(-bound, bound, size=(d_in, rank)).astype(np.float32)
    b_np = np.zeros((rank, d_out), dtype=np.float32)
    a_jax = (
        jax.device_put(a_np, replicated)
        if replicated is not None
        else jnp.asarray(a_np)
    )
    b_jax = (
        jax.device_put(b_np, replicated)
        if replicated is not None
        else jnp.asarray(b_np)
    )
    params[key_str] = {"lora_a": a_jax, "lora_b": b_jax}
  return params


def _wrap_leaf(var: Any, val: jax.Array) -> Any:
  return type(var)(val) if isinstance(var, nnx.Variable) else val


def _apply_lora_leaf(
    arr: jax.Array,
    entry: dict[str, jax.Array],
    scale: Any,
    out_dtype: Any,
) -> jax.Array:
  delta_2d = jnp.matmul(
      entry["lora_a"].astype(jnp.float32),
      entry["lora_b"].astype(jnp.float32),
      precision=jax.lax.Precision.HIGHEST,
  )
  delta = (delta_2d * jnp.float32(scale)).reshape(arr.shape)
  updated = (arr.astype(jnp.float32) + delta).astype(out_dtype)
  if not isinstance(arr, jax.core.Tracer) and isinstance(
      arr.sharding, jax.sharding.NamedSharding
  ):
    updated = jax.lax.with_sharding_constraint(updated, arr.sharding)
  return updated


def merge_lora_state(
    base_state: nnx.State,
    lora_params: LoraTree,
    *,
    scale: float = 1.0,
    target_dtype: jnp.dtype | None = None,
) -> nnx.State:
  """Returns an `nnx.State` with `W + scale * (lora_a @ lora_b)` merged."""
  merged_flat: list[tuple[tuple[Any, ...], Any]] = []
  for keys, var in nnx.to_flat_state(base_state):
    key_str = _flat_key_str(keys)
    arr = var.get_value() if isinstance(var, nnx.Variable) else var
    out_dtype = target_dtype if target_dtype is not None else arr.dtype
    entry = lora_params.get(key_str)
    if entry is None:
      cast_arr = arr.astype(out_dtype) if arr.dtype != out_dtype else arr
      merged_flat.append((keys, _wrap_leaf(var, cast_arr)))
    else:
      updated = _apply_lora_leaf(arr, entry, scale, out_dtype)
      merged_flat.append((keys, _wrap_leaf(var, updated)))
  return nnx.from_flat_state(merged_flat)


def prune_old_checkpoints(
    checkpoint_root: Path,
    *,
    current_step: int,
    keep_recent: int = 3,
    milestone_every: int = 25,
) -> None:
  """Retains recent full checkpoints and milestone `lora_weights.npz` files."""
  cutoff = int(current_step) - max(1, int(keep_recent))
  for child in sorted(Path(checkpoint_root).glob("step-*")):
    if not child.is_dir():
      continue
    suffix = child.name.removeprefix("step-")
    if not suffix.isdigit() or int(suffix) > cutoff:
      continue
    old_step = int(suffix)
    if milestone_every > 0 and old_step % milestone_every == 0:
      (child / "optimizer_state.npz").unlink(missing_ok=True)
    else:
      shutil.rmtree(child, ignore_errors=True)


def save_lora_checkpoint(
    checkpoint_root: Path,
    *,
    step: int,
    lora_params: LoraTree,
    opt_arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    publish_latest: bool = True,
    keep_recent: int = 3,
    milestone_every: int = 25,
) -> Path:
  """Persists a compact step checkpoint (`lora.npz` + `opt.npz` + `latest.json`)."""
  step_dir = Path(checkpoint_root) / f"step-{int(step):08d}"
  step_dir.mkdir(parents=True, exist_ok=True)
  lora_flat: dict[str, np.ndarray] = {}
  for key_str, pair in lora_params.items():
    lora_flat[f"{key_str}::lora_a"] = np.asarray(
        jax.device_get(pair["lora_a"]), dtype=np.float32
    )
    lora_flat[f"{key_str}::lora_b"] = np.asarray(
        jax.device_get(pair["lora_b"]), dtype=np.float32
    )
  np.savez(step_dir / "lora_weights.npz", **lora_flat)
  if opt_arrays:
    np.savez(step_dir / "optimizer_state.npz", **opt_arrays)
  (step_dir / "trainer_meta.json").write_text(
      json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
  )
  if publish_latest:
    latest_payload = {"step": int(step), "path": str(step_dir.resolve())}
    tmp_latest = Path(checkpoint_root) / "latest.json.tmp"
    tmp_latest.write_text(
        json.dumps(latest_payload, indent=2) + "\n", encoding="utf-8"
    )
    tmp_latest.replace(Path(checkpoint_root) / "latest.json")
    prune_old_checkpoints(
        Path(checkpoint_root),
        current_step=int(step),
        keep_recent=keep_recent,
        milestone_every=milestone_every,
    )
  return step_dir


def load_lora_weights(
    step_dir: Path,
    *,
    mesh: jax.sharding.Mesh | None = None,
) -> tuple[LoraTree, dict[str, np.ndarray], dict[str, Any]]:
  """Loads `lora_weights.npz`, `optimizer_state.npz`, and `trainer_meta.json`."""
  replicated = (
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      if mesh is not None
      else None
  )
  raw_lora = np.load(Path(step_dir) / "lora_weights.npz")
  params: LoraTree = {}
  for full_key in raw_lora.files:
    key_str, leaf = full_key.split("::", 1)
    arr = np.asarray(raw_lora[full_key], dtype=np.float32)
    val = (
        jax.device_put(arr, replicated)
        if replicated is not None
        else jnp.asarray(arr)
    )
    params.setdefault(key_str, {})[leaf] = val
  opt_path = Path(step_dir) / "optimizer_state.npz"
  opt_arrays = (
      {k: np.asarray(v) for k, v in np.load(opt_path).items()}
      if opt_path.is_file()
      else {}
  )
  meta_path = Path(step_dir) / "trainer_meta.json"
  meta = (
      json.loads(meta_path.read_text(encoding="utf-8"))
      if meta_path.is_file()
      else {}
  )
  return params, opt_arrays, meta
