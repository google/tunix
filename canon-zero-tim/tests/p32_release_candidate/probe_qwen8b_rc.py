#!/usr/bin/env python3
"""Run the default-off DP16xTP4 Qwen3-8B release-candidate stages."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


_HERE = Path(__file__).resolve().parent
_TESTS = _HERE.parent
sys.path.insert(0, str(_TESTS / "t1_tpu"))
from pathways_bootstrap import initialize_pathways  # pylint: disable=g-import-not-at-top


initialize_pathways()

from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top
from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import optax  # pylint: disable=g-import-not-at-top

from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top
from tunix.models.qwen3 import params as params_lib  # pylint: disable=g-import-not-at-top
from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top
from tunix.rl import dp_training  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(_TESTS / "p32_model_init"))
from probe_qwen8b_init import _topology_mesh  # pylint: disable=g-import-not-at-top
from probe_qwen8b_init import materialize_zero_state  # pylint: disable=g-import-not-at-top


_STAGES = ("checkpoint-forward", "backward", "one-update", "three-update")
_MODEL_ID = "Qwen/Qwen3-8B"
_EXPECTED_PARAM_LEAVES = 399
_GLOBAL_TRAJECTORIES = 256
_LOCAL_TRAJECTORIES = 16
_DP_SIZE = 16
_TP_SIZE = 4
_SEQ_LEN = 16
_ATTENTION_BACKEND = "dense-reference"


def _release_candidate_model_config() -> model_lib.ModelConfig:
  """Builds the bounded RC model contract used by all four stages.

  The RC deliberately uses a 16-token sequence to keep the real-checkpoint
  systems probe bounded. Splash Attention with the production block size of
  256 cannot represent that shape because the query block must divide the
  query sequence length. The production Splash path is therefore outside this
  RC's claim and must be admitted by a workload-scale gate.
  """
  config = model_lib.ModelConfig.qwen3_8b()
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  config.remat_config = model_lib.RematConfig.DECODER
  config.use_flash_attention = False
  config.shd_config = model_lib.ShardingConfig.get_data_parallel_sharding()
  return config


def _required_int(environ: Mapping[str, str], name: str) -> int:
  value = environ.get(name, "")
  if not value:
    raise ValueError(f"{name} is required")
  try:
    return int(value)
  except ValueError as exc:
    raise ValueError(f"{name} must be an integer") from exc


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
  return sorted(checkpoint_dir.glob("*.safetensors"))


def _checkpoint_identity(checkpoint_dir: Path) -> dict[str, Any]:
  files = _checkpoint_files(checkpoint_dir)
  if not files:
    raise RuntimeError(f"no safetensors found in {checkpoint_dir}")
  entries = [(path.name, int(path.stat().st_size)) for path in files]
  digest = hashlib.sha256()
  for name, size in entries:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\n")
  for name in ("model.safetensors.index.json", "config.json"):
    path = checkpoint_dir / name
    if path.is_file():
      digest.update(name.encode("utf-8"))
      digest.update(b"\0")
      digest.update(path.read_bytes())
  return {
      "files": len(entries),
      "bytes": sum(size for _, size in entries),
      "manifest_sha256": digest.hexdigest(),
  }


def _ensure_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
  if not _checkpoint_files(checkpoint_dir):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[P32.RC] checkpoint_download model={_MODEL_ID} "
        f"directory={checkpoint_dir}",
        flush=True,
    )
    oss_utils.hf_pipeline(_MODEL_ID, str(checkpoint_dir))
  return _checkpoint_identity(checkpoint_dir)


def _tree_exact(left: Any, right: Any) -> bool:
  @jax.jit
  def compare(left_tree, right_tree):
    flags = jax.tree.leaves(jax.tree.map(jnp.array_equal, left_tree, right_tree))
    result = jnp.asarray(True)
    for flag in flags:
      result = jnp.logical_and(result, flag)
    return result

  return bool(np.asarray(compare(left, right)))


def _tree_health(tree: Any) -> dict[str, Any]:
  @jax.jit
  def inspect(value_tree):
    leaves = jax.tree.leaves(value_tree)
    finite = jnp.asarray(True)
    nonzero = jnp.asarray(0.0, jnp.float32)
    squared = jnp.asarray(0.0, jnp.float32)
    for leaf in leaves:
      value = leaf.astype(jnp.float32)
      finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(value)))
      nonzero = nonzero + jnp.count_nonzero(value).astype(jnp.float32)
      squared = squared + jnp.sum(jnp.square(value))
    return finite, nonzero, jnp.sqrt(squared)

  finite, nonzero, norm = jax.device_get(inspect(tree))
  return {
      "finite": bool(finite),
      "nonzero": int(nonzero),
      "norm": float(norm),
  }


def _sample_tree_sha256(tree: Any, *, samples_per_leaf: int = 8) -> str:
  digest = hashlib.sha256()
  leaves = jax.tree.leaves(tree)
  for index, leaf in enumerate(leaves):
    flat = leaf.reshape(-1)
    count = min(samples_per_leaf, int(flat.size))
    if not count:
      continue
    sample = np.ascontiguousarray(jax.device_get(flat[:count]))
    digest.update(index.to_bytes(4, "little"))
    digest.update(str(tuple(leaf.shape)).encode("ascii"))
    digest.update(str(leaf.dtype).encode("ascii"))
    digest.update(sample.view(np.uint8))
  return digest.hexdigest()


def _tree_signature(tree: Any) -> Any:
  """Returns a compact device-side signature for one gradient contribution."""
  total = jnp.asarray(0.0, jnp.float32)
  absolute = jnp.asarray(0.0, jnp.float32)
  squared = jnp.asarray(0.0, jnp.float32)
  nonzero = jnp.asarray(0.0, jnp.float32)
  for leaf in jax.tree.leaves(tree):
    value = leaf.astype(jnp.float32)
    total = total + jnp.sum(value)
    absolute = absolute + jnp.sum(jnp.abs(value))
    squared = squared + jnp.sum(jnp.square(value))
    nonzero = nonzero + jnp.count_nonzero(value).astype(jnp.float32)
  return jnp.stack((total, absolute, squared, nonzero))


def _signature_sha256(signature: Any) -> str:
  value = np.ascontiguousarray(jax.device_get(signature))
  return hashlib.sha256(value.view(np.uint8)).hexdigest()


def _state_memory_kinds(tree: Any) -> tuple[str, ...]:
  return tuple(sorted({
      str(leaf.sharding.memory_kind)
      for leaf in jax.tree.leaves(tree)
      if isinstance(leaf, jax.Array)
  }))


def _put_memory_kind(tree: Any, memory_kind: str) -> Any:
  def move(value):
    if not isinstance(value, jax.Array):
      return value
    if value.sharding.memory_kind == memory_kind:
      return value
    return jax.device_put(value, value.sharding.with_memory_kind(memory_kind))

  with jax.transfer_guard("allow"):
    moved = jax.tree.map(move, tree)
  return jax.block_until_ready(moved)


def _make_inputs(
    mesh: jax.sharding.Mesh,
    global_batch: int,
    seq_len: int,
    vocab_size: int = 151936,
):
  if global_batch % mesh.shape["dp"]:
    raise ValueError("global batch must be divisible by DP")
  tokens = (
      np.arange(global_batch * seq_len, dtype=np.int32).reshape(
          global_batch, seq_len
      )
      * np.int32(17)
      + np.int32(23)
  ) % np.int32(vocab_size)
  positions = np.broadcast_to(
      np.arange(seq_len, dtype=np.int32), (global_batch, seq_len)
  ).copy()
  causal = np.tril(np.ones((seq_len, seq_len), dtype=np.bool_))
  attention = np.broadcast_to(
      causal, (global_batch, seq_len, seq_len)
  ).copy()
  return (
      jax.device_put(tokens, NamedSharding(mesh, P("dp", None))),
      jax.device_put(positions, NamedSharding(mesh, P("dp", None))),
      jax.device_put(attention, NamedSharding(mesh, P("dp", None, None))),
  )


def build_dp_programs(
    *,
    graphdef: Any,
    mesh: jax.sharding.Mesh,
    global_batch: int,
    vocab_size: int,
):
  """Builds global forward and one-rank-at-a-time gradient programs."""
  dp_size = int(mesh.shape["dp"])
  if global_batch % dp_size:
    raise ValueError("global batch must be divisible by DP")
  local_batch = global_batch // dp_size

  def model_rows(params, tokens, positions, attention_mask):
    model = nnx.merge(graphdef, params)
    logits, _ = model(tokens, positions, None, attention_mask)
    return logits[:, -1, :]

  def plain(params, tokens, positions, attention_mask):
    return model_rows(params, tokens, positions, attention_mask)

  def value_and_rank_grad(params, tokens, positions, attention_mask, rank):
    def rank_objective(candidate):
      rows = model_rows(candidate, tokens, positions, attention_mask)
      per_example = jnp.sum(
          jnp.square(rows.astype(jnp.float32)), axis=-1
      ) / jnp.asarray(vocab_size, jnp.float32)
      rank_losses = jnp.sum(
          per_example.reshape(dp_size, local_batch), axis=1
      ) / jnp.asarray(global_batch, jnp.float32)
      return jax.lax.dynamic_index_in_dim(
          rank_losses, rank, keepdims=False
      ), rows

    (rank_loss, rows), rank_gradients = jax.value_and_grad(
        rank_objective, has_aux=True
    )(params)
    return rank_loss, rows, rank_gradients, _tree_signature(rank_gradients)

  return jax.jit(plain), jax.jit(value_and_rank_grad)


def _stream_fixed_rank_gradient(
    rank_program: Any,
    params: Any,
    inputs: tuple[Any, Any, Any],
    *,
    dp_size: int,
) -> tuple[Any, Any, Any, tuple[str, ...]]:
  """Adds one nonzero-rank VJP at a time in registered rank order."""

  @functools.partial(jax.jit, donate_argnums=(0, 1))
  def add(left, right):
    return jax.tree.map(
        lambda x, y: (
            jax.lax.optimization_barrier(x)
            + jax.lax.optimization_barrier(y)
        ),
        left,
        right,
    )

  @jax.jit
  def add_loss(left, right):
    return (
        jax.lax.optimization_barrier(left)
        + jax.lax.optimization_barrier(right)
    )

  accumulator = None
  total_loss = jnp.asarray(0.0, jnp.float32)
  rows = None
  fingerprints = []
  for rank in range(dp_size):
    rank_loss, rank_rows, contribution, contribution_signature = rank_program(
        params, *inputs, jnp.asarray(rank, jnp.int32)
    )
    jax.block_until_ready(
        (rank_loss, rank_rows, contribution, contribution_signature)
    )
    fingerprints.append(_signature_sha256(contribution_signature))
    total_loss = add_loss(total_loss, rank_loss)
    total_loss.block_until_ready()
    if rows is None:
      rows = rank_rows
    if accumulator is None:
      accumulator = contribution
    else:
      accumulator = add(accumulator, contribution)
      jax.block_until_ready(accumulator)
  if accumulator is None or rows is None:
    raise RuntimeError("fixed rank reduction emitted no contribution")
  return total_loss, rows, accumulator, tuple(fingerprints)


def _replica_samples_exact(tree: Any, *, samples_per_shard: int = 8) -> bool:
  """Checks small physical samples for every replicated DP copy."""
  for leaf in jax.tree.leaves(tree)[:8]:
    groups: dict[str, list[np.ndarray]] = {}
    for shard in leaf.addressable_shards:
      key = repr(shard.index)
      sample = np.ascontiguousarray(
          jax.device_get(shard.data.reshape(-1)[:samples_per_shard])
      )
      groups.setdefault(key, []).append(sample)
    for values in groups.values():
      if not all(np.array_equal(values[0], value) for value in values[1:]):
        return False
  return True


def _build_optimizer_state(params: Any, mesh: jax.sharding.Mesh, tx: optax.GradientTransformation):
  abstract = jax.eval_shape(tx.init, params)
  abstract = dp_training.attach_adam_state_shardings(
      abstract, params=params, mesh=mesh
  )
  return materialize_zero_state(abstract, memory_kind="pinned_host")


def _commit_program(tx: optax.GradientTransformation):
  @jax.jit
  def commit(params, optimizer_state, gradients):
    updates, next_optimizer = tx.update(gradients, optimizer_state, params)
    next_params = optax.apply_updates(params, updates)
    return next_params, next_optimizer

  return commit


def _run_stage(
    *,
    stage: str,
    model: model_lib.Qwen3,
    mesh: jax.sharding.Mesh,
) -> dict[str, Any]:
  graphdef, params = nnx.split(model, nnx.Param)
  if len(jax.tree.leaves(params)) != _EXPECTED_PARAM_LEAVES:
    raise RuntimeError("Qwen3-8B parameter leaf count changed")
  inputs = _make_inputs(
      mesh, _GLOBAL_TRAJECTORIES, _SEQ_LEN, model.config.vocab_size
  )
  plain, value_and_rank_grad = build_dp_programs(
      graphdef=graphdef,
      mesh=mesh,
      global_batch=_GLOBAL_TRAJECTORIES,
      vocab_size=model.config.vocab_size,
  )
  before_sha = _sample_tree_sha256(params)
  first_rows = plain(params, *inputs)
  second_rows = plain(params, *inputs)
  jax.block_until_ready((first_rows, second_rows))
  forward_repeat_exact = bool(np.array_equal(
      np.asarray(jax.device_get(first_rows)),
      np.asarray(jax.device_get(second_rows)),
  ))
  if not forward_repeat_exact:
    raise RuntimeError("plain forward repeat changed bits")
  if _sample_tree_sha256(params) != before_sha:
    raise RuntimeError("plain forward mutated parameters")

  execution = {
      "forward": 2,
      "backward": 0,
      "optimizer_updates": 0,
      "training_steps": 0,
  }
  result: dict[str, Any] = {
      "forward_repeat_exact": forward_repeat_exact,
      "forward_shape": list(first_rows.shape),
      "parameter_sample_sha256_before": before_sha,
      "parameter_sample_sha256_after": before_sha,
      "third_program_exact": None,
      "gradient_repeat_exact": None,
      "gradient_health": None,
      "rank_local_stats_distinct": None,
      "post_reduction_replicas_exact": None,
      "dp_reduction_transactions": 0,
      "dp_reduction_rounds_per_transaction": 0,
      "dp_rank_pullbacks_per_transaction": 0,
      "dp_rank_ordered_additions_per_transaction": 0,
      "optimizer_state_memory_between_commits": None,
      "optimizer_state_memory_during_commit": None,
      "step_records": [],
  }
  if stage == "checkpoint-forward":
    result["execution"] = execution
    return result

  repeat_count = 2 if stage == "backward" else 1
  first_gradients = None
  first_gradient_sha = None
  gradient_repeat_exact = True
  tx = optax.adamw(
      learning_rate=1.0e-6, b1=0.9, b2=0.95, weight_decay=0.0
  )
  optimizer_state = None
  commit = None
  updates = 0 if stage == "backward" else (1 if stage == "one-update" else 3)
  if updates:
    optimizer_state = _build_optimizer_state(params, mesh, tx)
    if _state_memory_kinds(optimizer_state) != ("pinned_host",):
      raise RuntimeError("optimizer state did not land in pinned_host")
    commit = _commit_program(tx)
    result["optimizer_state_memory_between_commits"] = ["pinned_host"]

  iterations = repeat_count if stage == "backward" else updates
  for step in range(iterations):
    loss, vg_rows, gradients, rank_fingerprints = (
        _stream_fixed_rank_gradient(
            value_and_rank_grad,
            params,
            inputs,
            dp_size=_DP_SIZE,
        )
    )
    jax.block_until_ready((loss, vg_rows, gradients))
    execution["backward"] += _DP_SIZE
    execution["forward"] += _DP_SIZE
    third_program_exact = bool(np.array_equal(
        np.asarray(jax.device_get(plain(params, *inputs))),
        np.asarray(jax.device_get(vg_rows)),
    ))
    execution["forward"] += 1
    health = _tree_health(gradients)
    if not health["finite"] or health["nonzero"] <= 0 or health["norm"] <= 0:
      raise RuntimeError(f"gradient health failed: {health}")
    local_distinct = len(set(rank_fingerprints)) == _DP_SIZE
    reduced_exact = _replica_samples_exact(gradients)
    if not local_distinct:
      raise RuntimeError("rank-local gradient signatures are not all distinct")
    if not reduced_exact:
      raise RuntimeError("fixed DP reduction produced unequal replicas")
    gradient_sha = _sample_tree_sha256(gradients)
    if first_gradients is None:
      first_gradients = gradients
      first_gradient_sha = gradient_sha
    elif not _tree_exact(first_gradients, gradients):
      gradient_repeat_exact = False
      raise RuntimeError("backward repeat changed gradient bits")
    execution["optimizer_updates"] += int(updates > 0)
    execution["training_steps"] += int(updates > 0)
    step_record = {
        "step": step,
        "loss": float(np.asarray(loss)),
        "third_program_exact": third_program_exact,
        "gradient_sample_sha256": gradient_sha,
        "gradient_health": health,
        "rank_local_stats_distinct": local_distinct,
        "post_reduction_replicas_exact": reduced_exact,
        "rank_contribution_signature_sha256": list(rank_fingerprints),
    }
    if updates:
      assert optimizer_state is not None and commit is not None
      optimizer_state = _put_memory_kind(optimizer_state, "device")
      if _state_memory_kinds(optimizer_state) != ("device",):
        raise RuntimeError("optimizer state did not move to device for commit")
      result["optimizer_state_memory_during_commit"] = ["device"]
      params_before = params
      params, optimizer_state = commit(params, optimizer_state, gradients)
      jax.block_until_ready((params, optimizer_state))
      if _tree_exact(params_before, params):
        raise RuntimeError("optimizer commit did not change parameters")
      optimizer_state = _put_memory_kind(optimizer_state, "pinned_host")
      if _state_memory_kinds(optimizer_state) != ("pinned_host",):
        raise RuntimeError("optimizer state did not return to pinned_host")
      step_record["parameter_sample_sha256"] = _sample_tree_sha256(params)
      step_record["optimizer_sample_sha256"] = _sample_tree_sha256(
          optimizer_state
      )
      first_gradients = None
    result["step_records"].append(step_record)

  result.update({
      "third_program_exact": all(
          entry["third_program_exact"] for entry in result["step_records"]
      ),
      "gradient_repeat_exact": gradient_repeat_exact if stage == "backward" else None,
      "gradient_health": result["step_records"][-1]["gradient_health"],
      "rank_local_stats_distinct": True,
      "post_reduction_replicas_exact": True,
      "dp_reduction_transactions": iterations,
      "dp_reduction_rounds_per_transaction": _DP_SIZE - 1,
      "dp_rank_pullbacks_per_transaction": _DP_SIZE,
      "dp_rank_ordered_additions_per_transaction": _DP_SIZE - 1,
      "parameter_sample_sha256_after": _sample_tree_sha256(params),
      "gradient_sample_sha256": first_gradient_sha,
      "execution": execution,
  })
  return result


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--stage", choices=_STAGES, default=os.getenv("CANON_P32_RC_STAGE", "")
  )
  args = parser.parse_args(argv)
  env = os.environ
  if env.get("CANON_MODE") != "dp16-rc":
    raise RuntimeError("probe requires CANON_MODE=dp16-rc")
  if env.get("CANON_P32_RC") != "1":
    raise RuntimeError("CANON_P32_RC=1 is required")
  if env.get("CANON_P32_TRAIN_ADMITTED", "0") != "0":
    raise RuntimeError("release-candidate mode must not admit production training")
  if env.get("JOBSET_RESTART_ATTEMPT") != "0":
    raise RuntimeError("release-candidate evidence requires attempt 0")
  if not args.stage:
    raise RuntimeError("CANON_P32_RC_STAGE is required")
  dp = _required_int(env, "CANON_DP_SIZE")
  tp = _required_int(env, "CANON_TP_SIZE")
  total = _required_int(env, "CANON_TOTAL_DEVICES")
  if (dp, tp, total) != (_DP_SIZE, _TP_SIZE, 64):
    raise RuntimeError(
        f"release candidate requires DP16xTP4 on 64 devices, got "
        f"DP{dp}xTP{tp} on {total}"
    )
  checkpoint_dir = Path(
      env.get("CANON_P32_CHECKPOINT_DIR", "/tmp/models/Qwen3-8B")
  )
  print(
      f"[P32.RC] START stage={args.stage} attempt=0 dp={dp} tp={tp} "
      f"global_trajectories={_GLOBAL_TRAJECTORIES} "
      f"local_trajectories={_LOCAL_TRAJECTORIES} sequence_length={_SEQ_LEN}",
      flush=True,
  )
  checkpoint_before = _ensure_checkpoint(checkpoint_dir)
  devices = list(jax.devices())
  mesh = _topology_mesh(devices, dp, tp)
  config = _release_candidate_model_config()
  with jax.set_mesh(mesh):
    model = params_lib.create_model_from_safe_tensors(
        str(checkpoint_dir), config, mesh, dtype=jnp.float32
    )
  jax.block_until_ready(nnx.state(model, nnx.Param))
  checkpoint_after_load = _checkpoint_identity(checkpoint_dir)
  if checkpoint_after_load != checkpoint_before:
    raise RuntimeError("checkpoint identity changed while loading")
  inventory = dp_training.inspect_dp_replicated_state(
      nnx.state(model, nnx.Param), label="model"
  )
  result = _run_stage(
      stage=args.stage,
      model=model,
      mesh=mesh,
  )
  checkpoint_after = _checkpoint_identity(checkpoint_dir)
  if checkpoint_after != checkpoint_before:
    raise RuntimeError("checkpoint identity changed during release candidate")
  record = {
      "attempt": 0,
      "stage": args.stage,
      "topology": {
          "dp": dp,
          "tp": tp,
          "devices": len(devices),
          "mesh_shape": list(mesh.devices.shape),
          "unique_devices": len({int(device.id) for device in mesh.devices.flat}),
      },
      "batch": {
          "global_trajectories": _GLOBAL_TRAJECTORIES,
          "local_trajectories": _LOCAL_TRAJECTORIES,
          "sequence_length": _SEQ_LEN,
          "sample_to_rank_mapping": "frozen-contiguous-16",
      },
      "model": {
          "name": "qwen3-8b",
          "checkpoint_loaded": True,
          "checkpoint": checkpoint_before,
          "attention_backend": _ATTENTION_BACKEND,
          "compute_dtype": str(config.dtype),
          "param_dtype": str(config.param_dtype),
          "inventory": inventory,
      },
      "scope": {
          "production_training_admitted": False,
          "zero_tim_alignment": "NOT_MEASURED",
          "rollout_engine_initialized": False,
      },
      **result,
  }
  print(f"[P32.RC] JSON {json.dumps(record, sort_keys=True)}", flush=True)
  print(f"[P32.RC] VERDICT PASS stage={args.stage}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
