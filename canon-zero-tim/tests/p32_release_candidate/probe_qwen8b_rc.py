#!/usr/bin/env python3
"""Run the default-off DP16xTP4 Qwen3-8B checkpoint-forward gate."""

from __future__ import annotations

import argparse
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

from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top
from tunix.models.qwen3 import params as params_lib  # pylint: disable=g-import-not-at-top
from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top
from tunix.rl import dp_training  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(_TESTS / "p32_model_init"))
from probe_qwen8b_init import _topology_mesh  # pylint: disable=g-import-not-at-top


_STAGES = ("checkpoint-forward",)
_MODEL_ID = "Qwen/Qwen3-8B"
_EXPECTED_PARAM_LEAVES = 399
_GLOBAL_TRAJECTORIES = 256
_LOCAL_TRAJECTORIES = 16
_DP_SIZE = 16
_TP_SIZE = 4
_SEQ_LEN = 16


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


def _sample_tree_sha256(tree: Any, *, samples_per_leaf: int = 8) -> str:
  digest = hashlib.sha256()
  for index, leaf in enumerate(jax.tree.leaves(tree)):
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


def build_forward_program(graphdef: Any):
  """Builds the real native Tunix forward used by the checkpoint gate."""

  def model_rows(params, tokens, positions, attention_mask):
    model = nnx.merge(graphdef, params)
    logits, _ = model(tokens, positions, None, attention_mask)
    return logits[:, -1, :]

  return jax.jit(model_rows)


def _run_forward(model: model_lib.Qwen3, mesh: jax.sharding.Mesh) -> dict[str, Any]:
  graphdef, params = nnx.split(model, nnx.Param)
  if len(jax.tree.leaves(params)) != _EXPECTED_PARAM_LEAVES:
    raise RuntimeError("Qwen3-8B parameter leaf count changed")
  inputs = _make_inputs(
      mesh, _GLOBAL_TRAJECTORIES, _SEQ_LEN, model.config.vocab_size
  )
  plain = build_forward_program(graphdef)
  before_sha = _sample_tree_sha256(params)
  first_rows = plain(params, *inputs)
  second_rows = plain(params, *inputs)
  jax.block_until_ready((first_rows, second_rows))
  repeat_exact = bool(np.array_equal(
      np.asarray(jax.device_get(first_rows)),
      np.asarray(jax.device_get(second_rows)),
  ))
  if not repeat_exact:
    raise RuntimeError("plain forward repeat changed bits")
  after_sha = _sample_tree_sha256(params)
  if after_sha != before_sha:
    raise RuntimeError("plain forward mutated parameters")
  return {
      "forward_repeat_exact": repeat_exact,
      "forward_shape": list(first_rows.shape),
      "parameter_sample_sha256_before": before_sha,
      "parameter_sample_sha256_after": after_sha,
      "execution": {
          "forward": 2,
          "backward": 0,
          "optimizer_updates": 0,
          "training_steps": 0,
      },
  }


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
  config = model_lib.ModelConfig.qwen3_8b()
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  config.remat_config = model_lib.RematConfig.DECODER
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.shd_config = model_lib.ShardingConfig.get_data_parallel_sharding()
  with jax.set_mesh(mesh):
    model = params_lib.create_model_from_safe_tensors(
        str(checkpoint_dir), config, mesh, dtype=jnp.float32
    )
  jax.block_until_ready(nnx.state(model, nnx.Param))
  if _checkpoint_identity(checkpoint_dir) != checkpoint_before:
    raise RuntimeError("checkpoint identity changed while loading")
  inventory = dp_training.inspect_dp_replicated_state(
      nnx.state(model, nnx.Param), label="model"
  )
  result = _run_forward(model, mesh)
  if _checkpoint_identity(checkpoint_dir) != checkpoint_before:
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
