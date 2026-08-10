"""Model-free P38 discriminator for sampling and scoring aval splits.

This probe invokes the live TPU-inference sampling transform and the live
canonical logprob scorer. It deliberately does not load a model. A completed
red comparison is diagnostic evidence, while missing measurements, mismatched
inputs, or a failed negative control make the run inconclusive.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TP_SIZE = 4
LOCAL_M = 256
COMPACT_M = 16
VOCAB_SIZE = 151936
EXPECTED_MEASUREMENTS = 5
REPORT_ENV = "CANON_P38_AVAL_REPORT"


@dataclass(frozen=True)
class ShapeContract:
  device_count: int
  data_size: int
  tp_size: int
  compact_m: int
  local_m: int
  global_m: int


class ProbeContractError(RuntimeError):
  """Raised when the P38 model-free measurement is not admissible."""


def shape_contract(device_count: int) -> ShapeContract:
  """Return the only two admitted direct and target topology contracts."""
  if device_count not in (4, 64):
    raise ProbeContractError(
        f"P38 aval probe requires exactly 4 or 64 devices, got {device_count}"
    )
  data_size = device_count // TP_SIZE
  return ShapeContract(
      device_count=device_count,
      data_size=data_size,
      tp_size=TP_SIZE,
      compact_m=COMPACT_M,
      local_m=LOCAL_M,
      global_m=data_size * LOCAL_M,
  )


def bitwise_comparison(left: Any, right: Any) -> dict[str, Any]:
  """Compare two equal-shape arrays and preserve the first exact mismatch."""
  a = np.ascontiguousarray(np.asarray(left))
  b = np.ascontiguousarray(np.asarray(right))
  if a.shape != b.shape or a.dtype != b.dtype:
    raise ProbeContractError(
        "bitwise comparison requires equal shape and dtype: "
        f"{a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    )
  byte_view_a = a.view(np.uint8).reshape(a.size, a.dtype.itemsize)
  byte_view_b = b.view(np.uint8).reshape(b.size, b.dtype.itemsize)
  different = np.any(byte_view_a != byte_view_b, axis=1)
  coordinates = np.flatnonzero(different)
  first = None
  if coordinates.size:
    index = int(coordinates[0])
    first = {
        "index": index,
        "left": float(a.reshape(-1)[index]) if a.dtype.kind == "f" else int(a.reshape(-1)[index]),
        "right": float(b.reshape(-1)[index]) if b.dtype.kind == "f" else int(b.reshape(-1)[index]),
    }
  max_abs = 0.0
  if a.dtype.kind == "f" and a.size:
    max_abs = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
  return {
      "exact": not bool(coordinates.size),
      "differing_elements": int(coordinates.size),
      "total_elements": int(a.size),
      "max_abs": max_abs,
      "first_mismatch": first,
  }


def classify(comparisons: Mapping[str, Mapping[str, Any]], negative: Mapping[str, Any]) -> str:
  """Classify one complete measurement without treating numerical red as failure."""
  required = {
      "raw_target",
      "sampled_token",
      "processed_target",
      "target_logprob",
      "implied_normalizer",
  }
  if set(comparisons) != required:
    raise ProbeContractError(
        f"measurement names changed: {sorted(comparisons)}"
    )
  if len(comparisons) != EXPECTED_MEASUREMENTS:
    raise ProbeContractError("measurement count does not match the registered contract")
  if comparisons["raw_target"].get("exact") is not True:
    raise ProbeContractError("the two arms did not receive identical real-row logits")
  if negative.get("differing_elements") != 1:
    raise ProbeContractError("the one-bit negative control was not detected exactly once")

  transform_red = comparisons["processed_target"].get("exact") is not True
  score_red = comparisons["target_logprob"].get("exact") is not True
  normalizer_red = comparisons["implied_normalizer"].get("exact") is not True
  if normalizer_red and not (transform_red or score_red):
    raise ProbeContractError(
        "the implied normalizer changed while both of its operands were exact"
    )
  if transform_red and score_red:
    return "TRANSFORM_AND_SCORE_AVAL_CARRIER"
  if transform_red:
    return "TRANSFORM_AVAL_CARRIER"
  if score_red:
    return "SCORE_AVAL_CARRIER"
  return "MODEL_FREE_NOT_REPRODUCED"


def _sharding_contract(value: Any) -> dict[str, Any]:
  sharding = getattr(value, "sharding", None)
  mesh = getattr(sharding, "mesh", None)
  spec = getattr(sharding, "spec", None)
  return {
      "shape": [int(size) for size in value.shape],
      "dtype": str(value.dtype),
      "sharding_type": type(sharding).__name__,
      "partition_spec": repr(spec),
      "mesh_shape": dict(getattr(mesh, "shape", {})),
      "memory_kind": getattr(sharding, "memory_kind", None),
  }


def _lowered_sha(function: Any, *args: Any) -> str:
  text = function.lower(*args).as_text()
  return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _flip_one_float_bit(value: np.ndarray) -> np.ndarray:
  changed = np.ascontiguousarray(value).copy()
  if changed.dtype != np.float32 or not changed.size:
    raise ProbeContractError("negative control requires a nonempty float32 array")
  changed.reshape(-1).view(np.uint32)[0] ^= np.uint32(1)
  return changed


def _run_hardware_probe() -> dict[str, Any]:
  from unittest import mock

  from pathways_bootstrap import initialize_pathways

  initialize_pathways()

  import jax
  import jax.numpy as jnp
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
  from tpu_inference.layers.jax.sample.sampling import gather_logprobs, sample
  from tpu_inference.layers.jax.sample.sampling_metadata import (
      TPUSupportedSamplingMetadata,
  )
  from tpu_inference.runner.tpu_runner import _canon_pad_logprob_inputs
  from tunix.rl import canonical_qwen3_adapter

  devices = jax.devices()
  if jax.default_backend() not in ("tpu", "proxy"):
    raise ProbeContractError(
        f"P38 aval probe requires TPU or proxy, got {jax.default_backend()}"
    )
  contract = shape_contract(len(devices))
  arranged = mesh_utils.create_device_mesh(
      (contract.data_size, contract.tp_size),
      devices,
      allow_split_physical_axes=True,
  )
  mesh = Mesh(np.asarray(arranged, dtype=object), ("data", "model"))
  row_sharding = NamedSharding(mesh, P("data", None))
  vector_sharding = NamedSharding(mesh, P("data"))
  replicated = NamedSharding(mesh, P())

  def make_logits(rows: int):
    def build():
      row = jnp.arange(rows, dtype=jnp.int32)[:, None]
      column = jnp.arange(VOCAB_SIZE, dtype=jnp.int32)[None, :]
      integer = jnp.mod(column * jnp.int32(37) + row * jnp.int32(101), 4093)
      return (integer.astype(jnp.float32) / jnp.float32(257.0) - 8.0).astype(
          jnp.bfloat16
      )

    return jax.jit(build, out_shardings=row_sharding)()

  def make_vector(rows: int, value: int, dtype: Any):
    return jax.jit(
        lambda: jnp.full((rows,), value, dtype=dtype),
        out_shardings=vector_sharding,
    )()

  def metadata(rows: int):
    return TPUSupportedSamplingMetadata(
        temperature=jax.jit(
            lambda: jnp.full((rows,), 0.7, dtype=jnp.float32),
            out_shardings=vector_sharding,
        )(),
        top_k=make_vector(rows, -1, jnp.int32),
        top_p=jax.jit(
            lambda: jnp.ones((rows,), dtype=jnp.float32),
            out_shardings=vector_sharding,
        )(),
        do_sampling=True,
        logprobs=True,
    )

  compact_logits = make_logits(contract.compact_m)
  global_logits = make_logits(contract.global_m)
  compact_metadata = metadata(contract.compact_m)
  global_metadata = metadata(contract.global_m)
  rng = jax.random.PRNGKey(20260810)

  compact_tokens, compact_processed = sample(
      rng, mesh, compact_logits, compact_metadata
  )
  global_tokens, global_processed = sample(
      rng, mesh, global_logits, global_metadata
  )
  compact_processed.block_until_ready()
  global_processed.block_until_ready()

  compact_score_logits, compact_score_tokens, real_rows = (
      _canon_pad_logprob_inputs(
          compact_processed,
          compact_tokens,
          contract.local_m,
      )
  )
  if real_rows != contract.compact_m:
    raise ProbeContractError(
        f"decode padding changed the real-row count: {real_rows}"
    )
  base_global_targets = make_vector(contract.global_m, 0, jnp.int32)
  put_real_targets = jax.jit(
      lambda base, real: base.at[: contract.compact_m].set(real),
      out_shardings=vector_sharding,
  )
  global_score_tokens = put_real_targets(base_global_targets, compact_tokens)

  env = {
      "CANON_P32_TRAIN_ADMITTED": "1" if contract.data_size == 16 else "0",
      "CANON_DP_SIZE": str(contract.data_size),
      "CANON_TP_SIZE": str(contract.tp_size),
      "CANON_LOGPROB_M": str(contract.local_m),
      "CANON_TARGET_M": str(contract.local_m),
      "MIN_TOKEN_BUCKET": str(contract.global_m),
      "CANON_PALLAS_LOGSOFTMAX": "1",
      "CANON_P34_DEEPSWE": "0",
  }
  with mock.patch.dict(os.environ, env, clear=False):
    scorer = canonical_qwen3_adapter._make_canonical_compute_and_gather(  # pylint: disable=protected-access
        gather_logprobs, mesh
    )
    compact_scores = scorer(compact_score_logits, compact_score_tokens, 1)
    global_scores = scorer(global_processed, global_score_tokens, 1)
    compact_logps = compact_scores.logprobs[:, 0]
    global_logps = global_scores.logprobs[:, 0]
    compact_logps.block_until_ready()
    global_logps.block_until_ready()
    score_hlo = {
        "decode_m256": _lowered_sha(
            scorer, compact_score_logits, compact_score_tokens, 1
        ),
        "prompt_global": _lowered_sha(
            scorer, global_processed, global_score_tokens, 1
        ),
    }

  def replicate_prefix(value):
    return jax.jit(
        lambda array: array[: contract.compact_m],
        out_shardings=replicated,
    )(value)

  def select_prefix(logits, token_ids):
    return jax.jit(
        lambda matrix, ids: jnp.take_along_axis(
            matrix[: contract.compact_m],
            ids[: contract.compact_m, None],
            axis=1,
        )[:, 0],
        out_shardings=replicated,
    )(logits, token_ids)

  compact_raw_target = select_prefix(compact_logits, compact_tokens)
  global_raw_target = select_prefix(global_logits, global_score_tokens)
  compact_processed_target = select_prefix(compact_processed, compact_tokens)
  global_processed_target = select_prefix(global_processed, global_score_tokens)
  compact_logp = replicate_prefix(compact_logps)
  global_logp = replicate_prefix(global_logps)
  compact_token_prefix = replicate_prefix(compact_tokens)
  global_token_prefix = replicate_prefix(global_tokens)
  values = {
      "raw_target": (
          np.asarray(compact_raw_target),
          np.asarray(global_raw_target),
      ),
      "sampled_token": (
          np.asarray(compact_token_prefix),
          np.asarray(global_token_prefix),
      ),
      "processed_target": (
          np.asarray(compact_processed_target),
          np.asarray(global_processed_target),
      ),
      "target_logprob": (np.asarray(compact_logp), np.asarray(global_logp)),
      "implied_normalizer": (
          np.asarray(compact_processed_target - compact_logp),
          np.asarray(global_processed_target - global_logp),
      ),
  }
  comparisons = {
      name: bitwise_comparison(left, right)
      for name, (left, right) in values.items()
  }
  negative = bitwise_comparison(
      values["target_logprob"][0],
      _flip_one_float_bit(values["target_logprob"][0]),
  )
  verdict = classify(comparisons, negative)
  report = {
      "schema_version": 1,
      "verdict": verdict,
      "claim_scope": "model-free-aval-discriminator-only",
      "contract": asdict(contract),
      "backend": jax.default_backend(),
      "device_kind": devices[0].device_kind,
      "device_ids": [int(device.id) for device in np.asarray(arranged).flat],
      "same_sample_callable": True,
      "same_score_callable": True,
      "measurement_count": len(comparisons),
      "expected_measurements": EXPECTED_MEASUREMENTS,
      "arrays": {
          "sample_decode": _sharding_contract(compact_logits),
          "sample_prompt": _sharding_contract(global_logits),
          "score_decode": _sharding_contract(compact_score_logits),
          "score_prompt": _sharding_contract(global_processed),
      },
      "lowered_hlo_sha256": {
          "sample_decode": _lowered_sha(
              sample, rng, mesh, compact_logits, compact_metadata
          ),
          "sample_prompt": _lowered_sha(
              sample, rng, mesh, global_logits, global_metadata
          ),
          **score_hlo,
      },
      "comparisons": comparisons,
      "negative_control": negative,
  }
  return report


def main() -> int:
  output = os.environ.get(REPORT_ENV, "").strip()
  if not output:
    print(f"[P38.AVAL] REFUSING: {REPORT_ENV} is required", flush=True)
    return 2
  path = Path(output)
  if path.exists():
    print(f"[P38.AVAL] REFUSING: evidence path exists: {path}", flush=True)
    return 2
  try:
    report = _run_hardware_probe()
  except Exception as exc:
    print(
        f"[P38.AVAL] INCONCLUSIVE exception={type(exc).__name__}: {exc}",
        flush=True,
    )
    raise
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(f"[P38.AVAL.JSON] {json.dumps(report, sort_keys=True)}", flush=True)
  print(f"[P38.AVAL] artifact={path}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
