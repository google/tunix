"""Host-side persistence for bounded P38 terminal-logit observations."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


P38_TAIL_CHECKPOINTS = (
    "raw_target_logit",
    "raw_log_normalizer",
    "processed_target_logit",
    "processed_log_normalizer",
    "observer_target_logprob",
    "production_target_logprob",
)


def write_tail_record(
    directory: str,
    state: dict[str, int],
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping,
    max_bytes: int,
) -> tuple[int, str]:
  """Persist one exclusive and self-checking terminal-tail record."""
  record_index = int(state.get("records", 0))
  current_bytes = int(state.get("bytes", 0))
  root = Path(directory)
  root.mkdir(parents=True, exist_ok=True)
  base = root / f"p38_tail_{record_index:06d}"
  npz_path = Path(str(base) + ".npz")
  json_path = Path(str(base) + ".json")
  if npz_path.exists() or json_path.exists():
    raise RuntimeError(f"P38 tail evidence collision at {base}")

  payload = io.BytesIO()
  np.savez(payload, **arrays)
  npz_bytes = payload.getvalue()
  npz_sha256 = hashlib.sha256(npz_bytes).hexdigest()
  record = {
      **metadata,
      "array_keys": sorted(arrays),
      "npz_sha256": npz_sha256,
      "record_index": record_index,
      "schema": "p38-tail-values-v1",
  }
  json_bytes = (
      json.dumps(record, sort_keys=True, indent=2, default=str) + "\n"
  ).encode()
  new_bytes = len(npz_bytes) + len(json_bytes)
  if current_bytes + new_bytes > int(max_bytes):
    raise RuntimeError("P38 tail evidence exceeded its output byte bound")

  for path, content in ((npz_path, npz_bytes), (json_path, json_bytes)):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
      if os.write(descriptor, content) != len(content):
        raise RuntimeError(f"P38 tail write was incomplete: {path.name}")
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
  if hashlib.sha256(npz_path.read_bytes()).hexdigest() != npz_sha256:
    raise RuntimeError("P38 tail NPZ self-check failed")
  state["records"] = record_index + 1
  state["bytes"] = current_bytes + new_bytes
  return record_index, npz_sha256
