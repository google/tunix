#!/usr/bin/env python3
"""Create a deterministic synthetic P38 capsule for one-host admission only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "p38-frozenlake-mismatch-capsule-v1"


def _sha256(value: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--prompt-length", type=int, default=1788)
  parser.add_argument("--completion-length", type=int, default=16)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  if args.prompt_length <= 0 or args.completion_length < 16:
    raise ValueError("prompt length must be positive and completion length at least 16")

  prompt_ids = (
      np.arange(args.prompt_length, dtype=np.int32) % np.int32(127)
      + np.int32(1000)
  )[None, :]
  completion_ids = (
      np.arange(args.completion_length, dtype=np.int32) % np.int32(113)
      + np.int32(5000)
  )[None, :]
  action_pattern = np.asarray(
      [False, True, True, False, False, True, True, False,
       False, True, True, False, False, True, True, False],
      dtype=np.bool_,
  )
  action_mask = np.resize(action_pattern, args.completion_length)[None, :]
  action_indices = np.flatnonzero(action_mask[0])
  target_prefixes = (
      args.prompt_length + action_indices - 1
  ).astype(np.int64).tolist()
  arrays = {
      "prompt_ids": prompt_ids,
      "prompt_mask": np.ones_like(prompt_ids, dtype=np.bool_),
      "completion_ids": completion_ids,
      "completion_valid_mask": np.ones_like(completion_ids, dtype=np.bool_),
      "action_mask": action_mask,
      "s_decode": np.zeros_like(completion_ids, dtype=np.float32),
      "s_prefill": np.zeros_like(completion_ids, dtype=np.float32),
      "t_old": np.zeros_like(completion_ids, dtype=np.float32),
      "policy_version": np.asarray([[0]], dtype=np.int32),
      "sampling_values": np.asarray([[0.7, 1.0, 0.0]], dtype=np.float32),
  }
  selected_rows = np.asarray([900001], dtype=np.int32)
  metadata = {
      "schema": SCHEMA,
      "selected_rows": selected_rows.tolist(),
      "provenance": {
          "kind": "SYNTHETIC_CANARY",
          "claim_ceiling": "one-host admission only; not a production replay",
          "prompt_length": args.prompt_length,
          "completion_length": args.completion_length,
          "target_prefixes": target_prefixes,
      },
      "arrays": {
          name: {
              "shape": list(value.shape),
              "dtype": str(value.dtype),
              "sha256": _sha256(value),
          }
          for name, value in arrays.items()
      },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
      args.output,
      selected_rows=selected_rows,
      metadata_json=np.frombuffer(
          json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
      ),
      **arrays,
  )
  print(
      json.dumps(
          {
              "status": "SYNTHETIC_CANARY_CREATED",
              "path": str(args.output.resolve()),
              "prompt_length": args.prompt_length,
              "completion_length": args.completion_length,
              "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
          },
          sort_keys=True,
      )
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
