#!/usr/bin/env python3
"""Create one valid bounded serving-capture fixture for shell postflight tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _write(directory: Path, stage: str, arrays: dict, meta: dict) -> None:
  base = directory / f"p38_serving_0000_{stage}"
  npz_path = Path(str(base) + ".npz")
  with npz_path.open("xb") as stream:
    np.savez(stream, **arrays)
  record = {
      "schema_version": 1,
      "stage": stage,
      "seq": 0,
      "arrays": sorted(arrays),
      "describe": {},
      "meta": meta,
      "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
  }
  Path(str(base) + ".json").write_text(
      json.dumps(record), encoding="utf-8"
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  args = parser.parse_args()
  args.directory.mkdir(parents=True, exist_ok=False)
  names = {
      "input_ids",
      "input_positions",
      "active_mask",
      "md_input_positions",
      "md_block_tables",
      "md_seq_lens",
      "md_query_start_loc",
      "md_request_distribution",
      "tokens_indices_selector",
      "rng",
  }
  pre = {name: np.arange(4, dtype=np.int32) for name in names}
  pre["active_mask"] = np.ones(4, dtype=np.bool_)
  pre["sampling_leaf_0000"] = np.arange(4, dtype=np.float32)
  _write(args.directory, "pre", pre, {
      "continue_decode_enabled": True,
      "caller_update_kv_cache": True,
      "output_update_kv_cache": True,
      "request_ids": ["request-0"],
      "request_ids_by_dp": {"0": ["request-0"]},
      "requests": [{
          "request_id": "request-0",
          "block_ids": [[7, 11]],
          "token_ids": [101, 102, 103],
      }],
      "kv_caches_spec": [{"shape": [16, 256, 2, 1, 128]}],
      "block_size": 256,
      "observed_max_prefix": 1791,
      "capture_min_prefix": 1788,
      "rpa_block_tuples": {
          "CANON_RPA_D": "128,512,128,512",
          "CANON_RPA_P": "128,512,128,512",
          "CANON_RPA_M": "128,512,128,512",
      },
      "kv_unified": False,
  })
  post_names = {
      "generated_tokens",
      "final_input_positions",
      "final_seq_lens",
      "logprob_token_ids",
      "logprob_values",
      "logprob_ranks",
  }
  post = {
      name: np.arange(4, dtype=np.int32).reshape(1, 4)
      for name in post_names
  }
  _write(args.directory, "post", post, {
      "actual_steps": 1,
      "completed_records": 1,
      "expected_max_records": 1,
  })
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
