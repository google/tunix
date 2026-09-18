#!/usr/bin/env python3
"""Exact-image contract probe for the P58.22 runner hook."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np


def _load(path: Path):
  sys.path.insert(0, str(path.parent))
  spec = importlib.util.spec_from_file_location("p58_continue_kv_runner", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load runner: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--runner", type=Path, required=True)
  args = parser.parse_args()
  runner = _load(args.runner)
  required = (
      "_p58_continue_kv_after_burst",
      "_p38_kv_observer_after_standard",
      "_p38_kv_observer_effective_sharding",
      "_P58_CONTINUE_KV_DIAGNOSTIC",
      "_P58_CONTINUE_KV_MIN_PREFIX",
      "_P58_CONTINUE_KV_MAX_PREFIX",
  )
  if any(not hasattr(runner, name) for name in required):
    raise RuntimeError("P58.22 hook inventory is incomplete")

  class _Device:
    platform = "tpu"
    process_index = 0
    id = 3
    coords = (1, 0, 0)
    core_on_chip = 0

  class _Sharding:

    def devices_indices_map(self, shape):
      return {
          _Device(): (
              slice(None), slice(0, shape[1]), slice(2, 4),
          )
      }

  effective = runner._p38_kv_observer_effective_sharding(
      SimpleNamespace(shape=(8, 16, 8), sharding=_Sharding())
  )
  if effective != {
      "schema": "p38-effective-device-sharding-v1",
      "global_shape": [8, 16, 8],
      "devices": [{
          "platform": "tpu",
          "process_index": 0,
          "id": 3,
          "coords": [1, 0, 0],
          "core_on_chip": 0,
          "index": [
              {"kind": "slice", "start": 0, "stop": 8, "step": 1},
              {"kind": "slice", "start": 0, "stop": 16, "step": 1},
              {"kind": "slice", "start": 2, "stop": 4, "step": 1},
          ],
      }],
  }:
    raise RuntimeError("P58.22 effective sharding normalization drifted")

  negative_env = dict(os.environ)
  negative_env["CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC"] = "0"
  negative_env["PYTHONPATH"] = os.pathsep.join(
      [str(args.runner.parent), negative_env.get("PYTHONPATH", "")]
  ).rstrip(os.pathsep)
  negative = subprocess.run(
      [
          sys.executable,
          "-c",
          "import runpy,sys; runpy.run_path(sys.argv[1], run_name='p58_negative')",
          str(args.runner),
      ],
      env=negative_env,
      check=False,
      capture_output=True,
      text=True,
  )
  negative_text = negative.stdout + negative.stderr
  if negative.returncode == 0 or "requires serving capture" not in negative_text:
    raise RuntimeError(
        "ordinary P38 KV observer no longer requires serving capture"
    )

  saved = {
      "enabled": runner._P58_CONTINUE_KV_DIAGNOSTIC,
      "directory": runner._P38_KV_OBSERVER_DIR,
      "max_candidates": runner._P38_KV_OBSERVER_MAX_CANDIDATES,
      "candidates": dict(runner._P38_KV_OBSERVER_CANDIDATES),
      "a_records": list(runner._P38_KV_OBSERVER_A_RECORDS),
      "matched_a": set(runner._P38_KV_OBSERVER_MATCHED_A),
      "state": dict(runner._P58_CONTINUE_KV_STATE),
  }
  try:
    runner._P58_CONTINUE_KV_DIAGNOSTIC = "1"
    runner._P38_KV_OBSERVER_DIR = "/tmp/p58-continue-kv-contract"
    runner._P38_KV_OBSERVER_MAX_CANDIDATES = 1
    runner._P38_KV_OBSERVER_CANDIDATES.clear()
    runner._P58_CONTINUE_KV_STATE.update({
        "bursts": 0,
        "candidates": 0,
        "clean_empty_reported": set(),
        "clean_prefix_reported": set(),
    })
    request = SimpleNamespace(num_computed_tokens=2280, block_ids=[[3, 4]])
    owner = SimpleNamespace(requests={"req-a": request})
    scheduler = SimpleNamespace(num_scheduled_tokens={"req-a": 8})
    output = SimpleNamespace(req_ids=["req-a"])
    with mock.patch.object(runner, "_p38_primary_pages", return_value=[3, 4]), \
         mock.patch.object(runner, "_p38_kv_observer_after_standard") as capture:
      runner._p58_continue_kv_after_burst(
          owner, scheduler, {0: ["req-a"]}, output
      )
    candidate = runner._P38_KV_OBSERVER_CANDIDATES.get("req-a")
    if candidate is None or candidate["tag_prefix"] != 2280:
      raise RuntimeError("P58.22 did not select the signed seam candidate")
    capture.assert_called_once_with(
        owner, scheduler, {0: ["req-a"]}, frozenset(), output
    )

    with mock.patch.object(runner, "_p38_kv_observer_after_standard"):
      try:
        runner._p58_continue_kv_after_burst(
            owner, scheduler, {0: ["req-a", "req-b"]},
            SimpleNamespace(req_ids=["req-a", "req-b"]),
        )
      except RuntimeError as exc:
        if "exactly one active request" not in str(exc):
          raise
      else:
        raise RuntimeError("P58.22 admitted a multi-request diagnostic")

    # A prompt-logprob rescore intentionally need not execute the final input
    # token.  P58 must therefore capture B as soon as the clean KV contains the
    # entire selected A prefix; the ordinary P38 observer retains its original
    # full-request requirement.
    target_tokens = np.arange(2472, dtype=np.int32)
    full_tokens = np.arange(4000, dtype=np.int32).reshape(1, -1)
    clean_request = SimpleNamespace(
        num_computed_tokens=2304,
        num_tokens=4000,
        block_ids=[[int(value) for value in range(250)]],
    )
    clean_owner = SimpleNamespace(
        requests={"clean-b": clean_request},
        input_batch=SimpleNamespace(
            req_id_to_index={"clean-b": 0}, token_ids_cpu=full_tokens
        ),
    )
    clean_scheduler = SimpleNamespace(
        num_scheduled_tokens={"clean-b": 256}
    )
    live_record = {
        "record_index": 0,
        "request_id": "live-a",
        # P58 is a single-shot observer and therefore always uses round zero;
        # keep the assembled-overlay probe on the same schema as real A
        # records now that the shared M15 observer is multiround-aware.
        "diagnostic_round": 0,
        "target_token_ids": target_tokens,
        "target_seq_len": int(target_tokens.size),
    }
    runner._P38_KV_OBSERVER_A_RECORDS[:] = [live_record]
    runner._P38_KV_OBSERVER_MATCHED_A.clear()
    with mock.patch.object(
        runner, "_p38_primary_pages", return_value=list(range(250))
    ), mock.patch.object(runner, "_p38_kv_observer_capture") as capture_b:
      runner._p38_kv_observer_after_standard(
          clean_owner,
          clean_scheduler,
          {0: ["clean-b"]},
          frozenset({"clean-b"}),
          None,
      )
    if capture_b.call_count != 1 or runner._P38_KV_OBSERVER_MATCHED_A != {0}:
      raise RuntimeError("P58.22 did not capture a complete partial-rescore B")

    mismatch_tokens = full_tokens.copy()
    mismatch_tokens[0, 100] = -7
    mismatch_owner = SimpleNamespace(
        requests={"mismatch-b": clean_request},
        input_batch=SimpleNamespace(
            req_id_to_index={"mismatch-b": 0}, token_ids_cpu=mismatch_tokens
        ),
    )
    mismatch_scheduler = SimpleNamespace(
        num_scheduled_tokens={"mismatch-b": 256}
    )
    runner._P38_KV_OBSERVER_MATCHED_A.clear()
    runner._P58_CONTINUE_KV_STATE["clean_prefix_reported"] = set()
    with mock.patch.object(
        runner, "_p38_primary_pages", return_value=list(range(250))
    ), mock.patch.object(runner, "_p38_kv_observer_capture") as mismatch_capture:
      runner._p38_kv_observer_after_standard(
          mismatch_owner,
          mismatch_scheduler,
          {0: ["mismatch-b"]},
          frozenset({"mismatch-b"}),
          None,
      )
    if mismatch_capture.called or runner._P38_KV_OBSERVER_MATCHED_A:
      raise RuntimeError("P58.22 captured a mismatched clean prefix")
    if "mismatch-b" not in runner._P58_CONTINUE_KV_STATE[
        "clean_prefix_reported"
    ]:
      raise RuntimeError("P58.22 did not report the mismatched clean prefix")

    runner._P38_KV_OBSERVER_A_RECORDS.clear()
    runner._P58_CONTINUE_KV_STATE["clean_empty_reported"] = set()
    with mock.patch.object(runner, "_p38_kv_observer_capture") as empty_capture:
      runner._p38_kv_observer_after_standard(
          clean_owner,
          clean_scheduler,
          {0: ["clean-b"]},
          frozenset({"clean-b"}),
          None,
      )
    if empty_capture.called:
      raise RuntimeError("P58.22 captured B without a visible A record")
    if "clean-b" not in runner._P58_CONTINUE_KV_STATE[
        "clean_empty_reported"
    ]:
      raise RuntimeError("P58.22 did not report the missing A record")
    runner._P38_KV_OBSERVER_A_RECORDS[:] = [live_record]

    runner._P58_CONTINUE_KV_DIAGNOSTIC = "0"
    runner._P38_KV_OBSERVER_MATCHED_A.clear()
    with mock.patch.object(
        runner, "_p38_primary_pages", return_value=list(range(250))
    ), mock.patch.object(runner, "_p38_kv_observer_capture") as generic_capture:
      runner._p38_kv_observer_after_standard(
          clean_owner,
          clean_scheduler,
          {0: ["clean-b"]},
          frozenset({"clean-b"}),
          None,
      )
    if generic_capture.called or runner._P38_KV_OBSERVER_MATCHED_A:
      raise RuntimeError("ordinary P38 admitted a partial prompt rescore")
  finally:
    runner._P58_CONTINUE_KV_DIAGNOSTIC = saved["enabled"]
    runner._P38_KV_OBSERVER_DIR = saved["directory"]
    runner._P38_KV_OBSERVER_MAX_CANDIDATES = saved["max_candidates"]
    runner._P38_KV_OBSERVER_CANDIDATES.clear()
    runner._P38_KV_OBSERVER_CANDIDATES.update(saved["candidates"])
    runner._P38_KV_OBSERVER_A_RECORDS[:] = saved["a_records"]
    runner._P38_KV_OBSERVER_MATCHED_A.clear()
    runner._P38_KV_OBSERVER_MATCHED_A.update(saved["matched_a"])
    runner._P58_CONTINUE_KV_STATE.clear()
    runner._P58_CONTINUE_KV_STATE.update(saved["state"])
  print("P58_CONTINUE_KV_OVERLAY_PASS cases=8/8")


if __name__ == "__main__":
  main()
