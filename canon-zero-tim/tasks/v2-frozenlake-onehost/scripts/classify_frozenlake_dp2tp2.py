#!/usr/bin/env python3
"""Fail-closed classifier for a FrozenLake DP2xTP2 no-commit carrier."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


_FORWARD_RE = re.compile(
    r"^\[P32\.DP2\] (forward_group_(?:issued|done)) "
    r"group=(\d+)/8 .* n_real=\(([^)]*)\)$",
    re.MULTILINE,
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_RE = re.compile(r"[0-9a-f]{40}\Z")
_ARMS = {
    "r0": {"keep_tape": "0", "reduce_once": "0", "length_sort": "0"},
    "r1": {
        "keep_tape": "stream",
        "reduce_once": "0",
        "length_sort": "0",
    },
    "r2": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "0",
    },
    "r3": {
        "keep_tape": "stream",
        "reduce_once": "1",
        "length_sort": "1",
    },
}
_WORKLOADS = {
    "p45": {
        "name": "frozenlake-p45-onehost-dp2-tp2",
        "prompt": 4096,
        "response": 2048,
        "max_turns": 5,
        "vllm_hbm_utilization": 0.35,
        "min_landed_n_real": 2048,
        "min_action_tokens": 1024,
    },
    "m15": {
        "name": "frozenlake-m15-onehost-dp2-tp2",
        "prompt": 4096,
        "response": 8192,
        "max_turns": 15,
        "vllm_hbm_utilization": 0.37,
        # This is a landed-work threshold, not a claim that the 12,288-token
        # static cap was reached.  The classifier reports cap_coverage
        # separately and never promotes a CLI cap to measured work.
        "min_landed_n_real": 4096,
        "min_action_tokens": 4096,
    },
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"expected JSON object at {path}:{number}")
    rows.append(value)
  if not rows:
    raise ValueError(f"empty JSONL: {path}")
  return rows


def _exact_boundaries(rows: list[dict[str, Any]]) -> bool:
  if not rows:
    return False
  for row in rows:
    if row.get("verdict") != "PASS":
      return False
    if row.get("blocking_reds", []) or row.get("reds", []):
      return False
    boundaries = row.get("boundaries")
    if not isinstance(boundaries, dict) or not boundaries:
      return False
    for boundary in boundaries.values():
      if not isinstance(boundary, dict):
        return False
      if (
          boundary.get("valid") is not True
          or boundary.get("finite") is not True
          or boundary.get("differing_bytes") != 0
          or boundary.get("differing_elements") != 0
          or boundary.get("max_abs") != 0.0
      ):
        return False
  return True


def _peak_hbm(update: dict[str, Any]) -> tuple[int | None, int | None]:
  peaks = []
  limits = []
  for field in ("hbm_before", "hbm_after_reverse"):
    for snapshot in update.get(field, []):
      if not isinstance(snapshot, dict):
        continue
      peak = snapshot.get("peak_bytes_in_use")
      limit = snapshot.get("bytes_limit")
      if isinstance(peak, int):
        peaks.append(peak)
      if isinstance(limit, int):
        limits.append(limit)
  return (max(peaks) if peaks else None, min(limits) if limits else None)


def _forward_lengths(text: str) -> tuple[list[int], list[int], str | None]:
  matches = list(_FORWARD_RE.finditer(text))
  groups = []
  lengths = []
  marker = None
  for match in matches:
    this_marker = match.group(1)
    marker = this_marker if marker is None else marker
    if marker != this_marker:
      raise ValueError("mixed forward_group_issued/forward_group_done markers")
    groups.append(int(match.group(2)))
    fields = [field.strip() for field in match.group(3).split(",")]
    if len(fields) != 2:
      raise ValueError("DP2 forward receipt does not contain two rank lengths")
    lengths.extend(int(field) for field in fields)
  return groups, lengths, marker


def _anchor(
    registry: dict[str, Any], workload: str, arm: str
) -> tuple[list[float] | None, str | None]:
  if registry.get("schema") != "canon.v2-frozenlake-onehost.gradient-anchors.v1":
    raise ValueError("gradient anchor registry schema changed")
  anchors = registry.get("anchors")
  if not isinstance(anchors, dict):
    raise ValueError("gradient anchor registry has no anchors object")
  entry = anchors.get(f"{workload}:{arm}")
  if entry is None:
    return None, None
  if not isinstance(entry, dict):
    raise ValueError("gradient anchor entry is not an object")
  norms = entry.get("micro_gradient_norms")
  run_id = entry.get("run_id")
  if (
      not isinstance(norms, list)
      or len(norms) != 8
      or any(not isinstance(value, (int, float)) for value in norms)
      or not isinstance(run_id, str)
      or not run_id
  ):
    raise ValueError("gradient anchor entry is incomplete")
  return [float(value) for value in norms], run_id


def classify(
    root: Path,
    *,
    workload: str,
    arm: str,
    docker_exit: int,
    anchor_registry: Path,
    require_anchor: bool = False,
) -> dict[str, Any]:
  if workload not in _WORKLOADS:
    raise ValueError(f"unknown workload {workload!r}")
  if arm not in _ARMS:
    raise ValueError(f"unknown arm {arm!r}")
  spec = _WORKLOADS[workload]
  arm_spec = _ARMS[arm]
  required = {
      name: root / name
      for name in (
          "raw.log",
          "run_manifest.json",
          "runtime.json",
          "pre_alignment.jsonl",
          "alignment.jsonl",
          "updates.json",
      )
  }
  missing = [
      name
      for name, path in required.items()
      if not path.is_file() or path.stat().st_size == 0
  ]
  if missing:
    return {
        "schema": "canon.v2-frozenlake-onehost.classification.v1",
        "verdict": "INCONCLUSIVE",
        "workload": workload,
        "arm": arm,
        "reasons": [f"missing_or_empty:{name}" for name in missing],
    }

  raw = required["raw.log"].read_text(encoding="utf-8", errors="replace")
  manifest = _json(required["run_manifest.json"])
  runtime = _json(required["runtime.json"])
  pre = _jsonl(required["pre_alignment.jsonl"])
  alignment = _jsonl(required["alignment.jsonl"])
  update = _json(required["updates.json"])
  reasons = []

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  expected_manifest = {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "workload": workload,
      "workload_name": spec["name"],
      "arm": arm,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp2",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "global_prompts": 4,
      "num_generations": 4,
      "global_trajectories": 16,
      "gradient_groups": 8,
      "local_m": 256,
      "global_m": 512,
      "max_prompt_length": spec["prompt"],
      "max_response_length": spec["response"],
      "max_turns": spec["max_turns"],
      "data_shuffle_seed": 42,
      "vllm_global_seed": 0,
      "vllm_hbm_utilization": spec["vllm_hbm_utilization"],
      "selectors": arm_spec,
      "checked_vma": True,
      "wandb_mode": "disabled",
      "classification_mode": "certify" if require_anchor else "measure",
  }
  changed = {
      key: manifest.get(key)
      for key, expected in expected_manifest.items()
      if manifest.get(key) != expected
  }
  require(not changed, f"manifest:{changed}")
  require(_SOURCE_RE.fullmatch(str(manifest.get("source_commit"))) is not None,
          "manifest.source_commit")
  for key in (
      "source_diff_sha256",
      "image_id_sha256",
      "model_snapshot_sha256",
      "dataset_train_sha256",
      "dataset_test_sha256",
      "runner_sha256",
  ):
    require(_SHA256_RE.fullmatch(str(manifest.get(key))) is not None,
            f"manifest.{key}")

  require(docker_exit == 0, f"docker_exit={docker_exit}")
  require(runtime.get("verdict") == "PASS", "runtime")
  require(_exact_boundaries(pre), "pre_alignment_not_exact")
  require(_exact_boundaries(alignment), "alignment_not_exact")
  require(len(pre) == 1, f"pre_alignment_records={len(pre)}")
  require(len(alignment) == 8, f"alignment_records={len(alignment)}")
  require(
      "S_decode_vs_S_prefill" in pre[0].get("boundaries", {})
      and "S_prefill_vs_T_old" in pre[0].get("boundaries", {}),
      "pre_alignment_boundary_inventory",
  )
  action_tokens = pre[0].get("N_action")
  input_hashes = pre[0].get("hashes")
  valid_input_hashes = (
      isinstance(input_hashes, dict)
      and all(
          _SHA256_RE.fullmatch(str(input_hashes.get(key))) is not None
          for key in (
              "tokens",
              "action_mask",
              "S_decode",
              "S_prefill",
              "T_old",
          )
      )
  )
  require(valid_input_hashes, "input_hash_inventory")
  require(
      isinstance(action_tokens, int)
      and action_tokens >= spec["min_action_tokens"],
      f"landed_action_tokens={action_tokens}",
  )

  expected_update = {
      "contract_name": spec["name"],
      "dp_size": 2,
      "tp_size": 2,
      "global_m": 512,
      "verdict": "PASS",
      "mode": "backward-no-commit",
      "microsteps": 8,
      "commits": 0,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "gradient_finite": True,
      "dp_replicas_exact": True,
      "dp_axis": "dp",
      "dp_reduction_transactions": 1 if arm in ("r2", "r3") else 8,
      # Fixed DPRank reduction is one ordered reduce plus one broadcast.
      "dp_reduction_rounds_per_transaction": 2,
      "dp_rank_pullbacks_per_transaction": 2,
      "dp_pullback_invocations_per_transaction": 1,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "optimizer_placement": "pinned-host-offload",
      "optimizer_memory_kinds_before": ["pinned_host"],
  }
  wrong_update = {
      key: update.get(key)
      for key, expected in expected_update.items()
      if update.get(key) != expected
  }
  require(not wrong_update, f"update:{wrong_update}")
  activity = update.get("gradient_activity")
  norms = update.get("micro_gradient_norms")
  valid_norms = (
      isinstance(activity, list)
      and len(activity) == 8
      and all(isinstance(value, bool) for value in activity)
      and any(activity)
      and isinstance(norms, list)
      and len(norms) == 8
      and all(
          isinstance(value, (int, float)) and math.isfinite(value)
          for value in norms
      )
      and all((float(value) > 0.0) == active for value, active in zip(norms, activity))
  )
  require(valid_norms, "gradient_activity_or_norms")

  try:
    groups, n_real, forward_marker = _forward_lengths(raw)
  except ValueError as exc:
    groups, n_real, forward_marker = [], [], None
    reasons.append(f"forward_receipts:{exc}")
  require(groups == list(range(1, 9)), f"forward_groups={groups}")
  require(len(n_real) == 16, f"n_real_count={len(n_real)}")
  expected_marker = "forward_group_done" if arm == "r0" else "forward_group_issued"
  require(forward_marker == expected_marker, f"forward_marker={forward_marker}")
  max_n_real = max(n_real) if n_real else 0
  max_static = spec["prompt"] + spec["response"]
  require(max_n_real <= max_static, f"n_real_over_cap={max_n_real}>{max_static}")
  require(
      max_n_real >= spec["min_landed_n_real"],
      f"landed_n_real_max={max_n_real}",
  )
  group_chunks = [
      (max(n_real[index:index + 2]) + 255) // 256
      for index in range(0, len(n_real), 2)
  ] if len(n_real) == 16 else []

  p66_receipts = raw.count("[P66.VMA] outer_check_enabled")
  require(p66_receipts >= 1, f"p66_outer_check_receipts={p66_receipts}")
  seed_receipt = raw.count(
      "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
      "vllm_global_seed=0 per_request_seed=unsupported"
  )
  require(seed_receipt == 1, f"seed_receipts={seed_receipt}")
  disabled_wandb_receipt = raw.count("[V2.FL.WANDB] DISABLED_LOCAL_PASS")
  require(
      disabled_wandb_receipt == 1,
      f"disabled_wandb_receipts={disabled_wandb_receipt}",
  )
  sampler_receipt = raw.count(
      "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
      "use_rollout_logps=1 tis_weights=absent"
  )
  require(sampler_receipt == 1, f"sampler_receipts={sampler_receipt}")
  require("Traceback (most recent call last)" not in raw, "traceback")
  require("NUMERICAL_REJECT" not in raw, "numerical_reject")

  peak_hbm, hbm_limit = _peak_hbm(update)
  hbm_ratio = (
      peak_hbm / hbm_limit
      if isinstance(peak_hbm, int) and isinstance(hbm_limit, int) and hbm_limit
      else None
  )
  require(hbm_ratio is not None, "hbm_receipt")
  require(hbm_ratio is not None and hbm_ratio <= 0.90,
          f"hbm_headroom_ratio={hbm_ratio}")

  anchor_norms, anchor_run_id = _anchor(
      _json(anchor_registry), workload, arm
  )
  anchor_exact = anchor_norms is not None and norms == anchor_norms
  if require_anchor and anchor_norms is None:
    reasons.append("gradient_anchor_unregistered")
  if anchor_norms is not None:
    require(anchor_exact, "gradient_anchor_bitwise")

  verdict = (
      "FAIL"
      if reasons
      else "PASS"
      if anchor_norms is not None
      else "MEASUREMENT_ONLY"
  )
  return {
      "schema": "canon.v2-frozenlake-onehost.classification.v1",
      "verdict": verdict,
      "workload": workload,
      "arm": arm,
      "claim_level": "onehost-qwen8b-dp2-tp2-no-commit",
      "classification_mode": "certify" if require_anchor else "measure",
      "claim_excludes": [
          "DP8xTP8",
          "Pathways",
          "GKE",
          "optimizer-commit",
          "convergence",
          "target-performance",
      ],
      "reasons": reasons,
      "zero_tim": {
          "pre_alignment_records": len(pre),
          "post_alignment_records": len(alignment),
          "strict_exact": _exact_boundaries(pre + alignment),
          "input_hashes": input_hashes,
          "action_tokens": action_tokens,
      },
      "gradient": {
          "micro_gradient_norms": norms,
          "activity": activity,
          "anchor_registered": anchor_norms is not None,
          "anchor_exact": anchor_exact,
          "anchor_run_id": anchor_run_id,
      },
      "landed_shape": {
          "n_real": n_real,
          "min_n_real": min(n_real) if n_real else None,
          "max_n_real": max_n_real if n_real else None,
          "group_chunks": group_chunks,
          "max_group_chunks": max(group_chunks) if group_chunks else None,
          "static_cap": max_static,
          "cap_coverage": max_n_real == max_static,
      },
      "hbm": {
          "peak_bytes": peak_hbm,
          "limit_bytes": hbm_limit,
          "peak_to_limit": hbm_ratio,
          "max_admitted_ratio": 0.90,
      },
      "receipts": {
          "p66_outer_check_enabled": p66_receipts,
          "deterministic_rollout_seed": seed_receipt,
          "disabled_local_wandb": disabled_wandb_receipt,
          "sampler_contract": sampler_receipt,
          "forward_marker": forward_marker,
          "dp_reduction_transactions": update.get("dp_reduction_transactions"),
      },
      "artifacts": {
          name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
          for name, path in required.items()
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", required=True, type=Path)
  parser.add_argument("--workload", required=True, choices=sorted(_WORKLOADS))
  parser.add_argument("--arm", required=True, choices=sorted(_ARMS))
  parser.add_argument("--docker-exit", required=True, type=int)
  parser.add_argument("--anchor-registry", required=True, type=Path)
  parser.add_argument("--require-anchor", action="store_true")
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = classify(
      args.root,
      workload=args.workload,
      arm=args.arm,
      docker_exit=args.docker_exit,
      anchor_registry=args.anchor_registry,
      require_anchor=args.require_anchor,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True, separators=(",", ":")))
  return 0 if result["verdict"] in ("PASS", "MEASUREMENT_ONLY") else 1


if __name__ == "__main__":
  sys.exit(main())
