#!/usr/bin/env python3
"""Classify bounded M15 APC seam observations on the real DP8xTP8 carrier.

The generic P38 seam classifier requires every red action to have a standard-
path record.  M15 intentionally keeps CANON_CONTINUE_DECODE=8, so that
requirement is impossible: most suffix actions are produced inside the device
loop.  This classifier is stricter about the claim instead of weakening the
join.  It localizes only red actions with exact A and B standard-path records,
requires at least one completion-position-zero anchor, and accounts for every
other red action as unobserved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class M15WideSeamError(RuntimeError):
  pass


ROOT = Path(__file__).resolve().parents[4]
P38_CLASSIFIER_PATH = ROOT / (
    "canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_seam.py"
)
_P38_SPEC = importlib.util.spec_from_file_location(
    "classify_p38_seam_for_m15", P38_CLASSIFIER_PATH
)
assert _P38_SPEC and _P38_SPEC.loader
P38 = importlib.util.module_from_spec(_P38_SPEC)
_P38_SPEC.loader.exec_module(P38)

_SEAM_REQUIRED = {
    "row_indices", "positions", "token_ids", "request_ordinals",
    "token_prefix_sha256", "layer_fingerprints", "final_norm_fingerprints",
}
_SEAM_OPTIONAL = {"final_hidden_rows"}
_TAIL_REQUIRED = {
    "row_indices", "positions", "token_ids", "request_ordinals",
    "token_prefix_sha256", "logit_row_indices", "target_ids", "tail_values",
}
_TAIL_CHECKPOINTS = (
    "raw_target_logit",
    "raw_log_normalizer",
    "processed_target_logit",
    "processed_log_normalizer",
    "observer_target_logprob",
    "production_target_logprob",
)
_LAYER_CHECKPOINTS = ("layer_input", "layer_output")
_FULL_CHECKPOINTS = (
    "layer_input", "input_norm", "q_proj", "k_proj", "v_proj",
    "q_norm", "k_norm", "q_post_rope", "k_post_rope", "rpa_output",
    "o_proj", "attention_residual", "post_attention_norm", "mlp_output",
    "layer_output",
)
_MAX_ANCHORS = 16


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise M15WideSeamError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _same_candidate(left: dict[str, Any], right: dict[str, Any]) -> bool:
  scalar_keys = (
      "position", "token_id", "checkpoint_names", "layer_indices",
      "request_id", "target_id", "source_token_id",
  )
  for key in scalar_keys:
    if left.get(key) != right.get(key):
      return False
  for key in ("layer_fingerprints", "final_norm_fingerprints", "values"):
    if key in left or key in right:
      if key not in left or key not in right:
        return False
      if not np.array_equal(left[key], right[key]):
        return False
  return True


def _resolve_aliases(candidates: list[dict[str, Any]], label: str) -> dict[str, Any]:
  _require(candidates, f"no candidates for {label}")
  first = candidates[0]
  _require(
      all(_same_candidate(first, candidate) for candidate in candidates[1:]),
      f"numerically conflicting aliases for {label}",
  )
  return min(candidates, key=lambda item: (item["call_index"], item["record_index"]))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
  with np.load(path, allow_pickle=False) as archive:
    return {name: np.array(archive[name], copy=True) for name in archive.files}


def _load_seam_candidates(
    directory: Path, mode: str
) -> tuple[dict[tuple[int, bytes, str], list[dict[str, Any]]], dict[str, Any]]:
  paths = sorted(directory.glob("p38_seam_*.json"))
  _require(paths, "M15 wide seam observer produced no records")
  result: dict[tuple[int, bytes, str], list[dict[str, Any]]] = {}
  arms = set()
  record_bytes = 0
  for path in paths:
    record = json.loads(path.read_text(encoding="utf-8"))
    index = int(record.get("record_index", -1))
    _require(
        record.get("schema") == "p38-seam-fingerprint-v1"
        and record.get("observer_mode") == mode
        and index >= 0
        and path.name == f"p38_seam_{index:06d}.json",
        f"invalid M15 seam record: {path.name}",
    )
    npz_path = directory / f"p38_seam_{index:06d}.npz"
    _require(npz_path.is_file(), f"M15 seam NPZ is absent: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"M15 seam NPZ SHA failed: {npz_path.name}")
    arrays = _load_npz(npz_path)
    _require(
        _SEAM_REQUIRED <= set(arrays)
        and set(arrays) <= _SEAM_REQUIRED | _SEAM_OPTIONAL,
        f"M15 seam array inventory drifted: {npz_path.name}",
    )
    rows = arrays["row_indices"].reshape(-1)
    positions = arrays["positions"].reshape(-1)
    token_ids = arrays["token_ids"].reshape(-1)
    ordinals = arrays["request_ordinals"].reshape(-1)
    hashes = arrays["token_prefix_sha256"].reshape(-1)
    layer_values = arrays["layer_fingerprints"]
    final_values = arrays["final_norm_fingerprints"]
    _require(
        rows.size == positions.size == token_ids.size == ordinals.size
        == hashes.size == layer_values.shape[0] == final_values.shape[0],
        f"M15 seam row geometry drifted: {npz_path.name}",
    )
    _require(layer_values.ndim == 4 and layer_values.shape[-1] == 8,
             f"M15 seam fingerprint geometry drifted: {npz_path.name}")
    _require(final_values.shape == (rows.size, 8),
             f"M15 final-norm geometry drifted: {npz_path.name}")
    checkpoint_names = [str(value) for value in record.get("checkpoint_names", ())]
    layer_indices = [int(value) for value in record.get("layer_indices", ())]
    _require(
        layer_values.shape[1] == len(layer_indices)
        and layer_values.shape[2] == len(checkpoint_names),
        f"M15 seam metadata geometry drifted: {path.name}",
    )
    requests = record.get("requests")
    _require(isinstance(requests, list), f"M15 seam requests are absent: {path.name}")
    arm = str(record.get("arm"))
    diagnostic_round = int(record.get("diagnostic_round", -1))
    call_index = int(record.get("call_index", -1))
    _require(arm in ("A", "B") and diagnostic_round == 0 and call_index >= 0,
             f"M15 seam provenance drifted: {path.name}")
    arms.add(arm)
    for offset in range(rows.size):
      ordinal = int(ordinals[offset])
      _require(0 <= ordinal < len(requests),
               f"M15 seam request ordinal overflowed: {path.name}")
      request = requests[ordinal]
      _require(isinstance(request, dict) and request.get("request_id"),
               f"M15 seam request metadata is invalid: {path.name}")
      key = (diagnostic_round, bytes(hashes[offset]), arm)
      result.setdefault(key, []).append({
          "record_index": index,
          "row_offset": offset,
          "row_index": int(rows[offset]),
          "position": int(positions[offset]),
          "token_id": int(token_ids[offset]),
          "call_index": call_index,
          "request_id": str(request["request_id"]),
          "request": request,
          "checkpoint_names": checkpoint_names,
          "layer_indices": layer_indices,
          "layer_fingerprints": layer_values[offset],
          "final_norm_fingerprints": final_values[offset],
          "record_geometry": {
              "gather_bucket": int(record.get("gather_bucket", -1)),
              "layer_fingerprint_shape": record.get("layer_fingerprint_shape"),
              "final_fingerprint_shape": record.get("final_fingerprint_shape"),
              "layer_fingerprint_sharding": record.get(
                  "layer_fingerprint_sharding"),
              "final_fingerprint_sharding": record.get(
                  "final_fingerprint_sharding"),
          },
      })
    record_bytes += path.stat().st_size + npz_path.stat().st_size
  _require(arms == {"A", "B"}, f"M15 seam observer arms are incomplete: {arms}")
  return result, {
      "records": len(paths),
      "bytes": record_bytes,
      "arms": sorted(arms),
      "unique_keys": len(result),
  }


def _load_tail_candidates(
    directory: Path,
) -> tuple[dict[tuple[int, bytes, str, int], list[dict[str, Any]]], dict[str, Any]]:
  paths = sorted(directory.glob("p38_tail_*.json"))
  _require(paths, "M15 terminal-tail observer produced no records")
  result: dict[tuple[int, bytes, str, int], list[dict[str, Any]]] = {}
  arms = set()
  for path in paths:
    record = json.loads(path.read_text(encoding="utf-8"))
    index = int(record.get("record_index", -1))
    _require(
        record.get("schema") == "p38-tail-values-v1"
        and index >= 0
        and path.name == f"p38_tail_{index:06d}.json",
        f"invalid M15 tail record: {path.name}",
    )
    npz_path = directory / f"p38_tail_{index:06d}.npz"
    _require(npz_path.is_file(), f"M15 tail NPZ is absent: {npz_path.name}")
    _require(_sha256(npz_path) == record.get("npz_sha256"),
             f"M15 tail NPZ SHA failed: {npz_path.name}")
    arrays = _load_npz(npz_path)
    _require(set(arrays) == _TAIL_REQUIRED,
             f"M15 tail array inventory drifted: {npz_path.name}")
    rows = arrays["row_indices"].reshape(-1)
    positions = arrays["positions"].reshape(-1)
    tokens = arrays["token_ids"].reshape(-1)
    hashes = arrays["token_prefix_sha256"].reshape(-1)
    targets = arrays["target_ids"].reshape(-1)
    values = arrays["tail_values"]
    _require(
        rows.size == positions.size == tokens.size == hashes.size
        == targets.size == values.shape[0]
        and values.shape[1:] == (len(_TAIL_CHECKPOINTS),),
        f"M15 tail geometry drifted: {npz_path.name}",
    )
    _require(tuple(record.get("checkpoint_names", ())) == _TAIL_CHECKPOINTS,
             f"M15 tail checkpoints drifted: {path.name}")
    arm = str(record.get("arm"))
    diagnostic_round = int(record.get("diagnostic_round", -1))
    call_index = int(record.get("call_index", -1))
    _require(arm in ("A", "B") and diagnostic_round == 0 and call_index >= 0,
             f"M15 tail provenance drifted: {path.name}")
    arms.add(arm)
    for offset in range(rows.size):
      key = (diagnostic_round, bytes(hashes[offset]), arm, int(targets[offset]))
      result.setdefault(key, []).append({
          "record_index": index,
          "row_offset": offset,
          "row_index": int(rows[offset]),
          "position": int(positions[offset]),
          "source_token_id": int(tokens[offset]),
          "target_id": int(targets[offset]),
          "call_index": call_index,
          "checkpoint_names": list(_TAIL_CHECKPOINTS),
          "values": values[offset],
      })
  _require(arms == {"A", "B"}, f"M15 tail observer arms are incomplete: {arms}")
  return result, {"records": len(paths), "arms": sorted(arms), "unique_keys": len(result)}


def _load_alignment(path: Path) -> dict[str, Any]:
  rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
          if line.strip()]
  _require(len(rows) == 1, "M15 observer requires exactly one alignment round")
  row = rows[0]
  ab = row.get("boundaries", {}).get("S_decode_vs_S_prefill", {})
  bc = row.get("boundaries", {}).get("S_prefill_vs_T_old", {})
  _require(ab.get("valid") is True and ab.get("finite") is True,
           "M15 observer A-B boundary is invalid")
  _require(bc.get("valid") is True and bc.get("finite") is True,
           "M15 observer B-C boundary is invalid")
  _require(int(bc.get("differing_bytes", -1)) == 0,
           "M15 observer changed the independent B-C boundary")
  return row


def _source_anchor(checkpoint: str) -> dict[str, Any]:
  if checkpoint in ("layer_input", "layer_output", "final_norm"):
    relative = Path("canon-zero-tim/patches/tpu_inference/17-qwen3-p38-seam-observer.patch")
    needle = {
        "layer_input": "+            layer_input = x",
        "layer_output": "+                        fingerprint_tensor_rows(x)",
        "final_norm": "+            fingerprint_tensor_rows(x),",
    }[checkpoint]
  elif checkpoint in _TAIL_CHECKPOINTS:
    relative = Path("canon-zero-tim/patches/tpu_inference/19-tpu-runner-p38-terminal-tail.patch")
    needle = checkpoint
  else:
    relative = Path("canon-zero-tim/patches/tpu_inference/17-qwen3-p38-seam-observer.patch")
    needle = {
        "input_norm": "hidden_states = self.input_layernorm(x)",
        "q_proj": "_seam_q_proj = q",
        "k_proj": "_seam_k_proj = k",
        "v_proj": "_seam_v_proj = v",
        "q_norm": "_seam_q_norm = q",
        "k_norm": "_seam_k_norm = k",
        "q_post_rope": "_seam_q_post_rope = q",
        "k_post_rope": "_seam_k_post_rope = k",
        "rpa_output": "outputs, kv_cache = self.attention",
        "o_proj": "o = self.o_proj(outputs)",
        "attention_residual": "attention_residual = attn_projected + x",
        "post_attention_norm": "post_attention_norm = self.post_attention_layernorm",
        "mlp_output": "mlp_output = self.mlp(post_attention_norm)",
    }.get(checkpoint, checkpoint)
  path = ROOT / relative
  lines = path.read_text(encoding="utf-8").splitlines()
  matches = [index + 1 for index, line in enumerate(lines) if needle in line]
  _require(matches, f"source anchor is absent for checkpoint {checkpoint}")
  return {"file": str(relative), "line": matches[0], "anchor": needle}


def _last_exact_before(first: dict[str, Any], mode: str) -> dict[str, Any] | None:
  checkpoint = str(first["checkpoint"])
  if first.get("layer") is None:
    order = ["final_norm", *_TAIL_CHECKPOINTS]
    index = order.index(checkpoint)
    if index == 0:
      return None
    return {"layer": None, "checkpoint": order[index - 1]}
  layer = int(first["layer"])
  checkpoints = (
      list(_LAYER_CHECKPOINTS)
      if mode == "layer"
      else list(_FULL_CHECKPOINTS)
  )
  index = checkpoints.index(checkpoint)
  if index:
    return {"layer": layer, "checkpoint": checkpoints[index - 1]}
  if layer:
    return {"layer": layer - 1, "checkpoint": checkpoints[-1]}
  return None


def _ledger_receipts(
    path: Path | None, anchors: Iterable[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
  if path is None:
    return {}
  _require(path.is_file(), f"M15 replay ledger is absent: {path}")
  wanted = {}
  for anchor in anchors:
    for arm in ("A", "B"):
      item = anchor[arm.lower()]
      wanted[(arm, int(item["call_index"]), str(item["request_id"]))] = None
  with path.open(encoding="utf-8") as handle:
    for line in handle:
      if not line.strip():
        continue
      record = json.loads(line)
      _require(record.get("schema") == "m15-apc-serving-envelope-v1",
               "M15 replay ledger schema drifted")
      serving_arm = str(record.get("serving_arm"))
      call_index = int(record.get("call_index", -1))
      for request in record.get("requests", ()):
        key = (serving_arm, call_index, str(request.get("request_id")))
        if key in wanted:
          _require(wanted[key] is None, f"duplicate M15 ledger join: {key}")
          wanted[key] = request
  missing = [key for key, value in wanted.items() if value is None]
  _require(not missing, f"M15 replay ledger missed selected seam anchors: {missing}")
  return {
      f"{arm}:{call}:{request_id}": value
      for (arm, call, request_id), value in wanted.items()
  }


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
  return {
      key: candidate[key]
      for key in (
          "record_index", "row_offset", "row_index", "position",
          "call_index", "request_id", "request", "record_geometry",
      )
      if key in candidate
  }


def classify(
    *,
    directory: Path,
    alignment_report: Path,
    capsules: list[Path],
    mode: str,
    arm: str,
    replay_ledger: Path | None = None,
    expected_layer: int | None = None,
    require_first_action: bool = True,
) -> dict[str, Any]:
  _require(mode in ("layer", "full"), "invalid M15 seam mode")
  _require(arm in ("off", "on"), "invalid M15 APC arm")
  if mode == "full":
    _require(expected_layer is not None and 0 <= expected_layer < 36,
             "M15 full seam mode requires an expected layer in [0, 36)")
  else:
    _require(expected_layer is None,
             "M15 layer seam mode must not set an expected layer")
  alignment = _load_alignment(alignment_report)
  ab = alignment["boundaries"]["S_decode_vs_S_prefill"]
  ab_bytes = int(ab["differing_bytes"])
  seam, seam_inventory = _load_seam_candidates(directory, mode)
  require_tail = mode == "layer"
  tails, tail_inventory = (
      _load_tail_candidates(directory) if require_tail else ({}, None)
  )

  if ab_bytes == 0:
    _require(not capsules, "exact M15 observer arm unexpectedly has a capsule")
    return {
        "schema": "m15-apc-wide-seam-classification-v1",
        "status": "PASS",
        "classification": (
            "M15_OBSERVER_CONTROL_EXACT" if arm == "off"
            else "M15_OBSERVER_TREATMENT_EXACT"
        ),
        "arm": arm,
        "observer_mode": mode,
        "alignment": {
            "a_b_differing_bytes": 0,
            "b_c_differing_bytes": 0,
            "n_action": int(alignment.get("N_action", 0)),
        },
        "seam_inventory": seam_inventory,
        "tail_inventory": tail_inventory,
        "gate": "OBSERVER_REACHED_EXACT_ENDPOINT",
        "claim_ceiling": (
            "The observer reached both serving arms and the endpoint stayed "
            "exact; this arm contains no first-red interval."
        ),
    }

  _require(arm == "on", "APC-off control became A-B red under observation")
  _require(capsules, "red M15 observer arm has no mismatch capsule")
  red_points = P38._red_points(capsules)  # pylint: disable=protected-access
  joins = []
  for point in sorted(
      red_points,
      key=lambda item: (int(item["source_row"]), int(item["completion_position"])),
  ):
    base = (int(point["diagnostic_round"]), point["token_prefix_sha256"])
    seam_keys = ((*base, "A"), (*base, "B"))
    if not all(key in seam for key in seam_keys):
      continue
    a = _resolve_aliases(seam[seam_keys[0]], f"A seam {base}")
    b = _resolve_aliases(seam[seam_keys[1]], f"B seam {base}")
    _require(a["position"] == b["position"] == int(point["source_position"]),
             "M15 joined seam source position drifted")
    if mode == "full":
      _require(a["layer_indices"] == b["layer_indices"] == [expected_layer],
               "M15 full seam observed the wrong layer")
    first = P38._first_difference(a, b)  # pylint: disable=protected-access
    tail_a = tail_b = None
    if require_tail:
      target = int(point["target_id"])
      tail_keys = ((*base, "A", target), (*base, "B", target))
      if not all(key in tails for key in tail_keys):
        continue
      tail_a = _resolve_aliases(tails[tail_keys[0]], f"A tail {base}/{target}")
      tail_b = _resolve_aliases(tails[tail_keys[1]], f"B tail {base}/{target}")
      _require(
          tail_a["position"] == tail_b["position"] == int(point["source_position"]),
          "M15 joined tail source position drifted",
      )
      _require(
          float(tail_a["values"][-1]) == float(point["decode_logprob"])
          and float(tail_b["values"][-1]) == float(point["prefill_logprob"]),
          "M15 terminal production logprob differs from the capsule",
      )
      if first is None:
        first = P38._tail_first_difference(  # pylint: disable=protected-access
            tail_a, tail_b
        )
    _require(first is not None,
             "M15 red anchor stayed exact through every observed checkpoint")
    joins.append({
        "source_row": int(point["source_row"]),
        "completion_position": int(point["completion_position"]),
        "source_position": int(point["source_position"]),
        "target_id": int(point["target_id"]),
        "token_prefix_sha256": point["token_prefix_sha256"].decode("ascii"),
        "decode_logprob": float(point["decode_logprob"]),
        "prefill_logprob": float(point["prefill_logprob"]),
        "a": _public_candidate(a),
        "b": _public_candidate(b),
        "a_tail_record_index": tail_a["record_index"] if tail_a else None,
        "b_tail_record_index": tail_b["record_index"] if tail_b else None,
        "first_difference": first,
    })
  _require(joins, "no A-B-red M15 action joined exact standard-path seam records")
  first_action_joins = [join for join in joins if join["completion_position"] == 0]
  if require_first_action:
    _require(first_action_joins,
             "M15 observer did not join a completion-position-zero red anchor")
  preferred = first_action_joins or joins
  anchors = preferred[:_MAX_ANCHORS]
  ledger = _ledger_receipts(replay_ledger, anchors)
  signatures = sorted({
      (join["first_difference"].get("layer"),
       str(join["first_difference"]["checkpoint"]))
      for join in anchors
  }, key=lambda item: (10**9 if item[0] is None else int(item[0]), item[1]))
  first = anchors[0]["first_difference"]
  last_exact = _last_exact_before(first, mode)
  first_anchor = _source_anchor(str(first["checkpoint"]))
  last_anchor = (
      _source_anchor(str(last_exact["checkpoint"])) if last_exact else None
  )
  numeric_layers = sorted({
      int(layer) for layer, _ in signatures if layer is not None
  })
  if mode == "layer":
    gate = "COARSE_FIRST_RED_INTERVAL"
    classification = (
        "M15_LAYER_FIRST_RED_LOCALIZED"
        if numeric_layers
        else "M15_HIDDEN_EXACT_TAIL_FIRST_RED_LOCALIZED"
    )
    next_action = (
        f"rerun full observer at layer {numeric_layers[0]}"
        if numeric_layers
        else "localize the measured terminal-tail interval"
    )
  else:
    gate = "FIRST_RED_LOCALIZED"
    classification = "M15_INTERNAL_FIRST_RED_LOCALIZED"
    next_action = "test one bit-relevant degree of freedom inside the interval"
  return {
      "schema": "m15-apc-wide-seam-classification-v1",
      "status": "PASS",
      "classification": classification,
      "gate": gate,
      "arm": arm,
      "observer_mode": mode,
      "expected_layer": expected_layer,
      "alignment": {
          "a_b_differing_bytes": ab_bytes,
          "a_b_differing_elements": int(ab.get("differing_elements", -1)),
          "b_c_differing_bytes": 0,
          "n_action": int(alignment.get("N_action", 0)),
      },
      "coverage": {
          "total_red_points": len(red_points),
          "standard_joinable_red_points": len(joins),
          "unobserved_red_points": len(red_points) - len(joins),
          "first_action_joinable_red_points": len(first_action_joins),
          "selected_anchors": len(anchors),
          "max_selected_anchors": _MAX_ANCHORS,
      },
      "seam_inventory": seam_inventory,
      "tail_inventory": tail_inventory,
      "first_difference_signatures": [
          {"layer": layer, "checkpoint": checkpoint}
          for layer, checkpoint in signatures
      ],
      "mixed_first_difference_signatures": len(signatures) > 1,
      "selected_layer": numeric_layers[0] if numeric_layers else None,
      "last_exact_boundary": last_exact,
      "first_red_boundary": first,
      "source_interval": {
          "last_exact": last_anchor,
          "first_red": first_anchor,
      },
      "anchors": anchors,
      "replay_ledger_receipts": ledger,
      "next_action": next_action,
      "claim_ceiling": (
          "This report localizes exact standard-path red anchors only. "
          "Continue-decode red actions remain explicitly unobserved; integer "
          "fingerprint equality is not full-tensor byte equality."
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--capsule", type=Path, action="append", default=[])
  parser.add_argument("--mode", choices=("layer", "full"), required=True)
  parser.add_argument("--arm", choices=("off", "on"), required=True)
  parser.add_argument("--replay-ledger", type=Path)
  parser.add_argument("--expected-layer", type=int)
  parser.add_argument("--require-first-action", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = classify(
      directory=args.directory,
      alignment_report=args.alignment_report,
      capsules=args.capsule,
      mode=args.mode,
      arm=args.arm,
      replay_ledger=args.replay_ledger,
      expected_layer=args.expected_layer,
      require_first_action=args.require_first_action,
  )
  payload = json.dumps(report, sort_keys=True, indent=2) + "\n"
  args.output.write_text(payload, encoding="utf-8")
  print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
  main()
