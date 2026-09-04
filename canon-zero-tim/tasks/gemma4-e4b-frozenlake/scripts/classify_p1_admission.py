#!/usr/bin/env python3
"""Strict classifier for a Gemma E4B P1 one-host admission."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import tempfile


REPO_FROM_SCRIPT = Path(__file__).resolve().parents[4]
_ADMISSION_PATH = REPO_FROM_SCRIPT / "tunix/rl/gemma4_e4b_admission.py"
_SPEC = importlib.util.spec_from_file_location("gemma4_e4b_admission", _ADMISSION_PATH)
if _SPEC is None or _SPEC.loader is None:
  raise RuntimeError(f"cannot load {_ADMISSION_PATH}")
admission = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(admission)


MARKERS = {
    "host": "[GEMMA4_E4B_P1_HOST] ",
    "launch_preflight": "[GEMMA4_E4B_P1_LAUNCH_PREFLIGHT] ",
    "runtime": "[GEMMA4_E4B_P1_RUNTIME] ",
    "snapshot": "[GEMMA4_E4B_P1_SNAPSHOT] ",
    "tokenizer": "[GEMMA4_E4B_P1_TOKENIZER] ",
    "dataset": "[GEMMA4_E4B_P1_DATASET] ",
    "hbm": "[GEMMA4_E4B_P1_HBM] ",
    "optimizer": "[GEMMA4_E4B_P1_OPTIMIZER] ",
    "weights": "[GEMMA4_E4B_P1_WEIGHTS] ",
    "shape": "[GEMMA4_E4B_P1_SHAPE] ",
    "rollout": "[GEMMA4_E4B_P1_ROLLOUT] ",
}
HBM_STAGES = (
    "dataset_ready",
    "reference_loaded",
    "actor_loaded",
    "rlcluster_serving_ready",
    "learner_ready",
)
MIN_FREE_BYTES = 2 * 1024**3


def _records(raw: str) -> tuple[dict[str, list[dict]], list[str]]:
  found = {name: [] for name in MARKERS}
  failures = []
  for line_number, line in enumerate(raw.splitlines(), start=1):
    for name, prefix in MARKERS.items():
      if line.startswith(prefix):
        try:
          value = json.loads(line[len(prefix):])
        except (json.JSONDecodeError, TypeError):
          failures.append(f"malformed_receipt:{name}:line={line_number}")
          continue
        if not isinstance(value, dict):
          failures.append(f"non_object_receipt:{name}:line={line_number}")
          continue
        found[name].append(value)
  return found, failures


def classify(
    manifest: dict,
    raw: str,
    docker_exit: int,
    post_manifest: dict | None = None,
) -> dict:
  failures = []
  workload = manifest.get("workload")
  contract = admission.workload_contract(workload)
  found, record_failures = _records(raw)
  failures.extend(record_failures)
  if manifest.get("image_digest") != admission.RUNTIME_IMAGE_DIGEST:
    failures.append("manifest_image_digest")
  if len(str(manifest.get("source_tree_sha256", ""))) != 64:
    failures.append("manifest_source_tree_sha")
  if int(manifest.get("source_tree_files", 0)) <= 0:
    failures.append("manifest_source_tree_files")
  reconstruction = manifest.get("source_reconstruction", {})
  if reconstruction.get("schema") != "gemma4-e4b-p1-source-reconstruction-v1":
    failures.append("source_reconstruction_schema")
  if not re.fullmatch(r"[0-9a-f]{40}", str(manifest.get("source_commit", ""))):
    failures.append("manifest_source_commit")
  if not re.fullmatch(
      r"[0-9a-f]{64}", str(manifest.get("source_diff_sha256", ""))
  ):
    failures.append("manifest_source_diff_sha")
  try:
    tracked_diff = base64.b64decode(
        reconstruction["tracked_diff_base64"], validate=True
    )
  except (KeyError, TypeError, ValueError):
    tracked_diff = None
    failures.append("source_reconstruction_tracked_diff")
  if tracked_diff is not None and hashlib.sha256(tracked_diff).hexdigest() != manifest.get(
      "source_diff_sha256"
  ):
    failures.append("source_reconstruction_tracked_diff_sha")
  untracked = reconstruction.get("untracked", [])
  if not isinstance(untracked, list):
    untracked = []
    failures.append("source_reconstruction_untracked_type")
  if reconstruction.get("base_commit") != manifest.get("source_commit"):
    failures.append("source_reconstruction_base")
  if reconstruction.get("tracked_diff_sha256") != manifest.get(
      "source_diff_sha256"
  ):
    failures.append("source_reconstruction_declared_diff_sha")
  if reconstruction.get("untracked_count") != len(untracked):
    failures.append("source_reconstruction_count")
  paths = [
      record.get("path") if isinstance(record, dict) else None
      for record in untracked
  ]
  valid_paths = all(
      isinstance(path, str)
      and path
      and not Path(path).is_absolute()
      and ".." not in Path(path).parts
      for path in paths
  )
  if not valid_paths:
    failures.append("source_reconstruction_paths")
  elif paths != sorted(paths) or len(paths) != len(set(paths)):
    failures.append("source_reconstruction_order")
  for record in untracked:
    if not isinstance(record, dict):
      failures.append("source_reconstruction_record_type")
      continue
    if record.get("kind") not in ("file", "symlink"):
      failures.append(f"source_reconstruction_kind:{record.get('path')}")
    try:
      payload = base64.b64decode(record["base64"], validate=True)
    except (KeyError, TypeError, ValueError):
      failures.append(f"source_reconstruction_payload:{record.get('path')}")
      continue
    if len(payload) != record.get("bytes") or hashlib.sha256(payload).hexdigest() != record.get(
        "sha256"
    ):
      failures.append(f"source_reconstruction_hash:{record.get('path')}")
  if post_manifest is not None and post_manifest != manifest:
    failures.append("runtime_tree_changed_during_run")
  for name in (
      "host",
      "launch_preflight",
      "runtime",
      "snapshot",
      "tokenizer",
      "dataset",
      "optimizer",
      "weights",
      "shape",
      "rollout",
  ):
    if len(found[name]) != 1:
      failures.append(f"{name}_receipt_count={len(found[name])}")
  if docker_exit != 0:
    failures.append(f"docker_exit={docker_exit}")
  if "Traceback (most recent call last)" in raw:
    failures.append("python_traceback")

  if len(found["host"]) == 1:
    value = found["host"][0]
    if value.get("hostname") != "t1v-n-4a77ebd0-w-0":
      failures.append("host_identity")

  if len(found["launch_preflight"]) == 1:
    value = found["launch_preflight"][0]
    if value.get("workload") != workload or value.get("verdict") != "PASS":
      failures.append("launch_preflight_verdict")
    for gate in ("resolved_env", "contract_sweep", "intent_diff"):
      if value.get(gate, {}).get("status") != "PASS":
        failures.append(f"launch_preflight_{gate}")

  if len(found["runtime"]) == 1:
    value = found["runtime"][0]
    if value.get("device_count") != 4 or value.get("local_device_count") != 4:
      failures.append("runtime_device_count")
    if value.get("process_count") != 1 or value.get("platforms") != ["tpu"]:
      failures.append("runtime_platform")
    if value.get("device_kinds") != ["TPU v5"]:
      failures.append("runtime_device_kind")
    device_ids = value.get("device_ids", [])
    if len(device_ids) != 4 or len(set(device_ids)) != 4:
      failures.append("runtime_device_ids")
    if value.get("device_process_indices") != [0, 0, 0, 0]:
      failures.append("runtime_device_process_indices")
    if value.get("jax_version") != admission.EXPECTED_JAX_VERSION:
      failures.append("runtime_jax_version")
    if value.get("rollout_mesh_shape") != [1, 4] or value.get("trainer_mesh_shape") != [1, 4]:
      failures.append("runtime_mesh")
    if value.get("workload") != workload:
      failures.append("runtime_workload")

  if len(found["snapshot"]) == 1:
    value = found["snapshot"][0]
    if value.get("checkpoint_revision") != admission.CHECKPOINT_REVISION:
      failures.append("snapshot_revision")
    for name, expected in admission.SNAPSHOT_FILES.items():
      actual = value.get("files", {}).get(name, {})
      if actual.get("sha256") != expected["sha256"]:
        failures.append(f"snapshot_sha:{name}")

  if len(found["tokenizer"]) == 1:
    value = found["tokenizer"][0]
    expected = {
        "class": "GemmaTokenizer",
        "vocab_size": 262144,
        "bos_id": 2,
        "eos_id": 1,
        "pad_id": 0,
        "unk_id": 3,
        "chat_template_sha256": admission.CHAT_TEMPLATE_SHA256,
        "token_probe_sha256": admission.TOKEN_PROBE_SHA256,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
      failures.append("tokenizer_identity")

  if len(found["dataset"]) == 1:
    value = found["dataset"][0]
    if value.get("workload") != workload:
      failures.append("dataset_workload")
    if value.get("train_sha256") != contract["train_sha256"]:
      failures.append("dataset_train_sha")
    if value.get("eval_sha256") != contract["eval_sha256"]:
      failures.append("dataset_eval_sha")

  hbm_sequence = tuple(value.get("stage") for value in found["hbm"])
  hbm_by_stage = {value.get("stage"): value for value in found["hbm"]}
  if (
      set(hbm_by_stage) != set(HBM_STAGES)
      or len(found["hbm"]) != len(HBM_STAGES)
  ):
    failures.append(f"hbm_stages={sorted(hbm_by_stage)}")
  if hbm_sequence != HBM_STAGES:
    failures.append(f"hbm_stage_order={hbm_sequence}")
  for stage, value in hbm_by_stage.items():
    if len(value.get("devices", [])) != 4:
      failures.append(f"hbm_device_count:{stage}")
    if int(value.get("min_bytes_free", -1)) < 0:
      failures.append(f"hbm_invalid:{stage}")
  if hbm_by_stage and int(hbm_by_stage.get("learner_ready", {}).get("min_bytes_free", -1)) < MIN_FREE_BYTES:
    failures.append("learner_hbm_headroom_below_2gib")

  if len(found["weights"]) == 1:
    value = found["weights"][0]
    if value.get("equal") is not True or value.get("mismatch_indices") not in ([], ()):
      failures.append("live_weight_difference")
    if value.get("mapped_leaves") != 593 or value.get("live_leaves") != 593:
      failures.append("live_weight_leaf_count")
    if int(value.get("total_elements", 0)) <= 0:
      failures.append("live_weight_elements")
    mesh = dict(value.get("mesh_shape", []))
    if mesh.get("data") != 1 or mesh.get("model") != 4:
      failures.append("live_weight_mesh")
    expected_architecture = {
        "layers": 42,
        "heads": 8,
        "kv_heads": 2,
        "head_dim": 256,
        "hidden_size": 2560,
        "intermediate_size": 10240,
        "vocab_size": 262144,
        "sliding_window": 512,
        "final_logit_softcapping": 30.0,
        "vocab_size_per_layer_input": 262144,
        "hidden_size_per_layer_input": 256,
        "num_kv_shared_layers": 18,
        "tie_word_embeddings": True,
        "attention_k_eq_v": False,
        "enable_moe_block": False,
        "layer_types": list(admission.EXPECTED_LAYER_TYPES),
    }
    if value.get("architecture") != expected_architecture:
      failures.append("live_weight_architecture")

  if len(found["optimizer"]) == 1:
    value = found["optimizer"][0]
    if int(value.get("leaves", 0)) <= 0:
      failures.append("optimizer_state_empty")
    if (
        int(value.get("elements", 0)) <= 0
        or int(value.get("allocated_bytes", 0)) <= 0
    ):
      failures.append("optimizer_state_unallocated")

  if len(found["shape"]) == 1:
    value = found["shape"][0]
    expected = manifest["execution"]
    if value.get("workload") != workload or value.get("caller_global_rows") != 1:
      failures.append("shape_rows")
    if value.get("dp") != 1 or value.get("tp") != 4:
      failures.append("shape_mesh")
    if (
        value.get("mini_batch_size"),
        value.get("train_micro_batch_size"),
        value.get("compute_logps_micro_batch_size"),
    ) != (1, 1, 1):
      failures.append("shape_trainer_batches")
    if (
        expected.get("mini_batch_size"),
        expected.get("train_micro_batch_size"),
        expected.get("compute_logps_micro_batch_size"),
    ) != (1, 1, 1):
      failures.append("manifest_trainer_batches")
    if value.get("context_hard_cap") != expected["context_hard_cap"]:
      failures.append("shape_context_cap")

  if len(found["rollout"]) == 1:
    value = found["rollout"][0]
    if value.get("workload") != workload or value.get("trajectories") != 1:
      failures.append("rollout_coverage")
    if value.get("prompts") != 1 or value.get("generations") != 1:
      failures.append("rollout_shape")
    if int(value.get("max_prompt_tokens", 0)) <= 0:
      failures.append("rollout_prompt_tokens")
    if int(value.get("max_assistant_tokens", 0)) <= 0:
      failures.append("rollout_assistant_tokens")
    if not 1 <= int(value.get("max_interactions", 0)) <= contract["max_turns"]:
      failures.append("rollout_interactions")
    if int(value.get("max_active_rows", 0)) != 1:
      failures.append("rollout_active_rows")
    if not 1 <= int(value.get("max_kv_tokens_observed", 0)) <= contract["context_hard_cap"]:
      failures.append("rollout_kv_tokens")
    if value.get("train_steps_before") != value.get("train_steps_after"):
      failures.append("train_step_mutated")
    if value.get("global_steps_before") != value.get("global_steps_after"):
      failures.append("global_step_mutated")
    if value.get("backward") != 0 or value.get("optimizer_commits") != 0:
      failures.append("training_executed")

  return {
      "schema": "gemma4-e4b-p1-classification-v1",
      "verdict": "PASS" if not failures else "FAIL",
      "workload": workload,
      "failures": failures,
      "receipt_counts": {name: len(values) for name, values in found.items()},
      "minimum_required_learner_hbm_free_bytes": MIN_FREE_BYTES,
  }


def _write(path: Path, value: dict) -> None:
  if path.exists():
    raise FileExistsError(f"refusing to overwrite {path}")
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(value, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--raw", type=Path, required=True)
  parser.add_argument("--post-manifest", type=Path, required=True)
  parser.add_argument("--docker-exit", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = classify(
      json.loads(args.manifest.read_text(encoding="utf-8")),
      args.raw.read_text(encoding="utf-8", errors="replace"),
      args.docker_exit,
      json.loads(args.post_manifest.read_text(encoding="utf-8")),
  )
  _write(args.output, result)
  print(json.dumps(result, sort_keys=True), flush=True)
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
