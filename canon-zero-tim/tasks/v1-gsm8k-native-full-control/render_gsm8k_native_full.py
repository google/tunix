#!/usr/bin/env python3
"""Render one registered P33 GSM8K Native/mismatch full control."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Mapping

import yaml


_TASK_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _TASK_DIR.parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_CLUSTER_DIR = _PACKAGE_ROOT / "cluster"
for _path in (_REPO_ROOT, _CLUSTER_DIR):
  if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

import render_p33_jobsets as p33


_SHA_RE = re.compile(r"[0-9a-f]{40}")
_ORIGINAL_P33_PROFILE = "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env"
_NATIVE_PROFILE = (
    "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-native.env"
)
_WANDB_PROJECT = "zero-tim-gsm8k-dp16-tp4"
_WANDB_GROUP = "qwen3-1p7b-dp16-tp4"
_OUTPUT_NAME = "jobset-v1-gsm8k-native-mismatch-full.yaml"
_FORBIDDEN_ZERO_SELECTORS = (
    "CANON_P32_WORKLOAD",
    "CANON_P59_RANK_PARALLEL_BACKWARD",
    "CANON_P59_CHECKED_VMA",
    "CANON_V1_HP_FULL",
    "CANON_V1_HP_FIRST_UPDATE_GATE",
    "CANON_DP_COMPARE_MODE",
    "CANON_DP_DISTINCT_SCHEDULE",
    "CANON_DP_FINITE_FETCH",
    "CANON_P71_SCAN",
    "CANON_DP_COLLECTIVE_REDUCE",
    "CANON_P67_P66_VMA_P59_ONLY",
    "CANON_P63_OVERFLOW_SAFE_CLIP",
    "CANON_ALIGNMENT_GATE",
    "CANON_ALIGNMENT_GATE_ONLY",
    "CANON_ALIGNMENT_UPDATE_CANARY",
    "CANON_ALIGNMENT_TRAIN",
    "CANON_PRE_ALIGN_GATE",
    "CANON_GSM8K_AB_REPORT_ONLY",
    "CANON_GSM8K_ALIGNMENT_WARN_ONLY",
    "CANON_P38_FIXED_LM_HEAD",
    "CANON_PRE_ALIGN_REPORT",
    "CANON_ALIGN_REPORT",
    "CANON_UPDATE_REPORT",
    "CANON_P38_MISMATCH_CAPSULE",
    "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS",
)


def _registered_spec() -> p33.JobSpec:
  matches = [spec for spec in p33._SPECS if spec.key == "gsm8k-full"]
  if len(matches) != 1:
    raise ValueError("registered P33 gsm8k-full spec is missing or ambiguous")
  original = matches[0]
  expected = {
      "workload": "gsm8k",
      "stage": "full",
      "profile": _ORIGINAL_P33_PROFILE,
      "no_commit": False,
      "dp_size": 16,
      "tp_size": 4,
      "optimizer_resident": True,
      "rank_parallel_backward": False,
      "fixed_lm_head": True,
      "strict_alignment": False,
      "v1_hp_full": False,
      "command": p33._gsm8k_command(200),
  }
  wrong = {
      name: getattr(original, name)
      for name, value in expected.items()
      if getattr(original, name) != value
  }
  if wrong:
    raise ValueError(f"registered P33 gsm8k-full contract drifted: {wrong}")
  # Reuse the original scientific command and full-run restart identity, but
  # select the signed stock profile and stock lm-head. The renderer below
  # removes P33's canonical evidence defaults after its structural validator
  # has run.
  return dataclasses.replace(
      original,
      profile=_NATIVE_PROFILE,
      job_prefix="canon-v1ctl-gsm-native",
      fixed_lm_head=False,
  )


def _head_main(document: Mapping[str, Any]) -> Mapping[str, Any]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  matches = [item for item in pod["containers"] if item["name"] == "jax-tpu"]
  if len(matches) != 1:
    raise ValueError("expected exactly one jax-tpu container")
  return matches[0]


def _env(document: Mapping[str, Any]) -> dict[str, str]:
  entries = _head_main(document)["env"]
  names = [item["name"] for item in entries]
  if len(names) != len(set(names)):
    raise ValueError("generated Native environment contains duplicate names")
  return {
      item["name"]: item["value"] for item in entries if "value" in item
  }


def _set_env(document: dict[str, Any], values: Mapping[str, str]) -> None:
  entries = _head_main(document)["env"]
  by_name = {item["name"]: item for item in entries}
  for name, value in values.items():
    if name in by_name:
      by_name[name].clear()
      by_name[name].update({"name": name, "value": value})
    else:
      entries.append({"name": name, "value": value})


def _remove_env(document: dict[str, Any], names: tuple[str, ...]) -> None:
  remove = set(names)
  entries = _head_main(document)["env"]
  entries[:] = [item for item in entries if item["name"] not in remove]


def _remove_proxy_precision_pin(document: dict[str, Any]) -> None:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  matches = [
      item for item in pod["initContainers"] if item["name"] == "pathways-proxy"
  ]
  if len(matches) != 1:
    raise ValueError("expected exactly one pathways-proxy init container")
  proxy = matches[0]
  proxy["env"] = [
      item for item in proxy.get("env", []) if item["name"] != "XLA_FLAGS"
  ]


def _proxy_env(document: Mapping[str, Any]) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  proxy = next(
      item for item in pod["initContainers"] if item["name"] == "pathways-proxy"
  )
  return {
      item["name"]: item["value"]
      for item in proxy.get("env", [])
      if "value" in item
  }


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _validate_document(
    document: Mapping[str, Any],
    spec: p33.JobSpec,
    source_commit: str,
    run_id: str,
) -> None:
  labels = document.get("metadata", {}).get("labels", {})
  required_labels = {
      "canon.zero-tim/treatment": "native-mismatch",
      "canon.zero-tim/control-for": "v1-hp-zero",
      "canon.zero-tim/performance-profile": "stock-native",
      "canon.zero-tim/full-recipe": "gsm8k",
  }
  wrong_labels = {
      name: labels.get(name)
      for name, value in required_labels.items()
      if labels.get(name) != value
  }
  if wrong_labels:
    raise ValueError(f"GSM8K Native comparison labels drifted: {wrong_labels}")

  values = _env(document)
  required = {
      "CANON_PROFILE_FILE": _NATIVE_PROFILE,
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P33_SHARED_MESH": "16,4",
      "CANON_P33_RUN_STAGE": "full",
      "CANON_P33_NO_COMMIT": "0",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P32_DP_REDUCTION_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_GSM8K_TRAIN": "1",
      "CANON_GSM8K_VANILLA": "1",
      "CANON_RUN_CMD": shlex.join(p33._gsm8k_command(200)),
  }
  wrong = {
      name: values.get(name)
      for name, value in required.items()
      if values.get(name) != value
  }
  if wrong:
    raise ValueError(f"GSM8K Native full contract drifted: {wrong}")
  forbidden = {
      name: values.get(name)
      for name in _FORBIDDEN_ZERO_SELECTORS
      if name in values
  }
  if forbidden:
    raise ValueError(
        f"GSM8K Native unexpectedly carries Zero selectors: {forbidden}"
    )
  command = values["CANON_RUN_CMD"].split()
  if command.count(f"--wandb_project={_WANDB_PROJECT}") != 1:
    raise ValueError("GSM8K Native W&B project argument drifted")
  if command.count("--max_steps=200") != 1:
    raise ValueError("GSM8K Native update horizon drifted")
  if "XLA_FLAGS" in _proxy_env(document):
    raise ValueError("GSM8K Native proxy retained the canonical precision pin")
  expected_name = f"{spec.job_prefix}-{run_id}-{source_commit[:8]}"
  if document.get("metadata", {}).get("name") != expected_name:
    raise ValueError("GSM8K Native JobSet identity drifted")
  if document["spec"]["failurePolicy"].get("maxRestarts") != 3:
    raise ValueError("GSM8K Native full restart policy drifted")


def render_native_full(
    *, source_commit: str, output_dir: Path, run_id: str, base_path: Path
) -> Path:
  """Renders one fresh Native/mismatch full manifest and immutable index."""
  if not _SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be exactly 40 lowercase hex characters")
  if output_dir.exists():
    raise FileExistsError(f"refusing to overwrite output root: {output_dir}")

  spec = _registered_spec()
  document = p33.render_jobset(
      p33.load_base(base_path), spec, source_commit, run_id
  )
  _set_env(document, {
      "CANON_P32_TRAIN_ADMITTED": "0",
      "CANON_P32_DP_REDUCTION_ADMITTED": "0",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "0",
      "CANON_GSM8K_TRAIN": "1",
      "CANON_GSM8K_VANILLA": "1",
  })
  _remove_env(document, _FORBIDDEN_ZERO_SELECTORS)
  _remove_proxy_precision_pin(document)
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/treatment": "native-mismatch",
      "canon.zero-tim/control-for": "v1-hp-zero",
      "canon.zero-tim/performance-profile": "stock-native",
      "canon.zero-tim/full-recipe": "gsm8k",
  })
  _validate_document(document, spec, source_commit, run_id)

  output_dir.mkdir(parents=True)
  path = output_dir / _OUTPUT_NAME
  header = (
      "# Generated by v1-gsm8k-native-full-control renderer.\n"
      "# Do not edit; change the registered P33 spec or this validator.\n"
  )
  path.write_text(
      header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
  )
  receipt = {
      "schema": "v1-gsm8k-native-mismatch-full-v1",
      "arm": "native-mismatch",
      "comparison_arm": "v1-hp-zero",
      "wandb_project": _WANDB_PROJECT,
      "wandb_group": _WANDB_GROUP,
      "manifest": {
          "path": str(path),
          "sha256": _sha256(path),
          "jobset": document["metadata"]["name"],
          "source": source_commit,
          "run_id": run_id,
      },
      "launch_executed": False,
  }
  index = output_dir / "manifest-index.json"
  index.write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "V1_GSM8K_NATIVE_FULL_MANIFEST_PASS "
      f"path={path} sha256={receipt['manifest']['sha256']} "
      f"project={_WANDB_PROJECT} treatment=native-mismatch "
      "launch=not-executed",
      flush=True,
  )
  return path


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--run-id", required=True)
  parser.add_argument(
      "--base", type=Path, default=_CLUSTER_DIR / "jobset-64chip.yaml"
  )
  args = parser.parse_args()
  render_native_full(
      source_commit=args.source_commit,
      output_dir=args.output_dir,
      run_id=args.run_id,
      base_path=args.base,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
