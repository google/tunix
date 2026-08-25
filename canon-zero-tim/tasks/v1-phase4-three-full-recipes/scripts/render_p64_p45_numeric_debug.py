#!/usr/bin/env python3
"""Render one immutable P64 P45 DP8xTP8 first-red JobSet."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml


_TASK_DIR = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _TASK_DIR.parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
_CLUSTER_DIR = _PACKAGE_ROOT / "cluster"
for path in (_REPO_ROOT, _CLUSTER_DIR):
  if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import render_p33_jobsets as p33


_SHA_RE = re.compile(r"[0-9a-f]{40}")
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_PROFILE = (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-p64-debug.env"
)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _main(document: dict) -> dict:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document: dict) -> dict[str, str]:
  return {
      item["name"]: item["value"]
      for item in _main(document)["env"]
      if "value" in item
  }


def render(
    *,
    source_commit: str,
    run_id: str,
    output_dir: Path,
    base_path: Path,
    capsule_mode: str = "capture",
    capsule_gcs_uri: str = "",
    capsule_sha256: str = "",
    model_binding_sha256: str = "",
) -> Path:
  if not _SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be exactly 40 lowercase hex")
  if output_dir.exists():
    raise FileExistsError(f"refusing to overwrite output root: {output_dir}")
  if capsule_mode not in ("capture", "replay"):
    raise ValueError("capsule mode must be capture or replay")
  derived_gcs_uri = (
      "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/"
      f"p64/{run_id}/training-capsule.npz"
  )
  if capsule_mode == "capture":
    if capsule_gcs_uri or capsule_sha256 or model_binding_sha256:
      raise ValueError("capture mode derives its URI and forbids replay hashes")
    capsule_gcs_uri = derived_gcs_uri
  else:
    if not re.fullmatch(
        r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p64/"
        r"[a-z0-9][a-z0-9-]*/training-capsule\.npz",
        capsule_gcs_uri,
    ):
      raise ValueError("replay capsule URI is outside the P64 evidence root")
    if not _DIGEST_RE.fullmatch(capsule_sha256):
      raise ValueError("replay capsule SHA must be 64 lowercase hex")
    if not _DIGEST_RE.fullmatch(model_binding_sha256):
      raise ValueError("replay model-binding SHA must be 64 lowercase hex")
  output_dir.mkdir(parents=True)
  command = list(p33._frozenlake_command(  # pylint: disable=protected-access
      1, dp_size=8, tp_size=8
  ))
  command[:3] = (
      "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3"
  )
  command.extend(("--sampler_is=none", "--seed=42", "--eval_every_n_steps=0"))
  spec = p33.JobSpec(
      key="p64-p45-numeric-debug",
      workload="frozenlake",
      stage="backward-no-commit",
      profile=_PROFILE,
      no_commit=True,
      job_prefix="canon-p64-p45-num",
      command=tuple(command),
      dp_size=8,
      tp_size=8,
      optimizer_resident=True,
      rank_parallel_backward=True,
      fixed_lm_head=True,
      strict_alignment=True,
  )
  document = p33.render_jobset(
      p33.load_base(base_path), spec, source_commit, run_id
  )
  p33._set_named_env(  # pylint: disable=protected-access
      _main(document)["env"],
      {
          "CANON_P64_P45_NUMERIC_DEBUG": "1",
          "CANON_P64_TRAINING_CAPSULE_MODE": capsule_mode,
          "CANON_P64_TRAINING_CAPSULE": (
              f"{_env(document)['CANON_STATE']}/p64_training_capsule.npz"
          ),
          "CANON_P64_TRAINING_CAPSULE_GCS_URI": capsule_gcs_uri,
          "CANON_P64_TRAINING_CAPSULE_SHA256": capsule_sha256,
          "CANON_P64_MODEL_BINDING_SHA256": model_binding_sha256,
          "CANON_V1_HP_FULL": "0",
      },
      remove=(),
  )
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/diagnostic": "p64-p45-first-red",
      "canon.zero-tim/optimizer-commits": "0",
  })
  p33.validate_jobset(document, spec, source_commit, run_id)
  values = _env(document)
  required = {
      "CANON_PROFILE_FILE": _PROFILE,
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P38_FIXED_LM_HEAD": "1",
      "CANON_P64_P45_NUMERIC_DEBUG": "1",
      "CANON_P64_TRAINING_CAPSULE_MODE": capsule_mode,
      "CANON_P64_TRAINING_CAPSULE_GCS_URI": capsule_gcs_uri,
      "CANON_P64_TRAINING_CAPSULE_SHA256": capsule_sha256,
      "CANON_P64_MODEL_BINDING_SHA256": model_binding_sha256,
      "CANON_V1_HP_FULL": "0",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
  }
  wrong = {
      name: values.get(name)
      for name, expected in required.items()
      if values.get(name) != expected
  }
  if wrong:
    raise ValueError(f"rendered P64 contract drifted: {wrong}")
  state = values["CANON_STATE"]
  capsule_path = values["CANON_P64_TRAINING_CAPSULE"]
  if capsule_path != f"{state}/p64_training_capsule.npz":
    raise ValueError("P64 capsule path is not isolated by JobSet")
  run_log = values["CANON_RUN_LOG"]
  if run_log != f"{state}/run.log":
    raise ValueError("P64 full-log path is not isolated by JobSet")
  output = output_dir / "jobset-p64-p45-numeric-debug.yaml"
  output.write_text(
      "# Generated by render_p64_p45_numeric_debug.py. Do not edit.\n"
      + yaml.safe_dump(document, sort_keys=False),
      encoding="utf-8",
  )
  receipt = {
      "schema": "canon-p64-render-v2",
      "source_commit": source_commit,
      "run_id": run_id,
      "jobset": document["metadata"]["name"],
      "path": str(output),
      "sha256": _sha256(output),
      "optimizer_commits": 0,
      "state": state,
      "run_log": run_log,
      "classification": f"{state}/p64_p45_numeric.classification.json",
      "capsule_mode": capsule_mode,
      "capsule": capsule_path,
      "capsule_gcs_uri": capsule_gcs_uri,
      "capsule_sha256": capsule_sha256,
      "model_binding": f"{capsule_path}.model.json",
      "model_binding_sha256": model_binding_sha256,
  }
  (output_dir / "render-receipt.json").write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      "P64_P45_NUMERIC_RENDER_PASS "
      f"jobset={receipt['jobset']} sha256={receipt['sha256']} "
      f"capsule_mode={capsule_mode} optimizer_commits=0",
      flush=True,
  )
  return output


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument(
      "--capsule-mode", choices=("capture", "replay"), default="capture"
  )
  parser.add_argument("--capsule-gcs-uri", default="")
  parser.add_argument("--capsule-sha256", default="")
  parser.add_argument("--model-binding-sha256", default="")
  parser.add_argument(
      "--base", type=Path, default=_CLUSTER_DIR / "jobset-64chip.yaml"
  )
  args = parser.parse_args()
  render(
      source_commit=args.source_commit,
      run_id=args.run_id,
      output_dir=args.output_dir,
      base_path=args.base,
      capsule_mode=args.capsule_mode,
      capsule_gcs_uri=args.capsule_gcs_uri,
      capsule_sha256=args.capsule_sha256,
      model_binding_sha256=args.model_binding_sha256,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
