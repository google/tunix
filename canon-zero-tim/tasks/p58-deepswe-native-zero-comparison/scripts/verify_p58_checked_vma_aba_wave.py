#!/usr/bin/env python3
"""Verify a rendered P58 checked-VMA ABA wave before cluster use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


_SCRIPT_DIR = Path(__file__).resolve().parent
_PKG = _SCRIPT_DIR.parents[2]
_REPO = _PKG.parent
_CLUSTER = _PKG / "cluster"
for _path in (_REPO, _CLUSTER):
  if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

import render_p58_deepswe_tim as p58


_FILES = {
    "on-a": ("on", "01-on-a.yaml"),
    "off": ("off", "02-off.yaml"),
    "on-b": ("on", "03-on-b.yaml"),
}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output_dir: Path) -> dict:
  receipt_path = output_dir / "wave-render-receipt.json"
  receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  if receipt.get("schema") != "canon.p58.checked-vma-aba-render.v1":
    raise ValueError("ABA render receipt schema drifted")
  receipt_arms = {item["arm"]: item for item in receipt.get("arms", [])}
  if set(receipt_arms) != set(_FILES):
    raise ValueError("ABA receipt arm set drifted")

  documents = {}
  names = set()
  states = set()
  for arm, (selector, filename) in _FILES.items():
    path = output_dir / "jobsets" / filename
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    record = receipt_arms[arm]
    if record.get("yaml_sha256") != _sha256(path):
      raise ValueError(f"{arm} YAML digest drifted")
    if document["metadata"]["name"] != record.get("jobset"):
      raise ValueError(f"{arm} JobSet identity drifted")
    labels = document["metadata"].get("labels", {})
    if labels.get("canon.zero-tim/diagnostic-wave") != receipt["wave_id"]:
      raise ValueError(f"{arm} wave label drifted")
    if labels.get("canon.zero-tim/diagnostic-arm") != arm:
      raise ValueError(f"{arm} diagnostic-arm label drifted")
    if labels.get("canon.zero-tim/diagnostic-selector") != selector:
      raise ValueError(f"{arm} selector label drifted")
    p58.validate(
        document,
        source_commit=receipt["source_commit"],
        client_image=receipt["client_image"],
        stage="full",
        arm="zero",
        worker_nodepool=receipt["worker_nodepool"],
        checked_vma_off_diagnostic=selector == "off",
        checked_vma_on_diagnostic=selector == "on",
    )
    env = p58.p34._env(document)
    names.add(document["metadata"]["name"])
    states.add(env["CANON_STATE"])
    documents[arm] = document
  if len(names) != 3 or len(states) != 3:
    raise ValueError("ABA JobSet names or persistent roots collide")

  signatures = [p58.recipe_signature(documents[arm]) for arm in _FILES]
  if not all(signature == signatures[0] for signature in signatures[1:]):
    raise ValueError("ABA recipe signature drifted after serialization")
  on_a = p58.treatment_signature(documents["on-a"])
  off = p58.treatment_signature(documents["off"])
  on_b = p58.treatment_signature(documents["on-b"])
  if on_a != on_b:
    raise ValueError("ABA on controls are not identical")
  on_common = dict(on_a)
  off_common = dict(off)
  on_common.pop("checked_vma_diagnostic")
  off_common.pop("checked_vma_diagnostic")
  if on_common != off_common:
    raise ValueError("ABA arms differ outside the selector")
  if on_a["checked_vma_diagnostic"] != "on" or off[
      "checked_vma_diagnostic"
  ] != "off":
    raise ValueError("ABA selector sequence is not on/off/on")

  result = {
      "schema": "canon.p58.checked-vma-aba-verify.v1",
      "verdict": "PASS",
      "wave_id": receipt["wave_id"],
      "source_commit": receipt["source_commit"],
      "client_image": receipt["client_image"],
      "arm_order": list(_FILES),
      "execution_mode": receipt.get("execution_mode"),
      "jobsets": sorted(names),
      "persistent_roots": sorted(states),
      "aggregate_tpu_chips": 384,
      "aggregate_sandbox_concurrency": 384,
      "backward": 0,
      "optimizer_commits": 0,
  }
  if result["execution_mode"] != "parallel-requested":
    raise ValueError("ABA execution mode drifted")
  verify_path = output_dir / "wave-verify.json"
  if verify_path.exists():
    raise FileExistsError(f"refusing to overwrite verification: {verify_path}")
  verify_path.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      "P58_CHECKED_VMA_ABA_VERIFY_PASS jobs=3 selectors=on,off,on "
      "tpu_request=384 sandbox_concurrency=384 backward=0 "
      "optimizer_commits=0",
      flush=True,
  )
  return result


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  verify(args.output_dir)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
