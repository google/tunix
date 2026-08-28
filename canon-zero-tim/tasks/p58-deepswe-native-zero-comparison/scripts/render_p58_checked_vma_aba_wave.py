#!/usr/bin/env python3
"""Render three immutable P58 checked-VMA precheck JobSets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
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


_SHA = re.compile(r"[0-9a-f]{40}")
_WAVE = re.compile(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?")
_ARMS = (
    ("on-a", "on", "01-on-a.yaml"),
    ("off", "off", "02-off.yaml"),
    ("on-b", "on", "03-on-b.yaml"),
)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_arm(
    *, base: dict, source_commit: str, source_branch: str,
    client_image: str, run_id: str, selector: str, cpu_nodepool: str,
    worker_nodepool: str, model_pvc: str,
) -> dict:
  return p58.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage="full",
      arm="zero",
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      checked_vma_off_diagnostic=selector == "off",
      checked_vma_on_diagnostic=selector == "on",
  )


def render_wave(
    *, base_path: Path, output_dir: Path, source_commit: str,
    source_branch: str, client_image: str, wave_id: str,
    cpu_nodepool: str, worker_nodepool: str, model_pvc: str,
) -> dict:
  if not _SHA.fullmatch(source_commit):
    raise ValueError("source commit must be exactly 40 lowercase hex")
  if not _WAVE.fullmatch(wave_id) or len(wave_id) > 12:
    raise ValueError("wave id must be a DNS label of at most 12 characters")
  if output_dir.exists():
    raise FileExistsError(f"refusing to overwrite output root: {output_dir}")
  if not p58.p34._DIGEST_IMAGE.fullmatch(client_image):
    raise ValueError("client image must be pinned by registry digest")

  base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
  jobset_dir = output_dir / "jobsets"
  jobset_dir.mkdir(parents=True)
  documents: dict[str, dict] = {}
  arms: list[dict] = []
  for arm, selector, filename in _ARMS:
    run_id = f"{wave_id}-{arm.replace('-', '')}"
    document = _render_arm(
        base=base,
        source_commit=source_commit,
        source_branch=source_branch,
        client_image=client_image,
        run_id=run_id,
        selector=selector,
        cpu_nodepool=cpu_nodepool,
        worker_nodepool=worker_nodepool,
        model_pvc=model_pvc,
    )
    document["metadata"]["labels"].update({
        "canon.zero-tim/diagnostic-wave": wave_id,
        "canon.zero-tim/diagnostic-arm": arm,
    })
    p58.validate(
        document,
        source_commit=source_commit,
        client_image=client_image,
        stage="full",
        arm="zero",
        worker_nodepool=worker_nodepool,
        checked_vma_off_diagnostic=selector == "off",
        checked_vma_on_diagnostic=selector == "on",
    )
    path = jobset_dir / filename
    path.write_text(p58.p34.dump_jobset(document), encoding="utf-8")
    env = p58.p34._env(document)
    documents[arm] = document
    arms.append({
        "arm": arm,
        "selector": selector,
        "run_id": run_id,
        "jobset": document["metadata"]["name"],
        "yaml": str(path),
        "yaml_sha256": _sha256(path),
        "state": env["CANON_STATE"],
        "classification": (
            f"{env['CANON_STATE']}/"
            f"p58_checked_vma_{selector}.classification.json"
        ),
    })

  signatures = [p58.recipe_signature(documents[name]) for name, _, _ in _ARMS]
  if not all(signature == signatures[0] for signature in signatures[1:]):
    raise ValueError("ABA wave recipe signatures are not identical")
  on_a = p58.treatment_signature(documents["on-a"])
  off = p58.treatment_signature(documents["off"])
  on_b = p58.treatment_signature(documents["on-b"])
  if on_a != on_b:
    raise ValueError("ABA on-arm treatment signatures differ")
  if on_a["checked_vma_diagnostic"] != "on" or off[
      "checked_vma_diagnostic"
  ] != "off":
    raise ValueError("ABA selector ordering drifted")
  if {
      key: value for key, value in on_a.items()
      if key != "checked_vma_diagnostic"
  } != {
      key: value for key, value in off.items()
      if key != "checked_vma_diagnostic"
  }:
    raise ValueError("ABA arms differ outside the registered selector")

  receipt = {
      "schema": "canon.p58.checked-vma-aba-render.v1",
      "source_commit": source_commit,
      "source_branch": source_branch,
      "client_image": client_image,
      "worker_nodepool": worker_nodepool,
      "wave_id": wave_id,
      "arm_order": [arm for arm, _, _ in _ARMS],
      "execution_mode": "parallel-requested",
      "arms": arms,
      "parallel_capacity": {
          "jobsets": 3,
          "tpu_chips_per_jobset": 128,
          "aggregate_tpu_chips": 384,
          "head_nodes": 3,
          "sandbox_concurrency_per_jobset": 128,
          "aggregate_sandbox_concurrency": 384,
          "sandbox_cpu_requests": 768,
          "sandbox_memory_requests_gib": 1536,
      },
      "backward": 0,
      "optimizer_commits": 0,
      "claim": (
          "Render/construction evidence only. ON-A/OFF/ON-B is a logical arm "
          "order; a concurrent launch is a matched OFF control with two ON "
          "replicates, not a temporal ABA sandwich. Parallel target execution "
          "still requires TPU quota, three distinct head nodes, aggregate "
          "sandbox capacity, server dry-run, and explicit launch approval."
      ),
  }
  receipt_path = output_dir / "wave-render-receipt.json"
  receipt_path.write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      "P58_CHECKED_VMA_ABA_RENDER_PASS "
      f"wave={wave_id} source={source_commit} jobs=3 tpu_request=384 "
      "sandbox_concurrency=384 backward=0 optimizer_commits=0",
      flush=True,
  )
  return receipt


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, default=_CLUSTER / "jobset-64chip.yaml")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default="yuxzhang/canon-zero-tim")
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--wave-id", required=True)
  parser.add_argument("--cpu-nodepool", default="cpu-np")
  parser.add_argument("--worker-nodepool", required=True)
  parser.add_argument("--model-pvc", default="haoyugao-cpu-np-pvc")
  args = parser.parse_args()
  render_wave(
      base_path=args.base,
      output_dir=args.output_dir,
      source_commit=args.source_commit,
      source_branch=args.source_branch,
      client_image=args.client_image,
      wave_id=args.wave_id,
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      model_pvc=args.model_pvc,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
