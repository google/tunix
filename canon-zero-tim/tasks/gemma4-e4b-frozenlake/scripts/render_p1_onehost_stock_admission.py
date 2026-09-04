#!/usr/bin/env python3
"""Renders, but never launches, one Gemma E4B P1 admission manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


REPO_FROM_SCRIPT = Path(__file__).resolve().parents[4]
_ADMISSION_PATH = REPO_FROM_SCRIPT / "tunix/rl/gemma4_e4b_admission.py"
_SPEC = importlib.util.spec_from_file_location("gemma4_e4b_admission", _ADMISSION_PATH)
if _SPEC is None or _SPEC.loader is None:
  raise RuntimeError(f"cannot load {_ADMISSION_PATH}")
admission = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(admission)


PINNED_IMAGE_DIGEST = admission.RUNTIME_IMAGE_DIGEST
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA64 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_P1_UNTRACKED = frozenset({
    "canon-zero-tim/cluster/profiles/gemma4-e4b-dp1-tp4-frozenlake.env",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/HANDOFF.md",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/RUNBOOK.md",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/classify_p1_admission.py",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/preflight_p1_launch.py",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/render_p1_onehost_stock_admission.py",
    "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/run_p1_onehost_stock_admission.sh",
    "canon-zero-tim/tests/gemma4_e4b/test_p1_recipe.py",
    "tests/rl/gemma4_e4b_admission_test.py",
    "tunix/rl/gemma4_e4b_admission.py",
})


def _git(repo: Path, *args: str, binary: bool = False):
  return subprocess.check_output(
      ("git", "-C", str(repo), *args), text=not binary
  )


def _source_tree_sha256(repo: Path) -> tuple[str, int]:
  """Hashes tracked plus non-ignored untracked files in path/content order."""
  listing = subprocess.check_output(
      ("git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z")
  )
  paths = sorted(path for path in listing.split(b"\0") if path)
  digest = hashlib.sha256()
  for encoded in paths:
    relative = os.fsdecode(encoded)
    path = repo / relative
    if path.is_symlink():
      payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
      kind = b"symlink"
    elif path.is_file():
      payload = path.read_bytes()
      kind = b"file"
    else:
      raise ValueError(f"unsupported runtime-tree entry: {relative}")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)
    digest.update(kind)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
  return digest.hexdigest(), len(paths)


def _source_reconstruction(repo: Path, source_commit: str, diff: bytes) -> dict:
  """Embeds the dirty-tree delta so a manifest can reconstruct its source."""
  listing = subprocess.check_output(
      (
          "git", "-C", str(repo), "ls-files", "--others",
          "--exclude-standard", "-z",
      )
  )
  paths = sorted(
      os.fsdecode(path) for path in listing.split(b"\0") if path
  )
  unexpected = sorted(set(paths) - ALLOWED_P1_UNTRACKED)
  if unexpected:
    raise ValueError(f"unexpected untracked runtime-tree files: {unexpected}")
  records = []
  for relative in paths:
    path = repo / relative
    if path.is_symlink():
      payload = os.readlink(path).encode("utf-8", errors="surrogateescape")
      kind = "symlink"
    elif path.is_file():
      payload = path.read_bytes()
      kind = "file"
    else:
      raise ValueError(f"unsupported untracked runtime-tree entry: {relative}")
    records.append({
        "path": relative,
        "kind": kind,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "base64": base64.b64encode(payload).decode("ascii"),
    })
  return {
      "schema": "gemma4-e4b-p1-source-reconstruction-v1",
      "base_commit": source_commit,
      "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
      "tracked_diff_base64": base64.b64encode(diff).decode("ascii"),
      "untracked_count": len(records),
      "untracked": records,
  }


def render(repo: Path, workload: str, image_digest: str) -> dict:
  contract = admission.workload_contract(workload)
  if image_digest != PINNED_IMAGE_DIGEST:
    raise ValueError("P1 renderer requires the P0-pinned image digest")
  source_commit = _git(repo, "rev-parse", "HEAD").strip()
  if not SHA40.fullmatch(source_commit):
    raise ValueError("source commit is not a full Git SHA")
  diff = _git(repo, "diff", "--binary", "HEAD", binary=True)
  diff_sha256 = hashlib.sha256(diff).hexdigest()
  if not SHA64.fullmatch(diff_sha256):
    raise AssertionError("source diff hash construction failed")
  profile = repo / "canon-zero-tim/cluster/profiles/gemma4-e4b-dp1-tp4-frozenlake.env"
  if not profile.is_file():
    raise FileNotFoundError(profile)
  source_tree_sha256, source_tree_files = _source_tree_sha256(repo)
  source_reconstruction = _source_reconstruction(repo, source_commit, diff)
  return {
      "schema": "gemma4-e4b-p1-onehost-manifest-v1",
      "workload": workload,
      "source_commit": source_commit,
      "source_diff_sha256": diff_sha256,
      "source_tree_sha256": source_tree_sha256,
      "source_tree_files": source_tree_files,
      "source_reconstruction": source_reconstruction,
      "image_digest": image_digest,
      "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
      "identity": admission.identity_receipt(workload),
      "execution": {
          "host_count": 1,
          "local_tpu_devices": 4,
          "mesh": {"dp": 1, "tp": 4},
          "semantic_prompts": 1,
          "generations_per_prompt": 1,
          "mini_batch_size": 1,
          "train_micro_batch_size": 1,
          "compute_logps_micro_batch_size": 1,
          "max_turns": contract["max_turns"],
          "max_prompt_length": contract["max_prompt_length"],
          "max_response_length": contract["max_response_length"],
          "context_hard_cap": contract["context_hard_cap"],
          "rollout_only": True,
          "prefix_caching": False,
          "checkpoint": False,
          "backward": 0,
          "optimizer_commits": 0,
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--repo", type=Path, required=True)
  parser.add_argument("--workload", choices=tuple(admission.WORKLOADS), required=True)
  parser.add_argument("--image-digest", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  manifest = render(args.repo.resolve(), args.workload, args.image_digest)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=args.output.name + ".", dir=args.output.parent)
  try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
      json.dump(manifest, handle, indent=2, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.chmod(temporary, 0o600)
    os.replace(temporary, args.output)
  finally:
    if os.path.exists(temporary):
      os.unlink(temporary)
  print(
      "GEMMA4_E4B_P1_RENDER_PASS "
      f"workload={args.workload} output={args.output}",
      flush=True,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
