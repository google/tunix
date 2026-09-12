"""Immutable local input/evidence binding; never downloads or launches work."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from examples.frozenlake.gemma4_tim import recipe


def file_sha(path: Path) -> str:
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
  with path.open("xb") as stream:
    path.chmod(0o600)
    stream.write(recipe.json_bytes(value))


def _inside(root: Path, name: str) -> Path:
  rel = PurePosixPath(name)
  if rel.is_absolute() or ".." in rel.parts or str(rel) != name:
    raise recipe.RecipeError("unsafe artifact member")
  path = root / name
  if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
    raise recipe.RecipeError("missing or externally linked artifact: " + name)
  return path


def validate_model_config(config: dict) -> None:
  text = config.get("text_config", {})
  expected = {
      "num_hidden_layers": 35, "hidden_size": 1536, "vocab_size": 262144,
      "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256,
      "hidden_size_per_layer_input": 256, "num_kv_shared_layers": 20,
      "intermediate_size": 6144,
      "global_head_dim": 512, "sliding_window": 512,
  }
  if any(text.get(k) != v for k, v in expected.items()):
    raise recipe.RecipeError("checkpoint is not the registered E2B text geometry")
  if text.get("enable_moe_block", False) or not config.get("tie_word_embeddings", True):
    raise recipe.RecipeError("only dense tied-head E2B text is registered")


def input_members(snapshot: Path, data: Path) -> dict[str, Path]:
  config_path = _inside(snapshot, "config.json")
  validate_model_config(json.loads(config_path.read_text()))
  required = ("config.json", "tokenizer.json", "tokenizer_config.json")
  members = {"model/" + n: _inside(snapshot, n) for n in required}
  if (snapshot / "model.safetensors.index.json").exists():
    index = json.loads(_inside(snapshot, "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map", {})
    if not weight_map or not all(isinstance(k, str) and isinstance(v, str)
                                 for k, v in weight_map.items()):
      raise recipe.RecipeError("missing checkpoint weight map")
    shards = set(weight_map.values())
  else:
    shards = {"model.safetensors"}
  if {p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*.safetensors")} != shards:
    raise recipe.RecipeError("checkpoint shard inventory mismatch")
  for name in sorted(shards):
    if not name.endswith(".safetensors"):
      raise recipe.RecipeError("checkpoint shard must be safetensors")
    members["model/" + name] = _inside(snapshot, name)
  # Tokenizer/chat/generation side files are part of identity even when the
  # current loader does not consult them. Reject external HF cache symlinks:
  # remote operators must materialize a self-contained, read-only snapshot.
  for path in sorted(snapshot.rglob("*")):
    if path.is_file() and path.suffix in (".json", ".jinja", ".model", ".txt"):
      name = path.relative_to(snapshot).as_posix()
      members["model/" + name] = _inside(snapshot, name)
  for name in ("train.parquet", "test.parquet"):
    members["data/" + name] = _inside(data, name)
  return members


def seal_inputs(snapshot: Path, data: Path, revision: str, output: Path) -> dict:
  if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise recipe.RecipeError("model revision must be an immutable full commit")
  members = input_members(snapshot, data)
  value = {
      "schema": "gemma4-e2b-inputs-v1", "model_id": recipe.MODEL_ID,
      "model_revision": revision,
      "files": {n: {"sha256": file_sha(p), "bytes": p.stat().st_size}
                for n, p in sorted(members.items())},
      "claim": "local-payload-binding-not-loaded-weight-parity",
  }
  write_json(output, value)
  return value


def verify_inputs(path: Path, expected_sha: str, snapshot: Path, data: Path) -> dict:
  if file_sha(path) != expected_sha:
    raise recipe.RecipeError("input manifest SHA mismatch")
  value = json.loads(path.read_text())
  if (value.get("schema") != "gemma4-e2b-inputs-v1"
      or value.get("model_id") != recipe.MODEL_ID
      or not re.fullmatch(r"[0-9a-f]{40}", value.get("model_revision", ""))):
    raise recipe.RecipeError("input manifest identity mismatch")
  members = input_members(snapshot, data)
  actual = {n: {"sha256": file_sha(p), "bytes": p.stat().st_size}
            for n, p in sorted(members.items())}
  if actual != value.get("files"):
    raise recipe.RecipeError("input payload drift")
  return value


def seal_run(root: Path, verdict: dict) -> str:
  """Caller must first join all writers. No hashed file writes after return."""
  if verdict.get("status") not in ("GREEN", "RED", "INCONCLUSIVE"):
    raise recipe.RecipeError("invalid terminal status")
  write_json(root / "classification.json", verdict)
  # Exactly one terminal, finalized before calculating its digest.
  with (root / "driver.log").open("a", encoding="utf-8") as stream:
    stream.write("GEMMA4_E2B_TERMINAL " + verdict["status"] + "\n")
  members = {}
  for path in sorted(root.rglob("*")):
    if path.is_symlink():
      raise recipe.RecipeError("evidence must not contain symlinks")
    if path.is_file():
      name = path.relative_to(root).as_posix()
      if "\n" in name or "\r" in name or "\\" in name:
        raise recipe.RecipeError("unsafe evidence filename")
      members[name] = file_sha(path)
  manifest = root / "SHA256SUMS"
  with manifest.open("x", encoding="utf-8") as stream:
    stream.writelines(f"{sha}  {name}\n" for name, sha in members.items())
  verify_run(root)
  # Independent GNU verifier, after all writers have stopped. Its output is
  # not appended to any manifest member; a nonzero exit fails the runner.
  subprocess.run(["sha256sum", "--strict", "--check", "SHA256SUMS"], cwd=root,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
  return file_sha(manifest)


def verify_run(root: Path) -> None:
  lines = (root / "SHA256SUMS").read_text().splitlines()
  seen = set()
  for line in lines:
    sha, name = line.split("  ", 1)
    if name in seen or name == "SHA256SUMS" or file_sha(_inside(root, name)) != sha:
      raise recipe.RecipeError("evidence manifest mismatch")
    seen.add(name)
  actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
  if not seen or actual != seen | {"SHA256SUMS"}:
    raise recipe.RecipeError("evidence inventory mismatch")


def main():
  import argparse
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  seal = commands.add_parser("seal-inputs")
  for name in ("snapshot", "data", "output"):
    seal.add_argument("--" + name, required=True, type=Path)
  seal.add_argument("--revision", required=True)
  verify = commands.add_parser("verify-run")
  verify.add_argument("root", type=Path)
  args = parser.parse_args()
  if args.command == "seal-inputs":
    seal_inputs(args.snapshot, args.data, args.revision, args.output)
    print("GEMMA4_E2B_INPUTS_SEALED sha256=" + file_sha(args.output))
  else:
    verify_run(args.root)
    print("GEMMA4_E2B_MANIFEST_PASS sha256=" + file_sha(args.root / "SHA256SUMS"))


if __name__ == "__main__":
  main()
