# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare immutable student and teacher snapshots for the full OPD recipe.

Usage: python -m tunix.dist_rl.examples.on_policy_distillation.models prepare
    --output-dir /shared/opd/models

Only configuration, tokenizer files, and safetensors are downloaded. Model IDs
and full commit revisions are recipe constants, never environment overrides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from huggingface_hub import snapshot_download
import safetensors
from transformers import AutoTokenizer
from tunix.dist_rl.distillation import teacher

STUDENT_MODEL_ID = "Qwen/Qwen3-8B"
STUDENT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
STUDENT_MODEL_NAME = "Qwen3-8B"
TEACHER_MODEL_ID = "Qwen/Qwen3-32B"
TEACHER_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
TEACHER_MODEL_NAME = "Qwen3-32B"
MANIFEST_FILENAME = "models.json"
SCHEMA_VERSION = 1

_MODEL_SPECS = {
    "student": (STUDENT_MODEL_ID, STUDENT_REVISION, STUDENT_MODEL_NAME),
    "teacher": (TEACHER_MODEL_ID, TEACHER_REVISION, TEACHER_MODEL_NAME),
}
_DOWNLOAD_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
    "*.safetensors",
    "model.safetensors.index.json",
)


def _read_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as stream:
    value = json.load(stream)
  if not isinstance(value, dict):
    raise ValueError(f"Expected a JSON object in {path}.")
  return value


def _validate_snapshot(
    path: Path, repo_id: str, revision: str
) -> dict[str, Any]:
  """Checks local loader inputs without loading model tensors into memory."""
  expected_cache_name = "models--" + repo_id.replace("/", "--")
  if (
      path.name != revision
      or path.parent.name != "snapshots"
      or path.parent.parent.name != expected_cache_name
  ):
    raise ValueError(
        f"Expected the immutable HF snapshot for {repo_id}@{revision}."
    )
  required = (
      "config.json",
      "tokenizer.json",
      "tokenizer_config.json",
      "model.safetensors.index.json",
  )
  for filename in required:
    if not (path / filename).is_file():
      raise ValueError(f"Missing required model artifact: {path / filename}.")
  config = _read_json(path / "config.json")
  tokenizer_config = _read_json(path / "tokenizer_config.json")
  if config.get("model_type") != "qwen3":
    raise ValueError(
        "The fixed recipe requires the built-in Qwen3 model implementation."
    )
  if config.get("auto_map") or tokenizer_config.get("auto_map"):
    raise ValueError(
        "Remote model or tokenizer code is not supported by this recipe."
    )
  weight_map = _read_json(path / "model.safetensors.index.json").get(
      "weight_map"
  )
  if not isinstance(weight_map, dict) or not weight_map:
    raise ValueError(
        "The safetensors index must contain a nonempty weight_map."
    )
  shards: dict[str, set[str]] = {}
  for tensor_name, filename in weight_map.items():
    if (
        not isinstance(tensor_name, str)
        or not isinstance(filename, str)
        or Path(filename).name != filename
        or not filename.endswith(".safetensors")
    ):
      raise ValueError(
          "The safetensors index contains an invalid tensor or shard path."
      )
    shards.setdefault(filename, set()).add(tensor_name)
  for filename, expected_keys in shards.items():
    shard = path / filename
    if not shard.is_file():
      raise ValueError(f"Missing safetensors shard: {shard}.")
    with safetensors.safe_open(str(shard), framework="numpy") as handle:
      if not expected_keys.issubset(handle.keys()):
        raise ValueError(
            f"Safetensors shard is missing indexed tensors: {shard}."
        )
  if {item.name for item in path.glob("*.safetensors")} != set(shards):
    raise ValueError(
        "Snapshot contains safetensors files outside its pinned index."
    )
  # Small configuration/tokenizer files are hashed. Large weight files retain
  # their size and immutable Hub revision; this is not full weight attestation.
  inventory = {}
  metadata_files = {
      filename
      for filename in _DOWNLOAD_PATTERNS
      if "*" not in filename and (path / filename).is_file()
  }
  for filename in sorted(set(required) | set(shards) | metadata_files):
    artifact = path / filename
    entry: dict[str, Any] = {"size_bytes": artifact.stat().st_size}
    if not filename.endswith(".safetensors"):
      entry["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    inventory[filename] = entry
  return inventory


def _load_tokenizer(path: str) -> Any:
  tokenizer = AutoTokenizer.from_pretrained(
      path, use_fast=True, trust_remote_code=False, local_files_only=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def _tokenizer_fingerprint(student_path: str, teacher_path: str) -> str:
  scorer = teacher.SGLangTeacherScorer(
      "http://127.0.0.1:1",
      student_tokenizer=_load_tokenizer(student_path),
      teacher_tokenizer=_load_tokenizer(teacher_path),
  )
  return scorer.tokenizer_fingerprint


def _model_record(role: str, snapshot_path: str) -> dict[str, Any]:
  repo_id, revision, model_name = _MODEL_SPECS[role]
  path = Path(snapshot_path).expanduser().resolve()
  artifacts = _validate_snapshot(path, repo_id, revision)
  return {
      "repo_id": repo_id,
      "revision": revision,
      "model_name": model_name,
      "snapshot_path": str(path),
      "tokenizer_path": str(path),
      "declared_revision": f"hf:{repo_id}@{revision}",
      "artifacts": artifacts,
  }


def load_manifest(path: str | Path) -> dict[str, Any]:
  """Loads and revalidates a prepared manifest; never downloads artifacts."""
  manifest = _read_json(Path(path).expanduser())
  if manifest.get("schema_version") != SCHEMA_VERSION:
    raise ValueError("Unsupported models manifest schema_version.")
  for role in _MODEL_SPECS:
    record = manifest.get(role)
    if not isinstance(record, dict) or not isinstance(
        record.get("snapshot_path"), str
    ):
      raise ValueError(f"Missing {role} snapshot in the models manifest.")
    if not Path(record["snapshot_path"]).is_absolute():
      raise ValueError("Models manifest snapshot paths must be absolute.")
    expected = _model_record(role, record["snapshot_path"])
    if record != expected:
      raise ValueError(
          f"The {role} identity, paths, or artifacts differ from the fixed"
          " recipe."
      )
  fingerprint = _tokenizer_fingerprint(
      manifest["student"]["tokenizer_path"],
      manifest["teacher"]["tokenizer_path"],
  )
  if manifest.get("tokenizer_fingerprint") != fingerprint:
    raise ValueError("Tokenizer semantics differ from the prepared manifest.")
  return manifest


def prepare_models(
    output_dir: str | Path,
    *,
    local_files_only: bool = False,
    max_workers: int = 8,
) -> Path:
  """Downloads the two pinned snapshots and writes a validated models.json."""
  if (
      isinstance(max_workers, bool)
      or not isinstance(max_workers, int)
      or max_workers < 1
  ):
    raise ValueError("max_workers must be a positive integer.")
  directory = Path(output_dir).expanduser().resolve()
  directory.mkdir(parents=True, exist_ok=True)
  manifest: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
  for role, (repo_id, revision, _) in _MODEL_SPECS.items():
    snapshot = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(directory / "hf_cache"),
        endpoint="https://huggingface.co",
        allow_patterns=list(_DOWNLOAD_PATTERNS),
        local_files_only=local_files_only,
        max_workers=max_workers,
    )
    manifest[role] = _model_record(role, snapshot)
  manifest["tokenizer_fingerprint"] = _tokenizer_fingerprint(
      manifest["student"]["tokenizer_path"],
      manifest["teacher"]["tokenizer_path"],
  )
  destination = directory / MANIFEST_FILENAME
  temporary_path = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=directory,
        prefix=".models-",
        delete=False,
    ) as stream:
      temporary_path = Path(stream.name)
      json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
      stream.write("\n")
    os.replace(temporary_path, destination)
  finally:
    if temporary_path is not None:
      temporary_path.unlink(missing_ok=True)
  return destination


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  prepare = commands.add_parser(
      "prepare", help="Download the fixed student and teacher snapshots."
  )
  prepare.add_argument("--output-dir", required=True)
  prepare.add_argument("--local-files-only", action="store_true")
  prepare.add_argument("--max-workers", type=int, default=8)
  args = parser.parse_args(argv)
  print(
      prepare_models(
          args.output_dir,
          local_files_only=args.local_files_only,
          max_workers=args.max_workers,
      )
  )


if __name__ == "__main__":
  main()
