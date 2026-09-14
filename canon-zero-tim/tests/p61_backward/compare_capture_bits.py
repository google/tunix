#!/usr/bin/env python3
"""Compare sealed full FrozenLake P61 captures without admitting training.

The independently reviewed contract binds each source, raw log, run manifest,
algorithm config and all four tree manifests. It must be fixed before a
comparison; this tool never generates or refreshes its own expected hashes.
FULL_CAPTURE_BITWISE_EQUAL proves only equality at the P61 diagnostic boundary,
not A/B/C alignment, optimizer commits, performance or final-source admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any

import numpy as np


SCHEMA = "canon-p61-bitwise-contract-v1"
TREES = ("model_before", "example", "logps", "gradient")
SEALED_FILES = (
    "run_manifest.json", "raw.log", "p61_numerical/algo_config.json",
    *(f"p61_numerical/{name}/manifest.json" for name in TREES),
)
IDENTITY_FIELDS = {"label", "source_commit", "source_diff_sha256", "runner_sha256"}
REQUIRED_COMMON = {
    "schema", "stage", "workload", "model_id", "model_snapshot_sha256",
    "image_id_sha256", "topology", "checked_vma", "selectors",
    "training_capsule", "global_m", "local_m", "global_trajectories",
    "gradient_groups", "max_prompt_length", "max_response_length",
    "dataset_train_sha256", "dataset_test_sha256", "reducer_schedule",
}
CHUNK_ELEMENTS = 1_048_576


class InvalidEvidence(ValueError):
  """Missing, unsealed, mismatched or malformed evidence."""


class DifferentCapture(ValueError):
  """A complete comparison found different bits or tree schemas."""


class NoSignal(ValueError):
  """Equal all-zero gradients cannot supply an admission signal."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise InvalidEvidence(message)


def _unique_object(pairs):
  result = {}
  for key, value in pairs:
    _require(key not in result, f"duplicate JSON key: {key}")
    result[key] = value
  return result


def _json(path: Path) -> dict[str, Any]:
  def reject_constant(value):
    raise InvalidEvidence(f"non-finite JSON constant: {value}")
  value = json.loads(path.read_text(encoding="utf-8"),
                     object_pairs_hook=_unique_object, parse_constant=reject_constant)
  _require(isinstance(value, dict), f"expected JSON object: {path}")
  return value


def _hex(value, length: int) -> bool:
  return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _hash(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _regular(path: Path) -> None:
  _require(path.is_file() and not path.is_symlink(), f"not a regular evidence file: {path}")


def _tree(root: Path, name: str) -> dict[str, Any]:
  directory = root / "p61_numerical" / name
  _require(directory.is_dir() and not directory.is_symlink(), f"invalid tree: {name}")
  manifest = _json(directory / "manifest.json")
  _require(manifest.get("schema") == "canon-p61-full-tree-capture-v1"
           and manifest.get("capture") == name, f"invalid tree schema: {name}")
  leaves = manifest.get("leaves")
  _require(isinstance(leaves, list) and bool(leaves), f"empty tree: {name}")
  _require(type(manifest.get("leaf_count")) is int and
           manifest["leaf_count"] == len(leaves), f"leaf count differs: {name}")
  paths = set()
  total_bytes = nonzero = 0
  for index, leaf in enumerate(leaves):
    _require(isinstance(leaf, dict) and type(leaf.get("index")) is int
             and leaf["index"] == index, f"invalid index: {name}/{index}")
    path = leaf.get("path")
    _require(isinstance(path, str) and bool(path) and path not in paths,
             f"missing/duplicate leaf path: {name}/{index}")
    paths.add(path)
    filename = f"leaf_{index:05d}.npy"
    _require(leaf.get("file") == filename, f"invalid filename: {name}/{index}")
    _require(isinstance(leaf.get("shape"), list) and
             all(type(value) is int and value >= 0 for value in leaf["shape"]) and
             type(leaf.get("elements")) is int and leaf["elements"] >= 0 and
             type(leaf.get("data_bytes")) is int and leaf["data_bytes"] >= 0,
             f"invalid shape/count types: {name}/{index}")
    payload = directory / filename
    _regular(payload)
    _require(_hex(leaf.get("file_sha256"), 64) and
             _hash(payload) == leaf["file_sha256"], f"file hash differs: {name}/{path}")
    array = np.load(payload, mmap_mode="r", allow_pickle=False)
    _require(array.flags.c_contiguous and array.dtype.kind in "bifu",
             f"unsupported array layout/dtype: {name}/{path}")
    if name == "gradient":
      _require(array.dtype.kind == "f", f"non-floating gradient: {path}")
    _require(list(array.shape) == leaf.get("shape") and
             str(array.dtype) == leaf.get("dtype") and
             array.size == leaf.get("elements") and
             array.nbytes == leaf.get("data_bytes"), f"array metadata differs: {name}/{path}")
    flat = array.reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, flat.size, CHUNK_ELEMENTS):
      chunk = flat[start:start + CHUNK_ELEMENTS]
      _require(bool(np.all(np.isfinite(chunk))), f"non-finite array: {name}/{path}")
      nonzero += int(np.count_nonzero(chunk))
      digest.update(chunk.tobytes(order="C"))
    _require(_hex(leaf.get("data_sha256"), 64) and
             digest.hexdigest() == leaf["data_sha256"], f"data hash differs: {name}/{path}")
    total_bytes += array.nbytes
  _require(type(manifest.get("total_data_bytes")) is int and
           total_bytes == manifest["total_data_bytes"], f"byte count differs: {name}")
  _require({item.name for item in directory.iterdir()} ==
           {"manifest.json", *(leaf["file"] for leaf in leaves)},
           f"unlisted/missing tree payload: {name}")
  return {"manifest": manifest, "nonzero_elements": nonzero}


def _run(root: Path, binding: dict, common: dict, expected_leaves: int) -> dict:
  _require(root.is_dir() and root.absolute() == root.resolve(), "use a physical capture root")
  _require(not (root / "p61_numerical").is_symlink(), "symlinked numerical directory")
  for name in TREES:
    directory = root / "p61_numerical" / name
    _require(directory.is_dir() and not directory.is_symlink(), f"invalid tree directory: {name}")
  _require(isinstance(binding, dict) and
           set(binding) == {"source_commit", "files", "runtime_capture_root"},
           "invalid source binding fields")
  _require(_hex(binding["source_commit"], 40), "full source SHA required")
  runtime_root = binding["runtime_capture_root"]
  _require(isinstance(runtime_root, str) and runtime_root.startswith("/") and
           str(PurePosixPath(runtime_root)) == runtime_root and
           ".." not in PurePosixPath(runtime_root).parts and
           not any(char.isspace() for char in runtime_root),
           "invalid reviewed runtime capture root")
  _require(isinstance(binding["files"], dict) and
           set(binding["files"]) == set(SEALED_FILES), "incomplete sealed file set")
  for relative, expected in binding["files"].items():
    path = root / relative
    _regular(path)
    _require(_hex(expected, 64) and _hash(path) == expected,
             f"sealed artifact hash differs: {relative}")
  manifest = _json(root / "run_manifest.json")
  _require(manifest.get("source_commit") == binding["source_commit"], "wrong source")
  _require(manifest.get("source_diff_sha256") == hashlib.sha256(b"").hexdigest(), "dirty source")
  _require(_hex(manifest.get("runner_sha256"), 64), "missing runner identity")
  comparable = {key: value for key, value in manifest.items() if key not in IDENTITY_FIELDS}
  _require(comparable == common, "run contract/capsule producer differs from reviewed common contract")
  config = _json(root / "p61_numerical/algo_config.json")
  _require(set(config) == {"schema", "algo_config", "pad_id", "eos_id"} and
           config["schema"] == "canon-p61-algo-config-v1" and
           isinstance(config["algo_config"], dict) and bool(config["algo_config"]) and
           type(config["pad_id"]) is int and type(config["eos_id"]) is int,
           "invalid algorithm configuration")
  raw = (root / "raw.log").read_text(encoding="utf-8")
  plain_lines = [line for line in raw.splitlines()
                 if "[PATHTRACE] CANON_MATMUL_VJP_PLAIN" in line]
  _require(len(plain_lines) == 1 and re.fullmatch(
      r"\[PATHTRACE\] CANON_MATMUL_VJP_PLAIN=1 \S.*", plain_lines[0]) is not None,
           "expected exactly one plain-VJP runtime receipt")
  vma_count = raw.count("[P66.VMA] outer_check_enabled")
  _require(vma_count >= 1, "missing checked-VMA runtime receipt")
  terminals = {}
  for line in raw.splitlines():
    if "[P61.NUMERICAL] capture_complete" not in line:
      continue
    match = re.fullmatch(
        r"\[P61\.NUMERICAL\] capture_complete name=(\S+) "
        r"leaves=(\d+) bytes=(\d+) manifest=(\S+)", line)
    _require(match is not None, "malformed capture terminal")
    name, leaves, data_bytes, manifest_path = match.groups()
    _require(name in TREES and name not in terminals,
             "unknown or duplicate capture terminal: " + name)
    _require(manifest_path == f"{runtime_root}/{name}/manifest.json",
             "capture terminal path differs from reviewed runtime mapping: " + name)
    terminals[name] = (int(leaves), int(data_bytes))
  _require(set(terminals) == set(TREES), "missing capture terminal")
  trees = {name: _tree(root, name) for name in TREES}
  for name, tree in trees.items():
    m = tree["manifest"]
    _require(terminals[name] == (m["leaf_count"], m["total_data_bytes"]),
             "capture terminal counts differ: " + name)
  for name in ("model_before", "gradient"):
    _require(trees[name]["manifest"]["leaf_count"] == expected_leaves,
             f"incomplete expected leaf set: {name}")
  model_shape = [(leaf["path"], leaf["shape"])
                 for leaf in trees["model_before"]["manifest"]["leaves"]]
  gradient_shape = [(leaf["path"], leaf["shape"])
                    for leaf in trees["gradient"]["manifest"]["leaves"]]
  _require(model_shape == gradient_shape, "gradient does not cover the complete parameter tree")
  for relative, expected in binding["files"].items():
    _require(_hash(root / relative) == expected, f"artifact changed during comparison: {relative}")
  return {"trees": trees, "vma_receipts": vma_count}


def compare(left_root: Path, right_root: Path, contract: dict) -> dict:
  _require(contract.get("schema") == SCHEMA, "invalid comparison contract schema")
  _require(set(contract) == {"schema", "common", "left", "right", "gradient_leaves"},
           "unknown/missing comparison contract fields")
  common = contract["common"]
  _require(isinstance(common, dict) and REQUIRED_COMMON <= set(common), "incomplete common contract")
  _require(not IDENTITY_FIELDS.intersection(common), "source identities must be bound per arm")
  _require(common["schema"] == "canon.v2-frozenlake-onehost.run.v1"
           and common["stage"] == "backward-no-commit"
           and common["checked_vma"] is True, "unsupported capture boundary or unchecked VMA")
  topology = common["topology"]
  _require(isinstance(topology, dict) and set(topology) == {"devices", "dp", "tp"}
           and all(type(value) is int and value > 0 for value in topology.values())
           and topology["tp"] > 1 and topology["devices"] == topology["dp"] * topology["tp"],
           "invalid checked TP>1 topology")
  _require(type(common["local_m"]) is int and common["local_m"] > 0 and
           type(common["global_m"]) is int and
           common["global_m"] == topology["dp"] * common["local_m"],
           "global/local row geometry differs")
  capsule = common["training_capsule"]
  _require(isinstance(capsule, dict) and capsule.get("mode") == "replay", "replay capsule required")
  producer = capsule.get("capture_run")
  _require(isinstance(producer, str) and
           re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,127}", producer) is not None,
           "malformed capsule producer")
  for field in ("sha256", "model_binding_sha256"):
    _require(_hex(capsule.get(field), 64), f"invalid capsule {field}")
  for field in ("model_snapshot_sha256", "image_id_sha256",
                "dataset_train_sha256", "dataset_test_sha256"):
    _require(_hex(common[field], 64), f"missing common identity: {field}")
  count = contract["gradient_leaves"]
  _require(type(count) is int and count > 0, "positive complete leaf count required")
  _require(left_root.resolve() != right_root.resolve(), "two independent capture roots required")
  left = _run(left_root, contract["left"], common, count)
  right = _run(right_root, contract["right"], common, count)
  _require(right["vma_receipts"] >= left["vma_receipts"], "checked-VMA receipts decreased")
  for name in TREES:
    a = left["trees"][name]["manifest"]["leaves"]
    b = right["trees"][name]["manifest"]["leaves"]
    keys = ("index", "path", "shape", "dtype", "elements", "data_bytes", "data_sha256")
    identities = [[{key: leaf[key] for key in keys} for leaf in leaves] for leaves in (a, b)]
    if identities[0] != identities[1]:
      raise DifferentCapture(f"full tree bits/schema differ: {name}")
  if _hash(left_root / "p61_numerical/algo_config.json") != _hash(right_root / "p61_numerical/algo_config.json"):
    raise DifferentCapture("algorithm config bytes differ")
  if left["trees"]["gradient"]["nonzero_elements"] == 0:
    raise NoSignal("equal all-zero gradients: INCONCLUSIVE_NO_SIGNAL")
  return {
      "verdict": "FULL_CAPTURE_BITWISE_EQUAL",
      "gradient_leaves": count,
      "gradient_nonzero_elements": left["trees"]["gradient"]["nonzero_elements"],
      "trees": {name: left["trees"][name]["manifest"]["leaf_count"] for name in TREES},
      "sources": [contract[arm]["source_commit"] for arm in ("left", "right")],
      "vma_receipts": [left["vma_receipts"], right["vma_receipts"]],
      "optimizer_commits": 0,
      "runtime_alignment_admission": "NOT_EVALUATED",
      "performance_admission": "NOT_EVALUATED",
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--left-root", required=True, type=Path)
  parser.add_argument("--right-root", required=True, type=Path)
  parser.add_argument("--contract", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists() or args.output.is_symlink():
    parser.error("refusing to overwrite an existing comparison receipt")
  try:
    result = compare(args.left_root, args.right_root, _json(args.contract))
    status = 0
  except NoSignal as exc:
    result, status = {"verdict": "INCONCLUSIVE_NO_SIGNAL", "reason": str(exc)}, 2
  except DifferentCapture as exc:
    result, status = {"verdict": "FULL_CAPTURE_BITS_DIFFER", "reason": str(exc)}, 1
  except (ValueError, KeyError, TypeError, OSError) as exc:
    result, status = {"verdict": "INVALID_EVIDENCE", "reason": str(exc)}, 1
  result["tool_sha256"] = _hash(Path(__file__))
  if args.contract.is_file():
    result["contract_sha256"] = _hash(args.contract)
  with args.output.open("x", encoding="utf-8") as output:
    json.dump(result, output, indent=2, sort_keys=True)
    output.write("\n")
  print(json.dumps(result, sort_keys=True))
  return status


if __name__ == "__main__":
  sys.exit(main())
