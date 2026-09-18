"""Positive and adversarial controls for complete, source-bound bit comparison."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_PATH = Path(__file__).with_name("compare_capture_bits.py")
_SPEC = importlib.util.spec_from_file_location("p61_bits_test_module", _PATH)
BITS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(BITS)


def _json(path, value):
  path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _write_tree(root, name, arrays, paths=None):
  directory = root / "p61_numerical" / name
  directory.mkdir(parents=True, exist_ok=True)
  leaves = []
  for index, array in enumerate(arrays):
    array = np.ascontiguousarray(array)
    path = directory / f"leaf_{index:05d}.npy"
    np.save(path, array, allow_pickle=False)
    leaves.append({
        "index": index, "path": paths[index] if paths else f"['weight{index}']",
        "file": path.name, "dtype": str(array.dtype), "shape": list(array.shape),
        "elements": int(array.size), "data_bytes": int(array.nbytes),
        "file_sha256": BITS._hash(path),
        "data_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    })
  manifest = {
      "schema": "canon-p61-full-tree-capture-v1", "capture": name,
      "leaf_count": len(leaves), "leaves": leaves,
      "total_data_bytes": sum(leaf["data_bytes"] for leaf in leaves),
  }
  _json(directory / "manifest.json", manifest)


def _raw(root, *, plain="1", vma=2):
  lines = [f"[PATHTRACE] CANON_MATMUL_VJP_PLAIN={plain} projection VJPs"]
  lines.extend(["[P66.VMA] outer_check_enabled"] * vma)
  for name in BITS.TREES:
    path = root / "p61_numerical" / name / "manifest.json"
    m = json.loads(path.read_text())
    lines.append(f"[P61.NUMERICAL] capture_complete name={name} "
                 f"leaves={m['leaf_count']} bytes={m['total_data_bytes']} manifest={path}")
  (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seal(root, source):
  return {"source_commit": source,
          "runtime_capture_root": str(root / "p61_numerical"),
          "files": {name: BITS._hash(root / name) for name in BITS.SEALED_FILES}}


@pytest.fixture
def pair(tmp_path):
  common = {
      "schema": "canon.v2-frozenlake-onehost.run.v1", "stage": "backward-no-commit",
      "workload": "p45", "model_id": "synthetic-test-model",
      "model_snapshot_sha256": "1" * 64, "image_id_sha256": "2" * 64,
      "dataset_train_sha256": "3" * 64, "dataset_test_sha256": "4" * 64,
      "topology": {"dp": 2, "tp": 2, "devices": 4}, "checked_vma": True,
      "selectors": {"keep_tape": "stream", "reduce_once": "0"},
      "training_capsule": {"mode": "replay", "capture_run": "fixture_producer",
                           "sha256": "5" * 64, "model_binding_sha256": "6" * 64},
      "global_m": 512, "local_m": 256, "global_trajectories": 4,
      "gradient_groups": 2, "max_prompt_length": 4096, "max_response_length": 2048,
      "reducer_schedule": {"kind": "fixed-local-byte-buckets", "max_local_bytes": 2147483648},
  }
  contract = {"schema": BITS.SCHEMA, "common": common, "gradient_leaves": 2}
  roots = []
  for arm, source in (("left", "a" * 40), ("right", "b" * 40)):
    root = tmp_path / arm
    root.mkdir()
    roots.append(root)
    for name, arrays in (
        ("model_before", [np.array([1, 2], np.float32), np.array([3, 4], np.float32)]),
        ("gradient", [np.array([1, 0], np.float32), np.array([0.5, -2], np.float32)]),
        ("example", [np.array([1, 2], np.int32)]),
        ("logps", [np.array([-1, -2], np.float32)]),
    ):
      _write_tree(root, name, arrays)
    _json(root / "p61_numerical/algo_config.json", {
        "schema": "canon-p61-algo-config-v1", "algo_config": {"beta": 0.0},
        "pad_id": 0, "eos_id": 1,
    })
    _json(root / "run_manifest.json", {
        **common, "label": arm, "source_commit": source,
        "source_diff_sha256": hashlib.sha256(b"").hexdigest(), "runner_sha256": "7" * 64,
    })
    _raw(root)
    contract[arm] = _seal(root, source)
  return *roots, contract


def _reseal_right(pair, *, raw=True):
  _, right, contract = pair
  if raw:
    _raw(right)
  contract["right"] = _seal(right, contract["right"]["source_commit"])


def test_full_bits_equal_is_not_runtime_or_optimizer_admission(pair):
  result = BITS.compare(*pair)
  assert result["verdict"] == "FULL_CAPTURE_BITWISE_EQUAL"
  assert result["gradient_leaves"] == 2
  assert result["gradient_nonzero_elements"] == 3
  assert result["optimizer_commits"] == 0
  assert result["runtime_alignment_admission"] == "NOT_EVALUATED"
  assert result["performance_admission"] == "NOT_EVALUATED"


@pytest.mark.parametrize("kind", ["equal-norm", "signed-zero", "normal-bit", "times-tp", "dtype"])
def test_all_gradient_bits_and_dtypes_are_checked(pair, kind):
  _, right, _ = pair
  first = np.array([1, 0], np.float32)
  second = np.array([0.5, -2], np.float32)
  if kind == "equal-norm":
    first = first[::-1].copy()
  elif kind == "signed-zero":
    first[1] = -0.0
  elif kind == "normal-bit":
    first.view(np.uint32)[0] ^= np.uint32(1)
  elif kind == "times-tp":
    first *= 2
    second *= 2
  elif kind == "dtype":
    first = first.astype(np.float64)
  _write_tree(right, "gradient", [first, second])
  _reseal_right(pair)
  with pytest.raises(BITS.DifferentCapture, match="gradient"):
    BITS.compare(*pair)


@pytest.mark.parametrize("name", ["model_before", "example", "logps"])
def test_bound_input_weight_and_logps_bits_are_checked(pair, name):
  _, right, _ = pair
  manifest = json.loads((right / "p61_numerical" / name / "manifest.json").read_text())
  arrays = [np.load(right / "p61_numerical" / name / leaf["file"]) for leaf in manifest["leaves"]]
  arrays[0][0] += 1
  _write_tree(right, name, arrays)
  _reseal_right(pair)
  with pytest.raises(BITS.DifferentCapture, match=name):
    BITS.compare(*pair)


@pytest.mark.parametrize("kind", ["missing", "extra", "duplicate-path", "shape", "metadata", "nan", "symlink"])
def test_incomplete_or_corrupt_capture_is_rejected(pair, kind):
  _, right, _ = pair
  directory = right / "p61_numerical/gradient"
  manifest_path = directory / "manifest.json"
  manifest = json.loads(manifest_path.read_text())
  if kind == "missing":
    (directory / "leaf_00001.npy").unlink()
  elif kind == "extra":
    np.save(directory / "leaf_00002.npy", np.array([1], np.float32))
  elif kind == "duplicate-path":
    manifest["leaves"][1]["path"] = manifest["leaves"][0]["path"]
    _json(manifest_path, manifest)
  elif kind == "shape":
    _write_tree(right, "gradient", [np.array([[1, 0]], np.float32), np.array([0.5, -2], np.float32)])
  elif kind == "metadata":
    manifest["leaves"][0]["dtype"] = "float64"
    _json(manifest_path, manifest)
  elif kind == "nan":
    _write_tree(right, "gradient", [np.array([np.nan, 0], np.float32), np.array([0.5, -2], np.float32)])
  elif kind == "symlink":
    payload = directory / "leaf_00000.npy"
    target = directory / "original.npy"
    payload.rename(target)
    payload.symlink_to(target)
  _reseal_right(pair)
  with pytest.raises(BITS.InvalidEvidence):
    BITS.compare(*pair)


def test_truncating_both_trees_does_not_turn_prefix_equality_into_full_equality(pair):
  left, right, contract = pair
  for arm, root in (("left", left), ("right", right)):
    for name in ("model_before", "gradient"):
      _write_tree(root, name, [np.array([1, 0], np.float32)])
      (root / "p61_numerical" / name / "leaf_00001.npy").unlink()
    _raw(root)
    contract[arm] = _seal(root, contract[arm]["source_commit"])
  with pytest.raises(BITS.InvalidEvidence, match="complete expected leaf set"):
    BITS.compare(*pair)


@pytest.mark.parametrize("field", ["source_commit", "source_diff_sha256", "producer", "capsule-sha", "image", "selector"])
def test_provenance_mismatch_is_not_hidden_by_matching_arrays(pair, field):
  _, right, _ = pair
  path = right / "run_manifest.json"
  manifest = json.loads(path.read_text())
  if field == "source_commit":
    manifest[field] = "c" * 40
  elif field == "source_diff_sha256":
    manifest[field] = "0" * 64
  elif field == "producer":
    manifest["training_capsule"]["capture_run"] = "foreign_producer"
  elif field == "capsule-sha":
    manifest["training_capsule"]["sha256"] = "0" * 64
  elif field == "image":
    manifest["image_id_sha256"] = "0" * 64
  else:
    manifest["selectors"]["reduce_once"] = "1"
  _json(path, manifest)
  _reseal_right(pair)
  with pytest.raises(BITS.InvalidEvidence):
    BITS.compare(*pair)


@pytest.mark.parametrize("producer", [None, "", "../foreign", [], 7])
def test_malformed_reviewed_producer_is_rejected(pair, producer):
  pair[2]["common"]["training_capsule"]["capture_run"] = producer
  with pytest.raises(BITS.InvalidEvidence, match="producer"):
    BITS.compare(*pair)


@pytest.mark.parametrize("kind", ["raw-unsealed", "plain-off", "vma-decreased", "terminal-missing", "config"])
def test_runtime_binding_and_configuration_are_required(pair, kind):
  _, right, _ = pair
  if kind == "raw-unsealed":
    with (right / "raw.log").open("a") as output:
      output.write("unreviewed\n")
  elif kind == "plain-off":
    _raw(right, plain="0")
    _reseal_right(pair, raw=False)
  elif kind == "vma-decreased":
    _raw(right, vma=1)
    _reseal_right(pair, raw=False)
  elif kind == "terminal-missing":
    raw = (right / "raw.log").read_text()
    (right / "raw.log").write_text(raw.replace("name=gradient", "name=not_gradient"))
    _reseal_right(pair, raw=False)
  elif kind == "config":
    path = right / "p61_numerical/algo_config.json"
    config = json.loads(path.read_text())
    config["algo_config"]["beta"] = 0.1
    _json(path, config)
    _reseal_right(pair)
  expected = BITS.DifferentCapture if kind == "config" else BITS.InvalidEvidence
  with pytest.raises(expected):
    BITS.compare(*pair)


def test_equal_zero_gradients_are_no_signal_not_a_pass(pair):
  left, right, contract = pair
  for arm, root in (("left", left), ("right", right)):
    _write_tree(root, "gradient", [np.zeros(2, np.float32), np.zeros(2, np.float32)])
    _raw(root)
    contract[arm] = _seal(root, contract[arm]["source_commit"])
  with pytest.raises(BITS.NoSignal, match="INCONCLUSIVE_NO_SIGNAL"):
    BITS.compare(*pair)


def test_same_capture_cannot_be_used_as_two_independent_runs(pair):
  left, _, contract = pair
  with pytest.raises(BITS.InvalidEvidence, match="independent"):
    BITS.compare(left, left, contract)


@pytest.mark.parametrize("receipt", [
    "[P61.NUMERICAL] capture_complete name=gradient leaves=1 bytes=4 manifest=/foreign/manifest.json",
    "[P61.NUMERICAL] capture_complete name=gradient leaves=invalid bytes=4 manifest=/foreign/manifest.json",
    "[P61.NUMERICAL] capture_complete malformed",
    "[PATHTRACE] CANON_MATMUL_VJP_PLAIN=invalid projection VJPs",
])
def test_review_rejects_conflicting_or_malformed_receipts(pair, receipt):
  _, right, _ = pair
  with (right / "raw.log").open("a") as stream:
    stream.write(receipt + "\n")
  _reseal_right(pair, raw=False)
  with pytest.raises(BITS.InvalidEvidence):
    BITS.compare(*pair)


def test_terminal_runtime_path_can_differ_from_host_only_by_reviewed_mapping(pair):
  _, right, contract = pair
  path = right / "raw.log"
  path.write_text(path.read_text().replace(str(right / "p61_numerical"), "/capture/trees"))
  _reseal_right(pair, raw=False)
  contract["right"]["runtime_capture_root"] = "/capture/trees"
  assert BITS.compare(*pair)["verdict"] == "FULL_CAPTURE_BITWISE_EQUAL"
  path.write_text(path.read_text().replace("/capture/trees/gradient/", "/foreign/gradient/"))
  contract["right"]["files"]["raw.log"] = BITS._hash(path)
  with pytest.raises(BITS.InvalidEvidence, match="path differs"):
    BITS.compare(*pair)


@pytest.mark.parametrize("receipt", [
    "[P61.NUMERICAL] capture_complete name=gradient leaves=1 bytes=4 manifest=/foreign/manifest.json",
    "[PATHTRACE] CANON_MATMUL_VJP_PLAIN=invalid projection VJPs",
])
def test_conflicting_receipts_fail_the_real_cli(pair, tmp_path, receipt):
  left, right, contract = pair
  with (right / "raw.log").open("a") as stream:
    stream.write(receipt + "\n")
  _reseal_right(pair, raw=False)
  path, output = tmp_path / "contract.json", tmp_path / "receipt.json"
  _json(path, contract)
  result = subprocess.run(
      [sys.executable, str(_PATH), "--left-root", str(left), "--right-root", str(right),
       "--contract", str(path), "--output", str(output)],
      capture_output=True, text=True, check=False, timeout=30,
  )
  assert result.returncode == 1
  assert json.loads(output.read_text())["verdict"] == "INVALID_EVIDENCE"


def test_cli_writes_scoped_receipt_and_never_overwrites(pair, tmp_path):
  left, right, contract = pair
  contract_path, output = tmp_path / "contract.json", tmp_path / "receipt.json"
  _json(contract_path, contract)
  cmd = [sys.executable, str(_PATH), "--left-root", str(left), "--right-root", str(right),
         "--contract", str(contract_path), "--output", str(output)]
  result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
  assert result.returncode == 0, result.stderr
  before = output.read_bytes()
  assert json.loads(before)["verdict"] == "FULL_CAPTURE_BITWISE_EQUAL"
  assert json.loads(before)["contract_sha256"] == BITS._hash(contract_path)
  again = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
  assert again.returncode != 0
  assert output.read_bytes() == before


@pytest.mark.parametrize("kind,status,verdict", [
    ("bits", 1, "FULL_CAPTURE_BITS_DIFFER"),
    ("zero", 2, "INCONCLUSIVE_NO_SIGNAL"),
    ("duplicate-json-key", 1, "INVALID_EVIDENCE"),
])
def test_cli_negative_verdicts_have_nonzero_exit_codes(pair, tmp_path, kind, status, verdict):
  left, right, contract = pair
  if kind == "bits":
    _write_tree(right, "gradient", [np.array([2, 0], np.float32), np.array([1, -4], np.float32)])
    _reseal_right(pair)
  elif kind == "zero":
    for arm, root in (("left", left), ("right", right)):
      _write_tree(root, "gradient", [np.zeros(2, np.float32), np.zeros(2, np.float32)])
      _raw(root)
      contract[arm] = _seal(root, contract[arm]["source_commit"])
  contract_path, output = tmp_path / "contract.json", tmp_path / "receipt.json"
  _json(contract_path, contract)
  if kind == "duplicate-json-key":
    raw = contract_path.read_text()
    contract_path.write_text(raw.replace('"gradient_leaves": 2',
                                        '"gradient_leaves": 2, "gradient_leaves": 2'))
  result = subprocess.run(
      [sys.executable, str(_PATH), "--left-root", str(left), "--right-root", str(right),
       "--contract", str(contract_path), "--output", str(output)],
      capture_output=True, text=True, check=False, timeout=30,
  )
  assert result.returncode == status, result.stderr
  assert json.loads(output.read_text())["verdict"] == verdict
