"""Positive, corrupt-input and live-exit controls for fp64 criteria checks."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


_PATH = Path(__file__).with_name("check_fp64_repin.py")
_SPEC = importlib.util.spec_from_file_location("p61_repin_tests", _PATH)
CHECK = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CHECK)
LEAVES = ("['layers'][0]['attn']['w']", "['layers'][0]['norm']['w']")


def _metric(reference, value):
  ref_norm = float(np.linalg.norm(reference))
  got_norm = float(np.linalg.norm(value))
  diff_norm = float(np.linalg.norm(value - reference))
  return {
      "ref_norm": ref_norm, "got_norm": got_norm, "diff_norm": diff_norm,
      "rel_l2": diff_norm / ref_norm if ref_norm else (0.0 if not got_norm else math.inf),
      "one_minus_cos": (max(0.0, 1.0 - max(-1.0, min(1.0,
          float(np.dot(reference, value)) / (ref_norm * got_norm))))
          if ref_norm and got_norm else (0.0 if ref_norm == got_norm else 1.0)),
      "norm_ratio_error": abs(got_norm / ref_norm - 1) if ref_norm else math.inf,
  }


def _vectors():
  oracle = [np.array([1.0, 0.0]), np.array([0.01, 0.0])]
  a = [np.array([1.0, 0.1]), np.array([0.01, 0.001])]
  b = [np.array([1.0, 0.101]), np.array([0.01, 0.00101])]
  return oracle, a, b


def _from_vectors(oracle, a, b):
  result = {"schema": "canon-p61-fp64-reference-v1", "compute_dtype": "float64", "rows": 4}
  for label, reference, value in (("a_vs_fp64", oracle, a), ("b_vs_fp64", oracle, b), ("a_vs_b", a, b)):
    rows = [_metric(x, y) for x, y in zip(reference, value)]
    result[label] = {
        "overall": _metric(np.concatenate(reference), np.concatenate(value)),
        "groups": dict(zip(("layer[0].attn", "layer[0].norm"), copy.deepcopy(rows))),
        "leaves": dict(zip(LEAVES, copy.deepcopy(rows))),
    }
  return result


@pytest.fixture
def metrics():
  return _from_vectors(*_vectors())


def _bad_vectors(fault):
  oracle, a, b = _vectors()
  if fault == "overall":
    b[0][1] = 0.12
  elif fault == "group":
    b[1][1] = 0.00112
  elif fault == "material-cos":
    b[0][1] = 0.106
  elif fault == "ab-cos":
    b[0][1] = 0.18
  elif fault == "times-tp":
    b = [2 * row for row in a]
  elif fault == "candidate-zero":
    b = [np.zeros_like(row) for row in b]
  elif fault == "control-zero":
    a = [np.zeros_like(row) for row in a]
  elif fault == "oracle-zero":
    oracle = [np.zeros_like(row) for row in oracle]
  else:
    raise AssertionError(fault)
  return _from_vectors(oracle, a, b)


def _evaluate(metrics):
  return CHECK.evaluate(metrics, expected_rows=4, expected_leaves=2, expected_groups=2)


def test_registered_small_norm_criteria_accept_good_vectors(metrics):
  result = _evaluate(metrics)
  assert result["verdict"] == "FP64_REPIN_CRITERIA_PASS"
  assert result["material_groups"] == 1
  assert result["capture_and_g4_admission"] == "NOT_EVALUATED"


def test_immaterial_cosine_uses_the_registered_norm_guard():
  oracle, a, b = _vectors()
  a[1] = np.array([0.01005, 0.00005])
  b[1] = np.array([0.01, 0.000071])
  metrics = _from_vectors(oracle, a, b)
  ma = metrics["a_vs_fp64"]["groups"]["layer[0].norm"]
  mb = metrics["b_vs_fp64"]["groups"]["layer[0].norm"]
  assert mb["one_minus_cos"] / ma["one_minus_cos"] > 1.10
  assert _evaluate(metrics)["verdict"] == "FP64_REPIN_CRITERIA_PASS"


@pytest.mark.parametrize("fault,check", [
    ("overall", "overall_rel_l2_ratio_1.05"),
    ("group", "per_group_rel_l2_ratio_1.10"),
    ("material-cos", "material_group_cos_ratio_1.10"),
    ("ab-cos", "ab_overall_cos_1e-3"),
    ("times-tp", "per_group_b_over_a_norm_0.98_1.02"),
    ("candidate-zero", "candidate_nonzero"),
])
def test_bad_metrics_turn_criteria_red(fault, check):
  result = _evaluate(_bad_vectors(fault))
  assert result["verdict"] == "FP64_REPIN_CRITERIA_RED"
  assert check in result["failed"]


@pytest.mark.parametrize("fault", ["control-zero", "oracle-zero"])
def test_coherent_zero_control_or_oracle_is_no_signal(fault):
  with pytest.raises(CHECK.NoSignal, match="INCONCLUSIVE_NO_SIGNAL"):
    _evaluate(_bad_vectors(fault))


@pytest.mark.parametrize("fault", ["candidate-norms", "zero-candidate-leaves", "nan-ratio"])
def test_review_rejects_contradictory_or_nonfinite_metrics(metrics, fault):
  candidate = metrics["b_vs_fp64"]
  if fault == "candidate-norms":
    for family in ("groups", "leaves"):
      for row in candidate[family].values():
        row["got_norm"] *= 2
    candidate["overall"]["got_norm"] *= 2
  elif fault == "zero-candidate-leaves":
    for row in candidate["leaves"].values():
      row.update(got_norm=0, diff_norm=row["ref_norm"], rel_l2=1, one_minus_cos=1,
                 norm_ratio_error=1)
  else:
    candidate["overall"]["norm_ratio_error"] = float("nan")
  with pytest.raises(CHECK.InvalidMetrics):
    _evaluate(metrics)


@pytest.mark.parametrize("fault", ["schema", "dtype", "rows", "missing-leaf", "missing-group", "renamed-group", "nan", "oracle"])
def test_corrupt_or_incomplete_evidence_fails_closed(metrics, fault):
  if fault == "schema":
    metrics["schema"] = "unexpected"
  elif fault == "dtype":
    metrics["compute_dtype"] = "float32"
  elif fault == "rows":
    metrics["rows"] = 1
  elif fault == "missing-leaf":
    for name in CHECK.BLOCKS:
      del metrics[name]["leaves"][LEAVES[1]]
  elif fault == "missing-group":
    for name in CHECK.BLOCKS:
      del metrics[name]["groups"]["layer[0].norm"]
  elif fault == "renamed-group":
    groups = metrics["b_vs_fp64"]["groups"]
    groups["foreign"] = groups.pop("layer[0].norm")
  elif fault == "nan":
    metrics["a_vs_b"]["leaves"][LEAVES[0]]["ref_norm"] = float("nan")
  else:
    metrics["b_vs_fp64"]["groups"]["layer[0].norm"]["ref_norm"] *= 2
  with pytest.raises(CHECK.InvalidMetrics):
    _evaluate(metrics)


def test_zero_error_baseline_is_not_divided_by_zero_or_skipped():
  oracle, a, b = _vectors()
  a[1], b[1] = oracle[1].copy(), oracle[1].copy()
  assert _evaluate(_from_vectors(oracle, a, b))["verdict"] == "FP64_REPIN_CRITERIA_PASS"
  b[1][1] = 0.0001
  result = _evaluate(_from_vectors(oracle, a, b))
  assert result["verdict"] == "FP64_REPIN_CRITERIA_RED"
  assert "per_group_rel_l2_ratio_1.10" in result["failed"]


@pytest.mark.parametrize("fault", ["stale-ab", "stale-summary", "swapped-leaves", "inf-ratio"])
def test_coherent_subreports_cannot_be_spliced(metrics, fault):
  changed = _bad_vectors("candidate-zero")
  if fault == "stale-ab":
    metrics["b_vs_fp64"] = _bad_vectors("times-tp")["b_vs_fp64"]
  elif fault == "stale-summary":
    for name in CHECK.BLOCKS:
      metrics[name]["leaves"] = changed[name]["leaves"]
  elif fault == "swapped-leaves":
    for name in CHECK.BLOCKS:
      leaves = metrics[name]["leaves"]
      leaves[LEAVES[0]], leaves[LEAVES[1]] = leaves[LEAVES[1]], leaves[LEAVES[0]]
  else:
    metrics["a_vs_fp64"]["overall"]["norm_ratio_error"] = math.inf
  with pytest.raises(CHECK.InvalidMetrics):
    _evaluate(metrics)


def test_real_producer_statistics_and_grouping_are_compatible():
  # Execute the actual pure producer code, including streaming aggregation,
  # without importing its TPU/JAX/model dependencies or rewriting the producer.
  path = _PATH.with_name("fp64_reference.py")
  tree = ast.parse(path.read_text())
  names = {"metrics", "leaf_group", "_Stats", "compare_trees"}
  nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
           and node.name in names]
  assert len(nodes) == 4
  namespace = {"np": np, "math": math}
  exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
  rng = np.random.default_rng(61)
  paths = [*LEAVES, "['layers'][0]['attn']['bias']", "['embed']['w']"]
  oracle = {p: rng.normal(size=(7, 11)) for p in paths}
  a = {p: v + 0.01 * rng.normal(size=v.shape) for p, v in oracle.items()}
  b = {p: v.copy() for p, v in a.items()}
  data = {"schema": "canon-p61-fp64-reference-v1", "compute_dtype": "float64", "rows": 4}
  for name, ref, val in (("a_vs_fp64", oracle, a), ("b_vs_fp64", oracle, b), ("a_vs_b", a, b)):
    data[name] = namespace["compare_trees"](ref, val)
  assert all(CHECK._leaf_group(p) == namespace["leaf_group"](p) for p in paths)
  result = CHECK.evaluate(data, expected_rows=4, expected_leaves=4, expected_groups=3)
  assert result["verdict"] == "FP64_REPIN_CRITERIA_PASS"


@pytest.mark.parametrize("fault,status,verdict", [
    ("good", 0, "FP64_REPIN_CRITERIA_PASS"),
    ("worse", 1, "FP64_REPIN_CRITERIA_RED"),
    ("zero", 2, "INCONCLUSIVE_NO_SIGNAL"),
    ("sha", 1, "INVALID_METRICS"),
    ("duplicate-key", 1, "INVALID_METRICS"),
    ("mixed-blocks", 1, "INVALID_METRICS"),
    ("zero-leaves", 1, "INVALID_METRICS"),
])
def test_cli_exits_match_actual_verdicts(metrics, tmp_path, fault, status, verdict):
  if fault == "worse":
    metrics = _bad_vectors("overall")
  elif fault == "zero":
    metrics = _bad_vectors("control-zero")
  elif fault == "mixed-blocks":
    metrics["b_vs_fp64"] = _bad_vectors("times-tp")["b_vs_fp64"]
  elif fault == "zero-leaves":
    changed = _bad_vectors("candidate-zero")
    for name in CHECK.BLOCKS:
      metrics[name]["leaves"] = changed[name]["leaves"]
  payload = json.dumps(metrics)
  if fault == "duplicate-key":
    payload = payload.replace('"rows": 4', '"rows": 4, "rows": 4')
  path = tmp_path / "metrics.json"
  path.write_text(payload)
  sha = hashlib.sha256(path.read_bytes()).hexdigest() if fault != "sha" else "0" * 64
  result = subprocess.run(
      [sys.executable, str(_PATH), str(path), "--expected-sha256", sha,
       "--expected-rows", "4", "--expected-leaves", "2", "--expected-groups", "2"],
      capture_output=True, text=True, check=False, timeout=30,
  )
  assert result.returncode == status, result.stderr
  assert json.loads(result.stdout)["verdict"] == verdict
