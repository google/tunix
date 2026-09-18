#!/usr/bin/env python3
"""Fail-closed implementation of the registered small-norm fp64 re-pin checks.

Thresholds match the historical repin_check_v2.py: overall/per-kind 1.05,
per-group 1.10, material energy share 1e-3, B/A norm [0.98, 1.02], and
overall A/B one-minus-cos <= 1e-3. This checks sealed metrics only; G4,
capture provenance, geometry certification and approval remain separate gates.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import sys


BLOCKS = ("a_vs_fp64", "b_vs_fp64", "a_vs_b")
METRICS = ("ref_norm", "got_norm", "diff_norm", "rel_l2", "one_minus_cos", "norm_ratio_error")
# Report-consistency tolerances, NOT numerical admission thresholds. Different
# BLAS summation paths generate leaf and concatenation statistics independently.
CONSISTENCY_RTOL = 1e-12
CONSISTENCY_ATOL = 1e-12  # Only for dimensionless cosine / normalized square sums.


class InvalidMetrics(ValueError):
  pass


class NoSignal(ValueError):
  pass


def _require(condition, message):
  if not condition:
    raise InvalidMetrics(message)


def _metric(value, label):
  _require(isinstance(value, dict), f"invalid metric block: {label}")
  for key in METRICS:
    x = value.get(key)
    _require(type(x) in (int, float) and not math.isnan(x) and x >= 0,
             f"invalid nonnegative {key}: {label}")
    if key not in ("rel_l2", "norm_ratio_error"):
      _require(math.isfinite(x), f"invalid finite {key}: {label}")
  _require(value["one_minus_cos"] <= 2, f"invalid cosine: {label}")
  r, g, d = (value[k] for k in ("ref_norm", "got_norm", "diff_norm"))
  relative = d / r if r > 0 else (0.0 if g == 0 else math.inf)
  norm_error = abs(g / r - 1) if r > 0 else math.inf
  if r > 0:
    _require(math.isfinite(relative) and math.isfinite(norm_error),
             f"overflowed derived metric: {label}")
  _close(value["rel_l2"], relative, label + "/rel_l2")
  _close(value["norm_ratio_error"], norm_error, label + "/norm_ratio_error")
  if r == 0 or g == 0:
    _require(value["one_minus_cos"] == (0.0 if r == g else 1.0),
             f"inconsistent zero-vector cosine: {label}")
  scale = max(r, g, d)
  if scale:
    rs, gs, ds = r / scale, g / scale, d / scale
    expected = (rs - gs) ** 2 + 2 * rs * gs * value["one_minus_cos"]
    _require(math.isclose(ds ** 2, expected, rel_tol=CONSISTENCY_RTOL,
                         abs_tol=CONSISTENCY_ATOL), f"inconsistent distance/cosine: {label}")


def _close(actual, expected, label):
  # No absolute tolerance on norms: a positive norm must never become zero.
  _require(math.isclose(actual, expected, rel_tol=CONSISTENCY_RTOL, abs_tol=0),
           f"inconsistent metric: {label}")


def _leaf_group(path):
  # Mirror fp64_reference.leaf_group without importing its JAX/model runtime.
  # A regression executes the producer's pure definitions on the same trees.
  if path.startswith("['layers']"):
    rest = path[len("['layers']"):]
    index = rest[1:rest.index("]")]
    tail = rest[rest.index("]") + 1:]
    kind = tail.split("'")[1] if "'" in tail else tail
    return f"layer[{index}].{kind}"
  return path.split("'")[1] if "'" in path else path


def _aggregate(summary, children, label):
  for key in ("ref_norm", "got_norm", "diff_norm"):
    _close(summary[key], math.hypot(*(row[key] for row in children)), label + "/" + key)
  r, g = summary["ref_norm"], summary["got_norm"]
  if r and g:
    cosine = math.fsum((row["ref_norm"] / r) * (row["got_norm"] / g)
                       * (1 - row["one_minus_cos"]) for row in children)
    expected = max(0.0, 1 - max(-1.0, min(1.0, cosine)))
    _require(math.isclose(summary["one_minus_cos"], expected,
                         rel_tol=CONSISTENCY_RTOL, abs_tol=CONSISTENCY_ATOL),
             f"inconsistent aggregate cosine: {label}")


def _consistent_blocks(data):
  for name in BLOCKS:
    block = data[name]
    grouped = collections.defaultdict(list)
    for path, row in block["leaves"].items():
      grouped[_leaf_group(path)].append(row)
    _require(grouped.keys() == block["groups"].keys(), f"incorrect leaf/group mapping: {name}")
    for key, rows in grouped.items():
      _aggregate(block["groups"][key], rows, name + "/groups/" + key)
    _aggregate(block["overall"], list(block["leaves"].values()), name + "/overall/leaves")
    _aggregate(block["overall"], list(block["groups"].values()), name + "/overall/groups")
  a, b, ab = (data[name] for name in BLOCKS)
  rows = [(a["overall"], b["overall"], ab["overall"], "overall")]
  for family in ("groups", "leaves"):
    rows.extend((a[family][k], b[family][k], ab[family][k], family + "/" + k)
                for k in a[family])
  for am, bm, abm, label in rows:
    _require(am["ref_norm"] == bm["ref_norm"], f"different fp64 references: {label}")
    _close(am["got_norm"], abm["ref_norm"], label + "/control-gradient")
    _close(bm["got_norm"], abm["got_norm"], label + "/candidate-gradient")


def _ratio(new, old):
  # A nonzero error against an exactly-zero baseline must not be skipped.
  return new / old if old > 0 else (1.0 if new == 0 else math.inf)


def _kind_errors(block):
  sums = collections.defaultdict(lambda: [0.0, 0.0])
  for group, metric in block["groups"].items():
    kind = re.sub(r"^layer\[\d+\]\.", "", group)
    sums[kind][0] += metric["diff_norm"] ** 2
    sums[kind][1] += metric["ref_norm"] ** 2
  return {kind: math.sqrt(diff) / math.sqrt(ref) for kind, (diff, ref)
          in sums.items() if ref > 0}


def evaluate(data, *, expected_rows, expected_leaves, expected_groups, share_floor=1e-3):
  _require(isinstance(data, dict) and data.get("schema") == "canon-p61-fp64-reference-v1",
           "wrong fp64 metrics schema")
  _require(all(type(x) is int and x > 0 for x in
               (expected_rows, expected_leaves, expected_groups)), "invalid expected coverage")
  _require(data.get("compute_dtype") == "float64" and type(data.get("rows")) is int
           and data["rows"] == expected_rows, "incomplete fp64 dtype/row coverage")
  _require(math.isfinite(share_floor) and 0 < share_floor < 1, "invalid share floor")
  for name in BLOCKS:
    block = data.get(name)
    _require(isinstance(block, dict), f"missing comparison block: {name}")
    _metric(block.get("overall"), name + "/overall")
    for family, count in (("leaves", expected_leaves), ("groups", expected_groups)):
      members = block.get(family)
      _require(isinstance(members, dict) and len(members) == count,
               f"incomplete {family}: {name}")
      for key, value in members.items():
        _require(isinstance(key, str) and bool(key), f"invalid {family} key")
        _metric(value, name + "/" + key)
      _require(set(members) == set(data[BLOCKS[0]][family]), f"mismatched {family} set: {name}")
  _consistent_blocks(data)
  a, b, ab = (data[name] for name in BLOCKS)
  oa, ob, oab = (block["overall"] for block in (a, b, ab))
  if oa["got_norm"] == 0 or oa["ref_norm"] == 0 or oab["ref_norm"] == 0:
    raise NoSignal("INCONCLUSIVE_NO_SIGNAL: zero control gradient or fp64 reference")
  energy = sum(m["ref_norm"] ** 2 for m in a["groups"].values())
  if energy == 0:
    raise NoSignal("INCONCLUSIVE_NO_SIGNAL: zero grouped fp64 reference")
  checks = {}

  def check(name, condition):
    checks[name] = bool(condition)

  check("candidate_nonzero", ob["got_norm"] > 0)
  check("overall_rel_l2_ratio_1.05", ob["rel_l2"] <= 1.05 * oa["rel_l2"])
  check("overall_cos_ratio_1.05", ob["one_minus_cos"] <= 1.05 * oa["one_minus_cos"])
  check("ab_rel_l2_triangle", oab["rel_l2"] <= 2 * max(oa["rel_l2"], ob["rel_l2"]))
  ka, kb = _kind_errors(a), _kind_errors(b)
  _require(bool(ka) and ka.keys() == kb.keys(), "incomplete per-kind reference coverage")
  check("per_kind_rel_l2_ratio_1.05", all(_ratio(kb[k], ka[k]) <= 1.05 for k in ka))
  check("per_group_rel_l2_ratio_1.10", all(
      _ratio(b["groups"][key]["rel_l2"], m["rel_l2"]) <= 1.10
      for key, m in a["groups"].items()))
  material = [key for key, m in a["groups"].items() if m["ref_norm"] ** 2 / energy >= share_floor]
  _require(bool(material), "no material groups at the reviewed share floor")
  check("material_group_cos_ratio_1.10", all(
      _ratio(b["groups"][key]["one_minus_cos"], a["groups"][key]["one_minus_cos"]) <= 1.10
      for key in material))
  check("ab_overall_cos_1e-3", oab["one_minus_cos"] <= 1e-3)
  check("per_group_b_over_a_norm_0.98_1.02", all(
      0.98 <= _ratio(m["got_norm"], m["ref_norm"]) <= 1.02
      for m in ab["groups"].values()))
  failed = [name for name, passed in checks.items() if not passed]
  overall_ratio = _ratio(ob["rel_l2"], oa["rel_l2"])
  return {
      "verdict": "FP64_REPIN_CRITERIA_PASS" if not failed else "FP64_REPIN_CRITERIA_RED",
      "checks": checks, "failed": failed, "rows": expected_rows,
      "leaves": expected_leaves, "groups": expected_groups, "material_groups": len(material),
      "share_floor": share_floor,
      "overall_b_over_a_rel_l2": overall_ratio if math.isfinite(overall_ratio) else None,
      "capture_and_g4_admission": "NOT_EVALUATED",
  }


def _pairs(pairs):
  result = {}
  for key, value in pairs:
    _require(key not in result, f"duplicate metric key: {key}")
    result[key] = value
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("metrics", type=Path)
  parser.add_argument("--expected-sha256", required=True)
  parser.add_argument("--expected-rows", required=True, type=int)
  parser.add_argument("--expected-leaves", required=True, type=int)
  parser.add_argument("--expected-groups", required=True, type=int)
  parser.add_argument("--share-floor", type=float, default=1e-3)
  args = parser.parse_args()
  try:
    payload = args.metrics.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256) is not None
             and sha == args.expected_sha256, "sealed metrics SHA mismatch")
    data = json.loads(payload, object_pairs_hook=_pairs)
    result = evaluate(data, expected_rows=args.expected_rows,
                      expected_leaves=args.expected_leaves, expected_groups=args.expected_groups,
                      share_floor=args.share_floor)
    result["metrics_sha256"] = sha
    code = 0 if not result["failed"] else 1
  except NoSignal as exc:
    result, code = {"verdict": "INCONCLUSIVE_NO_SIGNAL", "reason": str(exc)}, 2
  except (ValueError, KeyError, TypeError, OverflowError, OSError) as exc:
    result, code = {"verdict": "INVALID_METRICS", "reason": str(exc)}, 1
  print(json.dumps(result, sort_keys=True, allow_nan=False))
  return code


if __name__ == "__main__":
  sys.exit(main())
