#!/usr/bin/env python3
"""Validate the immutable e0w5 render before a read-only evidence return."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml


TARGET_SOURCE = "2f61f8fc7cf073964a9adbd30e78de872426a4d2"
RUN_ID = "e0w5"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_GCS_ROOT = re.compile(
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
    r"[a-z0-9-]+/attempt-0"
)


class RenderContractError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise RenderContractError(message)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(root: Path) -> None:
  path = root / "SHA256SUMS"
  _require(path.is_file(), "render SHA256SUMS is absent")
  rows: dict[str, str] = {}
  for line in path.read_text(encoding="ascii").splitlines():
    digest, separator, name = line.partition("  ")
    _require(
        separator == "  "
        and _SHA256.fullmatch(digest) is not None
        and name not in rows
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts,
        "render manifest row is invalid",
    )
    rows[name] = digest
  actual = {
      path.relative_to(root).as_posix()
      for path in root.rglob("*")
      if path.is_file() and path.name != "SHA256SUMS"
  }
  _require(set(rows) == actual, "render manifest membership drifted")
  for name, digest in rows.items():
    _require(_sha256(root / name) == digest, f"render hash drifted: {name}")


def _container(document: dict[str, Any]) -> dict[str, Any]:
  containers = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]["containers"]
  values = [item for item in containers if item.get("name") == "jax-tpu"]
  _require(len(values) == 1 and containers[0] is values[0],
           "rendered jax-tpu container identity drifted")
  return values[0]


def _env(document: dict[str, Any]) -> dict[str, str]:
  return {
      str(row["name"]): str(row["value"])
      for row in _container(document)["env"] if "value" in row
  }


def _normalize(value: Any, arm: str) -> Any:
  if isinstance(value, str):
    return value.replace(f"-m15-{arm}-", "-m15-<ARM>-")
  if isinstance(value, list):
    return [_normalize(item, arm) for item in value]
  if isinstance(value, dict):
    return {key: _normalize(item, arm) for key, item in value.items()}
  return value


def validate(root: Path, target_source: str = TARGET_SOURCE) -> dict[str, Any]:
  _require(root.is_dir(), "original e0w5 render directory is absent")
  _manifest(root)
  contract_path = root / "RUN_CONTRACT.json"
  _require(contract_path.is_file(), "e0w5 RUN_CONTRACT.json is absent")
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  _require(isinstance(contract, dict), "e0w5 run contract is not an object")
  expected_contract = {
      "schema": "m15-e0v-tito-layer-render-v1",
      "source_commit": target_source,
      "run_id": RUN_ID,
      "program_identity": "m15-apc-debug-exact-tito-layer-v1",
      "observer": "layer",
      "rounds": 3,
      "zero_backward": True,
      "zero_optimizer_commit": True,
      "b_full_reset_immutable": True,
      "control_and_treatment_differ_only_at_apc": True,
      "tito_exact_both_arms": True,
      "launch_authorized": False,
      "target_executed": False,
      "remote_mutation": False,
  }
  wrong = {
      name: contract.get(name)
      for name, value in expected_contract.items()
      if contract.get(name) != value
  }
  _require(not wrong, f"e0w5 run contract drifted: {sorted(wrong)}")

  paths = sorted(root.glob("jobset-v1-apc-m15-*-*.yaml"))
  _require(len(paths) == 2, "e0w5 render must contain exactly two JobSet YAMLs")
  rows: dict[str, dict[str, str]] = {}
  normalized: list[dict[str, Any]] = []
  expected_values = {
      "CANON_EXPECT_COMMIT": target_source,
      "CANON_M15_TOKEN_CONTINUITY": "exact",
      "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
      "CANON_P38_DURABILITY_PROFILE": "m15-wide-v1",
      "CANON_P38_SEAM_OBSERVER": "layer",
      "CANON_P38_TAIL_OBSERVER": "1",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
  }
  for path in paths:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = _env(document)
    arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
    _require(arm in ("off", "on") and arm not in rows,
             "e0w5 arm membership drifted")
    jobset = str(document.get("metadata", {}).get("name", ""))
    expected_jobset = f"canon-v1-apc-m15-{arm}-{RUN_ID}-{target_source[:8]}"
    _require(jobset == expected_jobset, "e0w5 JobSet identity drifted")
    _require(
        path.name in (
            f"jobset-v1-apc-m15-{arm}-{RUN_ID}.yaml",
            f"jobset-v1-apc-m15-{arm}-layer.yaml",
        ),
        "e0w5 YAML filename drifted",
    )
    wrong_env = {
        name: env.get(name)
        for name, value in expected_values.items()
        if env.get(name) != value
    }
    _require(not wrong_env, f"e0w5 signed environment drifted: {sorted(wrong_env)}")
    _require(
        env.get("CANON_VLLM_ENABLE_PREFIX_CACHING")
        == ("1" if arm == "on" else "0"),
        "e0w5 APC treatment drifted",
    )
    _require(_GCS_ROOT.fullmatch(env.get("CANON_P38_GCS_PREFIX", "")) is not None,
             "e0w5 GCS locator drifted")
    _require(not any(name.startswith("CANON_P38_KV_OBSERVER") for name in env),
             "e0w5 render contains a historical KV observer")
    _require(
        document.get("metadata", {}).get("labels", {}).get(
            "canon.zero-tim/m15-token-continuity"
        ) == "exact",
        "e0w5 exact-TiTO label drifted",
    )
    candidate = copy.deepcopy(document)
    candidate_env = _container(candidate)["env"]
    for item in candidate_env:
      if item["name"] == "CANON_APC_M15_TARGET_DEBUG":
        item["value"] = "<ARM>"
      elif item["name"] == "CANON_VLLM_ENABLE_PREFIX_CACHING":
        item["value"] = "<APC>"
    candidate["metadata"]["labels"]["canon.zero-tim/apc-m15-arm"] = "<ARM>"
    normalized.append(_normalize(candidate, arm))
    rows[arm] = {
        "jobset": jobset,
        "yaml": path.name,
        "sha256": _sha256(path),
    }
  _require(set(rows) == {"off", "on"}, "e0w5 pair is incomplete")
  _require(normalized[0] == normalized[1],
           "e0w5 arms differ beyond the signed APC treatment")
  contract_arms = {
      str(row.get("arm")): {
          "jobset": row.get("jobset"),
          "yaml": row.get("yaml"),
          "sha256": row.get("sha256"),
      }
      for row in contract.get("arms", [])
  }
  _require(contract_arms == rows, "e0w5 RUN_CONTRACT arm binding drifted")
  return {
      "source_commit": target_source,
      "run_id": RUN_ID,
      "rounds": 3,
      "arms": rows,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--render-dir", required=True, type=Path)
  parser.add_argument("--target-source", default=TARGET_SOURCE)
  args = parser.parse_args()
  try:
    result = validate(args.render_dir, args.target_source)
  except (OSError, ValueError, KeyError, json.JSONDecodeError,
          yaml.YAMLError, RenderContractError) as exc:
    raise SystemExit(f"M15_E0W5_RENDER_RED {exc}") from exc
  print(
      "M15_E0W5_RENDER_PASS "
      f"source={result['source_commit']} run_id={result['run_id']} "
      f"rounds={result['rounds']} arms=2"
  )


if __name__ == "__main__":
  main()
