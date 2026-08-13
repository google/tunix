#!/usr/bin/env python3
"""Render bounded FrozenLake serving-capture and KV-unified JobSets."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping

import yaml


_P33_PATH = Path(__file__).with_name("render_p33_jobsets.py")
_P33_SPEC = importlib.util.spec_from_file_location("render_p33_jobsets", _P33_PATH)
assert _P33_SPEC and _P33_SPEC.loader
p33 = importlib.util.module_from_spec(_P33_SPEC)
sys.modules[_P33_SPEC.name] = p33
_P33_SPEC.loader.exec_module(p33)

_CAPTURE_PREFIX_BOUNDS = (1536, 1664, 1792, 1920, 2048)
_CAPTURE_RECORDS = len(_CAPTURE_PREFIX_BOUNDS) - 1
_DIAGNOSTIC_PROMPTS = 4
_NUM_GENERATIONS = 8
_ENGINE_DATA_SIZE = 16
_DIAGNOSTIC_UNITS = 8
_COVERED_PROMPTS = _DIAGNOSTIC_PROMPTS * _DIAGNOSTIC_UNITS


def _spec(*, unified: bool) -> Any:
  suffix = "unified" if unified else "stock"
  return p33.JobSpec(
      key=f"p38-serving-{suffix}",
      workload="frozenlake",
      stage="backward-no-commit",
      profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
      no_commit=True,
      job_prefix=f"canon-p38-fl-{suffix}",
      # Each four-prompt producer unit gives 32 trajectories and is divisible
      # by DP16. The P38 consumer waits for all eight units before alignment,
      # so the diagnostic covers the complete 32-prompt / 256-trajectory input
      # batch without admitting a non-divisible partial tail. Production/full-
      # training geometry is unchanged.
      command=p33._frozenlake_command(
          1, mini_batch_size=_DIAGNOSTIC_PROMPTS
      ),
  )


_SPECS = ((_spec(unified=False), False), (_spec(unified=True), True))


def _main_container(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(p33._head_pod(document)["containers"], "jax-tpu")


def _capture_values(document: Mapping[str, Any], *, unified: bool) -> dict[str, str]:
  env = p33._env_values(document)
  state = env["CANON_STATE"]
  return {
      "CANON_KV_UNIFIED": "1" if unified else "0",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "8",
      "CANON_P38_SERVING_CAPTURE_DIR": f"{state}/p38_serving_capture",
      "CANON_P38_REQUEST_JOURNAL": (
          f"{state}/p38_serving_capture/p38_request_journal.jsonl"
      ),
      "CANON_P38_SERVING_CAPTURE_MAX_CALLS": str(_CAPTURE_RECORDS),
      "CANON_P38_SERVING_CAPTURE_MIN_PREFIX": str(
          _CAPTURE_PREFIX_BOUNDS[0]
      ),
      "CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS": ",".join(
          str(value) for value in _CAPTURE_PREFIX_BOUNDS
      ),
      "CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER": "5",
      "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
      "CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS": str(_CAPTURE_RECORDS),
      "CANON_P38_SERVING_CAPTURE_CLASSIFICATION": (
          f"{state}/p38_serving_capture.classification.json"
      ),
      "CANON_P38_SERVING_CAPTURE_ARCHIVE": f"{state}/p38_serving_capture.tar",
  }


def validate_capture_jobset(
    document: Mapping[str, Any], *, unified: bool
) -> None:
  env = p33._env_values(document)
  expected = _capture_values(document, unified=unified)
  wrong = {name: env.get(name) for name, value in expected.items() if env.get(name) != value}
  if wrong:
    raise ValueError(f"P38 serving-capture environment drifted: {wrong}")
  if not env.get("CANON_P38_MISMATCH_CAPSULE", "").endswith(".npz"):
    raise ValueError("P38 serving capture requires a mismatch capsule path")
  capture_dir = env["CANON_P38_SERVING_CAPTURE_DIR"].rstrip("/")
  if env["CANON_P38_REQUEST_JOURNAL"] != (
      f"{capture_dir}/p38_request_journal.jsonl"
  ):
    raise ValueError("P38 request journal must live in the capture directory")
  labels = document["metadata"].get("labels", {})
  if labels.get("canon.zero-tim/diagnostic") != "p38-serving-capture":
    raise ValueError("P38 serving-capture label is missing")
  if labels.get("canon.zero-tim/kv-unified") != ("1" if unified else "0"):
    raise ValueError("P38 KV-unified label drifted")
  if document["spec"]["failurePolicy"].get("maxRestarts") != 0:
    raise ValueError("P38 serving-capture JobSet must not restart")
  command = shlex.split(env.get("CANON_RUN_CMD", ""))

  def integer_argument(name: str) -> int:
    prefix = f"--{name}="
    values = [
        value.removeprefix(prefix)
        for value in command
        if value.startswith(prefix)
    ]
    if len(values) != 1:
      raise ValueError(f"P38 command requires exactly one {prefix} argument")
    try:
      return int(values[0])
    except ValueError as exc:
      raise ValueError(
          f"P38 command has a non-integer {prefix} argument"
      ) from exc

  batch_size = integer_argument("batch_size")
  mini_batch_size = integer_argument("mini_batch_size")
  num_generations = integer_argument("num_generations")
  mesh_dp = integer_argument("mesh_dp")
  trajectories = mini_batch_size * num_generations
  if batch_size != 32:
    raise ValueError(f"P38 global prompt batch changed: {batch_size} != 32")
  if batch_size != _COVERED_PROMPTS:
    raise ValueError(
        "P38 diagnostic does not cover the full prompt batch: "
        f"{batch_size} vs {_COVERED_PROMPTS}"
    )
  if (mini_batch_size, num_generations, mesh_dp) != (
      _DIAGNOSTIC_PROMPTS,
      _NUM_GENERATIONS,
      _ENGINE_DATA_SIZE,
  ):
    raise ValueError(
        "P38 diagnostic batch geometry changed: "
        f"prompts={mini_batch_size} generations={num_generations} dp={mesh_dp}"
    )
  if trajectories % mesh_dp:
    raise ValueError(
        "P38 diagnostic trajectories are not divisible by engine DP: "
        f"{trajectories} vs {mesh_dp}"
    )


def render_jobset(
    base: Mapping[str, Any], spec: Any, source_commit: str, run_id: str,
    *, unified: bool,
) -> dict[str, Any]:
  document = p33.render_jobset(base, spec, source_commit, run_id)
  main = _main_container(document)
  p33._set_named_env(
      main["env"], _capture_values(document, unified=unified), remove=()
  )
  labels = document["metadata"].setdefault("labels", {})
  labels["canon.zero-tim/diagnostic"] = "p38-serving-capture"
  labels["canon.zero-tim/kv-unified"] = "1" if unified else "0"
  p33.validate_jobset(document, spec, source_commit, run_id)
  validate_capture_jobset(document, unified=unified)
  return document


def render_all(
    *, base_path: Path, output_dir: Path, source_commit: str, run_id: str,
    stock_only: bool = False,
) -> tuple[Path, ...]:
  base = p33.load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  specs = _SPECS[:1] if stock_only else _SPECS
  outputs = tuple(
      output_dir / f"jobset-p38-serving-{'unified' if unified else 'stock'}.yaml"
      for _, unified in specs
  )
  existing = [path for path in outputs if path.exists()]
  if existing:
    raise FileExistsError(
        "refusing to overwrite rendered P38 JobSets: "
        + ", ".join(str(path) for path in existing)
    )
  for (spec, unified), path in zip(specs, outputs, strict=True):
    document = render_jobset(
        base, spec, source_commit, run_id, unified=unified
    )
    header = (
        "# Generated by canon-zero-tim/cluster/render_p38_serving_jobsets.py.\n"
        "# Do not edit this output; change the reviewed renderer instead.\n"
    )
    path.write_text(
        header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    print(
        "[P38.SERVING.JOBSET] RENDERED "
        f"arm={'unified' if unified else 'stock'} path={path}"
    )
  return outputs


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument(
      "--stock-only", action="store_true",
      help="render only the known-red stock arm; U was already falsified",
  )
  parser.add_argument(
      "--base",
      type=Path,
      default=Path(__file__).with_name("jobset-64chip.yaml"),
  )
  args = parser.parse_args()
  outputs = render_all(
      base_path=args.base,
      output_dir=args.output_dir,
      source_commit=args.source_commit,
      run_id=args.run_id,
      stock_only=args.stock_only,
  )
  print(
      "[P38.SERVING.JOBSET] VERDICT PASS "
      f"count={len(outputs)} source={args.source_commit} run_id={args.run_id}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
