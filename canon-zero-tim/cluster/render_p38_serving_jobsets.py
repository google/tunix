#!/usr/bin/env python3
"""Render bounded FrozenLake serving-capture and KV-unified JobSets."""

from __future__ import annotations

import argparse
import dataclasses
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
_CAPSULE_MAX_ROWS = 256
_MIN_ACTION_KV = 1686
_DIAGNOSTIC_ROUNDS = 3
_LIVE_SNAPSHOT_INTERVAL_SECONDS = 30
_INCIDENT_MIN_PREFIX = 1400
_INCIDENT_MAX_PREFIX = 3072
_INCIDENT_MAX_BYTES = 128 * 1024 * 1024
_KV_OBSERVER_CANDIDATES = 3
_KV_OBSERVER_PAGES = 16
_KV_OBSERVER_MAX_BYTES = 128 * 1024 * 1024
_KV_OBSERVER_MAX_READ_BYTES = 640 * 1024 * 1024
_SEAM_MIN_POSITION = 1400
_SEAM_MAX_POSITION = 3072
_SEAM_MAX_BYTES = 4 * 1024 * 1024 * 1024
_SEAM_LAYER_COUNT = 36
_TAIL_MAX_BYTES = 256 * 1024 * 1024
_TERMINAL_MAX_BYTES = 4 * 1024 * 1024 * 1024
_LM_HEAD_ALGO_PRESET = "BF16_BF16_F32"
_ADMITTED_MAX_CONCURRENCY = (32, 256)
_ARTIFACT_BUCKET = "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38"


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


def _capture_values(
    document: Mapping[str, Any], *, unified: bool,
    seam_mode: str = "", seam_layer: int | None = None,
    terminal_tail: bool = False, terminal_discriminator: bool = False,
    lm_head_algo: bool = False,
) -> dict[str, str]:
  env = p33._env_values(document)
  state = env["CANON_STATE"]
  jobset = document["metadata"]["name"]
  values = {
      "CANON_KV_UNIFIED": "1" if unified else "0",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P38_DIAGNOSTIC_ROUNDS": str(_DIAGNOSTIC_ROUNDS),
      "CANON_P38_DIAGNOSTIC_ROUND_FILE": f"{state}/p38_diagnostic_round",
      "CANON_P38_ROUND_SEAL_REQUEST_DIR": f"{state}/p38_round_seal_requests",
      "CANON_P38_ROUND_SEAL_ACK_DIR": f"{state}/p38_round_seal_acks",
      "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": str(_CAPSULE_MAX_ROWS),
      "CANON_P38_MIN_ACTION_KV": str(_MIN_ACTION_KV),
      "CANON_P38_SERVING_CAPTURE_DIR": f"{state}/p38_serving_capture",
      "CANON_P38_REQUEST_JOURNAL": (
          f"{state}/p38_serving_capture/p38_request_journal.jsonl"
      ),
      "CANON_P38_INCIDENT_LEDGER": (
          f"{state}/p38_serving_capture/p38_incident_ledger.jsonl"
      ),
      "CANON_P38_INCIDENT_MIN_PREFIX": str(_INCIDENT_MIN_PREFIX),
      "CANON_P38_INCIDENT_MAX_PREFIX": str(_INCIDENT_MAX_PREFIX),
      "CANON_P38_INCIDENT_MAX_BYTES": str(_INCIDENT_MAX_BYTES),
      "CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS": str(
          _LIVE_SNAPSHOT_INTERVAL_SECONDS
      ),
      "CANON_P38_LIVE_SNAPSHOT_STOP_FILE": f"{state}/p38_live.stop",
      "CANON_P38_LIVE_SNAPSHOT_WORKER_LOG": f"{state}/p38_live_worker.log",
      "CANON_P38_LIVE_COLLECT_REQUEST_FILE": f"{state}/p38_collect.request",
      "CANON_P38_LIVE_COLLECT_ACK_FILE": f"{state}/p38_collect.ack",
      "CANON_P38_LIVE_COMPLETE_REQUEST_FILE": f"{state}/p38_complete.request",
      "CANON_P38_LIVE_COMPLETE_ACK_FILE": f"{state}/p38_complete.ack",
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
      "CANON_P38_GCS_PREFIX": f"{_ARTIFACT_BUCKET}/{jobset}/attempt-0",
  }
  # P38.2n is a single stock discriminator.  The already-falsified U arm is
  # still renderable as historical infrastructure, but it must not silently
  # masquerade as the live-vs-clean KV experiment.
  if seam_mode:
    if unified:
      raise ValueError("P38 seam observer is admitted only on the stock arm")
    if seam_mode not in ("layer", "full"):
      raise ValueError(f"P38 seam mode is invalid: {seam_mode}")
    if (seam_mode == "full") != (seam_layer is not None):
      raise ValueError("P38 full seam mode requires exactly one layer")
    if seam_layer is not None and not 0 <= seam_layer < _SEAM_LAYER_COUNT:
      raise ValueError(
          f"P38 seam layer is outside Qwen3-8B: {seam_layer}"
      )
    values.update({
        "CANON_P38_SEAM_OBSERVER": seam_mode,
        "CANON_P38_SEAM_OBSERVER_DIR": f"{state}/p38_serving_capture",
        "CANON_P38_SEAM_MIN_POSITION": str(_SEAM_MIN_POSITION),
        "CANON_P38_SEAM_MAX_POSITION": str(_SEAM_MAX_POSITION),
        "CANON_P38_SEAM_MAX_BYTES": str(_SEAM_MAX_BYTES),
        "CANON_P38_SEAM_CLASSIFICATION": (
            f"{state}/p38_seam.classification.json"
        ),
    })
    if seam_layer is not None:
      values["CANON_P38_SEAM_LAYER"] = str(seam_layer)
    if terminal_tail:
      if seam_mode != "layer":
        raise ValueError("P38 terminal tail requires layer seam mode")
      values.update({
          "CANON_P38_TAIL_OBSERVER": "1",
          "CANON_P38_TAIL_MAX_BYTES": str(_TAIL_MAX_BYTES),
      })
  elif terminal_tail:
    raise ValueError("P38 terminal tail requires seam observation")
  elif not unified:
    values.update({
        "CANON_P38_KV_OBSERVER_DIR": (
            f"{state}/p38_serving_capture"
        ),
        "CANON_P38_KV_OBSERVER_MAX_CANDIDATES": str(
            _KV_OBSERVER_CANDIDATES
        ),
        "CANON_P38_KV_OBSERVER_MAX_PAGES": str(_KV_OBSERVER_PAGES),
        "CANON_P38_KV_OBSERVER_MAX_BYTES": str(
            _KV_OBSERVER_MAX_BYTES
        ),
        "CANON_P38_KV_OBSERVER_MAX_READ_BYTES": str(
            _KV_OBSERVER_MAX_READ_BYTES
        ),
        "CANON_P38_KV_OBSERVER_CLASSIFICATION": (
            f"{state}/p38_kv_observer.classification.json"
        ),
    })
  if terminal_discriminator:
    if not terminal_tail:
      raise ValueError(
          "P38 terminal discriminator requires terminal-tail observation")
    values.update({
        "CANON_P38_TERMINAL_DISCRIMINATOR": "1",
        "CANON_P38_TERMINAL_MAX_BYTES": str(_TERMINAL_MAX_BYTES),
        "CANON_P38_TERMINAL_CLASSIFICATION": (
            f"{state}/p38_terminal.classification.json"
        ),
    })
  if lm_head_algo:
    if unified:
      raise ValueError("P38 lm-head algorithm arm is admitted only on stock")
    values.update({
        "CANON_MM_ALGO": "1",
        "CANON_MM_ALGO_PRESET": _LM_HEAD_ALGO_PRESET,
    })
  return values


def validate_capture_jobset(
    document: Mapping[str, Any], *, unified: bool, max_concurrency: int = 256,
    seam_mode: str = "", seam_layer: int | None = None,
    terminal_tail: bool = False, terminal_discriminator: bool = False,
    lm_head_algo: bool = False,
) -> None:
  env = p33._env_values(document)
  expected = _capture_values(
      document, unified=unified, seam_mode=seam_mode,
      seam_layer=seam_layer, terminal_tail=terminal_tail,
      terminal_discriminator=terminal_discriminator,
      lm_head_algo=lm_head_algo,
  )
  expected_gcs = (
      f"{_ARTIFACT_BUCKET}/{document['metadata']['name']}/attempt-0"
  )
  if env.get("CANON_P38_GCS_PREFIX") != expected_gcs:
    raise ValueError("P38 GCS evidence prefix drifted")
  wrong = {name: env.get(name) for name, value in expected.items() if env.get(name) != value}
  if wrong:
    raise ValueError(f"P38 serving-capture environment drifted: {wrong}")
  if not lm_head_algo and (
      "CANON_MM_ALGO" in env or "CANON_MM_ALGO_PRESET" in env
  ):
    raise ValueError("P38 stock baseline must leave the matmul algorithm unset")
  if lm_head_algo and env.get("CANON_PROFILE_FILE") != (
      "cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env"
  ):
    raise ValueError(
        "P38 lm-head algorithm isolation requires the canonical Qwen3-8B profile"
    )
  if not env.get("CANON_P38_MISMATCH_CAPSULE", "").endswith(".npz"):
    raise ValueError("P38 serving capture requires a mismatch capsule path")
  capture_dir = env["CANON_P38_SERVING_CAPTURE_DIR"].rstrip("/")
  if env["CANON_P38_REQUEST_JOURNAL"] != (
      f"{capture_dir}/p38_request_journal.jsonl"
  ):
    raise ValueError("P38 request journal must live in the capture directory")
  if env["CANON_P38_INCIDENT_LEDGER"] != (
      f"{capture_dir}/p38_incident_ledger.jsonl"
  ):
    raise ValueError("P38 incident ledger must live in the capture directory")
  if seam_mode:
    if env["CANON_P38_SEAM_OBSERVER_DIR"].rstrip("/") != capture_dir:
      raise ValueError("P38 seam observer must live in the capture directory")
    if "CANON_P38_KV_OBSERVER_DIR" in env:
      raise ValueError("P38 seam and KV observers may not share one target run")
    if not env["CANON_P38_SEAM_CLASSIFICATION"].endswith(
        "p38_seam.classification.json"
    ):
      raise ValueError("P38 seam classification path drifted")
    if (env.get("CANON_P38_SEAM_LAYER") is not None) != (
        seam_mode == "full"
    ):
      raise ValueError("P38 seam layer contract drifted")
    if terminal_tail and (
        env.get("CANON_P38_TAIL_OBSERVER") != "1"
        or env.get("CANON_P38_TAIL_MAX_BYTES") != str(_TAIL_MAX_BYTES)
    ):
      raise ValueError("P38 terminal-tail contract drifted")
    if terminal_discriminator and (
        env.get("CANON_P38_TERMINAL_DISCRIMINATOR") != "1"
        or env.get("CANON_P38_TERMINAL_MAX_BYTES")
        != str(_TERMINAL_MAX_BYTES)
        or not env.get("CANON_P38_TERMINAL_CLASSIFICATION", "").endswith(
            "p38_terminal.classification.json")
    ):
      raise ValueError("P38 terminal-discriminator contract drifted")
  elif not unified:
    if env["CANON_P38_KV_OBSERVER_DIR"].rstrip("/") != capture_dir:
      raise ValueError("P38 KV observer must live in the capture directory")
    if not env["CANON_P38_KV_OBSERVER_CLASSIFICATION"].endswith(
        "p38_kv_observer.classification.json"
    ):
      raise ValueError("P38 KV observer classification path drifted")
  if env["CANON_P38_LIVE_SNAPSHOT_STOP_FILE"] != (
      f"{env['CANON_STATE']}/p38_live.stop"
  ):
    raise ValueError("P38 live snapshot stop path drifted")
  if env["CANON_P38_LIVE_SNAPSHOT_WORKER_LOG"] != (
      f"{env['CANON_STATE']}/p38_live_worker.log"
  ):
    raise ValueError("P38 live snapshot worker log path drifted")
  expected_live_control_paths = {
      "CANON_P38_LIVE_COLLECT_REQUEST_FILE": "p38_collect.request",
      "CANON_P38_LIVE_COLLECT_ACK_FILE": "p38_collect.ack",
      "CANON_P38_LIVE_COMPLETE_REQUEST_FILE": "p38_complete.request",
      "CANON_P38_LIVE_COMPLETE_ACK_FILE": "p38_complete.ack",
  }
  for key, basename in expected_live_control_paths.items():
    if env[key] != f"{env['CANON_STATE']}/{basename}":
      raise ValueError(f"P38 live worker control path drifted: {key}")
  if env["CANON_P38_DIAGNOSTIC_ROUND_FILE"] != (
      f"{env['CANON_STATE']}/p38_diagnostic_round"
  ):
    raise ValueError("P38 diagnostic round path drifted")
  if env["CANON_P38_ROUND_SEAL_REQUEST_DIR"] != (
      f"{env['CANON_STATE']}/p38_round_seal_requests"
  ):
    raise ValueError("P38 round-seal request directory drifted")
  if env["CANON_P38_ROUND_SEAL_ACK_DIR"] != (
      f"{env['CANON_STATE']}/p38_round_seal_acks"
  ):
    raise ValueError("P38 round-seal acknowledgement directory drifted")
  labels = document["metadata"].get("labels", {})
  if labels.get("canon.zero-tim/diagnostic") != "p38-serving-capture":
    raise ValueError("P38 serving-capture label is missing")
  if labels.get("canon.zero-tim/kv-unified") != ("1" if unified else "0"):
    raise ValueError("P38 KV-unified label drifted")
  if labels.get("canon.zero-tim/seam-observer") != (seam_mode or "0"):
    raise ValueError("P38 seam-observer label drifted")
  if labels.get("canon.zero-tim/terminal-tail") != (
      "1" if terminal_tail else "0"
  ):
    raise ValueError("P38 terminal-tail label drifted")
  if labels.get("canon.zero-tim/terminal-discriminator") != (
      "1" if terminal_discriminator else "0"
  ):
    raise ValueError("P38 terminal-discriminator label drifted")
  if labels.get("canon.zero-tim/lm-head-algo") != (
      "1" if lm_head_algo else "0"
  ):
    raise ValueError("P38 lm-head-algo label drifted")
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
  observed_max_concurrency = integer_argument("max_concurrency")
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
  if max_concurrency not in _ADMITTED_MAX_CONCURRENCY:
    raise ValueError(
        f"P38 max concurrency is not admitted: {max_concurrency}"
    )
  if observed_max_concurrency != max_concurrency:
    raise ValueError(
        "P38 command max concurrency drifted: "
        f"{observed_max_concurrency} != {max_concurrency}"
    )
  labels = document["metadata"].get("labels", {})
  if labels.get("canon.zero-tim/max-concurrency") != str(max_concurrency):
    raise ValueError("P38 max-concurrency label drifted")


def render_jobset(
    base: Mapping[str, Any], spec: Any, source_commit: str, run_id: str,
    *, unified: bool, max_concurrency: int = 256,
    seam_mode: str = "", seam_layer: int | None = None,
    terminal_tail: bool = False, terminal_discriminator: bool = False,
    lm_head_algo: bool = False,
) -> dict[str, Any]:
  command = list(spec.command)
  target = "--max_concurrency=256"
  replacements = [index for index, value in enumerate(command) if value == target]
  if len(replacements) != 1:
    raise ValueError(
        "P38 base command must contain exactly one --max_concurrency=256"
    )
  command[replacements[0]] = f"--max_concurrency={max_concurrency}"
  effective_spec = dataclasses.replace(spec, command=tuple(command))
  document = p33.render_jobset(base, effective_spec, source_commit, run_id)
  main = _main_container(document)
  p33._set_named_env(
      main["env"], _capture_values(
          document, unified=unified, seam_mode=seam_mode,
          seam_layer=seam_layer, terminal_tail=terminal_tail,
          terminal_discriminator=terminal_discriminator,
          lm_head_algo=lm_head_algo,
      ), remove=()
  )
  labels = document["metadata"].setdefault("labels", {})
  labels["canon.zero-tim/diagnostic"] = "p38-serving-capture"
  labels["canon.zero-tim/kv-unified"] = "1" if unified else "0"
  labels["canon.zero-tim/max-concurrency"] = str(max_concurrency)
  labels["canon.zero-tim/seam-observer"] = seam_mode or "0"
  labels["canon.zero-tim/terminal-tail"] = "1" if terminal_tail else "0"
  labels["canon.zero-tim/terminal-discriminator"] = (
      "1" if terminal_discriminator else "0"
  )
  labels["canon.zero-tim/lm-head-algo"] = "1" if lm_head_algo else "0"
  p33.validate_jobset(document, effective_spec, source_commit, run_id)
  validate_capture_jobset(
      document, unified=unified, max_concurrency=max_concurrency,
      seam_mode=seam_mode, seam_layer=seam_layer,
      terminal_tail=terminal_tail,
      terminal_discriminator=terminal_discriminator,
      lm_head_algo=lm_head_algo,
  )
  return document


def render_all(
    *, base_path: Path, output_dir: Path, source_commit: str, run_id: str,
    stock_only: bool = False, max_concurrency: int = 256,
    seam_mode: str = "", seam_layer: int | None = None,
    terminal_tail: bool = False, terminal_discriminator: bool = False,
    lm_head_algo: bool = False,
) -> tuple[Path, ...]:
  base = p33.load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  if seam_mode and not stock_only:
    raise ValueError("P38 seam observer requires --stock-only")
  if seam_mode and max_concurrency != 256:
    raise ValueError("P38 seam observer requires max concurrency 256")
  if seam_layer is not None and seam_mode != "full":
    raise ValueError("P38 seam layer requires --seam-mode=full")
  if terminal_tail and seam_mode != "layer":
    raise ValueError("P38 terminal tail requires --seam-mode=layer")
  if terminal_discriminator and not terminal_tail:
    raise ValueError(
        "P38 terminal discriminator requires --terminal-tail")
  if lm_head_algo and not stock_only:
    raise ValueError("P38 lm-head algorithm arm requires --stock-only")
  if lm_head_algo and (
      max_concurrency != 256 or seam_mode or terminal_tail
      or terminal_discriminator
  ):
    raise ValueError(
        "P38 lm-head algorithm arm requires the slim stock concurrency-256 "
        "contract without seam/terminal observers"
    )
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
        base, spec, source_commit, run_id, unified=unified,
        max_concurrency=max_concurrency,
        seam_mode=seam_mode,
        seam_layer=seam_layer,
        terminal_tail=terminal_tail,
        terminal_discriminator=terminal_discriminator,
        lm_head_algo=lm_head_algo,
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
        f" max_concurrency={max_concurrency}"
        f" seam_mode={seam_mode or 'off'}"
        f" terminal_tail={int(terminal_tail)}"
        f" terminal_discriminator={int(terminal_discriminator)}"
        f" lm_head_algo={int(lm_head_algo)}"
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
      "--seam-mode", choices=("layer", "full"), default="",
      help="default-off hierarchical seam observer; requires --stock-only",
  )
  parser.add_argument(
      "--seam-layer", type=int,
      help="global layer index required only by --seam-mode=full",
  )
  parser.add_argument(
      "--terminal-tail", action="store_true",
      help="capture the bounded terminal tail; requires --seam-mode=layer",
  )
  parser.add_argument(
      "--terminal-discriminator", action="store_true",
      help="capture exact hidden rows and bounded logit-block signatures",
  )
  parser.add_argument(
      "--lm-head-algo", action="store_true",
      help=(
          "enable the preregistered BF16_BF16_F32 lm-head discriminator; "
          "requires the slim stock-only concurrency-256 contract"
      ),
  )
  parser.add_argument(
      "--max-concurrency",
      type=int,
      choices=_ADMITTED_MAX_CONCURRENCY,
      default=256,
      help="rollout concurrency; 32 is the preregistered P38s12b arm",
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
      max_concurrency=args.max_concurrency,
      seam_mode=args.seam_mode,
      seam_layer=args.seam_layer,
      terminal_tail=args.terminal_tail,
      terminal_discriminator=args.terminal_discriminator,
      lm_head_algo=args.lm_head_algo,
  )
  print(
      "[P38.SERVING.JOBSET] VERDICT PASS "
      f"count={len(outputs)} source={args.source_commit} run_id={args.run_id}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
