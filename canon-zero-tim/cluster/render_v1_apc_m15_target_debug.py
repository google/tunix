#!/usr/bin/env python3
"""Render bounded APC-off/on M15 target reproducers on DP8xTP8."""

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

_PROFILE = "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env"
_PREFIX_BOUNDS = (1152, 1216, 1280, 1408, 1696)
_INCIDENT_MIN_PREFIX = 1152
_INCIDENT_MAX_PREFIX = 7168
_INCIDENT_MAX_BYTES = 2 * 1024 * 1024 * 1024
_SEAM_MIN_POSITION = 960
_SEAM_MAX_POSITION = 4096
_SEAM_MAX_BYTES = 8 * 1024 * 1024 * 1024
_TAIL_MAX_BYTES = 256 * 1024 * 1024
_ARTIFACT_BUCKET = "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38"
_WORKLOAD_CANDIDATE = "m15"
_DATA_SPLIT = "main"


def _replace_arg(command: list[str], prefix: str, replacement: str) -> None:
  indices = [i for i, value in enumerate(command) if value.startswith(prefix)]
  if len(indices) != 1:
    raise ValueError(f"expected one {prefix} argument, found {len(indices)}")
  command[indices[0]] = replacement


def _command() -> tuple[str, ...]:
  command = list(p33._frozenlake_command(  # pylint: disable=protected-access
      1, dp_size=8, tp_size=8, mini_batch_size=32
  ))
  if tuple(command[:3]) != (
      "python3", "-u", "examples/frozenlake/train_frozenlake_qwen3.py"
  ):
    raise ValueError("FrozenLake base entrypoint drifted")
  command[:3] = (
      "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3"
  )
  _replace_arg(command, "--max_response_length=", "--max_response_length=8192")
  _replace_arg(command, "--env_max_steps=", "--env_max_steps=15")
  command.extend((
      "--sampler_is=none",
      "--seed=42",
      "--p57_workload_candidate=m15",
      "--p57_data_split=main",
      "--eval_every_n_steps=0",
  ))
  return tuple(command)


def _spec(arm: str) -> Any:
  return p33.JobSpec(
      key=f"v1-apc-m15-{arm}",
      workload="frozenlake",
      stage="backward-no-commit",
      profile=_PROFILE,
      no_commit=True,
      job_prefix=f"canon-v1-apc-m15-{arm}",
      command=_command(),
      enable_evaluation=False,
      dp_size=8,
      tp_size=8,
      optimizer_resident=True,
      rank_parallel_backward=True,
      fixed_lm_head=True,
      strict_alignment=True,
  )


def _container(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(p33._head_pod(document)["containers"], "jax-tpu")


def _replace_env(document: Mapping[str, Any], values: Mapping[str, str]) -> None:
  p33._set_named_env(_container(document)["env"], values, remove=())


def _capture_values(
    document: Mapping[str, Any],
    arm: str,
    *,
    observer: str = "none",
    seam_layer: int | None = None,
) -> dict[str, str]:
  env = p33._env_values(document)
  state = env["CANON_STATE"]
  name = document["metadata"]["name"]
  capture = f"{state}/p38_serving_capture"
  values = {
      "CANON_APC_M15_TARGET_DEBUG": arm,
      # train_frozenlake_qwen3 treats the CLI selector and these environment
      # fields as one signed workload identity.  Supplying only the CLI values
      # passes rendering/profile checks but fails before learner construction.
      "CANON_P57_WORKLOAD_CANDIDATE": _WORKLOAD_CANDIDATE,
      "CANON_P57_DATA_SPLIT": _DATA_SPLIT,
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "1" if arm == "on" else "0",
      "CANON_KV_UNIFIED": "0",
      "CANON_P38_PRECHECK_ONLY": "1",
      "CANON_P38_CONTROLLED_EXIT": "1",
      "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
      "CANON_P38_DIAGNOSTIC_ROUND_FILE": f"{state}/p38_diagnostic_round",
      "CANON_P38_ROUND_SEAL_REQUEST_DIR": f"{state}/p38_round_seal_requests",
      "CANON_P38_ROUND_SEAL_ACK_DIR": f"{state}/p38_round_seal_acks",
      "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "256",
      "CANON_P38_MIN_ACTION_KV": "1686",
      "CANON_P38_SERVING_CAPTURE_DIR": capture,
      "CANON_P38_REQUEST_JOURNAL": f"{capture}/p38_request_journal.jsonl",
      "CANON_P38_INCIDENT_LEDGER": f"{capture}/p38_incident_ledger.jsonl",
      "CANON_APC_M15_REPLAY_LEDGER": f"{capture}/m15_replay_envelope.jsonl",
      "CANON_P38_INCIDENT_MIN_PREFIX": str(_INCIDENT_MIN_PREFIX),
      "CANON_P38_INCIDENT_MAX_PREFIX": str(_INCIDENT_MAX_PREFIX),
      "CANON_P38_INCIDENT_MAX_BYTES": str(_INCIDENT_MAX_BYTES),
      "CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS": "30",
      "CANON_P38_DURABILITY_PROFILE": (
          "m15-wide-v1" if observer != "none" else "round-alignment-v1"
      ),
      "CANON_P38_LIVE_SNAPSHOT_STOP_FILE": f"{state}/p38_live.stop",
      "CANON_P38_LIVE_SNAPSHOT_WORKER_LOG": f"{state}/p38_live_worker.log",
      "CANON_P38_LIVE_COLLECT_REQUEST_FILE": f"{state}/p38_collect.request",
      "CANON_P38_LIVE_COLLECT_ACK_FILE": f"{state}/p38_collect.ack",
      "CANON_P38_LIVE_COMPLETE_REQUEST_FILE": f"{state}/p38_complete.request",
      "CANON_P38_LIVE_COMPLETE_ACK_FILE": f"{state}/p38_complete.ack",
      "CANON_P38_SERVING_CAPTURE_MAX_CALLS": "4",
      "CANON_P38_SERVING_CAPTURE_MIN_PREFIX": str(_PREFIX_BOUNDS[0]),
      "CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS": ",".join(map(str, _PREFIX_BOUNDS)),
      "CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER": "5",
      "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
      "CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS": "4",
      "CANON_P38_SERVING_CAPTURE_CLASSIFICATION": f"{state}/p38_serving_capture.classification.json",
      "CANON_P38_SERVING_CAPTURE_ARCHIVE": f"{state}/p38_serving_capture.tar",
      "CANON_P38_GCS_PREFIX": f"{_ARTIFACT_BUCKET}/{name}/attempt-0",
  }
  if observer != "none":
    values.update({
        "CANON_P38_SEAM_OBSERVER": observer,
        "CANON_P38_SEAM_OBSERVER_DIR": capture,
        "CANON_P38_SEAM_MIN_POSITION": str(_SEAM_MIN_POSITION),
        "CANON_P38_SEAM_MAX_POSITION": str(_SEAM_MAX_POSITION),
        "CANON_P38_SEAM_MAX_BYTES": str(_SEAM_MAX_BYTES),
        "CANON_P38_SEAM_CLASSIFICATION": f"{state}/p38_seam.classification.json",
        "CANON_APC_M15_SEAM_BUNDLE": f"{state}/m15_wide_seam_bundle.tar",
    })
    if observer == "layer":
      values.update({
          "CANON_P38_TAIL_OBSERVER": "1",
          "CANON_P38_TAIL_MAX_BYTES": str(_TAIL_MAX_BYTES),
      })
    elif observer == "full":
      if seam_layer is None:
        raise ValueError("full M15 seam observer requires --seam-layer")
      values["CANON_P38_SEAM_LAYER"] = str(seam_layer)
  return values


def validate(
    document: Mapping[str, Any],
    *,
    arm: str,
    source_commit: str,
    run_id: str,
    observer: str = "none",
    seam_layer: int | None = None,
) -> None:
  spec = _spec(arm)
  p33.validate_jobset(
      document, spec, source_commit, run_id,
      fixed_lm_head=True, alignment_warning_only=False,
  )
  env = p33._env_values(document)
  expected = _capture_values(
      document, arm, observer=observer, seam_layer=seam_layer
  )
  wrong = {name: env.get(name) for name, value in expected.items() if env.get(name) != value}
  if wrong:
    raise ValueError(f"M15 APC capture env drifted: {wrong}")
  if document["spec"]["failurePolicy"].get("maxRestarts") != 0:
    raise ValueError("M15 APC target debug must not restart")
  if any(name.startswith("CANON_" "P38_KV_OBSERVER") for name in env):
    raise ValueError("M15 wide seam runs must not attach the KV observer")
  if observer == "none":
    if any(
        name.startswith(("CANON_" "P38_SEAM", "CANON_" "P38_TAIL"))
        for name in env
    ):
      raise ValueError("observer=none must not attach a numerical observer")
  elif observer == "layer":
    if env.get("CANON_P38_SEAM_OBSERVER") != "layer" or \
       env.get("CANON_P38_TAIL_OBSERVER") != "1" or \
       "CANON_P38_SEAM_LAYER" in env:
      raise ValueError("M15 layer observer contract drifted")
  elif observer == "full":
    if seam_layer is None or not 0 <= seam_layer < 36:
      raise ValueError("M15 full seam layer must be in [0,36)")
    if env.get("CANON_P38_SEAM_OBSERVER") != "full" or \
       env.get("CANON_P38_SEAM_LAYER") != str(seam_layer) or \
       "CANON_P38_TAIL_OBSERVER" in env:
      raise ValueError("M15 full observer contract drifted")
  else:
    raise ValueError("observer must be none, layer, or full")
  command = shlex.split(env["CANON_RUN_CMD"])
  if tuple(command[:4]) != (
      "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3"
  ):
    raise ValueError("M15 APC command must use the package-safe module entrypoint")
  required = {
      "--mesh_dp=8", "--mesh_tp=8", "--batch_size=32",
      "--mini_batch_size=32", "--num_generations=8",
      "--max_prompt_length=4096", "--max_response_length=8192",
      "--max_concurrency=256", "--vllm_max_num_seqs=32",
      "--vllm_max_num_batched_tokens=256", "--env_max_steps=15",
      "--temperature=0.7", "--top_k=0", "--top_p=1.0", "--seed=42",
      f"--p57_workload_candidate={_WORKLOAD_CANDIDATE}",
      f"--p57_data_split={_DATA_SPLIT}",
      "--sampler_is=none", "--eval_every_n_steps=0",
  }
  missing = sorted(required - set(command))
  if missing:
    raise ValueError(f"M15 APC command drifted: {missing}")


def render_all(
    *,
    base_path: Path,
    output_dir: Path,
    source_commit: str,
    run_id: str,
    observer: str = "none",
    seam_layer: int | None = None,
) -> tuple[Path, ...]:
  if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
    raise ValueError("source commit must be a full lowercase SHA")
  if observer not in ("none", "layer", "full"):
    raise ValueError("observer must be none, layer, or full")
  if observer == "full" and (seam_layer is None or not 0 <= seam_layer < 36):
    raise ValueError("full M15 seam observer requires --seam-layer in [0,36)")
  if observer != "full" and seam_layer is not None:
    raise ValueError("--seam-layer is valid only with --observer=full")
  base = p33.load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  paths = []
  for arm in ("off", "on"):
    spec = _spec(arm)
    document = p33.render_jobset(base, spec, source_commit, run_id)
    _replace_env(
        document,
        _capture_values(
            document, arm, observer=observer, seam_layer=seam_layer
        ),
    )
    labels = document["metadata"].setdefault("labels", {})
    labels.update({
        # P33 keys the 256-row mismatch-capsule admission on this existing
        # diagnostic identity.  Observer mode is carried separately below.
        "canon.zero-tim/diagnostic": "p38-serving-capture",
        "canon.zero-tim/apc-m15-arm": arm,
        "canon.zero-tim/kv-unified": "0",
        "canon.zero-tim/seam-observer": observer,
        "canon.zero-tim/terminal-tail": "1" if observer == "layer" else "0",
        "canon.zero-tim/terminal-discriminator": "0",
        "canon.zero-tim/lm-head-algo": "0",
        "canon.zero-tim/fixed-lm-head": "1",
        "canon.zero-tim/durability-profile": (
            "m15-wide-v1" if observer != "none" else "round-alignment-v1"
        ),
    })
    validate(
        document,
        arm=arm,
        source_commit=source_commit,
        run_id=run_id,
        observer=observer,
        seam_layer=seam_layer,
    )
    suffix = "" if observer == "none" else f"-{observer}"
    path = output_dir / f"jobset-v1-apc-m15-{arm}{suffix}.yaml"
    if path.exists():
      raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(
        "# Generated by render_v1_apc_m15_target_debug.py; do not edit.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    paths.append(path)
  return tuple(paths)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, default=Path(__file__).with_name("jobset-64chip.yaml"))
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument(
      "--observer", choices=("none", "layer", "full"), default="none"
  )
  parser.add_argument("--seam-layer", type=int)
  args = parser.parse_args()
  for path in render_all(
      base_path=args.base, output_dir=args.output_dir,
      source_commit=args.source_commit, run_id=args.run_id,
      observer=args.observer, seam_layer=args.seam_layer,
  ):
    print(f"[V1.APC.M15] RENDERED path={path}")


if __name__ == "__main__":
  main()
