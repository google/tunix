#!/usr/bin/env python3
"""Render one immutable P59 DP16 backward timing or XProf JobSet."""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import shlex

import yaml

import render_p33_jobsets as p33


_WRAPPER = (
    "bash",
    "canon-zero-tim/tasks/p59-dp16-parallel-backward/scripts/"
    "run_and_persist.sh",
)
_KINDS = ("control", "candidate", "profile")
_EVIDENCE_ROOT = (
    "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p59"
)


def _spec(kind: str) -> tuple[p33.JobSpec, tuple[str, ...]]:
  if kind not in _KINDS:
    raise ValueError(f"unsupported P59 run kind: {kind!r}")
  inner = p33._frozenlake_command(3)  # pylint: disable=protected-access
  return (
      p33.JobSpec(
          key=f"p59-backward-{kind}",
          workload="frozenlake",
          stage="three-update",
          profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
          no_commit=False,
          job_prefix=f"canon-p59-fl-bwd-{kind}",
          command=_WRAPPER,
          rank_parallel_backward=kind != "control",
      ),
      inner,
  )


def render(
    *,
    base_path: Path,
    source_commit: str,
    run_id: str,
    kind: str,
) -> dict:
  spec, inner = _spec(kind)
  document = p33.render_jobset(
      p33.load_base(base_path), spec, source_commit, run_id
  )
  name = document["metadata"]["name"]
  env = p33._env_values(document)  # pylint: disable=protected-access
  state = env["CANON_STATE"]
  profile = kind == "profile"
  main = p33._container(  # pylint: disable=protected-access
      p33._head_pod(document)["containers"], "jax-tpu"  # pylint: disable=protected-access
  )
  p33._set_named_env(  # pylint: disable=protected-access
      main["env"],
      {
          "CANON_P59_INNER_RUN_CMD": shlex.join(inner),
          "CANON_P59_GCS_PREFIX": (
              f"{_EVIDENCE_ROOT}/{name}/attempt-0"
          ),
          "CANON_P59_REQUIRE_XPROF": "1" if profile else "0",
          "CANON_XPROF_DIR": f"{state}/xprof" if profile else "",
          "CANON_XPROF_SKIP_STEPS": "1",
          "CANON_XPROF_STEPS": "1",
          "CANON_XPROF_PHASE": "update",
          "CANON_XPROF_HOST_TRACER": "1",
          "CANON_XPROF_PYTHON_TRACER": "0",
          "CANON_XPROF_LABELS": "1" if profile else "0",
      },
      remove=(),
  )
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/diagnostic": "p59-dp16-backward-ab",
      "canon.zero-tim/p59-kind": kind,
      "canon.zero-tim/p59-rank-parallel": (
          "0" if kind == "control" else "1"
      ),
  })
  validate(document, spec, source_commit, run_id, kind=kind, inner=inner)
  return document


def validate(
    document: dict,
    spec: p33.JobSpec,
    source_commit: str,
    run_id: str,
    *,
    kind: str,
    inner: tuple[str, ...],
) -> None:
  p33.validate_jobset(document, spec, source_commit, run_id)
  env = p33._env_values(document)  # pylint: disable=protected-access
  name = document["metadata"]["name"]
  expected = {
      "CANON_RUN_CMD": shlex.join(_WRAPPER),
      "CANON_P59_INNER_RUN_CMD": shlex.join(inner),
      "CANON_P59_GCS_PREFIX": (
          f"{_EVIDENCE_ROOT}/{name}/attempt-0"
      ),
      "CANON_P59_REQUIRE_XPROF": "1" if kind == "profile" else "0",
      "CANON_P59_RANK_PARALLEL_BACKWARD": (
          "0" if kind == "control" else "1"
      ),
      "CANON_XPROF_DIR": (
          f"{env['CANON_STATE']}/xprof" if kind == "profile" else ""
      ),
      "CANON_XPROF_SKIP_STEPS": "1",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_LABELS": "1" if kind == "profile" else "0",
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P59 environment drifted: {wrong}")
  if "--max_steps=3" not in env["CANON_P59_INNER_RUN_CMD"]:
    raise ValueError("P59 inner workload is not the frozen three-update recipe")
  if document["spec"]["failurePolicy"]["maxRestarts"] != 0:
    raise ValueError("P59 evidence run must not restart")
  labels = document["metadata"].get("labels", {})
  if labels.get("canon.zero-tim/p59-kind") != kind:
    raise ValueError("P59 kind label drifted")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--kind", required=True, choices=_KINDS)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--base",
      type=Path,
      default=Path(__file__).with_name("jobset-64chip.yaml"),
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite rendered JobSet: {args.output}")
  document = render(
      base_path=args.base,
      source_commit=args.source_commit,
      run_id=args.run_id,
      kind=args.kind,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      "# Generated by render_p59_backward_ab.py; do not edit.\n"
      + yaml.safe_dump(document, sort_keys=False),
      encoding="utf-8",
  )
  print(
      "P59_BACKWARD_JOBSET_RENDER_PASS "
      f"kind={args.kind} job={document['metadata']['name']} "
      f"output={args.output}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
