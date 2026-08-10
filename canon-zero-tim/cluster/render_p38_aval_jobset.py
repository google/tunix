#!/usr/bin/env python3
"""Render one source-pinned Pathways P38 model-free aval JobSet."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


P36_RENDERER = Path(__file__).with_name("render_p36_proxy_xla_jobset.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "p38_p36_renderer", P36_RENDERER
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
  raise RuntimeError("cannot import the P36 proxy renderer")
p36 = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = p36
MODULE_SPEC.loader.exec_module(p36)
p33 = p36.p33


SCRATCH_ROOT = "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p38"


def _job_name(source_commit: str, run_id: str) -> str:
  name = f"canon-p38-aval-{run_id}-{source_commit[:8]}"
  if len(name) > 63:
    raise ValueError(f"generated JobSet name exceeds 63 characters: {name}")
  return name


def _replace_string_values(value: Any, old: str, new: str) -> Any:
  if isinstance(value, dict):
    return {
        key: _replace_string_values(item, old, new)
        for key, item in value.items()
    }
  if isinstance(value, list):
    return [_replace_string_values(item, old, new) for item in value]
  if isinstance(value, str):
    return value.replace(old, new)
  return value


def _main(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(p33._head_pod(document)["containers"], "jax-tpu")


def _proxy(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(
      p33._head_pod(document)["initContainers"], "pathways-proxy"
  )


def render(
    *, base: Mapping[str, Any], source_commit: str, run_id: str
) -> dict[str, Any]:
  """Return one no-model Attempt-0 P38 aval discriminator."""
  if not p33._SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be one lowercase 40-character SHA")
  if not p33._RUN_ID_RE.fullmatch(run_id):
    raise ValueError(
        "run id must be a 1-16 character lowercase DNS label component"
    )

  p36_document = p36.render(
      base=base, source_commit=source_commit, run_id=run_id
  )
  old_name = p36._job_name(source_commit, run_id)  # pylint: disable=protected-access
  name = _job_name(source_commit, run_id)
  document = _replace_string_values(p36_document, old_name, name)
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p38",
      "canon.zero-tim/stage": "model-free-aval",
  })
  state = f"/tmp/canon-state/{name}"
  main = _main(document)
  p33._set_named_env(
      main["env"],
      {
          "CANON_RUN_P38_AVAL": "1",
          "CANON_P38_AVAL_REPORT": f"{state}/p38_aval.result.json",
          "CANON_T1_LOG": f"{state}/p38_aval.raw.log",
      },
      remove=(),
  )
  head = p33._head_pod(document)
  for container_name in ("pathways-proxy", "pathways-rm"):
    container = p33._container(head["initContainers"], container_name)
    p33._replace_arg(
        container["args"],
        "--gcs_scratch_location=",
        f"--gcs_scratch_location={SCRATCH_ROOT}/{name}",
    )

  validate(document, source_commit=source_commit, run_id=run_id)
  return document


def validate(
    document: Mapping[str, Any], *, source_commit: str, run_id: str
) -> None:
  """Reject any target manifest that weakens the model-free contract."""
  name = _job_name(source_commit, run_id)
  if document.get("metadata", {}).get("name") != name:
    raise ValueError("generated P38 JobSet name drifted")
  if document["spec"]["failurePolicy"].get("maxRestarts") != 0:
    raise ValueError("P38 must not hide a red gate behind a restart")
  if any(
      job["template"]["spec"].get("backoffLimit") != 0
      for job in document["spec"]["replicatedJobs"]
  ):
    raise ValueError("P38 jobs must not retry a failed attempt")

  state = f"/tmp/canon-state/{name}"
  env = p33._env_values(document)
  expected = {
      "CANON_MODE": "gate-only",
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_RUN_P38_AVAL": "1",
      "CANON_P38_AVAL_REPORT": f"{state}/p38_aval.result.json",
      "CANON_T1_LOG": f"{state}/p38_aval.raw.log",
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"generated P38 environment drifted: {wrong}")
  if "CANON_RUN_CMD" in env or "WANDB_MODE" in env:
    raise ValueError("P38 model-free probe retained a workload environment")

  proxy = _proxy(document)
  proxy_flags = [
      entry
      for entry in proxy.get("env", [])
      if entry.get("name") == p36.PROXY_XLA_ENV
  ]
  if proxy_flags != [{
      "name": p36.PROXY_XLA_ENV,
      "value": p36.PROXY_XLA_FLAG,
  }]:
    raise ValueError("P38 proxy lost the canonical XLA_FLAGS environment")

  head = p33._head_pod(document)
  scratch = []
  for container_name in ("pathways-proxy", "pathways-rm"):
    container = p33._container(head["initContainers"], container_name)
    scratch.extend(
        arg
        for arg in container["args"]
        if arg.startswith("--gcs_scratch_location=")
    )
  expected_scratch = f"--gcs_scratch_location={SCRATCH_ROOT}/{name}"
  if scratch != [expected_scratch, expected_scratch]:
    raise ValueError("P38 Pathways services lost their isolated scratch")

  worker = p33._container(
      p33._worker_pod(document)["containers"], "pathways-worker"
  )
  address = f"{name}-pathways-head-0-0.{name}"
  if f"--resource_manager_address={address}:29001" not in worker["args"]:
    raise ValueError("P38 worker address drifted from the JobSet name")
  worker_env = {
      entry["name"]: entry.get("value") for entry in worker.get("env", [])
  }
  if worker_env.get("PATHWAYS_HEAD") != address:
    raise ValueError("P38 worker PATHWAYS_HEAD drifted")

  serialized = yaml.safe_dump(document, sort_keys=False)
  if p33._BRANCH not in serialized:
    raise ValueError("P38 JobSet does not fetch the canonical source branch")
  if any(marker in serialized for marker in ("wandb_v1_", "github_pat_", "ghp_")):
    raise ValueError("generated P38 JobSet contains a literal credential")


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument(
      "--base",
      type=Path,
      default=Path(__file__).with_name("jobset-64chip.yaml"),
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  document = render(
      base=p33.load_base(args.base),
      source_commit=args.source_commit,
      run_id=args.run_id,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      "# Generated by canon-zero-tim/cluster/render_p38_aval_jobset.py.\n"
      + yaml.safe_dump(document, sort_keys=False),
      encoding="utf-8",
  )
  print(f"[P38.AVAL.JOBSET] RENDERED path={args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
