#!/usr/bin/env python3
"""Render one strict Pathways proxy-XLA gate-only JobSet."""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


P33_RENDERER = Path(__file__).with_name("render_p33_jobsets.py")
MODULE_SPEC = importlib.util.spec_from_file_location(
    "p36_p33_renderer", P33_RENDERER
)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
  raise RuntimeError("cannot import the P33 JobSet renderer")
p33 = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = p33
MODULE_SPEC.loader.exec_module(p33)


PROXY_XLA_FLAG = "--xla_allow_excess_precision=false"
PROXY_XLA_PREFIX = "--xla_allow_excess_precision="
PROXY_IMAGE = (
    "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/"
    "proxy_server@sha256:7bdf61492b723c970b597812a90335a87358d279f92770d3ca11fc86ad15e312"
)
SCRATCH_ROOT = "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p36"
PROFILE = "cluster/profiles/qwen3-8b-dp16-tp4-rc.env"


def _job_name(source_commit: str, run_id: str) -> str:
  name = f"canon-p36-proxy-xla-{run_id}-{source_commit[:8]}"
  if len(name) > 63:
    raise ValueError(f"generated JobSet name exceeds 63 characters: {name}")
  return name


def _main(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(p33._head_pod(document)["containers"], "jax-tpu")


def _proxy(document: Mapping[str, Any]) -> dict[str, Any]:
  return p33._container(
      p33._head_pod(document)["initContainers"], "pathways-proxy"
  )


def _ensure_proxy_flag(args: list[str]) -> None:
  matches = [arg for arg in args if arg.startswith(PROXY_XLA_PREFIX)]
  if not matches:
    args.append(PROXY_XLA_FLAG)
    return
  if matches != [PROXY_XLA_FLAG]:
    raise ValueError(
        "base Pathways proxy has a conflicting or duplicate excess-precision flag"
    )


def render(
    *, base: Mapping[str, Any], source_commit: str, run_id: str
) -> dict[str, Any]:
  """Return one source-pinned attempt-zero flag-on topology probe."""
  if not p33._SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be one lowercase 40-character SHA")
  if not p33._RUN_ID_RE.fullmatch(run_id):
    raise ValueError(
        "run id must be a 1-16 character lowercase DNS label component"
    )

  document = copy.deepcopy(base)
  name = _job_name(source_commit, run_id)
  state = f"/tmp/canon-state/{name}"
  scratch = f"{SCRATCH_ROOT}/{name}"
  document["metadata"]["name"] = name
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/phase": "p36",
      "canon.zero-tim/stage": "proxy-xla-gate-only",
      "canon.zero-tim/source": source_commit[:8],
  })
  document["spec"]["failurePolicy"]["maxRestarts"] = 0

  head_job = document["spec"]["replicatedJobs"][0]["template"]["spec"]
  worker_job = document["spec"]["replicatedJobs"][1]["template"]["spec"]
  head_job["backoffLimit"] = 0
  worker_job["backoffLimit"] = 0

  head = p33._head_pod(document)
  proxy = p33._container(head["initContainers"], "pathways-proxy")
  manager = p33._container(head["initContainers"], "pathways-rm")
  _ensure_proxy_flag(proxy["args"])
  p33._replace_arg(
      proxy["args"],
      "--gcs_scratch_location=",
      f"--gcs_scratch_location={scratch}",
  )
  p33._replace_arg(
      manager["args"],
      "--gcs_scratch_location=",
      f"--gcs_scratch_location={scratch}",
  )

  main = _main(document)
  p33._set_named_env(
      main["env"],
      {
          "CANON_MODE": "gate-only",
          "CANON_PROFILE_FILE": PROFILE,
          "CANON_STATE": state,
          "CANON_EXPECT_COMMIT": source_commit,
          "CANON_P32_EXPECT_MODEL_MESH_IDS": "",
          "CANON_EXPECT_TRAIN_MESH_IDS": "",
          "CANON_REQUIRE_TRAIN_MESH_PIN": "0",
          "CANON_WAYCOUNT_WIDTHS": "2,4,8",
          "CANON_WAYCOUNT_DEPTHS": "8,15",
          "CANON_T1_LOG": f"{state}/p36_waycount.raw.log",
          "JAX_COMPILATION_CACHE_DIR": f"{state}/jax_compilation_cache",
          "CANON_GCS_CACHE_BUCKET": "",
      },
      remove=("CANON_P32_RC_STAGE",),
  )

  worker = p33._container(
      p33._worker_pod(document)["containers"], "pathways-worker"
  )
  address = f"{name}-pathways-head-0-0.{name}"
  p33._replace_arg(
      worker["args"],
      "--resource_manager_address=",
      f"--resource_manager_address={address}:29001",
  )
  worker_env = {entry["name"]: entry for entry in worker["env"]}
  if "PATHWAYS_HEAD" not in worker_env:
    raise ValueError("base JobSet worker lost PATHWAYS_HEAD")
  worker_env["PATHWAYS_HEAD"].clear()
  worker_env["PATHWAYS_HEAD"].update({
      "name": "PATHWAYS_HEAD",
      "value": address,
  })

  validate(document, source_commit=source_commit, run_id=run_id)
  return document


def validate(
    document: Mapping[str, Any], *, source_commit: str, run_id: str
) -> None:
  """Reject a manifest that changes more than the registered P36 variable."""
  name = _job_name(source_commit, run_id)
  if document.get("apiVersion") != "jobset.x-k8s.io/v1alpha2":
    raise ValueError("P36 requires the reviewed JobSet API version")
  if document.get("metadata", {}).get("name") != name:
    raise ValueError("generated P36 JobSet name drifted")
  if document["spec"]["failurePolicy"].get("maxRestarts") != 0:
    raise ValueError("P36 must not hide a red gate behind a restart")
  for job in document["spec"]["replicatedJobs"]:
    if job["template"]["spec"].get("backoffLimit") != 0:
      raise ValueError("P36 jobs must not retry a failed attempt")

  env = p33._env_values(document)
  expected = {
      "CANON_MODE": "gate-only",
      "CANON_PROFILE_FILE": PROFILE,
      "CANON_STATE": f"/tmp/canon-state/{name}",
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P32_EXPECT_MODEL_MESH_IDS": "",
      "CANON_EXPECT_TRAIN_MESH_IDS": "",
      "CANON_REQUIRE_TRAIN_MESH_PIN": "0",
      "CANON_WAYCOUNT_WIDTHS": "2,4,8",
      "CANON_WAYCOUNT_DEPTHS": "8,15",
      "CANON_T1_LOG": f"/tmp/canon-state/{name}/p36_waycount.raw.log",
      "JAX_COMPILATION_CACHE_DIR": (
          f"/tmp/canon-state/{name}/jax_compilation_cache"
      ),
      "CANON_GCS_CACHE_BUCKET": "",
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"generated P36 environment drifted: {wrong}")
  if "CANON_P32_RC_STAGE" in env:
    raise ValueError("P36 retained a model release-candidate stage")

  proxy = _proxy(document)
  if proxy.get("image") != PROXY_IMAGE:
    raise ValueError("P36 Pathways proxy image drifted from the pinned baseline")
  proxy_flags = [
      arg for arg in proxy["args"] if arg.startswith(PROXY_XLA_PREFIX)
  ]
  if proxy_flags != [PROXY_XLA_FLAG]:
    raise ValueError(
        "Pathways proxy must receive exactly one false excess-precision flag"
    )

  head = p33._head_pod(document)
  manager = p33._container(head["initContainers"], "pathways-rm")
  worker = p33._container(
      p33._worker_pod(document)["containers"], "pathways-worker"
  )
  misplaced_flags = [
      (container["name"], arg)
      for container in (manager, worker)
      for arg in container["args"]
      if arg.startswith(PROXY_XLA_PREFIX)
  ]
  if misplaced_flags:
    raise ValueError(
        f"excess-precision flag must be delivered only to the proxy: {misplaced_flags}"
    )
  scratch_args = []
  for container_name in ("pathways-proxy", "pathways-rm"):
    container = p33._container(head["initContainers"], container_name)
    scratch_args.extend(
        arg
        for arg in container["args"]
        if arg.startswith("--gcs_scratch_location=")
    )
  expected_scratch = f"--gcs_scratch_location={SCRATCH_ROOT}/{name}"
  if scratch_args != [expected_scratch, expected_scratch]:
    raise ValueError(
        "Pathways proxy and resource manager lost the isolated P36 scratch"
    )

  address = f"{name}-pathways-head-0-0.{name}"
  if f"--resource_manager_address={address}:29001" not in worker["args"]:
    raise ValueError("P36 worker address drifted from the generated JobSet name")
  worker_env = {
      entry["name"]: entry.get("value") for entry in worker["env"]
  }
  if worker_env.get("PATHWAYS_HEAD") != address:
    raise ValueError("P36 worker PATHWAYS_HEAD drifted")

  serialized = yaml.safe_dump(document, sort_keys=False)
  if p33._BRANCH not in serialized:
    raise ValueError("P36 JobSet does not fetch the canonical source branch")
  if "wandb_v1_" in serialized or "github_pat_" in serialized or "ghp_" in serialized:
    raise ValueError("generated P36 JobSet contains a literal credential")


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
      "# Generated by canon-zero-tim/cluster/render_p36_proxy_xla_jobset.py.\n"
      + yaml.safe_dump(document, sort_keys=False),
      encoding="utf-8",
  )
  print(f"[P36.JOBSET] RENDERED path={args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
