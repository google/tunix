#!/usr/bin/env python3
"""Render independent P45/M15 DP8xTP8 exact-TiTO diagnostics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

import yaml


_TASK_DIR = Path(__file__).resolve().parents[1]
_PACKAGE = _TASK_DIR.parents[1]
_CLUSTER = _PACKAGE / "cluster"
if str(_CLUSTER) not in sys.path:
  sys.path.insert(0, str(_CLUSTER))

import render_p57_frozenlake_tim as p57


_SHA_RE = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _container(document: dict) -> dict:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  return next(item for item in pod["containers"] if item["name"] == "jax-tpu")


def _env(document: dict) -> dict[str, str]:
  return {
      item["name"]: item["value"]
      for item in _container(document)["env"]
      if "value" in item
  }


def _scheduling_contract(document: dict) -> dict:
  """Returns only topology, exclusivity, and JobSet execution cardinality."""
  spec = document["spec"]
  result = {
      name: copy.deepcopy(spec.get(name))
      for name in ("failurePolicy", "network", "startupPolicy", "successPolicy")
  }
  jobs = []
  for job in spec["replicatedJobs"]:
    job_spec = job["template"]["spec"]
    pod_template = job_spec["template"]
    pod = pod_template["spec"]
    jobs.append({
        "name": job["name"],
        "replicas": job["replicas"],
        "job_execution": {
            name: copy.deepcopy(value)
            for name, value in job_spec.items()
            if name != "template"
        },
        "pod_metadata": copy.deepcopy(pod_template.get("metadata", {})),
        "pod_scheduling": {
            name: copy.deepcopy(pod.get(name))
            for name in (
                "priorityClassName",
                "nodeSelector",
                "hostNetwork",
                "dnsPolicy",
                "tolerations",
                "restartPolicy",
            )
        },
    })
  result["replicatedJobs"] = jobs
  return result


def render_pair(
    *,
    source_commit: str,
    output_dir: Path,
    p45_run_id: str,
    m15_run_id: str,
    campaign_root: str,
    base_path: Path,
    cpu_nodepool: str,
) -> tuple[Path, Path, Path]:
  if _SHA_RE.fullmatch(source_commit) is None:
    raise ValueError("source commit must be exactly 40 lowercase hex characters")
  if output_dir.exists():
    raise FileExistsError(f"refusing to overwrite diagnostic output: {output_dir}")
  if p45_run_id == m15_run_id:
    raise ValueError("P45 and M15 diagnostic run IDs must differ")
  output_dir.mkdir(parents=True)
  rendered = []
  for workload, candidate, split, run_id in (
      ("p45", "", "", p45_run_id),
      ("m15", "m15", "main", m15_run_id),
  ):
    outputs = p57.render_all(
        base_path=base_path,
        output_dir=output_dir / workload,
        source_commit=source_commit,
        run_id=run_id,
        campaign_tag=f"{campaign_root}-{workload}",
        checkpoint_mode="disabled",
        expected_updates=1,
        run_kind="tito-diagnostic",
        workload_candidate=candidate,
        data_split=split,
        arm="zero",
        stop_after_step=1,
        high_performance=False,
        disable_eval=True,
        cpu_nodepool=cpu_nodepool,
    )
    if len(outputs) != 1:
      raise ValueError(f"{workload} diagnostic rendered {len(outputs)} JobSets")
    rendered.append(outputs[0])

  records = []
  for workload, path in zip(("p45", "m15"), rendered, strict=True):
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidate, split = (("", "") if workload == "p45" else ("m15", "main"))
    control_spec = p57._spec(
        p57._ARM_BY_NAME["zero"],
        1,
        run_kind="tito-diagnostic",
        checkpoint_step=None,
        workload_candidate=candidate,
        data_split=split,
        high_performance=False,
        disable_eval=True,
    )
    expected_document = p57.p33.render_jobset(
        p57.p33.load_base(base_path), control_spec, source_commit, (
            p45_run_id if workload == "p45" else m15_run_id
        )
    )
    expected_head = expected_document["spec"]["replicatedJobs"][0][
        "template"
    ]["spec"]["template"]["spec"]
    expected_head["nodeSelector"]["cloud.google.com/gke-nodepool"] = cpu_nodepool
    expected_scheduling = _scheduling_contract(expected_document)
    if _scheduling_contract(document) != expected_scheduling:
      raise ValueError(
          f"{workload} diagnostic topology/exclusivity/autoscaling drifted"
      )
    env = _env(document)
    required = {
        "CANON_PROFILE_FILE": p57._TITO_DIAGNOSTIC_PROFILE,
        "CANON_P57_RUN_KIND": "tito-diagnostic",
        "CANON_P57_TIM_ARM": "zero",
        "CANON_P57_EXPECTED_UPDATES": "1",
        "CANON_P57_STOP_AFTER_STEP": "1",
        "CANON_P33_RUN_STAGE": "rollout-only",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P57_TOKEN_CONTINUITY": "exact",
        "CANON_P57_TOKEN_CONTINUITY_DEBUG": "collect-64",
        "CANON_P57_TITO_ROLLOUT_ONLY": "1",
        "CANON_P33_ENABLE_EVAL": "0",
        "CANON_P33_DISABLE_EVAL": "1",
        "CANON_P31_ENABLE_EVAL": "0",
        "CANON_FROZENLAKE_CKPT_MODE": "disabled",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "0",
        "CANON_V1_HP_FULL": "0",
    }
    wrong = {
        name: env.get(name)
        for name, expected in required.items()
        if env.get(name) != expected
    }
    if wrong:
      raise ValueError(f"{workload} diagnostic intent drifted: {wrong}")
    command = env["CANON_RUN_CMD"].split()
    for argument in (
        "--mesh_dp=8",
        "--mesh_tp=8",
        "--max_steps=1",
        "--num_test_batches=1",
        "--eval_every_n_steps=0",
        "--evaluation_only",
    ):
      if command.count(argument) != 1:
        raise ValueError(f"{workload} command lacks unique {argument}")
    expected_turns = "--env_max_steps=15" if workload == "m15" else "--env_max_steps=5"
    if command.count(expected_turns) != 1:
      raise ValueError(f"{workload} turn horizon drifted")
    records.append({
        "workload": workload,
        "path": path.relative_to(output_dir).as_posix(),
        "sha256": _sha256(path),
        "jobset": document["metadata"]["name"],
        "profile": required["CANON_PROFILE_FILE"],
        "stage": "rollout-only",
        "backward_calls": 0,
        "optimizer_commits": 0,
        "checkpoint_writes": 0,
        "launch": "not-executed",
    })
  index = output_dir / "manifest-index.json"
  index.write_text(
      json.dumps({
          "schema": "canon.p57-tito-diagnostic-pair.v1",
          "source_commit": source_commit,
          "manifests": records,
          "launch": "not-executed",
      }, sort_keys=True, indent=2) + "\n",
      encoding="utf-8",
  )
  return rendered[0], rendered[1], index


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--p45-run-id", required=True)
  parser.add_argument("--m15-run-id", required=True)
  parser.add_argument("--campaign-root", required=True)
  parser.add_argument(
      "--base", type=Path, default=_CLUSTER / "jobset-64chip.yaml"
  )
  parser.add_argument(
      "--cpu-nodepool",
      default="canon-cpu-pool",
      choices=("canon-cpu-pool", "cpu-np", "deepswe-cpu-pool-2"),
  )
  args = parser.parse_args()
  paths = render_pair(
      source_commit=args.source_commit,
      output_dir=args.output_dir,
      p45_run_id=args.p45_run_id,
      m15_run_id=args.m15_run_id,
      campaign_root=args.campaign_root,
      base_path=args.base,
      cpu_nodepool=args.cpu_nodepool,
  )
  print(
      "P57_TITO_DIAGNOSTIC_PAIR_READY "
      f"manifests=2 index={paths[2]} launch=not-executed"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
