#!/usr/bin/env python3
"""Render strict, independent P33 JobSets from the reviewed 64-chip base."""

from __future__ import annotations

import argparse
import copy
import dataclasses
from pathlib import Path
import re
import shlex
from typing import Any, Iterable, Mapping

import yaml


_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RUN_ID_RE = re.compile(r"[a-z0-9](?:[-a-z0-9]{0,14}[a-z0-9])?")
_BRANCH = "yuxzhang/canon-zero-tim"
_SCRATCH_ROOT = "gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p33"
_GSM8K_FULL_MAX_RESTARTS = 3
_PRIORITY_CLASS = "very-high"


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
  if re.match(r"^[0-9]+[eE][0-9]+$", data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')
  return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_representer, Dumper=yaml.SafeDumper)


@dataclasses.dataclass(frozen=True, slots=True)
class JobSpec:
  """Defines one immutable P33 queue entry."""

  key: str
  workload: str
  stage: str
  profile: str
  no_commit: bool
  job_prefix: str
  command: tuple[str, ...]
  enable_evaluation: bool = False
  dp_size: int = 16
  tp_size: int = 4
  optimizer_resident: bool = False

  @property
  def filename(self) -> str:
    return f"jobset-p33-{self.key}.yaml"


def _common_args(
    *,
    max_steps: int,
    prompt: int,
    response: int,
    dp_size: int = 16,
    tp_size: int = 4,
) -> tuple[str, ...]:
  return (
      f"--mesh_dp={dp_size}",
      f"--mesh_tp={tp_size}",
      "--batch_size=32",
      "--mini_batch_size=32",
      f"--train_trajectory_micro_batch_size={dp_size}",
      f"--max_steps={max_steps}",
      "--num_generations=8",
      f"--max_prompt_length={prompt}",
      f"--max_response_length={response}",
      "--max_concurrency=256",
  )


def _frozenlake_command(
    max_steps: int,
    *,
    short_alignment: bool = False,
    enable_evaluation: bool = False,
    dp_size: int = 16,
    tp_size: int = 4,
) -> tuple[str, ...]:
  local_trajectories = 256 // dp_size
  command = (
    "python3",
    "-u",
    "examples/frozenlake/train_frozenlake_qwen3.py",
    *_common_args(
        max_steps=max_steps,
        prompt=4096,
        response=512 if short_alignment else 2048,
        dp_size=dp_size,
        tp_size=tp_size,
    ),
    f"--vllm_max_num_seqs={local_trajectories}",
    "--vllm_max_num_batched_tokens=256",
    f"--env_max_steps={2 if short_alignment else 5}",
    "--num_batches=150",
    "--learning_rate=1e-6",
    "--b1=0.9",
    "--b2=0.95",
    "--weight_decay=0",
    "--beta=0",
    "--epsilon=0.003",
    "--epsilon_high=0.005",
    "--loss_algo=gspo-token",
    "--advantage_estimator=rloo",
    "--temperature=0.7",
    "--top_k=0",
    "--top_p=1.0",
  )
  if enable_evaluation:
    command += ("--num_test_batches=4", "--eval_every_n_steps=10")
  return command


def _gsm8k_command(max_steps: int) -> tuple[str, ...]:
  return (
      "python3",
      "-u",
      "examples/math_gsm8k/qwen3_grpo_demo.py",
      *_common_args(max_steps=max_steps, prompt=1024, response=1024),
      "--train_micro_batch_size=32",
      "--compute_logps_micro_batch_size=32",
      "--rollout_vllm_hbm_utilization=0.20",
      "--rollout_vllm_max_num_seqs=16",
      "--rollout_vllm_max_num_batched_tokens=256",
      "--wandb_project=zero-tim-gsm8k-dp16-tp4",
  )


_SPECS = (
    JobSpec(
        key="gsm8k-alignment-short",
        workload="gsm8k",
        stage="alignment-short",
        profile="cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env",
        no_commit=True,
        job_prefix="canon-p33-gsm8k-align",
        command=_gsm8k_command(1),
    ),
    JobSpec(
        key="frozenlake-alignment-short",
        workload="frozenlake",
        stage="alignment-short",
        profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
        no_commit=True,
        job_prefix="canon-p33-fl-align",
        command=_frozenlake_command(1, short_alignment=True),
    ),
    JobSpec(
        key="frozenlake-backward-no-commit",
        workload="frozenlake",
        stage="backward-no-commit",
        profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
        no_commit=True,
        job_prefix="canon-p33-fl-bwd",
        command=_frozenlake_command(1),
    ),
    JobSpec(
        key="gsm8k-full",
        workload="gsm8k",
        stage="full",
        profile="cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env",
        no_commit=False,
        job_prefix="canon-p33-gsm8k-full",
        command=_gsm8k_command(200),
    ),
    JobSpec(
        key="frozenlake-full",
        workload="frozenlake",
        stage="full",
        profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
        no_commit=False,
        job_prefix="canon-p33-fl-full",
        command=_frozenlake_command(450),
    ),
    JobSpec(
        key="frozenlake-full-eval",
        workload="frozenlake",
        stage="full",
        profile="cluster/profiles/qwen3-8b-dp16-tp4-frozenlake.env",
        no_commit=False,
        job_prefix="canon-p42-fl-eval",
        command=_frozenlake_command(450, enable_evaluation=True),
        enable_evaluation=True,
    ),
)


PROXY_XLA_ENV = "XLA_FLAGS"
PROXY_XLA_FLAG = "--xla_allow_excess_precision=false"
_PROXY_XLA_PREFIX = "--xla_allow_excess_precision="


def ensure_proxy_xla_env(proxy: dict[str, Any]) -> None:
  """Deliver the excess-precision flag through the proxy environment.

  Pathways compiles on the server side, so a client-container XLA_FLAGS value
  never reaches the TPU compiler, and the pinned proxy rejects the flag as a
  command-line argument (P36 flagon1: unknown command line flag).  The verified
  channel is the proxy container environment (P36 envon1: replicated arm
  0/262144 across widths 2/4/8 at depth 8).  Exactly one XLA_FLAGS entry with
  exactly this value is admitted; a raw argv flag or a conflicting entry is a
  contract violation, not something to repair silently.
  """
  raw = [a for a in proxy.get("args", []) if a.startswith(_PROXY_XLA_PREFIX)]
  if raw:
    raise ValueError(
        "Pathways proxy args carry a raw excess-precision flag; the pinned "
        "proxy rejects it as an unknown command line flag"
    )
  env = proxy.setdefault("env", [])
  matches = [e for e in env if e.get("name") == PROXY_XLA_ENV]
  if not matches:
    env.append({"name": PROXY_XLA_ENV, "value": PROXY_XLA_FLAG})
    return
  if matches != [{"name": PROXY_XLA_ENV, "value": PROXY_XLA_FLAG}]:
    raise ValueError(
        "Pathways proxy has a conflicting or duplicate XLA_FLAGS entry"
    )


def _head_pod(document: Mapping[str, Any]) -> dict[str, Any]:
  jobs = document["spec"]["replicatedJobs"]
  if [job["name"] for job in jobs] != ["pathways-head", "pathways-worker"]:
    raise ValueError("64-chip base JobSet replicated-job layout changed")
  return jobs[0]["template"]["spec"]["template"]["spec"]


def _worker_pod(document: Mapping[str, Any]) -> dict[str, Any]:
  return document["spec"]["replicatedJobs"][1]["template"]["spec"][
      "template"
  ]["spec"]


def _container(items: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
  matches = [item for item in items if item.get("name") == name]
  if len(matches) != 1:
    raise ValueError(f"expected exactly one container named {name!r}")
  return matches[0]


def _set_named_env(
    env: list[dict[str, Any]], values: Mapping[str, str], *, remove: Iterable[str]
) -> None:
  remove_set = set(remove)
  existing = {entry["name"]: entry for entry in env}
  missing_remove = remove_set - existing.keys()
  if missing_remove:
    raise ValueError(f"base JobSet lost expected env keys: {sorted(missing_remove)}")
  env[:] = [entry for entry in env if entry["name"] not in remove_set]
  existing = {entry["name"]: entry for entry in env}
  for name, value in values.items():
    if name in existing:
      existing[name].clear()
      existing[name].update({"name": name, "value": value})
    else:
      env.append({"name": name, "value": value})


def _replace_arg(args: list[str], prefix: str, value: str) -> None:
  indices = [index for index, arg in enumerate(args) if arg.startswith(prefix)]
  if len(indices) != 1:
    raise ValueError(f"expected exactly one {prefix!r} argument")
  args[indices[0]] = value


def _job_name(spec: JobSpec, source_commit: str, run_id: str) -> str:
  name = f"{spec.job_prefix}-{run_id}-{source_commit[:8]}"
  if len(name) > 63:
    raise ValueError(f"generated JobSet name exceeds 63 characters: {name}")
  return name


def render_jobset(
    base: Mapping[str, Any], spec: JobSpec, source_commit: str, run_id: str
) -> dict[str, Any]:
  """Returns one fail-closed P33 JobSet without mutating the base mapping."""
  if not _SHA_RE.fullmatch(source_commit):
    raise ValueError("source commit must be one lowercase 40-character SHA")
  if not _RUN_ID_RE.fullmatch(run_id):
    raise ValueError(
        "run id must be a 1-16 character lowercase DNS label component"
    )
  if spec.dp_size * spec.tp_size != 64:
    raise ValueError("P33/P45 JobSpecs must consume exactly 64 devices")
  if spec.dp_size not in (8, 16) or spec.tp_size not in (4, 8):
    raise ValueError("P33/P45 JobSpec topology is not registered")

  document = copy.deepcopy(base)
  name = _job_name(spec, source_commit, run_id)
  state = f"/tmp/canon-state/{name}"
  scratch = f"{_SCRATCH_ROOT}/{name}"
  document["metadata"]["name"] = name
  document["metadata"].setdefault("labels", {}).update({
      "canon.zero-tim/workload": spec.workload,
      "canon.zero-tim/stage": spec.stage,
      "canon.zero-tim/source": source_commit[:8],
  })
  max_restarts = (
      _GSM8K_FULL_MAX_RESTARTS if spec.key == "gsm8k-full" else 0
  )
  document["spec"]["failurePolicy"]["maxRestarts"] = max_restarts
  document["spec"]["replicatedJobs"][1]["template"]["spec"][
      "backoffLimit"
  ] = 0

  head = _head_pod(document)
  if head["restartPolicy"] != "Never":
    raise ValueError("strict P33 head must retain restartPolicy=Never")
  proxy = _container(head["initContainers"], "pathways-proxy")
  resource_manager = _container(head["initContainers"], "pathways-rm")
  ensure_proxy_xla_env(proxy)
  _replace_arg(
      proxy["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}"
  )
  _replace_arg(
      resource_manager["args"],
      "--gcs_scratch_location=",
      f"--gcs_scratch_location={scratch}",
  )

  main = _container(head["containers"], "jax-tpu")
  _set_named_env(
      main["env"],
      {
          "CANON_MODE": "run",
          "CANON_PROFILE_FILE": spec.profile,
          "CANON_STATE": state,
          "CANON_P32_EXPECT_MODEL_MESH_IDS": "",
          "CANON_EXPECT_TRAIN_MESH_IDS": "",
          "CANON_REQUIRE_TRAIN_MESH_PIN": "0",
          "CANON_EXPECT_COMMIT": source_commit,
          "CANON_P32_TRAIN_ADMITTED": "1",
          "CANON_P32_DP_REDUCTION_ADMITTED": "1",
          "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
          "CANON_P33_SHARED_MESH": f"{spec.dp_size},{spec.tp_size}",
          "CANON_P33_RUN_STAGE": spec.stage,
          "CANON_P33_NO_COMMIT": "1" if spec.no_commit else "0",
          "CANON_OPT_STATE_RESIDENT": (
              "1" if spec.optimizer_resident else "0"
          ),
          "CANON_P30_OPT_STATE_OFFLOAD": (
              "0" if spec.optimizer_resident else "1"
          ),
          "CANON_GSM8K_AB_REPORT_ONLY": "0",
          "CANON_GSM8K_ALIGNMENT_WARN_ONLY": (
              "1"
              if spec.workload == "gsm8k" and spec.stage == "full"
              else "0"
          ),
          "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": (
              "1"
              if spec.workload == "frozenlake" and spec.stage == "full"
              else "0"
          ),
          "CANON_P33_SHORT_ALIGNMENT": (
              "1" if spec.stage == "alignment-short" else "0"
          ),
          "CANON_PRE_ALIGN_GATE": "1",
          "CANON_RUN_CMD": shlex.join(spec.command),
          "CANON_RUN_LOG": f"{state}/run.log",
          "CANON_PRE_ALIGN_REPORT": f"{state}/pre_alignment.jsonl",
          "CANON_ALIGN_REPORT": f"{state}/alignment.jsonl",
          "CANON_UPDATE_REPORT": f"{state}/updates.jsonl",
          "CANON_P38_MISMATCH_CAPSULE": (
              f"{state}/p38_frozenlake_mismatch_capsule.npz"
              if spec.workload == "frozenlake"
              and spec.stage == "backward-no-commit"
              else ""
          ),
          "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "2",
          "CANON_WANDB_RUN_NAME": name,
          "MIN_TOKEN_BUCKET": str(spec.dp_size * 256),
          "CANON_WAYCOUNT_WIDTHS": "2,4,8",
          "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
          "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
          "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
          "CANON_GCS_CACHE_BUCKET": "gs://yuxzhang-tunix-models/cache/p33_compilation_cache",
      },
      remove=("CANON_P32_RC_STAGE",),
  )
  if spec.workload == "frozenlake":
    _set_named_env(
        main["env"],
        {
            "CANON_P33_ENABLE_EVAL": "1" if spec.enable_evaluation else "0",
            "CANON_P33_DISABLE_EVAL": "0" if spec.enable_evaluation else "1",
            "CANON_P31_ENABLE_EVAL": "1" if spec.enable_evaluation else "0",
        },
        remove=(),
    )

  worker = _container(_worker_pod(document)["containers"], "pathways-worker")
  address = f"{name}-pathways-head-0-0.{name}"
  _replace_arg(
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

  validate_jobset(document, spec, source_commit, run_id)
  return document


def _env_values(document: Mapping[str, Any]) -> dict[str, str]:
  main = _container(_head_pod(document)["containers"], "jax-tpu")
  names = [entry["name"] for entry in main["env"]]
  if len(names) != len(set(names)):
    raise ValueError("generated jax-tpu environment contains duplicate names")
  return {
      entry["name"]: entry["value"]
      for entry in main["env"]
      if "value" in entry
  }


def validate_jobset(
    document: Mapping[str, Any],
    spec: JobSpec,
    source_commit: str,
    run_id: str,
) -> None:
  """Rejects a generated manifest whose launch or isolation contract drifted."""
  name = _job_name(spec, source_commit, run_id)
  state = f"/tmp/canon-state/{name}"
  if document.get("apiVersion") != "jobset.x-k8s.io/v1alpha2":
    raise ValueError("P33 manifests require the reviewed JobSet API version")
  if document.get("metadata", {}).get("name") != name:
    raise ValueError("generated JobSet name drifted")
  expected_max_restarts = (
      _GSM8K_FULL_MAX_RESTARTS if spec.key == "gsm8k-full" else 0
  )
  if (
      document["spec"]["failurePolicy"].get("maxRestarts")
      != expected_max_restarts
  ):
    raise ValueError(
        "P33 JobSet restart policy drifted: "
        f"expected {expected_max_restarts} for {spec.key}"
    )
  head_job = document["spec"]["replicatedJobs"][0]
  if head_job["template"]["spec"].get("backoffLimit") != 0:
    raise ValueError("P33 head Job must not retry a failed training attempt")
  worker_job = document["spec"]["replicatedJobs"][1]
  if worker_job["template"]["spec"].get("backoffLimit") != 0:
    raise ValueError("P33 worker Job must not retry a failed training attempt")
  priorities = {
      "pathways-head": _head_pod(document).get("priorityClassName"),
      "pathways-worker": _worker_pod(document).get("priorityClassName"),
  }
  if any(value != _PRIORITY_CLASS for value in priorities.values()):
    raise ValueError(
        "P33 Pathways Pod priority class drifted: "
        f"expected {_PRIORITY_CLASS!r}, got {priorities}"
    )

  env = _env_values(document)
  expected = {
      "CANON_MODE": "run",
      "CANON_PROFILE_FILE": spec.profile,
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_P32_TRAIN_ADMITTED": "1",
      "CANON_P32_DP_REDUCTION_ADMITTED": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P33_SHARED_MESH": f"{spec.dp_size},{spec.tp_size}",
      "CANON_P33_RUN_STAGE": spec.stage,
      "CANON_P33_NO_COMMIT": "1" if spec.no_commit else "0",
      "CANON_OPT_STATE_RESIDENT": (
          "1" if spec.optimizer_resident else "0"
      ),
      "CANON_P30_OPT_STATE_OFFLOAD": (
          "0" if spec.optimizer_resident else "1"
      ),
      "CANON_GSM8K_AB_REPORT_ONLY": "0",
      "CANON_GSM8K_ALIGNMENT_WARN_ONLY": (
          "1"
          if spec.workload == "gsm8k" and spec.stage == "full"
          else "0"
      ),
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": (
          "1"
          if spec.workload == "frozenlake" and spec.stage == "full"
          else "0"
      ),
      "CANON_P33_SHORT_ALIGNMENT": (
          "1" if spec.stage == "alignment-short" else "0"
      ),
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_RUN_CMD": shlex.join(spec.command),
      "CANON_WANDB_RUN_NAME": name,
      "MIN_TOKEN_BUCKET": str(spec.dp_size * 256),
      "CANON_P38_MISMATCH_CAPSULE": (
          f"{state}/p38_frozenlake_mismatch_capsule.npz"
          if spec.workload == "frozenlake"
          and spec.stage == "backward-no-commit"
          else ""
      ),
      "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "2",
  }
  wrong = {
      key: env.get(key)
      for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"generated P33 environment drifted: {wrong}")
  if "CANON_P32_RC_STAGE" in env:
    raise ValueError("P33 JobSet retained a P32 release-candidate stage")
  if env.get("CANON_P32_EXPECT_MODEL_MESH_IDS") != "":
    raise ValueError("autoscaled P33 JobSet must not inherit old device ids")
  if env.get("CANON_EXPECT_TRAIN_MESH_IDS") != "":
    raise ValueError("autoscaled P33 JobSet must discover its train mesh ids")
  if spec.workload == "frozenlake":
    expected_eval = {
        "CANON_P33_ENABLE_EVAL": "1" if spec.enable_evaluation else "0",
        "CANON_P33_DISABLE_EVAL": "0" if spec.enable_evaluation else "1",
        "CANON_P31_ENABLE_EVAL": "1" if spec.enable_evaluation else "0",
    }
    wrong_eval = {
        key: env.get(key)
        for key, value in expected_eval.items()
        if env.get(key) != value
    }
    if wrong_eval:
      raise ValueError(f"FrozenLake evaluation contract drifted: {wrong_eval}")
    has_eval_args = (
        "--num_test_batches=4" in env["CANON_RUN_CMD"]
        and "--eval_every_n_steps=10" in env["CANON_RUN_CMD"]
    )
    if has_eval_args != spec.enable_evaluation:
      raise ValueError("FrozenLake evaluation command drifted")
  if env.get("CANON_P38_MISMATCH_CAPSULE_MAX_ROWS") != "2":
    raise ValueError("P38 mismatch capsule must retain its two-row bound")
  main = _container(_head_pod(document)["containers"], "jax-tpu")
  secret_refs = {
      entry["name"]: entry.get("valueFrom", {}).get("secretKeyRef", {})
      for entry in main["env"]
      if entry["name"] in ("INJECTED_HF_TOKEN", "INJECTED_WANDB_API_KEY")
  }
  if secret_refs.get("INJECTED_HF_TOKEN", {}).get("key") != "HF_TOKEN":
    raise ValueError("generated JobSet lost the Hugging Face Secret reference")
  if secret_refs.get("INJECTED_WANDB_API_KEY", {}).get("key") != "WANDB_API_KEY":
    raise ValueError("generated JobSet lost the W&B Secret reference")
  state_paths = (
      env.get("CANON_STATE", ""),
      env.get("CANON_RUN_LOG", ""),
      env.get("CANON_PRE_ALIGN_REPORT", ""),
      env.get("CANON_ALIGN_REPORT", ""),
      env.get("CANON_UPDATE_REPORT", ""),
  )
  if not all(path.startswith(f"/tmp/canon-state/{name}") for path in state_paths):
    raise ValueError("P33 state and evidence paths are not isolated by JobSet")

  head = _head_pod(document)
  scratch_args = []
  for container_name in ("pathways-proxy", "pathways-rm"):
    container = _container(head["initContainers"], container_name)
    scratch_args.extend(
        arg for arg in container["args"] if arg.startswith("--gcs_scratch_location=")
    )
  expected_scratch = f"--gcs_scratch_location={_SCRATCH_ROOT}/{name}"
  if scratch_args != [expected_scratch, expected_scratch]:
    raise ValueError("Pathways proxy and resource manager do not share one isolated scratch")

  worker = _container(_worker_pod(document)["containers"], "pathways-worker")
  address = f"{name}-pathways-head-0-0.{name}"
  if f"--resource_manager_address={address}:29001" not in worker["args"]:
    raise ValueError("worker resource-manager address does not follow generated JobSet name")
  worker_env = {
      entry["name"]: entry.get("value") for entry in worker["env"]
  }
  if worker_env.get("PATHWAYS_HEAD") != address:
    raise ValueError("worker PATHWAYS_HEAD does not follow generated JobSet name")

  serialized = yaml.safe_dump(document, sort_keys=False)
  if _BRANCH not in serialized:
    raise ValueError("P33 JobSet does not fetch the canonical source branch")
  if "wandb_v1_" in serialized or "github_pat_" in serialized or "ghp_" in serialized:
    raise ValueError("generated JobSet contains a literal credential")


def load_base(path: Path) -> dict[str, Any]:
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  if not isinstance(document, dict):
    raise ValueError(f"base JobSet is not one YAML mapping: {path}")
  return document


def render_all(
    *, base_path: Path, output_dir: Path, source_commit: str, run_id: str
) -> tuple[Path, ...]:
  base = load_base(base_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  outputs = tuple(output_dir / spec.filename for spec in _SPECS)
  existing = [path for path in outputs if path.exists()]
  if existing:
    raise FileExistsError(
        "refusing to overwrite rendered JobSets: "
        + ", ".join(str(path) for path in existing)
    )
  for spec, path in zip(_SPECS, outputs, strict=True):
    document = render_jobset(base, spec, source_commit, run_id)
    header = (
        "# Generated by canon-zero-tim/cluster/render_p33_jobsets.py.\n"
        "# Do not edit this output; change the reviewed base or renderer instead.\n"
    )
    path.write_text(
        header + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    print(f"[P33.JOBSET] RENDERED key={spec.key} path={path}")
  return outputs


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
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
  )
  print(
      "[P33.JOBSET] VERDICT PASS "
      f"count={len(outputs)} source={args.source_commit} run_id={args.run_id}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
