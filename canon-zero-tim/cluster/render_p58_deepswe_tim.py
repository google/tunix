#!/usr/bin/env python3
"""Render one arm of the paired 128-chip P58 DeepSWE TIM study."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
from typing import Any, Mapping

import yaml

import render_p34_jobset as p34
from v1_full_system_optimization import (
    FULL_SYSTEM_OPTIMIZATION_ENV_NAMES,
    full_system_optimization_additions,
)


MODEL = "Qwen/Qwen3-4B-Instruct-2507"
CLEAN_WHITELIST = (
    "canon-zero-tim/clean_data/p46_q4_learnable/"
    "p46q4census02_qwen3_4b_instruct_2507_n16_learnable_tasks.jsonl"
)
CLEAN_WHITELIST_SHA256 = (
    "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
)
CLEAN_ROWS = 1012
PROFILE = "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
HP_PROFILE = "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-v1-hp.env"
TOPOLOGY = "4x4x8"
WORKERS = 32
ROLE_DP = 8
ROLE_TP = 8
GLOBAL_PROMPTS = 8
GENERATIONS = 16
MAX_CONCURRENCY = GLOBAL_PROMPTS * GENERATIONS
FIXED_SEED = 42
_STAGE_STEPS = {"three-update": 3, "full": 1000}
_ARMS = ("native", "zero")
_KUEUE_MANAGED_WORKER_POOLS = frozenset({
    "auto",
    "none",
    "tpu-v5p-slice",
    "any",
})
_EXCLUSIVE_TOPOLOGY_ANNOTATION = (
    "alpha.jobset.sigs.k8s.io/exclusive-topology"
)
_KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"
_CPU_NODEPOOL = "cpu-np"
_JOBSET_REPLICATEDJOB_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"
_PATHWAYS_HEAD_REPLICATEDJOB = "pathways-head"
_HOSTNAME_TOPOLOGY_KEY = "kubernetes.io/hostname"
_TOKEN_TRANSPORT_LABEL = "canon.zero-tim/token-transport"
_TOKEN_TRANSPORT = "tito"
_KUBERNETES_DNS_LABEL = re.compile(
    r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?\Z"
)
_FILTER_STATUSES = (
    "MAX_STEPS_REACHED",
    "MAX_CONTEXT_LIMIT_REACHED",
    "TIMEOUT",
    "ENV_TIMEOUT",
    "MODEL_TIMEOUT",
    "REWARD_TIMEOUT",
)
_SEAM_LOCALIZATION_MODES = ("", "coarse")
_SEAM_DIAGNOSTIC_ROUNDS = 3
_SEAM_MIN_POSITION = 1686
_SEAM_MAX_POSITION = 4096
_SEAM_MAX_BYTES = 4 * 1024 * 1024 * 1024
_SEAM_TAIL_MAX_BYTES = 64 * 1024 * 1024
_SEAM_INCIDENT_MAX_BYTES = 128 * 1024 * 1024
_SEAM_CAPTURE_BOUNDS = (1686, 2512, 3072, 3584, 4096)
_RETIRED_DEVICE_PROBE_TRIGGER = "CANON_EXPECTED_SLICE_DEVICES"


def _service_containers(head: Mapping[str, Any]) -> list[dict[str, Any]]:
  return list(head.get("initContainers", [])) + list(head["containers"])


def _pathways_head_anti_affinity_term() -> dict[str, Any]:
  return {
      "labelSelector": {
          "matchExpressions": [{
              "key": _JOBSET_REPLICATEDJOB_LABEL,
              "operator": "In",
              "values": [_PATHWAYS_HEAD_REPLICATEDJOB],
          }],
      },
      "namespaceSelector": {},
      "topologyKey": _HOSTNAME_TOPOLOGY_KEY,
  }


def _remove_proxy_precision_pin(proxy: dict[str, Any]) -> None:
  env = proxy.get("env", [])
  proxy["env"] = [item for item in env if item.get("name") != p34.PROXY_XLA_ENV]


def _command(
    stage: str,
    *,
    run_root: str,
    whitelist: str,
    sampler_is: bool = False,
) -> tuple[str, ...]:
  if stage not in _STAGE_STEPS:
    raise ValueError("P58 admits only three-update or full")
  args = list(
      p34._command("three-update", run_root=run_root, whitelist=whitelist)
  )
  replacements = {
      "--model_version=Qwen3-32B": "--model_version=Qwen3-4B-Instruct-2507",
      "--num_generations=8": "--num_generations=16",
      "--episode_timeout_secs=4800": "--episode_timeout_secs=3000",
      "--step_timeout_secs=1800": "--step_timeout_secs=600",
      "--reward_timeout_secs=1800": "--reward_timeout_secs=600",
      "--rollout_batch_timeout_secs=5400": "--rollout_batch_timeout_secs=3600",
      "--max_concurrency=64": f"--max_concurrency={MAX_CONCURRENCY}",
      "--rollout_mesh_dp=16": "--rollout_mesh_dp=8",
      "--train_mesh_dp=16": "--train_mesh_dp=8",
      "--rollout_vllm_max_num_seqs=4": "--rollout_vllm_max_num_seqs=16",
      "--max_steps=3": f"--max_steps={_STAGE_STEPS[stage]}",
  }
  for old, new in replacements.items():
    if args.count(old) != 1:
      raise ValueError(f"P34 command no longer contains exactly one {old!r}")
    args[args.index(old)] = new
  checkpoint_args = [
      item for item in args if item.startswith("--ckpt_dir=")
  ]
  if len(checkpoint_args) != 1:
    raise ValueError(
        "P34 command no longer contains exactly one checkpoint directory"
    )
  args[args.index(checkpoint_args[0])] = "--ckpt_dir=none"
  for prefix in ("--save_interval_steps=", "--max_to_keep="):
    inherited = [item for item in args if item.startswith(prefix)]
    if len(inherited) != 1:
      raise ValueError(
          f"P34 command no longer contains exactly one {prefix!r} argument"
      )
    args.remove(inherited[0])
  args.extend((
      f"--seed={FIXED_SEED}",
      f"--expected_filtered_rows={CLEAN_ROWS}",
      "--loss_scale_factor=16384",
      "--loss_denominator_weighted_accumulation",
      "--overlong_filter",
      "--filter_statuses",
      *_FILTER_STATUSES,
  ))
  if sampler_is:
    args.extend((
        "--sampler_is=token",
        "--sampler_is_threshold=2.0",
    ))
  return tuple(args)


def render(
    base: Mapping[str, Any],
    *,
    source_commit: str,
    source_branch: str,
    client_image: str,
    run_id: str,
    stage: str,
    arm: str,
    cpu_nodepool: str,
    worker_nodepool: str,
    model_pvc: str,
    instance_type: str = TOPOLOGY,
    whitelist: str = CLEAN_WHITELIST,
    whitelist_sha256: str = CLEAN_WHITELIST_SHA256,
    sampler_is: bool = False,
    high_performance: bool = False,
    checked_vma_off_diagnostic: bool = False,
    checked_vma_on_diagnostic: bool = False,
    seam_localization: str = "",
) -> dict[str, Any]:
  """Returns one immutable P58 native or zero JobSet."""
  if stage not in _STAGE_STEPS:
    raise ValueError("P58 admits only three-update or full")
  if arm not in _ARMS:
    raise ValueError("P58 arm must be native or zero")
  if high_performance and (arm != "zero" or stage != "full"):
    raise ValueError("P58 high-performance is admitted only for Zero full")
  if checked_vma_off_diagnostic and checked_vma_on_diagnostic:
    raise ValueError("P58 checked-VMA diagnostic selectors are mutually exclusive")
  if seam_localization not in _SEAM_LOCALIZATION_MODES:
    raise ValueError("P58 seam localization must be empty or coarse")
  checked_vma_diagnostic = (
      "off" if checked_vma_off_diagnostic else
      "on" if checked_vma_on_diagnostic else ""
  )
  if checked_vma_diagnostic and (
      arm != "zero" or stage != "full" or high_performance
  ):
    raise ValueError(
        "P58 checked-VMA diagnostic is its own Zero/full HP selector"
    )
  if seam_localization and (
      arm != "zero"
      or stage != "full"
      or high_performance
      or bool(checked_vma_diagnostic)
  ):
    raise ValueError(
        "P58 seam localization is its own Zero/full HP diagnostic selector"
    )
  if sampler_is and arm != "native":
    raise ValueError("P58 sampler IS is admitted only for the native arm")
  if sampler_is and high_performance:
    raise ValueError("P58 sampler IS and Zero high-performance are disjoint")
  if cpu_nodepool != _CPU_NODEPOOL:
    raise ValueError("P58 requires the admitted cpu-np CPU node pool")
  if whitelist != CLEAN_WHITELIST or whitelist_sha256 != CLEAN_WHITELIST_SHA256:
    raise ValueError("P58 requires the reviewed 1012-task clean whitelist")

  # Use the bounded P34 render only as a structural JobSet constructor.  P58
  # replaces its workload contract below and validates the final document.
  document = p34.render(
      base,
      source_commit=source_commit,
      source_branch=source_branch,
      client_image=client_image,
      run_id=run_id,
      stage="three-update",
      cpu_nodepool=cpu_nodepool,
      worker_nodepool=worker_nodepool,
      model_pvc=model_pvc,
      whitelist=whitelist,
      whitelist_sha256=whitelist_sha256,
      fixed_lm_head=False,
  )
  document["metadata"].setdefault("annotations", {})[
      _EXCLUSIVE_TOPOLOGY_ANNOTATION
  ] = "cloud.google.com/gke-nodepool"

  hp_bundle = high_performance or bool(checked_vma_diagnostic) or bool(
      seam_localization
  )
  zero_hp_ab_warning = high_performance and arm == "zero" and stage == "full"
  treatment = (
      f"seam{seam_localization}"
      if seam_localization
      else f"vma{checked_vma_diagnostic}"
      if checked_vma_diagnostic
      else "zero-hp"
      if high_performance
      else "native-is"
      if sampler_is
      else arm
  )
  name = (
      f"canon-p58-{treatment}-"
      f"{'three' if stage == 'three-update' else 'full'}-{run_id}"
      if checked_vma_diagnostic or seam_localization
      else f"canon-p58-ds4b-{treatment}-"
      f"{'three' if stage == 'three-update' else 'full'}-{run_id}"
  )

  if len(name) > 63:
    raise ValueError("rendered P58 JobSet name exceeds 63 characters")
  run_root = f"/mnt/disks/linchai_data/deepswe_zero_tim/{name}"
  document["metadata"]["name"] = name
  document["metadata"]["labels"].update({
      "canon.zero-tim/phase": "p58-deepswe-tim",
      "canon.zero-tim/stage": stage,
      "canon.zero-tim/arm": arm,
      "canon.zero-tim/topology": "128",
      "canon.zero-tim/fixed-lm-head": "1" if hp_bundle else "0",
      _TOKEN_TRANSPORT_LABEL: _TOKEN_TRANSPORT,
  })
  if sampler_is:
    document["metadata"]["labels"]["canon.zero-tim/sampler-recipe"] = (
        "token-is"
    )
  if checked_vma_diagnostic:
    document["metadata"]["labels"].update({
        "canon.zero-tim/diagnostic": (
            f"p58-checked-vma-{checked_vma_diagnostic}"
        ),
        "canon.zero-tim/diagnostic-selector": checked_vma_diagnostic,
        "canon.zero-tim/backward": "0",
        "canon.zero-tim/optimizer-commits": "0",
    })
  if seam_localization:
    document["metadata"]["labels"].update({
        "canon.zero-tim/diagnostic": "p58-seam-localization",
        "canon.zero-tim/seam-observer": seam_localization,
        "canon.zero-tim/diagnostic-rounds": str(_SEAM_DIAGNOSTIC_ROUNDS),
        "canon.zero-tim/backward": "0",
        "canon.zero-tim/optimizer-commits": "0",
    })
  queue_name = str(
      document["metadata"]["labels"].get(_KUEUE_QUEUE_LABEL, "")
  )
  if (
      not queue_name
      or len(queue_name) > 63
      or not _KUBERNETES_DNS_LABEL.fullmatch(queue_name)
  ):
    raise ValueError("P58 requires an exact Kueue LocalQueue label")

  head = p34._head(document)
  # Pathways heads intentionally keep the proven host-network transport.  The
  # JobSet controller labels every head Pod with its replicated-job name, so a
  # required hostname anti-affinity term prevents two ResourceManagers from
  # sharing ports 29000/29001 on one CPU node.  P58f08 proved that merely
  # selecting cpu-np without this term lets Kubernetes pack a seventh head
  # onto one of six occupied nodes and connect the worker to a foreign RM.
  affinity = head.setdefault("affinity", {})
  pod_anti_affinity = affinity.setdefault("podAntiAffinity", {})
  required = pod_anti_affinity.setdefault(
      "requiredDuringSchedulingIgnoredDuringExecution", []
  )
  anti_affinity_term = _pathways_head_anti_affinity_term()
  if anti_affinity_term not in required:
    required.append(anti_affinity_term)
  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  manager = p34._container(services, "pathways-rm")
  main = p34._container(head["containers"], "jax-tpu")
  scratch = f"gs://yuxzhang-tunix-models/tmp/canon-zero-tim/p58/{name}"
  p34._replace_arg(
      proxy["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}"
  )
  p34._replace_arg(
      manager["args"], "--gcs_scratch_location=", f"--gcs_scratch_location={scratch}"
  )
  p34._replace_arg(
      manager["args"], "--instance_type=", f"--instance_type=tpuv5:{instance_type}"
  )
  if arm == "native":
    _remove_proxy_precision_pin(proxy)
  else:
    p34.ensure_proxy_xla_env(proxy)

  # P58 is an evidence-bearing paired experiment.  A JobSet-level retry
  # recreates the whole JobSet while retaining the same persistent run root;
  # that can mix attempt artifacts and invalidate the arm comparison.  Keep
  # the signed Attempt-0 contract until explicit attempt isolation exists.
  document["spec"]["failurePolicy"] = {
      "maxRestarts": 0,
      "restartStrategy": "Recreate",
  }
  p34._set_env(main, {
      "CANON_PROFILE_FILE": HP_PROFILE if hp_bundle else PROFILE,
      "CANON_STATE": run_root,
      # TiTO is selected by the DeepSWE workload identity itself.  Keep the
      # identity in the raw JobSet as well as the sourced profile so a
      # rendered full-training YAML cannot depend on a later implicit default.
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P34_TRAJECTORY_CAPTURE": "0",
      "CANON_P39_64CHIP_PILOT": "0",
      "CANON_P39_PILOT_ADMITTED": "0",
      "CANON_P43_DEEPSWE_DEBUG": "0",
      "CANON_P43_DEBUG_ADMITTED": "0",
      "CANON_P43_ROLLOUT_ONLY": "0",
      "CANON_P44_DEEPSWE_PARITY": "0",
      "CANON_P44_PARITY_ADMITTED": "0",
      "CANON_P44_TOPOLOGY": "none",
      "CANON_P44_ROLLOUT_ONLY": "0",
      "CANON_P46_DEEPSWE_TRAIN": "0",
      "CANON_P46_EVALUATION": "0",
      "CANON_P46_TOPOLOGY": "none",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P34_DISABLE_SAMPLER_IS": "0" if sampler_is else "1",
      "CANON_P34_DISABLE_TIS": "0" if sampler_is else "1",
      "CANON_P58_EXPECTED_UPDATES": str(_STAGE_STEPS[stage]),
      "CANON_P58_DEBUG_DIR": f"{run_root}/debug",
      "CANON_V1_HP_FULL": "1" if hp_bundle else "0",
      "CANON_P38_FIXED_LM_HEAD": "1" if hp_bundle else "0",
      "CANON_P34_CLEAN_ROWS": str(CLEAN_ROWS),
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": (
          "1" if arm == "native" or zero_hp_ab_warning else "0"
      ),
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_DEEPSWE_CLEANUP_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_ROLLOUT_BATCH_TIMEOUT_SECS": "3600",
      "CANON_DEEPSWE_PER_TURN_TIMEOUT_SECS": "300",
      "CANON_DEEPSWE_TRAJECTORY_TIMEOUT_SECS": "3000",
      "CANON_DEEPSWE_STEP_TIMEOUT_SECS": "600",
      "CANON_DEEPSWE_REWARD_TIMEOUT_SECS": "600",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
      "R2E_K8S_QUEUE_NAME": queue_name,
      "NODE_SELECTOR_VAL": cpu_nodepool,
      "MIN_TOKEN_BUCKET": "2048",
      "CANON_RUN_CMD": shlex.join(
          _command(
              stage,
              run_root=run_root,
              whitelist=whitelist,
              sampler_is=sampler_is,
          )
      ),
      "CANON_RUN_LOG": f"{run_root}/run.log",
      "CANON_P34_WEIGHT_REPORT": f"{run_root}/weight_attestation.jsonl",
      "CANON_PRE_ALIGN_REPORT": f"{run_root}/pre_alignment.jsonl",
      "CANON_ALIGN_REPORT": f"{run_root}/alignment.jsonl",
      "CANON_UPDATE_REPORT": f"{run_root}/updates.jsonl",
      "CANON_WANDB_RUN_NAME": name,
      "CANON_WANDB_PROJECT": "zero-tim-deepswe-4b-native-zero",
      "CANON_WANDB_GROUP": (
          f"qwen3-4b-p58-native-is-{stage}"
          if sampler_is
          else f"qwen3-4b-p58-{stage}"
      ),
      "CANON_OPTIMIZER_HBM_MIN_FREE_BYTES": str(8 * 1024**3),
  })
  if high_performance:
    p34._set_env(
        main, full_system_optimization_additions("deepswe-qwen4b")
    )
  if checked_vma_diagnostic:
    p34._set_env(main, {
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC": checked_vma_diagnostic,
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": (
            f"{run_root}/p38_diagnostic_round"
        ),
    })
  if seam_localization:
    capture = f"{run_root}/p38_serving_capture"
    p34._set_env(main, {
        "CANON_P58_SEAM_LOCALIZATION": seam_localization,
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": str(_SEAM_DIAGNOSTIC_ROUNDS),
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": (
            f"{run_root}/p38_diagnostic_round"
        ),
        "CANON_P38_ROUND_SEAL_REQUEST_DIR": (
            f"{run_root}/p38_round_seal_requests"
        ),
        "CANON_P38_ROUND_SEAL_ACK_DIR": (
            f"{run_root}/p38_round_seal_acks"
        ),
        "CANON_P38_MISMATCH_CAPSULE": f"{run_root}/p38_mismatch_capsule.npz",
        "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "256",
        "CANON_P38_DURABILITY_PROFILE": "p58-seam-v1",
        "CANON_P38_SERVING_CAPTURE_DIR": capture,
        "CANON_P38_REQUEST_JOURNAL": f"{capture}/p38_request_journal.jsonl",
        "CANON_P38_INCIDENT_LEDGER": f"{capture}/p38_incident_ledger.jsonl",
        "CANON_P38_INCIDENT_MIN_PREFIX": str(_SEAM_MIN_POSITION),
        "CANON_P38_INCIDENT_MAX_PREFIX": str(_SEAM_MAX_POSITION),
        "CANON_P38_INCIDENT_MAX_BYTES": str(_SEAM_INCIDENT_MAX_BYTES),
        "CANON_P38_LIVE_SNAPSHOT_INTERVAL_SECONDS": "30",
        "CANON_P38_LIVE_SNAPSHOT_STOP_FILE": f"{run_root}/p38_live.stop",
        "CANON_P38_LIVE_SNAPSHOT_WORKER_LOG": (
            f"{run_root}/p38_live_worker.log"
        ),
        "CANON_P38_LIVE_COLLECT_REQUEST_FILE": (
            f"{run_root}/p38_collect.request"
        ),
        "CANON_P38_LIVE_COLLECT_ACK_FILE": f"{run_root}/p38_collect.ack",
        "CANON_P38_LIVE_COMPLETE_REQUEST_FILE": (
            f"{run_root}/p38_complete.request"
        ),
        "CANON_P38_LIVE_COMPLETE_ACK_FILE": f"{run_root}/p38_complete.ack",
        "CANON_P38_SERVING_CAPTURE_MAX_CALLS": "4",
        "CANON_P38_SERVING_CAPTURE_MIN_PREFIX": str(_SEAM_CAPTURE_BOUNDS[0]),
        "CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS": ",".join(
            map(str, _SEAM_CAPTURE_BOUNDS)
        ),
        "CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER": "5",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_RECORDS": "4",
        "CANON_P38_MIN_ACTION_KV": str(_SEAM_MIN_POSITION),
        "CANON_P38_SERVING_CAPTURE_CLASSIFICATION": (
            f"{run_root}/p38_serving_capture.classification.json"
        ),
        "CANON_P38_SERVING_CAPTURE_ARCHIVE": (
            f"{run_root}/p38_serving_capture.tar"
        ),
        "CANON_P38_GCS_PREFIX": (
            "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p58/"
            f"{name}/attempt-0"
        ),
        "CANON_P38_SEAM_OBSERVER": "layer",
        "CANON_P38_SEAM_OBSERVER_DIR": capture,
        "CANON_P38_SEAM_MIN_POSITION": str(_SEAM_MIN_POSITION),
        "CANON_P38_SEAM_MAX_POSITION": str(_SEAM_MAX_POSITION),
        "CANON_P38_SEAM_MAX_BYTES": str(_SEAM_MAX_BYTES),
        "CANON_P38_SEAM_CLASSIFICATION": (
            f"{run_root}/p58_seam.classification.json"
        ),
        "CANON_P38_TAIL_OBSERVER": "1",
        "CANON_P38_TAIL_MAX_BYTES": str(_SEAM_TAIL_MAX_BYTES),
    })

  worker = p34._worker(document)
  worker["completions"] = WORKERS
  worker["parallelism"] = WORKERS
  worker_template_metadata = worker["template"].setdefault("metadata", {})
  worker_template_annotations = worker_template_metadata.get("annotations", {})
  worker_template_annotations.pop(_EXCLUSIVE_TOPOLOGY_ANNOTATION, None)
  if not worker_template_annotations:
    worker_template_metadata.pop("annotations", None)
  worker_pod = worker["template"]["spec"]
  if worker_nodepool in _KUEUE_MANAGED_WORKER_POOLS:
    # JobSet-level exclusive topology coordinates the selected/NAP-created
    # node pool across all indexed followers.  A Pod-template annotation does
    # not provide that context and caused K03's follower webhook rejection.
    worker_pod["nodeSelector"].pop("cloud.google.com/gke-nodepool", None)
  else:
    worker_pod["nodeSelector"][
        "cloud.google.com/gke-nodepool"
    ] = worker_nodepool
  worker_pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"] = TOPOLOGY
  worker_container = p34._container(worker_pod["containers"], "pathways-worker")
  p34._replace_arg(
      worker_container["args"],
      "--instance_type=",
      f"--instance_type=tpuv5:{instance_type}",
  )
  address = f"{name}-pathways-head-0-0.{name}"
  p34._replace_arg(
      worker_container["args"],
      "--resource_manager_address=",
      f"--resource_manager_address={address}:29001",
  )
  for item in worker_container.get("env", []):
    if item.get("name") == "PATHWAYS_HEAD" and "value" in item:
      item["value"] = address

  validate(
      document,
      source_commit=source_commit,
      client_image=client_image,
      stage=stage,
      arm=arm,
      worker_nodepool=worker_nodepool,
      instance_type=instance_type,
      sampler_is=sampler_is,
      high_performance=high_performance,
      checked_vma_off_diagnostic=checked_vma_off_diagnostic,
      checked_vma_on_diagnostic=checked_vma_on_diagnostic,
      seam_localization=seam_localization,
  )
  return document


def recipe_signature(document: Mapping[str, Any]) -> dict[str, Any]:
  """Returns only the fields that must be equal across the paired arms."""
  env = p34._env(document)
  omitted_prefixes = (
      "--gold_whitelist=",
      "--metric_logger_dir=",
      "--sampler_is=",
      "--sampler_is_threshold=",
  )
  command = tuple(
      item for item in shlex.split(env["CANON_RUN_CMD"])
      if not item.startswith(omitted_prefixes)
  )
  return {
      "command": command,
      "stage": env["CANON_P34_RUN_STAGE"],
      "deepswe_identity": env["CANON_P34_DEEPSWE"],
      "token_transport": document["metadata"]["labels"][
          _TOKEN_TRANSPORT_LABEL
      ],
      "source_commit": env["CANON_EXPECT_COMMIT"],
      "whitelist_sha256": env["CANON_P34_WHITELIST_SHA256"],
      "optimizer_resident": env["CANON_OPT_STATE_RESIDENT"],
      "optimizer_offload": env["CANON_P30_OPT_STATE_OFFLOAD"],
      "workers": p34._worker(document)["completions"],
  }


def treatment_signature(document: Mapping[str, Any]) -> dict[str, Any]:
  """Returns the explicitly registered numerical treatment fields."""
  env = p34._env(document)
  proxy = p34._container(
      _service_containers(p34._head(document)), "pathways-proxy"
  )
  proxy_xla = [
      item.get("value") for item in proxy.get("env", [])
      if item.get("name") == p34.PROXY_XLA_ENV
  ]
  return {
      "arm": env["CANON_P58_TIM_ARM"],
      "alignment_warning_only": env["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"],
      "proxy_xla": proxy_xla,
      "high_performance": env.get("CANON_V1_HP_FULL", "0"),
      "checked_vma_diagnostic": env.get(
          "CANON_P58_CHECKED_VMA_DIAGNOSTIC", ""
      ),
      "seam_localization": env.get("CANON_P58_SEAM_LOCALIZATION", ""),
      "disable_sampler_is": env["CANON_P34_DISABLE_SAMPLER_IS"],
      "disable_tis": env["CANON_P34_DISABLE_TIS"],
      "sampler_is": tuple(
          item
          for item in shlex.split(env["CANON_RUN_CMD"])
          if item.startswith(("--sampler_is=", "--sampler_is_threshold="))
      ),
  }


def validate(
    document: Mapping[str, Any],
    *,
    source_commit: str,
    client_image: str,
    stage: str,
    arm: str,
    worker_nodepool: str,
    instance_type: str = TOPOLOGY,
    sampler_is: bool = False,
    high_performance: bool = False,
    checked_vma_off_diagnostic: bool = False,
    checked_vma_on_diagnostic: bool = False,
    seam_localization: str = "",
) -> None:
  if stage not in _STAGE_STEPS or arm not in _ARMS:
    raise ValueError("invalid P58 stage or arm")
  if checked_vma_off_diagnostic and checked_vma_on_diagnostic:
    raise ValueError("P58 checked-VMA diagnostic selectors are mutually exclusive")
  if seam_localization not in _SEAM_LOCALIZATION_MODES:
    raise ValueError("P58 seam localization must be empty or coarse")
  checked_vma_diagnostic = (
      "off" if checked_vma_off_diagnostic else
      "on" if checked_vma_on_diagnostic else ""
  )
  hp_bundle = high_performance or bool(checked_vma_diagnostic) or bool(
      seam_localization
  )
  zero_hp_ab_warning = high_performance and arm == "zero" and stage == "full"
  head = p34._head(document)
  cpu_nodepool = head.get("nodeSelector", {}).get(
      "cloud.google.com/gke-nodepool", ""
  )
  worker = p34._worker(document)
  main = p34._container(head["containers"], "jax-tpu")
  env = p34._env(document)
  if _RETIRED_DEVICE_PROBE_TRIGGER in env:
    raise ValueError(
        "P58 must not re-enable the retired Step 65 device probe: "
        f"{_RETIRED_DEVICE_PROBE_TRIGGER}"
    )
  expected_failure_policy = {
      "maxRestarts": 0,
      "restartStrategy": "Recreate",
  }
  if document["spec"].get("failurePolicy") != expected_failure_policy:
    raise ValueError("P58 requires exact Attempt-0 failure policy")
  if document["metadata"]["labels"].get(
      "canon.zero-tim/fixed-lm-head"
  ) != ("1" if hp_bundle else "0"):
    raise ValueError("P58 fixed lm-head label drifted from the selected bundle")
  if document["metadata"]["labels"].get(
      _TOKEN_TRANSPORT_LABEL
  ) != _TOKEN_TRANSPORT:
    raise ValueError("P58 DeepSWE token transport must be TiTO")
  if cpu_nodepool != _CPU_NODEPOOL:
    raise ValueError("P58 CPU head lost the admitted cpu-np node pool")
  if (
      head.get("hostNetwork") is not True
      or head.get("dnsPolicy") != "ClusterFirstWithHostNet"
  ):
    raise ValueError("P58 CPU head must retain the Pathways host network")
  required_anti_affinity = (
      head.get("affinity", {})
      .get("podAntiAffinity", {})
      .get("requiredDuringSchedulingIgnoredDuringExecution", [])
  )
  if _pathways_head_anti_affinity_term() not in required_anti_affinity:
    raise ValueError("P58 CPU head lost required Pathways anti-affinity")
  network = document["spec"].get("network", {})
  if (
      network.get("enableDNSHostnames") is not True
      or network.get("publishNotReadyAddresses") is not True
  ):
    raise ValueError("P58 Pathways routing requires JobSet Pod DNS")
  if worker["backoffLimit"] != 0 or worker["completions"] != WORKERS or worker["parallelism"] != WORKERS:
    raise ValueError("P58 worker count does not match 4x4x8")
  if main["image"] != client_image or not p34._DIGEST_IMAGE.fullmatch(main["image"]):
    raise ValueError("P58 client image is not digest-pinned")
  expected = {
      "CANON_EXPECT_COMMIT": source_commit,
      "CANON_PROFILE_FILE": HP_PROFILE if hp_bundle else PROFILE,
      "CANON_P34_DEEPSWE": "1",
      "CANON_P34_RUN_STAGE": stage,
      "CANON_P34_NO_COMMIT": "0",
      "CANON_P58_DEEPSWE_TIM": "1",
      "CANON_P58_TIM_ADMITTED": "1",
      "CANON_P58_TIM_ARM": arm,
      "CANON_P34_DISABLE_SAMPLER_IS": "0" if sampler_is else "1",
      "CANON_P34_DISABLE_TIS": "0" if sampler_is else "1",
      "CANON_P58_EXPECTED_UPDATES": str(_STAGE_STEPS[stage]),
      "CANON_V1_HP_FULL": "1" if hp_bundle else "0",
      "CANON_P38_FIXED_LM_HEAD": "1" if hp_bundle else "0",
      "CANON_P34_CLEAN_ROWS": str(CLEAN_ROWS),
      "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": (
          "1" if arm == "native" or zero_hp_ab_warning else "0"
      ),
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "MIN_TOKEN_BUCKET": "2048",
      "R2E_ACTIVE_DEADLINE_SECONDS": "3300",
      "R2E_K8S_QUEUE_NAME": document["metadata"]["labels"].get(
          _KUEUE_QUEUE_LABEL
      ),
      "NODE_SELECTOR_VAL": cpu_nodepool,
  }
  if checked_vma_diagnostic:
    expected.update({
        "CANON_P58_CHECKED_VMA_DIAGNOSTIC": checked_vma_diagnostic,
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": (
            f"{env['CANON_STATE']}/p38_diagnostic_round"
        ),
    })
  elif "CANON_P58_CHECKED_VMA_DIAGNOSTIC" in env:
    raise ValueError("P58 production render contains a diagnostic selector")
  if seam_localization:
    capture = f"{env['CANON_STATE']}/p38_serving_capture"
    expected.update({
        "CANON_P58_SEAM_LOCALIZATION": seam_localization,
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": str(_SEAM_DIAGNOSTIC_ROUNDS),
        "CANON_P38_DIAGNOSTIC_ROUND_FILE": (
            f"{env['CANON_STATE']}/p38_diagnostic_round"
        ),
        "CANON_P38_ROUND_SEAL_REQUEST_DIR": (
            f"{env['CANON_STATE']}/p38_round_seal_requests"
        ),
        "CANON_P38_ROUND_SEAL_ACK_DIR": (
            f"{env['CANON_STATE']}/p38_round_seal_acks"
        ),
        "CANON_P38_MISMATCH_CAPSULE_MAX_ROWS": "256",
        "CANON_P38_DURABILITY_PROFILE": "p58-seam-v1",
        "CANON_P38_SERVING_CAPTURE_DIR": capture,
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
        "CANON_P38_MIN_ACTION_KV": str(_SEAM_MIN_POSITION),
        "CANON_P38_SEAM_OBSERVER": "layer",
        "CANON_P38_SEAM_OBSERVER_DIR": capture,
        "CANON_P38_SEAM_MIN_POSITION": str(_SEAM_MIN_POSITION),
        "CANON_P38_SEAM_MAX_POSITION": str(_SEAM_MAX_POSITION),
        "CANON_P38_SEAM_MAX_BYTES": str(_SEAM_MAX_BYTES),
        "CANON_P38_TAIL_OBSERVER": "1",
        "CANON_P38_TAIL_MAX_BYTES": str(_SEAM_TAIL_MAX_BYTES),
    })
  elif "CANON_P58_SEAM_LOCALIZATION" in env:
    raise ValueError("P58 production render contains a seam selector")
  wrong = {
      key: env.get(key) for key, value in expected.items()
      if env.get(key) != value
  }
  if wrong:
    raise ValueError(f"P58 rendered environment mismatch: {wrong}")
  optimization_additions = (
      full_system_optimization_additions("deepswe-qwen4b")
      if high_performance
      else {}
  )
  optimization_wrong = {
      key: env.get(key)
      for key, value in optimization_additions.items()
      if env.get(key) != value
  }
  if optimization_wrong:
    raise ValueError(
        "P58 production system-optimization bundle drifted: "
        f"{optimization_wrong}"
    )
  if not high_performance:
    leaked = [
        key for key in FULL_SYSTEM_OPTIMIZATION_ENV_NAMES if key in env
    ]
    if leaked:
      raise ValueError(
          "P58 non-production arm contains system-optimization selectors: "
          f"{leaked}"
      )
  if "CANON_DP_COLLECTIVE_REDUCE" in env:
    raise ValueError("P58 contains an uncertified DP collective reducer")
  unproven_transport_env = (
      "PATHWAYS_HEARTBEAT_TIMEOUT_SEC",
      "IFRT_PROXY_TIMEOUT_SECONDS",
      "GRPC_KEEPALIVE_TIME_MS",
      "GRPC_KEEPALIVE_TIMEOUT_MS",
      "GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS",
  )
  present_transport_env = [
      key for key in unproven_transport_env if key in env
  ]
  if present_transport_env:
    raise ValueError(
        "P58 contains unproven transport keepalive overrides: "
        f"{present_transport_env}"
    )

  args = shlex.split(env["CANON_RUN_CMD"])
  required = (
      "--model_version=Qwen3-4B-Instruct-2507",
      f"--batch_size={GLOBAL_PROMPTS}",
      "--mini_batch_size=8",
      "--train_micro_batch_size=8",
      "--compute_logps_micro_batch_size=8",
      f"--num_generations={GENERATIONS}",
      "--max_response_length=16384",
      "--max_turns=50",
      "--temperature=1.0",
      "--top_p=1.0",
      "--top_k=0",
      f"--seed={FIXED_SEED}",
      "--rollout_mesh_dp=8",
      "--rollout_mesh_tp=8",
      "--train_mesh_dp=8",
      "--train_mesh_tp=8",
      "--rollout_vllm_max_num_seqs=16",
      "--max_num_batched_tokens=256",
      f"--max_concurrency={MAX_CONCURRENCY}",
      "--loss_agg_mode=sequence-mean-token-scale",
      "--loss_scale_factor=16384",
      "--loss_denominator_weighted_accumulation",
      "--use_rollout_logps",
      "--overlong_filter",
      f"--expected_filtered_rows={CLEAN_ROWS}",
      f"--max_steps={_STAGE_STEPS[stage]}",
      "--no-optimizer-offload",
      "--ckpt_dir=none",
  )
  seed_args = tuple(item for item in args if item.startswith("--seed="))
  if seed_args != (f"--seed={FIXED_SEED}",):
    raise ValueError(
        "P58 command requires exactly one fixed seed: "
        f"expected=--seed={FIXED_SEED} actual={seed_args}"
    )
  checkpoint_args = tuple(
      item for item in args if item.startswith("--ckpt_dir=")
  )
  if checkpoint_args != ("--ckpt_dir=none",):
    raise ValueError(
        "P58 precomputed-gradient training requires exactly one "
        f"--ckpt_dir=none argument: actual={checkpoint_args}"
    )
  checkpoint_cadence_args = tuple(
      item for item in args
      if item.startswith(("--save_interval_steps=", "--max_to_keep="))
  )
  if checkpoint_cadence_args:
    raise ValueError(
        "P58 checkpoint-disabled command contains checkpoint cadence: "
        f"{checkpoint_cadence_args}"
    )
  missing = [item for item in required if item not in args]
  if missing:
    raise ValueError(f"P58 command lost signed fields: {missing}")
  status_index = args.index("--filter_statuses")
  if tuple(args[status_index + 1:status_index + 1 + len(_FILTER_STATUSES)]) != _FILTER_STATUSES:
    raise ValueError("P58 compact-filter status set drifted")
  sampler_args = tuple(
      item
      for item in args
      if item.startswith(("--sampler_is=", "--sampler_is_threshold="))
  )
  expected_sampler_args = (
      ("--sampler_is=token", "--sampler_is_threshold=2.0")
      if sampler_is
      else ()
  )
  if sampler_args != expected_sampler_args:
    raise ValueError(
        "P58 sampler-IS command drifted: "
        f"expected={expected_sampler_args} actual={sampler_args}"
    )
  forbidden = ("--group_clip_filter_threshold", "--optimizer-offload")
  if any(item == value or item.startswith(value + "=") for item in args for value in forbidden):
    raise ValueError("P58 command enabled an optional algorithm intervention")
  if not env.get("CANON_P58_DEBUG_DIR", "").endswith("/debug"):
    raise ValueError("P58 trajectory journal path is missing")

  services = _service_containers(head)
  proxy = p34._container(services, "pathways-proxy")
  proxy_pins = [
      item for item in proxy.get("env", [])
      if item.get("name") == p34.PROXY_XLA_ENV
  ]
  expected_proxy_pins = (
      [] if arm == "native" else [{"name": p34.PROXY_XLA_ENV, "value": p34.PROXY_XLA_FLAG}]
  )
  if proxy_pins != expected_proxy_pins:
    raise ValueError("P58 proxy precision treatment drifted")
  manager = p34._container(services, "pathways-rm")
  if f"--instance_type=tpuv5:{instance_type}" not in manager["args"]:
    raise ValueError("P58 resource-manager topology drifted")
  worker_pod = worker["template"]["spec"]
  annotations = document.get("metadata", {}).get("annotations", {})
  if annotations.get(_EXCLUSIVE_TOPOLOGY_ANNOTATION) != (
      "cloud.google.com/gke-nodepool"
  ):
    raise ValueError("P58 JobSet lost its exclusive-topology annotation")
  worker_template_annotations = worker["template"].get(
      "metadata", {}
  ).get("annotations", {})
  if _EXCLUSIVE_TOPOLOGY_ANNOTATION in worker_template_annotations:
    raise ValueError(
        "P58 exclusive-topology annotation must not be on the Pod template"
    )
  if (
      worker_pod.get("hostNetwork") is not True
      or worker_pod.get("dnsPolicy") != "ClusterFirstWithHostNet"
  ):
    raise ValueError("P58 TPU workers must retain the host network")
  worker_container = p34._container(
      worker_pod["containers"], "pathways-worker"
  )
  name = document["metadata"]["name"]
  address = f"{name}-pathways-head-0-0.{name}"
  rm_arg = f"--resource_manager_address={address}:29001"
  if rm_arg not in worker_container["args"]:
    raise ValueError("P58 worker lost the signed resource-manager address")
  worker_env = {
      item["name"]: item.get("value")
      for item in worker_container.get("env", [])
  }
  if worker_env.get("PATHWAYS_HEAD") != address:
    raise ValueError("P58 worker PATHWAYS_HEAD lost the JobSet Pod DNS name")
  if worker_pod["nodeSelector"].get("cloud.google.com/gke-tpu-topology") != TOPOLOGY:
    raise ValueError("P58 worker topology drifted")
  actual_worker_pool = worker_pod["nodeSelector"].get(
      "cloud.google.com/gke-nodepool"
  )
  expected_worker_pool = (
      None
      if worker_nodepool in _KUEUE_MANAGED_WORKER_POOLS
      else worker_nodepool
  )
  if actual_worker_pool != expected_worker_pool:
    raise ValueError("P58 worker node-pool affinity drifted")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--base", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-commit", required=True)
  parser.add_argument("--source-branch", default=p34.DEFAULT_SOURCE_BRANCH)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--stage", choices=tuple(_STAGE_STEPS), required=True)
  parser.add_argument("--arm", choices=_ARMS, required=True)
  parser.add_argument("--cpu-nodepool", default=_CPU_NODEPOOL)
  parser.add_argument("--worker-nodepool", required=True)
  parser.add_argument("--instance-type", default=TOPOLOGY)
  parser.add_argument("--model-pvc", default="haoyugao-cpu-np-pvc")
  parser.add_argument("--whitelist", default=CLEAN_WHITELIST)
  parser.add_argument("--whitelist-sha256", default=CLEAN_WHITELIST_SHA256)
  parser.add_argument("--sampler-is", action="store_true")
  parser.add_argument("--high-performance", action="store_true")
  parser.add_argument("--checked-vma-off-diagnostic", action="store_true")
  parser.add_argument("--checked-vma-on-diagnostic", action="store_true")
  parser.add_argument(
      "--seam-localization", choices=("coarse",), default=""
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite JobSet: {args.output}")
  document = render(
      yaml.safe_load(args.base.read_text()),
      source_commit=args.source_commit,
      source_branch=args.source_branch,
      client_image=args.client_image,
      run_id=args.run_id,
      stage=args.stage,
      arm=args.arm,
      cpu_nodepool=args.cpu_nodepool,
      worker_nodepool=args.worker_nodepool,
      instance_type=args.instance_type,
      model_pvc=args.model_pvc,
      whitelist=args.whitelist,
      whitelist_sha256=args.whitelist_sha256,
      sampler_is=args.sampler_is,
      high_performance=args.high_performance,
      checked_vma_off_diagnostic=args.checked_vma_off_diagnostic,
      checked_vma_on_diagnostic=args.checked_vma_on_diagnostic,
      seam_localization=args.seam_localization,
  )
  args.output.write_text(p34.dump_jobset(document))
  recipe = (
      "zero-hp-seam-coarse"
      if args.seam_localization
      else "zero-hp-vmaoff-precheck"
      if args.checked_vma_off_diagnostic
      else "zero-hp-vmaon-precheck"
      if args.checked_vma_on_diagnostic
      else "native-is"
      if args.sampler_is
      else "zero-hp"
      if args.high_performance
      else f"{args.arm}-raw"
  )
  print(
      "P58_DEEPSWE_TIM_RENDER_PASS "
      f"arm={args.arm} stage={args.stage} recipe={recipe} "
      "transport=token-in-token-out "
      f"output={args.output}"
  )


if __name__ == "__main__":
  main()
