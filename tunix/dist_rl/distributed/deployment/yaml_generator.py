# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates Kubernetes deployment YAML manifests from templates."""

import argparse
from collections.abc import Sequence
import json
import math
import os
import re
import string
from typing import Any

_K8S_LABEL_VALUE_RE = re.compile(r"^[A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?$")
_STANDALONE_PLACEHOLDER_RE = re.compile(
    r"^(?P<indent>[ \t]*)\$\{(?P<key>[A-Z0-9_]+)\}[ \t]*\n",
    re.MULTILINE,
)

_PATHWAYS_WORKER_TEMPLATE_ENV = frozenset({
    "RAIDEN_BROADCAST_K",
    "ENABLE_MULTI_NUMA",
    "TPU_RAIDEN_DATA_NICS",
    "TPU_MIN_LOG_LEVEL",
    "TF_CPP_MIN_LOG_LEVEL",
    "XCLOUD_ENVIRONMENT",
    "MEGASCALE_GRPC_ENABLE_XOR_TRACER",
    "MEGASCALE_NUM_SLICES",
    "JOBSET_NAME",
    "REPLICATED_JOB_NAME",
    "MEGASCALE_SLICE_ID",
    "PATHWAYS_HEAD",
    "MEGASCALE_COORDINATOR_ADDRESS",
})

_WORKER_NONZERO_EXIT_GUARD = """\
                  if [ "$EXIT_CODE" -eq 0 ]; then
                    echo "ERROR: Worker exited with code 0 unexpectedly; forcing exit code 1." >&2
                    EXIT_CODE=1
                  fi"""


def _substitute_template(
    template_text: str, substitutions: dict[str, Any]
) -> str:
  """Substitutes template variables, aligning or dropping standalone lines."""

  def _replace_standalone(match: re.Match[str]) -> str:
    key = match.group("key")
    if key not in substitutions:
      return match.group(0)
    indent = match.group("indent")
    val = substitutions[key]
    raw = "" if val is None else str(val).strip("\n")
    if not raw.strip():
      return ""
    lines = raw.splitlines()
    if indent and not all(
        not line or line.startswith(indent) for line in lines
    ):
      formatted = (
          "\n".join((indent + line if line else "") for line in lines) + "\n"
      )
    else:
      formatted = raw + "\n"
    return formatted.replace("$", "$$")

  expanded = _STANDALONE_PLACEHOLDER_RE.sub(_replace_standalone, template_text)
  return string.Template(expanded).substitute(substitutions)


def _env_bool(name: str) -> bool:
  return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def validate_gang_id(gang_id: str | None) -> None:
  """Validates gang_id against Kubernetes label value syntax rules."""
  if not gang_id:
    return
  if len(gang_id) > 63 or not _K8S_LABEL_VALUE_RE.fullmatch(gang_id):
    raise ValueError(
        f"Invalid gang_id {gang_id!r}: must be a valid Kubernetes label value "
        "(<=63 chars, alphanumeric start/end, [-A-Za-z0-9_.])."
    )


def render_gang_labels(gang_id: str | None) -> str:
  """Renders the gang-id label line if gang_id is provided."""
  if not gang_id:
    return ""
  validate_gang_id(gang_id)
  return f'    scheduling.x-k8s.io/gang-id: "{gang_id}"\n'


def parse_head_excluded_nodepools(value: str) -> tuple[str, ...]:
  """Parses `--head_excluded_nodepools` into validated GKE node pool names.

  Args:
    value: Comma-separated node pool names; empty for none.

  Returns:
    The node pool names, in the given order.

  Raises:
    ValueError: If a name is not a valid Kubernetes label value or repeats.
  """
  if not value:
    return ()
  pools = tuple(pool.strip() for pool in value.split(","))
  for pool in pools:
    if len(pool) > 63 or not _K8S_LABEL_VALUE_RE.fullmatch(pool):
      raise ValueError(
          f"Invalid --head_excluded_nodepools entry {pool!r} in {value!r}: must"
          " be a valid Kubernetes label value (<=63 chars, alphanumeric"
          " start/end, [-A-Za-z0-9_.])."
      )
  if len(set(pools)) != len(pools):
    raise ValueError(f"--head_excluded_nodepools repeats a name: {value!r}")
  return pools


def render_colocated_head_affinity(
    jobset_name: str, excluded_nodepools: tuple[str, ...]
) -> str:
  """Renders the affinity of a Pathways head pod that runs on a pw-node host.

  The head pod requests no `google.com/tpu`. Kueue matches a pod set against a
  ResourceFlavor using only the node selector keys the flavor lists in its
  `nodeLabels`, so the head pod's TPU `nodeSelector` does not rule out a CPU
  flavor that lists only `cloud.google.com/gke-nodepool`. If such a flavor comes
  before the TPU flavor in the ClusterQueue, Kueue admits the head pod to it and
  adds that node pool to its `nodeSelector`, which no pw-node host matches.
  Listing those node pools in `excluded_nodepools` makes the CPU flavors fail to
  match.

  Args:
    jobset_name: Name of the Pathways JobSet.
    excluded_nodepools: GKE node pools the head pod must not be admitted to.

  Returns:
    The `affinity` block of the head pod spec.
  """
  node_affinity = ""
  if excluded_nodepools:
    node_affinity = (
        "  nodeAffinity:\n"
        "    requiredDuringSchedulingIgnoredDuringExecution:\n"
        "      nodeSelectorTerms:\n"
        "      - matchExpressions:\n"
        "        - key: cloud.google.com/gke-nodepool\n"
        "          operator: NotIn\n"
        "          values:\n"
        + "".join(
            f"          - {json.dumps(pool)}\n" for pool in excluded_nodepools
        )
    )
  return (
      "affinity:\n"
      f"{node_affinity}"
      "  podAffinity:\n"
      "    requiredDuringSchedulingIgnoredDuringExecution:\n"
      "    # place on one of the pathways-worker nodes\n"
      "    - topologyKey: kubernetes.io/hostname\n"
      "      labelSelector:\n"
      "        matchExpressions:\n"
      "        - key: jobset.sigs.k8s.io/jobset-name\n"
      "          operator: In\n"
      "          values:\n"
      f"          - {jobset_name}\n"
      "        - key: jobset.sigs.k8s.io/replicatedjob-name\n"
      "          operator: In\n"
      "          values:\n"
      "          - pw-node"
  )


def render_teardown_on_exit(
    teardown_gang_on_exit: bool,
    gang_id: str | None,
    namespace: str,
    jobset_name: str,
) -> str:
  """Renders a shell block for the CPU orchestrator to delete sibling JobSets."""
  if not teardown_gang_on_exit:
    return ""
  if not gang_id:
    raise ValueError(
        "--teardown_gang_on_exit requires --gang_id (or GANG_ID) to be set."
    )
  validate_gang_id(gang_id)
  # A failed delete only warns, so the orchestrator keeps its own exit code.
  return (
      f'echo "Orchestrator finished (exit=${{EXIT_CODE}});'
      f' tearing down sibling JobSets in gang {gang_id}..."\n'
      "if ! kubectl delete jobsets.jobset.x-k8s.io"
      f' -n "{namespace}"'
      f' -l "scheduling.x-k8s.io/gang-id={gang_id}"'
      f' --field-selector="metadata.name!={jobset_name}"'
      " --ignore-not-found=true --wait=false; then\n"
      f'  echo "WARNING: Failed to tear down sibling JobSets in gang {gang_id}'
      ' (check kubectl and ServiceAccount RBAC)." >&2\n'
      "fi"
  )


def render_fail_fast_fields(
    fail_fast: bool, startup_retries: int
) -> dict[str, str]:
  """Renders JobSet/Job fail-fast and startup retry fields."""
  if startup_retries < 0:
    raise ValueError(f"--startup_retries must be >= 0, got {startup_retries}")
  if not fail_fast:
    return {
        "STARTUP_RETRIES": str(startup_retries),
        "BACKOFF_LIMIT": "2048000",
        "RESTART_POLICY": "OnFailure",
        "WORKER_NONZERO_EXIT_GUARD": "",
    }
  return {
      "STARTUP_RETRIES": str(startup_retries),
      "BACKOFF_LIMIT": "0",
      "RESTART_POLICY": "Never",
      "WORKER_NONZERO_EXIT_GUARD": _WORKER_NONZERO_EXIT_GUARD,
  }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description="Generate Kubernetes deployment YAML from template."
  )

  parser.add_argument("template_file", help="Path to the template file")

  parser.add_argument("--jobset_name", default=None, help="Name of the jobset")

  parser.add_argument(
      "--tpu_slice",
      default=None,
      help=(
          "TPU type and tpu_topology (e.g. tpu7x:4x4x8). Supported TPU types:"
          " tpu7x, tpuv5 (tpu-v5p-slice), tpuv5e (tpu-v5-lite-podslice), tpuv6e"
          " (tpu-v6e-slice), tpuv6ea (tpu-v6ea-slice)."
      ),
  )
  parser.add_argument(
      "--cpu_machine",
      default=None,
      help="CPU machine type (e.g. n2-standard-64)",
  )
  parser.add_argument(
      "--cpu_nodepool",
      default=os.environ.get("CPU_NODEPOOL") or None,
      help="GKE nodepool for CPU jobs (e.g. cpu-np, cpu-highmem-np).",
  )
  parser.add_argument(
      "--cpu_memory",
      default=os.environ.get("CPU_MEMORY") or None,
      help="Memory request for CPU jobs (e.g. 240G, 300Gi).",
  )
  parser.add_argument(
      "--namespace",
      default=os.environ.get("K8S_NAMESPACE", "default"),
      help="Kubernetes namespace to deploy into.",
  )
  parser.add_argument(
      "--queue_name",
      default=os.environ.get("KUEUE_QUEUE_NAME", ""),
      help="Kueue local queue name for scheduling (optional).",
  )
  parser.add_argument(
      "--preemptible",
      action=argparse.BooleanOptionalAction,
      default=None,
      help=(
          'Add scheduling.x-k8s.io/preemptible: "true" label to JobSet'
          " metadata (or set PREEMPTIBLE=true)."
      ),
  )
  parser.add_argument(
      "--gang_id",
      default=os.environ.get("GANG_ID"),
      help=(
          "Per-run unique identifier shared across all JobSets in a single RL"
          " run (emits scheduling.x-k8s.io/gang-id JobSet label; must be unique"
          " per concurrent run in the namespace so teardown only deletes"
          " sibling JobSets of the same run)."
      ),
  )
  parser.add_argument(
      "--service_account",
      default=os.environ.get("SERVICE_ACCOUNT") or None,
      help="Kubernetes service account for pods.",
  )
  parser.add_argument(
      "--priority_class",
      default=os.environ.get("PRIORITY_CLASS")
      or os.environ.get("KUEUE_PRIORITY_CLASS")
      or None,
      help="Kubernetes priority class for pods.",
  )
  parser.add_argument(
      "--use_dynamic_slicing",
      action=argparse.BooleanOptionalAction,
      default=None,
      help="Enable GKE dynamic slicing annotations and topology selectors.",
  )
  parser.add_argument(
      "--head_nodepool",
      default=None,
      help="Kubernetes nodepool for Pathways head pod (e.g. cpu-np).",
  )
  parser.add_argument(
      "--head_excluded_nodepools",
      default="",
      help=(
          "Comma-separated GKE node pools that the Pathways head pod, which"
          " runs on a pw-node host when --use_dynamic_slicing is off, must not"
          " be admitted to. Set it to the node pools of the Kueue CPU"
          " ResourceFlavors listed before the TPU flavor (see"
          " `render_colocated_head_affinity`)."
      ),
  )
  parser.add_argument(
      "--termination_grace_seconds",
      default=(
          int(os.environ["TERMINATION_GRACE_SECONDS"])
          if os.environ.get("TERMINATION_GRACE_SECONDS")
          else None
      ),
      type=int,
      help=(
          "Termination grace period in seconds for pods (default: 300 for CPU"
          " orchestrator, 10 for TPU/Pathways templates)."
      ),
  )
  parser.add_argument(
      "--teardown_gang_on_exit",
      action="store_true",
      default=None,
      help=(
          "When rendering jobset.cpu.yaml with --gang_id, append a kubectl"
          " delete of sibling JobSets in the same gang after the orchestrator"
          " exits (or set TEARDOWN_GANG_ON_EXIT=1)."
      ),
  )
  parser.add_argument(
      "--fail_fast",
      action="store_true",
      default=None,
      help=(
          "Disable in-place pod restarts (backoffLimit=0, restartPolicy=Never)"
          " and map unexpected zero worker exits to 1 (or set FAIL_FAST=1)."
      ),
  )
  parser.add_argument(
      "--startup_retries",
      type=int,
      default=int(os.environ.get("STARTUP_RETRIES") or "3"),
      help=(
          "JobSet failurePolicy.maxRestarts (default: STARTUP_RETRIES or 3):"
          " whole-JobSet recreations over the JobSet's lifetime, not only at"
          " startup. Trainer/reference workers can re-register only before"
          " bring_up; with --fail_fast, use --gang_id and"
          " --teardown_gang_on_exit so the orchestrator's exit deletes the"
          " sibling JobSets."
      ),
  )

  parser.add_argument(
      "--pathways_server_image",
      default=(
          "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:latest"
      ),
      help="Pathways server image",
  )
  parser.add_argument(
      "--pathways_proxy_server_image",
      default=(
          "us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/"
          "proxy_server:latest"
      ),
      help="Pathways proxy server image",
  )
  parser.add_argument(
      "--pathways_gcs_scratch_location",
      default="gs://cloud-pathways-staging/tmp",
      help="GCS scratch location",
  )
  parser.add_argument(
      "--pathways_proxy_memory_limit",
      default="100G",
      help="Memory limit of the Pathways proxy container",
  )
  parser.add_argument(
      "--pathways_proxy_memory",
      default="16G",
      help=(
          "Memory request for the Pathways proxy container. Kept well below"
          " --pathways_proxy_memory_limit: requests are the scheduling floor,"
          " and the head pod is co-located with pw-node via podAffinity, so"
          " requesting the full limit over-reserves the node."
      ),
  )
  parser.add_argument(
      "--pathways_rm_memory",
      default="4G",
      help="Memory request for the pathways-rm container",
  )
  parser.add_argument(
      "--user_container_memory",
      default="48G",
      help="Memory request for the user/worker container",
  )
  parser.add_argument(
      "--user_container_memory_limit",
      default="70G",
      help="Memory limit for the user/worker container",
  )
  parser.add_argument(
      "--pathways_worker_memory",
      default="100G",
      help="Memory request for the pathways-worker container",
  )
  parser.add_argument(
      "--ifrt_proxy_large_transfer_threshold_bytes",
      type=int,
      default=2**63 - 1,
      help=(
          "Pathways templates: host buffers of at least this many bytes go"
          " from the user container to pathways-proxy through a shared"
          " memory-backed directory instead of localhost gRPC (a 31 GB batch"
          " took ~1-1.6 s instead of ~6-10 s on v5p). Each such buffer also"
          " takes memory in the user container until it has been sent to the"
          " workers. The default, like the IFRT proxy client's, turns this"
          " off."
      ),
  )

  parser.add_argument(
      "--worker_container_name",
      default="main",
      help="Name of the worker container",
  )
  parser.add_argument(
      "--worker_container_image",
      default="python:3.12",
      help="Image of the worker container",
  )
  parser.add_argument(
      "--worker_container_port",
      type=int,
      default=12345,
      help="gRPC port of the worker container",
  )
  parser.add_argument(
      "--worker_startup_command",
      default="sleep infinity",
      help="Command to run on startup",
  )
  parser.add_argument(
      "--ray_head_pre_init_command",
      default="",
      help="Command to run on the Ray head pod before starting Ray.",
  )
  parser.add_argument(
      "--ray_worker_pre_init_command",
      default="",
      help="Command to run on Ray worker pods before joining the Ray cluster.",
  )

  args = parser.parse_args(argv)
  if not 0 < args.ifrt_proxy_large_transfer_threshold_bytes <= 2**63 - 1:
    # The IFRT proxy client parses the value as a positive int64.
    parser.error(
        "--ifrt_proxy_large_transfer_threshold_bytes must be in [1, 2**63 - 1],"
        f" got {args.ifrt_proxy_large_transfer_threshold_bytes}"
    )
  return args


def build_substitutions(
    args: argparse.Namespace, template_text: str = ""
) -> dict[str, Any]:
  """Computes substitution dictionary for template rendering."""
  tpu_type = None
  tpu_topology = None
  num_chips = None
  tpu_machine = None
  slice_topology = None
  slice_size = None
  pw_instance_type = None
  if args.tpu_slice and args.tpu_slice != ":":
    tpu_type, tpu_topology = args.tpu_slice.split(":")
    num_chips = math.prod([int(d) for d in tpu_topology.split("x")])
    assert num_chips >= 4 and num_chips % 4 == 0

    if tpu_type in ("tpu7x", "tpu-v7x-slice"):
      slice_topology = tpu_topology if num_chips <= 64 else "4x4x4"
      slice_size = num_chips // 4 if num_chips <= 64 else 16
      tpu_machine = "tpu7x-standard-4t"
      tpu_type = "tpu7x"
      pw_instance_type = "tpu7x"
    elif tpu_type in ("tpuv5", "tpuv5p", "tpu-v5p-slice"):
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct5p-hightpu-4t"
      tpu_type = "tpu-v5p-slice"
      pw_instance_type = "tpuv5"
    elif tpu_type in ("tpuv5e", "tpu-v5-lite-podslice"):
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct5lp-hightpu-4t"
      tpu_type = "tpu-v5-lite-podslice"
      pw_instance_type = "tpuv5e"
    elif tpu_type in ("tpuv6e", "tpu-v6e-slice"):
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct6e-standard-4t"
      tpu_type = "tpu-v6e-slice"
      pw_instance_type = "tpuv6e"
    elif tpu_type in ("tpuv6ea", "tpu-v6ea-slice"):
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct6ea-standard-4t"
      tpu_type = "tpu-v6ea-slice"
      pw_instance_type = "tpuv6ea"
    else:
      raise ValueError(f"Unsupported TPU type {tpu_type}")

  jobset_name = args.jobset_name
  if jobset_name is None:
    jobset_name = f"{os.environ.get('USER')}-{pw_instance_type}-{num_chips}"

  if args.preemptible is not None:
    preemptible = args.preemptible
  else:
    preemptible = _env_bool("PREEMPTIBLE")

  labels: list[str] = []
  if args.queue_name:
    labels.append(f"    kueue.x-k8s.io/queue-name: {args.queue_name}\n")
  if preemptible:
    labels.append('    scheduling.x-k8s.io/preemptible: "true"\n')
  if args.gang_id:
    labels.append(render_gang_labels(args.gang_id))
  queue_label = "  labels:\n" + "".join(labels) if labels else ""

  supports_fail_fast = (
      not template_text or "${WORKER_NONZERO_EXIT_GUARD}" in template_text
  )
  fail_fast = (
      bool(args.fail_fast)
      if args.fail_fast is not None
      else (_env_bool("FAIL_FAST") and supports_fail_fast)
  )
  supports_teardown = (
      not template_text or "${TEARDOWN_ON_EXIT}" in template_text
  )
  teardown_gang_on_exit = (
      bool(args.teardown_gang_on_exit)
      if args.teardown_gang_on_exit is not None
      else (_env_bool("TEARDOWN_GANG_ON_EXIT") and supports_teardown)
  )

  if args.cpu_nodepool:
    cpu_node_selector = f"cloud.google.com/gke-nodepool: {args.cpu_nodepool}"
  else:
    cpu_node_selector = f"node.kubernetes.io/instance-type: {args.cpu_machine}"

  if args.cpu_memory:
    cpu_resources = (
        "resources:\n"
        "  requests:\n"
        '    cpu: "24"\n'
        f"    memory: {args.cpu_memory}\n"
        "    ephemeral-storage: 20Gi"
    )
  else:
    cpu_resources = ""

  priority_class_line = (
      f"priorityClassName: {args.priority_class}" if args.priority_class else ""
  )
  service_account_line = (
      f"serviceAccountName: {args.service_account}"
      if args.service_account
      else ""
  )

  if args.termination_grace_seconds is not None:
    termination_grace_seconds = args.termination_grace_seconds
  elif os.path.basename(args.template_file or "") == "jobset.cpu.yaml":
    termination_grace_seconds = 300
  else:
    termination_grace_seconds = 10

  reservation_name = os.environ.get("TPU_RESERVATION", "").strip()
  reservation_selector = (
      f"cloud.google.com/reservation-name: {reservation_name}"
      if reservation_name
      else ""
  )

  if args.use_dynamic_slicing is not None:
    use_dynamic_slicing = args.use_dynamic_slicing
  else:
    use_dynamic_slicing = _env_bool("USE_DYNAMIC_SLICING")

  explicit_head_nodepool = (
      args.head_nodepool or os.environ.get("HEAD_NODEPOOL", "").strip()
  )
  if explicit_head_nodepool and not use_dynamic_slicing:
    raise ValueError(
        "--head_nodepool (or HEAD_NODEPOOL) requires --use_dynamic_slicing."
    )
  head_excluded_nodepools = parse_head_excluded_nodepools(
      args.head_excluded_nodepools
  )
  if head_excluded_nodepools and use_dynamic_slicing:
    raise ValueError(
        "--head_excluded_nodepools applies only without"
        " --use_dynamic_slicing; with it, set --head_nodepool."
    )

  if use_dynamic_slicing:
    if explicit_head_nodepool:
      head_nodepool = explicit_head_nodepool
      head_node_selector = f"cloud.google.com/gke-nodepool: {head_nodepool}"
    elif args.cpu_machine:
      head_nodepool = ""
      head_node_selector = (
          f"node.kubernetes.io/instance-type: {args.cpu_machine}"
      )
    else:
      head_nodepool = "cpu-np"
      head_node_selector = f"cloud.google.com/gke-nodepool: {head_nodepool}"
    head_affinity = ""
    toleration_op = "Equal" if head_nodepool else "Exists"
    toleration_val = f'  value: "{head_nodepool}"\n' if head_nodepool else ""
    head_tolerations = (
        "tolerations:\n"
        '- key: "cloud.google.com/gke-nodepool"\n'
        f'  operator: "{toleration_op}"\n'
        f"{toleration_val}"
        '  effect: "NoSchedule"'
    )
    tpu_topology_selector = ""
  else:
    head_selector_lines = [
        f"cloud.google.com/gke-tpu-accelerator: {tpu_type}",
        f"cloud.google.com/gke-tpu-topology: {tpu_topology}",
    ]
    if reservation_selector:
      head_selector_lines.append(reservation_selector)
    head_node_selector = "\n".join(head_selector_lines)
    head_affinity = render_colocated_head_affinity(
        jobset_name, head_excluded_nodepools
    )
    head_tolerations = ""
    tpu_topology_selector = (
        f"cloud.google.com/gke-tpu-topology: {tpu_topology}"
        if tpu_topology
        else ""
    )

  if use_dynamic_slicing and slice_topology:
    if slice_size and slice_size > 1:
      anno_lines = [
          f'cloud.google.com/gke-tpu-slice-topology: "{tpu_topology}"',
          'cloud.google.com/skip-tpu-webhook-check: "true"',
          (
              "kueue.x-k8s.io/podset-required-topology:"
              " cloud.google.com/gce-topology-block"
          ),
          (
              "kueue.x-k8s.io/podset-slice-required-topology:"
              f" cloud.google.com/gke-tpu-partition-{slice_topology}-id"
          ),
          f'kueue.x-k8s.io/podset-slice-size: "{slice_size}"',
      ]
    else:
      anno_lines = [
          f'cloud.google.com/gke-tpu-slice-topology: "{slice_topology}"',
          'cloud.google.com/skip-tpu-webhook-check: "true"',
      ]
    tpu_annotations = "\n".join(anno_lines)
  else:
    tpu_annotations = ""

  if use_dynamic_slicing:
    if slice_size and slice_size > 1:
      pw_node_affinity = (
          "affinity:\n"
          "  nodeAffinity:\n"
          "    requiredDuringSchedulingIgnoredDuringExecution:\n"
          "      nodeSelectorTerms:\n"
          "      - matchExpressions:\n"
          "        - key:"
          f" cloud.google.com/gke-tpu-partition-{slice_topology}-state\n"
          "          operator: In\n"
          '          values: ["HEALTHY", "DEGRADED"]'
      )
      tpu_affinity = pw_node_affinity
    else:
      pw_node_affinity = ""
      tpu_affinity = ""
  else:
    pw_node_affinity = (
        "affinity:\n"
        "  # place on nodes in the same nodepool\n"
        "  podAffinity:\n"
        "    requiredDuringSchedulingIgnoredDuringExecution:\n"
        "    - topologyKey: cloud.google.com/gke-nodepool\n"
        "      labelSelector:\n"
        "        matchExpressions:\n"
        "        - key: jobset.sigs.k8s.io/jobset-name\n"
        "          operator: In\n"
        "          values:\n"
        f"          - {jobset_name}\n"
        "  # ensure exclusive access to the nodepool\n"
        "  podAntiAffinity:\n"
        "    requiredDuringSchedulingIgnoredDuringExecution:\n"
        "    - topologyKey: cloud.google.com/gke-nodepool\n"
        "      labelSelector:\n"
        "        matchExpressions:\n"
        "        - key: jobset.sigs.k8s.io/jobset-name\n"
        "          operator: Exists\n"
        "        - key: jobset.sigs.k8s.io/jobset-name\n"
        "          operator: NotIn\n"
        "          values:\n"
        f"          - {jobset_name}"
    )
    tpu_affinity = ""

  sidecar_image = os.environ.get("COLOCATED_PYTHON_SIDECAR_IMAGE", "").strip()
  sidecar_shm = os.environ.get(
      "COLOCATED_PYTHON_SIDECAR_SHM", "1"
  ).strip().lower() not in ("0", "false", "no")
  sidecar_memory = os.environ.get(
      "COLOCATED_PYTHON_SIDECAR_MEMORY", "16Gi"
  ).strip()
  if sidecar_shm:
    worker_shm_mount = (
        "  - mountPath: /tmp/sidecar\n"
        "    name: sidecar-shared-memory\n"
    )
    sidecar_shm_env = (
        "\n  - name: CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY\n"
        "    value: /tmp/sidecar"
    )
    sidecar_volume_mount = (
        "  - mountPath: /tmp/sidecar\n"
        "    name: sidecar-shared-memory"
    )
  else:
    worker_shm_mount = ""
    sidecar_shm_env = ""
    sidecar_volume_mount = (
        "  - mountPath: /tmp\n"
        "    name: shared-tmp"
    )
  if sidecar_image:
    colocated_python_sidecar_block = (
        f"{worker_shm_mount}"
        "initContainers:\n"
        "- name: colocated-python-sidecar\n"
        f"  image: {sidecar_image}\n"
        "  imagePullPolicy: Always\n"
        "  env:\n"
        "  - name: GRPC_SERVER_ADDRESS\n"
        f'    value: "0.0.0.0:50051"{sidecar_shm_env}\n'
        "  ports:\n"
        "  - containerPort: 50051\n"
        "    protocol: TCP\n"
        "  resources:\n"
        "    requests:\n"
        '      cpu: "4"\n'
        f"      memory: {sidecar_memory}\n"
        "  restartPolicy: Always\n"
        "  volumeMounts:\n"
        f"{sidecar_volume_mount}"
    )
    colocated_python_proxy_args = "- --sidecar_name=external"
  else:
    colocated_python_sidecar_block = ""
    colocated_python_proxy_args = ""
  if sidecar_image and sidecar_shm:
    colocated_python_worker_args = (
        "- --cloud_pathways_sidecar_shm_directory=/tmp/sidecar"
    )
    colocated_python_sidecar_volume = (
        "- name: sidecar-shared-memory\n"
        "  emptyDir:\n"
        "    medium: Memory"
    )
  else:
    colocated_python_worker_args = ""
    colocated_python_sidecar_volume = ""

  worker_env = {}
  for line in (
      os.environ.get("PATHWAYS_WORKER_EXTRA_ENV", "").strip().splitlines()
  ):
    line = line.strip()
    if not line:
      continue
    if "=" not in line:
      raise ValueError(
          f"Invalid PATHWAYS_WORKER_EXTRA_ENV entry {line!r}; expected"
          " KEY=VALUE"
      )
    key, value = line.split("=", 1)
    key = key.strip()
    if not key:
      raise ValueError(
          f"Invalid PATHWAYS_WORKER_EXTRA_ENV entry {line!r}; empty key"
      )
    if key in _PATHWAYS_WORKER_TEMPLATE_ENV:
      raise ValueError(
          f"PATHWAYS_WORKER_EXTRA_ENV key {key!r} conflicts with a"
          " template-defined environment variable; configure"
          " RAIDEN_BROADCAST_K, ENABLE_MULTI_NUMA, or TPU_RAIDEN_DATA_NICS"
          " directly via environment variables."
      )
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
      value = value[1:-1]
    worker_env[key] = value

  pathways_worker_extra_env = "\n".join(
      f"- name: {key}\n  value: {json.dumps(value)}"
      for key, value in worker_env.items()
  )

  pathways_proxy_extra_args = "\n".join(
      f"- {json.dumps(flag)}"
      for flag in os.environ.get("PATHWAYS_PROXY_EXTRA_ARGS", "").split()
  )

  raiden_broadcast_k = os.environ.get("RAIDEN_BROADCAST_K", "64")
  enable_multi_numa = os.environ.get("ENABLE_MULTI_NUMA", "0")
  tpu_raiden_data_nics = os.environ.get("TPU_RAIDEN_DATA_NICS", "eth0")

  return {
      "JOBSET_NAME": jobset_name,
      "USER": os.environ.get("USER"),
      "NAMESPACE": args.namespace,
      "QUEUE_NAME": args.queue_name,
      "QUEUE_LABEL": queue_label,
      "SERVICE_ACCOUNT_LINE": service_account_line,
      "PRIORITY_CLASS_LINE": priority_class_line,
      "CPU_NODE_SELECTOR": cpu_node_selector,
      "CPU_RESOURCES": cpu_resources,
      "TERMINATION_GRACE_SECONDS": termination_grace_seconds,
      "TPU_TOPOLOGY_SELECTOR": tpu_topology_selector,
      "TPU_ANNOTATIONS": tpu_annotations,
      "TPU_AFFINITY": tpu_affinity,
      "PW_NODE_AFFINITY": pw_node_affinity,
      "HEAD_NODE_SELECTOR": head_node_selector,
      "HEAD_AFFINITY": head_affinity,
      "HEAD_TOLERATIONS": head_tolerations,
      "RESERVATION_SELECTOR": reservation_selector,
      "COLOCATED_PYTHON_SIDECAR_BLOCK": colocated_python_sidecar_block,
      "COLOCATED_PYTHON_SIDECAR_VOLUME": colocated_python_sidecar_volume,
      "COLOCATED_PYTHON_WORKER_ARGS": colocated_python_worker_args,
      "COLOCATED_PYTHON_PROXY_ARGS": colocated_python_proxy_args,
      "PATHWAYS_WORKER_EXTRA_ENV": pathways_worker_extra_env,
      "PATHWAYS_PROXY_EXTRA_ARGS": pathways_proxy_extra_args,
      "RAIDEN_BROADCAST_K": raiden_broadcast_k,
      "ENABLE_MULTI_NUMA": enable_multi_numa,
      "TPU_RAIDEN_DATA_NICS": tpu_raiden_data_nics,
      "TEARDOWN_ON_EXIT": render_teardown_on_exit(
          teardown_gang_on_exit,
          args.gang_id,
          args.namespace,
          jobset_name,
      ),
      **render_fail_fast_fields(fail_fast, args.startup_retries),
      "SERVER_IMAGE": args.pathways_server_image,
      "PROXY_IMAGE": args.pathways_proxy_server_image,
      "GCS_SCRATCH_LOCATION": args.pathways_gcs_scratch_location,
      "PATHWAYS_PROXY_MEMORY_LIMIT": args.pathways_proxy_memory_limit,
      "PATHWAYS_PROXY_MEMORY": args.pathways_proxy_memory,
      "PATHWAYS_RM_MEMORY": args.pathways_rm_memory,
      "USER_CONTAINER_MEMORY": args.user_container_memory,
      "USER_CONTAINER_MEMORY_LIMIT": args.user_container_memory_limit,
      "PATHWAYS_WORKER_MEMORY": args.pathways_worker_memory,
      "IFRT_PROXY_LARGE_TRANSFER_THRESHOLD_BYTES": (
          args.ifrt_proxy_large_transfer_threshold_bytes
      ),
      "CPU_MACHINE": args.cpu_machine,
      "TPU_MACHINE": tpu_machine,
      "TPU_TYPE": tpu_type,
      "TPU_TOPOLOGY": tpu_topology,
      "NUM_CHIPS": num_chips,
      "PW_INSTANCE_TYPE": pw_instance_type,
      "REPLICAS": 1,
      "COMPLETIONS": num_chips // 4 if num_chips else None,
      "PARALLELISM": num_chips // 4 if num_chips else None,
      "PODSET_SLICE_TOPOLOGY": slice_topology,
      "PODSET_SLICE_SIZE": slice_size,
      "USER_CONTAINER": args.worker_container_name,
      "USER_CONTAINER_IMAGE": args.worker_container_image,
      "USER_CONTAINER_PORT": args.worker_container_port,
      "STARTUP_COMMAND": args.worker_startup_command,
      "RAY_HEAD_PRE_INIT_COMMAND": args.ray_head_pre_init_command,
      "RAY_WORKER_PRE_INIT_COMMAND": args.ray_worker_pre_init_command,
  }


def render_template(template_text: str, args: argparse.Namespace) -> str:
  """Validates template compatibility and renders substituted YAML."""
  if args.fail_fast and "${WORKER_NONZERO_EXIT_GUARD}" not in template_text:
    raise ValueError(
        f"--fail_fast is not supported by template {args.template_file}"
    )
  if args.teardown_gang_on_exit and "${TEARDOWN_ON_EXIT}" not in template_text:
    raise ValueError(
        f"--teardown_gang_on_exit is not supported by template"
        f" {args.template_file}"
    )
  is_pathways = "name: pathways-worker" in template_text
  sidecar_image = os.environ.get("COLOCATED_PYTHON_SIDECAR_IMAGE", "").strip()
  if is_pathways and sidecar_image:
    missing_placeholders = [
        p
        for p in (
            "${COLOCATED_PYTHON_SIDECAR_BLOCK}",
            "${COLOCATED_PYTHON_WORKER_ARGS}",
            "${COLOCATED_PYTHON_PROXY_ARGS}",
            "${COLOCATED_PYTHON_SIDECAR_VOLUME}",
        )
        if p not in template_text
    ]
    if missing_placeholders:
      raise ValueError(
          f"COLOCATED_PYTHON_SIDECAR_IMAGE is set, but template "
          f"{args.template_file} lacks {', '.join(missing_placeholders)}"
      )
  substitutions = build_substitutions(args, template_text=template_text)
  return _substitute_template(template_text, substitutions)


def main(argv: Sequence[str] | None = None) -> None:
  """Parses command-line arguments and renders a deployment YAML from a template."""
  args = parse_args(argv)
  with open(args.template_file, "r") as f:
    template_text = f.read()
  content = render_template(template_text, args)
  print(content)


if __name__ == "__main__":
  main()
