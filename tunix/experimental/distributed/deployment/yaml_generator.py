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
import dataclasses
import json
import math
import os
import re
import string

_K8S_LABEL_VALUE_RE = re.compile(r"^([a-zA-Z0-9][-a-zA-Z0-9_.]*)?[a-zA-Z0-9]$")

# Exit code a worker container uses to say "I failed before registering with
# the orchestrator, recreating me is safe". Anything else after registration is
# fatal for the JobSet. Kept in sync with
# examples/deepswe_dist/cluster_reaper.py (STARTUP_RETRY_EXIT_CODE).
STARTUP_RETRY_EXIT_CODE = 75

# Written by K8sDiscoveryContext.register() once registration succeeded. It
# lives in the container's own /dev/shm (fresh per container, and containers
# are never restarted in place under --fail_fast), not in the hostPath /tmp
# that pods on the same node share.
REGISTERED_MARKER_PATH = "/dev/shm/tunix_ft_registered"

# Indentation of the PID-1 wrapper script inside the `command: - |` block of
# the worker templates.
_WRAPPER_INDENT = " " * 18


@dataclasses.dataclass(frozen=True)
class FailFastPlaceholders:
  """Template placeholder values controlling worker fail-fast behaviour.

  Every field renders to the templates' pre-fail-fast text when disabled, so
  `--fail_fast` off produces byte-identical manifests.
  """

  # Line prefix that comments out a template's own `maxRestarts:` line.
  FAIL_FAST_LINE_DISABLE: str
  # Appended after `restartStrategy: Recreate` in the JobSet failurePolicy.
  FAIL_FAST_JOBSET_FAILURE_POLICY: str
  # Job `backoffLimit` and pod `restartPolicy` of the tpu / mcjax.ray workers.
  FAIL_FAST_WORKER_BACKOFF_LIMIT: str
  FAIL_FAST_WORKER_RESTART_POLICY: str
  # Appended after the worker Job's `backoffLimit:` line.
  FAIL_FAST_POD_FAILURE_POLICY: str
  # Line prefixes in the PID-1 wrapper: before the workload subshell starts,
  # and after its exit code is collected.
  FAIL_FAST_WRAPPER_PRE: str
  FAIL_FAST_WRAPPER_POST: str
  # Same as FAIL_FAST_WRAPPER_POST, but only pod 0 (the one that registers)
  # may ask for a startup retry. Used by the Ray template, whose other pods
  # only join the Ray cluster.
  FAIL_FAST_WRAPPER_POST_POD0: str
  # Pathways head pod: key holding pathways-rm/proxy, and a line prefix that
  # opens `containers:` for the user container. With fail-fast rm/proxy
  # become native sidecars so the pod terminates when the user container
  # exits (as regular `restartPolicy: Always` containers they keep the pod
  # Running forever).
  FAIL_FAST_PROC_SIDECARS_KEY: str
  FAIL_FAST_PROC_MAIN_CONTAINERS: str


def _indent_lines(lines: list[str], indent: str) -> str:
  return "".join(f"{indent}{line}\n" if line else "\n" for line in lines)


def _wrapper_post(retry_only_on_pod_zero: bool) -> str:
  """Maps the workload exit code to startup-retry or fatal.

  Workers are servers that never finish on their own, so every exit is
  abnormal, including exit 0 after a forwarded SIGTERM (which would otherwise
  complete the Job and the JobSet silently while the orchestrator waits).

  Args:
    retry_only_on_pod_zero: Only pod 0 registers (Ray template); other pods
      never ask for a startup retry.

  Returns:
    Wrapper script lines, indented for the templates.
  """
  startup_condition = f'[ ! -f "{REGISTERED_MARKER_PATH}" ]'
  if retry_only_on_pod_zero:
    startup_condition += ' && [ "${POD_INDEX}" = "0" ]'
  return _indent_lines(
      [
          "# fail-fast: a worker never exits on its own. Before registration",
          f"# ask the JobSet to retry startup (exit {STARTUP_RETRY_EXIT_CODE});"
          " after it, fail the JobSet.",
          f"if {startup_condition}; then",
          '  echo "fail-fast: exited (code $EXIT_CODE) before registering;'
          f' exiting {STARTUP_RETRY_EXIT_CODE} to retry startup"',
          f"  EXIT_CODE={STARTUP_RETRY_EXIT_CODE}",
          'elif [ "$EXIT_CODE" -eq 0 ] || [ "$EXIT_CODE" -eq'
          f" {STARTUP_RETRY_EXIT_CODE} ]; then",
          '  echo "fail-fast: stopped (code $EXIT_CODE,'
          ' sigterm=$FT_GOT_SIGTERM); exiting 1 to fail the JobSet"',
          "  EXIT_CODE=1",
          "fi",
      ],
      _WRAPPER_INDENT,
  )


def render_fail_fast(
    enabled: bool, startup_retries: int, user_container: str
) -> FailFastPlaceholders:
  """Returns template placeholder values for worker fail-fast.

  With fail-fast, pods are never restarted in place (a restarted worker would
  re-register under the same worker_id, which the orchestrator rejects, and the
  run deadlocks). Instead:

  * The PID-1 wrapper exits STARTUP_RETRY_EXIT_CODE if the workload stopped
    before registering, and non-zero for any stop after registering.
  * The Job's podFailurePolicy counts the startup exit code (backoffLimit 0 ->
    the Job fails -> JobSet restarts, bounded by maxRestarts=startup_retries)
    and fails the Job with reason PodFailurePolicy for anything else.
  * The JobSet fails immediately on a PodFailurePolicy Job failure.

  Args:
    enabled: Whether to render fail-fast.
    startup_retries: JobSet maxRestarts for pre-registration failures.
    user_container: Name of the worker container running the wrapper.

  Returns:
    Placeholder values for string.Template substitution.
  """
  if not enabled:
    return FailFastPlaceholders(
        FAIL_FAST_LINE_DISABLE="",
        FAIL_FAST_JOBSET_FAILURE_POLICY="",
        FAIL_FAST_WORKER_BACKOFF_LIMIT="2048000",
        FAIL_FAST_WORKER_RESTART_POLICY="OnFailure",
        FAIL_FAST_POD_FAILURE_POLICY="",
        FAIL_FAST_WRAPPER_PRE="",
        FAIL_FAST_WRAPPER_POST="",
        FAIL_FAST_WRAPPER_POST_POD0="",
        FAIL_FAST_PROC_SIDECARS_KEY="containers",
        FAIL_FAST_PROC_MAIN_CONTAINERS="",
    )

  jobset_failure_policy = (
      f"\n    maxRestarts: {startup_retries}"
      "\n    rules:"
      "\n    - name: failJobSetOnPodFailurePolicy"
      "\n      action: FailJobSet"
      "\n      onJobFailureReasons:"
      "\n      - PodFailurePolicy"
  )
  # First matching rule wins. A disrupted pod whose workload had not
  # registered yet still exits with the startup code, so rule 1 retries it;
  # disruptions without a container exit code (e.g. node lost) are fatal.
  pod_failure_policy = (
      "\n        podFailurePolicy:"
      "\n          rules:"
      "\n          - action: Count"
      "\n            onExitCodes:"
      f"\n              containerName: {user_container}"
      "\n              operator: In"
      f"\n              values: [{STARTUP_RETRY_EXIT_CODE}]"
      "\n          - action: FailJob"
      "\n            onPodConditions:"
      "\n            - type: DisruptionTarget"
      "\n          - action: FailJob"
      "\n            onExitCodes:"
      f"\n              containerName: {user_container}"
      "\n              operator: NotIn"
      f"\n              values: [0, {STARTUP_RETRY_EXIT_CODE}]"
  )
  wrapper_pre = _indent_lines(
      [
          "# fail-fast: the workload writes this marker once registered.",
          f"export FT_REGISTERED_MARKER={REGISTERED_MARKER_PATH}",
          'rm -f "$FT_REGISTERED_MARKER"',
          'FT_GOT_SIGTERM=""',
          '_sigterm() { FT_GOT_SIGTERM=1; kill -SIGTERM "$!" 2>/dev/null; }',
          "",
      ],
      _WRAPPER_INDENT,
  )
  return FailFastPlaceholders(
      FAIL_FAST_LINE_DISABLE="# overridden by fail-fast: ",
      FAIL_FAST_JOBSET_FAILURE_POLICY=jobset_failure_policy,
      FAIL_FAST_WORKER_BACKOFF_LIMIT="0",
      FAIL_FAST_WORKER_RESTART_POLICY="Never",
      FAIL_FAST_POD_FAILURE_POLICY=pod_failure_policy,
      FAIL_FAST_WRAPPER_PRE=wrapper_pre,
      FAIL_FAST_WRAPPER_POST=_wrapper_post(retry_only_on_pod_zero=False),
      FAIL_FAST_WRAPPER_POST_POD0=_wrapper_post(retry_only_on_pod_zero=True),
      FAIL_FAST_PROC_SIDECARS_KEY="initContainers",
      FAIL_FAST_PROC_MAIN_CONTAINERS="            containers:\n",
  )


def main() -> None:
  """Parses command-line arguments and renders a deployment YAML from a template."""
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
      default=os.environ.get("CPU_NODEPOOL", "cpu-np"),
      help="GKE nodepool for CPU jobs (e.g. cpu-np, cpu-highmem-np).",
  )
  parser.add_argument(
      "--cpu_memory",
      default=os.environ.get("CPU_MEMORY", "240G"),
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
      default=os.environ.get("GANG_ID", ""),
      help=(
          'Add scheduling.x-k8s.io/gang-id: "<gang_id>" label to JobSet'
          " metadata (or set GANG_ID)."
      ),
  )
  parser.add_argument(
      "--service_account",
      default=os.environ.get("SERVICE_ACCOUNT", "xpk-sa"),
      help="Kubernetes service account for pods.",
  )
  parser.add_argument(
      "--priority_class",
      default=os.environ.get("PRIORITY_CLASS", "medium"),
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
      "--termination_grace_seconds",
      default=int(os.environ.get("TERMINATION_GRACE_SECONDS") or 360),
      type=int,
      help="Termination grace period in seconds for pods (default: 360).",
  )

  parser.add_argument(
      "--pathways_server_image",
      default="us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:latest",
      help="Pathways server image",
  )
  parser.add_argument(
      "--pathways_proxy_server_image",
      default=(
          "us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:latest"
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
      "--fail_fast",
      action="store_true",
      default=False,
      help=(
          "Render worker templates (tpu, mcjax.ray, pathways) so that a worker"
          " failure after it registered with the orchestrator fails the"
          " JobSet instead of restarting pods in place. Failures before"
          " registration are retried up to --startup_retries times by"
          " recreating the JobSet."
      ),
  )
  parser.add_argument(
      "--startup_retries",
      type=int,
      default=3,
      help=(
          "With --fail_fast: JobSet maxRestarts for failures that happen"
          " before the worker registered with the orchestrator."
      ),
  )

  args = parser.parse_args()
  if args.startup_retries < 0:
    raise ValueError(
        f"--startup_retries must be >= 0, got {args.startup_retries}"
    )
  fail_fast = render_fail_fast(
      enabled=args.fail_fast,
      startup_retries=args.startup_retries,
      user_container=args.worker_container_name,
  )

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
  if args.jobset_name is None:
    jobset_name = f"{os.environ.get('USER')}-{pw_instance_type}-{num_chips}"

  if args.preemptible is not None:
    preemptible = args.preemptible
  else:
    preemptible = os.environ.get("PREEMPTIBLE", "").lower() in ("true", "1")

  labels = []
  if args.queue_name:
    labels.append(f"    kueue.x-k8s.io/queue-name: {args.queue_name}\n")
  if preemptible:
    labels.append('    scheduling.x-k8s.io/preemptible: "true"\n')
  if args.gang_id:
    if len(args.gang_id) > 63 or not _K8S_LABEL_VALUE_RE.match(args.gang_id):
      raise ValueError(
          f"Invalid gang_id '{args.gang_id}': must be a valid Kubernetes label"
          " value (63 characters or less, consisting of alphanumeric"
          " characters, '-', '_', or '.', and must start and end with an"
          " alphanumeric character)."
      )
    labels.append(f'    scheduling.x-k8s.io/gang-id: "{args.gang_id}"\n')
  queue_label = "  labels:\n" + "".join(labels) if labels else ""

  # Optional reservation pin. NAP will not create a large TPU slice without being
  # told which reservation to draw from -- it fails the scale-up with
  # "Invalid Reservation: ." and backs off -- so this has to be emitted when set.
  # Rendered as a whole nodeSelector line, because string.Template cannot express
  # "omit this key when the value is empty".
  # Kueue orders its queue by priority. On a busy ClusterQueue an unprioritised
  # workload (priority 0) can sit behind tens of thousands of pending ones and never
  # be evaluated, while everything with a class jumps ahead.
  priority_class = os.environ.get("KUEUE_PRIORITY_CLASS", "").strip()
  priority_class_line = (
      f"\n            priorityClassName: {priority_class}" if priority_class else ""
  )

  reservation_name = os.environ.get("TPU_RESERVATION", "").strip()
  reservation_selector = (
      f"\n              cloud.google.com/reservation-name: {reservation_name}"
      if reservation_name
      else ""
  )

  if args.use_dynamic_slicing is not None:
    use_dynamic_slicing = args.use_dynamic_slicing
  else:
    use_dynamic_slicing = os.environ.get("USE_DYNAMIC_SLICING", "").lower() in (
        "true",
        "1",
    ) or (tpu_type in ("tpu7x", "tpu-v7x-slice"))

  head_nodepool = args.head_nodepool or os.environ.get("HEAD_NODEPOOL", "")
  if not head_nodepool and use_dynamic_slicing:
    head_nodepool = "cpu-np"

  if use_dynamic_slicing:
    head_node_selector = (
        f"              cloud.google.com/gke-nodepool: {head_nodepool}"
        if head_nodepool
        else f"              node.kubernetes.io/instance-type: {args.cpu_machine or 'n2d-standard-64'}"
    )
    head_affinity = ""
    head_tolerations = (
        "\n            tolerations:\n"
        "            - key: \"cloud.google.com/gke-nodepool\"\n"
        f"              operator: \"{'Equal' if head_nodepool else 'Exists'}\"\n"
        + (f"              value: \"{head_nodepool}\"\n" if head_nodepool else "")
        + "              effect: \"NoSchedule\""
    )
    tpu_topology_selector = ""
  else:
    head_node_selector = (
        f"              cloud.google.com/gke-tpu-accelerator: {tpu_type}\n"
        f"              cloud.google.com/gke-tpu-topology: {tpu_topology}"
        f"{reservation_selector}"
    )
    head_affinity = (
        "            affinity:\n"
        "              podAffinity:\n"
        "                requiredDuringSchedulingIgnoredDuringExecution:\n"
        "                # place on one of the pathways-worker nodes\n"
        "                - topologyKey: kubernetes.io/hostname\n"
        "                  labelSelector:\n"
        "                    matchExpressions:\n"
        "                    - key: jobset.sigs.k8s.io/jobset-name\n"
        "                      operator: In\n"
        "                      values:\n"
        f"                      - {jobset_name}\n"
        "                    - key: jobset.sigs.k8s.io/replicatedjob-name\n"
        "                      operator: In\n"
        "                      values:\n"
        "                      - pw-node\n"
    )
    head_tolerations = ""
    tpu_topology_selector = (
        f"\n              cloud.google.com/gke-tpu-topology: {tpu_topology}"
        if tpu_topology
        else ""
    )

  if use_dynamic_slicing and slice_topology:
    if slice_size and slice_size > 1:
      # The slice-topology annotation must be the job's FULL shape: the
      # mjobset webhook checks the requested TPU count against it, so a
      # 4x4x8 job annotated 4x4x4 is refused ("128 TPUs requested, but must
      # be exactly 64"). The partition levels below stay at the 4x4x4 unit.
      anno_lines = [
          f'cloud.google.com/gke-tpu-slice-topology: "{tpu_topology}"',
          'cloud.google.com/skip-tpu-webhook-check: "true"',
          "kueue.x-k8s.io/podset-required-topology: cloud.google.com/gce-topology-block",
          f"kueue.x-k8s.io/podset-slice-required-topology: cloud.google.com/gke-tpu-partition-{slice_topology}-id",
          f'kueue.x-k8s.io/podset-slice-size: "{slice_size}"',
      ]
    else:
      anno_lines = [
          f'cloud.google.com/gke-tpu-slice-topology: "{slice_topology}"',
          'cloud.google.com/skip-tpu-webhook-check: "true"',
      ]
    tpu_annotations = "\n" + "\n".join(f"              {line}" for line in anno_lines)
  else:
    tpu_annotations = ""

  if use_dynamic_slicing:
    if slice_size and slice_size > 1:
      pw_node_affinity = (
          "            affinity:\n"
          "              nodeAffinity:\n"
          "                requiredDuringSchedulingIgnoredDuringExecution:\n"
          "                  nodeSelectorTerms:\n"
          "                  - matchExpressions:\n"
          f"                    - key: cloud.google.com/gke-tpu-partition-{slice_topology}-state\n"
          "                      operator: In\n"
          "                      values: [\"HEALTHY\", \"DEGRADED\"]\n"
      )
      tpu_affinity = pw_node_affinity
    else:
      pw_node_affinity = ""
      tpu_affinity = ""
  else:
    pw_node_affinity = (
        "            affinity:\n"
        "              # place on nodes in the same nodepool\n"
        "              podAffinity:\n"
        "                requiredDuringSchedulingIgnoredDuringExecution:\n"
        "                - topologyKey: cloud.google.com/gke-nodepool\n"
        "                  labelSelector:\n"
        "                    matchExpressions:\n"
        f"                    - key: jobset.sigs.k8s.io/jobset-name\n"
        "                      operator: In\n"
        "                      values:\n"
        f"                      - {jobset_name}\n"
        "              # ensure exclusive access to the nodepool (among all jobsets created with this yaml)\n"
        "              podAntiAffinity:\n"
        "                requiredDuringSchedulingIgnoredDuringExecution:\n"
        "                - topologyKey: cloud.google.com/gke-nodepool\n"
        "                  labelSelector:\n"
        "                    matchExpressions:\n"
        f"                    - key: jobset.sigs.k8s.io/jobset-name\n"
        "                      operator: Exists\n"
        "                    - key: jobset.sigs.k8s.io/jobset-name\n"
        "                      operator: NotIn\n"
        "                      values:\n"
        f"                      - {jobset_name}\n"
    )
    tpu_affinity = ""
  # Colocated-python checkpointing sidecar. Emitted as a whole block for the same reason
  # as reservation_selector above: string.Template cannot omit a key when unset, and an
  # initContainer with an empty image would wedge every pathways-worker pod.
  #
  # `restartPolicy: Always` on an initContainer is the k8s native-sidecar pattern: it starts
  # before, and stays running alongside, the worker container.
  #
  # The image MUST match the head image's jax/jaxlib exactly and must contain orbax, since
  # Orbax ships its serialization callables here by reference via cloudpickle.
  sidecar_image = os.environ.get("COLOCATED_PYTHON_SIDECAR_IMAGE", "").strip()
  sidecar_shm = os.environ.get("COLOCATED_PYTHON_SIDECAR_SHM", "1").strip().lower() not in (
      "0",
      "false",
      "no",
  )
  sidecar_memory = os.environ.get("COLOCATED_PYTHON_SIDECAR_MEMORY", "16Gi").strip()
  worker_shm_mount = (
      "\n              - mountPath: /tmp/sidecar\n                name: sidecar-shared-memory"
      if sidecar_shm
      else ""
  )
  sidecar_shm_env = (
      "\n              - name: CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY\n                value: /tmp/sidecar"
      if sidecar_shm
      else ""
  )
  sidecar_volume_mount = (
      "- mountPath: /tmp/sidecar\n                name: sidecar-shared-memory"
      if sidecar_shm
      else "- mountPath: /tmp\n                name: shared-tmp"
  )
  colocated_python_sidecar_block = (
      f"""{worker_shm_mount}
            initContainers:
            - name: colocated-python-sidecar
              image: {sidecar_image}
              imagePullPolicy: Always
              env:
              - name: GRPC_SERVER_ADDRESS
                value: "0.0.0.0:50051"{sidecar_shm_env}
              ports:
              - containerPort: 50051
                protocol: TCP
              resources:
                requests:
                  cpu: "4"
                  memory: {sidecar_memory}
              restartPolicy: Always
              volumeMounts:
              {sidecar_volume_mount}"""
      if sidecar_image
      else ""
  )
  colocated_python_worker_args = (
      "\n              - --cloud_pathways_sidecar_shm_directory=/tmp/sidecar"
      if (sidecar_image and sidecar_shm)
      else ""
  )
  colocated_python_proxy_args = (
      "\n              - --sidecar_name=external" if sidecar_image else ""
  )
  colocated_python_sidecar_volume = (
      "\n            - name: sidecar-shared-memory\n              emptyDir:\n                medium: Memory"
      if (sidecar_image and sidecar_shm)
      else ""
  )

  # RAIDEN_BROADCAST_K, TPU_RAIDEN_DATA_NICS, and ENABLE_MULTI_NUMA are set in
  # every container env of the jobset templates; rendering them again here would
  # duplicate the entry.
  template_env = (
      "RAIDEN_BROADCAST_K",
      "TPU_RAIDEN_DATA_NICS",
      "ENABLE_MULTI_NUMA",
  )
  worker_env = {}

  # Under Pathways the TPU program runs in the worker, not the user container, so
  # LIBTPU_INIT_ARGS has to be set there. The server rejects the xpk-level
  # --extra_env_vars/--extra_flags with "Unknown command line flag"; xpk turns
  # those into container env, which is what this reproduces. One KEY=VALUE per
  # line, because a value may contain spaces.
  for line in os.environ.get("PATHWAYS_WORKER_EXTRA_ENV", "").strip().splitlines():
    line = line.strip()
    if not line or "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key.strip() in template_env:
      continue
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
      value = value[1:-1]
    worker_env[key.strip()] = value

  pathways_worker_extra_env = "".join(
      f"\n              - name: {key}\n                value: {json.dumps(value)}"
      for key, value in worker_env.items()
  )

  # The proxy, not the worker, compiles the TPU program under Pathways, so XLA
  # compiler flags (e.g. --xla_tpu_scoped_vmem_limit_kib) only take effect as
  # proxy args; in the worker's LIBTPU_INIT_ARGS they are ignored at compile
  # time. Whitespace-separated, like xpk's --custom-pathways-proxy-server-args.
  pathways_proxy_extra_args = "".join(
      f"\n              - {json.dumps(flag)}"
      for flag in os.environ.get("PATHWAYS_PROXY_EXTRA_ARGS", "").split()
  )

  enable_multi_numa = os.environ.get("ENABLE_MULTI_NUMA", "0")
  tpu_raiden_data_nics = os.environ.get("TPU_RAIDEN_DATA_NICS", "eth0")

  with open(args.template_file, "r") as f:
    template_text = f.read()
    if args.fail_fast and "${FAIL_FAST_POD_FAILURE_POLICY}" not in template_text:
      raise ValueError(
          f"--fail_fast is not supported by template {args.template_file}"
      )

    # Guard: only Pathways templates run the colocated sidecar; non-Pathways
    # templates (e.g. CPU orchestrator, Ray/MCJAX rollout) share the launcher
    # environment but do not include pathways-worker or the sidecar.
    is_pathways = "name: pathways-worker" in template_text
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
        raise SystemExit(
            f"COLOCATED_PYTHON_SIDECAR_IMAGE is set, but template "
            f"{args.template_file} lacks {', '.join(missing_placeholders)}"
        )
    template = string.Template(template_text)
    content = template.substitute(
        JOBSET_NAME=jobset_name,
        USER=os.environ.get("USER"),
        NAMESPACE=args.namespace,
        QUEUE_NAME=args.queue_name,
        QUEUE_LABEL=queue_label,
        SERVICE_ACCOUNT=args.service_account,
        PRIORITY_CLASS=args.priority_class,
        SERVER_IMAGE=args.pathways_server_image,
        PROXY_IMAGE=args.pathways_proxy_server_image,
        GCS_SCRATCH_LOCATION=args.pathways_gcs_scratch_location,
        PATHWAYS_PROXY_MEMORY_LIMIT=args.pathways_proxy_memory_limit,
        PATHWAYS_PROXY_MEMORY=args.pathways_proxy_memory,
        PATHWAYS_RM_MEMORY=args.pathways_rm_memory,
        USER_CONTAINER_MEMORY=args.user_container_memory,
        USER_CONTAINER_MEMORY_LIMIT=args.user_container_memory_limit,
        PATHWAYS_WORKER_MEMORY=args.pathways_worker_memory,
        CPU_MACHINE=args.cpu_machine,
        CPU_NODEPOOL=args.cpu_nodepool,
        CPU_MEMORY=args.cpu_memory,
        TPU_MACHINE=tpu_machine,
        TPU_TYPE=tpu_type,
        TPU_TOPOLOGY=tpu_topology,
        TPU_TOPOLOGY_SELECTOR=tpu_topology_selector,
        TPU_ANNOTATIONS=tpu_annotations,
        TPU_AFFINITY=tpu_affinity,
        PW_NODE_AFFINITY=pw_node_affinity,
        HEAD_NODE_SELECTOR=head_node_selector,
        HEAD_AFFINITY=head_affinity,
        HEAD_TOLERATIONS=head_tolerations,
        RESERVATION_SELECTOR=reservation_selector,
        PRIORITY_CLASS_LINE=priority_class_line,
        COLOCATED_PYTHON_SIDECAR_BLOCK=colocated_python_sidecar_block,
        COLOCATED_PYTHON_SIDECAR_VOLUME=colocated_python_sidecar_volume,
        COLOCATED_PYTHON_WORKER_ARGS=colocated_python_worker_args,
        COLOCATED_PYTHON_PROXY_ARGS=colocated_python_proxy_args,
        TERMINATION_GRACE_SECONDS=args.termination_grace_seconds,
        PW_INSTANCE_TYPE=pw_instance_type,
        REPLICAS=1,
        COMPLETIONS=num_chips // 4 if num_chips else None,
        PARALLELISM=num_chips // 4 if num_chips else None,
        PODSET_SLICE_TOPOLOGY=slice_topology,
        PODSET_SLICE_SIZE=slice_size,
        USER_CONTAINER=args.worker_container_name,
        USER_CONTAINER_IMAGE=args.worker_container_image,
        USER_CONTAINER_PORT=args.worker_container_port,
        STARTUP_COMMAND=args.worker_startup_command,
        PATHWAYS_WORKER_EXTRA_ENV=pathways_worker_extra_env,
        PATHWAYS_PROXY_EXTRA_ARGS=pathways_proxy_extra_args,
        ENABLE_MULTI_NUMA=enable_multi_numa,
        TPU_RAIDEN_DATA_NICS=tpu_raiden_data_nics,
        **dataclasses.asdict(fail_fast),
    )
    print(content)


if __name__ == "__main__":
  main()
