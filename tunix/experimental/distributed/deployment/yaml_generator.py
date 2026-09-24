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
import json
import math
import os
import string


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
      "--omit_slice_topology",
      action="store_true",
      default=False,
      help=(
          "Omit cloud.google.com/gke-tpu-slice-topology annotation from initial"
          " JobSet manifest (used for single-host dynamic slice admission flow)."
      ),
  )
  parser.add_argument(
      "--head_nodepool",
      default=None,
      help="Kubernetes nodepool for Pathways head pod (e.g. cpu-np).",
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

  args = parser.parse_args()

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

  queue_label = (
      f"  labels:\n    kueue.x-k8s.io/queue-name: {args.queue_name}\n"
      if args.queue_name
      else ""
  )

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

  # Under Pathways the TPU program runs in the worker, not the user container, so
  # LIBTPU_INIT_ARGS has to be set there. The server rejects the xpk-level
  # --extra_env_vars/--extra_flags with "Unknown command line flag"; xpk turns
  # those into container env, which is what this reproduces. One KEY=VALUE per
  # line, because a value may contain spaces.
  _worker_env_entries = []
  for _line in os.environ.get("PATHWAYS_WORKER_EXTRA_ENV", "").strip().splitlines():
    _line = _line.strip()
    if not _line or "=" not in _line:
      continue
    _k, _v = _line.split("=", 1)
    _worker_env_entries.append(
        f"\n              - name: {_k.strip()}"
        f"\n                value: {json.dumps(_v)}"
    )
  pathways_worker_extra_env = "".join(_worker_env_entries)

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
      anno_lines = [
          f'cloud.google.com/gke-tpu-slice-topology: "{tpu_topology}"',
          'cloud.google.com/skip-tpu-webhook-check: "true"',
          "kueue.x-k8s.io/podset-required-topology: cloud.google.com/gce-topology-block",
          "kueue.x-k8s.io/podset-slice-required-topology: cloud.google.com/gke-tpu-partition-4x4x4-id",
          f'kueue.x-k8s.io/podset-slice-size: "{slice_size}"',
      ]
    else:
      anno_lines = [
          'cloud.google.com/skip-tpu-webhook-check: "true"',
          'kueue.x-k8s.io/podset-required-topology: cloud.google.com/gke-tpu-partition-4x4x4-id',
      ]
      if not args.omit_slice_topology:
        anno_lines.insert(0, f'cloud.google.com/gke-tpu-slice-topology: "{slice_topology}"')
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
          "                    - key:"
          " cloud.google.com/gke-tpu-partition-4x4x4-state\n"
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
  sidecar_memory = os.environ.get("COLOCATED_PYTHON_SIDECAR_MEMORY", "16Gi").strip()
  colocated_python_sidecar_block = (
      f"""
            initContainers:
            - name: colocated-python-sidecar
              image: {sidecar_image}
              imagePullPolicy: Always
              env:
              - name: GRPC_SERVER_ADDRESS
                value: "0.0.0.0:50051"
              ports:
              - containerPort: 50051
                protocol: TCP
              resources:
                requests:
                  cpu: "4"
                  memory: {sidecar_memory}
                limits:
                  memory: {sidecar_memory}
              restartPolicy: Always
              volumeMounts:
              - mountPath: /tmp
                name: shared-tmp"""
      if sidecar_image
      else ""
  )

  tpu_raiden_data_nics = os.environ.get("TPU_RAIDEN_DATA_NICS", "").strip()
  if not tpu_raiden_data_nics and tpu_type in ("tpu7x", "tpu-v7x-slice"):
    tpu_raiden_data_nics = "eth0"

  if tpu_raiden_data_nics and "- name: TPU_RAIDEN_DATA_NICS\n" not in pathways_worker_extra_env:
    pathways_worker_extra_env += (
        f"\n              - name: TPU_RAIDEN_DATA_NICS\n                value: \"{tpu_raiden_data_nics}\""
    )

  with open(args.template_file, "r") as f:
    template = string.Template(f.read())
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
        PW_INSTANCE_TYPE=pw_instance_type,
        PATHWAYS_WORKER_EXTRA_ENV=pathways_worker_extra_env,
        REPLICAS=1,
        COMPLETIONS=num_chips // 4 if num_chips else None,
        PARALLELISM=num_chips // 4 if num_chips else None,
        PODSET_SLICE_TOPOLOGY=slice_topology,
        PODSET_SLICE_SIZE=slice_size,
        USER_CONTAINER=args.worker_container_name,
        USER_CONTAINER_IMAGE=args.worker_container_image,
        USER_CONTAINER_PORT=args.worker_container_port,
        STARTUP_COMMAND=args.worker_startup_command,
    )
    print(content)


if __name__ == "__main__":
  main()
