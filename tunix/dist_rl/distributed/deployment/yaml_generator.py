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
    if not raw:
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

  labels: list[str] = []
  if args.queue_name:
    labels.append(f"    kueue.x-k8s.io/queue-name: {args.queue_name}\n")
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

  return {
      "JOBSET_NAME": jobset_name,
      "USER": os.environ.get("USER"),
      "NAMESPACE": args.namespace,
      "QUEUE_NAME": args.queue_name,
      "QUEUE_LABEL": queue_label,
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
