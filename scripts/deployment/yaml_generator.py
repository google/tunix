import argparse
import math
import os
import string


def main():
  parser = argparse.ArgumentParser()

  parser.add_argument("template_file", help="Path to the template file")

  parser.add_argument("--jobset_name", default=None, help="Name of the jobset")

  parser.add_argument(
      "--tpu_slice",
      default=None,
      help="TPU type and tpu_topology (e.g. tpu7x:4x4x8)",
  )
  parser.add_argument(
      "--cpu_machine",
      default=None,
      help="CPU machine type (e.g. n2-standard-64)",
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

    if tpu_type == "tpu7x":
      slice_topology = tpu_topology if num_chips <= 64 else "4x4x4"
      slice_size = num_chips // 4 if num_chips <= 64 else 16
      tpu_machine = "tpu7x-standard-4t"
      pw_instance_type = "tpu7x"
    elif tpu_type == "tpuv5" or tpu_type == "tpu-v5p-slice":
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct5p-hightpu-4t"
      tpu_type = "tpu-v5p-slice"
      pw_instance_type = "tpuv5"
    elif tpu_type == "tpuv5e" or tpu_type == "tpu-v5-lite-podslice":
      slice_topology = tpu_topology
      slice_size = num_chips // 4
      tpu_machine = "ct5lp-hightpu-4t"
      tpu_type = "tpu-v5-lite-podslice"
      pw_instance_type = "tpuv5e"
    else:
      raise ValueError(f"Unsupported TPU type {tpu_type}")

  jobset_name = args.jobset_name
  if args.jobset_name is None:
    jobset_name = f"{os.environ.get('USER')}-{pw_instance_type}-{num_chips}"

  with open(args.template_file, "r") as f:
    template = string.Template(f.read())
    content = template.substitute(
        JOBSET_NAME=jobset_name,
        USER=os.environ.get("USER"),
        SERVER_IMAGE=args.pathways_server_image,
        PROXY_IMAGE=args.pathways_proxy_server_image,
        GCS_SCRATCH_LOCATION=args.pathways_gcs_scratch_location,
        CPU_MACHINE=args.cpu_machine,
        TPU_MACHINE=tpu_machine,
        TPU_TYPE=tpu_type,
        TOPOLOGY=tpu_topology,
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
    )
    print(content)


if __name__ == "__main__":
  main()
