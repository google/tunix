# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout Worker demo."""

import argparse
import logging
import time

import grpc
import jax
from tunix.distributed.plugin import jax as jax_plugin
from tunix.distributed.plugin import k8s
from tunix.distributed.service import registration_service_pb2
from tunix.distributed.service import registration_service_pb2_grpc


def main():
  parser = argparse.ArgumentParser(description="Rollout Worker")
  parser.add_argument(
      "--name", type=str, default="rollout", help="Name of the rollout worker"
  )
  parser.add_argument(
      "--registration_service_address",
      type=str,
      default="localhost:12345",
      help="Address of the registration service",
  )
  parser.add_argument(
      "--rollout_service_port",
      type=int,
      default=11111,
      help="Port of the rollout worker service",
  )
  args = parser.parse_args()

  is_local = (
      "localhost" in args.registration_service_address
      or "127.0.0.1" in args.registration_service_address
  )
  logging.basicConfig(level=logging.INFO, force=True)

  # 1. Initialization

  # init jax
  if not is_local:
    jax_plugin.init_pathways()
  logging.info(f"[{args.name}] jax devices: {jax.devices()}")
  # TODO: init RolloutService

  # 2. Register RolloutService

  logging.info(f"[{args.name}] register to {args.registration_service_address}")
  with grpc.insecure_channel(args.registration_service_address) as channel:
    stub = registration_service_pb2_grpc.RegistrationServiceStub(channel)

    hostname = "localhost" if is_local else k8s.get_jobset_hostname()

    request_payload = registration_service_pb2.RegisterRolloutWorkerRequest(
        worker_id=args.name,
        rollout_service_address=f"{hostname}:{args.rollout_service_port}",
    )

    while True:
      try:
        response = stub.RegisterRolloutWorker(request_payload)
        logging.info(
            f"[{args.name}] register"
            f" {'succeeded' if response.success else 'failed'}:"
            f" {response.reply}"
        )
        break
      except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
          time.sleep(60)
          continue
        else:
          raise RuntimeError(
              f"[{args.name}] register failed: {e.code()} - {e.details()}"
          )

  # 3. Main loop

  # TODO: handle RolloutService requests
  try:
    logging.info(f"[{args.name}] start serving")
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    logging.info(f"[{args.name}] stop serving")

  # 4. Cleanup

  # TODO: wait RolloutService finish


if __name__ == "__main__":
  main()
