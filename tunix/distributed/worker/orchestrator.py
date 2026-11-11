# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Orchestrator demo."""

import argparse
from concurrent import futures
import logging
import time

import grpc
from tunix.distributed.service import registration_service_pb2
from tunix.distributed.service import registration_service_pb2_grpc


class RegisterRolloutWorkerHandler(
    registration_service_pb2_grpc.RegistrationServiceServicer
):

  def __init__(self, name):
    self.name = name

  def RegisterRolloutWorker(self, request, context):
    logging.info(
        f"[{self.name}] registered worker {request.worker_id} with"
        f" RolloutService at {request.rollout_service_address}"
    )

    return registration_service_pb2.RegisterRolloutWorkerResponse(
        reply=f"{request.worker_id} registered.", success=True
    )


def main():
  parser = argparse.ArgumentParser(description="Orchestrator")
  parser.add_argument(
      "--name",
      type=str,
      default="orchestrator",
      help="Name of the orchestrator",
  )
  parser.add_argument(
      "--registration_service_port",
      type=int,
      default=12345,
      help="Port for registration service",
  )
  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, force=True)

  # 1. Initialization

  # start gRPC server
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

  # register handlers
  registration_service_pb2_grpc.add_RegistrationServiceServicer_to_server(
      RegisterRolloutWorkerHandler(name=args.name), server
  )

  # start server
  server.add_insecure_port(f"[::]:{args.registration_service_port}")
  server.start()
  logging.info(
      f"[{args.name}] Registration service started at port"
      f" {args.registration_service_port}"
  )

  # 2. Main loop

  # TODO: orchestration logic
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    server.stop(0)

  # 3. Cleanup


if __name__ == "__main__":
  main()
