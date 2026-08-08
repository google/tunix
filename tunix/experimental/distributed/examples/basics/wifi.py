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

import argparse
import logging
import pickle
import time

from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv: list[str], context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="this is wifi!", help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  context.ipc.discovery.on_connect(
      on_client_connected=lambda client_id, h, p, m, is_rec: logging.info(
          f"Phone {client_id} connected (reconnect={is_rec})"
      ),
      on_client_disconnected=lambda client_id, h, p, reason: logging.warning(
          f"Phone {client_id} disconnected ({reason})"
      ),
  )

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass
