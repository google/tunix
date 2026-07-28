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

"""Test doubles for orchestrator components."""

from typing import Any

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker


class FakeWorker(abstract_worker.Worker):
  """A unified test double for all worker types."""

  def __init__(self):
    self.call_counts = {
        "initialize": 0,
        "compile": 0,
        "start": 0,
        "stop": 0,
    }
    self.state = "CREATED"

  def initialize(self) -> datatypes.Response:
    self.call_counts["initialize"] += 1
    self.state = "INITIALIZED"
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    self.call_counts["compile"] += 1
    self.state = "COMPILED"
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    self.call_counts["start"] += 1
    self.state = "READY"
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.call_counts["stop"] += 1
    self.state = "STOPPED"
    return datatypes.Response()
