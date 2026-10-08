# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Payload store for sub-batch checkpoint tests, without a Trajectory Store."""

from collections.abc import Sequence
import copy
from typing import Any

from tunix.experimental.trajectory import store as store_lib


class TestPayloadStore:
  """In-memory `sub_batch_checkpoint.TrajectoryPayloadStore` for tests.

  Like a durable store, it keeps its own copy of every committed payload and
  returns a fresh copy from every load, so a restore never aliases what the
  saving run holds. References are sequential ids that are never reused.
  Unlike `TrajectoryStorePayloads`, it does not validate payloads: tests of
  what production accepts use the real store.
  """

  def __init__(self) -> None:
    self.payloads: dict[str, tuple[Any, Any]] = {}
    self._next_id = 0

  def commit(
      self, payloads: Sequence[tuple[Any, Any]]
  ) -> list[dict[str, str]]:
    """Keeps a copy of each payload and returns one new reference to each."""
    refs = []
    for payload in payloads:
      ref_id = str(self._next_id)
      self._next_id += 1
      self.payloads[ref_id] = copy.deepcopy(payload)
      refs.append({"id": ref_id})
    return refs

  def load(self, refs: Sequence[dict[str, str]]) -> list[tuple[Any, Any]]:
    """Returns a fresh copy of each referenced payload.

    Args:
      refs: References returned by `commit`.

    Returns:
      The payloads, in the order of `refs`.

    Raises:
      store_lib.TrajectoryNotFoundError: If a reference is unknown, as the
        Trajectory Store raises for an id it does not hold.
    """
    loaded = []
    for ref in refs:
      if ref["id"] not in self.payloads:
        raise store_lib.TrajectoryNotFoundError(ref["id"])
      loaded.append(copy.deepcopy(self.payloads[ref["id"]]))
    return loaded
