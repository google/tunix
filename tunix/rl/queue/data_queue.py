# Copyright 2025 Google LLC
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

"""Data queue interface."""

import abc
import queue
from typing import Generic, TypeVar

_T = TypeVar("_T")

ABC = abc.ABC
abstractmethod = abc.abstractmethod


class AbstractDataQueue(ABC, Generic[_T]):
  """Abstract base class for data queues."""

  @abstractmethod
  def put(self, item: _T) -> None:
    """Puts an item into the queue."""

  @abstractmethod
  def get(self, block: bool = True, timeout: float | None = None) -> _T:
    """Gets an item from the queue."""

  @abstractmethod
  def qsize(self) -> int:
    """Returns the approximate size of the queue."""

  @abstractmethod
  def close(self) -> None:
    """Closes the queue."""


class SimpleDataQueue(AbstractDataQueue[_T]):
  """A simple data queue based on queue.Queue."""

  def __init__(self, maxsize: int = 0):
    self._queue = queue.Queue(maxsize=maxsize)

  def put(self, item: _T) -> None:
    self._queue.put(item)

  def get(self, block: bool = True, timeout: float | None = None) -> _T:
    return self._queue.get(block=block, timeout=timeout)

  def qsize(self) -> int:
    return self._queue.qsize()

  def close(self) -> None:
    if not self._queue.empty():
      while True:
        try:
          self._queue.get_nowait()
        except queue.Empty:
          break
