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

"""Moving trained weights to the processes that generate with them.

When the trainer and the samplers share a process, a weight sync is a pointer
assignment and there is nothing to transport. Once they are separate
processes, something has to carry the bytes, and the sync protocol needs a
seam it can name without knowing which mechanism is behind it: a trainer
stages a version and hands back coordinates, a replica fetches those
coordinates, and the version is released when no one needs it any more.

The implementation here writes to a shared directory. That is the honest
starting rung -- it works across processes on one machine, it is inspectable
when something goes wrong, and it needs no accelerator-specific machinery.
Faster rungs (device-to-device transfer, a checkpoint service) replace it
behind the same three calls.

Versions are retained rather than overwritten, because a replica that was
slow, retried, or quarantined may still be asking for one the trainer has
moved past; releasing is explicit so that decision stays with the coordinator.
"""

from __future__ import annotations

import abc
import dataclasses
import os
import shutil
from typing import Any, Mapping

from absl import logging
import numpy as np


@dataclasses.dataclass(frozen=True)
class TransportMetadata:
  """Where a staged version can be fetched from.

  Attributes:
    version: The policy version these weights carry.
    location: Transport-specific coordinates; for the filesystem rung, a path.
    method: Names the transport that produced this, so a replica can tell it
      was handed coordinates it does not know how to read.
  """

  version: int
  location: str
  method: str


class WeightTransport(abc.ABC):
  """Carries one version of weights from a trainer to its replicas."""

  @abc.abstractmethod
  def stage(self, params: Any, version: int) -> TransportMetadata:
    """Publishes `params` as `version` and returns how to fetch them."""

  @abc.abstractmethod
  def fetch(self, metadata: TransportMetadata) -> Any:
    """Returns the params a `stage` call published."""

  @abc.abstractmethod
  def release(self, version: int) -> None:
    """Discards a version no replica will ask for again."""


class FileWeightTransport(WeightTransport):
  """Stages weights into a shared directory, one file per version."""

  METHOD = "file"

  def __init__(self, directory: str):
    """Initializes the transport.

    Args:
      directory: Shared location both sides can reach. Created if absent.
    """
    self._directory = directory
    os.makedirs(directory, exist_ok=True)

  @property
  def directory(self) -> str:
    return self._directory

  def stage(self, params: Any, version: int) -> TransportMetadata:
    """Writes the params, then names them, so a reader never sees a partial file.

    Args:
      params: A flat mapping of name to array.
      version: The version being published.

    Returns:
      Coordinates for `fetch`.
    """
    flat = {key: np.asarray(value) for key, value in params.items()}
    final = self._path(version)
    partial = f"{final}.partial"
    np.savez(partial, **flat)
    # savez appends .npz when the name has no suffix; rename what it wrote.
    written = partial if os.path.exists(partial) else f"{partial}.npz"
    os.replace(written, final)
    logging.info("Staged weight version %d at %s", version, final)
    return TransportMetadata(
        version=version, location=final, method=self.METHOD
    )

  def fetch(self, metadata: TransportMetadata) -> Mapping[str, np.ndarray]:
    """Reads a staged version.

    Args:
      metadata: What `stage` returned.

    Returns:
      The params, by name.

    Raises:
      ValueError: If the coordinates came from a different transport.
      FileNotFoundError: If the version was already released, or never staged.
    """
    if metadata.method != self.METHOD:
      raise ValueError(
          f"These coordinates were produced by {metadata.method!r} transport,"
          f" which this one cannot read."
      )
    with np.load(metadata.location) as loaded:
      return {key: loaded[key] for key in loaded.files}

  def release(self, version: int) -> None:
    """Discards a version; releasing one that is already gone is fine."""
    path = self._path(version)
    if os.path.exists(path):
      os.remove(path)

  def release_all(self) -> None:
    """Discards everything staged, for teardown."""
    shutil.rmtree(self._directory, ignore_errors=True)

  def _path(self, version: int) -> str:
    return os.path.join(self._directory, f"weights-v{version}.npz")
