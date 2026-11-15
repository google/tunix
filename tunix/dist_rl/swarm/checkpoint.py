# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Durable local round manifests pointing at immutable trainer checkpoints.

Callers must wait for the referenced model/optimizer checkpoint before writing.
This helper commits only the small control-state file; it does not synchronize
remote checkpoint stores or make several independent optimizers atomic.
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

FORMAT_VERSION = 1


def _encode(state):
  return json.dumps(
      state, sort_keys=True, separators=(",", ":"), allow_nan=False
  ).encode()


def write_manifest(path, state):
  """Atomically publish JSON state after fsync; preserve the previous on failure."""
  if not isinstance(state, dict):
    raise TypeError("Round state must be a JSON object.")
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  raw = _encode(state)
  envelope = {
      "format_version": FORMAT_VERSION,
      "sha256": hashlib.sha256(raw).hexdigest(),
      "state": state,
  }
  filename = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix="." + path.name, delete=False
    ) as stream:
      filename = stream.name
      stream.write(_encode(envelope) + b"\n")
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(filename, path)
    filename = None
    fd = os.open(path.parent, os.O_RDONLY)
    try:
      os.fsync(fd)
    finally:
      os.close(fd)
  finally:
    if filename is not None:
      Path(filename).unlink(missing_ok=True)


def read_manifest(path):
  """Read and integrity-check one committed round state."""
  envelope = json.loads(Path(path).read_text())
  if (
      not isinstance(envelope, dict)
      or envelope.get("format_version") != FORMAT_VERSION
  ):
    raise ValueError("Unknown round-manifest format.")
  state = envelope.get("state")
  if (
      not isinstance(state, dict)
      or envelope.get("sha256") != hashlib.sha256(_encode(state)).hexdigest()
  ):
    raise ValueError("Round-manifest integrity check failed.")
  return state
