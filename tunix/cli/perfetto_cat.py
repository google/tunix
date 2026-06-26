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

"""Concatenate sharded perfetto trace files into a single trace.

Long-running jobs split their perfetto trace into a directory of sharded files
plus a single in-flight pending file::

    trace.shard_0001.binpb
    trace.shard_0002.binpb
    ...
    trace.shard_pending.binpb
    trace.manifest.json

Perfetto's ``TracePacket`` format is concatenable, so a complete trace is
``cat trace.shard_*.binpb trace.shard_pending.binpb``. This module provides a
small CLI wrapper that reads the manifest (for ordering) and emits a single
file -- handy when the trace directory lives on remote storage and you want
one local file to drop into https://ui.perfetto.dev.

Usage::

    python -m tunix.cli.perfetto_cat <trace_dir>                  # writes to stdout
    python -m tunix.cli.perfetto_cat <trace_dir> -o trace.binpb    # writes to file
    python -m tunix.cli.perfetto_cat <trace_dir> --no-pending      # sealed only

Remote paths supported by ``etils.epath`` (e.g. ``gs://...``) are accepted for
the input directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys

from etils import epath


_MANIFEST_FILE = "trace.manifest.json"
_PENDING_FILE = "trace.shard_pending.binpb"
_SHARD_FILE_RE = re.compile(r"^trace\.shard_(\d{4,})\.binpb$")


def _shard_index(name: str) -> int | None:
  """Returns the numeric shard index for a sealed-shard filename, or None."""
  m = _SHARD_FILE_RE.match(name)
  return int(m.group(1)) if m else None


def list_sealed_shards(trace_dir: epath.Path) -> list[epath.Path]:
  """Lists sealed shard files in deterministic concatenation order.

  Prefers the manifest's sealed-shard list when one is present. Falls back to
  a glob-based listing sorted by the numeric shard index, which is the same
  ordering the writer produces. This lets the CLI work even on a directory
  whose manifest is missing or corrupt.

  Args:
    trace_dir: Directory containing the sharded trace files.

  Returns:
    The sealed shards in concatenation order.
  """
  manifest_path = trace_dir / _MANIFEST_FILE
  if manifest_path.exists():
    try:
      payload = json.loads(manifest_path.read_text())
      names = payload.get("sealed_shards") or []
      shards = [trace_dir / name for name in names]
      if all(p.exists() for p in shards):
        return shards
    except Exception:  # pylint: disable=broad-except
      # Fall through to glob-based discovery.
      pass

  found: list[tuple[int, epath.Path]] = []
  for child in trace_dir.iterdir():
    idx = _shard_index(child.name)
    if idx is None:
      continue
    found.append((idx, child))
  found.sort(key=lambda item: item[0])
  return [p for _, p in found]


def concat_trace(
    trace_dir: epath.Path,
    *,
    include_pending: bool = True,
) -> bytes:
  """Concatenates all trace fragments under ``trace_dir`` into a single blob.

  Args:
    trace_dir: Directory containing the sharded trace.
    include_pending: When True, append ``trace.shard_pending.binpb`` (if it
      exists) so the result contains in-flight data too.

  Returns:
    The concatenated trace bytes.
  """
  parts: list[bytes] = []
  for shard in list_sealed_shards(trace_dir):
    parts.append(shard.read_bytes())
  if include_pending:
    pending = trace_dir / _PENDING_FILE
    if pending.exists():
      parts.append(pending.read_bytes())
  return b"".join(parts)


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
      prog="python -m tunix.cli.perfetto_cat",
      description=(
          "Concatenate sharded perfetto trace files into a single binary"
          " trace suitable for https://ui.perfetto.dev."
      ),
  )
  parser.add_argument(
      "trace_dir",
      help=(
          "Directory containing trace.shard_NNNN.binpb files (and optional"
          " trace.shard_pending.binpb). Remote paths supported by etils.epath"
          " are accepted (e.g. gs://bucket/path)."
      ),
  )
  parser.add_argument(
      "-o",
      "--output",
      default="-",
      help=(
          "Destination file. Use '-' (the default) to write to stdout. Remote"
          " paths supported by etils.epath are accepted."
      ),
  )
  parser.add_argument(
      "--no-pending",
      action="store_true",
      help=(
          "Skip the in-flight pending file; emit only sealed shards. Useful"
          " when copying a completed trace; not needed for the live view."
      ),
  )
  return parser


def main(argv: list[str] | None = None) -> int:
  parser = _build_parser()
  args = parser.parse_args(argv)
  trace_dir = epath.Path(args.trace_dir)
  if not trace_dir.exists():
    print(f"Trace directory not found: {trace_dir}", file=sys.stderr)
    return 1
  if not trace_dir.is_dir():
    print(f"Not a directory: {trace_dir}", file=sys.stderr)
    return 1

  payload = concat_trace(trace_dir, include_pending=not args.no_pending)
  if not payload:
    print(
        f"No trace files found under {trace_dir}; nothing to concatenate.",
        file=sys.stderr,
    )
    return 1

  if args.output == "-":
    # Use the underlying buffer so we don't accidentally re-encode bytes.
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
  else:
    out_path = epath.Path(args.output)
    out_path.write_bytes(payload)
  return 0


if __name__ == "__main__":
  sys.exit(main())
