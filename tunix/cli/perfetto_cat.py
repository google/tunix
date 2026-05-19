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
both the input directory and the output file.

Memory: the concatenation is streamed in fixed-size chunks (default 8 MiB)
from each shard directly into the output handle, so multi-gigabyte traces
reassemble without buffering the full trace in RAM. The chunk size is tunable
via ``--chunk-bytes``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from typing import BinaryIO

from etils import epath


_MANIFEST_FILE = "trace.manifest.json"
_PENDING_FILE = "trace.shard_pending.binpb"
_SHARD_FILE_RE = re.compile(r"^trace\.shard_(\d{4,})\.binpb$")

# Default streaming chunk size. Sized to balance syscall overhead against
# peak RSS during concatenation: 8 MiB is small enough to keep memory flat
# even for traces in the hundreds of gigabytes, and large enough that read
# throughput is dominated by storage bandwidth rather than read() overhead.
_DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024


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


def _copy_file_to(
    src: epath.Path,
    dst: BinaryIO,
    chunk_bytes: int,
) -> int:
  """Streams ``src`` into ``dst`` in ``chunk_bytes``-sized reads.

  Returns the total number of bytes copied.
  """
  total = 0
  with src.open("rb") as f:
    while True:
      chunk = f.read(chunk_bytes)
      if not chunk:
        break
      dst.write(chunk)
      total += len(chunk)
  return total


def stream_trace_to(
    trace_dir: epath.Path,
    output: BinaryIO,
    *,
    include_pending: bool = True,
    chunk_bytes: int = _DEFAULT_CHUNK_BYTES,
) -> int:
  """Streams all trace fragments under ``trace_dir`` into ``output``.

  Sealed shards are emitted in concatenation order, followed by the in-flight
  pending file if it exists and ``include_pending`` is True. Memory use stays
  bounded to ~``chunk_bytes`` regardless of total trace size, so this is the
  preferred entry point for multi-gigabyte traces.

  Args:
    trace_dir: Directory containing the sharded trace.
    output: A writable binary file-like object (e.g. ``open(path, 'wb')``,
      ``sys.stdout.buffer``, or an ``epath.Path.open('wb')`` handle).
    include_pending: When True, append ``trace.shard_pending.binpb`` (if it
      exists) so the result contains in-flight data too.
    chunk_bytes: Read/write chunk size. Must be positive.

  Returns:
    The total number of bytes written to ``output``.
  """
  if chunk_bytes <= 0:
    raise ValueError(f"chunk_bytes must be positive, got {chunk_bytes!r}")
  total = 0
  for shard in list_sealed_shards(trace_dir):
    total += _copy_file_to(shard, output, chunk_bytes)
  if include_pending:
    pending = trace_dir / _PENDING_FILE
    if pending.exists():
      total += _copy_file_to(pending, output, chunk_bytes)
  return total


def concat_trace(
    trace_dir: epath.Path,
    *,
    include_pending: bool = True,
) -> bytes:
  """Concatenates all trace fragments under ``trace_dir`` into one ``bytes``.

  Convenience wrapper for callers that want the bytes in-process (tests,
  small ad-hoc analyses). Buffers the full trace in memory; prefer
  :func:`stream_trace_to` for large traces.

  Args:
    trace_dir: Directory containing the sharded trace.
    include_pending: When True, append ``trace.shard_pending.binpb`` (if it
      exists).

  Returns:
    The concatenated trace bytes.
  """
  buf = io.BytesIO()
  stream_trace_to(trace_dir, buf, include_pending=include_pending)
  return buf.getvalue()


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
  parser.add_argument(
      "--chunk-bytes",
      type=int,
      default=_DEFAULT_CHUNK_BYTES,
      help=(
          "Streaming chunk size in bytes. Defaults to %(default)d (8 MiB)."
          " Memory use stays bounded to roughly this value regardless of"
          " total trace size."
      ),
  )
  return parser


def _has_any_input(
    trace_dir: epath.Path, *, include_pending: bool
) -> bool:
  if list_sealed_shards(trace_dir):
    return True
  if include_pending and (trace_dir / _PENDING_FILE).exists():
    return True
  return False


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

  include_pending = not args.no_pending
  if not _has_any_input(trace_dir, include_pending=include_pending):
    print(
        f"No trace files found under {trace_dir}; nothing to concatenate.",
        file=sys.stderr,
    )
    return 1

  if args.output == "-":
    stream_trace_to(
        trace_dir,
        sys.stdout.buffer,
        include_pending=include_pending,
        chunk_bytes=args.chunk_bytes,
    )
    sys.stdout.buffer.flush()
  else:
    out_path = epath.Path(args.output)
    with out_path.open("wb") as f:
      stream_trace_to(
          trace_dir,
          f,
          include_pending=include_pending,
          chunk_bytes=args.chunk_bytes,
      )
  return 0


if __name__ == "__main__":
  sys.exit(main())
