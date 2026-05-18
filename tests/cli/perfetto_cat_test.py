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

import io
import json
import os
import tempfile
from unittest import mock

from absl.testing import absltest
from etils import epath
from tunix.cli import perfetto_cat


def _populate_trace_dir(
    tmp_dir,
    *,
    sealed_shards=("trace.shard_0001.binpb", "trace.shard_0002.binpb"),
    pending_content=b"PENDING_BYTES",
    manifest=True,
):
  """Writes a fixture trace directory with known byte payloads per file."""
  for name in sealed_shards:
    epath.Path(os.path.join(tmp_dir, name)).write_bytes(name.encode())
  if pending_content is not None:
    epath.Path(os.path.join(tmp_dir, "trace.shard_pending.binpb")).write_bytes(
        pending_content
    )
  if manifest:
    epath.Path(os.path.join(tmp_dir, "trace.manifest.json")).write_text(
        json.dumps(
            {
                "version": 1,
                "shard_steps": 1,
                "sealed_shards": list(sealed_shards),
                "sealed_step_count": len(sealed_shards),
                "pending_file": "trace.shard_pending.binpb",
            }
        )
    )


class ListSealedShardsTest(absltest.TestCase):

  def test_uses_manifest_order_when_present(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(
          tmp_dir,
          sealed_shards=("trace.shard_0001.binpb", "trace.shard_0002.binpb"),
      )
      shards = perfetto_cat.list_sealed_shards(epath.Path(tmp_dir))
      names = [p.name for p in shards]
      self.assertEqual(
          names, ["trace.shard_0001.binpb", "trace.shard_0002.binpb"]
      )

  def test_falls_back_to_glob_when_manifest_missing(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(
          tmp_dir,
          sealed_shards=(
              "trace.shard_0002.binpb",
              "trace.shard_0010.binpb",
              "trace.shard_0001.binpb",
          ),
          manifest=False,
      )
      shards = perfetto_cat.list_sealed_shards(epath.Path(tmp_dir))
      names = [p.name for p in shards]
      self.assertEqual(
          names,
          [
              "trace.shard_0001.binpb",
              "trace.shard_0002.binpb",
              "trace.shard_0010.binpb",
          ],
      )

  def test_falls_back_to_glob_when_manifest_lists_missing_shard(self):
    """If the manifest references a shard that isn't on disk, glob fallback
    must kick in rather than silently dropping the file from the output."""
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(
          tmp_dir,
          sealed_shards=("trace.shard_0001.binpb",),
      )
      # Manifest claims a second shard that doesn't exist.
      epath.Path(os.path.join(tmp_dir, "trace.manifest.json")).write_text(
          json.dumps(
              {
                  "version": 1,
                  "shard_steps": 1,
                  "sealed_shards": [
                      "trace.shard_0001.binpb",
                      "trace.shard_0099.binpb",
                  ],
                  "sealed_step_count": 2,
                  "pending_file": "trace.shard_pending.binpb",
              }
          )
      )
      shards = perfetto_cat.list_sealed_shards(epath.Path(tmp_dir))
      # Only the on-disk shard should appear; manifest is treated as a hint.
      self.assertEqual([p.name for p in shards], ["trace.shard_0001.binpb"])


class ConcatTraceTest(absltest.TestCase):

  def test_concat_includes_pending_by_default(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(tmp_dir, pending_content=b"PENDING")
      payload = perfetto_cat.concat_trace(epath.Path(tmp_dir))
      self.assertEqual(
          payload, b"trace.shard_0001.binpb" + b"trace.shard_0002.binpb" + b"PENDING"
      )

  def test_concat_skips_pending_when_requested(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(tmp_dir, pending_content=b"PENDING")
      payload = perfetto_cat.concat_trace(
          epath.Path(tmp_dir), include_pending=False
      )
      self.assertEqual(
          payload, b"trace.shard_0001.binpb" + b"trace.shard_0002.binpb"
      )

  def test_concat_handles_missing_pending(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(tmp_dir, pending_content=None)
      payload = perfetto_cat.concat_trace(epath.Path(tmp_dir))
      self.assertEqual(
          payload, b"trace.shard_0001.binpb" + b"trace.shard_0002.binpb"
      )


class MainTest(absltest.TestCase):

  def test_main_writes_to_output_file(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(tmp_dir, pending_content=b"P")
      out_path = os.path.join(tmp_dir, "combined.binpb")
      rc = perfetto_cat.main([tmp_dir, "-o", out_path])
      self.assertEqual(rc, 0)
      self.assertEqual(
          epath.Path(out_path).read_bytes(),
          b"trace.shard_0001.binpb" + b"trace.shard_0002.binpb" + b"P",
      )

  def test_main_writes_to_stdout(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      _populate_trace_dir(tmp_dir, pending_content=b"P")
      fake_stdout = io.BytesIO()
      fake_wrapper = mock.MagicMock()
      fake_wrapper.buffer = fake_stdout
      with mock.patch("sys.stdout", fake_wrapper):
        rc = perfetto_cat.main([tmp_dir])
      self.assertEqual(rc, 0)
      self.assertEqual(
          fake_stdout.getvalue(),
          b"trace.shard_0001.binpb" + b"trace.shard_0002.binpb" + b"P",
      )

  def test_main_missing_directory_returns_error(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      missing = os.path.join(tmp_dir, "does-not-exist")
      rc = perfetto_cat.main([missing, "-o", "/dev/null"])
      self.assertEqual(rc, 1)

  def test_main_empty_directory_returns_error(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      rc = perfetto_cat.main([tmp_dir, "-o", os.path.join(tmp_dir, "out.bin")])
      self.assertEqual(rc, 1)


if __name__ == "__main__":
  absltest.main()
