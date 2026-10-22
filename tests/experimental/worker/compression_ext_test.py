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

"""Unit tests for `_compression_ext.cc` and Bulk TCP LZ4 framing helpers."""

from concurrent import futures
import os
import struct
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.worker import _compression_ext
from tunix.experimental.worker import network as network_lib
from tunix.rl import packing


def _ref_decode_lz4_block(block: bytes, expected_len: int) -> bytes:
  """Spec-direct reference LZ4 raw block decoder."""
  out = bytearray()
  pos = 0
  n = len(block)
  while pos < n:
    token = block[pos]
    pos += 1
    lit_len = token >> 4
    match_code = token & 0x0F
    if lit_len == 15:
      while True:
        if pos >= n:
          raise ValueError("Truncated literal length")
        b = block[pos]
        pos += 1
        lit_len += b
        if b != 255:
          break
    if pos + lit_len > n:
      raise ValueError("Truncated literal run")
    out.extend(block[pos : pos + lit_len])
    pos += lit_len
    if pos == n:
      break
    if pos + 2 > n:
      raise ValueError("Truncated match offset")
    offset = block[pos] | (block[pos + 1] << 8)
    pos += 2
    if offset == 0 or offset > len(out):
      raise ValueError(f"Invalid match offset {offset}")
    match_len = match_code + 4
    if match_code == 15:
      while True:
        if pos >= n:
          raise ValueError("Truncated match length")
        b = block[pos]
        pos += 1
        match_len += b
        if b != 255:
          break
    for _ in range(match_len):
      out.append(out[-offset])
  if len(out) != expected_len:
    raise ValueError(f"Decoded length {len(out)} != expected {expected_len}")
  return bytes(out)


def _ref_encode_lz4_block(raw: bytes) -> bytes:
  """Spec-direct reference LZ4 raw block encoder (keeps final 5 literals)."""
  n = len(raw)
  if n < 13:
    out = bytearray()
    if n < 15:
      out.append(n << 4)
    else:
      out.append(15 << 4)
      rem = n - 15
      while rem >= 255:
        out.append(255)
        rem -= 255
      out.append(rem)
    out.extend(raw)
    return bytes(out)

  out = bytearray()
  anchor = 0
  ip = 0
  limit = n - 12
  last_literals = n - 5
  seen: dict[bytes, int] = {}
  while ip <= limit:
    key = raw[ip : ip + 4]
    ref = seen.get(key, -1)
    seen[key] = ip
    if ref < 0 or (ip - ref) > 65535 or raw[ref : ref + 4] != key:
      ip += 1
      continue
    lit_len = ip - anchor
    offset = ip - ref
    m_len = 4
    while (
        ip + m_len < last_literals and raw[ref + m_len] == raw[ip + m_len]
    ):
      m_len += 1
    m_code = m_len - 4
    token = ((15 if lit_len >= 15 else lit_len) << 4) | (
        15 if m_code >= 15 else m_code
    )
    out.append(token)
    if lit_len >= 15:
      rem = lit_len - 15
      while rem >= 255:
        out.append(255)
        rem -= 255
      out.append(rem)
    out.extend(raw[anchor:ip])
    out.append(offset & 0xFF)
    out.append((offset >> 8) & 0xFF)
    if m_code >= 15:
      rem = m_code - 15
      while rem >= 255:
        out.append(255)
        rem -= 255
      out.append(rem)
    ip += m_len
    anchor = ip
  tail_len = n - anchor
  if tail_len < 15:
    out.append(tail_len << 4)
  else:
    out.append(15 << 4)
    rem = tail_len - 15
    while rem >= 255:
      out.append(255)
      rem -= 255
    out.append(rem)
  out.extend(raw[anchor:])
  return bytes(out)


class CompressionExtTest(absltest.TestCase):

  def test_lz4_bulk_codec_roundtrip_and_incompressible_bypass(self):
    rng = np.random.default_rng(12345)
    # 1. Realistic MoE routed_experts (uint8 Mode 1 and uint10 Mode 2) with
    # ~45% unset padding (-1) and skewed top-8 expert tuple selection.
    expert_tuples = np.array(
        [
            [0, 1, 2, 3, 5, 8, 13, 21],
            [0, 1, 2, 4, 7, 11, 18, 29],
            [1, 2, 3, 5, 8, 13, 21, 34],
            [0, 2, 4, 6, 8, 10, 12, 14],
            [3, 5, 8, 13, 21, 34, 55, 89],
            [1, 3, 7, 15, 31, 63, 95, 127],
        ],
        dtype=np.int16,
    )
    raw_i16 = np.full((8, 1024, 16, 8), -1, dtype=np.int16)
    tuple_idx = rng.integers(0, len(expert_tuples), size=(8, 560, 16))
    raw_i16[:, :560, :, :] = expert_tuples[tuple_idx]
    for num_experts in (128, 384):
      packed = packing.encode_routed_experts(
          raw_i16, num_experts=num_experts, batched=True
      )
      mv = memoryview(packed).cast("B")
      try:
        comp = _compression_ext.try_compress_bulk_buffer(mv, 0.85, 256 * 1024)
      finally:
        mv.release()
      self.assertIsNotNone(comp)
      self.assertLess(comp.nbytes, int(packed.nbytes * 0.45))
      out = np.empty(packed.nbytes, dtype=np.uint8)
      _compression_ext.decompress_bulk_buffer(comp, out)
      np.testing.assert_array_equal(out.reshape(packed.shape), packed)
      np.testing.assert_array_equal(
          packing.decode_routed_experts(
              out.reshape(packed.shape), batched=True
          ),
          raw_i16,
      )

    # 2. Power-of-two batched tensor (B=64) with high-entropy prefix and
    # row-tail padding (-1) does not alias with the prime-13 sample probe.
    row_tail_i16 = np.full((64, 512, 8, 8), -1, dtype=np.int16)
    row_tail_i16[:, :280, :, :] = rng.integers(
        0, 896, size=(64, 280, 8, 8), dtype=np.int16
    )
    row_tail_u10 = packing.encode_routed_experts(
        row_tail_i16, num_experts=896, batched=True
    )
    row_tail_mv = memoryview(row_tail_u10).cast("B")
    try:
      comp_tail = _compression_ext.try_compress_bulk_buffer(
          row_tail_mv, 0.85, 256 * 1024
      )
    finally:
      row_tail_mv.release()
    self.assertIsNotNone(comp_tail)
    out_tail = np.empty(row_tail_u10.nbytes, dtype=np.uint8)
    _compression_ext.decompress_bulk_buffer(comp_tail, out_tail)
    np.testing.assert_array_equal(
        out_tail.reshape(row_tail_u10.shape), row_tail_u10
    )

    # 3. High-entropy incompressible float32 buffer is bypassed (< 2 us probe).
    incomp = rng.standard_normal(256 * 1024, dtype=np.float32)
    incomp_mv = memoryview(incomp).cast("B")
    try:
      self.assertIsNone(
          _compression_ext.try_compress_bulk_buffer(incomp_mv, 0.85, 256 * 1024)
      )
    finally:
      incomp_mv.release()

    # 3. Multi-block buffer mixing compressible (0x00) blocks with a raw
    # incompressible 256 KiB block (exercising 0x80000000U raw-block flag).
    mixed_blocks = np.zeros(4 * 256 * 1024, dtype=np.uint8)
    mixed_blocks[256 * 1024 : 512 * 1024] = rng.integers(
        0, 256, size=256 * 1024, dtype=np.uint8
    )
    comp_mixed = _compression_ext.try_compress_bulk_buffer(
        mixed_blocks, 0.85, 256 * 1024
    )
    self.assertIsNotNone(comp_mixed)
    out_mixed = np.empty_like(mixed_blocks)
    _compression_ext.decompress_bulk_buffer(comp_mixed, out_mixed)
    np.testing.assert_array_equal(out_mixed, mixed_blocks)

    # 4. Periodic overlapping matches (1 <= offset < match_len) across periods.
    for period in (1, 2, 5, 8, 9, 10, 15, 23):
      pattern = np.arange(1, period + 1, dtype=np.uint8)
      src = np.tile(pattern, (64 * 1024) // period + 1)[: 64 * 1024]
      comp_per = _compression_ext.try_compress_bulk_buffer(
          src, max_ratio=0.85, block_size=8192
      )
      self.assertIsNotNone(comp_per)
      dst_per = np.empty_like(src)
      _compression_ext.decompress_bulk_buffer(comp_per, dst_per)
      np.testing.assert_array_equal(dst_per, src)

    # 5. Corrupt frame header, mismatched size, and parallel multi-block
    # (>= 4 blocks) corrupt block payload fail loudly without killing threads.
    bad_frame = np.zeros(32, dtype=np.uint8)
    dst_32 = np.empty(32, dtype=np.uint8)
    with self.assertRaisesRegex(ValueError, "Invalid LZ4 frame: magic"):
      _compression_ext.decompress_bulk_buffer(bad_frame, dst_32)
    corrupt_blk = comp_mixed.copy()
    corrupt_blk[20 + 4 * 4] = 0xFF
    with self.assertRaisesRegex(ValueError, "Corrupted LZ4 block"):
      _compression_ext.decompress_bulk_buffer(corrupt_blk, out_mixed)

  def test_lz4_bulk_codec_reference_block_conformance(self):
    # 1. Direction A (C++ encoder -> Reference LZ4 raw block decoder):
    # Include >300-byte literal runs, >300-byte match runs, overlapping RLE,
    # and multi-block boundaries.
    rng = np.random.default_rng(1234)
    part_literals = rng.integers(0, 256, size=400, dtype=np.uint8)
    part_rle = np.tile(np.array([7, 13, 29], dtype=np.uint8), 600)
    part_repeats = np.tile(np.arange(64, dtype=np.uint8), 80)
    raw_arr = np.concatenate(
        [part_repeats, part_literals, part_rle, part_repeats]
    )
    block_size = 2048
    frame_arr = _compression_ext.try_compress_bulk_buffer(
        raw_arr, max_ratio=0.90, block_size=block_size
    )
    self.assertIsNotNone(frame_arr)
    frame_bytes = frame_arr.tobytes()
    self.assertEqual(frame_bytes[:4], b"TLZ4")
    raw_size, blk_sz, num_blocks = struct.unpack("<QII", frame_bytes[4:20])
    self.assertEqual(raw_size, raw_arr.nbytes)
    self.assertEqual(blk_sz, block_size)
    dir_entries = struct.unpack(
        f"<{num_blocks}I", frame_bytes[20 : 20 + 4 * num_blocks]
    )
    cursor = 20 + 4 * num_blocks

    reconstructed = bytearray()
    compressed_block_count = 0
    for b_idx, entry in enumerate(dir_entries):
      is_raw = (entry & 0x80000000) != 0
      csize = entry & 0x7FFFFFFF
      blk_payload = frame_bytes[cursor : cursor + csize]
      cursor += csize
      expected_blk_raw = min(block_size, raw_arr.nbytes - b_idx * block_size)
      if is_raw:
        reconstructed.extend(blk_payload)
      else:
        compressed_block_count += 1
        reconstructed.extend(
            _ref_decode_lz4_block(blk_payload, expected_blk_raw)
        )
    self.assertLen(frame_bytes, cursor)
    self.assertGreater(compressed_block_count, 0)
    self.assertEqual(bytes(reconstructed), raw_arr.tobytes())

    # 2. Direction B (Reference LZ4 raw block encoder -> C++ decoder):
    ref_blocks: list[bytes] = []
    ref_dir: list[int] = []
    raw_bytes = raw_arr.tobytes()
    for off in range(0, len(raw_bytes), block_size):
      chunk = raw_bytes[off : off + block_size]
      enc_blk = _ref_encode_lz4_block(chunk)
      if len(enc_blk) < len(chunk):
        ref_blocks.append(enc_blk)
        ref_dir.append(len(enc_blk))
      else:
        ref_blocks.append(chunk)
        ref_dir.append(0x80000000 | len(chunk))
    ref_frame = bytearray()
    ref_frame.extend(b"TLZ4")
    ref_frame.extend(
        struct.pack("<QII", len(raw_bytes), block_size, len(ref_blocks))
    )
    for d in ref_dir:
      ref_frame.extend(struct.pack("<I", d))
    for blk in ref_blocks:
      ref_frame.extend(blk)

    dst_from_ref = np.empty(len(raw_bytes), dtype=np.uint8)
    _compression_ext.decompress_bulk_buffer(
        np.frombuffer(bytes(ref_frame), dtype=np.uint8), dst_from_ref
    )
    np.testing.assert_array_equal(dst_from_ref, raw_arr)

  def test_lz4_bulk_codec_randomized_corruption_fuzz(self):
    rng = np.random.default_rng(20261011)
    # Build a multi-block frame with both compressed and raw blocks.
    raw = np.zeros(4 * 4096 + 517, dtype=np.uint8)
    raw[:4096] = np.tile(np.arange(16, dtype=np.uint8), 256)
    raw[4096:8192] = rng.integers(0, 256, size=4096, dtype=np.uint8)
    raw[8192:12288] = np.tile(np.array([1, 2, 3, 4, 5], dtype=np.uint8), 820)[
        :4096
    ]
    valid_frame = _compression_ext.try_compress_bulk_buffer(
        raw, max_ratio=0.95, block_size=4096
    )
    self.assertIsNotNone(valid_frame)
    dst = np.empty_like(raw)

    # 1. Deterministic boundary corruptions on header, directory, and block
    # tokens/offsets.
    boundary_values = (
        0,
        1,
        0x7FFFFFFF,
        0x80000000,
        0x80000001,
        0xFFFFFFFF,
    )
    for offset in (4, 8, 12, 16, 20, 24, 28, 32, 36):
      for val in boundary_values:
        mut = valid_frame.copy()
        mut[offset : offset + 4] = np.frombuffer(
            struct.pack("<I", val), dtype=np.uint8
        )
        if np.array_equal(mut, valid_frame):
          continue
        with self.assertRaises(ValueError):
          _compression_ext.decompress_bulk_buffer(mut, dst)

    # 2. Multi-seed randomized truncation and byte/word corruption fuzzing
    # across 250 seeds (run under ASan via --config=asan as well).
    header_and_dir_len = 20 + 5 * 4
    for seed in range(250):
      sub_rng = np.random.default_rng(seed)
      mode = seed % 4
      if mode == 0:
        # Random truncation anywhere before the final byte.
        cut = int(sub_rng.integers(0, len(valid_frame)))
        with self.assertRaises(ValueError):
          _compression_ext.decompress_bulk_buffer(valid_frame[:cut], dst)
      elif mode == 1:
        # Corrupt header or block directory entry.
        mut = valid_frame.copy()
        pos = int(sub_rng.integers(0, header_and_dir_len))
        mut[pos] ^= np.uint8(int(sub_rng.integers(1, 256)))
        with self.assertRaises(ValueError):
          _compression_ext.decompress_bulk_buffer(mut, dst)
      elif mode == 2:
        # Append trailing garbage bytes.
        extra = sub_rng.integers(
            0, 256, size=int(sub_rng.integers(1, 32)), dtype=np.uint8
        )
        mut = np.concatenate([valid_frame, extra])
        with self.assertRaises(ValueError):
          _compression_ext.decompress_bulk_buffer(mut, dst)
      else:
        # Corrupt compressed block stream (tokens, offsets, match lengths).
        mut = valid_frame.copy()
        num_flips = int(sub_rng.integers(1, 8))
        positions = sub_rng.integers(
            header_and_dir_len, len(valid_frame), size=num_flips
        )
        for pos in positions:
          mut[int(pos)] = np.uint8(int(sub_rng.integers(0, 256)))
        try:
          _compression_ext.decompress_bulk_buffer(mut, dst)
        except ValueError:
          pass

  def test_lz4_bulk_codec_multithreaded_concurrency(self):
    # Verify concurrent multi-block encode/decode + incompressible rejection
    # across 8 Python threads (exercising g_pool_mu_, Lz4WorkerPool, and nogil
    # return paths concurrently).
    rng = np.random.default_rng(999)
    compressible = np.tile(np.arange(32, dtype=np.uint8), 32 * 1024)
    incompressible = rng.integers(0, 256, size=1024 * 1024, dtype=np.uint8)

    def _worker(tid: int) -> None:
      for step in range(8):
        if (tid + step) % 3 == 0:
          self.assertIsNone(
              _compression_ext.try_compress_bulk_buffer(
                  incompressible, max_ratio=0.85, block_size=64 * 1024
              )
          )
        else:
          src = compressible + np.uint8(tid + step)
          frame = _compression_ext.try_compress_bulk_buffer(
              src, max_ratio=0.85, block_size=64 * 1024
          )
          self.assertIsNotNone(frame)
          dst = np.empty_like(src)
          _compression_ext.decompress_bulk_buffer(frame, dst)
          np.testing.assert_array_equal(dst, src)

    with futures.ThreadPoolExecutor(max_workers=8) as pool:
      futs = [pool.submit(_worker, i) for i in range(8)]
      for fut in futs:
        fut.result()

  def test_parse_bulk_lz4_env_and_validate_bulk_manifest_compressed_lengths(
      self,
  ):
    # 1. _parse_bulk_lz4_env: default (unset), truthy, falsy, case/whitespace,
    # and invalid value raising ValueError.
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("TUNIX_BULK_TCP_LZ4", None)
      self.assertTrue(network_lib._parse_bulk_lz4_env())

    for truthy in ("1", "true", "yes", "on", "TRUE", "  Yes \n", "\tON "):
      with self.subTest(env_val=truthy):
        with mock.patch.dict(os.environ, {"TUNIX_BULK_TCP_LZ4": truthy}):
          self.assertTrue(network_lib._parse_bulk_lz4_env())

    for falsy in ("0", "false", "no", "off", "FALSE", "  No \n", "\tOFF "):
      with self.subTest(env_val=falsy):
        with mock.patch.dict(os.environ, {"TUNIX_BULK_TCP_LZ4": falsy}):
          self.assertFalse(network_lib._parse_bulk_lz4_env())

    for invalid in ("maybe", "2", "-1", "", "   ", "enabled", "disabled"):
      with self.subTest(invalid_env=invalid):
        with mock.patch.dict(os.environ, {"TUNIX_BULK_TCP_LZ4": invalid}):
          with self.assertRaisesRegex(
              ValueError, "TUNIX_BULK_TCP_LZ4 must be one of"
          ):
            network_lib._parse_bulk_lz4_env()

    # 2. _validate_bulk_manifest rejection of malformed compressed_lengths.
    tid = b"a" * 16
    header_len = 128
    buffer_lengths = (1024, 2048)
    valid_comp = (512, 0)
    valid_layout = network_lib._plan_bulk_buffer_layout(
        header_len, buffer_lengths, valid_comp
    )
    valid_manifest = network_lib._BulkManifest(
        op=network_lib._OP_PUSH,
        tid=tid,
        port=12345,
        total_bulk_bytes=valid_layout.total_bulk_bytes,
        header_len=header_len,
        buffer_lengths=buffer_lengths,
        compressed_lengths=valid_comp,
    )
    self.assertEqual(
        network_lib._validate_bulk_manifest(valid_manifest), valid_layout
    )

    bad_compressed_cases = (
        ("wrong_len_short", (512,)),
        ("wrong_len_long", (512, 0, 0)),
        ("negative_entry", (-1, 0)),
        ("equal_to_buffer_len", (1024, 0)),
        ("greater_than_buffer_len", (1025, 0)),
        ("compressed_for_zero_len_buf", (0, 2048)),
        ("bool_instead_of_int", (True, 0)),
        ("float_instead_of_int", (512.0, 0)),  # type: ignore[arg-type]
        ("list_instead_of_tuple", [512, 0]),  # type: ignore[arg-type]
    )
    for case_name, bad_comp in bad_compressed_cases:
      with self.subTest(case_name=case_name):
        bad_manifest = valid_manifest._replace(compressed_lengths=bad_comp)
        with self.assertRaisesRegex(
            ValueError, "Invalid bulk manifest compressed_lengths"
        ):
          network_lib._validate_bulk_manifest(bad_manifest)

    # Also verify compressed_lengths > 0 on a zero-length buffer is rejected.
    zero_buf_manifest = network_lib._BulkManifest(
        op=network_lib._OP_PUSH,
        tid=tid,
        port=12345,
        total_bulk_bytes=header_len,
        header_len=header_len,
        buffer_lengths=(0,),
        compressed_lengths=(1,),
    )
    with self.assertRaisesRegex(
        ValueError, "Invalid bulk manifest compressed_lengths"
    ):
      network_lib._validate_bulk_manifest(zero_buf_manifest)


if __name__ == "__main__":
  absltest.main()
