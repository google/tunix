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

"""Tests for bit-packed holding of router-replay routing on the orchestrator.

Three properties, each with a check that is shown to fail on a deliberately
broken implementation:
1. Exactness: every trainer micro-batch is byte-identical with and without
   `compact_held_routed_experts`, through the real adapter and assemblers.
2. Memory: a held trajectory's bytes drop to ~0.6x, and the dense routing array
   is actually freed.
3. Safety: anything the encoding cannot restore bit for bit stays dense.
"""

import asyncio
import copy
import dataclasses
import gc
import types
from unittest import mock
import weakref

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import distributed_rl_engine
from tunix.experimental.orchestrator import held_routing
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.worker import remote_execution
from tunix.rl import algorithm_config

UNSET = datatypes.UNSET_ROUTED_EXPERT
# Qwen3.5-397B-A17B: 60 MoE layers, top-10 of 512 experts.
NUM_LAYERS, TOP_K, NUM_EXPERTS = 60, 10, 512


def _routing(rng, tokens, num_layers=NUM_LAYERS, top_k=TOP_K, unset_tail=0):
  """Random top-k routing, distinct ids per token-layer, int16, -1 tail.

  Slot k is `(a + k * b) mod 512` with a random start `a` and a random odd
  stride `b`: distinct ids over all of [0, 512) without materialising a
  `[tokens, layers, 512]` array.
  """
  shape = (tokens, num_layers, 1)
  start = rng.integers(0, NUM_EXPERTS, size=shape)
  stride = 2 * rng.integers(0, NUM_EXPERTS // 2, size=shape) + 1
  ids = ((start + np.arange(top_k) * stride) % NUM_EXPERTS).astype(np.int16)
  if unset_tail:
    ids[-unset_tail:] = UNSET
  return ids


def _deepswe_item(
    rng,
    *,
    group,
    index,
    prompt_len,
    conv_len,
    routed_len=None,
    num_layers=NUM_LAYERS,
    routing="random",
):
  """A finished trajectory shaped like the collector's Token-mode output."""
  prompt = rng.integers(0, 50_000, prompt_len, dtype=np.int32)
  conv = rng.integers(0, 50_000, conv_len, dtype=np.int32)
  masks = np.zeros(conv_len, dtype=np.float32)
  masks[: conv_len // 3] = 1.0  # Assistant turns trained, env turns not.
  traj = {
      "prompt_tokens": prompt,
      "conversation_tokens": conv,
      "conversation_masks": masks,
      "old_logprobs": rng.normal(size=conv_len).astype(np.float32),
      "status": "SUCCEEDED",
      "trajectory_reward": float(index % 2),
      "conversation_text": [
          {"role": "user", "content": "x" * 64},
          {"role": "assistant", "content": "y" * 64},
      ],
      "policy_version": 0,
  }
  n = prompt_len + conv_len if routed_len is None else routed_len
  if routing == "random":
    traj["routed_experts"] = _routing(rng, n, num_layers=num_layers)
  elif routing == "adversarial":
    ids = _routing(rng, n, num_layers=num_layers)
    ids[1, 0, 3] = UNSET  # A partly unset token-layer.
    traj["routed_experts"] = ids
  elif routing == "out_of_range":
    ids = _routing(rng, n, num_layers=num_layers)
    ids[0, 1, 0] = 700
    traj["routed_experts"] = ids
  elif routing != "none":
    raise ValueError(routing)
  return datatypes.TrajectoryItem(
      prompt_id=f"p{group}",
      group_index=index,
      start_step=0,
      traj=traj,
      metadata={
          "prompt_id": f"p{group}",
          "group_index": index,
          "batch_idx": 0,
          "prompt_idx": group,
          "intra_batch_idx": group,
          "policy_version": 0,
      },
  )


def _array_owner(arr):
  """The object that owns `arr`'s memory (follows the `.base` chain)."""
  owner = arr
  while isinstance(owner, np.ndarray) and owner.base is not None:
    owner = owner.base
  return owner


def _held_bytes(obj):
  """Bytes of every distinct buffer reachable from `obj`.

  Counts each owning buffer once, so a small view that keeps a large base
  alive is charged the whole base, the way the allocator sees it.
  """
  seen_objs: set[int] = set()
  owners: dict[int, int] = {}
  stack = [obj]
  while stack:
    x = stack.pop()
    if id(x) in seen_objs:
      continue
    seen_objs.add(id(x))
    if isinstance(x, np.ndarray):
      owner = _array_owner(x)
      if isinstance(owner, np.ndarray):
        owners[id(owner)] = owner.nbytes
      else:
        owners[id(owner)] = memoryview(owner).nbytes
    elif isinstance(x, held_routing.BitPackedRoutedExperts):
      stack.extend([x.lo, x.hi])
    elif isinstance(x, dict):
      stack.extend(x.values())
    elif isinstance(x, (list, tuple)):
      stack.extend(x)
    elif isinstance(x, (str, bytes, int, float, bool, type(None))):
      continue
    elif dataclasses.is_dataclass(x):
      stack.extend(getattr(x, f.name) for f in dataclasses.fields(x))
      stack.extend(getattr(x, "__dict__", {}).values())
    elif hasattr(x, "__dict__"):
      stack.extend(vars(x).values())
  return sum(owners.values())


def _batch_fields(batch):
  """Every field of an assembled micro-batch, for byte comparison."""
  out = {
      "is_final_batch": batch.is_final_batch,
      "trajectory_ids": tuple(batch.trajectory_ids),
  }
  payload = batch.payload
  for field in dataclasses.fields(payload):
    value = getattr(payload, field.name)
    if field.name == "metadata":
      out["metadata.trajectory_ids"] = tuple(
          (value or {}).get("trajectory_ids", ())
      )
    else:
      out[field.name] = value
  return out


def batches_identical(a, b):
  """(True, "") if two micro-batch lists match field for field, byte for byte."""
  if len(a) != len(b):
    return False, f"{len(a)} vs {len(b)} micro-batches"
  for i, (ba, bb) in enumerate(zip(a, b)):
    fa, fb = _batch_fields(ba), _batch_fields(bb)
    if fa.keys() != fb.keys():
      return False, f"batch {i}: fields {sorted(fa)} vs {sorted(fb)}"
    for name in fa:
      va, vb = fa[name], fb[name]
      if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
        if not (
            isinstance(va, np.ndarray)
            and isinstance(vb, np.ndarray)
            and va.dtype == vb.dtype
            and va.shape == vb.shape
            and va.tobytes() == vb.tobytes()
        ):
          return False, f"batch {i}: field {name} differs"
      elif va != vb:
        return False, f"batch {i}: field {name}: {va!r} vs {vb!r}"
  return True, ""


def _groups(rng, *, num_groups=3, group_size=4, adversarial=False):
  """Groups of finished trajectories covering every alignment case."""
  groups = []
  for g in range(num_groups):
    group = []
    for i in range(group_size):
      prompt_len = int(rng.integers(3, 9))
      conv_len = int(rng.integers(5, 30))
      seq_len = prompt_len + conv_len
      kind = (g * group_size + i) % 5
      routed_len = {
          0: seq_len,  # Exact capture.
          1: seq_len - 1,  # Last token not prefilled.
          2: seq_len + 3,  # Longer than the payload: truncated.
          3: max(seq_len - 4, 1),  # Short capture: unset padding.
          4: seq_len,
      }[kind]
      routing = "random"
      if adversarial and i == 1:
        routing = "adversarial"
      if adversarial and i == 2:
        routing = "out_of_range"
      group.append(
          _deepswe_item(
              rng,
              group=g,
              index=i,
              prompt_len=prompt_len,
              conv_len=conv_len,
              routed_len=routed_len,
              num_layers=4,
              routing=routing,
          )
      )
    groups.append(group)
  return groups


def _adapter(group_size):
  # beta 0: the MLPerf recipe has no KL penalty, so no reference model.
  return algorithm_adapter.GRPOAdapter(
      algo_config=algorithm_config.GRPOConfig(
          num_generations=group_size, beta=0.0
      ),
      mini_batch_size=1,
  )


def _packed_assembler(group_size):
  return batch_assembly.SequencePackedBatchAssembler(
      batch_size=2,
      num_generations=group_size,
      mini_batch_size=2,
      max_packed_len=64,
      pad_id=0,
      segment_align_multiple=1,
  )


def _padded_assembler(group_size):
  return batch_assembly.PaddedBatchAssembler(
      batch_size=4,
      max_prompt_length=8,
      max_response_length=32,
      pad_id=0,
      num_generations=group_size,
      mini_batch_size=2,
  )


def _assemble(groups, make_assembler, *, compact_first):
  """Runs the real adapter and assembler over `groups`; returns micro-batches."""
  groups = copy.deepcopy(groups)
  group_size = len(groups[0])
  if compact_first:
    for group in groups:
      for item in group:
        held_routing.compact_item_routing(item)
  adapter = _adapter(group_size)
  assembler = make_assembler(group_size)
  batches = []
  for group in groups:
    rewards = [float(i % 2) for i in range(len(group))]
    payloads = [
        # As `StandardRLProgram._train_loop` does before feeding.
        dataclasses.replace(
            payload, metadata={**payload.metadata, "traj_id": item.traj_id}
        )
        for item, payload in zip(
            group, adapter.create_trainer_payloads(group, rewards=rewards)
        )
    ]
    batches.extend(assembler.feed(payloads))
  batches.extend(assembler.flush())
  return batches


class CodecTest(parameterized.TestCase):

  @parameterized.parameters(1, 2, 7, 8, 9, 10, 15)
  def test_round_trip_is_exact_for_every_top_k(self, top_k):
    rng = np.random.default_rng(top_k)
    ids = rng.integers(0, 512, size=(97, 3, top_k), dtype=np.int16)
    ids[[0, 5, 96]] = UNSET
    ids[7, 1] = UNSET  # One unset token-layer inside a set token.
    ids[9, 2, :] = 511
    ids[10, 0, :] = 256
    ids[11, 0, :] = 255
    packed = held_routing.compact(ids)
    self.assertIsNotNone(packed)
    out = held_routing.expand(packed)
    self.assertEqual(out.dtype, np.int16)
    self.assertEqual(out.shape, ids.shape)
    np.testing.assert_array_equal(out, ids)
    self.assertEqual(packed.nbytes, ids.size + 2 * ids.shape[0] * ids.shape[1])

  def test_round_trip_across_encode_blocks(self):
    rng = np.random.default_rng(0)
    # 1,200 token-layers per 20 tokens; 70,000 tokens x 1 layer spans two
    # 65,536-row encode blocks.
    ids = rng.integers(0, 512, size=(70_000, 1, 10), dtype=np.int16)
    ids[65_535:65_538] = UNSET
    np.testing.assert_array_equal(
        held_routing.expand(held_routing.compact(ids)), ids
    )

  def test_size_is_point_six_at_top_10(self):
    ids = _routing(np.random.default_rng(1), 1000)
    packed = held_routing.compact(ids)
    self.assertEqual(packed.dense_nbytes, ids.nbytes)
    self.assertAlmostEqual(packed.nbytes / ids.nbytes, 0.6)

  def test_wider_and_unsigned_inputs_match_the_dense_int16_cast(self):
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 512, size=(40, 2, 10))
    ids[3] = UNSET
    for dtype in (np.int32, np.int64, np.uint16):
      x = (
          ids.astype(dtype)
          if dtype != np.uint16
          else ids.astype(np.int16).view(np.uint16)
      )
      packed = held_routing.compact(x)
      self.assertIsNotNone(packed, dtype)
      np.testing.assert_array_equal(
          held_routing.expand(packed), np.asarray(x, dtype=np.int16)
      )

  @parameterized.named_parameters(
      ("partly_unset_token_layer", lambda a: a.__setitem__((2, 1, 4), UNSET)),
      ("id_512", lambda a: a.__setitem__((0, 0, 0), 512)),
      ("id_minus_2", lambda a: a.__setitem__((5, 1, 9), -2)),
      ("id_32767", lambda a: a.__setitem__((1, 0, 2), 32767)),
  )
  def test_refuses_what_it_cannot_restore(self, corrupt):
    ids = _routing(np.random.default_rng(3), 8, num_layers=2)
    corrupt(ids)
    self.assertIsNone(held_routing.compact(ids))

  def test_refuses_non_routing_inputs(self):
    ids = _routing(np.random.default_rng(4), 8, num_layers=2)
    self.assertIsNone(held_routing.compact(ids[:, 0]))  # 2-D.
    self.assertIsNone(held_routing.compact(ids.astype(np.float32)))
    self.assertIsNone(held_routing.compact(ids.tolist()))
    self.assertIsNone(
        held_routing.compact(np.zeros((4, 2, 16), dtype=np.int16))
    )  # top_k 16.
    wide = ids.astype(np.int32)
    wide[0, 0, 0] = 65535  # int16 cast would wrap to -1.
    self.assertIsNone(held_routing.compact(wide))
    wrapping = ids.astype(np.int32)
    wrapping[0, 0, 0] = 65536 + 4  # A 16-bit cast would wrap it to expert 4.
    self.assertIsNone(held_routing.compact(wrapping))

  def test_empty_routing(self):
    ids = np.zeros((0, 60, 10), dtype=np.int16)
    out = held_routing.expand(held_routing.compact(ids))
    self.assertEqual(out.shape, (0, 60, 10))
    self.assertEqual(out.dtype, np.int16)

  def test_is_not_array_like(self):
    packed = held_routing.compact(_routing(np.random.default_rng(5), 4))
    with self.assertRaises(TypeError):
      np.asarray(packed)
    with self.assertRaises(TypeError):
      np.asarray(packed, dtype=np.int16)

  def test_fit_tokens_matches_dense_alignment(self):
    ids = _routing(np.random.default_rng(6), 12, num_layers=3)
    packed = held_routing.compact(ids)
    np.testing.assert_array_equal(
        held_routing.expand(held_routing.fit_tokens(packed, 7)), ids[:7]
    )
    padded = np.full((20, 3, TOP_K), UNSET, dtype=np.int16)
    padded[:12] = ids
    np.testing.assert_array_equal(
        held_routing.expand(held_routing.fit_tokens(packed, 20)), padded
    )

  def test_does_not_modify_its_input(self):
    ids = _routing(np.random.default_rng(7), 30, unset_tail=3)
    before = ids.copy()
    held_routing.compact(ids)
    np.testing.assert_array_equal(ids, before)

  def test_survives_the_orchestrator_wire_format(self):
    packed = held_routing.compact(_routing(np.random.default_rng(8), 50))
    chunks = list(remote_execution._iter_serialized_chunks(packed))  # pylint: disable=protected-access
    back = remote_execution._deserialize_from_chunks(chunks)  # pylint: disable=protected-access
    self.assertTrue(held_routing.is_bitpacked(back))
    np.testing.assert_array_equal(
        held_routing.expand(back), held_routing.expand(packed)
    )


class CompactItemRoutingTest(absltest.TestCase):

  def test_replaces_every_reference_with_one_encoding(self):
    rng = np.random.default_rng(10)
    ids = _routing(rng, 40)
    item = datatypes.TrajectoryItem(
        traj={"routed_experts": ids}, metadata={"routed_experts": ids}
    )
    item.routed_experts = ids
    stats = held_routing.compact_item_routing(item)
    self.assertEqual(
        (stats.items, stats.compacted, stats.kept_dense, stats.errors),
        (1, 1, 0, 0),
    )
    packed = item.traj["routed_experts"]
    self.assertTrue(held_routing.is_bitpacked(packed))
    self.assertIs(item.metadata["routed_experts"], packed)
    self.assertIs(item.routed_experts, packed)
    np.testing.assert_array_equal(held_routing.expand(packed), ids)

  def test_frees_the_dense_array(self):
    rng = np.random.default_rng(11)
    item = _deepswe_item(rng, group=0, index=0, prompt_len=10, conv_len=90)
    dense_ref = weakref.ref(item.traj["routed_experts"])
    held_routing.compact_item_routing(item)
    gc.collect()
    self.assertIsNone(dense_ref())

  def test_frees_a_wire_buffer_kept_alive_only_by_the_routing_view(self):
    # The orchestrator receives routing as a view into a reassembly buffer
    # (pickle protocol 5). Replacing the view must free the whole buffer.
    rng = np.random.default_rng(12)
    item = _deepswe_item(rng, group=0, index=0, prompt_len=10, conv_len=90)
    chunks = list(remote_execution._iter_serialized_chunks(item))  # pylint: disable=protected-access
    received = remote_execution._deserialize_from_chunks(chunks)  # pylint: disable=protected-access
    owner = _array_owner(received.traj["routed_experts"])
    self.assertIsInstance(owner, np.ndarray)
    owner_ref = weakref.ref(owner)
    del owner
    held_routing.compact_item_routing(received)
    gc.collect()
    self.assertIsNone(owner_ref())

  def test_unrestorable_routing_stays_dense_and_untouched(self):
    rng = np.random.default_rng(13)
    item = _deepswe_item(
        rng, group=0, index=0, prompt_len=4, conv_len=8, routing="adversarial"
    )
    before = item.traj["routed_experts"]
    stats = held_routing.compact_item_routing(item)
    self.assertEqual((stats.compacted, stats.kept_dense), (0, 1))
    self.assertIs(item.traj["routed_experts"], before)

  def test_items_without_routing_are_untouched(self):
    item = datatypes.TrajectoryItem(traj={"status": "FAILED"})
    stats = held_routing.compact_item_routing(item)
    self.assertEqual(
        (stats.compacted, stats.kept_dense, stats.errors), (0, 0, 0)
    )
    self.assertNotIn("routed_experts", item.traj)

  def test_an_encoding_error_leaves_the_item_exactly_as_it_was(self):
    rng = np.random.default_rng(14)
    ids = _routing(rng, 20)
    item = datatypes.TrajectoryItem(
        traj={"routed_experts": ids}, metadata={"routed_experts": ids}
    )
    with mock.patch.object(
        held_routing, "compact", side_effect=RuntimeError("boom")
    ):
      stats = held_routing.compact_item_routing(item)
    self.assertEqual(stats.errors, 1)
    self.assertIs(item.traj["routed_experts"], ids)
    self.assertIs(item.metadata["routed_experts"], ids)


class PerTrajectoryBytesTest(absltest.TestCase):
  """MEASURES what one held trajectory costs before and after."""

  def test_a_deepswe_sized_trajectory_drops_to_point_six(self):
    # ~30k tokens per trajectory, as the production packing lines imply.
    rng = np.random.default_rng(20)
    item = _deepswe_item(
        rng, group=0, index=0, prompt_len=2_000, conv_len=28_000
    )
    dense = _held_bytes(item)
    routing = item.traj["routed_experts"].nbytes
    held_routing.compact_item_routing(item)
    gc.collect()
    held = _held_bytes(item)
    print(
        "\n[per-trajectory bytes] tokens=30000"
        f" routing_dense={routing / 1e6:.2f} MB total_dense={dense / 1e6:.2f}"
        f" MB total_held={held / 1e6:.2f} MB ratio={held / dense:.3f}"
    )
    self.assertEqual(routing, 30_000 * NUM_LAYERS * TOP_K * 2)
    self.assertGreater(routing / dense, 0.97)  # Routing is the trajectory.
    self.assertLess(held / dense, 0.62)
    self.assertGreater(held / dense, 0.58)

  def test_bytes_gate_fails_if_the_dense_array_is_kept(self):
    # Deliberately broken: encodes, but also keeps the dense array around.
    rng = np.random.default_rng(21)
    item = _deepswe_item(rng, group=0, index=0, prompt_len=100, conv_len=900)
    dense = _held_bytes(item)
    original = held_routing.compact

    def leaky_compact(ids):
      packed = original(ids)
      item.metadata["_debug_dense_copy"] = ids
      return packed

    with mock.patch.object(held_routing, "compact", leaky_compact):
      held_routing.compact_item_routing(item)
    self.assertGreater(_held_bytes(item) / dense, 1.5)


class TrainerPayloadIdentityTest(parameterized.TestCase):
  """The trainer sees the same bytes with and without held compaction."""

  @parameterized.named_parameters(
      ("sequence_packed", _packed_assembler, False),
      ("sequence_packed_adversarial", _packed_assembler, True),
      ("padded", _padded_assembler, False),
      ("padded_adversarial", _padded_assembler, True),
  )
  def test_micro_batches_are_byte_identical(self, make_assembler, adversarial):
    groups = _groups(np.random.default_rng(30), adversarial=adversarial)
    dense = _assemble(groups, make_assembler, compact_first=False)
    held = _assemble(groups, make_assembler, compact_first=True)
    self.assertNotEmpty(dense)
    self.assertTrue(
        any(b.payload.routed_experts is not None for b in dense),
        "the fixture must exercise router replay",
    )
    ok, why = batches_identical(dense, held)
    self.assertTrue(ok, why)

  def test_feed_error_leaves_the_same_buffer_as_the_dense_path(self):
    # `feed` expands a bit-packed group on threads; an item that fails must
    # still leave exactly the items before it buffered, as the plain loop does.
    def failing_feed(compact_first):
      groups = _groups(np.random.default_rng(34), num_groups=1)
      group = copy.deepcopy(groups[0])
      if compact_first:
        for item in group:
          held_routing.compact_item_routing(item)
      payloads = _adapter(len(group)).create_trainer_payloads(
          group, rewards=[0.0, 1.0, 0.0, 1.0]
      )
      bad = payloads[2].routed_experts
      payloads[2] = dataclasses.replace(
          payloads[2],
          routed_experts=(
              held_routing.BitPackedRoutedExperts(lo=bad.lo, hi=bad.hi[:, :1])
              if compact_first
              else bad[:, 0]
          ),
      )
      assembler = _packed_assembler(len(group))
      with self.assertRaises(ValueError):
        assembler.feed(payloads)
      return len(assembler._buffer)  # pylint: disable=protected-access

    self.assertEqual(failing_feed(False), 2)
    self.assertEqual(failing_feed(True), 2)

  def test_a_trajectory_without_routing_keeps_its_place(self):
    groups = _groups(np.random.default_rng(31))
    groups[1][2].traj.pop("routed_experts")
    dense = _assemble(groups, _packed_assembler, compact_first=False)
    held = _assemble(groups, _packed_assembler, compact_first=True)
    ok, why = batches_identical(dense, held)
    self.assertTrue(ok, why)

  def test_identity_gate_fails_on_a_decoder_that_drops_the_ninth_bit(self):
    groups = _groups(np.random.default_rng(32))
    dense = _assemble(groups, _packed_assembler, compact_first=False)
    original = held_routing.expand

    def broken_expand(packed):  # Deliberately broken: ids >= 256 lose bit 8.
      out = original(packed)
      return np.where(out >= 0, out & 0xFF, out).astype(np.int16)

    with mock.patch.object(held_routing, "expand", broken_expand):
      held = _assemble(groups, _packed_assembler, compact_first=True)
    ok, why = batches_identical(dense, held)
    self.assertFalse(ok)
    self.assertIn("routed_experts", why)

  def test_identity_gate_fails_on_an_encoder_without_the_exactness_check(self):
    # Deliberately broken: always encodes, so a partly unset token-layer or an
    # out-of-range id comes back as a fully unset token-layer (lossy).
    groups = _groups(np.random.default_rng(33), adversarial=True)
    dense = _assemble(groups, _packed_assembler, compact_first=False)
    original = held_routing._encode_rows  # pylint: disable=protected-access

    def lossy_encode_rows(u, lo, hi):
      original(u, lo, hi)
      return 0

    with mock.patch.object(held_routing, "_encode_rows", lossy_encode_rows):
      held = _assemble(groups, _packed_assembler, compact_first=True)
    ok, why = batches_identical(dense, held)
    self.assertFalse(ok)
    self.assertIn("routed_experts", why)


class ProgramTest(absltest.TestCase):
  """The flag through StandardRLProgram's real polling, critique and train."""

  def _engine(self, items):
    engine = mock.MagicMock(spec=distributed_rl_engine.DistributedRLEngine)
    pending = [list(items)]

    async def _poll(*args, **kwargs):
      del args, kwargs
      await asyncio.sleep(0)
      return pending.pop(0) if pending else []

    async def _logps(role, items=None, **kwargs):
      del role, kwargs
      shape = np.asarray(items.completion_tokens).shape
      return types.SimpleNamespace(
          per_token_logps=np.zeros(shape, dtype=np.float32), error=None
      )

    engine.poll_rollouts = mock.AsyncMock(side_effect=_poll)
    # The sampler-trainer agreement request carries the packed routing too.
    engine.per_token_logps = mock.AsyncMock(side_effect=_logps)
    engine.train_step = mock.AsyncMock(return_value={"metrics": {}})
    engine.save_checkpoint = mock.AsyncMock(
        return_value=types.SimpleNamespace(metadata={"checkpoint_path": "x"})
    )
    engine.sync_weights = mock.AsyncMock(return_value=1)
    engine.get_metrics = mock.AsyncMock(return_value=None)
    return engine

  def _run(self, groups, *, compact):
    groups = copy.deepcopy(groups)
    group_size = len(groups[0])
    items = [item for group in groups for item in group]
    program = rl_program.StandardRLProgram(
        algo=_adapter(group_size),
        dataset=[f"p{g}" for g in range(len(groups))],
        batch_size=len(groups),
        max_steps=1,
        assembler=_packed_assembler(group_size),
        sync_weights=False,
        compact_held_routed_experts=compact,
    )
    program.algo.algo_config.temperature = 1.0
    program.engine = self._engine(items)
    held = {}

    async def _go():
      program._in_flight_rollouts = len(items)  # pylint: disable=protected-access
      program._dispatch_done.set()  # pylint: disable=protected-access
      await program.polling_stage()
      held["after_poll"] = _held_bytes(items)
      held["routing_types"] = {
          type(item.traj.get("routed_experts")).__name__ for item in items
      }
      await program.critique_stage()
      await program.train_stage()

    asyncio.run(_go())
    batches = [
        call.args[0] for call in program.engine.train_step.call_args_list
    ]
    batches += [
        call.kwargs["items"]
        for call in program.engine.per_token_logps.call_args_list
    ]
    stats = program._held_routing_stats  # pylint: disable=protected-access
    program.close()
    return batches, held, stats

  def test_polled_routing_is_held_bit_packed_and_trains_identically(self):
    groups = _groups(np.random.default_rng(40), num_groups=2, adversarial=True)
    dense_batches, dense_held, _ = self._run(groups, compact=False)
    packed_batches, packed_held, stats = self._run(groups, compact=True)

    self.assertEqual(dense_held["routing_types"], {"ndarray"})
    self.assertEqual(
        packed_held["routing_types"], {"ndarray", "BitPackedRoutedExperts"}
    )
    # Two adversarial trajectories per group cannot be encoded exactly.
    self.assertEqual(stats.kept_dense, 4)
    self.assertEqual(stats.compacted, 4)
    self.assertLess(packed_held["after_poll"], dense_held["after_poll"])

    self.assertNotEmpty(dense_batches)
    self.assertLen(packed_batches, len(dense_batches))
    for a, b in zip(dense_batches, packed_batches):
      self.assertIs(type(a), type(b))
      for field in dataclasses.fields(a):
        va, vb = getattr(a, field.name), getattr(b, field.name)
        self.assertEqual(va is None, vb is None, field.name)
        if isinstance(va, np.ndarray):
          self.assertIsInstance(vb, np.ndarray, field.name)
          self.assertEqual(va.dtype, vb.dtype, field.name)
          self.assertEqual(va.shape, vb.shape, field.name)
          self.assertEqual(va.tobytes(), vb.tobytes(), field.name)
    self.assertTrue(any(b.routed_experts is not None for b in packed_batches))

  def test_flag_off_creates_no_threads(self):
    program = rl_program.StandardRLProgram(
        algo=_adapter(4), dataset=["p0"], batch_size=1
    )
    self.assertIsNone(program._held_routing_executor)  # pylint: disable=protected-access
    program.close()


if __name__ == "__main__":
  absltest.main()
