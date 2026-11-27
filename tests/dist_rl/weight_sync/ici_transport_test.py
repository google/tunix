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

"""Tests for the ICI weight transport.

Planning and packing invariants are tested in-process. The data plane is tested
for real across OS processes joining one JAX world over Gloo CPU collectives.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.testing import ici_testing
from tunix.dist_rl.weight_sync import ici_transport

_TIMEOUT_S = 120


def _leaf(
    name: str, shape: tuple[int, ...], dtype: str = "float32"
) -> ici_transport.LeafSpec:
  return ici_transport.LeafSpec(name, tuple(shape), dtype)


class PlanTest(parameterized.TestCase):

  def test_leaf_spec_validates_name_shape_and_dtype(self):
    with self.assertRaisesRegex(ValueError, "must not be empty"):
      ici_transport.LeafSpec("", (4,), "float32")
    with self.assertRaisesRegex(ValueError, "invalid shape"):
      ici_transport.LeafSpec("w", (-1, 4), "float32")
    with self.assertRaisesRegex(ValueError, "invalid dtype"):
      ici_transport.LeafSpec("w", (4,), "not_a_dtype")

  def test_chunks_are_order_preserving_and_bounded(self):
    leaves = [_leaf(f"l{i}", (4,)) for i in range(5)]  # 16 bytes each
    self.assertEqual(
        ici_transport.plan_chunks(leaves, chunk_bytes=32),
        ((0, 1), (2, 3), (4,)),
    )

  def test_oversized_leaf_gets_its_own_chunk(self):
    leaves = [_leaf("a", (4,)), _leaf("big", (100,)), _leaf("b", (4,))]
    self.assertEqual(
        ici_transport.plan_chunks(leaves, chunk_bytes=32), ((0,), (1,), (2,))
    )

  def test_bfloat16_is_two_bytes(self):
    self.assertEqual(_leaf("w", (3, 4), "bfloat16").nbytes, 24)

  @parameterized.parameters(
      # sources, destinations, expected rounds
      (1, 1, 1),
      (1, 3, 2),
      (1, 7, 3),
      (1, 8, 4),
      (2, 6, 2),
      (4, 2, 1),
  )
  def test_relay_rounds_double_the_holders(self, sources, destinations, want):
    src = list(range(sources))
    dst = list(range(sources, sources + destinations))
    rounds = ici_transport.plan_rounds(src, dst)
    self.assertLen(rounds, want)
    holders = set(src)
    served: list[int] = []
    for pairs in rounds:
      senders = [s for s, _ in pairs]
      receivers = [d for _, d in pairs]
      self.assertLen(set(senders), len(senders))
      self.assertLen(set(receivers), len(receivers))
      self.assertLessEqual(set(senders), holders)
      self.assertEmpty(set(receivers) & holders)
      holders |= set(receivers)
      served += receivers
    self.assertEqual(served, dst)

  def test_participants_are_the_whole_world(self):
    plan = ici_transport.build_plan(
        [_leaf("w", (4,))],
        source_processes=[0],
        destination_processes=[2],
        width=2,
        world_processes=[0, 1, 2, 3],
    )
    self.assertEqual(plan.participants, (0, 1, 2, 3))
    self.assertEqual(plan.destination_processes, (2,))
    with self.assertRaisesRegex(ValueError, "not part of the JAX world"):
      ici_transport.build_plan(
          [_leaf("w", (4,))],
          source_processes=[0],
          destination_processes=[5],
          width=2,
          world_processes=[0, 1],
      )

  def test_build_plan_rejects_duplicate_world_processes(self):
    with self.assertRaisesRegex(ValueError, "duplicate process in world"):
      ici_transport.build_plan(
          [_leaf("w", (4,))],
          source_processes=[0],
          destination_processes=[1],
          width=2,
          world_processes=[0, 1, 1],
      )

  def test_build_plan_rejects_overlapping_roles(self):
    with self.assertRaisesRegex(ValueError, "both source and destination"):
      ici_transport.build_plan(
          [_leaf("w", (4,))],
          source_processes=[0],
          destination_processes=[0, 1],
          width=2,
      )

  def test_canonical_spec_splits_first_divisible_axis(self):
    self.assertEqual(
        ici_transport.canonical_spec((3, 8, 4), 4), (None, "ici", None)
    )
    self.assertEqual(ici_transport.canonical_spec((3, 5), 4), (None, None))

  @parameterized.parameters(
      ((8, 6), 2),
      ((3, 8, 4), 4),
      ((3, 5), 4),
      ((4,), 4),
  )
  def test_pack_and_unpack_roundtrip(self, shape, width):
    arr = jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(shape)
    packed = ici_transport._pack_canonical_leaf(arr, shape, width)
    self.assertEqual(packed.shape[0], width)
    unpacked = ici_transport._unpack_payload_leaf(packed, shape, width)
    np.testing.assert_array_equal(np.asarray(unpacked), np.asarray(arr))

  def test_check_plan_rejects_wrong_source_and_destination_sharding(self):
    dev_p0 = absltest.mock.MagicMock(id=0, process_index=0)
    dev_p0_b = absltest.mock.MagicMock(id=1, process_index=0)
    dev_p1 = absltest.mock.MagicMock(id=2, process_index=1)
    dev_p1_b = absltest.mock.MagicMock(id=3, process_index=1)

    class _FakeArray:

      def __init__(self, device_set, *, is_fully_addressable=True):
        self.shape = (4,)
        self.dtype = np.dtype("float32")
        self.is_fully_addressable = is_fully_addressable
        self.sharding = absltest.mock.MagicMock(device_set=device_set)

    bad_plan = ici_transport.build_plan(
        [_leaf("w", (4,))],
        source_processes=[0],
        destination_processes=[1],
        width=1,
        world_processes=[0, 1],
    )
    with absltest.mock.patch.object(
        jax, "process_index", return_value=0
    ), absltest.mock.patch.object(
        jax, "devices", return_value=[dev_p0, dev_p1]
    ):
      with self.assertRaisesRegex(ValueError, "outside plan source_processes"):
        ici_transport.check_plan(bad_plan, source={"w": _FakeArray({dev_p1})})

    class _BadSink:

      def target_sharding(self, name: str) -> Any:
        del name
        return absltest.mock.MagicMock(device_set={dev_p0})

      def release(self, names):
        del names

      def install(self, arrays):
        del arrays

    with absltest.mock.patch.object(
        jax, "process_index", return_value=1
    ), absltest.mock.patch.object(
        jax, "devices", return_value=[dev_p0, dev_p1]
    ):
      with self.assertRaisesRegex(ValueError, "must stay local to process 1"):
        ici_transport.check_plan(bad_plan, sink=_BadSink())

    # Multi-host source leaf spanning both processes but missing a device fails.
    mh_plan = ici_transport.build_plan(
        [_leaf("w", (4,))],
        source_processes=[0, 1],
        destination_processes=[2],
        width=2,
        world_processes=[0, 1, 2],
    )
    dev_p2 = absltest.mock.MagicMock(id=4, process_index=2)
    dev_p2_b = absltest.mock.MagicMock(id=5, process_index=2)
    all_devs = [dev_p0, dev_p0_b, dev_p1, dev_p1_b, dev_p2, dev_p2_b]
    with absltest.mock.patch.object(
        jax, "process_index", return_value=0
    ), absltest.mock.patch.object(jax, "devices", return_value=all_devs):
      partial_arr = _FakeArray(
          {dev_p0, dev_p1}, is_fully_addressable=False
      )
      with self.assertRaisesRegex(ValueError, "must span all devices"):
        ici_transport.check_plan(mh_plan, source={"w": partial_arr})


def _run_world(
    *,
    processes: int,
    width: int,
    sources: list[int],
    destinations: list[int],
    **scenario: Any,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
  """Runs one multi-process scenario; returns `(participants, everyone)`."""
  port = ici_testing.free_port()
  env = ici_testing.clean_env()
  participants = sorted({*sources, *destinations})
  cmd_prefix = ici_testing.worker_command("ici_world_process", __file__)
  procs: dict[int, subprocess.Popen[str]] = {}
  for rank in range(processes):
    cfg = dict(
        rank=rank,
        port=port,
        processes=processes,
        width=width,
        sources=sources,
        destinations=destinations,
        chunk_bytes=scenario.get("chunk_bytes", 64),
        syncs=scenario.get("syncs", 1),
        reverse_device_order=scenario.get("reverse_device_order", False),
    )
    procs[rank] = subprocess.Popen(
        [*cmd_prefix, json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
  results: dict[int, dict[str, Any]] = {}
  try:
    for rank, proc in procs.items():
      out, _ = proc.communicate(timeout=_TIMEOUT_S)
      lines = [line for line in out.splitlines() if line.startswith("RESULT ")]
      if proc.returncode != 0 or not lines:
        raise AssertionError(
            f"world process {rank} failed (exit {proc.returncode}):\n"
            + out[-4000:]
        )
      results[rank] = json.loads(lines[-1][len("RESULT ") :])
  finally:
    for proc in procs.values():
      if proc.poll() is None:
        proc.kill()
      proc.wait()
  return {rank: results[rank] for rank in participants}, results


@absltest.skipUnless(
    ici_testing.gloo_available(), "needs JAX CPU collectives (gloo)"
)
class MultiProcessTransferTest(absltest.TestCase):

  def _assert_delivered(self, participants, sources, destinations):
    for rank in sources:
      self.assertEqual(participants[rank]["role"], "source")
      self.assertTrue(participants[rank]["source_intact"])
    for rank in destinations:
      self.assertEqual(participants[rank]["role"], "destination")
      self.assertEmpty(participants[rank]["mismatched"])
      self.assertGreater(participants[rank]["bytes"], 0)

  def test_one_source_fans_out_to_three_replicas(self):
    participants, _ = _run_world(
        processes=4, width=2, sources=[0], destinations=[1, 2, 3]
    )
    self._assert_delivered(participants, [0], [1, 2, 3])
    # 3 replicas from one source relay in 2 rounds, not 3.
    self.assertEqual(participants[0]["rounds"], 2)
    self.assertGreater(participants[0]["chunks"], 1)

  def test_repeated_syncs_deliver_each_new_version(self):
    participants, _ = _run_world(
        processes=3, width=2, sources=[0], destinations=[1, 2], syncs=3
    )
    self._assert_delivered(participants, [0], [1, 2])
    self.assertEqual(participants[1]["released"], 3 * 5)

  def test_bystander_process_joins_with_a_throwaway_payload(self):
    participants, everyone = _run_world(
        processes=4, width=2, sources=[0], destinations=[1, 2]
    )
    self._assert_delivered(participants, [0], [1, 2])
    self.assertEqual(everyone[3]["role"], "bystander")

  def test_source_role_spanning_two_processes(self):
    participants, _ = _run_world(
        processes=4, width=2, sources=[0, 1], destinations=[2, 3]
    )
    self._assert_delivered(participants, [0, 1], [2, 3])
    self.assertEqual(participants[0]["rounds"], 1)

  def test_permuted_device_order_meshes_stage_and_install_cleanly(self):
    # Simulates 2D snake-ordered TPU meshes whose device ordering differs from
    # ascending device-ID order on both source and destination.
    participants, _ = _run_world(
        processes=4,
        width=2,
        sources=[0, 1],
        destinations=[2, 3],
        reverse_device_order=True,
    )
    self._assert_delivered(participants, [0, 1], [2, 3])

  def test_single_chunk_plan(self):
    participants, _ = _run_world(
        processes=2, width=4, sources=[0], destinations=[1], chunk_bytes=1 << 20
    )
    self._assert_delivered(participants, [0], [1])
    self.assertEqual(participants[0]["chunks"], 1)


if __name__ == "__main__":
  absltest.main()
