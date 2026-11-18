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

Planning is tested in-process. The data plane is tested for real: several OS
processes join one JAX world over gloo CPU collectives and move weights with
the same `ppermute` program that runs over ICI on TPU.
"""

import json
import os
import socket
import subprocess
import sys

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.weight_sync import ici_transport

_WORLD_PROCESS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ici_world_process.py"
)
_TIMEOUT_S = 120


def _leaf(name, shape, dtype="float32"):
  return ici_transport.LeafSpec(name, tuple(shape), dtype)


class PlanTest(parameterized.TestCase):

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
    served = []
    for pairs in rounds:
      senders = [s for s, _ in pairs]
      receivers = [d for _, d in pairs]
      # A valid partial permutation whose senders already hold the data.
      self.assertLen(set(senders), len(senders))
      self.assertLen(set(receivers), len(receivers))
      self.assertTrue(set(senders) <= holders)
      self.assertFalse(set(receivers) & holders)
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

  def test_build_plan_rejects_overlapping_roles(self):
    with self.assertRaisesRegex(ValueError, "both source and destination"):
      ici_transport.build_plan(
          [_leaf("w", (4,))],
          source_processes=[0],
          destination_processes=[0, 1],
          width=2,
      )

  def test_canonical_spec_splits_first_divisible_axis(self):
    self.assertEqual(ici_transport.canonical_spec((3, 8, 4), 4), (None, "ici", None))
    self.assertEqual(ici_transport.canonical_spec((3, 5), 4), (None, None))


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _run_world(*, processes, width, sources, destinations, **scenario):
  """Runs one scenario; returns `{rank: result}` for the participants."""
  port = _free_port()
  env = dict(os.environ)
  # `jax.distributed.initialize` can hang behind proxy settings.
  for key in list(env):
    if "proxy" in key.lower():
      env.pop(key)
  participants = sorted({*sources, *destinations})
  procs = {}
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
    )
    procs[rank] = subprocess.Popen(
        [sys.executable, _WORLD_PROCESS, json.dumps(cfg)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
  results = {}
  try:
    for rank, proc in procs.items():
      out, _ = proc.communicate(timeout=_TIMEOUT_S)
      lines = [l for l in out.splitlines() if l.startswith("RESULT ")]
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
  return {rank: results[rank] for rank in participants}, results


def _gloo_available() -> bool:
  try:
    import jax  # pylint: disable=g-import-not-at-top

    return hasattr(jax.config, "jax_cpu_collectives_implementation")
  except ImportError:
    return False


@absltest.skipUnless(_gloo_available(), "needs JAX CPU collectives (gloo)")
class MultiProcessTransferTest(absltest.TestCase):

  def _assert_delivered(self, participants, sources, destinations):
    for rank in sources:
      self.assertEqual(participants[rank]["role"], "source")
      self.assertTrue(participants[rank]["source_intact"])
    for rank in destinations:
      self.assertEqual(participants[rank]["role"], "destination")
      self.assertEqual(participants[rank]["mismatched"], [])
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
    # Every sync released (dropped) every leaf before receiving it.
    self.assertEqual(participants[1]["released"], 3 * 5)

  def test_bystander_process_joins_with_a_throwaway_payload(self):
    # Process 3 is in the JAX world but neither sends nor receives. It still
    # enters every collective; were it to sit them out, the others would hang.
    participants, everyone = _run_world(
        processes=4, width=2, sources=[0], destinations=[1, 2]
    )
    self._assert_delivered(participants, [0], [1, 2])
    self.assertEqual(everyone[3]["role"], "bystander")

  def test_source_role_spanning_two_processes(self):
    # The trainer's weights are sharded ACROSS its two processes, so staging
    # is itself a collective; both then act as initial holders.
    participants, _ = _run_world(
        processes=4, width=2, sources=[0, 1], destinations=[2, 3]
    )
    self._assert_delivered(participants, [0, 1], [2, 3])
    self.assertEqual(participants[0]["rounds"], 1)

  def test_single_chunk_plan(self):
    participants, _ = _run_world(
        processes=2, width=4, sources=[0], destinations=[1], chunk_bytes=1 << 20
    )
    self._assert_delivered(participants, [0], [1])
    self.assertEqual(participants[0]["chunks"], 1)


if __name__ == "__main__":
  absltest.main()
