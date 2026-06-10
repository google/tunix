# Copyright 2025 Google LLC
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

"""Real 2-process ``jax.distributed`` simulation of multi-controller data flow.

This is the load-bearing proof that Plan C's per-process data sharding -- one
generic ``cli.utils.data.shard_by_process`` strided slice -- is correct under
genuine multi-controller JAX for BOTH the SFT and the GRPO/RL paths. It does NOT
mock ``jax.process_count``: it spawns two OS processes that each call
``jax.distributed.initialize()`` against a shared coordinator (the exact same
mechanism a 2-host TPU job uses), builds the per-process dataset, and then runs
the real ``sharding_utils.shard_input`` to assemble the global batch from the
two process-local sub-batches.

Two workers exercise the two batching sites:

  ``--worker-sft``: the SFT translation ``grain.MapDataset`` loader
  (``translation_dataset._build_data_loader``).
  ``--worker-grpo``: the RL ``post_init_dataset`` ``grain.MapDataset`` pipeline
  (``cli.utils.data.post_init_dataset``) that feeds the GRPO learner.

For each path it asserts:

  * Each process initializes with ``process_count == 2`` and a disjoint
    ``process_index``.
  * Each process's pipeline yields a LOCAL batch of ``global_batch // 2`` rows
    drawn from a DISJOINT strided shard (proc 0 -> even records, proc 1 -> odd),
    and the two shards together cover every record.
  * ``shard_input`` -> ``jax.make_array_from_process_local_data`` assembles a
    global array of shape ``(global_batch, seq_len)`` that is NOT fully
    addressable (it spans both processes), and each process's addressable slice
    equals the local batch it fed in. This is the precise
    ``make_array_from_process_local_data`` contract the trainers rely on.
  * A token-weighted ``sum(value)/sum(weight)`` reduction computed inside a
    ``jax.jit`` on the GLOBAL mesh over the process-local shards produces a
    REPLICATED scalar that equals the reduction over the FULL global batch and
    is NOT equal to either process's local-shard-only value -- the proof that
    the assembled global array is the true global batch, not a per-host shard.

Run directly (it self-spawns the workers):

    python -m tests.examples.data.translation_dataset_multiprocess_test
"""

import os
import socket
import sys

from absl.testing import absltest

_GLOBAL_BATCH = 4
_NUM_RECORDS = 8
_SEQ_LEN = 8
_NUM_PROCESSES = 2
# 2 devices per process -> 4 global devices, enough for a (fsdp=4, tp=1) mesh
# that shards the global batch of 4 across the fsdp axis.
_DEVICES_PER_PROCESS = 2


def _pick_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("localhost", 0))
    return s.getsockname()[1]


def _init_worker(process_id: int, coordinator_address: str):
  """Sets up CPU devices + ``jax.distributed`` and returns the global mesh."""
  # Must be set before jax initializes its backend.
  os.environ["XLA_FLAGS"] = (
      f"--xla_force_host_platform_device_count={_DEVICES_PER_PROCESS}"
  )
  os.environ["JAX_PLATFORMS"] = "cpu"

  import jax
  import jax.sharding as shd
  import numpy as np

  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      num_processes=_NUM_PROCESSES,
      process_id=process_id,
  )
  assert jax.process_count() == _NUM_PROCESSES, jax.process_count()
  assert jax.process_index() == process_id, jax.process_index()
  assert jax.device_count() == _NUM_PROCESSES * _DEVICES_PER_PROCESS

  mesh = shd.Mesh(
      np.array(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )
  return mesh


def _assemble_and_check(local_tokens, mesh, record_ids_of_local_batch):
  """Runs ``shard_input`` on a local batch and proves the global assembly.

  Returns a JSON-serializable dict of results for the parent test, including the
  replicated global token-weighted metric and its references.
  """
  import jax
  import jax.numpy as jnp
  import jax.sharding as shd
  import numpy as np

  from tunix.sft import sharding_utils

  with mesh:
    global_arr = sharding_utils.shard_input(local_tokens, ("fsdp",))

  assert global_arr.shape == (_GLOBAL_BATCH, _SEQ_LEN), global_arr.shape
  # The assembled global array spans both processes -> not fully addressable.
  assert not global_arr.is_fully_addressable

  # This process's addressable slice must equal the local batch it fed in.
  addressable = np.concatenate(
      [np.asarray(shard.data) for shard in global_arr.addressable_shards],
      axis=0,
  )
  addressable = addressable[np.argsort(addressable[:, 0])]
  expected = np.asarray(local_tokens)[
      np.argsort(np.asarray(local_tokens)[:, 0])
  ]
  assert np.array_equal(addressable, expected), (addressable, expected)

  # GLOBAL-metric proof: a sum/sum reduction over the sharded global array,
  # computed inside a jit on the global mesh, is a REPLICATED scalar equal to
  # the reduction over the full global batch and unequal to either local shard.
  def _value(tokens):
    rec = tokens[:, 0].astype(jnp.float32)  # == record_id + 1
    return jnp.sum(rec * rec), jnp.sum(rec)

  @jax.jit
  def _global_mean(global_tokens):
    total_value, total_weight = _value(global_tokens)
    return total_value / total_weight

  with mesh:
    global_mean_arr = _global_mean(global_arr)
  metric_replicated = bool(global_mean_arr.sharding.spec == shd.PartitionSpec())
  assert metric_replicated, global_mean_arr.sharding.spec
  global_mean = float(jax.device_get(global_mean_arr))

  def _np_mean(record_ids):
    rec = np.array([rid + 1 for rid in record_ids], dtype=np.float64)
    return float(np.sum(rec * rec) / np.sum(rec))

  # The full global batch is the union of both processes' strided first-batch
  # shards: proc p contributes records p, p+2, ... so the first local batch of
  # size 2 is {0, 2} for p0 and {1, 3} for p1 -> global {0, 1, 2, 3}.
  local_bs = _GLOBAL_BATCH // _NUM_PROCESSES
  full_global_ids = sorted(
      p + _NUM_PROCESSES * j
      for p in range(_NUM_PROCESSES)
      for j in range(local_bs)
  )
  global_ref = _np_mean(full_global_ids)
  local_only_ref = _np_mean(sorted(record_ids_of_local_batch))

  assert abs(global_mean - global_ref) < 1e-4, (global_mean, global_ref)
  assert abs(global_mean - local_only_ref) > 1e-4, (global_mean, local_only_ref)

  return {
      "global_shape": list(global_arr.shape),
      "fully_addressable": bool(global_arr.is_fully_addressable),
      "global_metric_replicated": metric_replicated,
      "global_metric": global_mean,
      "global_metric_ref": global_ref,
      "local_only_metric_ref": local_only_ref,
  }


def _sft_worker(process_id: int, coordinator_address: str) -> None:
  """Exercises the SFT translation MapDataset loader end to end."""
  mesh = _init_worker(process_id, coordinator_address)

  import numpy as np

  from tunix.examples.data import translation_dataset

  class _FakeTokenizer:

    def pad_id(self):
      return 0

    def tokenize(self, text, prefix="", suffix="", add_eos=False):
      record_id = int(text.lstrip("r")) + 1  # +1 so token != pad id (0)
      return np.full((2,), record_id, dtype=np.int32)

  class _ListDataSource:

    def __init__(self, n):
      self._records = [
          {"src": f"r{i}".encode(), "dst": f"r{i}".encode()} for i in range(n)
      ]

    def __len__(self):
      return len(self._records)

    def __getitem__(self, idx):
      return self._records[idx]

  loader = translation_dataset._build_data_loader(
      data_source=_ListDataSource(_NUM_RECORDS),
      batch_size=_GLOBAL_BATCH,
      num_epochs=1,
      max_seq_len=_SEQ_LEN,
      tokenizer=_FakeTokenizer(),
      input_template=translation_dataset.INPUT_TEMPLATE,
  )

  local_bs = _GLOBAL_BATCH // _NUM_PROCESSES
  all_batches = list(loader)
  assert all_batches, "loader produced no batches"
  local_record_ids = []
  for batch in all_batches:
    tokens = np.asarray(batch.input_tokens)
    assert tokens.shape == (local_bs, _SEQ_LEN), tokens.shape
    local_record_ids.extend(int(row[0]) - 1 for row in tokens)
  local_record_ids = sorted(local_record_ids)

  local_tokens = np.asarray(all_batches[0].input_tokens)
  first_batch_ids = [int(row[0]) - 1 for row in local_tokens]
  metrics = _assemble_and_check(local_tokens, mesh, first_batch_ids)

  _emit_result(process_id, local_tokens, local_record_ids, metrics)


def _grpo_worker(process_id: int, coordinator_address: str) -> None:
  """Exercises the RL post_init_dataset MapDataset pipeline end to end."""
  mesh = _init_worker(process_id, coordinator_address)

  import grain
  import numpy as np

  from tunix.cli.utils import data as data_lib

  class _FakeTokenizer:

    def encode(self, text):
      return text.split()

  # A GRPO-style MapDataset: each record carries a "prompts" string and an
  # integer record id. post_init_dataset will shard it across processes.
  dataset = grain.MapDataset.source(
      [{"prompts": f"r{i}", "rid": i} for i in range(_NUM_RECORDS)]
  )
  first_segment, second_segment = data_lib.post_init_dataset(
      dataset,
      tokenizer=_FakeTokenizer(),
      batch_size=_GLOBAL_BATCH,
      num_batches=None,
      max_prompt_length=None,
  )
  assert second_segment is None

  local_bs = _GLOBAL_BATCH // _NUM_PROCESSES
  all_batches = list(first_segment)
  assert all_batches, "post_init_dataset produced no batches"
  local_record_ids = []
  for batch in all_batches:
    rids = np.asarray(batch["rid"])
    assert rids.shape == (local_bs,), rids.shape
    local_record_ids.extend(int(r) for r in rids)
  local_record_ids = sorted(local_record_ids)

  # Build a token tensor analogous to the rollout's prompt_ids: each row encodes
  # its record id (+1 so token != pad id 0), shaped (local_bs, seq_len). This is
  # exactly the fully-addressable per-process array grpo_learner feeds to
  # rl_cluster.shard_input.
  first_rids = [int(r) for r in np.asarray(all_batches[0]["rid"])]
  local_tokens = np.stack(
      [np.full((_SEQ_LEN,), rid + 1, dtype=np.int32) for rid in first_rids]
  )
  metrics = _assemble_and_check(local_tokens, mesh, first_rids)

  _emit_result(process_id, local_tokens, local_record_ids, metrics)


_ODD_NUM_RECORDS = 7
_ODD_GLOBAL_BATCH = 2  # local batch size 1 -> one record per batch per host


def _odd_count_worker(process_id: int, coordinator_address: str) -> None:
  """Regression for B1: odd record count must yield equal per-host batch counts.

  With ``_ODD_NUM_RECORDS`` not a multiple of ``process_count``,
  ``shard_by_process`` must truncate to an even length so both hosts produce the
  SAME number of batches; otherwise the cross-host
  ``make_array_from_process_local_data`` collective (run every training step)
  hangs because the host with fewer batches exhausts first. Each step here runs
  the real ``shard_input`` -> that collective, so an unequal-shard regression
  surfaces as a hang (300s timeout) rather than a silent wrong number. The
  global batch (``_ODD_GLOBAL_BATCH``) is sized to the number of processes so the
  per-step global array shards one row per host across the fsdp sub-axis. The
  worker reports its batch count and consumed record ids for the parent test.
  """
  mesh = _init_worker(process_id, coordinator_address)

  import json

  import grain
  import jax
  import jax.sharding as shd
  import numpy as np

  from tunix.cli.utils import data as data_lib
  from tunix.sft import sharding_utils

  class _FakeTokenizer:

    def encode(self, text):
      return text.split()

  dataset = grain.MapDataset.source(
      [{"prompts": f"r{i}", "rid": i} for i in range(_ODD_NUM_RECORDS)]
  )
  first_segment, _ = data_lib.post_init_dataset(
      dataset,
      tokenizer=_FakeTokenizer(),
      batch_size=_ODD_GLOBAL_BATCH,
      num_batches=None,
      max_prompt_length=None,
  )

  # Use an fsdp axis sized to the process count so the global batch of
  # _ODD_GLOBAL_BATCH (== _NUM_PROCESSES) shards exactly one row per host. The
  # default _init_worker mesh has fsdp == device_count (4), which would not
  # divide a global batch of 2; build a (process_count, _) mesh instead.
  odd_mesh = shd.Mesh(
      np.array(jax.devices()).reshape(_NUM_PROCESSES, -1),
      axis_names=("fsdp", "tp"),
  )
  del mesh

  local_bs = _ODD_GLOBAL_BATCH // _NUM_PROCESSES
  consumed_ids = []
  num_batches = 0
  for batch in first_segment:
    rids = np.asarray(batch["rid"])
    assert rids.shape == (local_bs,), rids.shape
    consumed_ids.extend(int(r) for r in rids)
    # Assemble the global batch for this step, exercising the cross-host
    # collective. With unequal shards this is exactly where a host would block.
    tokens = np.stack(
        [np.full((_SEQ_LEN,), rid + 1, dtype=np.int32) for rid in rids]
    )
    with odd_mesh:
      global_arr = sharding_utils.shard_input(tokens, ("fsdp",))
    assert global_arr.shape == (_ODD_GLOBAL_BATCH, _SEQ_LEN), global_arr.shape
    num_batches += 1

  print(
      "RESULT "
      + json.dumps({
          "process_id": process_id,
          "process_count": jax.process_count(),
          "num_batches": num_batches,
          "consumed_ids": sorted(consumed_ids),
      }),
      flush=True,
  )
  jax.distributed.shutdown()


def _emit_result(process_id, local_tokens, local_record_ids, metrics) -> None:
  import json

  import jax
  import numpy as np

  result = {
      "process_id": process_id,
      "process_count": jax.process_count(),
      "local_batch_shape": list(np.asarray(local_tokens).shape),
      "local_record_ids": local_record_ids,
      **metrics,
  }
  print("RESULT " + json.dumps(result), flush=True)
  jax.distributed.shutdown()


class _MultiProcessShardingTest(absltest.TestCase):
  """Shared 2-process assertions parameterized by which worker flag to spawn."""

  WORKER_FLAG = None

  def _run(self):
    import json
    import subprocess

    coordinator_address = f"localhost:{_pick_free_port()}"
    procs = []
    for process_id in range(_NUM_PROCESSES):
      procs.append(
          subprocess.Popen(
              [
                  sys.executable,
                  __file__,
                  self.WORKER_FLAG,
                  str(process_id),
                  coordinator_address,
              ],
              stdout=subprocess.PIPE,
              stderr=subprocess.STDOUT,
              text=True,
          )
      )

    results = {}
    outputs = {}
    for process_id, proc in enumerate(procs):
      out, _ = proc.communicate(timeout=300)
      outputs[process_id] = out
      self.assertEqual(
          proc.returncode, 0, msg=f"worker {process_id} failed:\n{out}"
      )
      for line in out.splitlines():
        if line.startswith("RESULT "):
          results[process_id] = json.loads(line[len("RESULT ") :])

    self.assertLen(results, _NUM_PROCESSES, msg=outputs)

    local_bs = _GLOBAL_BATCH // _NUM_PROCESSES
    for process_id, res in results.items():
      self.assertEqual(res["process_count"], _NUM_PROCESSES)
      self.assertEqual(res["process_id"], process_id)
      # Each process fed only its LOCAL sub-batch...
      self.assertEqual(res["local_batch_shape"], [local_bs, _SEQ_LEN])
      # ...but shard_input assembled the full GLOBAL batch.
      self.assertEqual(res["global_shape"], [_GLOBAL_BATCH, _SEQ_LEN])
      self.assertFalse(res["fully_addressable"])

      # GLOBAL-metric proof: replicated, equals full-global, != local-only.
      self.assertTrue(res["global_metric_replicated"])
      self.assertAlmostEqual(
          res["global_metric"], res["global_metric_ref"], places=4
      )
      self.assertNotAlmostEqual(
          res["global_metric"], res["local_only_metric_ref"], places=4
      )

    # Both processes observe the SAME replicated global metric...
    self.assertAlmostEqual(
        results[0]["global_metric"], results[1]["global_metric"], places=6
    )
    # ...while their local-only values genuinely differ (non-degenerate proof).
    self.assertNotAlmostEqual(
        results[0]["local_only_metric_ref"],
        results[1]["local_only_metric_ref"],
        places=4,
    )

    # The two processes read disjoint strided shards covering all records:
    # proc 0 -> even record ids, proc 1 -> odd record ids.
    ids0 = set(results[0]["local_record_ids"])
    ids1 = set(results[1]["local_record_ids"])
    self.assertEqual(ids0, set(range(0, _NUM_RECORDS, 2)))
    self.assertEqual(ids1, set(range(1, _NUM_RECORDS, 2)))
    self.assertEqual(ids0 & ids1, set(), msg="record shards must be disjoint")
    self.assertEqual(ids0 | ids1, set(range(_NUM_RECORDS)))


class SftTranslationMultiProcessTest(_MultiProcessShardingTest):
  WORKER_FLAG = "--worker-sft"

  def test_two_process_sharding_assembles_global_batch(self):
    self._run()


class GrpoPostInitMultiProcessTest(_MultiProcessShardingTest):
  WORKER_FLAG = "--worker-grpo"

  def test_two_process_sharding_assembles_global_batch(self):
    self._run()


class OddCountMultiProcessTest(absltest.TestCase):
  """Regression for B1: an odd record count must not hang multi-host training.

  Spawns the two real ``jax.distributed`` processes against
  ``post_init_dataset`` with ``_ODD_NUM_RECORDS`` (7) records, which is NOT a
  multiple of ``_NUM_PROCESSES`` (2). Each step runs the real cross-host
  ``shard_input``; if ``shard_by_process`` failed to equalize the shard lengths
  the host with fewer batches would exhaust first and the collective would block
  until the 300s timeout. The test asserts both hosts emit the SAME number of
  batches, read disjoint strided shards, and drop only the tail record (6).
  """

  def test_odd_record_count_yields_equal_batch_counts(self):
    import json
    import subprocess

    coordinator_address = f"localhost:{_pick_free_port()}"
    procs = [
        subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--worker-oddcount",
                str(process_id),
                coordinator_address,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for process_id in range(_NUM_PROCESSES)
    ]

    results = {}
    outputs = {}
    for process_id, proc in enumerate(procs):
      out, _ = proc.communicate(timeout=300)
      outputs[process_id] = out
      self.assertEqual(
          proc.returncode, 0, msg=f"worker {process_id} failed:\n{out}"
      )
      for line in out.splitlines():
        if line.startswith("RESULT "):
          results[process_id] = json.loads(line[len("RESULT ") :])

    self.assertLen(results, _NUM_PROCESSES, msg=outputs)

    # The load-bearing assertion: equal batch counts -> no host blocks forever.
    self.assertEqual(
        results[0]["num_batches"],
        results[1]["num_batches"],
        msg="hosts must emit equal batch counts or the collective hangs",
    )
    # 7 records / 2 procs -> truncate to 6 -> 3 records each -> local_bs 1 -> 3.
    self.assertEqual(results[0]["num_batches"], 3)

    ids0 = set(results[0]["consumed_ids"])
    ids1 = set(results[1]["consumed_ids"])
    self.assertEqual(ids0, {0, 2, 4})
    self.assertEqual(ids1, {1, 3, 5})
    self.assertEqual(ids0 & ids1, set(), msg="record shards must be disjoint")
    # The tail record beyond the even-length truncation is dropped on every host.
    self.assertNotIn(_ODD_NUM_RECORDS - 1, ids0 | ids1)


if __name__ == "__main__":
  if len(sys.argv) >= 4 and sys.argv[1] == "--worker-sft":
    _sft_worker(int(sys.argv[2]), sys.argv[3])
  elif len(sys.argv) >= 4 and sys.argv[1] == "--worker-grpo":
    _grpo_worker(int(sys.argv[2]), sys.argv[3])
  elif len(sys.argv) >= 4 and sys.argv[1] == "--worker-oddcount":
    _odd_count_worker(int(sys.argv[2]), sys.argv[3])
  else:
    absltest.main()
