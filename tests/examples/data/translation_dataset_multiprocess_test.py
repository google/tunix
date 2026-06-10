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

"""Real 2-process ``jax.distributed`` simulation of multi-controller SFT data.

This is the load-bearing proof that the per-process data sharding in
``translation_dataset`` is correct under genuine multi-controller JAX. It does
NOT mock ``jax.process_count``: it spawns two OS processes that each call
``jax.distributed.initialize()`` against a shared coordinator (the exact same
mechanism a 2-host TPU job uses), builds the per-process data loader, and then
runs the real ``sharding_utils.shard_input`` to assemble the global batch from
the two process-local sub-batches.

It asserts:

  * Each process initializes with ``process_count == 2`` and a disjoint
    ``process_index``.
  * Each process's loader yields a LOCAL batch of ``global_batch // 2`` rows.
  * The two processes read DISJOINT record shards.
  * ``shard_input`` -> ``jax.make_array_from_process_local_data`` assembles a
    global array of shape ``(global_batch, seq_len)`` that is NOT fully
    addressable (it spans both processes), and each process's addressable slice
    equals the local batch it fed in. This is the precise
    ``make_array_from_process_local_data`` contract the SFT trainer relies on.
  * A loss-style token-weighted ``sum(value)/sum(weight)`` reduction, computed
    inside a ``jax.jit`` on the GLOBAL mesh over the process-local shards,
    produces a REPLICATED scalar that equals the reduction over the FULL global
    batch and is NOT equal to either process's local-shard-only value. This is
    the load-bearing proof that the loss / grad_norm scalars the SFT trainer
    feeds to ``metrics_logger`` (which only reads them on ``process_index == 0``)
    are the true GLOBAL reductions, not process-0's local-shard value.

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


def _worker_main(process_id: int, coordinator_address: str) -> None:
  """Body run inside each spawned process.

  Prints ``RESULT <process_id> <json>`` on success or raises (non-zero exit) on
  failure so the parent test can assert per-process outcomes.
  """
  # Must be set before jax initializes its backend.
  os.environ["XLA_FLAGS"] = (
      f"--xla_force_host_platform_device_count={_DEVICES_PER_PROCESS}"
  )
  os.environ["JAX_PLATFORMS"] = "cpu"

  import json

  import jax
  import jax.numpy as jnp
  import jax.sharding as shd
  import numpy as np

  jax.distributed.initialize(
      coordinator_address=coordinator_address,
      num_processes=_NUM_PROCESSES,
      process_id=process_id,
  )

  # Import after distributed init so jax.process_count() is already correct
  # wherever module-level code might read it.
  from tunix.examples.data import translation_dataset
  from tunix.sft import sharding_utils

  assert jax.process_count() == _NUM_PROCESSES, jax.process_count()
  assert jax.process_index() == process_id, jax.process_index()
  assert jax.device_count() == _NUM_PROCESSES * _DEVICES_PER_PROCESS

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
  # Record ids this process consumed across ALL of its batches (token0 ==
  # record_id + 1). This is what proves the shards are disjoint and cover every
  # record.
  local_record_ids = []
  for batch in all_batches:
    tokens = np.asarray(batch.input_tokens)
    # Every batch this process produces is exactly the LOCAL batch size.
    assert tokens.shape == (local_bs, _SEQ_LEN), tokens.shape
    local_record_ids.extend(int(row[0]) - 1 for row in tokens)
  local_record_ids = sorted(local_record_ids)

  # Use the first batch to exercise shard_input below.
  local_batch = all_batches[0]
  local_tokens = np.asarray(local_batch.input_tokens)

  # Build the GLOBAL mesh spanning both processes and shard the local batch via
  # the real production code path.
  mesh = shd.Mesh(
      np.array(jax.devices()).reshape(jax.device_count(), 1),
      axis_names=("fsdp", "tp"),
  )
  with mesh:
    global_arr = sharding_utils.shard_input(local_batch.input_tokens, ("fsdp",))

  assert global_arr.shape == (_GLOBAL_BATCH, _SEQ_LEN), global_arr.shape
  # The assembled global array spans both processes, so it is not fully
  # addressable from a single process.
  assert not global_arr.is_fully_addressable

  # This process's addressable slice must equal the local batch it fed in.
  addressable = np.concatenate(
      [np.asarray(shard.data) for shard in global_arr.addressable_shards],
      axis=0,
  )
  addressable = addressable[np.argsort(addressable[:, 0])]
  expected = local_tokens[np.argsort(local_tokens[:, 0])]
  assert np.array_equal(addressable, expected), (addressable, expected)

  # ---------------------------------------------------------------------------
  # Load-bearing GLOBAL-metric proof.
  #
  # The SFT trainer computes loss as sum(per-token NLL) / sum(target_mask) and
  # grad_norm via optax.global_norm INSIDE a jit on the global mesh. Both are
  # full-array reductions over a process-sharded array, so XLA inserts an
  # all-reduce and the resulting scalar is REPLICATED (identical on every
  # process). metrics_logger then reads that scalar only on process 0. We prove
  # that contract here with a representative loss-style sum/sum reduction over
  # the global array assembled from the per-process LOCAL shards.
  #
  # Define a per-example weight/value from the row's record token so the two
  # processes contribute DIFFERENT quantities; this guarantees the global mean
  # cannot accidentally equal a single process's local-only mean.
  def _value(tokens):  # token-weighted "loss-like" quantity, jnp or np.
    rec = tokens[:, 0].astype(jnp.float32)  # == record_id + 1
    weight = rec  # per-example token weight (analogous to target_mask sum)
    value = rec * rec  # per-example weighted contribution
    return jnp.sum(value), jnp.sum(weight)

  @jax.jit
  def _global_token_weighted_mean(global_tokens):
    total_value, total_weight = _value(global_tokens)
    return total_value / total_weight  # global sum / global sum

  with mesh:
    global_mean_arr = _global_token_weighted_mean(global_arr)
  # A scalar produced by reductions over a sharded array is REPLICATED: its
  # sharding carries an empty PartitionSpec (no dimension is sharded), so every
  # process holds the identical value. (`is_fully_addressable` can still be
  # False because the replicated sharding nominally spans all processes'
  # devices; replication is the property that matters, and the parent test
  # additionally cross-checks the value is identical across processes.)
  metric_replicated = bool(global_mean_arr.sharding.spec == shd.PartitionSpec())
  assert metric_replicated, (
      "global metric must be replicated",
      global_mean_arr.sharding.spec,
  )
  global_mean = float(jax.device_get(global_mean_arr))

  # numpy reference over the FULL global batch (both processes' records). The
  # fake tokenizer maps record i -> token value (i+1).
  def _np_mean(record_ids):
    rec = np.array([rid + 1 for rid in record_ids], dtype=np.float64)
    return float(np.sum(rec * rec) / np.sum(rec))

  # Records in THIS process's local first batch (the one fed to shard_input).
  local_first_ids = sorted(int(row[0]) - 1 for row in local_tokens)
  # The full global batch is the union of every process's first-batch shard.
  # ShardByJaxProcess assigns process p the contiguous record block starting at
  # p * (_NUM_RECORDS // _NUM_PROCESSES), and grain.Batch takes the first
  # local_bs of those for the first batch. With _NUM_RECORDS=8,
  # _NUM_PROCESSES=2, local_bs=2 that is {0,1} for p0 and {4,5} for p1.
  full_global_ids = sorted(
      p * (_NUM_RECORDS // _NUM_PROCESSES) + j
      for p in range(_NUM_PROCESSES)
      for j in range(local_bs)
  )

  global_ref = _np_mean(full_global_ids)
  local_only_ref = _np_mean(local_first_ids)

  assert abs(global_mean - global_ref) < 1e-4, (global_mean, global_ref)
  # The proof: the jitted global reduction is NOT process p's local-only value.
  assert abs(global_mean - local_only_ref) > 1e-4, (
      global_mean,
      local_only_ref,
  )

  print(
      "RESULT "
      + json.dumps({
          "process_id": process_id,
          "process_count": jax.process_count(),
          "local_batch_shape": list(local_tokens.shape),
          "global_shape": list(global_arr.shape),
          "local_record_ids": local_record_ids,
          "fully_addressable": bool(global_arr.is_fully_addressable),
          "global_metric_replicated": metric_replicated,
          "global_metric": global_mean,
          "global_metric_ref": global_ref,
          "local_only_metric_ref": local_only_ref,
      }),
      flush=True,
  )
  jax.distributed.shutdown()


class TranslationDatasetMultiProcessTest(absltest.TestCase):

  def test_two_process_sharding_assembles_global_batch(self):
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
                  "--worker",
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
          proc.returncode,
          0,
          msg=f"worker {process_id} failed:\n{out}",
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

      # GLOBAL-metric proof (mirrors the loss / grad_norm reduction path):
      #   * the jitted sum/sum reduction over the sharded global array yields a
      #     REPLICATED scalar (fully addressable on every process),
      #   * it equals the reduction over the FULL global batch, and
      #   * it does NOT equal this process's local-shard-only value.
      self.assertTrue(
          res["global_metric_replicated"],
          msg="global loss-style metric must be a replicated scalar",
      )
      self.assertAlmostEqual(
          res["global_metric"], res["global_metric_ref"], places=4
      )
      self.assertNotAlmostEqual(
          res["global_metric"], res["local_only_metric_ref"], places=4
      )

    # Both processes must observe the SAME global metric value (replicated).
    self.assertAlmostEqual(
        results[0]["global_metric"], results[1]["global_metric"], places=6
    )
    # And the two processes' local-only values genuinely differ, so "equals the
    # global, differs from local-only" is a real (non-degenerate) discriminator.
    self.assertNotAlmostEqual(
        results[0]["local_only_metric_ref"],
        results[1]["local_only_metric_ref"],
        places=4,
    )

    # The two processes read disjoint record shards covering all records.
    ids0 = set(results[0]["local_record_ids"])
    ids1 = set(results[1]["local_record_ids"])
    self.assertEqual(ids0 & ids1, set(), msg="record shards must be disjoint")
    self.assertEqual(ids0 | ids1, set(range(_NUM_RECORDS)))


if __name__ == "__main__":
  if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
    _worker_main(int(sys.argv[2]), sys.argv[3])
  else:
    absltest.main()
