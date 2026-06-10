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

  print(
      "RESULT "
      + json.dumps({
          "process_id": process_id,
          "process_count": jax.process_count(),
          "local_batch_shape": list(local_tokens.shape),
          "global_shape": list(global_arr.shape),
          "local_record_ids": local_record_ids,
          "fully_addressable": bool(global_arr.is_fully_addressable),
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
