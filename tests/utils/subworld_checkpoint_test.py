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

"""Unit tests for sub-world Orbax checkpointing utilities."""

from __future__ import annotations

import contextlib
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import jax
import orbax.checkpoint as ocp_v0
from orbax.checkpoint import checkpoint_manager as ocp_cm
from orbax.checkpoint import multihost as ocp_multihost
from orbax.checkpoint import v1 as ocp_v1
from tunix.utils import subworld_checkpoint


@contextlib.contextmanager
def _world(
    *,
    process_count: int,
    runtime_id: int,
    distributed_ids: dict[int, int] | None = None,
):
  """Simulates this process in a `process_count`-process JAX world.

  Args:
    process_count: Number of processes in the simulated world.
    runtime_id: This process's `jax.process_index()`.
    distributed_ids: Runtime -> distributed id map that the collective gather
      returns; identity when None.
  """

  def _gather(procs):
    if distributed_ids is None:
      return {p: p for p in procs}
    return {p: distributed_ids[p] for p in procs}

  with (
      mock.patch.object(jax, "process_count", return_value=process_count),
      mock.patch.object(jax, "process_index", return_value=runtime_id),
      mock.patch.object(
          subworld_checkpoint, "_gather_distributed_ids", side_effect=_gather
      ),
  ):
    yield


class SubworldCheckpointTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # pylint: disable-next=protected-access
    subworld_checkpoint._DISTRIBUTED_IDS.clear()

  def test_is_subworld(self):
    with _world(process_count=4, runtime_id=1):
      self.assertTrue(subworld_checkpoint.is_subworld({1, 2}))
      self.assertFalse(subworld_checkpoint.is_subworld({0, 1, 2, 3}))
      self.assertFalse(subworld_checkpoint.is_subworld(None))
    self.assertFalse(subworld_checkpoint.is_subworld({0}))

  def test_multiprocessing_options_scopes_primary_host_and_prefix(self):
    with _world(process_count=4, runtime_id=2):
      mp = subworld_checkpoint.multiprocessing_options(
          {2, 1}, barrier_sync_key_prefix="trainer"
      )
    self.assertEqual(
        mp,
        ocp_v1.options.MultiprocessingOptions(
            primary_host=1,
            active_processes={1, 2},
            barrier_sync_key_prefix="trainer",
        ),
    )

  def test_multiprocessing_options_rejects_outside_process(self):
    with _world(process_count=4, runtime_id=0):
      with self.assertRaisesRegex(ValueError, "not in active_processes"):
        subworld_checkpoint.multiprocessing_options({1, 2})
      with self.assertRaisesRegex(ValueError, "non-empty"):
        subworld_checkpoint.multiprocessing_options(set())

  def test_rejects_id_mismatch_without_distributed_id_mode(self):
    with _world(process_count=4, runtime_id=1, distributed_ids={1: 5, 2: 7}):
      with self.assertRaisesRegex(RuntimeError, "enable_distributed_proc"):
        subworld_checkpoint.multiprocessing_options({1, 2})
      with self.assertRaisesRegex(RuntimeError, "enable_distributed_proc"):
        subworld_checkpoint.configure_v1_context(ocp_v1.Context(), {1, 2})

  @flagsaver.flagsaver
  def test_translates_to_distributed_ids_in_distributed_id_mode(self):
    with (
        mock.patch.object(
            ocp_multihost,
            "is_runtime_to_distributed_ids_initialized",
            return_value=False,
        ),
        mock.patch.object(
            ocp_multihost, "initialize_runtime_to_distributed_ids"
        ) as init_ids,
    ):
      subworld_checkpoint.enable_distributed_process_ids()
    init_ids.assert_called_once()
    self.assertTrue(flags.FLAGS.experimental_orbax_use_distributed_process_id)

    with _world(process_count=4, runtime_id=1, distributed_ids={1: 7, 2: 5}):
      mp = subworld_checkpoint.multiprocessing_options({1, 2})
      ctx = subworld_checkpoint.configure_v1_context(ocp_v1.Context(), {1, 2})
    self.assertEqual(mp.primary_host, 5)
    self.assertEqual(mp.active_processes, {5, 7})
    self.assertEqual(ctx.multiprocessing_options.primary_host, 5)
    self.assertEqual(ctx.multiprocessing_options.active_processes, {5, 7})

  def test_configure_v1_context_scopes_primary_host_and_prefix(self):
    base = ocp_v1.Context()
    base.multiprocessing.barrier_sync_key_prefix = "keep_me"
    with _world(process_count=4, runtime_id=1):
      scoped = subworld_checkpoint.configure_v1_context(base, {1, 2})
      full_world = subworld_checkpoint.configure_v1_context(base, {0, 1, 2, 3})
    self.assertEqual(scoped.multiprocessing_options.primary_host, 1)
    self.assertEqual(scoped.multiprocessing_options.active_processes, {1, 2})
    self.assertEqual(
        scoped.multiprocessing_options.barrier_sync_key_prefix, "keep_me"
    )
    # The base context is left unscoped, and a full world keeps it as is.
    self.assertEqual(base.multiprocessing.primary_host, 0)
    self.assertIsNone(base.multiprocessing.active_processes)
    self.assertIs(full_world, base)

  def test_v1_checkpointer_creates_root_directory_with_active_processes(self):
    root = os.path.join(self.create_tempdir().full_path, "a", "b")
    context = ocp_v1.Context()
    context.multiprocessing.active_processes = {0}

    subworld_checkpoint.allow_subworld_root_directory_creation()
    subworld_checkpoint.allow_subworld_root_directory_creation()  # Idempotent.
    with ocp_v1.training.Checkpointer(root, context=context) as ckptr:
      ckptr.save_pytree(1, {"x": jax.numpy.arange(4)})
      self.assertEqual(ckptr.latest.step, 1)
    self.assertTrue(os.path.isdir(root))

    # Without `active_processes`, Orbax's own implementation runs.
    full_root = os.path.join(self.create_tempdir().full_path, "full")
    ocp_cm._create_root_directory(  # pylint: disable=protected-access
        full_root, ocp_v0.options.MultiprocessingOptions()
    )
    self.assertTrue(os.path.isdir(full_root))

  def test_root_directory_patch_skipped_when_orbax_supports_it(self):
    orig = ocp_cm._create_root_directory  # pylint: disable=protected-access
    with (
        mock.patch.object(
            subworld_checkpoint, "_ROOT_DIRECTORY_CHECKED", False
        ),
        mock.patch.object(
            subworld_checkpoint,
            "_orbax_creates_subworld_root_directory",
            return_value=True,
        ) as probe,
    ):
      subworld_checkpoint.allow_subworld_root_directory_creation()
    probe.assert_called_once()
    # pylint: disable-next=protected-access
    self.assertIs(ocp_cm._create_root_directory, orig)


if __name__ == "__main__":
  absltest.main()
