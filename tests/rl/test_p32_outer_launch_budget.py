"""Pins the eager launches of the grouped update's OUTER loop per group (CPU).

The chunk loops are covered by test_p32_launch_budget.py; this file runs
the whole streamed update with the stream-tape harness (fake chunk loops,
real outer loop) on 64 forced CPU devices and counts the launches that
``stream_lookahead`` issues per group.
"""

import importlib.util
import os
import pathlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=64"
  ).strip()

import launch_budget  # noqa: E402  (patches JAX at import; before the adapter)
import types
from unittest import mock

import jax
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter


def _load(name):
  path = pathlib.Path(__file__).with_name(name)
  spec = importlib.util.spec_from_file_location(name[:-3], path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_stream_lookahead_issues_few_programs_per_group():
  if len(jax.devices()) < 64:
    pytest.skip("requires sixty-four forced CPU devices")
  tape = _load("test_p32_stream_tape.py")
  tape._run("stream", reduce_once="1")  # pylint: disable=protected-access  # compile
  with launch_budget.counting() as records:
    result, *_ = tape._run("stream", reduce_once="1")  # pylint: disable=protected-access
  groups = len(result["reports"])
  lookahead = launch_budget.in_function(records, "stream_lookahead")
  named = set(launch_budget.named_programs(lookahead))
  eager = [entry for entry in lookahead if entry not in named]
  per_group = len(eager) / groups
  # One stream-step program per group replaces the two scatters, the
  # cotangent program, the two group-row slices and the logps slice.
  assert per_group <= 2.0, (per_group, launch_budget.by_site(eager).most_common())


def test_group_specs_take_host_lengths_and_launch_no_lengths_program():
  """tasks/v2_dispatch Phase 11: the update hands every group its host row
  lengths, so the per-group lengths program (and its device_get) is gone
  while the pack program still runs once per group."""
  if len(jax.devices()) < 64:
    pytest.skip("requires sixty-four forced CPU devices")
  tape = _load("test_p32_stream_tape.py")
  tape._run("stream", reduce_once="1")  # pylint: disable=protected-access  # compile
  with launch_budget.counting() as records:
    result, *_ = tape._run("stream", reduce_once="1")  # pylint: disable=protected-access
  programs = launch_budget.by_program(records)
  groups = len(result["reports"])
  assert programs.get("group_lengths", 0) == 0, programs
  assert programs.get("group_pack", 0) == groups, programs


def test_update_refuses_host_lengths_that_disagree_with_the_masks():
  """The pack program's host-vs-device length check is verified once per
  update, before anything is committed."""
  if len(jax.devices()) < 64:
    pytest.skip("requires sixty-four forced CPU devices")
  tape = _load("test_p32_stream_tape.py")
  case = tape.harness.CanonicalQwen3AdapterTest(
      "test_p32_dp16_rejects_the_legacy_data1_segmented_reverse"
  )
  adapter, _ = case._make_p32_group_adapter(sequence_bucket=256)  # pylint: disable=protected-access
  original = adapter._p32_group_spec  # pylint: disable=protected-access

  def lying_group_spec(self, *args, **kwargs):
    del self
    kwargs["host_prompt_length"] = (
        np.asarray(kwargs["host_prompt_length"], np.int32) + 1
    )
    return original(*args, **kwargs)

  adapter._p32_group_spec = types.MethodType(lying_group_spec, adapter)  # pylint: disable=protected-access
  with pytest.raises(
      canonical_qwen3_adapter.FunctionalMappingError, match="host row lengths"
  ):
    tape._run("stream", reduce_once="1", adapter=adapter)  # pylint: disable=protected-access


def test_stream_step_takes_the_example_and_the_index_committed_to_the_mesh():
  """tasks/v2_dispatch Phase 11: with a NamedSharding engine every example
  leaf the loss reads and the group index reach the stream step committed
  to the engine mesh (replicated), so no call re-slices or re-copies them
  (the jit__multi_slice launch and the host copies per group on the
  one-host census); the update's outputs are unchanged."""
  if len(jax.devices()) < 64:
    pytest.skip("requires sixty-four forced CPU devices")
  tape = _load("test_p32_stream_tape.py")
  case = tape.harness.CanonicalQwen3AdapterTest(
      "test_p32_dp16_rejects_the_legacy_data1_segmented_reverse"
  )
  adapter, _ = case._make_p32_group_adapter(sequence_bucket=256)  # pylint: disable=protected-access
  # The harness's own (16, 4) data/model mesh over the sixty-four devices.
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(16, 4), ("data", "model")
  )
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      mesh, jax.sharding.PartitionSpec("data", None)
  )
  expected = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
  seen = []
  original_jit = jax.jit

  def checking_jit(fun, *args, **kwargs):
    compiled = original_jit(fun, *args, **kwargs)
    if getattr(fun, "__name__", "") != "stream_step":
      return compiled

    def checked(stream_logps, stream_entropy, logps, entropy, index, example):
      assert isinstance(index, jax.Array) and index.committed
      assert index.sharding == expected, index.sharding
      for name in canonical_qwen3_adapter._P32_EXAMPLE_FIELDS:  # pylint: disable=protected-access
        leaf = getattr(example, name, None)
        if leaf is None:
          continue
        assert isinstance(leaf, jax.Array) and leaf.committed, name
        assert leaf.sharding == expected, (name, leaf.sharding)
      seen.append(int(index))
      return compiled(stream_logps, stream_entropy, logps, entropy, index, example)

    return checked

  with mock.patch.object(jax, "jit", checking_jit):
    result, *_ = tape._run("stream", reduce_once="1", adapter=adapter)  # pylint: disable=protected-access
  assert seen == list(range(len(result["reports"])))
  # The committed example is a new object; the caller's example is untouched.
  example = tape._train_example()  # pylint: disable=protected-access
  committed = adapter._p32_commit_example(example)  # pylint: disable=protected-access
  assert committed is not example and not example.advantages.committed
  assert committed.advantages.sharding == expected
  assert np.asarray(committed.advantages).tobytes() == np.asarray(example.advantages).tobytes()
  assert adapter._p32_commit_example(committed) is committed  # pylint: disable=protected-access
  scalar = adapter._p32_group_index_scalar(3)  # pylint: disable=protected-access
  assert scalar is adapter._p32_group_index_scalar(3) and int(scalar) == 3  # pylint: disable=protected-access
