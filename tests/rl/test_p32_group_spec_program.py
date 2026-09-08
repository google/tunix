"""The grouped update builds each group's schedule with one program.

``_p32_group_spec`` packs the valid prompt/completion tokens of one group,
derives the next-token targets, the completion source rows and the
per-rank lengths.  Eagerly that was ~40 tiny programs per group; the
device math now lives in one cached program per input shape.  The test
pins byte identity against a plain reference of the same math and the
launch budget.
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
      + " --xla_force_host_platform_device_count=16"
  ).strip()

import launch_budget  # noqa: E402  (patches JAX at import; before the adapter)
import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _adapter(sequence_bucket=4):
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  path = pathlib.Path(__file__).with_name("canonical_qwen3_adapter_test.py")
  spec = importlib.util.spec_from_file_location("group_spec_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  case = module.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, _ = case._make_p32_group_adapter(sequence_bucket=sequence_bucket)  # pylint: disable=protected-access
  return adapter


def _inputs():
  rng = np.random.default_rng(7)
  prompt = rng.integers(1, 9, size=(16, 5), dtype=np.int32)
  completion = rng.integers(1, 9, size=(16, 6), dtype=np.int32)
  # Ragged: rank r has 2 + r % 3 prompt tokens and 1 + r % 5 completion tokens,
  # left-padded prompts and right-padded completions like the learner's rows.
  prompt_len = 2 + np.arange(16) % 3
  completion_len = 1 + np.arange(16) % 5
  prompt_valid = np.arange(5)[None, :] >= (5 - prompt_len)[:, None]
  completion_valid = np.arange(6)[None, :] < completion_len[:, None]
  return prompt, completion, prompt_valid, completion_valid, prompt_len, completion_len


def _reference(adapter, prompt, completion, prompt_valid, completion_valid, temperature):
  bucket = adapter._sequence_bucket  # pylint: disable=protected-access
  width = adapter._p32_glue_width()  # pylint: disable=protected-access
  full = np.concatenate((prompt, completion), axis=1)
  valid = np.concatenate((prompt_valid, completion_valid), axis=1)
  n_real = valid.sum(axis=1).astype(np.int32)
  prompt_length = prompt_valid.sum(axis=1).astype(np.int32)
  num_chunks = int((n_real.max() + bucket - 1) // bucket)
  packed = np.zeros((16, width), np.int32)
  for row in range(16):
    tokens = full[row][valid[row]]
    packed[row, : tokens.size] = tokens
  next_ids = np.concatenate((packed[:, 1:], np.zeros((16, 1), np.int32)), axis=1)
  ordinal = np.cumsum(completion_valid, axis=1).astype(np.int32) - 1
  source_rows = np.clip(prompt_length[:, None] + ordinal - 1, 0, num_chunks * bucket - 1).astype(np.int32)
  return {
      "packed_ids": packed,
      "next_ids": next_ids,
      "source_rows": source_rows,
      "completion_valid": completion_valid,
      "n_real": n_real,
      "num_chunks": num_chunks,
      "temperature": np.asarray(temperature, np.float32),
  }


def test_group_spec_matches_the_reference_bitwise():
  adapter = _adapter()
  prompt, completion, prompt_valid, completion_valid, prompt_len, completion_len = _inputs()
  spec = adapter._p32_group_spec(  # pylint: disable=protected-access
      jnp.asarray(prompt), jnp.asarray(completion), jnp.asarray(prompt_valid),
      jnp.asarray(completion_valid), 0.7,
      host_prompt_length=prompt_len, host_completion_length=completion_len,
  )
  want = _reference(adapter, prompt, completion, prompt_valid, completion_valid, 0.7)
  assert spec["num_chunks"] == want["num_chunks"]
  assert spec["host_n_real"] == tuple(int(v) for v in want["n_real"])
  assert spec["host_completion_length"] == tuple(int(v) for v in completion_len)
  for key in ("packed_ids", "next_ids", "source_rows", "completion_valid", "n_real", "temperature"):
    got = np.asarray(spec[key])
    assert got.dtype == want[key].dtype, key
    assert got.tobytes() == want[key].tobytes(), key
  assert bool(spec["host_lengths_match"])


def test_group_spec_refuses_wrong_host_lengths():
  adapter = _adapter()
  prompt, completion, prompt_valid, completion_valid, prompt_len, completion_len = _inputs()
  spec = adapter._p32_group_spec(  # pylint: disable=protected-access
      jnp.asarray(prompt), jnp.asarray(completion), jnp.asarray(prompt_valid),
      jnp.asarray(completion_valid), 1.0,
      host_prompt_length=prompt_len, host_completion_length=completion_len + 1,
  )
  assert not bool(spec["host_lengths_match"])


def test_group_spec_costs_one_program_per_group():
  adapter = _adapter()
  prompt, completion, prompt_valid, completion_valid, prompt_len, completion_len = _inputs()
  args = (jnp.asarray(prompt), jnp.asarray(completion), jnp.asarray(prompt_valid), jnp.asarray(completion_valid), 1.0)
  kwargs = dict(host_prompt_length=prompt_len, host_completion_length=completion_len)
  adapter._p32_group_spec(*args, **kwargs)  # pylint: disable=protected-access  # compile
  with launch_budget.counting() as records:
    spec = adapter._p32_group_spec(*args, **kwargs)  # pylint: disable=protected-access
    jax.block_until_ready(spec["packed_ids"])
  launches = launch_budget.in_function(records, "_p32_group_spec")
  assert len(launches) <= 1, launch_budget.by_program(launches).most_common()


def test_group_spec_arrays_are_committed_to_the_engine_row_sharding():
  """Uncommitted spec arrays were re-sliced per device by shard_args on every
  program that consumed them (the census's jit__multi_slice launches); the
  pack program now pins its outputs to the rows-over-data sharding derived
  from the engine's input sharding (named ('dp', 'tp') on the one host)."""
  adapter = _adapter()
  engine_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:16]).reshape(8, 2), ("dp", "tp")
  )
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      engine_mesh, jax.sharding.PartitionSpec("dp")
  )
  adapter.__dict__.pop("_p32_group_pack_program", None)
  prompt, completion, prompt_valid, completion_valid, prompt_len, completion_len = _inputs()
  spec = adapter._p32_group_spec(  # pylint: disable=protected-access
      jnp.asarray(prompt), jnp.asarray(completion), jnp.asarray(prompt_valid),
      jnp.asarray(completion_valid), 0.7,
      host_prompt_length=prompt_len, host_completion_length=completion_len,
  )
  rows = jax.sharding.NamedSharding(engine_mesh, jax.sharding.PartitionSpec("dp"))
  scalar = jax.sharding.NamedSharding(engine_mesh, jax.sharding.PartitionSpec())
  for key in ("packed_ids", "next_ids", "source_rows", "completion_valid", "n_real"):
    assert spec[key].sharding == rows, key
    assert spec[key].committed, key
  assert spec["temperature"].sharding == scalar and spec["temperature"].committed
  want = _reference(adapter, prompt, completion, prompt_valid, completion_valid, 0.7)
  for key in ("packed_ids", "next_ids", "source_rows", "completion_valid", "n_real", "temperature"):
    assert np.asarray(spec[key]).tobytes() == want[key].tobytes(), key
  # Without an engine NamedSharding (the plain CPU fixture) nothing is pinned.
  adapter.__dict__.pop("_p32_group_pack_program", None)
  adapter._input_sharding = None  # pylint: disable=protected-access
  plain = adapter._p32_group_spec(  # pylint: disable=protected-access
      jnp.asarray(prompt), jnp.asarray(completion), jnp.asarray(prompt_valid),
      jnp.asarray(completion_valid), 0.7,
      host_prompt_length=prompt_len, host_completion_length=completion_len,
  )
  assert np.asarray(plain["packed_ids"]).tobytes() == want["packed_ids"].tobytes()


def test_split_groups_is_one_program_and_matches_the_eager_slices():
  """The rank-major regrouping and per-group split of the update's four
  stacks run as one program; each group's arrays are byte-identical to the
  eager _group_batch_rows slices and, with an engine sharding, committed to
  the spec row sharding."""
  adapter = _adapter()
  rng = np.random.default_rng(3)
  stacks = tuple(
      jnp.asarray(rng.integers(0, 9, size=(32, width), dtype=np.int32))
      for width in (5, 6, 5, 6)
  )
  fn = adapter._p32_split_groups_fn(2, 4)  # pylint: disable=protected-access
  fn(stacks)  # compile
  with launch_budget.counting() as records:
    groups = fn(stacks)
    jax.block_until_ready(groups)
  assert len(records) == 1, launch_budget.by_program(records).most_common()
  assert len(groups) == 2 and all(len(group) == 4 for group in groups)
  for index in range(2):
    for stack, got in zip(stacks, groups[index]):
      want = adapter._group_batch_rows(stack)[index]  # pylint: disable=protected-access
      assert np.asarray(got).tobytes() == np.asarray(want).tobytes()
      assert got.shape == want.shape and got.dtype == want.dtype
  engine_mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:16]).reshape(8, 2), ("dp", "tp")
  )
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      engine_mesh, jax.sharding.PartitionSpec("dp")
  )
  adapter.__dict__.pop("_p32_split_groups_programs", None)
  committed = adapter._p32_split_groups_fn(2, 4)(stacks)  # pylint: disable=protected-access
  rows = jax.sharding.NamedSharding(engine_mesh, jax.sharding.PartitionSpec("dp"))
  for group in committed:
    for value in group:
      assert value.sharding == rows and value.committed
  assert np.asarray(committed[1][2]).tobytes() == np.asarray(groups[1][2]).tobytes()
