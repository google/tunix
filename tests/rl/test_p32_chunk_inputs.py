"""Chunk inputs for the grouped forward/reverse are one program per chunk.

``_p32_group_chunk_inputs`` builds one chunk's engine-call inputs (ids,
targets, RPA metadata).  The eager arm issued ~61 tiny programs per call
(measured on the one-host xplanes: 122 launches per chunk pass, a third of
all dispatches); the fused program ``_fused_chunk_metadata`` computes the
same integers in one program whose outputs already carry the engine input
sharding.  These tests pin (a) byte-identical outputs against the eager
reference and (b) the launch budget of one program per call.
"""

import dataclasses
import importlib.util
import os
import pathlib
import types
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get(
    "XLA_FLAGS", ""
):
  os.environ["XLA_FLAGS"] = (
      os.environ.get("XLA_FLAGS", "")
      + " --xla_force_host_platform_device_count=16"
  ).strip()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax._src import dispatch as jax_dispatch
from tunix.rl import canonical_qwen3_adapter


def _fixtures():
  path = pathlib.Path(__file__).with_name("canonical_qwen3_adapter_test.py")
  spec = importlib.util.spec_from_file_location("p32_chunk_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _Metadata:
  input_positions: object
  block_tables: object
  seq_lens: object
  query_start_loc: object
  request_distribution: object


_FIELDS = (
    "input_positions",
    "block_tables",
    "seq_lens",
    "query_start_loc",
    "request_distribution",
)


def _adapter(sequence_bucket=4):
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  fixtures = _fixtures()
  case = fixtures.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, _ = case._make_p32_group_adapter(sequence_bucket=sequence_bucket)  # pylint: disable=protected-access
  # The fixture stubs the chunk-input builder; put the production one back.
  adapter._p32_group_chunk_inputs = types.MethodType(  # pylint: disable=protected-access
      canonical_qwen3_adapter.Qwen3EngineForwardAdapter._p32_group_chunk_inputs,  # pylint: disable=protected-access
      adapter,
  )
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()[:16]).reshape(16, 1), ("data", "model")
  )
  adapter._input_sharding = jax.sharding.NamedSharding(  # pylint: disable=protected-access
      mesh, jax.sharding.PartitionSpec("data")
  )
  adapter._metadata_cls = _Metadata  # pylint: disable=protected-access
  return adapter


def _three_chunk_spec(adapter):
  row = jnp.arange(16, dtype=jnp.int32)[:, None]
  prompt = jnp.concatenate((1 + row % 2, 2 + row % 2, 1 + row % 3), axis=1)
  completion = jnp.concatenate(
      (2 + row % 2, 1 + row % 3, 1 + row % 2, 3 + row % 2, 1 + row % 4),
      axis=1,
  )
  # Ragged completions: rank r keeps 5 - (r % 3) completion tokens, so the
  # late chunks mix active and empty rows (q_len == 0) across ranks.
  completion_mask = (jnp.arange(5)[None, :] < (5 - row % 3)).astype(bool)
  spec = adapter._p32_group_spec(  # pylint: disable=protected-access
      prompt,
      completion,
      jnp.ones_like(prompt, dtype=bool),
      completion_mask,
      1.0,
  )
  assert spec["num_chunks"] == 2
  return spec


def _eager_reference(adapter, spec, chunk_index):
  """The retired eager arm, kept verbatim as the test oracle."""
  bucket = adapter._sequence_bucket  # pylint: disable=protected-access
  start = chunk_index * bucket
  rows = jnp.arange(bucket, dtype=jnp.int32)
  q_len = jnp.clip(spec["n_real"] - start, 0, bucket)
  kv_len = jnp.where(
      q_len > 0, jnp.minimum(spec["n_real"], start + bucket), 0
  )
  ids = spec["packed_ids"][:, start:start + bucket]
  targets = spec["next_ids"][:, start:start + bucket]
  positions = jnp.where(rows[None, :] < q_len[:, None], start + rows[None, :], 0)
  block_tables, seq_lens, query_start, request_distribution = (
      canonical_qwen3_adapter._canonical_dp_attention_metadata_arrays(  # pylint: disable=protected-access
          data_size=adapter._data_size,  # pylint: disable=protected-access
          max_num_reqs=adapter._max_num_reqs,  # pylint: disable=protected-access
          blocks_per_req=adapter._blocks_per_req,  # pylint: disable=protected-access
          q_len=q_len,
          kv_len=kv_len,
      )
  )
  return {
      "ids": ids.reshape(-1),
      "targets": targets.reshape(-1),
      "input_positions": positions.reshape(-1),
      "block_tables": block_tables,
      "seq_lens": seq_lens,
      "query_start_loc": query_start,
      "request_distribution": request_distribution,
  }


def _as_dict(ids, targets, metadata):
  values = {"ids": ids, "targets": targets}
  for field in _FIELDS:
    values[field] = getattr(metadata, field)
  return values


def test_chunk_inputs_match_the_eager_reference_bitwise_on_every_chunk():
  adapter = _adapter()
  spec = _three_chunk_spec(adapter)
  for chunk_index in range(spec["num_chunks"]):
    ids, targets, metadata = adapter._p32_group_chunk_inputs(spec, chunk_index)  # pylint: disable=protected-access
    got = _as_dict(ids, targets, metadata)
    want = _eager_reference(adapter, spec, chunk_index)
    for key, value in want.items():
      assert got[key].dtype == value.dtype, (chunk_index, key)
      assert got[key].shape == value.shape, (chunk_index, key)
      assert np.asarray(got[key]).tobytes() == np.asarray(value).tobytes(), (
          chunk_index,
          key,
      )
    assert metadata.padded_num_reqs == adapter._max_num_reqs  # pylint: disable=protected-access


def test_chunk_inputs_land_on_the_engine_input_sharding():
  adapter = _adapter()
  spec = _three_chunk_spec(adapter)
  sharding = adapter._input_sharding  # pylint: disable=protected-access
  for chunk_index in range(spec["num_chunks"]):
    ids, targets, metadata = adapter._p32_group_chunk_inputs(spec, chunk_index)  # pylint: disable=protected-access
    for key, value in _as_dict(ids, targets, metadata).items():
      assert value.sharding == sharding, (chunk_index, key)


def test_chunk_inputs_cost_one_program_and_no_reshard_per_call():
  adapter = _adapter()
  spec = _three_chunk_spec(adapter)
  adapter._p32_group_chunk_inputs(spec, 0)  # pylint: disable=protected-access  # compile
  primitive_calls = []
  original = jax_dispatch.apply_primitive

  def counting_apply(prim, *args, **params):
    primitive_calls.append(prim.name)
    return original(prim, *args, **params)

  with mock.patch.object(jax_dispatch, "apply_primitive", counting_apply), \
      mock.patch.object(
          canonical_qwen3_adapter.jax, "reshard", side_effect=AssertionError("reshard")
      ) as reshard:
    for chunk_index in range(spec["num_chunks"]):
      ids, targets, metadata = adapter._p32_group_chunk_inputs(spec, chunk_index)  # pylint: disable=protected-access
      jax.block_until_ready((ids, targets, metadata))
  assert reshard.call_count == 0
  # One fused program per call; the only other device work is the
  # chunk_start scalar transfer, which is not a primitive application.
  assert primitive_calls == [], primitive_calls
