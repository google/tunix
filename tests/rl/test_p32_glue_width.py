"""The grouped update's glue arrays have one fixed width per run.

`_p32_group_spec` lays the packed rows out at the model limit rounded up
to whole chunks, whatever the group's longest row, so the programs that
carry them (packing, the flattened logps, the flattened cotangents)
compile once instead of once per distinct chunk count.  The chunk loop
still runs `num_chunks` passes and masked slots still clip to the last
column of the last real chunk.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np

_HARNESS_PATH = Path(__file__).with_name("test_p71_fwd_scan.py")
_spec = importlib.util.spec_from_file_location("p71_harness_glue", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


def _spec_for(adapter, prompt_tokens, completion_tokens):
  row = jnp.arange(16, dtype=jnp.int32)[:, None]
  prompt = jnp.concatenate([1 + row % (2 + i) for i in range(prompt_tokens)], axis=1)
  completion = jnp.concatenate([2 + row % (2 + i) for i in range(completion_tokens)], axis=1)
  return adapter._p32_group_spec(  # pylint: disable=protected-access
      prompt,
      completion,
      jnp.ones_like(prompt, dtype=bool),
      jnp.ones_like(completion, dtype=bool),
      1.0,
  )


def test_glue_width_is_fixed_across_chunk_counts():
  adapter, _ = harness._group_adapter(rank_parallel=False)  # pylint: disable=protected-access
  bucket = adapter._sequence_bucket  # pylint: disable=protected-access
  width = adapter._p32_glue_width()  # pylint: disable=protected-access
  assert width % bucket == 0 and width >= adapter._max_model_len  # pylint: disable=protected-access
  two = _spec_for(adapter, 3, 3)  # n_real = 6 > bucket = 4: two chunks
  one = _spec_for(adapter, 2, 1)  # n_real = 3: one chunk
  assert (two["num_chunks"], one["num_chunks"]) == (2, 1)
  for spec in (two, one):
    assert spec["packed_ids"].shape == (adapter._data_size, width)  # pylint: disable=protected-access
    assert spec["next_ids"].shape == (adapter._data_size, width)  # pylint: disable=protected-access
    # Nothing real lives past n_real; the tail is zeros in both layouts.
    packed = np.asarray(spec["packed_ids"])
    for rank, count in enumerate(np.asarray(spec["n_real"])):
      assert not packed[rank, count:].any()
    # Masked completion slots clip to the last column of the last real chunk.
    assert int(np.asarray(spec["source_rows"]).max()) <= spec["num_chunks"] * bucket - 1
  # The chunk inputs come out identical whether the row layout is wide or
  # not: the operand-driven slice reads the same columns.
  ids, targets, _ = adapter._p32_group_chunk_inputs(two, 1)  # pylint: disable=protected-access
  packed = np.asarray(two["packed_ids"])
  assert np.array_equal(np.asarray(ids), packed[:, bucket : 2 * bucket].reshape(-1))
  assert np.array_equal(
      np.asarray(targets), np.asarray(two["next_ids"])[:, bucket : 2 * bucket].reshape(-1)
  )
