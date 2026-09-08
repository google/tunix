"""The learner's per-microbatch sidecar rows are sliced on the host.

The alignment batch check hashes host bytes, so one device-to-host copy per
per-trajectory leaf per update replaces one device gather and one copy per
leaf per microbatch (1,760 launches per update on the one-host census).
Byte-identical rows, zero device launches per microbatch.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import launch_budget  # noqa: E402  (patches JAX at import; before the learner)
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import alignment
from tunix.rl.agentic import agentic_rl_learner


def _sidecar(num_trajectories=16, width=5):
  rng = np.random.default_rng(11)
  rows = lambda dtype=np.float32: jnp.asarray(  # pylint: disable=g-long-lambda
      rng.standard_normal((num_trajectories, width)).astype(dtype)
  )
  return alignment.ObservedTrainExample(
      train_example={
          "prompt_ids": jnp.asarray(rng.integers(1, 9, (num_trajectories, 3), dtype=np.int32)),
          "advantages": jnp.asarray(rng.standard_normal(num_trajectories).astype(np.float32)),
      },
      s_decode=rows(),
      s_prefill=rows(),
      t_old=rows(),
      action_mask=jnp.asarray(rng.integers(0, 2, (num_trajectories, width)).astype(bool)),
      completion_valid_mask=jnp.asarray(rng.integers(0, 2, (num_trajectories, width)).astype(bool)),
      prompt_mask=jnp.asarray(rng.integers(0, 2, (num_trajectories, 3)).astype(bool)),
      tokens=jnp.asarray(rng.integers(1, 9, (num_trajectories, width), dtype=np.int32)),
      policy_version=jnp.asarray(7, jnp.int32),
      sampling_values=jnp.asarray(rng.standard_normal((num_trajectories, 3)).astype(np.float32)),
  )


def _device_rows(sidecar, logps, rows, num_trajectories):
  index = np.asarray(rows, dtype=np.int32)
  pair = jax.tree.map(
      lambda value: (
          value[index]
          if hasattr(value, "shape") and value.shape and value.shape[0] == num_trajectories
          else value
      ),
      sidecar,
  )
  return pair, logps[index]


def test_host_views_match_the_device_gathers_bitwise_and_launch_nothing():
  num_trajectories = 16
  sidecar = _sidecar(num_trajectories)
  logps = jnp.asarray(np.random.default_rng(5).standard_normal((num_trajectories, 5)).astype(np.float32))
  microbatches = [(0, 1, 2, 3), (4, 5, 6, 7), (12, 13, 14, 15)]
  wanted = [_device_rows(sidecar, logps, rows, num_trajectories) for rows in microbatches]
  with launch_budget.counting() as records:
    host_sidecar, host_logps = agentic_rl_learner._host_microbatch_views(  # pylint: disable=protected-access
        sidecar, logps, num_trajectories
    )
    got = []
    for rows in microbatches:
      index = np.asarray(rows, dtype=np.int32)
      pair = jax.tree.map(
          lambda value: (
              value[index]
              if hasattr(value, "shape") and value.shape and value.shape[0] == num_trajectories
              else value
          ),
          host_sidecar,
      )
      got.append((pair, host_logps[index]))
  assert not records, launch_budget.by_program(records).most_common()
  for (pair_want, t_want), (pair_got, t_got) in zip(wanted, got):
    want_leaves = jax.tree.leaves(pair_want)
    got_leaves = jax.tree.leaves(pair_got)
    assert len(want_leaves) == len(got_leaves)
    for a, b in zip(want_leaves, got_leaves):
      assert isinstance(b, np.ndarray) or np.ndim(b) == 0
      assert np.asarray(a).dtype == np.asarray(b).dtype
      assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    assert np.asarray(t_want).tobytes() == np.asarray(t_got).tobytes()
    assert alignment._hash(pair_want.s_decode) == alignment._hash(pair_got.s_decode)  # pylint: disable=protected-access
  # Leaves without a leading trajectory axis are untouched.
  assert int(host_sidecar.policy_version) == 7


def test_learner_slices_microbatch_rows_from_the_host_views():
  import inspect
  source = inspect.getsource(
      agentic_rl_learner.AgenticRLLearner._run_p28_g6_update  # pylint: disable=protected-access
  )
  assert "host_sidecar, host_logps = _host_microbatch_views(" in source
  assert "t_current=host_logps[row_index]" in source
  assert 'result["per_token_logps"][' not in source.split("_host_microbatch_views(")[1].split("records.append(record)")[0]
