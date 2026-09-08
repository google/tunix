"""The stream path's batch loss output and scale run as ONE program.

Eagerly the loss construction issued about 300 tiny launches per update
(once for the scale at the first group, once for the loss output at the
end).  The scale is exact under the program (a mask count and one
division) and gates the committed gradient; the loss value is reporting
only.  The eager batch-loss pullback oracle is untouched.
"""

import importlib.util
import os
import pathlib
import types

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

from tunix.rl import algo_core
from tunix.rl import common
from tunix.sft import utils as sft_utils

_ALGO = types.SimpleNamespace(
    beta=0.0,
    epsilon=0.2,
    epsilon_high=0.2,
    epsilon_c=None,
    loss_algo="grpo",
    loss_agg_mode="sequence-mean-token-mean",
    temperature=1.0,
    kl_loss_mode="k1",
    kl_clamp_value=None,
)


def _adapter():
  if len(jax.devices()) < 16:
    pytest.skip("requires sixteen forced CPU devices")
  path = pathlib.Path(__file__).with_name("canonical_qwen3_adapter_test.py")
  spec = importlib.util.spec_from_file_location("stream_loss_fixtures", path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  case = module.CanonicalQwen3AdapterTest(
      "test_p32_dp16_group_spec_preserves_rank_local_order"
  )
  adapter, _ = case._make_p32_group_adapter()  # pylint: disable=protected-access
  return adapter


def _inputs(groups=2, data_size=16, width=6):
  rng = np.random.default_rng(13)
  rows = groups * data_size
  completion_ids = rng.integers(1, 9, (rows, width), dtype=np.int32)
  completion_ids[:, 4:] = 0
  completion_ids[3, 2:] = 0
  example = common.TrainExample(
      prompt_ids=jnp.asarray(rng.integers(1, 9, (rows, 3), dtype=np.int32)),
      prompt_mask=jnp.ones((rows, 3), bool),
      completion_ids=jnp.asarray(completion_ids),
      completion_mask=jnp.asarray(completion_ids != 0),
      old_per_token_logps=jnp.asarray(
          -0.5 * np.abs(rng.standard_normal((rows, width))).astype(np.float32)
      ),
      ref_per_token_logps=None,
      advantages=jnp.asarray(np.linspace(-1.0, 1.0, rows, dtype=np.float32)),
      sampler_is_weights=None,
      segment_ids=None,
  )
  stream_logps = jnp.asarray(
      -0.4 * np.abs(rng.standard_normal((groups, data_size, width))).astype(np.float32)
  )
  stream_entropy = jnp.asarray(
      np.abs(rng.standard_normal((groups, data_size, width))).astype(np.float32)
  )
  return example, stream_logps, stream_entropy


def test_stream_loss_program_matches_the_eager_loss_and_is_one_launch():
  adapter = _adapter()
  example, stream_logps, stream_entropy = _inputs()
  fn = adapter._p32_stream_loss_fn(_ALGO)  # pylint: disable=protected-access
  assert adapter._p32_stream_loss_fn(_ALGO) is fn  # pylint: disable=protected-access
  fn(stream_logps, stream_entropy, example)  # compile
  with launch_budget.counting() as records:
    output, scale = fn(stream_logps, stream_entropy, example)
    jax.block_until_ready((output, scale))
  assert len(records) == 1, launch_budget.by_program(records).most_common()
  assert isinstance(output, sft_utils.LossOutput)
  eager = algo_core.grpo_loss_from_precomputed_logps(
      adapter._ungroup_batch_rows(stream_logps),  # pylint: disable=protected-access
      adapter._ungroup_batch_rows(stream_entropy),  # pylint: disable=protected-access
      example,
      _ALGO,
  )
  eager_scale = eager.primary_loss.compute_scale()
  # The scale (a mask count and one division) is exact: byte-identical.
  assert np.asarray(scale).tobytes() == np.asarray(eager_scale).tobytes()
  assert np.asarray(output.primary_loss.denominator).tobytes() == np.asarray(
      eager.primary_loss.denominator
  ).tobytes()
  assert np.asarray(output.primary_loss.compute_scale()).tobytes() == np.asarray(
      eager_scale
  ).tobytes()
  # The loss value is reporting only; the program may fuse its reductions.
  np.testing.assert_allclose(
      np.asarray(output.primary_loss.unreduced_sum),
      np.asarray(eager.primary_loss.unreduced_sum),
      rtol=1e-6,
  )
  assert set(output.aux_metrics) == set(eager.aux_metrics)
  # A different algo config object rebuilds the program.
  other = types.SimpleNamespace(**vars(_ALGO))
  assert adapter._p32_stream_loss_fn(other) is not fn  # pylint: disable=protected-access
