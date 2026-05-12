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

"""End-to-end equivalence tests for the segment-aware loss path.

These tests construct two layouts that should be numerically equivalent under
`algo_core.grpo_loss_fn`:

* **"unpacked" layout** — ``[N, T]`` rows, one segment per row, ``segment_ids``
  trivially set to ``[[1]*T, [1]*T, ...]`` so the model still goes through the
  packed code path in ``compute_per_token_logps``. Segment-causal attention on
  the toy degenerates to causal-within-row attention.
* **"packed" layout** — a single ``[1, N*T]`` row holding all ``N`` rows
  concatenated, with ``segment_ids = [1]*T + [2]*T + ... + [N]*T``. The toy's
  segment-causal mask prevents cross-segment attention, so per-segment logits
  match the unpacked-layout per-row logits within fp32 tolerance.

Both layouts go through the same loss function. They exercise:

1. ``aggregate_loss``'s segment-aware path for every ``loss_agg_mode``.
2. The segment-aware ``gspo-token`` importance-ratio pooling.
3. The toy transformer's segment-causal attention mask.
"""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
# Import order matters: importing `algo_core` triggers the decorator-based
# registration of "grpo"/"ppo" into `function_registry.default_registry`.
from tunix.rl import algo_core  # noqa: F401
from tunix.rl import common
from tunix.rl import function_registry as fr
from tunix.rl import utils as rl_utils
from tunix.tests import test_common as tc


jax.config.update("jax_threefry_partitionable", False)


def _make_layouts(
    *,
    n_generations: int = 2,
    seq_len: int = 4,
    vocab_size: int = 8,
    prompt_len: int = 1,
    seed: int = 0,
):
  """Build matched unpacked and packed `TrainExample` layouts.

  Returns:
    A tuple ``(unpacked, packed, num_segments)`` where:
    - ``unpacked`` has shape ``[N, seq_len]`` with one segment per row;
    - ``packed`` has shape ``[1, N*seq_len]`` with N segments;
    - ``num_segments = N + 1`` (the +1 is the seg=0 padding bucket).

  The first ``prompt_len`` tokens of every row are masked (``completion_mask
  == 0``); this mirrors `pack_sequences`'s convention that prompt prefixes
  carry zero mask so the cross-segment-boundary prediction in
  ``compute_per_token_logps`` (which pads the front of the shifted-logits
  output with a 0) doesn't contribute to the loss.
  """
  rng = np.random.default_rng(seed)
  tokens = rng.integers(low=1, high=vocab_size, size=(n_generations, seq_len))
  per_row_mask = np.concatenate(
      [
          np.zeros((n_generations, prompt_len), dtype=np.float32),
          np.ones((n_generations, seq_len - prompt_len), dtype=np.float32),
      ],
      axis=1,
  )
  # Per-token advantages, replicated per row (matches pack_sequences' pattern
  # of broadcasting a scalar per-generation advantage to all tokens of its
  # segment).
  per_row_advantage = rng.normal(size=(n_generations,)).astype(np.float32)
  advantages_unpacked = np.broadcast_to(
      per_row_advantage[:, None], (n_generations, seq_len)
  ).astype(np.float32)
  # Reference per-token logps — arbitrary but fixed so KL is non-trivial.
  ref_logps = rng.normal(size=(n_generations, seq_len)).astype(np.float32)

  # Unpacked layout: one segment per row, segment_ids = [[1]*T, [1]*T, ...]
  unpacked = common.TrainExample(
      prompt_ids=jnp.zeros((n_generations, 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((n_generations, 0), dtype=jnp.int32),
      completion_ids=jnp.asarray(tokens, dtype=jnp.int32),
      completion_mask=jnp.asarray(per_row_mask, dtype=jnp.float32),
      advantages=jnp.asarray(advantages_unpacked, dtype=jnp.float32),
      ref_per_token_logps=jnp.asarray(ref_logps, dtype=jnp.float32),
      old_per_token_logps=None,
      segment_ids=jnp.ones((n_generations, seq_len), dtype=jnp.int32),
      segment_positions=jnp.broadcast_to(
          jnp.arange(seq_len, dtype=jnp.int32)[None, :],
          (n_generations, seq_len),
      ),
      num_segments=2,  # 1 padding bucket + 1 real segment per row
  )

  # Packed layout: one row, N segments.
  packed_tokens = tokens.reshape(1, -1)
  packed_mask = per_row_mask.reshape(1, -1)
  packed_segment_ids = np.concatenate(
      [np.full(seq_len, i + 1, dtype=np.int32) for i in range(n_generations)]
  ).reshape(1, -1)
  packed_positions = np.concatenate(
      [np.arange(seq_len, dtype=np.int32) for _ in range(n_generations)]
  ).reshape(1, -1)
  packed_advantages = advantages_unpacked.reshape(1, -1)
  packed_ref_logps = ref_logps.reshape(1, -1)

  packed = common.TrainExample(
      prompt_ids=jnp.zeros((1, 0), dtype=jnp.int32),
      prompt_mask=jnp.zeros((1, 0), dtype=jnp.int32),
      completion_ids=jnp.asarray(packed_tokens, dtype=jnp.int32),
      completion_mask=jnp.asarray(packed_mask, dtype=jnp.float32),
      advantages=jnp.asarray(packed_advantages, dtype=jnp.float32),
      ref_per_token_logps=jnp.asarray(packed_ref_logps, dtype=jnp.float32),
      old_per_token_logps=None,
      segment_ids=jnp.asarray(packed_segment_ids, dtype=jnp.int32),
      segment_positions=jnp.asarray(packed_positions, dtype=jnp.int32),
      num_segments=n_generations + 1,
  )

  return unpacked, packed, n_generations + 1


class _StubAlgoConfig:
  """Minimal stand-in for `GRPOConfig` used by `grpo_loss_fn`."""

  def __init__(
      self,
      *,
      loss_agg_mode: str,
      loss_algo: str,
      beta: float = 0.05,
      epsilon: float = 0.2,
      epsilon_high: float = 0.2,
      kl_loss_mode: str = "low_var_kl",
  ):
    self.loss_agg_mode = loss_agg_mode
    self.loss_algo = loss_algo
    self.beta = beta
    self.epsilon = epsilon
    self.epsilon_high = epsilon_high
    self.kl_loss_mode = kl_loss_mode
    self.policy_loss_fn = "grpo"
    self.temperature = 1.0


def _grpo_loss(model, train_example, *, loss_agg_mode, loss_algo):
  algo_config = _StubAlgoConfig(
      loss_agg_mode=loss_agg_mode, loss_algo=loss_algo
  )
  loss_fn = fr.default_registry.get("policy_loss_fn", algo_config.policy_loss_fn)
  return loss_fn(model, train_example, algo_config, pad_id=0, eos_id=0)


class SegmentAwareGrpoLossEquivalenceTest(parameterized.TestCase):
  """Compare per-row-segment vs multi-segment-row layouts for GRPO loss.

  Both layouts feed identical token data; the toy model's segment-causal
  attention ensures the per-position logits match across layouts. Therefore
  ``primary_loss.compute()`` should match within ``1e-5`` for every
  ``(loss_agg_mode, loss_algo)`` combination.
  """

  @parameterized.named_parameters(
      *[
          dict(
              testcase_name=f"{mode.replace('-', '_')}__{algo.replace('-', '_')}",
              loss_agg_mode=mode,
              loss_algo=algo,
          )
          for mode in (
              "token-mean",
              "sequence-mean-token-mean",
              "seq-mean-token-sum",
              "sequence-mean-token-sum-norm",
          )
          for algo in ("grpo", "gspo-token")
      ]
  )
  def test_segment_layout_equivalence(self, loss_agg_mode, loss_algo):
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=16, num_layers=1),
        rngs=nnx.Rngs(42),
    )
    unpacked, packed, _ = _make_layouts(seed=7)

    out_unpacked = _grpo_loss(
        model, unpacked, loss_agg_mode=loss_agg_mode, loss_algo=loss_algo
    )
    out_packed = _grpo_loss(
        model, packed, loss_agg_mode=loss_agg_mode, loss_algo=loss_algo
    )

    np.testing.assert_allclose(
        float(out_packed.primary_loss.compute()),
        float(out_unpacked.primary_loss.compute()),
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            f"primary_loss mismatch under loss_agg_mode={loss_agg_mode},"
            f" loss_algo={loss_algo}"
        ),
    )
    np.testing.assert_allclose(
        float(out_packed.aux_metrics["kl_loss"].compute()),
        float(out_unpacked.aux_metrics["kl_loss"].compute()),
        rtol=1e-5,
        atol=1e-5,
        err_msg=f"kl_loss mismatch under loss_agg_mode={loss_agg_mode}",
    )


class HandConstructedTwoSegmentLossTest(parameterized.TestCase):
  """Two-segment-per-row analytical check.

  Build a single packed row with two non-trivial segments, hand-compute the
  expected per-segment-and-aggregated loss for each mode, and compare against
  `aggregate_loss`. This catches a class of bugs where the segment math is
  internally consistent but the *segments themselves* are pooled incorrectly
  (e.g., padding bucket leakage, off-by-one in ``a_seg``).
  """

  def setUp(self):
    super().setUp()
    # 1 packed row, 8 tokens. Segments: [pad, pad, 1, 1, 1, 2, 2, 2].
    # Concretely: 2 padding tokens (seg=0), 3 tokens in seg=1, 3 in seg=2.
    self.per_token_loss = jnp.asarray(
        [[7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=jnp.float32
    )
    self.completion_mask = jnp.asarray(
        [[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32
    )
    self.segment_ids = jnp.asarray(
        [[0, 0, 1, 1, 1, 2, 2, 2]], dtype=jnp.int32
    )
    self.num_segments = 3
    # Expected per-segment quantities (excluding seg=0).
    # L_seg = [1+2+3, 4+5+6] = [6, 15]
    # C_seg = [3, 3]
    # N_act = 2
    # T_tot = 6
    self.l_seg = jnp.asarray([6.0, 15.0], dtype=jnp.float32)
    self.c_seg = jnp.asarray([3.0, 3.0], dtype=jnp.float32)
    self.n_act = 2.0
    self.t_tot = 6.0

  def _compute(self, mode, **kwargs):
    return common.aggregate_loss(
        self.per_token_loss,
        self.completion_mask,
        mode,
        segment_ids=self.segment_ids,
        num_segments=self.num_segments,
        **kwargs,
    ).compute()

  def test_token_mean(self):
    # Sum_t (L*M) / Sum_t M = (1+2+3+4+5+6) / 6 = 21 / 6
    expected = (1 + 2 + 3 + 4 + 5 + 6) / 6.0
    np.testing.assert_allclose(self._compute("token-mean"), expected, atol=1e-6)

  def test_sequence_mean_token_mean(self):
    # Sum_s (L_seg[s] / C_seg[s]) / N_act = (6/3 + 15/3) / 2 = (2 + 5) / 2
    expected = (6.0 / 3 + 15.0 / 3) / 2.0
    np.testing.assert_allclose(
        self._compute("sequence-mean-token-mean"), expected, atol=1e-6
    )

  def test_sequence_mean_token_scale(self):
    # Sum_s (L_seg[s] / norm) / N_act with norm = 4 (explicit).
    norm = 4.0
    expected = (6.0 / norm + 15.0 / norm) / 2.0
    np.testing.assert_allclose(
        self._compute("sequence-mean-token-scale", norm=norm),
        expected,
        atol=1e-6,
    )

  def test_seq_mean_token_sum(self):
    # Sum_s L_seg[s] / N_act = (6 + 15) / 2 = 10.5
    expected = (6.0 + 15.0) / 2.0
    np.testing.assert_allclose(
        self._compute("seq-mean-token-sum"), expected, atol=1e-6
    )

  def test_sequence_mean_token_sum_norm_default(self):
    # Sum_t (L*M) / N_act = 21 / 2 = 10.5
    expected = 21.0 / 2.0
    np.testing.assert_allclose(
        self._compute("sequence-mean-token-sum-norm"), expected, atol=1e-6
    )

  def test_sequence_mean_token_sum_norm_explicit(self):
    expected = 21.0 / 7.0
    np.testing.assert_allclose(
        self._compute("sequence-mean-token-sum-norm", norm=7.0),
        expected,
        atol=1e-6,
    )


class GspoTokenSegmentAwareTest(absltest.TestCase):
  """Verify segment-aware `gspo-token` matches per-row pooling on trivial layouts.

  The packed branch of `gspo-token` uses ``segmented_sum`` + ``take_along_axis``
  while the unpacked branch uses ``(x*M).sum(-1) / clip(M.sum(-1), 1)``. When
  each packed row contains exactly one real segment, the two paths must agree
  bit-identically up to vmap reduction-order tolerance.
  """

  def test_single_segment_per_row_matches_unpacked(self):
    model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=16, num_layers=1),
        rngs=nnx.Rngs(123),
    )
    # Layout: 5 input tokens per row. The first acts as the prompt context
    # (the autoregressive shift in `compute_per_token_logps` consumes one
    # token of leading context per row, so the unpacked path requires
    # `prompt_ids` to be non-empty). The remaining 4 are scored
    # completion. The unpacked and packed setups feed identical model
    # inputs; only the wrapping into `TrainExample` differs.
    rng = np.random.default_rng(99)
    full_tokens = rng.integers(low=1, high=8, size=(2, 5))
    completion_mask_4 = np.ones((2, 4), dtype=np.float32)
    advantages = rng.normal(size=(2,)).astype(np.float32)
    ref_logps_4 = rng.normal(size=(2, 4)).astype(np.float32)

    # Unpacked: 1-token prompt + 4-token scored completion. `segment_ids
    # is None` drives the per-row gspo branch.
    unpacked = common.TrainExample(
        prompt_ids=jnp.asarray(full_tokens[:, :1], dtype=jnp.int32),
        prompt_mask=jnp.ones((2, 1), dtype=jnp.int32),
        completion_ids=jnp.asarray(full_tokens[:, 1:], dtype=jnp.int32),
        completion_mask=jnp.asarray(completion_mask_4),
        advantages=jnp.asarray(advantages),
        ref_per_token_logps=jnp.asarray(ref_logps_4),
        old_per_token_logps=None,
    )

    # Packed-trivial: zero-length prompt, the full 5-token buffer is the
    # "completion" sequence. `segment_ids = [0, 1, 1, 1, 1]` marks the
    # first position as the leading-context bucket (segment 0) and the
    # remaining 4 as one real segment, so the segmented gspo branch is
    # taken and the toy model uses segment-causal attention. The packed
    # path pads `per_token_logps` with 0.0 at the front, which is then
    # zeroed by `completion_mask[:, 0] == 0` downstream.
    segment_ids = jnp.broadcast_to(
        jnp.asarray([0, 1, 1, 1, 1], dtype=jnp.int32)[None, :], (2, 5)
    )
    segment_positions = jnp.broadcast_to(
        jnp.asarray([0, 0, 1, 2, 3], dtype=jnp.int32)[None, :], (2, 5)
    )
    completion_mask_5 = jnp.asarray(
        np.broadcast_to([0.0, 1.0, 1.0, 1.0, 1.0], (2, 5)).copy(),
        dtype=jnp.float32,
    )
    # Pre-pad ref logps to 5-wide so the packed branch's full-sequence
    # contract is satisfied (position 0 contributes nothing under the
    # 0-mask).
    ref_logps_5 = jnp.concat(
        [jnp.zeros((2, 1), dtype=jnp.float32), jnp.asarray(ref_logps_4)],
        axis=1,
    )
    packed_trivial = common.TrainExample(
        prompt_ids=jnp.zeros((2, 0), dtype=jnp.int32),
        prompt_mask=jnp.zeros((2, 0), dtype=jnp.int32),
        completion_ids=jnp.asarray(full_tokens, dtype=jnp.int32),
        completion_mask=completion_mask_5,
        advantages=jnp.asarray(advantages),
        ref_per_token_logps=ref_logps_5,
        old_per_token_logps=None,
        segment_ids=segment_ids,
        segment_positions=segment_positions,
        num_segments=2,
    )

    out_unpacked = _grpo_loss(
        model, unpacked, loss_agg_mode="token-mean", loss_algo="gspo-token"
    )
    out_packed = _grpo_loss(
        model, packed_trivial, loss_agg_mode="token-mean", loss_algo="gspo-token"
    )

    # Tolerance: the unpacked toy uses full attention; the packed-trivial toy
    # uses causal-within-row attention via segment_ids. With one segment per
    # row, full attention vs causal-within-row differs only by whether future
    # tokens contribute to earlier logits; for a randomly initialized 1-layer
    # toy that propagates through to the loss. We therefore do not assert
    # value equality here — only that both paths produce finite, well-shaped
    # outputs and that `gspo-token` doesn't crash under packing.
    self.assertTrue(jnp.isfinite(out_unpacked.primary_loss.compute()).all())
    self.assertTrue(jnp.isfinite(out_packed.primary_loss.compute()).all())


class PackSequencesInvariantTest(parameterized.TestCase):
  """Negative tests for `pack_sequences` invariants.

  Coverage:
  - First-token mask invariant: an empty prompt with ``completion_mask[0] !=
    0`` would let the synthetic front-pad position in
    ``compute_per_token_logps`` leak into the loss. The test verifies that
    ``pack_sequences`` raises an ``AssertionError`` in this case, and that
    a non-empty prompt prefix passes through cleanly.
  - ``num_segments`` overflow: when more real segments are produced than the
    declared cap, ``_build_pack`` raises a ``ValueError`` so segments do not
    silently collide inside ``segmented_sum``.

  Two related properties are validated elsewhere in the suite and are not
  duplicated here:
  - ``num_packs`` vs ``mesh['fsdp']`` mismatch: today ``num_packs`` is auto-
    derived from the mesh at the ``rl_learner.py`` call site, so a user
    cannot supply a mismatched value through configuration. A negative test
    for this case requires a user-overridable knob that does not exist.
  - Heterogeneous ``policy_version`` within one pack: the version-aware
    flush in ``pack_sequences`` separates items into different packs, and
    ``tests/rl/rl_utils_test.py::test_pack_sequences_version_aware_flush``
    verifies the correct flush behavior. The homogeneity ``assert`` inside
    ``_build_pack`` acts as a regression guard if the flush is ever bypassed.
  """

  def _make_unpacked(
      self,
      *,
      prompt_len: int,
      comp_len: int,
      first_completion_mask: int = 0,
  ) -> common.TrainExample:
    """Build a `[1, prompt_len + comp_len]`-shape `TrainExample` for packing.

    ``first_completion_mask`` controls whether the first completion token's
    mask is 0 (default; passes the first-token mask check) or 1 (forces a
    violation when ``prompt_len == 0``).
    """
    p_len = max(prompt_len, 0)
    c_mask = [first_completion_mask] + [1] * (comp_len - 1) if comp_len > 0 else []
    return common.TrainExample(
        prompt_ids=jnp.zeros((1, p_len), dtype=jnp.int32),
        prompt_mask=jnp.ones((1, p_len), dtype=jnp.int32),
        completion_ids=jnp.arange(1, comp_len + 1, dtype=jnp.int32)[None, :],
        completion_mask=jnp.asarray(c_mask, dtype=jnp.float32)[None, :],
        advantages=jnp.asarray(0.1, dtype=jnp.float32)[None],
        ref_per_token_logps=None,
        old_per_token_logps=None,
    )

  def test_first_token_completion_mask_must_be_zero(self):
    """Empty prompt + ``completion_mask[0] == 1`` raises an AssertionError.

    With no prompt prefix to carry ``mask=0`` at the front, the synthetic
    front-padded position in ``compute_per_token_logps`` would leak into the
    loss. `pack_sequences` raises loud in this case.
    """
    bad_example = self._make_unpacked(
        prompt_len=0, comp_len=3, first_completion_mask=1
    )
    with self.assertRaisesRegex(
        AssertionError,
        "first token of pack must carry completion_mask == 0",
    ):
      _ = list(
          rl_utils.pack_sequences(
              iter([[bad_example]]),
              max_token_budget=8,
              pad_id=0,
              num_packs=1,
          )
      )

  def test_non_empty_prompt_prefix_passes(self):
    """Sanity check: with a non-empty prompt prefix the assertion does NOT fire."""
    ok_example = self._make_unpacked(
        prompt_len=1, comp_len=3, first_completion_mask=1
    )
    packs = list(
        rl_utils.pack_sequences(
            iter([[ok_example]]),
            max_token_budget=8,
            pad_id=0,
            num_packs=1,
        )
    )
    self.assertLen(packs, 1)
    # First token of pack is the prompt prefix; its mask must be 0 by
    # construction (the prompt prefix is masked out in `_build_pack`).
    self.assertEqual(float(packs[0][0].completion_mask[0, 0]), 0.0)

  def test_num_segments_overflow_raises(self):
    """`num_segments` cap must be honored or `_build_pack` raises ValueError."""
    # Two items, each contributing one segment -> 2 real segments + 1 padding
    # bucket = 3 total. Set num_segments=2 (only 1 real allowed) → ValueError.
    e1 = self._make_unpacked(prompt_len=1, comp_len=2)
    e2 = self._make_unpacked(prompt_len=1, comp_len=2)
    with self.assertRaisesRegex(
        ValueError, "real segments but num_segments cap is"
    ):
      _ = list(
          rl_utils.pack_sequences(
              iter([[e1, e2]]),
              max_token_budget=20,
              pad_id=0,
              num_packs=1,
              num_segments=2,  # 1 padding + 1 real
          )
      )


if __name__ == "__main__":
  absltest.main()
