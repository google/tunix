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

"""Tests for the router-replay rollout/trainer contract."""

import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.common import router_replay

_MISSING = router_replay.MISSING_ROUTE
_PADDING = router_replay.PADDING_ROUTE
_LAYERS, _TOP_K = 2, 2
# A 3-token prompt and 2 generated tokens means 4 routed forwards: the last
# generated token is never fed back in.
_PROMPT_LEN, _COMPLETION_LEN = 3, 2


def _routes(length, fill):
  """`[length, num_layers, top_k]` tagged by `fill`.

  Real ids are SPREAD across the top-k axis rather than repeated. A real row
  comes from `top_k`, which cannot select the same expert twice, so a repeated
  row is indistinguishable from an unwritten slot and `full_sequence` now
  demotes it to `MISSING_ROUTE`. Multiplying by `_TOP_K` keeps the id blocks for
  different fills disjoint, so the placement assertions stay exact.

  Sentinel fills are left constant, because a sentinel row is all -1 or all -2
  by contract.
  """
  if fill < 0:
    return np.full((length, _LAYERS, _TOP_K), fill, dtype=np.int32)
  row = fill * _TOP_K + np.arange(_TOP_K, dtype=np.int32)
  return np.broadcast_to(row, (length, _LAYERS, _TOP_K)).copy()


def _full(prompt_routes, completion_routes):
  return router_replay.full_sequence(prompt_routes, completion_routes, _PROMPT_LEN, _COMPLETION_LEN)


class FullSequenceTest(parameterized.TestCase):
  """Routes must land on the tokens they were captured for, or not at all."""

  def test_prompt_and_completion_land_in_input_order(self):
    full = _full(_routes(_PROMPT_LEN, 7), _routes(_COMPLETION_LEN - 1, 9))
    np.testing.assert_array_equal(full[:_PROMPT_LEN], _routes(_PROMPT_LEN, 7))
    np.testing.assert_array_equal(full[_PROMPT_LEN], _routes(1, 9)[0])
    # The final generated token is consumed by no forward, so it has no route.
    np.testing.assert_array_equal(full[-1], _routes(1, _MISSING)[0])
    self.assertEqual(full.dtype, router_replay.ROUTE_DTYPE)

  def test_completion_only_leaves_the_prompt_unrouted(self):
    """A decode-only capture is G-1 rows and starts at the prompt boundary."""
    full = _full(None, _routes(_COMPLETION_LEN - 1, 9))
    np.testing.assert_array_equal(full[:_PROMPT_LEN], _routes(_PROMPT_LEN, _MISSING))
    np.testing.assert_array_equal(full[_PROMPT_LEN], _routes(1, 9)[0])

  def test_vllm_full_sequence_capture_lands_in_input_order(self):
    """What vLLM actually returns: one array spanning prompt + completion.

    `CompletionOutput.routed_experts` is `[seq_len, layers, top_k]` -- the
    output processor concatenates the prefill chunk with the decode chunks --
    and there is no separate prompt array. A healthy capture is P+G-1 rows,
    because the final generated token is never fed into a forward.
    """
    total = _PROMPT_LEN + _COMPLETION_LEN
    captured = np.concatenate(
        [_routes(_PROMPT_LEN, 7), _routes(_COMPLETION_LEN - 1, 9)]
    )
    self.assertLen(captured, total - 1)
    full = _full(None, captured)
    np.testing.assert_array_equal(full[:_PROMPT_LEN], _routes(_PROMPT_LEN, 7))
    np.testing.assert_array_equal(
        full[_PROMPT_LEN : total - 1], _routes(_COMPLETION_LEN - 1, 9)
    )
    # Only the final generated token is unrouted.
    np.testing.assert_array_equal(full[-1], _routes(1, _MISSING)[0])

  def test_full_capture_is_not_shifted_by_one(self):
    """Regression: right-aligning at P+G moves every route one token late.

    Reproduces the shape that failed on the cluster (P=118, G=512, n=629) at
    small scale. The first prompt token must carry the first captured route.
    """
    total = _PROMPT_LEN + _COMPLETION_LEN
    captured = np.stack([_routes(1, i)[0] for i in range(total - 1)])
    full = _full(None, captured)
    np.testing.assert_array_equal(full[0], captured[0])
    np.testing.assert_array_equal(full[total - 2], captured[-1])
    np.testing.assert_array_equal(full[total - 1], _routes(1, _MISSING)[0])

  def test_prefix_cache_hit_tail_aligns(self):
    """A cached prefix of C tokens shortens the capture from the front."""
    total = _PROMPT_LEN + _COMPLETION_LEN
    cached = 1
    captured = np.concatenate([
        _routes(_PROMPT_LEN - cached, 7),
        _routes(_COMPLETION_LEN - 1, 9),
    ])
    full = _full(None, captured)
    np.testing.assert_array_equal(full[:cached], _routes(cached, _MISSING))
    np.testing.assert_array_equal(
        full[cached:_PROMPT_LEN], _routes(_PROMPT_LEN - cached, 7)
    )
    np.testing.assert_array_equal(
        full[_PROMPT_LEN : total - 1], _routes(_COMPLETION_LEN - 1, 9)
    )

  def test_prompt_only_leaves_the_completion_unrouted(self):
    full = _full(_routes(_PROMPT_LEN, 7), None)
    np.testing.assert_array_equal(full[:_PROMPT_LEN], _routes(_PROMPT_LEN, 7))
    np.testing.assert_array_equal(full[_PROMPT_LEN:], _routes(_COMPLETION_LEN, _MISSING))

  def test_nothing_captured_is_none(self):
    self.assertIsNone(_full(None, None))

  def test_short_prompt_capture_tail_aligns(self):
    """A prefix-cache hit drops a prefix, so what remains is the prompt tail."""
    full = _full(_routes(_PROMPT_LEN - 1, 7), _routes(1, 9))
    # The uncaptured head is the model's own problem, not padding.
    np.testing.assert_array_equal(full[0], _routes(1, _MISSING)[0])
    np.testing.assert_array_equal(full[1:_PROMPT_LEN], _routes(_PROMPT_LEN - 1, 7))
    np.testing.assert_array_equal(full[_PROMPT_LEN], _routes(1, 9)[0])

  @parameterized.named_parameters(
      # Guessing an alignment here would shift every expert by one token.
      ("long_prompt", _routes(_PROMPT_LEN + 1, 7), _routes(1, 9)),
      # Longer than prompt+completion cannot belong to this request at all.
      # (Longer than the completion alone is legitimate and normal -- see the
      # full-sequence tests above.)
      (
          "longer_than_the_whole_sequence",
          None,
          _routes(_PROMPT_LEN + _COMPLETION_LEN + 1, 9),
      ),
      ("trailing_shape_disagrees", _routes(_PROMPT_LEN, 7)[:, :1], _routes(1, 9)),
      ("not_three_dimensional", None, _routes(1, 9)[:, :, 0]),
  )
  def test_unplaceable_capture_raises(self, prompt_routes, completion_routes):
    with self.assertRaises(ValueError):
      _full(prompt_routes, completion_routes)


class ValidateTest(parameterized.TestCase):
  """A row is all real, all -1, or all -2. Anything else has no meaning."""

  def test_accepts_each_kind_of_row(self):
    router_replay.validate(np.array([[[0, 1]], [[_MISSING] * 2], [[_PADDING] * 2]]), num_experts=2)

  @parameterized.named_parameters(
      ("row_mixes_sentinel_and_expert", np.array([[[1, _MISSING]]]), None),
      ("row_mixes_both_sentinels", np.array([[[_MISSING, _PADDING]]]), None),
      ("row_repeats_an_expert", np.array([[[2, 2]]]), None),
      ("expert_id_out_of_range", np.array([[[0, 5]]]), 4),
      ("float_dtype", np.zeros((1, 1, 2)), None),
      ("no_top_k_axis", np.array([1, 2]), None),
      ("empty_top_k_axis", np.zeros((1, 0), dtype=np.int32), None),
  )
  def test_rejects(self, routed_experts, num_experts):
    with self.assertRaises(ValueError):
      router_replay.validate(routed_experts, num_experts=num_experts)

  def test_top_k_of_one_cannot_repeat(self):
    """K=1 has no pair to compare; the duplicate check must not misfire."""
    router_replay.validate(np.array([[[3]], [[3]]]))


class RequireMaxtextSupportTest(parameterized.TestCase):
  """An older MaxText reads -1 as 'route nowhere'; that must not run."""

  def _install_fake_maxtext(self, moe_module):
    """Puts `moe_module` behind `from maxtext.layers import moe`."""
    maxtext = types.ModuleType("maxtext")
    layers = types.ModuleType("maxtext.layers")
    maxtext.layers = layers
    layers.moe = moe_module
    for name, module in (
        ("maxtext", maxtext),
        ("maxtext.layers", layers),
        ("maxtext.layers.moe", moe_module),
    ):
      self.enterContext(mock.patch.dict(sys.modules, {name: module}))

  def _moe(self, **attrs):
    moe = types.ModuleType("maxtext.layers.moe")
    for name, value in attrs.items():
      setattr(moe, name, value)
    return moe

  def test_accepts_a_maxtext_that_agrees(self):
    self._install_fake_maxtext(
        self._moe(
            ROUTER_REPLAY_MISSING=router_replay.MISSING_ROUTE,
            ROUTER_REPLAY_PADDING=router_replay.PADDING_ROUTE,
        )
    )
    router_replay.require_maxtext_support()  # Must not raise.

  def test_rejects_a_maxtext_predating_the_protocol(self):
    """No sentinels at all: -1 would silently zero the MoE block."""
    self._install_fake_maxtext(self._moe())
    with self.assertRaisesRegex(RuntimeError, "ROUTER_REPLAY_MISSING"):
      router_replay.require_maxtext_support()

  def test_rejects_a_maxtext_that_disagrees_on_a_value(self):
    self._install_fake_maxtext(
        self._moe(
            ROUTER_REPLAY_MISSING=router_replay.MISSING_ROUTE,
            ROUTER_REPLAY_PADDING=-7,
        )
    )
    with self.assertRaisesRegex(RuntimeError, "ROUTER_REPLAY_PADDING is -7"):
      router_replay.require_maxtext_support()

  def test_rejects_a_missing_maxtext(self):
    self.enterContext(
        mock.patch.dict(sys.modules, {"maxtext.layers": None})
    )
    with self.assertRaisesRegex(RuntimeError, "maxtext.layers.moe"):
      router_replay.require_maxtext_support()


if __name__ == "__main__":
  absltest.main()
