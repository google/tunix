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

"""The MLPerf GRPO recipe knobs have to survive the trip to `grpo_loss_fn`.

`algo_core.grpo_loss_fn` reads every sampler-vs-trainer option off `algo_config`
with `getattr(..., default)`, and every sampler-vs-trainer input off
`train_example` the same way. That is deliberately forgiving, which means a knob
that never reaches either object does not raise -- the feature simply does not
happen, for the whole run, silently. That exact failure has already cost this
project once: `trainable_parameters_mask` was read only by a code path the RL
trainer never called, so a recipe that meant to freeze the MoE router trained it
instead and nothing said so.

These tests pin the two carriers:

  * `GRPOAdapter` -> `algo_config`, for the options.
  * `RLTrainerPayload` / `batch_assembly` -> `train_example`, for the per-token
    rollout log-probabilities and the per-sequence overlong verdict.

They assert plumbing, not numerics. The numerics of the gates themselves live in
`tests/rl/algo_core_test.py`.
"""

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter


def _mlperf_adapter(**overrides):
  """A GRPOAdapter configured the way the MLPerf recipe asks for."""
  kwargs = dict(
      group_size=4,
      overlong_loss_masking=True,
      seq_logprob_error_threshold=2.0,
      truncated_importance_sampling_type="seq-mask-tis",
      truncated_importance_sampling_ratio_min=0.999,
      truncated_importance_sampling_ratio=1.002,
  )
  kwargs.update(overrides)
  return algorithm_adapter.GRPOAdapter(**kwargs)


class AlgoConfigCarriesRecipeOptionsTest(absltest.TestCase):
  """GRPOAdapter -> the `algo_config` namespace `grpo_loss_fn` reads."""

  def _algo_config(self, adapter):
    """The namespace the adapter bakes into its model-input function."""
    fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
    # build_gen_model_input_fn returns functools.partial(_algo_model_input,
    # algo_config=..., ...); the namespace is the whole point of this test, so
    # read it back off the partial rather than reconstructing it.
    return fn.keywords["algo_config"]

  def test_recipe_options_reach_algo_config(self):
    cfg = self._algo_config(_mlperf_adapter())
    self.assertTrue(cfg.overlong_loss_masking)
    self.assertEqual(cfg.seq_logprob_error_threshold, 2.0)
    self.assertEqual(cfg.truncated_importance_sampling_type, "seq-mask-tis")
    self.assertEqual(cfg.truncated_importance_sampling_ratio_min, 0.999)
    self.assertEqual(cfg.truncated_importance_sampling_ratio, 1.002)

  def test_defaults_are_off_so_existing_runs_are_unchanged(self):
    cfg = self._algo_config(algorithm_adapter.GRPOAdapter(group_size=4))
    self.assertFalse(cfg.overlong_loss_masking)
    self.assertIsNone(cfg.seq_logprob_error_threshold)
    self.assertIsNone(cfg.truncated_importance_sampling_type)
    self.assertIsNone(cfg.truncated_importance_sampling_ratio_min)
    self.assertIsNone(cfg.truncated_importance_sampling_ratio)

  def test_preexisting_options_still_reach_algo_config(self):
    """The additions must not displace what was already carried."""
    adapter = algorithm_adapter.GRPOAdapter(
        group_size=4,
        clip_epsilon=0.2,
        epsilon_high=0.28,
        beta_kl=0.0,
        temperature=1.0,
        loss_agg_mode="token-mean",
    )
    cfg = self._algo_config(adapter)
    self.assertEqual(cfg.epsilon, 0.2)
    self.assertEqual(cfg.epsilon_high, 0.28)
    self.assertEqual(cfg.beta, 0.0)
    self.assertEqual(cfg.temperature, 1.0)
    self.assertEqual(cfg.loss_agg_mode, "token-mean")

  def test_epsilon_high_defaults_to_epsilon(self):
    cfg = self._algo_config(
        algorithm_adapter.GRPOAdapter(group_size=4, clip_epsilon=0.3)
    )
    self.assertEqual(cfg.epsilon_high, 0.3)

  def test_report_bands_and_length_buckets_reach_algo_config(self):
    adapter = _mlperf_adapter(
        sampler_is_report_bands=((0.999, 1.002), (0.99, 1.01)),
        sampler_is_length_buckets=(512, 2048),
    )
    cfg = self._algo_config(adapter)
    self.assertEqual(
        cfg.sampler_is_report_bands, ((0.999, 1.002), (0.99, 1.01))
    )
    self.assertEqual(cfg.sampler_is_length_buckets, (512, 2048))


class ForceOnPolicyComposesWithTisTest(absltest.TestCase):
  """force_on_policy_ratio and seq-mask-tis must not be mutually exclusive.

  The MLPerf recipe sets BOTH. In this stack there is no `force_on_policy_ratio`
  flag; the equivalent is `use_rollout_logps=False`, which leaves
  `old_per_token_logps` unset so the trainer recomputes and the PPO ratio pins to
  1. That flag must not also suppress `rollout_per_token_logps`, which is what
  the TIS gates compare against -- otherwise the recipe is inexpressible.
  """

  def test_tis_with_force_on_policy_is_allowed(self):
    adapter = _mlperf_adapter(use_rollout_logps=False)
    self.assertFalse(adapter.use_rollout_logps)
    self.assertEqual(
        adapter.truncated_importance_sampling_type, "seq-mask-tis"
    )
    self.assertTrue(adapter.requires_rollout_logps)

  def test_requires_rollout_logps_is_false_when_no_gate_is_on(self):
    adapter = algorithm_adapter.GRPOAdapter(group_size=4)
    self.assertFalse(adapter.requires_rollout_logps)

  def test_either_gate_alone_sets_requires_rollout_logps(self):
    self.assertTrue(
        algorithm_adapter.GRPOAdapter(
            group_size=4, seq_logprob_error_threshold=2.0
        ).requires_rollout_logps
    )
    self.assertTrue(
        algorithm_adapter.GRPOAdapter(
            group_size=4,
            truncated_importance_sampling_type="seq-mask-tis",
            truncated_importance_sampling_ratio_min=0.999,
            truncated_importance_sampling_ratio=1.002,
        ).requires_rollout_logps
    )

  def test_overlong_masking_alone_does_not_require_rollout_logps(self):
    """Overlong masking reads the collector's verdict, not the sampler."""
    adapter = algorithm_adapter.GRPOAdapter(
        group_size=4, overlong_loss_masking=True, use_rollout_logps=False
    )
    self.assertTrue(adapter.overlong_loss_masking)
    self.assertFalse(adapter.requires_rollout_logps)

  def test_rollout_logps_are_carried_even_with_force_on_policy(self):
    """The payload builder must split the two uses of the sampler logprobs."""
    src = algorithm_adapter.__file__
    with open(src, "r") as fh:
      text = fh.read()
    # rollout_lp is extracted unconditionally; only old_lp is gated.
    self.assertIn("rollout_lp = _extract_old_logps(item, len(c_arr))", text)
    self.assertIn(
        "old_lp = rollout_lp if self.use_rollout_logps else None", text
    )
    self.assertIn("rollout_per_token_logps=rollout_lp", text)


class PayloadCarriesRolloutInputsTest(absltest.TestCase):
  """RLTrainerPayload -> the `train_example` `grpo_loss_fn` reads."""

  def _payload(self, n_tokens=4, **overrides):
    kwargs = dict(
        prompt_ids=np.zeros((2,), np.int32),
        prompt_mask=np.ones((2,), np.int32),
        completion_ids=np.zeros((n_tokens,), np.int32),
        completion_mask=np.ones((n_tokens,), np.int32),
        advantages=np.zeros((1,), np.float32),
    )
    kwargs.update(overrides)
    return datatypes.RLTrainerPayload(**kwargs)

  def test_fields_exist_and_default_to_none(self):
    payload = self._payload()
    self.assertIsNone(payload.rollout_per_token_logps)
    self.assertIsNone(payload.overlong)

  def test_fields_round_trip(self):
    logps = np.array([-0.1, -0.2, -0.3, -0.4], np.float32)
    payload = self._payload(rollout_per_token_logps=logps, overlong=1.0)
    np.testing.assert_allclose(payload.rollout_per_token_logps, logps)
    self.assertEqual(payload.overlong, 1.0)

  def test_packer_actually_returns_padded_rollout_logps(self):
    """Run the real packer. Registering the name is not enough.

    `optional_fields` only makes `_pack_chunk` STACK a field; the payload it
    returns has to be handed the array too. The first version of this port
    registered the name and omitted the kwarg, so the padded rows were computed
    and dropped -- and a test that grepped `optional_fields` passed anyway.
    """
    from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top

    max_prompt, max_resp = 4, 6
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=max_prompt,
        max_response_length=max_resp,
        pad_id=0,
        group_size=2,
        mini_batch_size=1,
    )
    # Two rows of DIFFERENT completion length, so padding is actually exercised.
    rows = []
    for n in (3, 5):
      rows.append(
          datatypes.RLTrainerPayload(
              prompt_ids=np.arange(2, dtype=np.int32),
              prompt_mask=np.ones(2, np.float32),
              completion_ids=np.arange(n, dtype=np.int32),
              completion_mask=np.ones(n, np.float32),
              advantages=np.full(n, 0.5, np.float32),
              rollout_per_token_logps=np.full(n, -0.25, np.float32),
          )
      )
    packed = assembler._pack_chunk(rows)  # pylint: disable=protected-access

    self.assertIsNotNone(
        packed.rollout_per_token_logps,
        "packer dropped rollout_per_token_logps despite the allowlist entry",
    )
    got = np.asarray(packed.rollout_per_token_logps)
    self.assertEqual(got.shape, (2, max_resp))
    # Real values preserved, tail padded, and padded exactly like the field it
    # is supposed to mirror.
    np.testing.assert_allclose(got[0, :3], -0.25)
    np.testing.assert_allclose(got[1, :5], -0.25)
    np.testing.assert_allclose(got[0, 3:], 0.0)

  def test_packer_omits_the_field_when_no_row_carries_it(self):
    """Optional means optional: absent everywhere must stay None, not zeros."""
    from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top

    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=4,
        max_response_length=6,
        pad_id=0,
        group_size=2,
        mini_batch_size=1,
    )
    rows = [
        datatypes.RLTrainerPayload(
            prompt_ids=np.arange(2, dtype=np.int32),
            prompt_mask=np.ones(2, np.float32),
            completion_ids=np.arange(3, dtype=np.int32),
            completion_mask=np.ones(3, np.float32),
            advantages=np.full(3, 0.5, np.float32),
        )
        for _ in range(2)
    ]
    packed = assembler._pack_chunk(rows)  # pylint: disable=protected-access
    self.assertIsNone(packed.rollout_per_token_logps)


class AuxMetricsReachTheMetricStreamTest(absltest.TestCase):
  """The loss's aux dict must survive to the metric writer.

  `_record_fwd_bwd` used to hand `aux` to `_post_process_train_step`, whose base
  implementation is `pass`, so every metric `grpo_loss_fn` produces was computed
  and discarded. Enabling TIS then had no observable effect at all.
  """

  def test_scalar_aux_metrics_are_converted_for_the_buffer(self):
    from tunix.experimental.train import peft_trainer_v2  # pylint: disable=g-import-not-at-top

    aux = {
        "tis/is_oob_ratio": np.float32(0.25),
        "sample_mask/kept_frac": np.float32(0.8),
        "sampler_is/seq_geomean_max": np.float32(1.4),
        "sampler_is/seq_geomean_min": np.float32(0.6),
    }
    out = peft_trainer_v2._aux_to_additional_metrics(aux)  # pylint: disable=protected-access
    self.assertIsNotNone(out)
    self.assertCountEqual(out.keys(), aux.keys())
    # Reducer inferred by name: an extreme stays an extreme under pooling.
    self.assertIs(out["sampler_is/seq_geomean_max"][1], np.max)
    self.assertIs(out["sampler_is/seq_geomean_min"][1], np.min)
    self.assertIs(out["tis/is_oob_ratio"][1], np.mean)

  def test_non_scalar_entries_are_skipped_not_crashed_on(self):
    from tunix.experimental.train import peft_trainer_v2  # pylint: disable=g-import-not-at-top

    out = peft_trainer_v2._aux_to_additional_metrics(  # pylint: disable=protected-access
        {"per_token": np.zeros((2, 4), np.float32), "scalar": np.float32(1.0)}
    )
    self.assertEqual(list(out.keys()), ["scalar"])

  def test_non_dict_aux_is_none(self):
    from tunix.experimental.train import peft_trainer_v2  # pylint: disable=g-import-not-at-top

    self.assertIsNone(peft_trainer_v2._aux_to_additional_metrics(None))  # pylint: disable=protected-access

  def test_metrics_buffer_is_still_a_dataclass(self):
    """Regression: helpers were once inserted between the decorator and class.

    That left `@dataclasses.dataclass` applied to a function, which fails only
    at import time with `'function' object has no attribute '__mro__'` -- and
    only in the trainer subprocess, where it is easy to misread as a runtime
    fault rather than a syntax-level mistake.
    """
    import dataclasses  # pylint: disable=g-import-not-at-top
    from tunix.experimental.train import peft_trainer_v2  # pylint: disable=g-import-not-at-top

    self.assertTrue(dataclasses.is_dataclass(peft_trainer_v2.MetricsBuffer))


if __name__ == "__main__":
  absltest.main()
