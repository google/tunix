# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for wiring the sequence-level GRPO options to the loss."""

import os

from absl.testing import absltest
import numpy as np
import tunix
from tunix.experimental.common import datatypes
from tunix.experimental.examples.math_gsm8k_dist import run_gsm8k_dist_grpo
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.train import peft_trainer_v2
from tunix.rl import algorithm_config
from tunix.rl.agentic.agents import agent_types


def _launcher_orchestrator_cmd() -> str:
  path = os.path.join(
      os.path.dirname(os.path.abspath(tunix.__file__)),
      "experimental", "examples", "math_gsm8k_dist", "launcher.sh",
  )
  with open(path) as f:
    text = f.read()
  start = text.index("ORCHESTRATOR_CMD=(")
  return text[start : text.index("\n  export JAX_PLATFORMS", start)]


def _recipe_config(**overrides) -> algorithm_config.GRPOConfig:
  kwargs = dict(
      num_generations=4,
      temperature=1.0,
      use_rollout_logps=False,
      overlong_loss_masking=True,
      seq_logprob_error_threshold=2.0,
      truncated_importance_sampling_type="seq-mask-tis",
      truncated_importance_sampling_ratio_min=0.999,
      truncated_importance_sampling_ratio=1.002,
  )
  kwargs.update(overrides)
  return algorithm_config.GRPOConfig(**kwargs)


def _loss_algo_config(algo_config):
  adapter = algorithm_adapter.GRPOAdapter(algo_config=algo_config)
  fn = adapter.build_gen_model_input_fn(pad_id=0, eos_id=1)
  return fn.keywords["algo_config"]


def _trajectory(n_completion: int, status=None, old_logprobs=None):
  traj = {
      "prompt_tokens": np.arange(2, dtype=np.int32),
      "conversation_tokens": np.arange(n_completion, dtype=np.int32),
      "conversation_masks": np.ones(n_completion, dtype=np.float32),
  }
  if status is not None:
    traj["status"] = status
  if old_logprobs is not None:
    traj["old_logprobs"] = old_logprobs
  return datatypes.TrajectoryItem(prompt_id="p", traj=traj)


def _row(n_completion: int, **overrides) -> datatypes.RLTrainerPayload:
  kwargs = dict(
      prompt_ids=np.arange(2, dtype=np.int32),
      prompt_mask=np.ones(2, dtype=np.float32),
      completion_ids=np.arange(n_completion, dtype=np.int32),
      completion_mask=np.ones(n_completion, dtype=np.float32),
      advantages=np.full(n_completion, 0.5, dtype=np.float32),
  )
  kwargs.update(overrides)
  return datatypes.RLTrainerPayload(**kwargs)


def _assembler() -> batch_assembly.PaddedBatchAssembler:
  return batch_assembly.PaddedBatchAssembler(
      batch_size=2,
      max_prompt_length=4,
      max_response_length=6,
      pad_id=0,
      num_generations=2,
      mini_batch_size=1,
  )


class GRPOConfigValidationTest(absltest.TestCase):
  """Tests for validation of the sequence-level options."""

  def test_defaults_are_off(self):
    cfg = algorithm_config.GRPOConfig(num_generations=4)
    self.assertFalse(cfg.overlong_loss_masking)
    self.assertIsNone(cfg.seq_logprob_error_threshold)
    self.assertIsNone(cfg.truncated_importance_sampling_type)
    self.assertIsNone(cfg.truncated_importance_sampling_ratio_min)
    self.assertIsNone(cfg.truncated_importance_sampling_ratio)
    self.assertIsNone(cfg.sampler_is_length_buckets)

  def test_unsupported_tis_type_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "seq-mask-tis"):
      _recipe_config(truncated_importance_sampling_type="token-tis")

  def test_tis_without_a_band_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "requires a keep-band"):
      _recipe_config(
          truncated_importance_sampling_ratio_min=None,
          truncated_importance_sampling_ratio=None,
      )

  def test_half_specified_band_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "must be set together"):
      _recipe_config(truncated_importance_sampling_ratio=None)

  def test_reversed_band_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "must not exceed"):
      _recipe_config(
          truncated_importance_sampling_ratio_min=1.002,
          truncated_importance_sampling_ratio=0.999,
      )

  def test_non_increasing_length_buckets_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "strictly increasing"):
      _recipe_config(sampler_is_length_buckets=(2048, 512))

  def test_empty_length_buckets_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "non-empty"):
      _recipe_config(sampler_is_length_buckets=())

  def test_leave_one_out_estimator_is_accepted(self):
    cfg = algorithm_config.GRPOConfig(
        num_generations=4, temperature=1.0, advantage_estimator="grpo-loo"
    )
    self.assertEqual(cfg.advantage_estimator, "grpo-loo")

  def test_length_buckets_are_normalized_to_a_tuple(self):
    cfg = _recipe_config(sampler_is_length_buckets=[512, 2048])
    self.assertEqual(cfg.sampler_is_length_buckets, (512, 2048))

  def test_tis_requires_a_recomputed_ratio_denominator(self):
    # With use_rollout_logps=True the sampler's log-probs are already the
    # surrogate ratio's denominator, so seq-mask-tis would apply the same
    # correction a second time and square it.
    with self.assertRaisesRegex(ValueError, "use_rollout_logps=False"):
      _recipe_config(use_rollout_logps=True)

  def test_token_level_and_sequence_level_tis_cannot_both_be_set(self):
    # Both write the per-token importance weights, so the second would
    # overwrite the first.
    with self.assertRaisesRegex(ValueError, "only one may be used"):
      _recipe_config(sampler_is="token")


class AlgoConfigReachesTheLossTest(absltest.TestCase):
  """Tests for the config the adapter hands to the loss function."""

  def test_options_reach_the_loss_config(self):
    cfg = _loss_algo_config(
        _recipe_config(
            sampler_is_length_buckets=(512, 2048),
        )
    )
    self.assertTrue(cfg.overlong_loss_masking)
    self.assertEqual(cfg.seq_logprob_error_threshold, 2.0)
    self.assertEqual(cfg.truncated_importance_sampling_type, "seq-mask-tis")
    self.assertEqual(cfg.truncated_importance_sampling_ratio_min, 0.999)
    self.assertEqual(cfg.truncated_importance_sampling_ratio, 1.002)
    self.assertEqual(cfg.sampler_is_length_buckets, (512, 2048))

  def test_existing_options_still_reach_the_loss_config(self):
    cfg = _loss_algo_config(
        _recipe_config(
            epsilon=0.2, epsilon_high=0.28, beta=0.0, loss_agg_mode="token-mean"
        )
    )
    self.assertEqual(cfg.epsilon, 0.2)
    self.assertEqual(cfg.epsilon_high, 0.28)
    self.assertEqual(cfg.beta, 0.0)
    self.assertEqual(cfg.loss_agg_mode, "token-mean")


class RolloutLogprobRequestTest(absltest.TestCase):
  """Tests for when the rollout is asked to return log-probabilities."""

  def _return_logprobs(self, algo_config) -> bool:
    program = rl_program.StandardRLProgram(
        algo=algorithm_adapter.GRPOAdapter(algo_config=algo_config)
    )
    return program.generation_args.return_logprobs

  def test_requested_when_the_ppo_denominator_uses_them(self):
    self.assertTrue(
        self._return_logprobs(
            algorithm_config.GRPOConfig(
                num_generations=4, temperature=1.0, use_rollout_logps=True
            )
        )
    )

  def test_not_requested_when_nothing_needs_them(self):
    self.assertFalse(
        self._return_logprobs(
            algorithm_config.GRPOConfig(
                num_generations=4, temperature=1.0, use_rollout_logps=False
            )
        )
    )

  def test_use_rollout_logps_false_requires_batch_size_equal_to_mini_batch_size(
      self,
  ):
    adapter = algorithm_adapter.GRPOAdapter(
        algo_config=_recipe_config(), mini_batch_size=2
    )
    with self.assertRaisesRegex(
        ValueError, "use_rollout_logps=False requires batch_size == mini_batch_size"
    ):
      rl_program.StandardRLProgram(algo=adapter, batch_size=4)
    program = rl_program.StandardRLProgram(algo=adapter, batch_size=2)
    self.assertEqual(program.full_batch_size, 2)
    self.assertEqual(program.mini_batch_size, 2)

  def test_requested_for_the_gates_alone(self):
    # A recipe may recompute the PPO denominator on the trainer and still gate
    # on the sampler-vs-trainer comparison.
    self.assertTrue(
        self._return_logprobs(_recipe_config(use_rollout_logps=False))
    )
    self.assertTrue(
        self._return_logprobs(
            algorithm_config.GRPOConfig(
                num_generations=4,
                temperature=1.0,
                use_rollout_logps=False,
                seq_logprob_error_threshold=2.0,
            )
        )
    )

  def test_overlong_masking_alone_does_not_request_them(self):
    self.assertFalse(
        self._return_logprobs(
            algorithm_config.GRPOConfig(
                num_generations=4,
                temperature=1.0,
                use_rollout_logps=False,
                overlong_loss_masking=True,
            )
        )
    )


class PayloadFieldsTest(absltest.TestCase):
  """Tests for the per-sequence fields the adapter puts on a payload."""

  def _payload(self, item, **config_overrides):
    """The payload for `item`, built as the first of a full prompt group."""
    config = _recipe_config(num_generations=2, **config_overrides)
    adapter = algorithm_adapter.GRPOAdapter(algo_config=config)
    group = [item, _trajectory(3)]
    return adapter.create_trainer_payloads(group, rewards=[1.0, 0.0])[0]

  def test_rollout_logprobs_are_carried_when_the_denominator_is_recomputed(
      self,
  ):
    logps = [-0.1, -0.2, -0.3]
    payload = self._payload(
        _trajectory(3, old_logprobs=logps), use_rollout_logps=False
    )
    self.assertIsNone(payload.old_per_token_logps)
    np.testing.assert_allclose(
        payload.rollout_per_token_logps, np.float32(logps)
    )

  def test_both_logprob_fields_are_set_when_the_denominator_uses_them(self):
    # Reachable only without seq-mask-tis: the two together would apply the
    # sampler correction twice, and the config refuses that combination.
    logps = [-0.1, -0.2, -0.3]
    payload = self._payload(
        _trajectory(3, old_logprobs=logps),
        use_rollout_logps=True,
        truncated_importance_sampling_type=None,
        truncated_importance_sampling_ratio_min=None,
        truncated_importance_sampling_ratio=None,
    )
    np.testing.assert_allclose(payload.old_per_token_logps, np.float32(logps))
    np.testing.assert_allclose(
        payload.rollout_per_token_logps, np.float32(logps)
    )

  def test_truncated_trajectory_is_flagged_overlong(self):
    status = agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED.name
    payload = self._payload(_trajectory(3, status=status))
    self.assertEqual(payload.overlong, 1.0)

  def test_a_status_enum_is_read_like_its_name(self):
    # The agentic collector serialises `status.name`; the orchestrator's
    # critique stage resolves it back to a TrajectoryStatus member. Both reach
    # the adapter and must give the same verdict.
    enum_status = agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
    self.assertEqual(
        self._payload(_trajectory(3, status=enum_status)).overlong, 1.0
    )
    self.assertEqual(
        self._payload(
            _trajectory(3, status=agent_types.TrajectoryStatus.SUCCEEDED)
        ).overlong,
        0.0,
    )

  def test_finished_trajectory_is_not_flagged_overlong(self):
    status = agent_types.TrajectoryStatus.SUCCEEDED.name
    payload = self._payload(_trajectory(3, status=status))
    self.assertEqual(payload.overlong, 0.0)

  def test_overlong_is_absent_when_the_rollout_reports_no_status(self):
    self.assertIsNone(self._payload(_trajectory(3)).overlong)


class PaddedBatchAssemblerTest(absltest.TestCase):
  """Tests for carrying the new payload fields through batch assembly."""

  def test_rollout_logprobs_are_padded_like_the_completion(self):
    rows = [
        _row(n, rollout_per_token_logps=np.full(n, -0.25, dtype=np.float32))
        for n in (3, 5)
    ]
    packed = _assembler()._pack_chunk(rows)  # pylint: disable=protected-access
    got = np.asarray(packed.rollout_per_token_logps)
    self.assertEqual(got.shape, (2, 6))
    np.testing.assert_allclose(got[0, :3], -0.25)
    np.testing.assert_allclose(got[0, 3:], 0.0)
    np.testing.assert_allclose(got[1, :5], -0.25)

  def test_rollout_logprobs_stay_none_when_no_row_carries_them(self):
    packed = _assembler()._pack_chunk([_row(3), _row(3)])  # pylint: disable=protected-access
    self.assertIsNone(packed.rollout_per_token_logps)

  def test_overlong_is_stacked_one_value_per_sequence(self):
    rows = [
        _row(3, overlong=np.float32(1.0)),
        _row(5, overlong=np.float32(0.0)),
    ]
    packed = _assembler()._pack_chunk(rows)  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(packed.overlong), [1.0, 0.0])

  def test_overlong_stays_none_when_no_row_carries_it(self):
    packed = _assembler()._pack_chunk([_row(3), _row(3)])  # pylint: disable=protected-access
    self.assertIsNone(packed.overlong)

  def test_trailing_rows_are_padded_as_not_overlong(self):
    packed = _assembler()._pack_chunk([_row(3, overlong=np.float32(1.0))])  # pylint: disable=protected-access
    np.testing.assert_allclose(np.asarray(packed.overlong), [1.0, 0.0])


class ExampleCommandLineTest(absltest.TestCase):
  """Tests for the gsm8k example's flags for the sequence-level options."""

  def _config(self, argv):
    args = run_gsm8k_dist_grpo._parse_args(argv)  # pylint: disable=protected-access
    return run_gsm8k_dist_grpo._build_algo(args).algo_config  # pylint: disable=protected-access

  def test_defaults_leave_every_option_off(self):
    cfg = self._config([])
    self.assertFalse(cfg.overlong_loss_masking)
    self.assertIsNone(cfg.seq_logprob_error_threshold)
    self.assertIsNone(cfg.truncated_importance_sampling_type)
    self.assertEqual(cfg.advantage_estimator, "grpo")
    self.assertIsNone(cfg.sampler_is_length_buckets)

  def test_flags_reach_the_config(self):
    cfg = self._config([
        "--no-use_rollout_logps",
        "--overlong_loss_masking",
        "--seq_logprob_error_threshold=2.0",
        "--truncated_importance_sampling_type=seq-mask-tis",
        "--truncated_importance_sampling_ratio_min=0.999",
        "--truncated_importance_sampling_ratio=1.002",
        "--advantage_estimator=grpo-loo",
        "--sampler_is_length_buckets=512,2048",
    ])
    self.assertTrue(cfg.overlong_loss_masking)
    self.assertEqual(cfg.seq_logprob_error_threshold, 2.0)
    self.assertEqual(cfg.truncated_importance_sampling_type, "seq-mask-tis")
    self.assertEqual(cfg.truncated_importance_sampling_ratio_min, 0.999)
    self.assertEqual(cfg.truncated_importance_sampling_ratio, 1.002)
    self.assertEqual(cfg.advantage_estimator, "grpo-loo")
    self.assertFalse(cfg.use_rollout_logps)
    self.assertEqual(cfg.sampler_is_length_buckets, (512, 2048))

  def test_launcher_passes_every_option_it_exposes(self):
    block = _launcher_orchestrator_cmd()
    for flag in (
        "--epsilon_high=",
        "--loss_agg_mode=",
        "--advantage_estimator=",
        "--overlong_loss_masking",
        "--seq_logprob_error_threshold=",
        "--truncated_importance_sampling_type=",
        "--truncated_importance_sampling_ratio_min=",
        "--truncated_importance_sampling_ratio=",
        "--sampler_is_length_buckets=",
    ):
      self.assertIn(flag, block, f"launcher does not pass {flag}")


class AuxMetricForwardingTest(absltest.TestCase):
  """Tests for converting a loss function's aux metrics for the buffer."""

  def test_scalars_are_forwarded_with_a_reducer(self):
    aux = {
        "tis/is_oob_ratio": np.float32(0.25),
        "sampler_is/seq_geomean_max": np.float32(1.4),
        "sampler_is/seq_geomean_min": np.float32(0.6),
    }
    out = peft_trainer_v2._aux_to_additional_metrics(aux)  # pylint: disable=protected-access
    self.assertCountEqual(out.keys(), aux.keys())
    self.assertIs(out["tis/is_oob_ratio"][1], np.mean)
    self.assertIs(out["sampler_is/seq_geomean_max"][1], np.max)
    self.assertIs(out["sampler_is/seq_geomean_min"][1], np.min)

  def test_weighted_metrics_are_reduced_to_their_value(self):
    from tunix.sft import utils  # pylint: disable=g-import-not-at-top

    out = peft_trainer_v2._aux_to_additional_metrics(  # pylint: disable=protected-access
        {"kl": utils.WeightedMetric(np.float32(4.0), np.float32(2.0))}
    )
    self.assertAlmostEqual(float(out["kl"][0]), 2.0)

  def test_non_scalar_entries_are_skipped(self):
    out = peft_trainer_v2._aux_to_additional_metrics(  # pylint: disable=protected-access
        {"per_token": np.zeros((2, 4), np.float32), "scalar": np.float32(1.0)}
    )
    self.assertEqual(list(out.keys()), ["scalar"])

  def test_a_non_dict_yields_nothing(self):
    self.assertIsNone(peft_trainer_v2._aux_to_additional_metrics(None))  # pylint: disable=protected-access


if __name__ == "__main__":
  absltest.main()
