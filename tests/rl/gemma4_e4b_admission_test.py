"""Routing tests for the Gemma 4 E4B P1 live-engine admission."""

import os
import unittest
from unittest import mock

import optax
from tunix.rl import gemma4_e4b_admission
from tunix.rl import rl_cluster
from tunix.rl.rollout import vllm_rollout


class Gemma4E4bAdmissionTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.environ = mock.patch.dict(
        os.environ,
        {gemma4_e4b_admission.SELECTOR_ENV: "p45"},
    )
    self.environ.start()
    self.addCleanup(self.environ.stop)

  def test_vllm_rollout_routes_only_to_gemma_observer(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = None
    rollout._sampler = object()
    expected = {"equal": True, "schema": "test"}
    with mock.patch.object(
        gemma4_e4b_admission,
        "attest_exact_live_engine_weights",
        return_value=expected,
    ) as observer:
      actual = rollout.attest_exact_engine_weights("trainer-state")
    self.assertEqual(actual, expected)
    observer.assert_called_once_with(
        sampler=rollout._sampler,
        trainer_state="trainer-state",
    )

  def test_vllm_rollout_rejects_canonical_adapter_in_p1(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = object()
    rollout._sampler = object()
    with self.assertRaisesRegex(RuntimeError, "forbids a canonical"):
      rollout.attest_exact_engine_weights("trainer-state")

  def test_p1_unit_microbatches_satisfy_real_training_config(self):
    config = rl_cluster.RLTrainingConfig(
        actor_optimizer=optax.sgd(0.0),
        eval_every_n_steps=0,
        mini_batch_size=1,
        train_micro_batch_size=1,
        compute_logps_micro_batch_size=1,
    )
    self.assertEqual(config.gradient_accumulation_steps, 1)


if __name__ == "__main__":
  unittest.main()
