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

"""CPU contracts for canonical re-score through both in-process vLLM modes."""

import inspect
import os
from itertools import count
from absl.testing import absltest
import numpy as np
import types
from unittest import mock

from tunix.generate import vllm_sampler
from tunix.rl.rollout import vllm_rollout


class _Tokenizer:

  def pad_id(self):
    return 0

  def eos_id(self):
    return 7


class _RescoreSampler:

  def __init__(self):
    self.tokenizer = _Tokenizer()
    self.reset_calls = 0
    self.prompts = None
    self.prompt_batches = []

  def reset_prefix_cache_when_idle(self):
    self.reset_calls += 1

  def generate_request_outputs(
      self, prompts, sampling_params, *, reset_prefix_cache=False
  ):
    del sampling_params
    if reset_prefix_cache:
      self.reset_calls += 1
    self.prompts = prompts
    self.prompt_batches.append(prompts)
    outputs = []
    for prompt in prompts:
      token_ids = prompt["prompt_token_ids"]
      prompt_logprobs = [None]
      for token_id in token_ids[1:]:
        prompt_logprobs.append({
            token_id: types.SimpleNamespace(logprob=-float(token_id))
        })
      outputs.append(types.SimpleNamespace(
          prompt_logprobs=prompt_logprobs,
          num_cached_tokens=0,
      ))
    return outputs


class VllmRolloutCanonicalTest(absltest.TestCase):

  def test_jax_seed_route_uses_engine_global_and_rejects_per_request(self):
    self.assertIn("seed", inspect.signature(vllm_sampler.EngineArgs).parameters)
    config = types.SimpleNamespace(
        seed=None,
        rollout_vllm_tpu_backend_type="jax",
        rollout_vllm_kwargs={"seed": 42},
    )
    self.assertEqual(
        vllm_rollout._validated_vllm_seed_route(config),  # pylint: disable=protected-access
        (None, 42),
    )

    config.seed = 42
    with self.assertRaisesRegex(ValueError, "does not support per-request"):
      vllm_rollout._validated_vllm_seed_route(config)  # pylint: disable=protected-access

    config.seed = None
    config.rollout_vllm_kwargs["seed"] = 42.0
    with self.assertRaisesRegex(ValueError, "must be an integer"):
      vllm_rollout._validated_vllm_seed_route(config)  # pylint: disable=protected-access

  def test_p28_full_chain_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.run_p28_full_chain_gate.return_value = (
        "ok"
    )

    self.assertEqual(rollout.run_p28_full_chain_gate(), "ok")
    rollout._canonical_engine_adapter.run_p28_full_chain_gate.assert_called_once_with()

  def test_p28_block_vjp_layer_index_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.run_p28_block_vjp_gate.return_value = (
        "ok"
    )

    self.assertEqual(rollout.run_p28_block_vjp_gate(layer_index=17), "ok")
    rollout._canonical_engine_adapter.run_p28_block_vjp_gate.assert_called_once_with(
        layer_index=17
    )

  def test_sampler_driver_mode_uses_locked_reset_and_driver_generate(self):
    sampler = object.__new__(vllm_sampler.VllmSampler)
    sampler.llm = None
    sampler._driver = mock.Mock()
    sampler._generate_server_mode = mock.Mock(return_value=["driver-output"])
    prompts = [{"prompt_token_ids": [1, 2]}]
    params = object()

    self.assertEqual(
        sampler.generate_request_outputs(
            prompts, params, reset_prefix_cache=True
        ),
        ["driver-output"],
    )
    sampler._generate_server_mode.assert_called_once_with(
        prompts,
        params,
        reset_prefix_cache=True,
        reset_timeout_s=300.0,
    )

  def test_server_mode_deadline_aborts_unfinished_request(self):
    sampler = object.__new__(vllm_sampler.VllmSampler)
    sampler.llm = None
    sampler._request_counter = count()
    sampler._driver = mock.Mock()
    future = mock.Mock()
    future.result.side_effect = TimeoutError()
    future.done.return_value = False
    sampler._driver.submit_requests.return_value = [future]

    with self.assertRaisesRegex(TimeoutError, 'aborted 1 unfinished'):
      sampler._generate_server_mode(
          [{"prompt_token_ids": [1, 2]}],
          object(),
          request_timeout_s=0.01,
      )

    sampler._driver.cancel.assert_called_once_with("0")

  def test_prefill_rescore_uses_mode_independent_sampler_contract(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_prefill_rescore_provenance = None
    result = rollout.get_prefill_rescore_logps(
        prompt_tokens=np.asarray([[0, 1, 2], [0, 0, 5]], np.int32),
        completion_tokens=np.asarray([[3, 4, 0], [6, 0, 0]], np.int32),
        processed=False,
        completion_lengths=np.asarray([2, 1], np.int32),
    )

    np.testing.assert_array_equal(
        result, np.asarray([[-3.0, -4.0, 0.0], [-6.0, 0.0, 0.0]], np.float32)
    )
    self.assertEqual(rollout._sampler.reset_calls, 1)
    self.assertEqual(
        [prompt["prompt_token_ids"] for prompt in rollout._sampler.prompts],
        [[1, 2, 3, 4], [5, 6]],
    )

  def test_processed_rescore_skips_engine_for_empty_completion_batch(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = None
    rollout._last_prefill_rescore_provenance = None
    with mock.patch.dict(
        os.environ, {"CANON_PROMPT_PROCESSED_LOGPROBS": "1"}, clear=True
    ):
      result = rollout.get_prefill_rescore_logps(
          prompt_tokens=np.asarray([[0, 1, 2], [0, 0, 5]], np.int32),
          completion_tokens=np.zeros((2, 3), np.int32),
          completion_lengths=np.zeros((2,), np.int32),
          processed=True,
      )

    np.testing.assert_array_equal(result, np.zeros((2, 3), np.float32))
    self.assertEqual(rollout._sampler.prompt_batches, [])
    self.assertFalse(
        rollout._last_prefill_rescore_provenance["engine_called"]
    )
    self.assertEqual(
        rollout._last_prefill_rescore_provenance["skip_reason"],
        "empty-completion-batch",
    )

  def test_processed_rescore_still_requires_provenance_for_any_target(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = None
    with mock.patch.dict(
        os.environ, {"CANON_PROMPT_PROCESSED_LOGPROBS": "1"}, clear=True
    ), self.assertRaisesRegex(RuntimeError, "must follow generate"):
      rollout.get_prefill_rescore_logps(
          prompt_tokens=np.asarray([[1, 2], [3, 4]], np.int32),
          completion_tokens=np.asarray([[0, 0], [5, 0]], np.int32),
          completion_lengths=np.asarray([0, 1], np.int32),
          processed=True,
      )

  def test_p58_native_processed_rescore_uses_only_signed_stock_observer(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    rollout._last_prefill_rescore_provenance = None
    native = {
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
        ),
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "native",
        "CANON_ENGINE_MODULE_C": "0",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "0",
        "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "1",
    }
    with mock.patch.dict(os.environ, native, clear=True):
      result = rollout.get_prefill_rescore_logps(
          prompt_tokens=np.asarray([[1, 2]], np.int32),
          completion_tokens=np.asarray([[3, 0]], np.int32),
          completion_lengths=np.asarray([1], np.int32),
          processed=True,
      )

    np.testing.assert_array_equal(
        result, np.asarray([[-3.0, 0.0]], np.float32)
    )
    self.assertEqual(
        rollout._last_prefill_rescore_provenance["processor"],
        "p58-native-stock-observer",
    )

  def test_p58_native_processed_rescore_rejects_missing_observer(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    native = {
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
        ),
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "native",
        "CANON_ENGINE_MODULE_C": "0",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "0",
        "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "0",
    }
    with mock.patch.dict(os.environ, native, clear=True):
      with self.assertRaisesRegex(RuntimeError, "signed stock prompt observer"):
        rollout.get_prefill_rescore_logps(
            np.asarray([[1, 2]], np.int32),
            np.asarray([[3]], np.int32),
            processed=True,
            completion_lengths=np.asarray([1], np.int32),
        )

  def test_p58_zero_processed_rescore_rejects_native_observer(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    zero = {
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
        ),
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "zero",
        "CANON_ENGINE_MODULE_C": "1",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
        "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "1",
    }
    with mock.patch.dict(os.environ, zero, clear=True):
      with self.assertRaisesRegex(RuntimeError, "outside its signed arm"):
        rollout.get_prefill_rescore_logps(
            np.asarray([[1, 2]], np.int32),
            np.asarray([[3]], np.int32),
            processed=True,
            completion_lengths=np.asarray([1], np.int32),
        )

  def test_p58_zero_processed_rescore_keeps_canonical_processor(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_sampling_transforms = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    zero = {
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env"
        ),
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "zero",
        "CANON_ENGINE_MODULE_C": "1",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
        "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "0",
    }
    with mock.patch.dict(os.environ, zero, clear=True):
      rollout.get_prefill_rescore_logps(
          np.asarray([[1, 2]], np.int32),
          np.asarray([[3]], np.int32),
          processed=True,
          completion_lengths=np.asarray([1], np.int32),
      )
    self.assertEqual(
        rollout._last_prefill_rescore_provenance["processor"],
        "canonical-processed",
    )

  def test_grouped_prefill_rescore_changes_only_submission_grouping(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_prefill_rescore_provenance = None
    rollout._last_grouped_prefill_rescore_provenance = None
    prompts = np.asarray(
        [[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.int32
    )
    completions = np.asarray(
        [[5, 0], [6, 0], [7, 0], [8, 0]], dtype=np.int32
    )
    with mock.patch.dict(
        os.environ, {"CANON_P35_ENVELOPE": "1"}, clear=False
    ):
      result = rollout.get_grouped_prefill_rescore_logps(
          prompts,
          completions,
          completion_lengths=np.ones((4,), dtype=np.int32),
          group_size=2,
          processed=False,
          source_row_indices=np.asarray([0, 2, 1, 3]),
          diagnostic_arm="B",
      )

    np.testing.assert_array_equal(
        result,
        np.asarray(
            [[-5.0, 0.0], [-6.0, 0.0], [-7.0, 0.0], [-8.0, 0.0]],
            dtype=np.float32,
        ),
    )
    self.assertEqual(rollout._sampler.reset_calls, 2)
    self.assertEqual(
        [len(batch) for batch in rollout._sampler.prompt_batches], [2, 2]
    )
    self.assertEqual(
        rollout._last_grouped_prefill_rescore_provenance["group_size"], 2
    )
    self.assertEqual(
        rollout._last_grouped_prefill_rescore_provenance["groups"], 2
    )
    self.assertEqual(
        rollout._last_grouped_prefill_rescore_provenance["source_row_indices"],
        (0, 2, 1, 3),
    )
    self.assertEqual(
        rollout._last_grouped_prefill_rescore_provenance["diagnostic_arm"], "B"
    )
    contract = rollout.p35_grouped_prefill_contract()
    self.assertEqual(len(contract["group_provenance"]), 2)
    self.assertTrue(
        all(
            group["reset_prefix_cache"] is True
            for group in contract["group_provenance"]
        )
    )
    self.assertNotIn("CANON_P35_ARM", os.environ)

  def test_grouped_prefill_rescore_rejects_partial_group(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    with self.assertRaisesRegex(ValueError, "exact number of groups"):
      rollout.get_grouped_prefill_rescore_logps(
          np.ones((3, 2), dtype=np.int32),
          np.ones((3, 2), dtype=np.int32),
          completion_lengths=np.ones((3,), dtype=np.int32),
          group_size=2,
          processed=False,
      )

  def test_grouped_prefill_rescore_preserves_multichunk_sequences(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    rollout._last_prefill_rescore_provenance = None
    rollout._last_grouped_prefill_rescore_provenance = None
    prompts = np.arange(2 * 300, dtype=np.int32).reshape(2, 300) + 1
    completions = np.arange(2 * 256, dtype=np.int32).reshape(2, 256) + 1000
    with mock.patch.dict(
        os.environ, {"CANON_P35_ENVELOPE": "1"}, clear=False
    ):
      result = rollout.get_grouped_prefill_rescore_logps(
          prompts,
          completions,
          completion_lengths=np.asarray([256, 200], dtype=np.int32),
          group_size=2,
          processed=False,
          source_row_indices=np.asarray([0, 2]),
          diagnostic_arm="B",
      )

    self.assertEqual(result.shape, (2, 256))
    self.assertEqual(
        rollout._last_grouped_prefill_rescore_provenance[
            "group_provenance"
        ][0]["sequence_lengths"],
        (556, 500),
    )
    self.assertEqual(
        [len(prompt["prompt_token_ids"]) for prompt in rollout._sampler.prompts],
        [556, 500],
    )

  def test_grouped_prefill_rescore_rejects_duplicate_source_rows(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._sampler = _RescoreSampler()
    with self.assertRaisesRegex(ValueError, "duplicates"):
      rollout.get_grouped_prefill_rescore_logps(
          np.ones((2, 2), dtype=np.int32),
          np.ones((2, 2), dtype=np.int32),
          completion_lengths=np.ones((2,), dtype=np.int32),
          group_size=2,
          processed=False,
          source_row_indices=np.asarray([1, 1]),
      )

  def test_exact_weight_attestation_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.attest_exact_live_weights.return_value = {
        "equal": True
    }
    state = object()
    self.assertEqual(
        rollout.attest_canonical_engine_weights(state), {"equal": True}
    )
    rollout._canonical_engine_adapter.attest_exact_live_weights.assert_called_once_with(
        state
    )

  def test_selected_engine_weight_attestation_uses_registered_adapter(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.attest_exact_live_weights.return_value = {
        "equal": True
    }
    state = object()

    self.assertEqual(
        rollout.attest_exact_engine_weights(state), {"equal": True}
    )
    rollout._canonical_engine_adapter.attest_exact_live_weights.assert_called_once_with(
        state
    )

  def test_p58_native_uses_observer_without_registering_adapter(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = None
    rollout._sampler = object()
    state = object()
    signed_native = {
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "native",
        "CANON_ENGINE_MODULE_C": "0",
    }
    with mock.patch.dict(os.environ, signed_native, clear=True), mock.patch(
        "tunix.rl.canonical_qwen3_adapter.attest_exact_live_engine_weights",
        return_value={"equal": True},
    ) as observer:
      self.assertEqual(
          rollout.attest_exact_engine_weights(state), {"equal": True}
      )

    observer.assert_called_once_with(
        sampler=rollout._sampler, trainer_state=state
    )
    self.assertIsNone(rollout._canonical_engine_adapter)

  def test_stock_weight_observer_rejects_unsigned_or_zero_arm(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = None
    rollout._sampler = object()
    base = {
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "native",
        "CANON_ENGINE_MODULE_C": "0",
    }
    for changed in (
        {"CANON_P58_DEEPSWE_TIM": "0"},
        {"CANON_P58_TIM_ADMITTED": "0"},
        {"CANON_P58_TIM_ARM": "zero"},
        {"CANON_ENGINE_MODULE_C": "1"},
    ):
      with self.subTest(changed=changed), mock.patch.dict(
          os.environ, {**base, **changed}, clear=True
      ):
        with self.assertRaisesRegex(RuntimeError, "signed P58 native"):
          rollout.attest_exact_engine_weights(object())

  def test_p58_native_rejects_registered_canonical_adapter(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._sampler = object()
    signed_native = {
        "CANON_P34_DEEPSWE": "1",
        "CANON_P58_DEEPSWE_TIM": "1",
        "CANON_P58_TIM_ADMITTED": "1",
        "CANON_P58_TIM_ARM": "native",
        "CANON_ENGINE_MODULE_C": "0",
    }
    with mock.patch.dict(os.environ, signed_native, clear=True):
      with self.assertRaisesRegex(RuntimeError, "forbids a registered"):
        rollout.attest_exact_engine_weights(object())
    rollout._canonical_engine_adapter.attest_exact_live_weights.assert_not_called()

  def test_p35_adapter_contract_is_forwarded(self):
    rollout = object.__new__(vllm_rollout.VllmRollout)
    rollout._canonical_engine_adapter = mock.Mock()
    rollout._canonical_engine_adapter.p35_envelope_contract_attestation.return_value = {
        "local_m": 256
    }
    self.assertEqual(
        rollout.canonical_p35_adapter_contract(), {"local_m": 256}
    )


if __name__ == "__main__":
  absltest.main()
