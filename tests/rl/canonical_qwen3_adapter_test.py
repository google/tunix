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

"""Tests for the functional trainer-to-engine Qwen3 weight map."""

from absl.testing import absltest
import contextlib
import dataclasses
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import os
import types
from unittest import mock

from tunix.rl import canonical_qwen3_adapter


def _state(tree):
  return nnx.State(jax.tree.map(lambda value: nnx.Param(value), tree))


class _ModelConfig:

  max_model_len = 1024
  max_logprobs = 1

  def get_vocab_size(self):
    return 8

  def get_total_num_kv_heads(self):
    return 1

  def get_head_size(self):
    return 2


class _Runner:

  def __init__(self, state, mesh):
    self.state = state
    self.state_leaves = tuple(jax.tree.leaves(state))
    self.model_fn = lambda *args: args
    self.compute_logits_fn = lambda *args: args
    self.mesh = mesh
    self.kv_caches = [jnp.zeros((1, 1, 1, 1, 2), jnp.bfloat16)]
    self.layer_name_to_kvcache_index = {"layer.0": 0}
    self.is_first_rank = True
    self.is_last_rank = True
    self.model_config = _ModelConfig()


@dataclasses.dataclass
class _AttentionMetadata:
  input_positions: jax.Array
  block_tables: jax.Array
  seq_lens: jax.Array
  query_start_loc: jax.Array
  request_distribution: jax.Array
  padded_num_reqs: int | None = None


class _ForwardRunner(_Runner):

  def __init__(self, state, mesh):
    super().__init__(state, mesh)
    cache_sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(None, None, "model", None, None)
    )
    self.kv_caches = [
        jax.device_put(jnp.zeros((4, 2, 1, 2, 2), jnp.bfloat16), cache_sharding)
    ]
    self.max_num_reqs = 2
    self.block_size = 2
    self.vllm_config = object()
    self._canonical_attention_metadata_cls = _AttentionMetadata
    self._canonical_set_forward_context = lambda *args: contextlib.nullcontext()

    @dataclasses.dataclass
    class _SamplingMetadata:
      temperature: jax.Array
      top_k: jax.Array
      top_p: jax.Array
      do_sampling: bool
      logprobs: bool

    @dataclasses.dataclass
    class _Logprobs:
      logprob_token_ids: jax.Array
      logprobs: jax.Array
      selected_token_ranks: jax.Array

    def sample(key, mesh_arg, logits, metadata):
      del key, mesh_arg
      return jnp.zeros((logits.shape[0],), jnp.int32), (
          logits / metadata.temperature[:, None]
      )

    def compute_and_gather(logits, token_ids, max_logprobs):
      del max_logprobs
      logprobs = jax.nn.log_softmax(logits, axis=-1)
      target = jnp.take_along_axis(logprobs, token_ids[:, None], axis=-1)
      return _Logprobs(token_ids[:, None], target, jnp.zeros_like(token_ids))

    self._canonical_sampling_metadata_cls = _SamplingMetadata
    self._canonical_sample = sample
    self._canonical_compute_and_gather_logprobs = compute_and_gather

    def model_fn(
        leaves,
        caches,
        input_ids,
        metadata,
        inputs_embeds,
        positions,
        static_kv_indices,
        lora_metadata,
        intermediate_tensors,
        is_first_rank,
        is_last_rank,
    ):
      del (
          inputs_embeds,
          positions,
          static_kv_indices,
          lora_metadata,
          intermediate_tensors,
          is_first_rank,
          is_last_rank,
      )
      scale = leaves[1][0].astype(jnp.float32)
      q_len = metadata.query_start_loc[1]
      previous = caches[0][0, 0, 0, 0, 0].astype(jnp.float32)
      first_row_bonus = jnp.where(
          (jnp.arange(input_ids.shape[0]) == 0) & (q_len > 0),
          previous / 10,
          0.0,
      )
      hidden = (
          input_ids.astype(jnp.float32) + first_row_bonus
      )[:, None] * scale
      last_row = jnp.clip(q_len - 1, 0, input_ids.shape[0] - 1)
      last_value = input_ids[last_row].astype(caches[0].dtype)
      next_cache = caches[0].at[0, 0, 0, 0, 0].set(
          jnp.where(q_len > 0, last_value, caches[0][0, 0, 0, 0, 0])
      )
      return [next_cache], hidden, None, None

    def compute_logits_fn(leaves, hidden, lora_metadata):
      del lora_metadata
      vocab = jnp.arange(8, dtype=jnp.float32)
      bias = jnp.sum(leaves[0].astype(jnp.float32))
      return hidden * (vocab[None, :] + 1) + bias * vocab[None, :] / 100

    self.model_fn = model_fn
    self.compute_logits_fn = compute_logits_fn


class _Sampler:

  def __init__(self, runner, mapping, *, driver_mode=False):
    self.llm = None if driver_mode else object()
    self._driver = object() if driver_mode else None
    self._model_runner = runner
    self.to_hf_key_mappings = mapping
    self.to_hf_transpose_keys = {"w": (1, 0)}
    self.to_hf_hook_fns = None
    self.args = {"tensor_parallel_size": 1}


class CanonicalQwen3AdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.target = _state({
        "engine": {
            "a": jnp.full((3, 2), -7, jnp.bfloat16),
            "b": jnp.full((2,), -9, jnp.bfloat16),
        }
    })
    self.mapping = {
        "trainer.w": ("engine.a", (None, None)),
        "trainer.n": ("engine.b", (None,)),
    }

  def _map(self, w, n):
    source = _state({"trainer": {"w": w, "n": n}})
    return canonical_qwen3_adapter.map_trainer_state_to_engine_leaves(
        trainer_state=source,
        engine_state_contract=self.target,
        key_mappings=self.mapping,
        transpose_keys={"w": (1, 0)},
    )

  def test_mapping_is_ordered_cast_and_non_mutating(self):
    target_before = [np.asarray(x).copy() for x in jax.tree.leaves(self.target)]
    mapped = self._map(
        jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        jnp.asarray([4, 5], jnp.float32),
    )
    self.assertEqual(mapped.paths, ("engine.a", "engine.b"))
    np.testing.assert_array_equal(
        np.asarray(mapped.leaves[0]),
        np.arange(6, dtype=np.float32).reshape(2, 3).T.astype(np.float16),
    )
    self.assertEqual(mapped.leaves[0].dtype, jnp.bfloat16)
    self.assertEqual(mapped.leaves[1].dtype, jnp.bfloat16)
    for before, after in zip(target_before, jax.tree.leaves(self.target)):
      np.testing.assert_array_equal(before, np.asarray(after))

    manifest = canonical_qwen3_adapter.inspect_trainer_state_to_engine_contract(
        trainer_state=_state({"trainer": {
            "w": jnp.zeros((2, 3), jnp.float32),
            "n": jnp.zeros((2,), jnp.float32),
        }}),
        engine_state_contract=self.target,
        key_mappings=self.mapping,
        transpose_keys={"w": (1, 0)},
    )
    self.assertEqual(manifest.target_paths, mapped.paths)
    self.assertLen(manifest.entries, 2)
    self.assertEqual(manifest.entries[0].mapped_shape, (3, 2))
    self.assertEqual(manifest.entries[0].mapped_dtype, "bfloat16")

  def test_mapping_is_jittable_and_has_live_vjp(self):
    def loss(w, n):
      leaves = self._map(w, n).leaves
      return sum(jnp.sum(x.astype(jnp.float32)) for x in leaves)

    w = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    n = jnp.asarray([4, 5], jnp.float32)
    eager = loss(w, n)
    compiled = jax.jit(loss)(w, n)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(compiled))
    gw, gn = jax.grad(loss, argnums=(0, 1))(w, n)
    np.testing.assert_array_equal(np.asarray(gw), np.ones((2, 3), np.float32))
    np.testing.assert_array_equal(np.asarray(gn), np.ones((2,), np.float32))

  def test_missing_source_or_target_is_rejected(self):
    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError, "trainer leaves"
    ):
      canonical_qwen3_adapter.map_trainer_state_to_engine_leaves(
          trainer_state=_state({"trainer": {
              "w": jnp.zeros((2, 3)),
              "n": jnp.zeros((2,)),
              "extra": jnp.zeros((1,)),
          }}),
          engine_state_contract=self.target,
          key_mappings=self.mapping,
          transpose_keys={"w": (1, 0)},
      )

    incomplete = {"trainer.w": ("engine.a", (None, None))}
    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError, "target-complete"
    ):
      canonical_qwen3_adapter.map_trainer_state_to_engine_leaves(
          trainer_state=_state(
              {"trainer": {"w": jnp.zeros((2, 3))}}
          ),
          engine_state_contract=self.target,
          key_mappings=incomplete,
          transpose_keys={"w": (1, 0)},
      )

  def test_live_engine_contract_and_negatives(self):
    mesh = jax.make_mesh((1,), ("model",), devices=jax.devices()[:1])
    runner = _Runner(self.target, mesh)
    sampler = _Sampler(runner, self.mapping)
    source = _state({"trainer": {
        "w": jnp.zeros((2, 3), jnp.float32),
        "n": jnp.zeros((2,), jnp.float32),
    }})
    with mock.patch.dict(os.environ, {"CANON_RPA_VJP2": "1"}, clear=False):
      contract = canonical_qwen3_adapter.inspect_live_engine_contract(
          sampler=sampler, trainer_state=source
      )
    self.assertEqual(contract.mapping_entries, 2)
    self.assertEqual(contract.state_leaves, 2)
    self.assertEqual(contract.kv_caches, 1)

    driver_sampler = _Sampler(runner, self.mapping, driver_mode=True)
    with mock.patch.dict(os.environ, {"CANON_RPA_VJP2": "1"}, clear=False):
      driver_contract = canonical_qwen3_adapter.inspect_live_engine_contract(
          sampler=driver_sampler, trainer_state=source
      )
    self.assertEqual(driver_contract, contract)

    sampler.llm = None
    sampler._driver = None
    del sampler._model_runner
    with mock.patch.dict(os.environ, {"CANON_RPA_VJP2": "1"}, clear=False):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError, "live model runner"
      ):
        canonical_qwen3_adapter.inspect_live_engine_contract(
            sampler=sampler, trainer_state=source
        )

  def test_shared_logprob_pipeline_is_one_function_object(self):
    def stock(logits, token_ids, max_logprobs):
      del max_logprobs
      return (jnp.take_along_axis(logits, token_ids[:, None], axis=-1),)

    def gather(logprobs, token_ids, max_logprobs):
      del max_logprobs
      return (jnp.take_along_axis(logprobs, token_ids[:, None], axis=-1),)

    mesh = jax.make_mesh((1,), ("model",), devices=jax.devices()[:1])
    runner = types.SimpleNamespace(mesh=mesh)
    runner_module = types.SimpleNamespace(compute_and_gather_logprobs=stock)
    sampling_module = types.SimpleNamespace(compute_and_gather_logprobs=stock)
    with mock.patch.dict(
        os.environ, {"CANON_PALLAS_LOGSOFTMAX": "1"}, clear=False
    ), mock.patch.object(
        canonical_qwen3_adapter.canonical_logsoftmax,
        "log_softmax",
        new=lambda value: jax.nn.log_softmax(value, axis=-1),
    ):
      installed = canonical_qwen3_adapter._install_shared_logprob_pipeline(
          runner,
          stock_compute_and_gather=stock,
          gather_logprobs=gather,
          runner_module=runner_module,
          sampling_module=sampling_module,
      )
      self.assertIs(installed, runner_module.compute_and_gather_logprobs)
      self.assertIs(installed, sampling_module.compute_and_gather_logprobs)
      self.assertIs(installed, runner._canonical_compute_and_gather_logprobs)
      result = installed(
          jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32),
          jnp.asarray([2], jnp.int32),
          1,
      )
      np.testing.assert_allclose(
          np.asarray(result[0]),
          np.asarray([[jax.nn.log_softmax(jnp.asarray([1.0, 2.0, 3.0]))[2]]]),
      )

      hostile = types.SimpleNamespace(compute_and_gather_logprobs=lambda *_: ())
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError, "unknown runner"
      ):
        canonical_qwen3_adapter._install_shared_logprob_pipeline(
            types.SimpleNamespace(mesh=mesh),
            stock_compute_and_gather=stock,
            gather_logprobs=gather,
            runner_module=hostile,
            sampling_module=types.SimpleNamespace(
                compute_and_gather_logprobs=stock
            ),
        )

  def test_live_adapter_primal_and_vjp(self):
    names = (
        "data",
        "attn_dp",
        "attn_dp_expert",
        "expert",
        "model",
        "dcp",
    )
    mesh = jax.make_mesh(
        (1, 1, 1, 1, 1, 1),
        names,
        devices=jax.devices()[:1],
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    runner = _ForwardRunner(self.target, mesh)
    sampler = _Sampler(runner, self.mapping)
    source = _state({"trainer": {
        "w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 100,
        "n": jnp.asarray([0.5, 0.25], jnp.float32),
    }})
    env = {
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
        "CANON_LOGPROB_M": "256",
        "MIN_TOKEN_BUCKET": "256",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )

      def loss(state):
        logps, entropy = adapter.compute_per_token_logps(
            graphdef=None,
            state=state,
            prompt_tokens=jnp.asarray([[0, 1, 2], [0, 2, 1]], jnp.int32),
            completion_tokens=jnp.asarray([[3, 4, 0], [4, 3, 0]], jnp.int32),
            pad_id=0,
            eos_id=7,
            stop_gradient=False,
            return_entropy=True,
            temperature=1.0,
        )
        self.assertEqual(logps.shape, (2, 3))
        self.assertEqual(entropy.shape, (2, 3))
        return jnp.sum(logps + entropy)

      value_and_grad = jax.jit(jax.value_and_grad(loss))
      primal, grad = value_and_grad(source)
      primal_repeat, grad_repeat = value_and_grad(source)
      with mock.patch.dict(
          os.environ, {"CANON_L3_A3_DIAG": "1"}, clear=False
      ):
        diag_logps, diag_entropy, diagnostics = jax.jit(
            lambda st: adapter.compute_per_token_diagnostics(
                graphdef=None,
                state=st,
                prompt_tokens=jnp.asarray(
                    [[0, 1, 2], [0, 2, 1]], jnp.int32
                ),
                completion_tokens=jnp.asarray(
                    [[3, 4, 0], [4, 3, 0]], jnp.int32
                ),
                pad_id=0,
                eos_id=7,
                temperature=1.0,
            )
        )(source)
      self.assertEqual(diag_logps.shape, (2, 3))
      self.assertEqual(diag_entropy.shape, (2, 3))
      self.assertEqual(diagnostics["raw_rows"].shape, (2, 3, 8))
      self.assertEqual(diagnostics["processed_rows"].shape, (2, 3, 8))
      self.assertEqual(diagnostics["target_ids"].shape, (2, 3))
      np.testing.assert_array_equal(
          np.asarray(diagnostics["processed_targets"])
          - np.asarray(diag_logps),
          np.asarray(diagnostics["implied_log_normalizers"]),
      )
    self.assertTrue(np.isfinite(np.asarray(primal)))
    np.testing.assert_array_equal(np.asarray(primal), np.asarray(primal_repeat))
    grad_leaves = [np.asarray(x) for x in jax.tree.leaves(grad)]
    grad_repeat_leaves = [np.asarray(x) for x in jax.tree.leaves(grad_repeat)]
    self.assertTrue(all(np.isfinite(x).all() for x in grad_leaves))
    self.assertGreater(sum(np.count_nonzero(x) for x in grad_leaves), 0)
    for actual, repeated in zip(grad_leaves, grad_repeat_leaves):
      np.testing.assert_array_equal(actual, repeated)

    with mock.patch.dict(os.environ, env, clear=False):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError,
          "neutral top-k/top-p",
      ):
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
            sampler=sampler, sampling_kwargs={"top_k": 7, "top_p": 1.0}
        )
    sampler.llm = object()
    with mock.patch.dict(os.environ, {"CANON_RPA_VJP2": "0"}, clear=False):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError, "CANON_RPA_VJP2"
      ):
        canonical_qwen3_adapter.inspect_live_engine_contract(
            sampler=sampler, trainer_state=source
        )

  def test_long_sequence_uses_cache_carried_fixed_m_chunks(self):
    names = (
        "data",
        "attn_dp",
        "attn_dp_expert",
        "expert",
        "model",
        "dcp",
    )
    mesh = jax.make_mesh(
        (1, 1, 1, 1, 1, 1),
        names,
        devices=jax.devices()[:1],
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    runner = _ForwardRunner(self.target, mesh)
    sampler = _Sampler(runner, self.mapping)
    source = _state({"trainer": {
        "w": jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 100,
        "n": jnp.asarray([0.5, 0.25], jnp.float32),
    }})
    # Static width 512+4 produces three M256 slots.  Only 259 tokens are real:
    # two chunks execute and the third must be a true no-op.  The first action
    # crosses the 255/256 chunk boundary.
    prompt = np.zeros((512,), np.int32)
    prompt[-256:] = 1 + (np.arange(256, dtype=np.int32) % 7)
    completion = np.asarray([3, 4, 5, 0], np.int32)
    env = {
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
        "CANON_LOGPROB_M": "256",
        "MIN_TOKEN_BUCKET": "256",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )

      def value(state):
        return adapter.compute_per_token_logps(
            graphdef=None,
            state=state,
            prompt_tokens=jnp.asarray(prompt[None]),
            completion_tokens=jnp.asarray(completion[None]),
            pad_id=0,
            eos_id=7,
            stop_gradient=False,
            return_entropy=True,
            temperature=1.0,
        )

      def value_with_real_pad_token(state):
        return adapter.compute_per_token_logps(
            graphdef=None,
            state=state,
            prompt_tokens=jnp.asarray(prompt[None]),
            completion_tokens=jnp.asarray(completion[None]),
            pad_id=0,
            eos_id=7,
            stop_gradient=False,
            return_entropy=True,
            temperature=1.0,
            prompt_mask=jnp.asarray((prompt != 0)[None]),
            completion_mask=jnp.ones((1, completion.shape[0]), jnp.bool_),
        )

      def loss(state):
        logps, entropy = value(state)
        return jnp.sum(logps + entropy)

      compiled = jax.jit(value)
      logps, entropy = compiled(source)
      explicit_logps, explicit_entropy = jax.jit(value_with_real_pad_token)(
          source
      )
      logps_repeat, entropy_repeat = compiled(source)
      primal, grad = jax.jit(jax.value_and_grad(loss))(source)
      primal_repeat, grad_repeat = jax.jit(jax.value_and_grad(loss))(source)

    scale = 0.5
    mapped_w = np.asarray(
        jnp.asarray(
            np.arange(6, dtype=np.float32).reshape(2, 3) / 100
        ).T.astype(jnp.bfloat16).astype(jnp.float32)
    )
    bias = float(mapped_w.sum())
    last_prompt = float(prompt[-1])
    source_values = np.asarray(
        [last_prompt, completion[0] + last_prompt / 10, completion[1]],
        np.float32,
    )
    target_ids = completion[:3]
    vocab = np.arange(8, dtype=np.float32)
    rows = (
        source_values[:, None] * scale * (vocab[None, :] + 1)
        + bias * vocab[None, :] / 100
    )
    expected_logps = jax.nn.log_softmax(jnp.asarray(rows), axis=-1)[
        np.arange(3), target_ids
    ]
    normalized = jax.nn.log_softmax(jnp.asarray(rows), axis=-1)
    probabilities = jnp.exp(normalized)
    expected_entropy = -jnp.sum(probabilities * normalized, axis=-1)

    np.testing.assert_array_equal(
        np.asarray(logps[0, :3]), np.asarray(expected_logps)
    )
    np.testing.assert_array_equal(
        np.asarray(entropy[0, :3]), np.asarray(expected_entropy)
    )
    np.testing.assert_array_equal(np.asarray(logps[0, 3]), 0.0)
    np.testing.assert_array_equal(np.asarray(entropy[0, 3]), 0.0)
    self.assertNotEqual(float(explicit_logps[0, 3]), 0.0)
    self.assertNotEqual(float(explicit_entropy[0, 3]), 0.0)
    np.testing.assert_array_equal(np.asarray(logps), np.asarray(logps_repeat))
    np.testing.assert_array_equal(
        np.asarray(entropy), np.asarray(entropy_repeat)
    )
    np.testing.assert_array_equal(np.asarray(primal), np.asarray(primal_repeat))
    grad_leaves = [np.asarray(x) for x in jax.tree.leaves(grad)]
    grad_repeat_leaves = [np.asarray(x) for x in jax.tree.leaves(grad_repeat)]
    self.assertTrue(all(np.isfinite(x).all() for x in grad_leaves))
    self.assertGreater(sum(np.count_nonzero(x) for x in grad_leaves), 0)
    for actual, repeated in zip(grad_leaves, grad_repeat_leaves):
      np.testing.assert_array_equal(actual, repeated)

    with mock.patch.dict(os.environ, env, clear=False):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError,
          "max-model-length",
      ):
        adapter.compute_per_token_logps(
            graphdef=None,
            state=source,
            prompt_tokens=jnp.ones((1, 1024), jnp.int32),
            completion_tokens=jnp.ones((1, 1), jnp.int32),
            pad_id=0,
            eos_id=7,
        )

if __name__ == "__main__":
  absltest.main()
