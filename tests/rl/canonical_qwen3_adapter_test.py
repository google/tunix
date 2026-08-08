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
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import os
import types
from unittest import mock

from tunix.rl import algo_core
from tunix.rl import canonical_qwen3_adapter
from tunix.rl import dp_training


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
    model_width = int(mesh.shape.get("model", 1))
    self.kv_caches = [
        jax.device_put(
            jnp.zeros((4, 2, model_width, 2, 2), jnp.bfloat16),
            cache_sharding,
        )
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


class _SegmentedEmbed(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(0.5, jnp.float32))

  def __call__(self, token_ids):
    return token_ids[:, None].astype(jnp.float32) * self.scale[...]


class _SegmentedNorm(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(0.25, jnp.float32))

  def __call__(self, hidden):
    return hidden * self.scale[...]


class _SegmentedHead(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(0.75, jnp.float32))

  def __call__(self, hidden):
    return hidden * self.scale[...]


class _SegmentedLayer(nnx.Module):

  def __init__(self, scale):
    self.scale = nnx.Param(jnp.asarray(scale, jnp.float32))

  def __call__(self, cache, hidden, metadata):
    output = hidden * self.scale[...] + metadata + cache * 0.1
    return cache + jnp.sum(output), output


class _SegmentedBackbone(nnx.Module):

  def __init__(self):
    self.embed_tokens = _SegmentedEmbed()
    self.layers = nnx.List(
        [_SegmentedLayer(1.5), _SegmentedLayer(2.0)]
    )
    self.start_layer = 0
    self.end_layer = 2
    self.norm = _SegmentedNorm()

  def __call__(self, caches, token_ids, metadata):
    hidden = self.embed_tokens(token_ids)
    next_caches = []
    for layer, cache in zip(self.layers, caches):
      cache, hidden = layer(cache, hidden, metadata)
      next_caches.append(cache)
    return next_caches, self.norm(hidden)


class _SegmentedModel(nnx.Module):

  def __init__(self):
    self.model = _SegmentedBackbone()
    self.lm_head = _SegmentedHead()


class _SegmentedRunner:

  def __init__(self):
    self.model = _SegmentedModel()
    _, self.state = nnx.split(self.model)
    self.state_leaves = tuple(jax.tree.leaves(self.state))
    self.kv_caches = [jnp.asarray(0.0), jnp.asarray(0.0)]
    self.is_first_rank = True
    self.is_last_rank = True
    self.compute_logits_fn = lambda leaves, hidden, _: hidden * leaves[-1]


class _CompleteSegmentedHead(nnx.Module):

  def __init__(self):
    self.scale = nnx.Param(jnp.asarray(0.75, jnp.float32))

  def __call__(self, hidden):
    columns = jnp.asarray([0.5, 1.0, 1.5], jnp.float32)
    return (hidden * self.scale[...] * columns[None, :]).astype(jnp.bfloat16)


class _CompleteSegmentedModel(nnx.Module):

  def __init__(self):
    self.model = _SegmentedBackbone()
    self.lm_head = _CompleteSegmentedHead()


class _CompleteSegmentedRunner:

  def __init__(self):
    self.model = _CompleteSegmentedModel()
    _, self.state = nnx.split(self.model)
    self.state_leaves = tuple(jax.tree.leaves(self.state))
    self.kv_caches = [jnp.asarray(0.0), jnp.asarray(0.0)]
    self.is_first_rank = True
    self.is_last_rank = True
    self.vllm_config = object()
    self.model_config = types.SimpleNamespace(
        max_model_len=4096,
        get_total_num_kv_heads=lambda: 1,
        get_head_size=lambda: 1,
    )


class CanonicalQwen3AdapterTest(absltest.TestCase):

  def _make_p32_group_adapter(self, *, sequence_bucket=4):
    runner = _CompleteSegmentedRunner()
    adapter = object.__new__(
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter
    )
    adapter._runner = runner  # pylint: disable=protected-access
    adapter._data_size = 16  # pylint: disable=protected-access
    adapter._tp_size = 4  # pylint: disable=protected-access
    adapter._sequence_bucket = sequence_bucket  # pylint: disable=protected-access
    adapter._bucket = 16 * sequence_bucket  # pylint: disable=protected-access
    adapter._max_model_len = 64  # pylint: disable=protected-access
    adapter._max_num_reqs = 16  # pylint: disable=protected-access
    adapter._blocks_per_req = 4  # pylint: disable=protected-access
    adapter._engine_state_contract = runner.state  # pylint: disable=protected-access
    adapter._key_mappings = {}  # pylint: disable=protected-access
    adapter._transpose_keys = None  # pylint: disable=protected-access
    adapter._hook_fns = None  # pylint: disable=protected-access
    adapter._set_forward_context = (  # pylint: disable=protected-access
        lambda *_: contextlib.nullcontext()
    )
    adapter._fresh_caches = types.MethodType(  # pylint: disable=protected-access
        lambda self: [jnp.asarray(0.0), jnp.asarray(0.0)], adapter
    )

    def group_chunk_inputs(self, spec, chunk_index):
      start = chunk_index * self._sequence_bucket
      end = start + self._sequence_bucket
      return (
          spec["packed_ids"][:, start:end].reshape(-1),
          spec["next_ids"][:, start:end].reshape(-1),
          jnp.asarray(0.125, jnp.float32),
      )

    adapter._p32_group_chunk_inputs = types.MethodType(  # pylint: disable=protected-access
        group_chunk_inputs, adapter
    )

    def processed_rows(logits, target_ids, temperature):
      normalized = jax.nn.log_softmax(logits / temperature, axis=-1)
      selected = jnp.take_along_axis(
          normalized, target_ids[:, None], axis=-1
      )[:, 0]
      probabilities = jnp.exp(normalized)
      entropy = -jnp.sum(probabilities * normalized, axis=-1)
      return selected, entropy

    def processed_rows_pullback(
        logits, target_ids, temperature, dlogps, dentropy
    ):
      _, pullback = jax.vjp(
          lambda values: processed_rows(values, target_ids, temperature),
          logits,
      )
      return pullback((dlogps, dentropy))[0]

    adapter._p28_processed_rows_fn = jax.jit(  # pylint: disable=protected-access
        processed_rows
    )
    adapter._p28_processed_rows_pullback_fn = jax.jit(  # pylint: disable=protected-access
        processed_rows_pullback
    )
    return adapter, runner

  def test_p32_dp16_grouped_forward_and_reverse_replay_exactly(self):
    adapter, runner = self._make_p32_group_adapter()
    row = jnp.arange(16, dtype=jnp.int32)[:, None]
    prompt = jnp.concatenate(
        (1 + row % 2, 2 + row % 2, jnp.zeros_like(row)), axis=1
    )
    completion = jnp.concatenate(
        (2 + row % 2, 1 + row % 2, jnp.zeros_like(row)), axis=1
    )
    prompt_valid = prompt != 0
    completion_valid = completion != 0
    spec = adapter._p32_group_spec(  # pylint: disable=protected-access
        prompt,
        completion,
        prompt_valid,
        completion_valid,
        1.0,
    )
    env = {
        "CANON_P28_SEGMENTED_FORWARD": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
      forward = adapter._p32_forward_group(  # pylint: disable=protected-access
          segmented, tuple(runner.state_leaves), spec, keep_cache_inputs=False
      )
      reverse = adapter._p32_reverse_group(  # pylint: disable=protected-access
          segmented,
          tuple(runner.state_leaves),
          spec,
          jnp.ones_like(forward["logps"]),
          jnp.zeros_like(forward["entropy"]),
      )

    self.assertEqual(forward["logps"].shape, (16, 3))
    np.testing.assert_array_equal(
        np.asarray(reverse["replay_logps"]), np.asarray(forward["logps"])
    )
    gradient_leaves = [
        np.asarray(value) for value in reverse["engine_gradients"]
    ]
    self.assertTrue(all(np.isfinite(value).all() for value in gradient_leaves))
    self.assertGreater(sum(np.count_nonzero(value) for value in gradient_leaves), 0)
    self.assertGreater(
        sum(
            np.count_nonzero(np.asarray(value))
            for value in jax.tree.leaves(
                reverse["initial_cache_cotangents"]
            )
        ),
        0,
    )

  def test_p32_dp16_group_spec_preserves_rank_local_order(self):
    adapter, _ = self._make_p32_group_adapter()
    prompt = jnp.zeros((16, 3), jnp.int32)
    completion = jnp.zeros((16, 3), jnp.int32)
    prompt = prompt.at[:, :2].set(
        jnp.arange(16, dtype=jnp.int32)[:, None] * 10
        + jnp.asarray([1, 2], jnp.int32)
    )
    completion = completion.at[:, :2].set(
        jnp.arange(16, dtype=jnp.int32)[:, None] * 10
        + jnp.asarray([3, 4], jnp.int32)
    )
    spec = adapter._p32_group_spec(  # pylint: disable=protected-access
        prompt,
        completion,
        prompt != 0,
        completion != 0,
        0.7,
    )
    self.assertEqual(spec["host_n_real"], (4,) * 16)
    self.assertEqual(spec["num_chunks"], 1)
    np.testing.assert_array_equal(
        np.asarray(spec["packed_ids"][:, :4]),
        np.asarray(
            jnp.arange(16, dtype=jnp.int32)[:, None] * 10
            + jnp.asarray([1, 2, 3, 4], jnp.int32)
        ),
    )
    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError,
        "one row per DP rank",
    ):
      adapter._p32_group_spec(  # pylint: disable=protected-access
          prompt[:15],
          completion[:15],
          prompt[:15] != 0,
          completion[:15] != 0,
          0.7,
      )

  def test_p32_dp16_segmented_transaction_streams_sixteen_groups(self):
    adapter, runner = self._make_p32_group_adapter(sequence_bucket=256)
    adapter._max_model_len = 6144  # pylint: disable=protected-access
    adapter._p32_d3b_segmented_engine = object()  # pylint: disable=protected-access
    adapter.map_engine_cotangents_to_trainer_state = types.MethodType(
        lambda self, state, cotangents: tuple(cotangents), adapter
    )

    def forward_group(self, segmented, leaves, spec, *, keep_cache_inputs):
      del segmented, leaves, keep_cache_inputs
      shape = spec["completion_valid"].shape
      return {
          "logps": jnp.zeros(shape, jnp.float32),
          "entropy": jnp.zeros(shape, jnp.float32),
          "counts": {"forward": 1},
      }

    def reverse_group(self, segmented, leaves, spec, dlogps, dentropy):
      del segmented, leaves, dentropy
      signal = jnp.sum(dlogps).astype(jnp.float32)
      return {
          "engine_gradients": (signal[None],),
          "initial_cache_cotangents": (jnp.asarray([1.0]),),
          "counts": {"reverse": 1},
          "replay_logps": jnp.zeros(
              spec["completion_valid"].shape, jnp.float32
          ),
          "replay_entropy": jnp.zeros(
              spec["completion_valid"].shape, jnp.float32
          ),
      }

    adapter._p32_forward_group = types.MethodType(  # pylint: disable=protected-access
        forward_group, adapter
    )
    adapter._p32_reverse_group = types.MethodType(  # pylint: disable=protected-access
        reverse_group, adapter
    )

    class FakeReducer:

      def __init__(self, template, *, dp_size):
        del template
        self.dp_size = dp_size
        self.values = []

      def begin(self):
        self.values = []

      def add(self, rank, contribution):
        if rank != len(self.values):
          raise ValueError("rank cadence changed")
        copied = jax.tree.map(lambda value: value + 0, contribution)
        jax.block_until_ready(copied)
        self.values.append(copied)

      def finalize(self):
        reduced = dp_training.fixed_dp_sum(self.values)
        fingerprints = tuple(f"rank-{rank}" for rank in range(self.dp_size))
        return reduced, {
            "dp_size": self.dp_size,
            "rank_contributions": self.dp_size,
            "rank_local_fingerprints": fingerprints,
            "rank_local_fingerprints_distinct": True,
            "reduction_transactions": 1,
            "reduction_rounds": 8,
            "replica_check_flags": self.dp_size,
            "post_reduction_replicas_exact": True,
        }

    adapter._p33_gradient_reducer_factory = FakeReducer  # pylint: disable=protected-access

    prompt_ids = jnp.zeros((256, 4096), jnp.int32).at[:, :2].set(
        jnp.asarray([1, 2], jnp.int32)
    )
    completion_ids = jnp.zeros((256, 2048), jnp.int32).at[:, :2].set(
        jnp.asarray([2, 1], jnp.int32)
    )
    prompt_mask = prompt_ids != 0
    completion_mask = completion_ids != 0
    train_example = types.SimpleNamespace(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        completion_valid_mask=completion_mask,
        old_per_token_logps=jnp.zeros((256, 2048), jnp.float32),
        ref_per_token_logps=None,
        advantages=jnp.linspace(-1.0, 1.0, 256, dtype=jnp.float32),
        sampler_is_weights=None,
        segment_ids=None,
    )
    algo_config = types.SimpleNamespace(
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
    mapped = canonical_qwen3_adapter.FunctionalEngineLeaves(
        paths=(), leaves=(jnp.asarray([1.0]),), source_to_target=()
    )
    streamed = []

    def consume(index, gradient, multiplier):
      streamed.append((
          index,
          tuple(np.asarray(value).copy() for value in jax.tree.leaves(gradient)),
          float(np.asarray(multiplier)),
      ))

    env = {
        "CANON_P32_DP16_SEGMENTED": "1",
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_P32_DP_REDUCTION_ADMITTED": "1",
            "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
            "CANON_P33_RUN_STAGE": "full",
            "CANON_P33_DISABLE_EVAL": "1",
            "CANON_WANDB_ONLINE_REQUIRED": "1",
          "CANON_P31_MONOTONIC_METRICS": "1",
          "CANON_WANDB_PROJECT": "zero-tim-frozenlake-dp16-tp4",
          "CANON_WANDB_GROUP": "qwen3-8b-dp16-tp4",
          "CANON_WANDB_RUN_NAME": "p33-frozenlake-adapter-test",
          "WANDB_MODE": "online",
          "WANDB_API_KEY": "test-key-not-a-credential",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P32_WORKLOAD": "frozenlake",
        "CANON_DP_SIZE": "16",
        "CANON_TP_SIZE": "4",
        "CANON_TOTAL_DEVICES": "64",
        "CANON_GLOBAL_PROMPTS": "32",
        "CANON_LOCAL_PROMPTS": "2",
        "CANON_NUM_GENERATIONS": "8",
        "CANON_LOCAL_TRAJECTORIES": "16",
        "CANON_GLOBAL_TRAJECTORIES": "256",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "MIN_TOKEN_BUCKET": "4096",
        "CANON_FIXED_AR": "1",
        "CANON_FIXED_AR_EMBED": "1",
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
        "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
        "CANON_PALLAS_LOGSOFTMAX": "1",
        "CANON_P28_SEGMENTED_FORWARD": "1",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_P29_FULL_TRAIN": "1",
        "CANON_ALIGNMENT_GATE": "1",
        "CANON_ALIGNMENT_GATE_ONLY": "0",
        "CANON_ALIGNMENT_UPDATE_CANARY": "0",
        "CANON_ALIGNMENT_TRAIN": "1",
        "CANON_P30_OPT_STATE_OFFLOAD": "1",
        "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
        "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
        "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
        "CANON_P30_RELEASE_CAPTURED_STATE": "1",
        "CANON_P30_RESHARD_ACCUMULATOR": "1",
        "FL_SHARED_MESH": "16,4",
        "XLA_FLAGS": "--xla_allow_excess_precision=false",
    }
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      result = adapter.segmented_dp_grpo_value_and_grad(
          trainer_state=(jnp.asarray([1.0]),),
          train_example=train_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
          gradient_microbatch_sink=consume,
      )

    self.assertIsNone(result["gradients"])
    self.assertEqual(result["gradient_microbatches"], 16)
    self.assertEqual([item[0] for item in streamed], list(range(16)))
    expected_multiplier = float(np.asarray(
        result["loss_output"].primary_loss.compute_scale()
    )) * 16.0
    self.assertTrue(all(
        item[2] == expected_multiplier for item in streamed
    ))
    self.assertEqual(
        tuple(report["trajectory_rows"] for report in result["reports"]),
        tuple(
            tuple(local + 16 * rank for rank in range(16))
            for local in range(16)
        ),
    )
    self.assertEqual(result["dp_reduction_visibility"], "EXPLICIT_FIXED_TREE")
    self.assertTrue(result["replica_equality"])
    self.assertEqual(result["dp_reduction_transactions"], 16)
    self.assertEqual(result["dp_reduction_rounds_per_transaction"], 8)
    self.assertEqual(result["dp_rank_pullbacks_per_transaction"], 16)
    self.assertTrue(all(
        len(set(fingerprints)) == 16
        for fingerprints in result["rank_local_gradient_fingerprints"]
    ))

  def test_p32_dp16_rejects_the_legacy_data1_segmented_reverse(self):
    adapter, _ = self._make_p32_group_adapter(sequence_bucket=256)
    with (
        mock.patch.dict(
            os.environ, {"CANON_P28_SEGMENTED_TRAIN": "1"}, clear=False
        ),
        self.assertRaisesRegex(
            canonical_qwen3_adapter.FunctionalMappingError,
            "data1 segmented reverse cannot run on a DP mesh",
        ),
    ):
      adapter.segmented_grpo_value_and_grad(
          trainer_state=(jnp.asarray([1.0]),),
          train_example=types.SimpleNamespace(),
          algo_config=types.SimpleNamespace(),
          pad_id=0,
          eos_id=2,
      )

  def test_dp16_topology_contract_uses_global_m4096_and_local_m256(self):
    env = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_DP_SIZE": "16",
        "CANON_TP_SIZE": "4",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "MIN_TOKEN_BUCKET": "4096",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      self.assertEqual(
          canonical_qwen3_adapter._canonical_topology_contract(),
          (16, 4, 256, 4096),
      )
      self.assertEqual(
          canonical_qwen3_adapter._canonical_logprob_bucket(), 4096
      )
      mesh = types.SimpleNamespace(
          axis_names=("data", "model"),
          shape={"data": 16, "model": 4},
      )
      self.assertEqual(
          canonical_qwen3_adapter._canonical_logprob_row_spec(mesh),
          jax.sharding.PartitionSpec("data", None),
      )

    with mock.patch.dict(
        os.environ, {**env, "MIN_TOKEN_BUCKET": "256"}, clear=False
    ):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError,
          r"dp\*CANON_LOGPROB_M",
      ):
        canonical_qwen3_adapter._canonical_topology_contract()

  def test_default_data1_adapter_retains_live_tp4(self):
    if len(jax.devices()) < 4:
      self.skipTest("requires at least four forced CPU or accelerator devices")
    names = (
        "data",
        "attn_dp",
        "attn_dp_expert",
        "expert",
        "model",
        "dcp",
    )
    mesh = jax.make_mesh(
        (1, 1, 1, 1, 4, 1),
        names,
        devices=jax.devices()[:4],
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    runner = _ForwardRunner(self.target, mesh)
    sampler = _Sampler(runner, self.mapping)
    sampler.args["tensor_parallel_size"] = 4
    env = {
        "CANON_P32_TRAIN_ADMITTED": "0",
        "CANON_LOGPROB_M": "256",
        "MIN_TOKEN_BUCKET": "256",
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )
    self.assertEqual(adapter._data_size, 1)
    self.assertEqual(adapter._tp_size, 4)
    self.assertEqual(adapter._bucket, 256)

  def test_dp16_grouping_and_metadata_are_rank_major(self):
    adapter = object.__new__(
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter
    )
    adapter._data_size = 16
    values = jnp.arange(256 * 3, dtype=jnp.int32).reshape(256, 3)
    grouped = adapter._group_batch_rows(values)
    self.assertEqual(grouped.shape, (16, 16, 3))
    np.testing.assert_array_equal(
        np.asarray(grouped[:, 0]), np.asarray(values[:16])
    )
    np.testing.assert_array_equal(
        np.asarray(grouped[:, 15]), np.asarray(values[-16:])
    )
    np.testing.assert_array_equal(
        np.asarray(adapter._ungroup_batch_rows(grouped)), np.asarray(values)
    )

    q_len = jnp.asarray([256] * 15 + [0], jnp.int32)
    kv_len = jnp.asarray([512] * 15 + [0], jnp.int32)
    metadata = canonical_qwen3_adapter._canonical_dp_attention_metadata_arrays(
        data_size=16,
        max_num_reqs=256,
        blocks_per_req=32,
        q_len=q_len,
        kv_len=kv_len,
    )
    block_tables, seq_lens, query_start, distribution = map(
        np.asarray, metadata
    )
    self.assertEqual(block_tables.shape, (8192,))
    self.assertEqual(seq_lens.shape, (256,))
    self.assertEqual(query_start.shape, (272,))
    self.assertEqual(distribution.shape, (48,))
    np.testing.assert_array_equal(block_tables[:32], np.arange(32))
    np.testing.assert_array_equal(block_tables[-512:-480], np.arange(32))
    self.assertEqual(seq_lens[0], 512)
    self.assertEqual(seq_lens[-16], 0)
    np.testing.assert_array_equal(distribution[-3:], [0, 0, 0])

  def test_dp16_grouping_rejects_partial_global_batch(self):
    adapter = object.__new__(
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter
    )
    adapter._data_size = 16
    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError, "divisible"
    ):
      adapter._group_batch_rows(jnp.zeros((255, 3), jnp.int32))

  def test_dp16_live_adapter_admits_only_the_frozen_mesh_contract(self):
    if len(jax.devices()) != 64:
      self.skipTest("requires exactly 64 forced CPU or accelerator devices")
    names = (
        "data",
        "attn_dp",
        "attn_dp_expert",
        "expert",
        "model",
        "dcp",
    )
    mesh = jax.make_mesh(
        (16, 1, 1, 1, 4, 1),
        names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    runner = _ForwardRunner(self.target, mesh)
    runner.max_num_reqs = 256
    sampler = _Sampler(runner, self.mapping)
    sampler.args["tensor_parallel_size"] = 4
    env = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_DP_SIZE": "16",
        "CANON_TP_SIZE": "4",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "MIN_TOKEN_BUCKET": "4096",
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )
      self.assertEqual(adapter._data_size, 16)
      self.assertEqual(adapter._tp_size, 4)
      self.assertEqual(adapter._bucket, 4096)
      self.assertEqual(adapter._sequence_bucket, 256)
      self.assertEqual(adapter._local_max_num_reqs, 16)
      self.assertEqual(adapter._cache_shape[0], 8192)

      sampler.args["tensor_parallel_size"] = 2
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError,
          "TP sizes differ",
      ):
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter(sampler=sampler)

  def test_dp16_grouped_forward_and_vjp_are_exactly_repeatable(self):
    if len(jax.devices()) != 64:
      self.skipTest("requires exactly 64 forced CPU or accelerator devices")
    names = (
        "data",
        "attn_dp",
        "attn_dp_expert",
        "expert",
        "model",
        "dcp",
    )
    mesh = jax.make_mesh(
        (16, 1, 1, 1, 4, 1),
        names,
        devices=jax.devices(),
        axis_types=(jax.sharding.AxisType.Auto,) * 6,
    )
    runner = _ForwardRunner(self.target, mesh)
    runner.max_num_reqs = 256
    cache_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec("data", None, "model", None, None),
    )
    runner.kv_caches = [
        jax.device_put(
            jnp.zeros((8192, 2, 4, 2, 2), jnp.bfloat16), cache_sharding
        )
    ]

    def grouped_model_fn(
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
          metadata,
          inputs_embeds,
          positions,
          static_kv_indices,
          lora_metadata,
          intermediate_tensors,
          is_first_rank,
          is_last_rank,
      )
      scale = leaves[1][0].astype(jnp.float32)
      return caches, input_ids.astype(jnp.float32)[:, None] * scale, None, None

    runner.model_fn = grouped_model_fn
    sampler = _Sampler(runner, self.mapping)
    sampler.args["tensor_parallel_size"] = 4
    env = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_DP_SIZE": "16",
        "CANON_TP_SIZE": "4",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "MIN_TOKEN_BUCKET": "4096",
        "CANON_RPA_VJP2": "1",
        "CANON_VJP2_MAX_SEQS": "1",
    }
    prompts = jnp.tile(jnp.asarray([[1, 2, 3]], jnp.int32), (16, 1))
    completions = jnp.tile(jnp.asarray([[4, 5]], jnp.int32), (16, 1))
    valid_prompt = jnp.ones_like(prompts, jnp.bool_)
    valid_completion = jnp.ones_like(completions, jnp.bool_)
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )
      engine_leaves = tuple(jax.tree.leaves(self.target))

      def loss(leaves):
        logps, entropy = adapter._sequence_group(
            leaves,
            prompts,
            completions,
            valid_prompt,
            valid_completion,
            0,
            1.0,
        )
        return jnp.sum(logps + entropy), (logps, entropy)

      compiled = jax.jit(jax.value_and_grad(loss, has_aux=True))
      (value, outputs), gradients = compiled(engine_leaves)
      (repeat_value, repeat_outputs), repeat_gradients = compiled(engine_leaves)
    self.assertTrue(np.isfinite(np.asarray(value)))
    self.assertEqual(outputs[0].shape, (16, 2))
    self.assertEqual(outputs[1].shape, (16, 2))
    np.testing.assert_array_equal(np.asarray(value), np.asarray(repeat_value))
    for actual, repeated in zip(outputs, repeat_outputs, strict=True):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(repeated))
      self.assertTrue(np.isfinite(np.asarray(actual)).all())
    gradient_arrays = [np.asarray(value) for value in jax.tree.leaves(gradients)]
    repeat_arrays = [
        np.asarray(value) for value in jax.tree.leaves(repeat_gradients)
    ]
    self.assertGreater(sum(np.count_nonzero(value) for value in gradient_arrays), 0)
    for actual, repeated in zip(
        gradient_arrays, repeat_arrays, strict=True
    ):
      np.testing.assert_array_equal(actual, repeated)

  def test_dp16_logprob_pipeline_reassembles_all_local_m256_rows(self):
    if len(jax.devices()) != 64:
      self.skipTest("requires exactly 64 forced CPU or accelerator devices")
    mesh = jax.sharding.Mesh(
        np.asarray(jax.devices()).reshape(16, 4), ("data", "model")
    )
    env = {
        "CANON_P32_TRAIN_ADMITTED": "1",
        "CANON_DP_SIZE": "16",
        "CANON_TP_SIZE": "4",
        "CANON_LOGPROB_M": "256",
        "CANON_TARGET_M": "256",
        "MIN_TOKEN_BUCKET": "4096",
    }

    def return_logprobs(logprobs, next_tokens, max_logprobs):
      del next_tokens, max_logprobs
      return logprobs

    logits = jnp.arange(4096 * 8, dtype=jnp.float32).reshape(4096, 8)
    logits = jnp.mod(logits, 17.0) / 7.0
    def local_log_softmax(value):
      return value + jnp.float32(1.0)
    with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
        canonical_qwen3_adapter.canonical_logsoftmax,
        "log_softmax",
        local_log_softmax,
    ):
      scorer = canonical_qwen3_adapter._make_canonical_compute_and_gather(
          return_logprobs, mesh
      )
      actual = scorer(logits, jnp.zeros((4096,), jnp.int32), 1)
    expected = logits + jnp.float32(1.0)
    self.assertEqual(actual.shape, (4096, 8))
    self.assertEqual(
        actual.sharding.spec, jax.sharding.PartitionSpec("data")
    )
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

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

  def test_p28_segmented_forward_is_host_bounded_and_value_exact(self):
    runner = _SegmentedRunner()
    env = {"CANON_P28_SEGMENTED_FORWARD": "1"}
    with mock.patch.dict(os.environ, env, clear=False):
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
    token_ids = jnp.asarray([2, 4], jnp.int32)
    metadata = jnp.asarray(0.125, jnp.float32)
    expected_caches, expected_hidden = runner.model.model(
        runner.kv_caches, token_ids, metadata
    )
    actual_caches, actual_hidden = segmented.run(
        runner.state_leaves,
        runner.kv_caches,
        token_ids,
        metadata,
    )
    np.testing.assert_array_equal(actual_hidden, expected_hidden)
    for actual, expected in zip(actual_caches, expected_caches):
      np.testing.assert_array_equal(actual, expected)
    self.assertEqual(segmented.contract.block_depth, 1)
    self.assertEqual(segmented.contract.end_layer, 2)

    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError, "host boundary"
    ):
      jax.jit(
          lambda leaves, caches, ids, meta: segmented.run(
              leaves, caches, ids, meta
          )
      )(runner.state_leaves, runner.kv_caches, token_ids, metadata)

    with mock.patch.dict(
        os.environ, {"CANON_P28_SEGMENTED_FORWARD": "0"}, clear=False
    ):
      with self.assertRaisesRegex(
          canonical_qwen3_adapter.FunctionalMappingError,
          "CANON_P28_SEGMENTED_FORWARD=1",
      ):
        canonical_qwen3_adapter.build_p28_segmented_engine_forward(runner)

  def test_p28_isolated_block_vjp_is_exact_and_has_cache_cotangent(self):
    runner = _SegmentedRunner()
    with mock.patch.dict(
        os.environ, {"CANON_P28_SEGMENTED_FORWARD": "1"}, clear=False
    ):
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
    hidden = jnp.asarray([[1.0], [2.0]], jnp.float32)
    result = segmented.run_block_vjp(
        0,
        runner.state_leaves,
        jnp.asarray(0.75, jnp.float32),
        hidden,
        jnp.asarray(0.125, jnp.float32),
    )
    for actual, expected in zip(result["isolated"], result["reference"]):
      np.testing.assert_array_equal(actual, expected)
    explicit_state = list(runner.state_leaves)
    layer_leaf = segmented._local_layer_full_indices[0][0]  # pylint: disable=protected-access
    explicit_state[layer_leaf] = explicit_state[layer_leaf] + 0.25
    explicit_reference, explicit_isolated = segmented.run_block_forward(
        0,
        tuple(explicit_state),
        jnp.asarray(0.75, jnp.float32),
        hidden,
        jnp.asarray(0.125, jnp.float32),
    )
    for actual, expected in zip(explicit_isolated, explicit_reference):
      np.testing.assert_array_equal(actual, expected)
    self.assertFalse(
        bool(np.asarray(jnp.array_equal(explicit_isolated[1], result["isolated"][1])))
    )
    parameter_grads, cache_grad, hidden_grad = result["gradients"]
    self.assertTrue(any(np.any(np.asarray(x) != 0) for x in parameter_grads))
    self.assertNotEqual(float(cache_grad), 0.0)
    self.assertTrue(np.all(np.asarray(hidden_grad) != 0))
    self.assertEqual(result["contract"].layer_index, 0)
    self.assertEqual(result["contract"].block_depth, 1)

  def test_p28_two_segment_pullback_matches_monolithic_oracle(self):
    runner = _SegmentedRunner()
    with mock.patch.dict(
        os.environ, {"CANON_P28_SEGMENTED_FORWARD": "1"}, clear=False
    ):
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
    prefix_hidden = jnp.asarray([[0.5], [1.25]], jnp.float32)
    chunk_hidden = jnp.asarray([[1.5], [-0.75]], jnp.float32)
    metadata = jnp.asarray(0.125, jnp.float32)
    caches = tuple(runner.kv_caches)
    layer_leaves = segmented._local_layer_leaves  # pylint: disable=protected-access

    def objective(leaves, prefix_input, chunk_input, initial_caches):
      prefix_tape_caches = []
      hidden = prefix_input
      for index, layer_fn in enumerate(
          segmented._local_layer_fns  # pylint: disable=protected-access
      ):
        cache, hidden = layer_fn(
            leaves[index], initial_caches[index], hidden, metadata
        )
        prefix_tape_caches.append(cache)
      hidden = chunk_input
      for index, layer_fn in enumerate(
          segmented._local_layer_fns  # pylint: disable=protected-access
      ):
        _, hidden = layer_fn(
            leaves[index], prefix_tape_caches[index], hidden, metadata
        )
      seed = jnp.asarray([[0.25], [-0.5]], jnp.float32)
      return jnp.sum(hidden * seed)

    _, oracle = jax.value_and_grad(
        objective, argnums=(0, 1, 2, 3)
    )(layer_leaves, prefix_hidden, chunk_hidden, caches)

    prefix_tape = []
    prefix_caches = []
    hidden = prefix_hidden
    for index, layer_fn in enumerate(
        segmented._local_layer_fns  # pylint: disable=protected-access
    ):
      prefix_tape.append((caches[index], hidden))
      cache, hidden = layer_fn(
          layer_leaves[index], caches[index], hidden, metadata
      )
      prefix_caches.append(cache)
    chunk_tape = []
    hidden = chunk_hidden
    for index, layer_fn in enumerate(
        segmented._local_layer_fns  # pylint: disable=protected-access
    ):
      chunk_tape.append((prefix_caches[index], hidden))
      _, hidden = layer_fn(
          layer_leaves[index], prefix_caches[index], hidden, metadata
      )

    dhidden = jnp.asarray([[0.25], [-0.5]], jnp.float32)
    chunk_param_grads = [None] * len(layer_leaves)
    prefix_cache_cotangents = [None] * len(layer_leaves)
    for index in reversed(range(len(layer_leaves))):
      cache_in, hidden_in = chunk_tape[index]
      dcache_out = jnp.zeros_like(cache_in)
      grads, dcache, dhidden = segmented.run_block_pullback(
          index,
          cache_in,
          hidden_in,
          metadata,
          dcache_out,
          dhidden,
      )
      chunk_param_grads[index] = grads
      prefix_cache_cotangents[index] = dcache
    chunk_input_grad = dhidden

    dhidden = jnp.zeros_like(prefix_tape[-1][1])
    prefix_param_grads = [None] * len(layer_leaves)
    initial_cache_grads = [None] * len(layer_leaves)
    for index in reversed(range(len(layer_leaves))):
      cache_in, hidden_in = prefix_tape[index]
      grads, dcache, dhidden = segmented.run_block_pullback(
          index,
          cache_in,
          hidden_in,
          metadata,
          prefix_cache_cotangents[index],
          dhidden,
      )
      prefix_param_grads[index] = grads
      initial_cache_grads[index] = dcache
    combined = tuple(
        jax.tree.map(lambda a, b: a + b, prefix_grad, chunk_grad)
        for prefix_grad, chunk_grad in zip(
            prefix_param_grads, chunk_param_grads, strict=True
        )
    )
    staged = (combined, dhidden, chunk_input_grad, tuple(initial_cache_grads))
    for actual, expected in zip(
        jax.tree.leaves(staged), jax.tree.leaves(oracle), strict=True
    ):
      np.testing.assert_array_equal(actual, expected)

  def test_p28_full_loss_endpoints_cover_and_pull_back_full_state(self):
    runner = _SegmentedRunner()
    env = {
        "CANON_P28_SEGMENTED_FORWARD": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
    token_ids = jnp.asarray([2, 4], jnp.int32)
    embedded = segmented.run_embed_forward(token_ids)
    np.testing.assert_array_equal(
        embedded, runner.model.model.embed_tokens(token_ids)
    )
    hidden = jnp.asarray([[1.0], [-2.0]], jnp.float32)
    normalized = segmented.run_norm_forward(hidden)
    logits = segmented.run_head_forward(normalized)
    np.testing.assert_array_equal(normalized, runner.model.model.norm(hidden))
    np.testing.assert_array_equal(logits, runner.model.lm_head(normalized))
    explicit_state = list(runner.state_leaves)
    embed_leaf = segmented._embed_full_indices[0]  # pylint: disable=protected-access
    explicit_state[embed_leaf] = explicit_state[embed_leaf] + 0.5
    explicit_embedded = segmented.run_embed_forward(
        token_ids, state_leaves=tuple(explicit_state)
    )
    np.testing.assert_array_equal(
        explicit_embedded,
        token_ids[:, None].astype(jnp.float32) * explicit_state[embed_leaf],
    )

    dembed = segmented.run_embed_pullback(
        token_ids, jnp.asarray([[0.25], [-0.5]], jnp.float32)
    )
    dnorm, dhidden = segmented.run_norm_pullback(
        hidden, jnp.asarray([[0.125], [0.25]], jnp.float32)
    )
    dhead, dnormalized = segmented.run_head_pullback(
        normalized, jnp.asarray([[0.5], [-0.25]], jnp.float32)
    )
    self.assertTrue(all(np.any(np.asarray(x) != 0) for x in dembed))
    self.assertTrue(all(np.any(np.asarray(x) != 0) for x in dnorm))
    self.assertTrue(all(np.any(np.asarray(x) != 0) for x in dhead))
    self.assertTrue(np.any(np.asarray(dhidden) != 0))
    self.assertTrue(np.any(np.asarray(dnormalized) != 0))

    layer_grads = tuple(
        jax.tree.map(jnp.ones_like, leaves)
        for leaves in segmented._local_layer_leaves  # pylint: disable=protected-access
    )
    full = segmented.assemble_full_state_gradient(
        embed=jax.tree.map(jnp.ones_like, dembed),
        layers=layer_grads,
        norm=jax.tree.map(jnp.ones_like, dnorm),
        head=jax.tree.map(jnp.ones_like, dhead),
    )
    self.assertLen(full, len(runner.state_leaves))
    for leaf in full:
      np.testing.assert_array_equal(leaf, jnp.ones_like(leaf))

    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError, "host boundary"
    ):
      jax.jit(lambda values: segmented.assemble_full_state_gradient(
          embed=dembed, layers=values, norm=dnorm, head=dhead
      ))(layer_grads)

  def test_p30_sparse_gradient_assembly_matches_legacy_bits(self):
    runner = _SegmentedRunner()
    base_env = {
        "CANON_P28_SEGMENTED_FORWARD": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
    }
    with mock.patch.dict(os.environ, base_env, clear=False):
      legacy = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
    with mock.patch.dict(
        os.environ,
        {**base_env, "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1"},
        clear=False,
    ):
      sparse = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )

    def signed_values(leaves):
      values = []
      for index, leaf in enumerate(leaves):
        value = jnp.full_like(leaf, index + 1)
        if value.size:
          value = value.reshape((-1,)).at[0].set(
              jnp.asarray(-0.0, value.dtype)
          ).reshape(value.shape)
        values.append(value)
      return tuple(values)

    embed = signed_values(legacy._embed_local_leaves)  # pylint: disable=protected-access
    layers = tuple(
        signed_values(leaves)
        for leaves in legacy._local_layer_leaves  # pylint: disable=protected-access
    )
    norm = signed_values(legacy._norm_local_leaves)  # pylint: disable=protected-access
    head = signed_values(legacy._head_local_leaves)  # pylint: disable=protected-access
    legacy_full = legacy.assemble_full_state_gradient(
        embed=embed, layers=layers, norm=norm, head=head
    )
    sparse_full = sparse.assemble_full_state_gradient(
        embed=embed, layers=layers, norm=norm, head=head
    )
    for actual, expected in zip(sparse_full, legacy_full, strict=True):
      np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
      self.assertEqual(
          hashlib.sha256(np.asarray(actual).tobytes()).hexdigest(),
          hashlib.sha256(np.asarray(expected).tobytes()).hexdigest(),
      )

  def test_p28_complete_loss_schedule_matches_monolithic_oracle(self):
    runner = _CompleteSegmentedRunner()
    adapter = object.__new__(
        canonical_qwen3_adapter.Qwen3EngineForwardAdapter
    )
    adapter._runner = runner  # pylint: disable=protected-access
    adapter._bucket = 256  # pylint: disable=protected-access
    adapter._max_model_len = 4096  # pylint: disable=protected-access
    adapter._max_num_reqs = 1  # pylint: disable=protected-access
    adapter._blocks_per_req = 16  # pylint: disable=protected-access
    adapter._engine_state_contract = runner.state  # pylint: disable=protected-access
    adapter._key_mappings = {}  # pylint: disable=protected-access
    adapter._transpose_keys = None  # pylint: disable=protected-access
    adapter._hook_fns = None  # pylint: disable=protected-access
    adapter._tp_size = 1  # pylint: disable=protected-access
    adapter._set_forward_context = (  # pylint: disable=protected-access
        lambda *_: contextlib.nullcontext()
    )
    adapter._fresh_caches = types.MethodType(  # pylint: disable=protected-access
        lambda self: [jnp.asarray(0.0), jnp.asarray(0.0)], adapter
    )

    def chunk_inputs(self, spec, chunk_index):
      start = chunk_index * self._bucket
      return (
          spec["packed_ids"][start : start + self._bucket],
          spec["next_ids"][start : start + self._bucket],
          jnp.asarray(0.125, jnp.float32),
      )

    adapter._p28_chunk_inputs = types.MethodType(  # pylint: disable=protected-access
        chunk_inputs, adapter
    )

    def processed_rows(logits, target_ids, temperature):
      normalized = jax.nn.log_softmax(logits / temperature, axis=-1)
      selected = jnp.take_along_axis(
          normalized, target_ids[:, None], axis=-1
      )[:, 0]
      probabilities = jnp.exp(normalized)
      entropy = -jnp.sum(probabilities * normalized, axis=-1)
      return selected, entropy

    def processed_rows_pullback(
        logits, target_ids, temperature, dlogps, dentropy
    ):
      _, pullback = jax.vjp(
          lambda values: processed_rows(values, target_ids, temperature),
          logits,
      )
      return pullback((dlogps, dentropy))[0]

    adapter._p28_processed_rows_fn = jax.jit(  # pylint: disable=protected-access
        processed_rows
    )
    adapter._p28_processed_rows_pullback_fn = jax.jit(  # pylint: disable=protected-access
        processed_rows_pullback
    )
    adapter.map_engine_cotangents_to_trainer_state = types.MethodType(
        lambda self, state, cotangents: tuple(cotangents), adapter
    )

    token_row = (jnp.arange(2048, dtype=jnp.int32) % 3)
    prompt_ids = jnp.broadcast_to(token_row[None, :], (8, 2048))
    prompt_mask = jnp.broadcast_to(
        (jnp.arange(2048) < 260)[None, :], (8, 2048)
    )
    completion_ids = jnp.broadcast_to(
        (jnp.arange(64, dtype=jnp.int32) % 3)[None, :], (8, 64)
    )
    completion_mask = jnp.broadcast_to(
        (jnp.arange(64) < 4)[None, :], (8, 64)
    )
    train_example = types.SimpleNamespace(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        old_per_token_logps=jnp.zeros((8, 64), jnp.float32),
        ref_per_token_logps=None,
        advantages=jnp.asarray(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0],
            dtype=jnp.float32,
        ),
        sampler_is_weights=None,
        segment_ids=None,
    )
    algo_config = types.SimpleNamespace(
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
    engine_leaves = tuple(runner.state_leaves)
    env = {
        "CANON_P28_SEGMENTED_FORWARD": "1",
        "CANON_P28_SEGMENTED_TRAIN": "1",
        "CANON_P28_G5C_ONLY": "1",
    }
    mapped = canonical_qwen3_adapter.FunctionalEngineLeaves(
        paths=(), leaves=engine_leaves, source_to_target=()
    )
    with (
        mock.patch.dict(os.environ, env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      result = adapter.segmented_grpo_value_and_grad(
          trainer_state=engine_leaves,
          train_example=train_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
      )
      segmented = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
          runner
      )
      specs = tuple(
          adapter._p28_sequence_spec(  # pylint: disable=protected-access
              prompt_ids[index],
              completion_ids[index],
              prompt_mask[index],
              completion_mask[index],
              1.0,
          )
          for index in range(8)
      )

      def oracle(leaves):
        batch_logps = []
        batch_entropy = []
        for spec in specs:
          caches = (jnp.asarray(0.0), jnp.asarray(0.0))
          chunk_logps = []
          chunk_entropy = []
          for chunk_index in range(spec["num_chunks"]):
            ids, targets, metadata = chunk_inputs(
                adapter, spec, chunk_index
            )
            embed_leaves = tuple(
                leaves[i] for i in segmented._embed_full_indices  # pylint: disable=protected-access
            )
            hidden = segmented._embed_local_fn(  # pylint: disable=protected-access
                embed_leaves, ids
            )
            next_caches = []
            for layer_index, cache in enumerate(caches):
              local_leaves = tuple(
                  leaves[i]
                  for i in segmented._local_layer_full_indices[layer_index]  # pylint: disable=protected-access
              )
              cache, hidden = segmented._local_layer_fns[layer_index](  # pylint: disable=protected-access
                  local_leaves, cache, hidden, metadata
              )
              next_caches.append(cache)
            caches = tuple(next_caches)
            norm_leaves = tuple(
                leaves[i] for i in segmented._norm_full_indices  # pylint: disable=protected-access
            )
            normalized = segmented._norm_local_fn(  # pylint: disable=protected-access
                norm_leaves, hidden
            )
            head_leaves = tuple(
                leaves[i] for i in segmented._head_full_indices  # pylint: disable=protected-access
            )
            logits = segmented._head_local_fn(  # pylint: disable=protected-access
                head_leaves, normalized
            ).astype(jnp.float32)
            logps, entropy = processed_rows(logits, targets, 1.0)
            chunk_logps.append(logps)
            chunk_entropy.append(entropy)
          flat_logps = jnp.concatenate(chunk_logps)
          flat_entropy = jnp.concatenate(chunk_entropy)
          batch_logps.append(jnp.where(
              spec["completion_valid"],
              flat_logps[spec["source_rows"]],
              0.0,
          ))
          batch_entropy.append(jnp.where(
              spec["completion_valid"],
              flat_entropy[spec["source_rows"]],
              0.0,
          ))
        logps = jnp.stack(batch_logps)
        entropy = jnp.stack(batch_entropy)
        loss_output = algo_core.grpo_loss_from_precomputed_logps(
            logps, entropy, train_example, algo_config
        )
        return loss_output.primary_loss.compute(), logps

      (expected_loss, expected_logps), expected_grads = jax.value_and_grad(
          oracle, has_aux=True
      )(engine_leaves)

    np.testing.assert_array_equal(result["per_token_logps"], expected_logps)
    np.testing.assert_allclose(result["loss"], expected_loss, rtol=1e-6)
    for actual, expected in zip(
        jax.tree.leaves(result["gradients"]),
        jax.tree.leaves(expected_grads),
        strict=True,
    ):
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    self.assertLen(result["reports"], 8)
    self.assertEqual(
        tuple(report["boundary"] for report in result["reports"]),
        ("pending",) * 7 + ("final",),
    )
    expected_mapping_leaves = len(jax.tree.leaves(engine_leaves))
    self.assertTrue(all(
        report["trainer_gradient"]["mapping_adjoint_leaves"]
        == expected_mapping_leaves
        for report in result["reports"]
    ))
    for report in result["reports"][:6]:
      self.assertEqual(report["loss_cotangent"]["nonzero"], 0)
      self.assertEqual(report["trainer_gradient"]["nonzero"], 0)
      self.assertTrue(all(
          group["nonzero"] == 0
          for group in report["engine_groups"].values()
      ))
    for report in result["reports"][6:]:
      self.assertGreater(report["loss_cotangent"]["nonzero"], 0)
      self.assertGreater(report["trainer_gradient"]["nonzero"], 0)
      self.assertTrue(all(
          group["nonzero"] > 0
          for group in report["engine_groups"].values()
      ))

    streamed = []
    g6_env = dict(env)
    g6_env["CANON_P28_G5C_ONLY"] = "0"
    g6_env["CANON_P28_G6_UPDATE"] = "1"
    g6_env["CANON_P30_REUSE_SEGMENTED_ENGINE"] = "1"
    g6_env["CANON_P30_RELEASE_CAPTURED_STATE"] = "1"
    with (
        mock.patch.dict(os.environ, g6_env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      streamed_result = adapter.segmented_grpo_value_and_grad(
          trainer_state=engine_leaves,
          train_example=train_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
          gradient_microbatch_sink=lambda index, gradient: streamed.append(
              (index, gradient)
          ),
      )
    cached_segmented = adapter._p30_segmented_engine  # pylint: disable=protected-access
    self.assertTrue(cached_segmented._captured_state_released)  # pylint: disable=protected-access
    self.assertTrue(all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaf in cached_segmented._full_state_leaves  # pylint: disable=protected-access
    ))
    self.assertTrue(all(
        isinstance(leaf, jax.ShapeDtypeStruct)
        for leaves in cached_segmented._local_layer_leaves  # pylint: disable=protected-access
        for leaf in leaves
    ))
    with self.assertRaisesRegex(
        canonical_qwen3_adapter.FunctionalMappingError,
        "requires explicit current state after captured state was released",
    ):
      cached_segmented.run_embed_forward(jnp.zeros((1,), jnp.int32))
    self.assertIsNone(streamed_result["gradients"])
    self.assertEqual(streamed_result["gradient_microbatches"], 4)
    self.assertEqual([index for index, _ in streamed], [0, 1, 2, 3])
    averaged = jax.tree.map(
        lambda *values: sum(values) / 4.0,
        *(gradient for _, gradient in streamed),
    )
    for actual, expected in zip(
        jax.tree.leaves(averaged),
        jax.tree.leaves(result["gradients"]),
        strict=True,
    ):
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    streamed_pairs = []
    with (
        mock.patch.dict(os.environ, g6_env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      pair_result = adapter.segmented_grpo_value_and_grad(
          trainer_state=engine_leaves,
          train_example=train_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
          gradient_pair_sink=(
              lambda index, left, right, multiplier: streamed_pairs.append(
                  (index, left, right, multiplier)
              )
          ),
      )
    self.assertIs(
        adapter._p30_segmented_engine, cached_segmented  # pylint: disable=protected-access
    )
    self.assertEqual(pair_result["gradient_microbatches"], 4)
    self.assertEqual([item[0] for item in streamed_pairs], [0, 1, 2, 3])
    for (_, left, right, multiplier), (_, expected) in zip(
        streamed_pairs, streamed, strict=True
    ):
      actual = jax.tree.map(
          lambda a, b: (a + b) * multiplier.astype(a.dtype), left, right
      )
      for actual_leaf, expected_leaf in zip(
          jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
      ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)

    # P31 keeps the same two-trajectory math but grows the real batch from
    # 8 to 32 trajectories and therefore streams 16 accumulator steps.
    p31_prompt_ids = jnp.pad(
        jnp.tile(prompt_ids, (4, 1)), ((0, 0), (0, 2048))
    )
    p31_prompt_mask = jnp.pad(
        jnp.tile(prompt_mask, (4, 1)), ((0, 0), (0, 2048))
    )
    p31_completion_ids = jnp.pad(
        jnp.tile(completion_ids, (4, 1)), ((0, 0), (0, 1984))
    )
    p31_completion_mask = jnp.pad(
        jnp.tile(completion_mask, (4, 1)), ((0, 0), (0, 1984))
    )
    # Positions 1 and 3 model environment/parser tokens: they do not
    # contribute to the policy loss but must remain in the causal sequence.
    p31_action_mask = p31_completion_mask.at[:, 1].set(False)
    p31_action_mask = p31_action_mask.at[:, 3].set(False)
    p31_example = types.SimpleNamespace(
        prompt_ids=p31_prompt_ids,
        prompt_mask=p31_prompt_mask,
        completion_ids=p31_completion_ids,
        completion_mask=p31_action_mask,
        completion_valid_mask=p31_completion_mask,
        old_per_token_logps=jnp.zeros((32, 2048), jnp.float32),
        ref_per_token_logps=None,
        advantages=jnp.tile(
            jnp.asarray([-1.0, 1.0], jnp.float32), 16
        ),
        sampler_is_weights=None,
        segment_ids=None,
    )
    p31_aggregate_env = {
        **env,
        "CANON_P31_CONVERGENCE": "1",
        "CANON_P28_G5C_ONLY": "1",
        "CANON_P28_G6_UPDATE": "0",
        "CANON_P30_REUSE_SEGMENTED_ENGINE": "0",
        "CANON_P30_RELEASE_CAPTURED_STATE": "0",
    }
    with (
        mock.patch.dict(os.environ, p31_aggregate_env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      p31_aggregate = adapter.segmented_grpo_value_and_grad(
          trainer_state=engine_leaves,
          train_example=p31_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
      )
    p31_streamed = []
    p31_stream_env = {
        **p31_aggregate_env,
        "CANON_P28_G5C_ONLY": "0",
        "CANON_P28_G6_UPDATE": "1",
    }
    with (
        mock.patch.dict(os.environ, p31_stream_env, clear=False),
        mock.patch.object(
            canonical_qwen3_adapter,
            "map_trainer_state_to_engine_leaves",
            return_value=mapped,
        ),
    ):
      p31_stream = adapter.segmented_grpo_value_and_grad(
          trainer_state=engine_leaves,
          train_example=p31_example,
          algo_config=algo_config,
          pad_id=0,
          eos_id=2,
          gradient_microbatch_sink=lambda index, gradient: p31_streamed.append(
              (index, gradient)
          ),
      )
    self.assertEqual(p31_stream["gradient_microbatches"], 16)
    self.assertLen(p31_stream["reports"], 32)
    self.assertEqual([index for index, _ in p31_streamed], list(range(16)))
    self.assertTrue(bool(np.asarray(jnp.all(
        p31_aggregate["per_token_logps"][:, 1] != 0.0
    ))))
    self.assertTrue(bool(np.asarray(jnp.all(
        p31_aggregate["per_token_logps"][:, 3] != 0.0
    ))))
    p31_average = jax.tree.map(
        lambda *values: sum(values) / 16.0,
        *(gradient for _, gradient in p31_streamed),
    )
    for actual, expected in zip(
        jax.tree.leaves(p31_average),
        jax.tree.leaves(p31_aggregate["gradients"]),
        strict=True,
    ):
      np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

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
        "CANON_P28_SEGMENTED_TRAIN": "1",
    }
    with mock.patch.dict(os.environ, env, clear=False):
      adapter = canonical_qwen3_adapter.Qwen3EngineForwardAdapter(
          sampler=sampler
      )

      mapped = self._map(
          source["trainer"]["w"].value,
          source["trainer"]["n"].value,
      ).leaves
      cotangents = tuple(
          jnp.arange(value.size, dtype=value.dtype).reshape(value.shape) + 1
          for value in mapped
      )

      def mapped_objective(state):
        values = canonical_qwen3_adapter.map_trainer_state_to_engine_leaves(
            trainer_state=state,
            engine_state_contract=self.target,
            key_mappings=self.mapping,
            transpose_keys={"w": (1, 0)},
        ).leaves
        return sum(
            jnp.sum(value * cotangent)
            for value, cotangent in zip(values, cotangents, strict=True)
        )

      expected_mapping_grad = jax.grad(mapped_objective)(source)
      actual_mapping_grad = adapter.map_engine_cotangents_to_trainer_state(
          source, cotangents
      )
      for actual, expected in zip(
          jax.tree.leaves(actual_mapping_grad),
          jax.tree.leaves(expected_mapping_grad),
          strict=True,
      ):
        np.testing.assert_array_equal(actual, expected)

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
