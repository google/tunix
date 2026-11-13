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

"""Tests for TcpDestinationDelegate and manifest-based HF metadata derivation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import dataclasses

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import tcp_delegate
from tunix.dist_rl.weight_sync import tcp_handler
from tunix.models import hf_mapping
from tunix.models.qwen3 import mapping_vllm_jax


@dataclasses.dataclass
class _FakeQwen3HfConfig:
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool


class _FakeHfSampler:
  """Typed test double implementing `HfWeightUpdatableSampler`."""

  def __init__(
      self,
      spec: hf_mapping.WeightMappingSpec,
      *,
      error: Exception | None = None,
  ):
    self._spec = spec
    self.error = error
    self.received: list[dict[str, jax.Array]] = []
    self.delete_cache_calls = 0
    self.reinit_cache_calls = 0

  def from_hf_manifest(self) -> hf_mapping.WeightMappingSpec:
    return self._spec

  def update_params_from_hf(
      self, hf_weights: Mapping[str, jax.Array]
  ) -> None:
    if self.error is not None:
      raise self.error
    self.received.append(dict(hf_weights))

  def delete_cache(self) -> None:
    self.delete_cache_calls += 1

  def reinitialize_cache(self) -> None:
    self.reinit_cache_calls += 1


def _identity_spec(name: str) -> hf_mapping.WeightMappingSpec:
  return hf_mapping.WeightMappingSpec(
      rules=(
          hf_mapping.ParamRule(
              source_keys=(name,),
              target_keys=(name,),
              ops=(),
          ),
      ),
      model_type="identity",
  )


def _qwen3_vllm_jax_state(
    cfg: _FakeQwen3HfConfig,
    *,
    tp_size: int,
    dtype: jax.typing.DTypeLike = jnp.float32,
) -> dict[str, jax.Array]:
  v_pad = ((cfg.vocab_size + tp_size - 1) // tp_size) * tp_size
  hd = cfg.head_dim or (cfg.hidden_size // cfg.num_attention_heads)
  h_pad = 64 if hd == 64 else ((hd + 127) // 128) * 128
  nq = cfg.num_attention_heads
  nkv = cfg.num_key_value_heads or nq
  rep_q = tp_size // nq if nq < tp_size else 1
  rep_kv = tp_size // nkv if nkv < tp_size else 1
  d = cfg.hidden_size
  inter = cfg.intermediate_size

  state: dict[str, jax.Array] = {
      "model.embed_tokens.weight": jnp.zeros((v_pad, d), dtype=dtype),
      "model.norm.weight": jnp.zeros((d,), dtype=dtype),
  }
  for layer_idx in range(cfg.num_hidden_layers):
    p = f"model.layers.{layer_idx}"
    state.update({
        f"{p}.input_layernorm.weight": jnp.zeros((d,), dtype=dtype),
        f"{p}.post_attention_layernorm.weight": jnp.zeros((d,), dtype=dtype),
        f"{p}.self_attn.q_norm.weight": jnp.zeros((h_pad,), dtype=dtype),
        f"{p}.self_attn.k_norm.weight": jnp.zeros((h_pad,), dtype=dtype),
        f"{p}.self_attn.q_proj.weight": jnp.zeros(
            (d, nq * rep_q, h_pad), dtype=dtype
        ),
        f"{p}.self_attn.k_proj.weight": jnp.zeros(
            (d, nkv * rep_kv, h_pad), dtype=dtype
        ),
        f"{p}.self_attn.v_proj.weight": jnp.zeros(
            (d, nkv * rep_kv, h_pad), dtype=dtype
        ),
        f"{p}.self_attn.o_proj.weight": jnp.zeros(
            (nq * rep_q, h_pad, d), dtype=dtype
        ),
        f"{p}.mlp.gate_proj.weight": jnp.zeros((d, inter), dtype=dtype),
        f"{p}.mlp.up_proj.weight": jnp.zeros((d, inter), dtype=dtype),
        f"{p}.mlp.down_proj.weight": jnp.zeros((inter, d), dtype=dtype),
    })
  if not cfg.tie_word_embeddings:
    state["lm_head.weight"] = jnp.zeros((d, v_pad), dtype=dtype)
  return state


class TcpDestinationDelegateTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.delegate = tcp_delegate.TcpDestinationDelegate(
        server_id="rollout_0",
        worker_index=0,
    )

  def tearDown(self):
    self.delegate.close()
    super().tearDown()

  def test_bind_and_metadata(self):
    sampler = _FakeHfSampler(_identity_spec("model.layers.0.weight"))
    state = {"model.layers.0.weight": np.zeros((4, 4), dtype=np.float32)}

    res = asyncio.run(
        self.delegate.bind_weight_sync(sampler=sampler, state=state)
    )
    self.assertTrue(res)
    self.assertTrue(self.delegate.is_bounded())

    meta = asyncio.run(self.delegate.get_weight_sync_metadata())
    self.assertLen(meta, 1)
    self.assertEqual(meta[0].unit.job_name, "rollout_0")
    self.assertEqual(meta[0].transport_mode, "tcp")
    self.assertEqual(meta[0].variables[0].name, "model.layers.0.weight")
    self.assertEqual(meta[0].variables[0].shape, (4, 4))

  def test_bind_rejects_invalid_sampler_or_missing_state(self):
    with self.assertRaises(TypeError):
      asyncio.run(
          self.delegate.bind_weight_sync(sampler=None, state={"w": np.zeros(1)})
      )
    with self.assertRaises(TypeError):
      asyncio.run(
          self.delegate.bind_weight_sync(
              sampler=object(), state={"w": np.zeros(1)}
          )
      )
    sampler = _FakeHfSampler(_identity_spec("w"))
    with self.assertRaisesRegex(ValueError, "state is required"):
      asyncio.run(self.delegate.bind_weight_sync(sampler=sampler, state=None))

  def test_derive_hf_variables_from_manifest_qwen3(self):
    cfg_tied = _FakeQwen3HfConfig(
        vocab_size=1003,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        tie_word_embeddings=True,
        attention_bias=False,
    )
    spec_tied = mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
        cfg_tied, tp_size=4
    )
    flat_tied = _qwen3_vllm_jax_state(cfg_tied, tp_size=4)
    # Build an nnx.State containing an extra unmapped "rng" leaf.
    nested: dict[str, object] = {"rng": nnx.Param(jnp.zeros((2,)))}
    for dot_key, arr in flat_tied.items():
      cur: dict[str, object] = nested
      parts = dot_key.split(".")
      for p in parts[:-1]:
        child = cur.get(p)
        if not isinstance(child, dict):
          child = {}
          cur[p] = child
        cur = child
      cur[parts[-1]] = nnx.Param(arr)
    nnx_state_tied = nnx.State(nested)

    vars_tied = tcp_delegate.derive_hf_variables_from_manifest(
        spec_tied, nnx_state_tied
    )
    names_tied = {v.name: v for v in vars_tied}
    self.assertNotIn("rng", names_tied)
    self.assertIn("model.embed_tokens.weight", names_tied)
    self.assertNotIn("lm_head.weight", names_tied)
    self.assertEqual(
        names_tied["model.embed_tokens.weight"].shape, (1003, 128)
    )
    self.assertEqual(
        names_tied["model.layers.0.self_attn.q_norm.weight"].shape, (32,)
    )
    self.assertEqual(
        names_tied["model.layers.0.self_attn.q_proj.weight"].shape, (128, 128)
    )
    self.assertEqual(
        names_tied["model.layers.0.self_attn.k_proj.weight"].shape, (64, 128)
    )
    self.assertEqual(
        names_tied["model.layers.0.self_attn.o_proj.weight"].shape, (128, 128)
    )
    self.assertEqual(
        names_tied["model.layers.0.mlp.down_proj.weight"].shape, (128, 256)
    )

    # Case B: tie_word_embeddings=False retains lm_head.weight.
    cfg_untied = dataclasses.replace(cfg_tied, tie_word_embeddings=False)
    spec_untied = mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
        cfg_untied, tp_size=4
    )
    flat_untied = _qwen3_vllm_jax_state(cfg_untied, tp_size=4)
    vars_untied = tcp_delegate.derive_hf_variables_from_manifest(
        spec_untied, flat_untied
    )
    names_untied = {v.name: v for v in vars_untied}
    self.assertIn("model.embed_tokens.weight", names_untied)
    self.assertIn("lm_head.weight", names_untied)
    self.assertEqual(names_untied["lm_head.weight"].shape, (1003, 128))

    # Missing required target key fails fast.
    incomplete = dict(flat_untied)
    incomplete.pop("lm_head.weight")
    with self.assertRaisesRegex(KeyError, "lm_head.weight"):
      tcp_delegate.derive_hf_variables_from_manifest(spec_untied, incomplete)

  def test_tcp_destination_delegate_does_not_touch_kv_cache(self):
    sampler = _FakeHfSampler(_identity_spec("w"))
    self.assertIsInstance(sampler, tcp_delegate.HfWeightUpdatableSampler)

    asyncio.run(
        self.delegate.bind_weight_sync(
            sampler=sampler,
            state={"w": np.zeros((2, 2), dtype=np.uint16)},
        )
    )

    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r1", 1)] = {
        "w": tcp_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
        )
    }
    asyncio.run(self.delegate.weight_sync(req))
    asyncio.run(self.delegate.post_weight_sync(req))

    self.assertLen(sampler.received, 1)
    self.assertEqual(sampler.delete_cache_calls, 0)
    self.assertEqual(sampler.reinit_cache_calls, 0)

  def test_full_lifecycle_and_corruption_invariant(self):
    sampler = _FakeHfSampler(_identity_spec("w"))

    asyncio.run(
        self.delegate.bind_weight_sync(
            sampler=sampler,
            state={"w": np.zeros((2, 2), dtype=np.uint16)},
        )
    )

    # 1. Pre weight sync
    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r1", "uuid": 1},
        policy_version=10,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))

    # Manually inject round into receiver
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r1", 1)] = {
        "w": tcp_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
        )
    }

    # 2. Weight sync
    v = asyncio.run(self.delegate.weight_sync(req))
    self.assertEqual(v, 10)
    self.assertLen(sampler.received, 1)
    self.assertEqual(
        sampler.received[0]["w"].devices(), {jax.devices("cpu")[0]}
    )

    # 3. Post weight sync
    asyncio.run(self.delegate.post_weight_sync(req))
    status = self.delegate.get_weight_sync_status()
    self.assertEqual(status["phase"], "committed")

    # Invariant 5: Fail during update_params_from_hf sets serving_corrupted
    req2 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r2", "uuid": 2},
        policy_version=11,
    )
    asyncio.run(self.delegate.pre_weight_sync(req2))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r2", 2)] = {
        "w": tcp_handler.RawTensorPayload(
            buffer=np.ones((2, 2), dtype=np.uint16), shape=(2, 2), dtype="bf16"
        )
    }
    sampler.error = RuntimeError("TPU HBM corrupted!")

    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.weight_sync(req2))

    self.assertTrue(self.delegate.serving_corrupted)

    # Abort must now fail and refuse to clear corruption
    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.abort_weight_sync(req2))

    # New round must also be refused
    req3 = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r3", "uuid": 3},
        policy_version=12,
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(self.delegate.pre_weight_sync(req3))

  def test_phase_methods_reject_invalid_sync_request_without_tracker_pollution(
      self,
  ):
    initial_status = self.delegate.get_weight_sync_status()
    invalid_requests = [
        None,
        datatypes.WeightSyncRequest(request_id="123", extra_config={}),
        datatypes.WeightSyncRequest(extra_config={"req_id": "r_no_uuid"}),
        datatypes.WeightSyncRequest(
            extra_config={"req_id": "r_zero_uuid", "uuid": 0}
        ),
    ]
    for bad_req in invalid_requests:
      for phase_fn in (
          self.delegate.pre_weight_sync,
          self.delegate.weight_sync,
          self.delegate.post_weight_sync,
          self.delegate.abort_weight_sync,
      ):
        with self.assertRaises(ValueError):
          asyncio.run(phase_fn(bad_req))
        self.assertEqual(
            self.delegate.get_weight_sync_status(), initial_status
        )

  def test_weight_sync_preserves_uint16_and_bitcasts_bf16(self):
    spec = hf_mapping.WeightMappingSpec(
        rules=(
            hf_mapping.ParamRule(
                source_keys=("u16",), target_keys=("u16",), ops=()
            ),
            hf_mapping.ParamRule(
                source_keys=("bf16",), target_keys=("bf16",), ops=()
            ),
            hf_mapping.ParamRule(
                source_keys=("bfloat16",), target_keys=("bfloat16",), ops=()
            ),
        ),
        model_type="identity",
    )
    sampler = _FakeHfSampler(spec)
    state = {
        "u16": np.zeros((1,), dtype=np.uint16),
        "bf16": np.zeros((1,), dtype=jnp.bfloat16),
        "bfloat16": np.zeros((1,), dtype=jnp.bfloat16),
    }
    asyncio.run(self.delegate.bind_weight_sync(sampler=sampler, state=state))

    req = datatypes.WeightSyncRequest(
        extra_config={"req_id": "r_dtypes", "uuid": 1},
        policy_version=1,
    )
    asyncio.run(self.delegate.pre_weight_sync(req))
    assert self.delegate._receiver is not None
    self.delegate._receiver._staged_rounds[("r_dtypes", 1)] = {
        "u16": tcp_handler.RawTensorPayload(
            buffer=np.array([0x3F80], dtype=np.uint16),
            shape=(1,),
            dtype="uint16",
        ),
        "bf16": tcp_handler.RawTensorPayload(
            buffer=np.array([0x3F80], dtype=np.uint16),
            shape=(1,),
            dtype="bf16",
        ),
        "bfloat16": tcp_handler.RawTensorPayload(
            buffer=np.array([0x4000], dtype=np.uint16),
            shape=(1,),
            dtype="bfloat16",
        ),
    }
    asyncio.run(self.delegate.weight_sync(req))
    asyncio.run(self.delegate.post_weight_sync(req))

    self.assertLen(sampler.received, 1)
    received = sampler.received[0]
    self.assertEqual(received["u16"].dtype, jnp.uint16)
    self.assertEqual(int(received["u16"][0]), 0x3F80)
    self.assertEqual(received["bf16"].dtype, jnp.bfloat16)
    self.assertEqual(float(received["bf16"][0]), 1.0)
    self.assertEqual(received["bfloat16"].dtype, jnp.bfloat16)
    self.assertEqual(float(received["bfloat16"][0]), 2.0)


if __name__ == "__main__":
  absltest.main()
