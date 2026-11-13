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

"""End-to-end test for TorchTitan TrainerWorker -> basic weight sync."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import dataclasses
import json
import os
from typing import Any

from absl.testing import absltest
import jax
from jax import typing as jax_typing
import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch
from torchtitan.experiments.tpu import qwen3 as tpu_qwen3
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.testing import worker_registry_testing
from tunix.dist_rl.weight_sync import basic_delegate
from tunix.dist_rl.weight_sync import basic_handler
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import abstract_worker
from tunix.experimental.torchtitan import model_registry
from tunix.experimental.torchtitan.engine import TorchTitanEngine
from tunix.experimental.torchtitan.engine import TorchTitanEngineConfig
from tunix.experimental.torchtitan.worker import TorchTitanTrainerWorker
from tunix.generate import vllm_hf_manifest
from tunix.generate.vllm_hf_manifests import qwen3 as qwen3_manifest
from tunix.models import hf_mapping


@dataclasses.dataclass
class _Qwen3HfConfig:
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool
  model_type: str = "qwen3"


def _qwen3_hf_config_from_titan(
    titan_cfg: Any,
) -> _Qwen3HfConfig:
  layer0 = titan_cfg.layers[0]
  attn = layer0.attention
  ffn = layer0.feed_forward
  n_heads = attn.n_heads
  n_kv_heads = n_heads if attn.n_kv_heads is None else attn.n_kv_heads
  head_dim = (
      attn.head_dim if attn.head_dim is not None else (titan_cfg.dim // n_heads)
  )
  return _Qwen3HfConfig(
      vocab_size=titan_cfg.vocab_size,
      hidden_size=titan_cfg.dim,
      intermediate_size=ffn.w1.out_features,
      num_hidden_layers=len(titan_cfg.layers),
      num_attention_heads=n_heads,
      num_key_value_heads=n_kv_heads,
      head_dim=head_dim,
      tie_word_embeddings=bool(titan_cfg.enable_weight_tying),
      attention_bias=False,
  )


class FakeVllmQwen3Sampler:
  """Typed test double implementing `HfWeightUpdatableSampler`."""

  def __init__(
      self,
      hf_cfg: _Qwen3HfConfig,
      *,
      tp_size: int,
      dtype: jax_typing.DTypeLike,
  ):
    self.hf_cfg = hf_cfg
    self.tp_size = tp_size
    self._manifest = vllm_hf_manifest.build_from_hf_manifest(
        hf_cfg, tp_size=tp_size
    )
    self.received_hf_weights: dict[str, jax.Array] | None = None

    v_pad = ((hf_cfg.vocab_size + tp_size - 1) // tp_size) * tp_size
    hd = hf_cfg.head_dim or (hf_cfg.hidden_size // hf_cfg.num_attention_heads)
    h_pad = 64 if hd == 64 else ((hd + 127) // 128) * 128
    nq = hf_cfg.num_attention_heads
    nkv = hf_cfg.num_key_value_heads or nq
    rep_q = tp_size // nq if nq < tp_size else 1
    rep_kv = tp_size // nkv if nkv < tp_size else 1
    d = hf_cfg.hidden_size
    inter = hf_cfg.intermediate_size

    self.state: dict[str, jax.Array] = {
        "model.embed_tokens.weight": jnp.zeros((v_pad, d), dtype=dtype),
        "model.norm.weight": jnp.zeros((d,), dtype=dtype),
    }
    for layer_idx in range(hf_cfg.num_hidden_layers):
      p = f"model.layers.{layer_idx}"
      self.state.update({
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
    if not hf_cfg.tie_word_embeddings:
      self.state["lm_head.weight"] = jnp.zeros((d, v_pad), dtype=dtype)

  def expected_hf_variables(self) -> tuple[weight_sync.TensorMetadata, ...]:
    return vllm_hf_manifest.derive_hf_variables_from_manifest(
        self._manifest, self.state
    )

  def update_params_from_hf(
      self,
      hf_weights: Mapping[
          str, basic_handler.RawTensorPayload | jax.Array | np.ndarray
      ],
      *,
      preserve_active_kv_cache: bool | None = None,
  ) -> None:
    del preserve_active_kv_cache
    local_cpus = jax.local_devices(backend="cpu")
    cpu_device = local_cpus[0] if local_cpus else jax.devices("cpu")[0]
    unpacked: dict[str, jax.Array] = {}
    with jax.default_device(cpu_device):
      for k, v in hf_weights.items():
        if isinstance(v, basic_handler.RawTensorPayload):
          if v.dtype in ("bf16", "bfloat16"):
            unpacked[k] = jax.lax.bitcast_convert_type(
                jnp.asarray(v.buffer), jnp.bfloat16
            )
          else:
            unpacked[k] = jnp.asarray(v.buffer)
        else:
          unpacked[k] = jnp.asarray(v)
    self.received_hf_weights = unpacked
    converted = hf_mapping.from_hf(
        json.dumps(self._manifest, sort_keys=True), unpacked
    )
    for k, arr in converted.items():
      self.state[k] = arr.astype(self.state[k].dtype)


def _jax_rms_norm(
    x: jax.Array, weight: jax.Array, eps: float = 1e-6
) -> jax.Array:
  x_f = x.astype(jnp.float32)
  mean_sq = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
  inv_rms = jax.lax.rsqrt(mean_sq + eps)
  return (x_f * inv_rms).astype(x.dtype) * weight


def _jax_qwen3_forward_vllm_layout(
    state: Mapping[str, jax.Array],
    cfg: _Qwen3HfConfig,
    tokens: np.ndarray,
    *,
    tp_size: int,
) -> np.ndarray:
  """Pure-JAX Qwen3 forward pass over vLLM-JAX 3D/transposed/padded weights."""
  v_size = cfg.vocab_size
  nq = cfg.num_attention_heads
  nkv = cfg.num_key_value_heads or nq
  hd = cfg.head_dim or (cfg.hidden_size // nq)
  h_pad = 64 if hd == 64 else ((hd + 127) // 128) * 128
  rep_q = tp_size // nq if nq < tp_size else 1
  rep_kv = tp_size // nkv if nkv < tp_size else 1
  gqa_groups = nq // nkv

  toks = jnp.asarray(tokens, dtype=jnp.int32)
  seq_len = toks.shape[1]
  embed = state["model.embed_tokens.weight"][:v_size, :]
  x = embed[toks]

  inv_freq = 1.0 / (
      1000000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / float(hd))
  )
  freqs = jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)
  emb = jnp.concatenate([freqs, freqs], axis=-1)
  cos = jnp.cos(emb)[None, :, None, :]
  sin = jnp.sin(emb)[None, :, None, :]

  def _rotate_half(t: jax.Array) -> jax.Array:
    return jnp.concatenate([-t[..., hd // 2 :], t[..., : hd // 2]], axis=-1)

  causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))[
      None, None, :, :
  ]

  for layer_idx in range(cfg.num_hidden_layers):
    p = f"model.layers.{layer_idx}"
    h = _jax_rms_norm(x, state[f"{p}.input_layernorm.weight"])

    q = jnp.einsum("BTD,DNH->BTNH", h, state[f"{p}.self_attn.q_proj.weight"])
    k = jnp.einsum("BTD,DKH->BTKH", h, state[f"{p}.self_attn.k_proj.weight"])
    v = jnp.einsum("BTD,DKH->BTKH", h, state[f"{p}.self_attn.v_proj.weight"])
    if rep_q > 1:
      q = q[:, :, ::rep_q, :]
    if rep_kv > 1:
      k = k[:, :, ::rep_kv, :]
      v = v[:, :, ::rep_kv, :]
    if hd < h_pad:
      q = q[:, :, :, :hd]
      k = k[:, :, :, :hd]
      v = v[:, :, :, :hd]

    q = _jax_rms_norm(q, state[f"{p}.self_attn.q_norm.weight"][:hd])
    k = _jax_rms_norm(k, state[f"{p}.self_attn.k_norm.weight"][:hd])
    q_f = q.astype(jnp.float32)
    k_f = k.astype(jnp.float32)
    q = (q_f * cos + _rotate_half(q_f) * sin).astype(q.dtype)
    k = (k_f * cos + _rotate_half(k_f) * sin).astype(k.dtype)

    k = jnp.repeat(k, gqa_groups, axis=2)
    v = jnp.repeat(v, gqa_groups, axis=2)
    scores = jnp.einsum("BTNH,BSNH->BNTS", q, k) / jnp.sqrt(float(hd))
    scores = jnp.where(causal_mask, scores, -1e9)
    probs = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(q.dtype)
    attn_out = jnp.einsum("BNTS,BSNH->BTNH", probs, v)
    if hd < h_pad:
      attn_out = jnp.pad(attn_out, ((0, 0), (0, 0), (0, 0), (0, h_pad - hd)))
    o_w = state[f"{p}.self_attn.o_proj.weight"]
    if rep_q > 1:
      attn_out = jnp.repeat(attn_out, rep_q, axis=2)
    x = x + jnp.einsum("BTNH,NHD->BTD", attn_out, o_w)

    h2 = _jax_rms_norm(x, state[f"{p}.post_attention_layernorm.weight"])
    gate = jax.nn.silu(
        jnp.einsum("BTD,DI->BTI", h2, state[f"{p}.mlp.gate_proj.weight"])
    )
    up = jnp.einsum("BTD,DI->BTI", h2, state[f"{p}.mlp.up_proj.weight"])
    x = x + jnp.einsum(
        "BTI,ID->BTD", gate * up, state[f"{p}.mlp.down_proj.weight"]
    )

  x = _jax_rms_norm(x, state["model.norm.weight"])
  if cfg.tie_word_embeddings:
    lm_w = state["model.embed_tokens.weight"][:v_size, :].T
  else:
    lm_w = state["lm_head.weight"][:, :v_size]
  return np.asarray(jnp.einsum("BTD,DV->BTV", x, lm_w), dtype=np.float32)


class RolloutWorkerTestDouble(abstract_worker.Worker):
  """Worker test double adapting BasicDestinationDelegate to WorkerRegistry."""

  def __init__(self, delegate: basic_delegate.BasicDestinationDelegate):
    self.delegate = delegate
    self.worker_id = "rollout_0"
    self.roles = frozenset({"rollout", "weight_sync"})

  def initialize(self) -> datatypes.Response:
    return datatypes.Response()

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    del dummy_data
    return datatypes.Response()

  def start(self) -> datatypes.Response:
    return datatypes.Response()

  def stop(self, *, abort: bool = False) -> datatypes.Response:
    del abort
    return datatypes.Response()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self.worker_id,
        roles=self.roles,
        resources={"weight_sync_mode": "basic"},
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=datatypes.WorkerState.READY)

  async def bind_weight_sync(self, sync_request=None, **kwargs):
    return await self.delegate.bind_weight_sync(sync_request, **kwargs)

  async def get_weight_sync_metadata(self, **kwargs):
    return await self.delegate.get_weight_sync_metadata(**kwargs)

  async def pre_weight_sync(self, sync_request=None, **kwargs):
    return await self.delegate.pre_weight_sync(sync_request, **kwargs)

  async def weight_sync(self, sync_request=None, **kwargs):
    return await self.delegate.weight_sync(sync_request, **kwargs)

  async def post_weight_sync(self, sync_request=None, **kwargs):
    return await self.delegate.post_weight_sync(sync_request, **kwargs)

  async def abort_weight_sync(self, sync_request=None, **kwargs):
    return await self.delegate.abort_weight_sync(sync_request, **kwargs)

  def get_weight_sync_status(self):
    return self.delegate.get_weight_sync_status()


class WeightSyncE2ETest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path

    # Build dummy HF assets
    model_spec = tpu_qwen3.model_registry("debugmodel", attn_backend="sdpa")
    model_spec.model.enable_weight_tying = False
    dummy_model = model_spec.model.build().to(torch.float32)
    torch.manual_seed(42)
    with torch.no_grad():
      for p in dummy_model.parameters():
        p.data.normal_(0.0, 0.02)
    adapter = Qwen3StateDictAdapter(model_spec.model, None)
    hf_sd = adapter.to_hf(dummy_model.state_dict())

    # Save to temp safetensors
    safetensors.torch.save_file(
        hf_sd, os.path.join(self.temp_dir, "model.safetensors")
    )
    with open(os.path.join(self.temp_dir, "config.json"), "w") as f:
      json.dump({"tie_word_embeddings": False}, f)

    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    self.engine = TorchTitanEngine(
        config=cfg,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    self.trainer_worker = TorchTitanTrainerWorker(
        engine=self.engine,
        worker_id="trainer_0",
    )
    self.trainer_worker.initialize()

    # Destination sampler double with vLLM-JAX 3D/transposed layout.
    self.qwen3_hf_cfg = _qwen3_hf_config_from_titan(model_spec.model)
    self.mock_sampler = FakeVllmQwen3Sampler(
        self.qwen3_hf_cfg, tp_size=2, dtype=jnp.float32
    )
    self.assertIsInstance(
        self.mock_sampler, basic_delegate.HfWeightUpdatableSampler
    )

    self.delegate = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_0",
        worker_index=0,
    )
    self.rollout_worker = RolloutWorkerTestDouble(self.delegate)

    asyncio.run(self.delegate.bind_weight_sync(sampler=self.mock_sampler))

    # Setup WorkerRegistry
    self.registry = worker_registry.WorkerRegistry()
    self.registry.register(self.trainer_worker)
    self.registry.register(self.rollout_worker)

    for wid in self.registry.worker_ids():
      worker_registry_testing.set_status(
          self.registry, wid, worker_registry.AdmissionStatus.ACTIVE
      )

    self.handler = basic_handler.BasicWeightSyncHandler()
    self.coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=self.registry,
        handler=self.handler,
    )

  def tearDown(self):
    self.trainer_worker.stop()
    self.delegate.close()
    super().tearDown()

  def test_e2e_weight_sync_round(self):
    # Run Round 1
    res = asyncio.run(self.coordinator.sync(policy_version=1))
    self.assertTrue(res.success, f"Round 1 failed: {res.failures}")
    self.assertEqual(self.coordinator.last_committed_version, 1)

    # Verify that mock_sampler received the weights and converted state
    self.assertIsNotNone(self.mock_sampler.received_hf_weights)
    assert self.mock_sampler.received_hf_weights is not None
    self.assertIn(
        "model.embed_tokens.weight", self.mock_sampler.received_hf_weights
    )
    self.assertIn(
        "model.layers.0.self_attn.q_proj.weight",
        self.mock_sampler.received_hf_weights,
    )
    q_proj_hf = self.mock_sampler.received_hf_weights[
        "model.layers.0.self_attn.q_proj.weight"
    ]
    self.assertEqual(q_proj_hf.dtype, jnp.float32)

    # Forward-pass logit parity: TorchTitan PyTorch vs pure-JAX vLLM layout.
    rng = np.random.default_rng(123)
    tokens_np = rng.integers(
        0, self.qwen3_hf_cfg.vocab_size, size=(2, 8), dtype=np.int64
    )
    tokens_1d = torch.from_numpy(tokens_np.reshape(-1))
    seq_len = tokens_np.shape[1]
    positions = torch.from_numpy(
        np.tile(np.arange(seq_len, dtype=np.int64), tokens_np.shape[0])
    )
    self.engine.model.eval()
    with torch.no_grad(), unflatten_shim.set_unflatten_seq_len(seq_len):
      pt_out = self.engine.model(tokens_1d, positions=positions)
    pt_logits = (
        (pt_out[0] if isinstance(pt_out, (tuple, list)) else pt_out)
        .reshape(tokens_np.shape[0], seq_len, -1)
        .cpu()
        .numpy()
    )
    jax_logits = _jax_qwen3_forward_vllm_layout(
        self.mock_sampler.state,
        self.qwen3_hf_cfg,
        tokens_np,
        tp_size=2,
    )
    np.testing.assert_allclose(jax_logits, pt_logits, rtol=1e-4, atol=1e-4)

    # Run Round 2
    res2 = asyncio.run(self.coordinator.sync(policy_version=2))
    self.assertTrue(res2.success, f"Round 2 failed: {res2.failures}")
    self.assertEqual(self.coordinator.last_committed_version, 2)

  def test_e2e_padded_vocab_head_dim_and_kv_repeat_logit_parity(self):
    custom_spec = tpu_qwen3.model_registry("debugmodel", attn_backend="sdpa")
    cfg_m = custom_spec.model
    cfg_m.vocab_size = 1003
    cfg_m.dim = 128
    cfg_m.enable_weight_tying = False
    cfg_m.tok_embeddings.num_embeddings = 1003
    cfg_m.tok_embeddings.embedding_dim = 128
    cfg_m.lm_head.in_features = 128
    cfg_m.lm_head.out_features = 1003
    cfg_m.norm.normalized_shape = 128
    cfg_m.layers = cfg_m.layers[:2]
    head_dim = 32
    inter_dim = 256
    for layer in cfg_m.layers:
      layer.attention_norm.normalized_shape = 128
      layer.ffn_norm.normalized_shape = 128
      layer.attention.dim = 128
      layer.attention.n_heads = 2
      layer.attention.n_kv_heads = 1
      layer.attention.head_dim = head_dim
      layer.attention.qkv_linear.n_heads = 2
      layer.attention.qkv_linear.n_kv_heads = 1
      layer.attention.qkv_linear.head_dim = head_dim
      layer.attention.qkv_linear.wqkv.in_features = 128
      layer.attention.qkv_linear.wqkv.out_features = (2 + 2 * 1) * head_dim
      layer.attention.wo.in_features = 2 * head_dim
      layer.attention.wo.out_features = 128
      layer.attention.qk_norm.normalized_shape = head_dim
      layer.attention.rope = CosSinRoPE.Config(
          dim=head_dim,
          max_context_length=64,
          theta=1000000.0,
      )
      layer.feed_forward.w1.in_features = 128
      layer.feed_forward.w1.out_features = inter_dim
      layer.feed_forward.w2.in_features = inter_dim
      layer.feed_forward.w2.out_features = 128
      layer.feed_forward.w3.in_features = 128
      layer.feed_forward.w3.out_features = inter_dim

    custom_model = cfg_m.build().to(torch.float32)
    torch.manual_seed(99)
    with torch.no_grad():
      for p in custom_model.parameters():
        p.data.normal_(0.0, 0.02)

    engine_pad = TorchTitanEngine(
        config=TorchTitanEngineConfig(
            model_name="qwen3",
            flavor="debugmodel",
            dtype="float32",
            mixed_precision_param="float32",
            learning_rate=1e-3,
            device_type="cpu",
            load_hf_weights=False,
        ),
        model=custom_model,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    trainer_pad = TorchTitanTrainerWorker(
        engine=engine_pad,
        worker_id="trainer_pad",
    )
    trainer_pad.initialize()

    hf_cfg = _qwen3_hf_config_from_titan(cfg_m)
    sampler_pad = FakeVllmQwen3Sampler(hf_cfg, tp_size=4, dtype=jnp.float32)
    self.assertEqual(
        sampler_pad.state["model.embed_tokens.weight"].shape, (1004, 128)
    )
    self.assertEqual(
        sampler_pad.state["model.layers.0.self_attn.k_proj.weight"].shape,
        (128, 4, 128),
    )

    delegate_pad = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_pad", worker_index=0
    )
    rollout_pad = RolloutWorkerTestDouble(delegate_pad)
    asyncio.run(delegate_pad.bind_weight_sync(sampler=sampler_pad))

    registry = worker_registry.WorkerRegistry()
    registry.register(trainer_pad)
    registry.register(rollout_pad)
    for wid in registry.worker_ids():
      worker_registry_testing.set_status(
          registry, wid, worker_registry.AdmissionStatus.ACTIVE
      )
    handler = basic_handler.BasicWeightSyncHandler()
    coord = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry, handler=handler
    )
    try:
      res = asyncio.run(coord.sync(policy_version=1))
      self.assertTrue(res.success, f"Padded sync failed: {res.failures}")
      rng = np.random.default_rng(321)
      tokens_np = rng.integers(
          0, hf_cfg.vocab_size, size=(2, 8), dtype=np.int64
      )
      tokens_1d = torch.from_numpy(tokens_np.reshape(-1))
      positions = torch.from_numpy(
          np.tile(
              np.arange(tokens_np.shape[1], dtype=np.int64), tokens_np.shape[0]
          )
      )
      engine_pad.model.eval()
      with torch.no_grad(), unflatten_shim.set_unflatten_seq_len(
          tokens_np.shape[1]
      ):
        pt_out = engine_pad.model(tokens_1d, positions=positions)
      pt_logits = (
          (pt_out[0] if isinstance(pt_out, (tuple, list)) else pt_out)
          .reshape(tokens_np.shape[0], tokens_np.shape[1], -1)
          .cpu()
          .numpy()
      )
      jax_logits = _jax_qwen3_forward_vllm_layout(
          sampler_pad.state, hf_cfg, tokens_np, tp_size=4
      )
      np.testing.assert_allclose(jax_logits, pt_logits, rtol=1e-4, atol=1e-4)
    finally:
      trainer_pad.stop()
      delegate_pad.close()

  def test_bfloat16_weight_sync_e2e(self):
    cfg_bf16 = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="bfloat16",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine_bf16 = TorchTitanEngine(
        config=cfg_bf16,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    self.assertTrue(
        all(p.dtype == torch.float32 for p in engine_bf16.model.parameters())
    )
    trainer_worker_bf16 = TorchTitanTrainerWorker(
        engine=engine_bf16,
        worker_id="trainer_bf16",
    )
    trainer_worker_bf16.initialize()
    sampler_bf16 = FakeVllmQwen3Sampler(
        self.qwen3_hf_cfg, tp_size=2, dtype=jnp.bfloat16
    )

    delegate_bf16 = basic_delegate.BasicDestinationDelegate(
        server_id="rollout_bf16",
        worker_index=0,
    )
    rollout_worker_bf16 = RolloutWorkerTestDouble(delegate_bf16)
    asyncio.run(delegate_bf16.bind_weight_sync(sampler=sampler_bf16))

    registry = worker_registry.WorkerRegistry()
    registry.register(trainer_worker_bf16)
    registry.register(rollout_worker_bf16)
    for wid in registry.worker_ids():
      worker_registry_testing.set_status(
          registry, wid, worker_registry.AdmissionStatus.ACTIVE
      )

    handler = basic_handler.BasicWeightSyncHandler()
    coord = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=handler,
    )
    try:
      res = asyncio.run(coord.sync(policy_version=1))
      self.assertTrue(res.success, f"Bfloat16 sync failed: {res.failures}")
      self.assertIsNotNone(sampler_bf16.received_hf_weights)
      assert sampler_bf16.received_hf_weights is not None
      q_proj_val = sampler_bf16.received_hf_weights[
          "model.layers.0.self_attn.q_proj.weight"
      ]
      self.assertEqual(q_proj_val.dtype, jnp.bfloat16)
      self.assertEqual(
          sampler_bf16.state["model.layers.0.self_attn.q_proj.weight"].dtype,
          jnp.bfloat16,
      )
      self.assertTrue(
          all(p.dtype == torch.float32 for p in engine_bf16.model.parameters())
      )
    finally:
      trainer_worker_bf16.stop()
      delegate_bf16.close()

  def test_qwen3_0_6b_model_registry_and_vllm_manifest_shapes(self):
    name, flavor = model_registry.resolve_model_name_and_flavor(
        "Qwen/Qwen3-0.6B"
    )
    self.assertEqual((name, flavor), ("qwen3", "0.6B"))
    spec = model_registry.resolve_model_spec(
        name, flavor, max_context_length=2048
    )
    self.assertIsInstance(spec.model, tpu_qwen3.Qwen3Model.Config)
    assert isinstance(spec.model, tpu_qwen3.Qwen3Model.Config)
    self.assertTrue(spec.model.enable_weight_tying)
    self.assertEqual(spec.model.vocab_size, 151936)
    self.assertEqual(spec.model.dim, 1024)
    self.assertLen(spec.model.layers, 28)

    hf_cfg = _qwen3_hf_config_from_titan(spec.model)
    self.assertEqual(hf_cfg.num_attention_heads, 16)
    self.assertEqual(hf_cfg.num_key_value_heads, 8)
    self.assertEqual(hf_cfg.head_dim, 128)
    self.assertTrue(hf_cfg.tie_word_embeddings)

    manifest = qwen3_manifest.build_from_hf_manifest(hf_cfg, tp_size=4)
    self.assertEqual(manifest["dropped_keys"], ["lm_head.weight"])
    rules_by_target = {r["target_keys"][0]: r for r in manifest["rules"]}
    self.assertNotIn("lm_head.weight", rules_by_target)
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.q_proj.weight"]["ops"],
        [
            {
                "reshape": {
                    "src_shape": [2048, 1024],
                    "dst_shape": [16, 128, 1024],
                }
            },
            {"transpose": {"axes": [2, 0, 1]}},
        ],
    )
    self.assertEqual(
        rules_by_target["model.layers.0.self_attn.k_proj.weight"]["ops"],
        [
            {
                "reshape": {
                    "src_shape": [1024, 1024],
                    "dst_shape": [8, 128, 1024],
                }
            },
            {"transpose": {"axes": [2, 0, 1]}},
        ],
    )


if __name__ == "__main__":
  absltest.main()
