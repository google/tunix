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

import inspect
from unittest.mock import MagicMock

import jax
from jax import numpy as jnp
import numpy as np
import pytest
import torch
import transformers
from vllm.config import set_current_vllm_config
from vllm.model_executor.model_loader import get_model_loader

try:
  from transformers import Gemma4TextConfig
except ImportError:
  Gemma4TextConfig = None

from tpu_inference.distributed.jax_parallel_state import init_pp_distributed_environment
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import get_kv_cache_shape
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.jax.quantization import get_tpu_quantization_config
from tpu_inference.models.jax.gemma4 import (
    Gemma4DecoderLayer,
    Gemma4ForCausalLM,
)

K_MASK = -2.3819763e38


def create_pytorch_causal_mask(seq_len):
  """Creates a causal attention mask for a sequence of a given length."""
  mask = torch.ones(seq_len, seq_len, dtype=torch.float).tril(diagonal=0)
  mask = mask.masked_fill(mask == 0, K_MASK)
  mask = mask.masked_fill(mask == 1, 0)
  return mask


def get_hf_output(model, seq_len: int):
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)
  return model(x, attn_mask, position_ids).logits.detach().numpy()


def get_per_layer_hf_output(model, seq_len: int, num_layer_to_run: int = 1):
  """Get the first decoder layer output from the HF model."""
  x = (torch.arange(seq_len) + 1).reshape(1, -1)
  position_ids = torch.arange(seq_len).reshape(1, -1)
  attn_mask = create_pytorch_causal_mask(seq_len).unsqueeze(0).unsqueeze(0)

  m = model.get_decoder()
  emb = m.embed_tokens(x)

  try:
    position_embeddings = m.rotary_emb(emb, position_ids)
  except Exception:
    position_embeddings = None

  logits = emb
  for i in range(num_layer_to_run):
    layer = m.layers[i]
    sig = inspect.signature(layer.forward)
    kwargs = {}
    if position_embeddings is not None:
      if "position_embeddings" in sig.parameters:
        kwargs["position_embeddings"] = position_embeddings
      elif "rotary_pos_emb" in sig.parameters:
        kwargs["rotary_pos_emb"] = position_embeddings
    logits = layer(logits, attn_mask, position_ids, **kwargs)

  return logits[0].detach().numpy()


def get_per_layer_jax_output(model, seq_len: int, num_layer_to_run: int = 1):
  """Get the first N decoder layer output from the tpu_inference model."""
  input_ids = jnp.arange(seq_len) + 1
  input_ids = input_ids.reshape(1, -1)

  x = model.model.embed_tokens(input_ids)

  start_layer_idx = model.model.start_layer
  layer_0 = model.model.layers[start_layer_idx]
  num_key_value_heads = layer_0.self_attn.num_kv_heads
  qk_head_dim = layer_0.self_attn.head_dim_original
  kv_dtype = jnp.bfloat16

  block_size = 16
  num_blocks = 8
  cache_shape = get_kv_cache_shape(
      num_blocks, block_size, num_key_value_heads, qk_head_dim, kv_dtype
  )
  kv_cache = jnp.zeros(cache_shape, dtype=kv_dtype)

  attn_metadata = AttentionMetadata(
      input_positions=jnp.arange(seq_len),
      block_tables=jnp.array(list(range(1))),
      seq_lens=jnp.array([seq_len]),
      query_start_loc=jnp.array([0, seq_len]),
      request_distribution=jnp.array([0, 0, 1]),
  )

  for i in range(num_layer_to_run):
    layer = model.model.layers[start_layer_idx + i]
    kv_cache, x = layer(
        kv_cache=kv_cache,
        x=x,
        attention_metadata=attn_metadata,
    )

  return x


def get_jax_output(model, seq_len: int):
  input_ids = jnp.arange(seq_len) + 1
  input_ids = input_ids.reshape(1, -1)

  start_layer_idx = model.model.start_layer
  layer_0 = model.model.layers[start_layer_idx]
  num_key_value_heads = layer_0.self_attn.num_kv_heads
  qk_head_dim = layer_0.self_attn.head_dim_original
  kv_dtype = jnp.bfloat16

  block_size = 16
  num_blocks = 8
  cache_shape = get_kv_cache_shape(
      num_blocks, block_size, num_key_value_heads, qk_head_dim, kv_dtype
  )
  kv_caches = [jnp.zeros(cache_shape, dtype=kv_dtype) for _ in range(len(model.model.layers))]

  attn_metadata = AttentionMetadata(
      input_positions=jnp.arange(seq_len),
      block_tables=jnp.array(list(range(1))),
      seq_lens=jnp.array([seq_len]),
      query_start_loc=jnp.array([0, seq_len]),
      request_distribution=jnp.array([0, 0, 1]),
  )

  _, x, _ = model(
      kv_caches=kv_caches,
      input_ids=input_ids,
      attention_metadata=attn_metadata,
  )
  logits = model.compute_logits(x)
  return logits


class TestGemma4Alignment:

  @pytest.mark.skipif(
      condition=Gemma4TextConfig is None,
      reason=(
          "Gemma4 requires transformers v5.5.0, which will break other models."
          " This test cannot be enabled until vLLM upgrades to transformers"
          " v5.5.0 or later."
      ),
  )
  @pytest.mark.parametrize(
      "model_name",
      [
          "google/gemma-4-E2B-it",
      ],
  )
  def test_gemma4_alignment(
      self,
      model_name,
      rng,
      mesh,
      mock_vllm_config,
      assert_weight_loading_memory_bounded,
  ):
    kv_cache_type = "auto"
    vllm_config = mock_vllm_config(model_name, kv_cache_type)
    vllm_config.load_config.load_format = "auto"
    vllm_config.parallel_config = MagicMock()
    vllm_config.parallel_config.enable_expert_parallel = False

    init_pp_distributed_environment(
        ip="",
        rank=0,
        world_size=1,
        device=jax.devices()[0],
        need_pp=False,
    )
    vllm_config.quant_config = get_tpu_quantization_config(vllm_config)
    model_config = vllm_config.model_config

    with jax.set_mesh(mesh):
      jax_model = Gemma4ForCausalLM(vllm_config, rng, mesh)

    # load weights from HF model
    with jax.set_mesh(mesh):
      loader = get_model_loader(vllm_config.load_config)
      with assert_weight_loading_memory_bounded(
          jax_model,
          description=f"load_weights({model_name})",
          threshold_multiplier=0.3,
      ), set_current_vllm_config(vllm_config):
        loader.load_weights(jax_model, model_config)
    print("JAX model loaded.")

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32
    )
    print("HF model loaded.")

    # Make sure model weights are the same (check q_proj weight)
    hf_query_weight = (
        hf_model.get_decoder()
        .layers[0]
        .self_attn.q_proj.weight.detach()
        .numpy()
    )
    jax_query_weight = jax_model.model.layers[jax_model.model.start_layer].self_attn.q_proj.weight.value
    n, d, h = jax_query_weight.shape
    jax_query_weight = jax_query_weight.transpose(0, 2, 1).reshape(-1, d)
    np.testing.assert_equal(
        hf_query_weight,
        jax_query_weight,
        err_msg=(
            "Query weights are not equal, are you sure the loaded model weight"
            " between HF and JAX is identical?"
        ),
    )

    seq_len = 16
    tolerance = 2e-3

    # Compare per-layer output
    layer_to_run = 4
    hf_logits = get_per_layer_hf_output(hf_model, seq_len, layer_to_run)
    jax_logits = get_per_layer_jax_output(jax_model, seq_len, layer_to_run)
    np.testing.assert_allclose(
        hf_logits.squeeze(),
        jax_logits.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )

    # Compare full model output
    hf_output = get_hf_output(hf_model, seq_len)
    jax_output = get_jax_output(jax_model, seq_len)
    np.testing.assert_allclose(
        hf_output.squeeze(),
        jax_output.squeeze(),
        atol=tolerance,
        rtol=tolerance,
    )
    print("Logits are close! Model alignment check passed :)")
