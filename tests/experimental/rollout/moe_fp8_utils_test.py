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

"""Tests for MoE FP8 rollout quantization utilities."""

from flax import nnx
import jax
import jax.numpy as jnp
from tunix.experimental.rollout import moe_fp8_utils
from tunix.utils import maxtext_utils


class RoutedMoE(nnx.Module):
  """Mock RoutedMoE module matching MaxText's RoutedMoE attributes."""

  def __init__(self, num_experts: int = 4, emb_dim: int = 16, mlp_dim: int = 8):
    self.wi_kernel_axes = ("expert", None, "mlp")
    self.wo_kernel_axes = ("expert", "mlp", None)
    self.wi = nnx.Param(
        jnp.linspace(-2.0, 2.0, num_experts * emb_dim * (2 * mlp_dim), dtype=jnp.bfloat16)
        .reshape(num_experts, emb_dim, 2 * mlp_dim),
        out_sharding=self.wi_kernel_axes,
    )
    self.wo = nnx.Param(
        jnp.linspace(-1.0, 1.0, num_experts * mlp_dim * emb_dim, dtype=jnp.bfloat16)
        .reshape(num_experts, mlp_dim, emb_dim),
        out_sharding=self.wo_kernel_axes,
    )
    self.wi_scale = None
    self.wo_scale = None
    self.weight_dtype = jnp.bfloat16
    self.quant = None


class DummyDecoder(nnx.Module):

  def __init__(self):
    self.moe_0 = RoutedMoE()
    self.moe_1 = RoutedMoE()


def test_quantize_moe_weight_3d_shapes_and_accuracy():
  key = jax.random.PRNGKey(0)
  w = jax.random.normal(key, (8, 32, 16), dtype=jnp.bfloat16) * 0.5
  q_w, scale = moe_fp8_utils.quantize_moe_weight_3d(w)

  assert q_w.dtype == jnp.float8_e4m3fn
  assert q_w.shape == (8, 32, 16)
  assert scale.dtype == jnp.float32
  assert scale.shape == (8, 1, 16)

  recon = q_w.astype(jnp.float32) * scale
  rel_err = jnp.linalg.norm(w.astype(jnp.float32) - recon) / jnp.linalg.norm(
      w.astype(jnp.float32)
  )
  assert float(rel_err) < 0.05


def test_quantize_routed_moe_modules_in_place():
  model = DummyDecoder()
  n = moe_fp8_utils.quantize_routed_moe_modules_in_place(model)
  assert n == 2

  for moe in (model.moe_0, model.moe_1):
    assert moe.wi[...].dtype == jnp.float8_e4m3fn
    assert moe.wo[...].dtype == jnp.float8_e4m3fn
    assert moe.wi_scale is not None
    assert moe.wo_scale is not None
    assert moe.wi_scale[...].dtype == jnp.float32
    assert moe.wo_scale[...].dtype == jnp.float32
    assert moe.wi_scale[...].shape == (4, 1, 16)
    assert moe.wo_scale[...].shape == (4, 1, 16)
    assert moe.weight_dtype == jnp.float8_e4m3fn
    assert moe.quant is not None

  state_dict = nnx.to_pure_dict(nnx.state(model, nnx.Param))
  assert "wi_scale" in state_dict["moe_0"]
  assert "wo_scale" in state_dict["moe_0"]

  # Idempotency check
  n2 = moe_fp8_utils.quantize_routed_moe_modules_in_place(model)
  assert n2 == 2
  assert model.moe_0.wi[...].dtype == jnp.float8_e4m3fn


def test_quantize_converted_state_nested_and_flat(monkeypatch):
  monkeypatch.setenv("ROLLOUT_MOE_FP8", "true")
  wi_bf16 = jnp.ones((4, 16, 8), dtype=jnp.bfloat16)
  wo_bf16 = jnp.ones((4, 4, 16), dtype=jnp.bfloat16) * 2.0

  # Test flat tuple-keyed dict (both MoeBlock_0 and routed_experts)
  flat = {
      ("decoder", "layers_0", "MoeBlock_0", "wi"): wi_bf16,
      ("decoder", "layers_0", "mlp", "routed_experts", "wo"): wo_bf16,
      ("decoder", "layers_0", "MoeBlock_0", "gate", "kernel"): jnp.ones(
          (16, 4), dtype=jnp.float32
      ),
  }
  count_flat = moe_fp8_utils.quantize_converted_state_moe_fp8(flat)
  assert count_flat == 2
  assert flat[("decoder", "layers_0", "MoeBlock_0", "wi")].dtype == jnp.float8_e4m3fn
  assert flat[("decoder", "layers_0", "MoeBlock_0", "wi_scale")].dtype == jnp.float32
  assert flat[("decoder", "layers_0", "MoeBlock_0", "wi_scale")].shape == (4, 1, 8)
  assert flat[("decoder", "layers_0", "mlp", "routed_experts", "wo")].dtype == jnp.float8_e4m3fn
  assert flat[("decoder", "layers_0", "mlp", "routed_experts", "wo_scale")].dtype == jnp.float32
  assert flat[("decoder", "layers_0", "mlp", "routed_experts", "wo_scale")].shape == (4, 1, 16)

  # Test nested dict (as returned by WeightConverter.convert)
  nested = {
      "decoder": {
          "layers_0": {
              "mlp": {
                  "routed_experts": {
                      "wi": wi_bf16,
                      "wo": wo_bf16,
                  }
              }
          }
      }
  }
  count_nested = moe_fp8_utils.quantize_converted_state_moe_fp8(nested)
  assert count_nested == 2
  moe_dict = nested["decoder"]["layers_0"]["mlp"]["routed_experts"]
  assert moe_dict["wi"].dtype == jnp.float8_e4m3fn
  assert moe_dict["wi_scale"].dtype == jnp.float32
  assert moe_dict["wo"].dtype == jnp.float8_e4m3fn
  assert moe_dict["wo_scale"].dtype == jnp.float32


def test_patch_trainer_converter_moe_fp8(monkeypatch):
  monkeypatch.setenv("ROLLOUT_MOE_FP8", "true")

  class FakeDirectConverter:

    def _execute_group_target_free(self, group, source_flat):
      del group, source_flat
      return [
          (
              ("decoder", "layers_0", "mlp", "routed_experts", "wi"),
              jnp.ones((4, 16, 8), dtype=jnp.bfloat16),
          ),
          (
              ("decoder", "layers_0", "mlp", "routed_experts", "wo"),
              jnp.ones((4, 4, 16), dtype=jnp.bfloat16) * 2.0,
          ),
      ]

  class FakeWeightConverter:

    def __init__(self):
      self._direct = FakeDirectConverter()

    def convert(self, params_state, target_state=None):
      del params_state, target_state
      outs = dict(self._direct._execute_group_target_free(None, None))
      return {
          "decoder": {
              "layers_0": {
                  "mlp": {
                      "routed_experts": {
                          k[-1]: v for k, v in outs.items()
                      }
                  }
              }
          }
      }

  conv = FakeWeightConverter()
  assert moe_fp8_utils.patch_trainer_converter_moe_fp8(conv) is True
  out = conv.convert({})
  moe_out = out["decoder"]["layers_0"]["mlp"]["routed_experts"]
  assert moe_out["wi"].dtype == jnp.float8_e4m3fn
  assert moe_out["wi_scale"].dtype == jnp.float32
  assert moe_out["wo"].dtype == jnp.float8_e4m3fn
  assert moe_out["wo_scale"].dtype == jnp.float32


def test_vllm_additional_config_defaults_float32_logits_when_moe_fp8(monkeypatch):
  monkeypatch.setenv("ROLLOUT_MOE_FP8", "true")
  monkeypatch.delenv("FLOAT32_LOGITS", raising=False)
  cfg = maxtext_utils.build_vllm_maxtext_additional_config("qwen3.5-397b-a17b")
  assert cfg["maxtext_config"]["float32_gate_logits"] is True
  assert cfg["maxtext_config"]["float32_logits"] is True
