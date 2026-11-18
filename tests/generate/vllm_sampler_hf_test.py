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

"""Hermetic CPU tests for VllmSampler HF manifest and weight updates."""

from collections.abc import Sequence
import dataclasses
import os
from unittest import mock

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
os.environ["JAX_PLATFORMS"] = "cpu"

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

_require_tpu_inference = False

try:
  from tpu_inference.layers.jax import pp_utils as tpu_pp_utils
  from tpu_inference.models.jax import qwen3 as tpu_qwen3
except ImportError:
  tpu_pp_utils = None
  tpu_qwen3 = None

from tunix.generate import vllm_sampler
from tunix.models import hf_mapping_jax
from tunix.models.qwen3 import mapping_vllm_jax
# pylint: enable=g-import-not-at-top


@dataclasses.dataclass
class _FakeQwen3HfConfig:
  model_type: str
  vocab_size: int
  hidden_size: int
  intermediate_size: int
  num_hidden_layers: int
  num_attention_heads: int
  num_key_value_heads: int | None
  head_dim: int | None
  tie_word_embeddings: bool
  attention_bias: bool
  rms_norm_eps: float = 1e-6
  hidden_act: str = "silu"
  rope_parameters: dict[str, object] | None = None
  rope_theta: float = 1000000.0
  rope_scaling: dict[str, object] | None = None


class _FakeModelConfig:

  def __init__(
      self,
      hf_config: _FakeQwen3HfConfig,
      *,
      dtype: jax.typing.DTypeLike = jnp.bfloat16,
      runner_type: str = "generate",
  ):
    self.hf_config = hf_config
    self.dtype = dtype
    self.runner_type = runner_type

  def get_vocab_size(self) -> int:
    return self.hf_config.vocab_size


class _FakeParallelConfig:

  def __init__(self, tensor_parallel_size: int):
    self.tensor_parallel_size = tensor_parallel_size


@dataclasses.dataclass(frozen=True)
class _FakeCacheConfig:
  cache_dtype: str = "auto"


@dataclasses.dataclass(frozen=True)
class _FakePpGroup:
  is_first_rank: bool = True
  is_last_rank: bool = True
  rank_in_group: int = 0
  world_size: int = 1


class _FakeVllmConfig:

  def __init__(
      self,
      tensor_parallel_size: int,
      *,
      model_config: _FakeModelConfig | None = None,
  ):
    self.parallel_config = _FakeParallelConfig(tensor_parallel_size)
    self.model_config = model_config
    self.cache_config = _FakeCacheConfig()
    self.quant_config = None
    self.speculative_config = None


class _FakeRunner:

  def __init__(
      self,
      *,
      state: nnx.State | dict[str, jax.Array],
      mesh: jax.sharding.Mesh,
      hf_config: _FakeQwen3HfConfig,
      tp_size: int,
  ):
    self.state = state
    self.state_leaves = ()
    self.mesh = mesh
    self.model = object()
    self.model_config = _FakeModelConfig(hf_config)
    self.vllm_config = _FakeVllmConfig(tp_size)


class _FakeLlmEngine:

  def __init__(self, runner: _FakeRunner):
    self.model_executor = mock.MagicMock()
    self.model_executor.driver_worker.model_runner = runner
    self.prefix_resets = 0
    self.rpc_calls: list[str] = []

  def reset_prefix_cache(self) -> None:
    self.prefix_resets += 1

  def collective_rpc(self, method: str) -> None:
    self.rpc_calls.append(method)


class _FakeLlm:

  def __init__(self, engine: _FakeLlmEngine):
    self.llm_engine = engine

  def reset_prefix_cache(self) -> None:
    self.llm_engine.reset_prefix_cache()

  def collective_rpc(self, method: str) -> None:
    self.llm_engine.collective_rpc(method)


class VllmSamplerFromHfTest(absltest.TestCase):
  """Hermetic CPU tests for `VllmSampler` HF manifest and updates."""

  def _make_sampler(
      self,
      *,
      tie_word_embeddings: bool = True,
      model_type: str = "qwen3",
      free_kv: bool = True,
      delete_dst_buffers: bool = True,
      use_dict_state: bool = False,
  ) -> tuple[
      vllm_sampler.VllmSampler,
      _FakeQwen3HfConfig,
      _FakeLlmEngine,
      jax.sharding.Mesh,
  ]:
    cpu_dev = jax.devices("cpu")[0]
    mesh = jax.sharding.Mesh(np.array([cpu_dev]), ("model",))
    hf_cfg = _FakeQwen3HfConfig(
        model_type=model_type,
        vocab_size=19,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        tie_word_embeddings=tie_word_embeddings,
        attention_bias=False,
    )
    spec = mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
        hf_cfg, tp_size=1, head_sharding_size=1
    )
    rng = np.random.default_rng(7)
    hf_zero = self._random_hf_weights(hf_cfg, rng, scale=0.0)
    with jax.default_device(cpu_dev):
      dst_arrays = hf_mapping_jax.from_hf_impl(spec, hf_zero)
    sharded: dict[str, jax.Array] = {}
    for r in spec.rules:
      tgt_key = r.target_keys[0]
      sharded[tgt_key] = jax.device_put(
          dst_arrays[tgt_key].astype(jnp.bfloat16),
          jax.sharding.NamedSharding(
              mesh, jax.sharding.PartitionSpec(*(r.sharding or (None,)))
          ),
      )
    state: nnx.State | dict[str, jax.Array]
    if use_dict_state:
      state = sharded
    else:
      # Build a nested nnx.State preserving exact dot-path structure.
      nested: dict[str, object] = {}
      for dot_key, arr in sharded.items():
        cur: dict[str, object] = nested
        parts = dot_key.split(".")
        for p in parts[:-1]:
          child = cur.get(p)
          if not isinstance(child, dict):
            child = {}
            cur[p] = child
          cur = child
        cur[parts[-1]] = nnx.Param(arr)
      state = nnx.State(nested)

    runner = _FakeRunner(
        state=state, mesh=mesh, hf_config=hf_cfg, tp_size=1
    )
    engine = _FakeLlmEngine(runner)
    sampler = object.__new__(vllm_sampler.VllmSampler)
    sampler.config = vllm_sampler.VllmConfig(
        free_kv_cache_during_weight_sync=free_kv,
        delete_dst_buffers=delete_dst_buffers,
    )
    sampler._driver = None
    sampler.llm = _FakeLlm(engine)  # pyrefly: ignore[bad-assignment]
    sampler._kv_cache_deleted = False
    return sampler, hf_cfg, engine, mesh

  def _random_hf_weights(
      self,
      cfg: _FakeQwen3HfConfig,
      rng: np.random.Generator,
      *,
      scale: float = 0.1,
  ) -> dict[str, jax.Array]:
    d = cfg.hidden_size
    inter = cfg.intermediate_size
    nq = cfg.num_attention_heads
    nkv = cfg.num_key_value_heads or nq
    hd = cfg.head_dim or (d // nq)

    def _w(shape: tuple[int, ...]) -> jax.Array:
      arr = rng.standard_normal(shape, dtype=np.float32) * scale
      return jax.device_put(jnp.asarray(arr), jax.devices("cpu")[0])

    out: dict[str, jax.Array] = {
        "model.embed_tokens.weight": _w((cfg.vocab_size, d)),
        "model.norm.weight": _w((d,)),
    }
    for layer_idx in range(cfg.num_hidden_layers):
      p = f"model.layers.{layer_idx}"
      out.update({
          f"{p}.input_layernorm.weight": _w((d,)),
          f"{p}.post_attention_layernorm.weight": _w((d,)),
          f"{p}.self_attn.q_norm.weight": _w((hd,)),
          f"{p}.self_attn.k_norm.weight": _w((hd,)),
          f"{p}.self_attn.q_proj.weight": _w((nq * hd, d)),
          f"{p}.self_attn.k_proj.weight": _w((nkv * hd, d)),
          f"{p}.self_attn.v_proj.weight": _w((nkv * hd, d)),
          f"{p}.self_attn.o_proj.weight": _w((d, nq * hd)),
          f"{p}.mlp.gate_proj.weight": _w((inter, d)),
          f"{p}.mlp.up_proj.weight": _w((inter, d)),
          f"{p}.mlp.down_proj.weight": _w((d, inter)),
      })
    if not cfg.tie_word_embeddings:
      out["lm_head.weight"] = _w((cfg.vocab_size, d))
    return out

  def test_from_hf_manifest_rejects_unsupported_configs(self):
    sampler_bad_family, _, _, _ = self._make_sampler(model_type="llama")
    with self.assertRaisesRegex(NotImplementedError, "model_type='llama'"):
      sampler_bad_family.from_hf_manifest()

    sampler_ok, _, _, _ = self._make_sampler()
    with mock.patch.dict(os.environ, {"LAYOUT_Q_PROJ_AS_NDH": "1"}):
      with self.assertRaisesRegex(NotImplementedError, "LAYOUT_Q_PROJ_AS_NDH"):
        sampler_ok.from_hf_manifest()

    runner = sampler_ok._model_runner
    runner.state = {}
    runner.model = mock.MagicMock()
    runner.model.load_canonical_weights = mock.MagicMock()
    with self.assertRaisesRegex(NotImplementedError, "torchax"):
      sampler_ok.from_hf_manifest()

  def test_update_params_from_hf_installs_sharded_weights_and_deletes_old(self):
    sampler, hf_cfg, engine, _ = self._make_sampler(
        tie_word_embeddings=False,
        free_kv=True,
        delete_dst_buffers=True,
    )
    old_norm = sampler.transformer_state.model.norm.weight.value
    old_sharding = old_norm.sharding
    hf_weights = self._random_hf_weights(hf_cfg, np.random.default_rng(11))

    sampler.update_params_from_hf(hf_weights)

    self.assertTrue(old_norm.is_deleted())
    new_norm = sampler.transformer_state.model.norm.weight.value
    self.assertEqual(new_norm.dtype, jnp.bfloat16)
    self.assertEqual(new_norm.sharding, old_sharding)
    np.testing.assert_allclose(
        np.asarray(new_norm.astype(jnp.float32)),
        np.asarray(hf_weights["model.norm.weight"].astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )
    q_proj = (
        sampler.transformer_state.model.layers["0"]
        .self_attn.q_proj.weight.value
    )
    self.assertEqual(q_proj.shape, (8, 2, 128))
    self.assertEqual(
        engine.rpc_calls, ["delete_kv_cache", "reinitialize_kv_cache"]
    )

  def test_update_params_from_hf_dict_state_and_fail_fast_validation(self):
    sampler, hf_cfg, engine, _ = self._make_sampler(
        tie_word_embeddings=True,
        free_kv=False,
        use_dict_state=True,
    )
    hf_weights = self._random_hf_weights(hf_cfg, np.random.default_rng(13))
    # Missing key fails before touching KV cache.
    bad_missing = dict(hf_weights)
    bad_missing.pop("model.norm.weight")
    with self.assertRaisesRegex(ValueError, "missing="):
      sampler.update_params_from_hf(bad_missing)
    self.assertEqual(engine.prefix_resets, 0)

    # Unexpected key fails before touching KV cache.
    bad_extra = dict(hf_weights)
    bad_extra["extra.weight"] = jnp.zeros((2,))
    with self.assertRaisesRegex(ValueError, "unexpected="):
      sampler.update_params_from_hf(bad_extra)
    self.assertEqual(engine.prefix_resets, 0)

    # Shape mismatch fails before deleting old buffers.
    old_embed = sampler.transformer_state["model.embed_tokens.weight"]
    old_norm = sampler.transformer_state["model.norm.weight"]
    bad_shape = dict(hf_weights)
    bad_shape["model.norm.weight"] = jnp.zeros((4,), dtype=jnp.float32)
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      sampler.update_params_from_hf(bad_shape)
    self.assertFalse(old_embed.is_deleted())
    self.assertFalse(old_norm.is_deleted())
    self.assertEqual(engine.prefix_resets, 0)

    # Valid update on dict state with tied lm_head.weight ignored.
    hf_with_tied = dict(hf_weights)
    hf_with_tied["lm_head.weight"] = hf_weights["model.embed_tokens.weight"]
    sampler.update_params_from_hf(hf_with_tied)
    self.assertTrue(old_norm.is_deleted())
    self.assertEqual(engine.prefix_resets, 1)

  def test_update_params_from_hf_tied_keys_allows_untrimmed_hf_and_fails_on_missing_dst(
      self,
  ):
    sampler, hf_cfg, engine, _ = self._make_sampler(
        tie_word_embeddings=True,
        free_kv=False,
        use_dict_state=True,
    )
    base_spec = sampler.from_hf_manifest()
    tied_spec = dataclasses.replace(
        base_spec,
        tied_keys=(("model.embed_tokens.weight", "tied_embed.weight"),),
        dropped_keys=(),
    )
    sampler.transformer_state["tied_embed.weight"] = jnp.zeros_like(
        sampler.transformer_state["model.embed_tokens.weight"]
    )
    hf_weights = self._random_hf_weights(hf_cfg, np.random.default_rng(17))
    hf_weights["tied_embed.weight"] = jnp.zeros_like(
        hf_weights["model.embed_tokens.weight"]
    )
    with mock.patch.object(
        sampler, "from_hf_manifest", return_value=tied_spec
    ):
      # (a) Untrimmed hf_weights still containing the tied destination key
      # succeeds without unexpected-key ValueError.
      sampler.update_params_from_hf(hf_weights)
      self.assertEqual(engine.prefix_resets, 1)
      np.testing.assert_array_equal(
          np.asarray(sampler.transformer_state["tied_embed.weight"]),
          np.asarray(sampler.transformer_state["model.embed_tokens.weight"]),
      )

      # (b) Non-dropped tied dst_key missing from destination state fails fast
      # before mutating buffers or resetting prefix cache.
      sampler.transformer_state.pop("tied_embed.weight")
      old_embed = sampler.transformer_state["model.embed_tokens.weight"]
      with self.assertRaisesRegex(
          KeyError, r"Tied target key 'tied_embed\.weight' missing"
      ):
        sampler.update_params_from_hf(hf_weights)
      self.assertFalse(old_embed.is_deleted())
      self.assertEqual(engine.prefix_resets, 1)

  def test_manifest_and_update_match_real_tpu_inference_qwen3(self):
    if _require_tpu_inference:
      self.assertIsNotNone(tpu_qwen3)
      self.assertIsNotNone(tpu_pp_utils)
    elif tpu_qwen3 is None or tpu_pp_utils is None:
      self.skipTest("tpu_inference is not installed")
    assert tpu_qwen3 is not None
    assert tpu_pp_utils is not None

    cpu_dev = jax.devices("cpu")[0]
    mesh = jax.sharding.Mesh(
        np.array([cpu_dev]).reshape((1, 1, 1, 1)),
        ("data", "attn_dp", "expert", "model"),
    )

    def _canon_sharding(
        pspec: Sequence[object] | None,
    ) -> tuple[object, ...]:
      items = list(pspec or ())
      while items and items[-1] is None:
        items.pop()
      return tuple(items)

    pp_group = _FakePpGroup()
    with (
        mock.patch.object(
            tpu_qwen3, "get_pp_group", return_value=pp_group
        ),
        mock.patch.object(
            tpu_pp_utils, "get_pp_group", return_value=pp_group
        ),
    ):
      for tie_word_embeddings in (True, False):
        hf_cfg = _FakeQwen3HfConfig(
            model_type="qwen3",
            vocab_size=1003,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            tie_word_embeddings=tie_word_embeddings,
            attention_bias=False,
        )
        model_cfg = _FakeModelConfig(hf_cfg, dtype=jnp.bfloat16)
        vllm_cfg = _FakeVllmConfig(
            tensor_parallel_size=2, model_config=model_cfg
        )
        with jax.set_mesh(mesh):
          model = tpu_qwen3.Qwen3ForCausalLM(
              vllm_cfg,  # pyrefly: ignore[bad-argument-type]
              jax.random.PRNGKey(0),
              mesh,
          )
        real_state = nnx.state(model, nnx.Param)
        real_vars = {
            ".".join(str(k) for k in path): var
            for path, var in real_state.flat_state()
        }

        spec = mapping_vllm_jax.build_vllm_jax_from_hf_manifest(
            hf_cfg,
            tp_size=2,
            head_sharding_size=mesh.shape["model"],
            vocab_size=model_cfg.get_vocab_size(),
        )
        hf_weights = self._random_hf_weights(
            hf_cfg, np.random.default_rng(23 + int(tie_word_embeddings))
        )
        with jax.default_device(cpu_dev):
          converted = hf_mapping_jax.from_hf_impl(spec, hf_weights)

        self.assertEqual(set(real_vars.keys()), set(converted.keys()))
        rule_by_target = {r.target_keys[0]: r for r in spec.rules}
        for dot_path, var in real_vars.items():
          self.assertEqual(
              tuple(var.value.shape),
              tuple(converted[dot_path].shape),
              msg=f"Shape mismatch at {dot_path} (tie={tie_word_embeddings})",
          )
          meta = var.get_metadata()
          var_sharding = meta["out_sharding"] if "out_sharding" in meta else ()
          self.assertEqual(
              _canon_sharding(var_sharding),
              _canon_sharding(rule_by_target[dot_path].sharding),
              msg=(
                  f"Sharding mismatch at {dot_path}"
                  f" (tie={tie_word_embeddings}): real={var_sharding!r}"
                  f" vs rule={rule_by_target[dot_path].sharding!r}"
              ),
          )

        if not tie_word_embeddings:
          runner = _FakeRunner(
              state=real_state, mesh=mesh, hf_config=hf_cfg, tp_size=2
          )
          engine = _FakeLlmEngine(runner)
          sampler = object.__new__(vllm_sampler.VllmSampler)
          sampler.config = vllm_sampler.VllmConfig(
              free_kv_cache_during_weight_sync=False,
              delete_dst_buffers=True,
          )
          sampler._driver = None
          sampler.llm = _FakeLlm(engine)  # pyrefly: ignore[bad-assignment]
          sampler._kv_cache_deleted = False

          sampler.update_params_from_hf(hf_weights)
          updated_lm_head = real_state.lm_head.weight.value
          self.assertEqual(updated_lm_head.dtype, jnp.bfloat16)
          np.testing.assert_allclose(
              np.asarray(updated_lm_head.astype(jnp.float32)),
              np.asarray(
                  converted["lm_head.weight"]
                  .astype(jnp.bfloat16)
                  .astype(jnp.float32)
              ),
              rtol=1e-2,
              atol=1e-2,
          )


if __name__ == "__main__":
  absltest.main()
