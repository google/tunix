import dataclasses
import inspect
import itertools
import json
import os
import sys
import types
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
from tunix.models import automodel
from tunix.models import hf_mapping
from tunix.models import naming
from tunix.models import registry
from tunix.models import safetensors_loader
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma import params as gemma_params
from tunix.models.gemma4 import model as gemma4_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.models.qwen3p5 import model as qwen3p5_model
from tunix.models.qwen3p5 import params as qwen3p5_params
from tunix.oss import utils as oss_utils
from tunix.utils import env_utils

_MAXTEXT_ONLY_REPOS = frozenset({
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3.5-397B-A17B",
})

_MODEL_CLASS_BY_CATEGORY = {
    "gemma": "Gemma",
    "gemma3": "Gemma3",
    "gemma4": "Gemma4",
    "llama3": "Llama3",
    "qwen2": "Qwen2",
    "qwen3": "Qwen3",
    "qwen3p5": "Qwen3p5",
}


def _hf_variant_of(model):
  return model._hf_variant  # pylint: disable=protected-access


def _shrink_cfg_for_eval_shape(category: str, cfg):
  """Caps layer/expert counts while preserving every param-leaf pattern."""
  if category == "gemma":
    n = min(cfg.num_layers, 4)
    return dataclasses.replace(
        cfg, num_layers=n, attention_types=tuple(cfg.attention_types)[:n]
    )
  if category == "gemma3":
    v_cfg = cfg.vision_config
    if v_cfg is not None:
      v_cfg = dataclasses.replace(v_cfg, depth=min(v_cfg.depth, 2))
    return dataclasses.replace(
        cfg, num_layers=min(cfg.num_layers, 4), vision_config=v_cfg
    )
  if category == "gemma4":
    pat = cfg.attention_pattern or gemma4_model.GEMMA4_ATTENTION_PATTERN
    has_shared = cfg.frac_shared_layers > 0
    n = min(cfg.num_layers, 2 * len(pat) if has_shared else len(pat))
    v_cfg = cfg.vision_encoder
    if v_cfg is not None:
      v_cfg = dataclasses.replace(v_cfg, num_layers=min(v_cfg.num_layers, 2))
    a_cfg = cfg.audio_encoder
    if a_cfg is not None:
      a_cfg = dataclasses.replace(a_cfg, num_layers=min(a_cfg.num_layers, 2))
    ne = min(cfg.num_experts, 4) if cfg.num_experts is not None else None
    ept = (
        min(cfg.num_experts_per_tok, ne)
        if cfg.num_experts_per_tok is not None and ne is not None
        else cfg.num_experts_per_tok
    )
    return dataclasses.replace(
        cfg,
        num_layers=n,
        frac_shared_layers=0.5 if has_shared else 0.0,
        vision_encoder=v_cfg,
        audio_encoder=a_cfg,
        num_experts=ne,
        num_experts_per_tok=ept,
    )
  if category == "qwen3p5":
    n = min(cfg.num_layers, cfg.full_attention_interval * 2)
    return dataclasses.replace(
        cfg, num_layers=n, layer_types=cfg.layer_types[:n]
    )
  kw = {"num_layers": min(cfg.num_layers, 4)}
  if getattr(cfg, "num_experts", None) is not None:
    kw["num_experts"] = min(cfg.num_experts, 4)
    kw["num_experts_per_tok"] = min(
        cfg.num_experts_per_tok, kw["num_experts"]
    )
  return dataclasses.replace(cfg, **kw)


def _manifest_target_keys(manifest: dict[str, Any]) -> set[str]:
  dropped = frozenset(manifest["dropped_keys"])
  active_rules = [
      r
      for r in manifest["rules"]
      if not dropped.intersection((*r["source_keys"], *r["target_keys"]))
  ]
  targets = set(
      itertools.chain.from_iterable(r["target_keys"] for r in active_rules)
  )
  for _, dst in manifest["tied_keys"]:
    if dst not in dropped:
      targets.add(dst)
  return targets


def _get_all_models_test_parameters():
  return (
      dict(testcase_name="gemma-2b", model_name="gemma-2b"),
      dict(testcase_name="gemma-2b-it", model_name="gemma-2b-it"),
      dict(testcase_name="gemma-7b", model_name="gemma-7b"),
      dict(testcase_name="gemma-7b-it", model_name="gemma-7b-it"),
      dict(testcase_name="gemma-1.1-2b-it", model_name="gemma-1.1-2b-it"),
      dict(testcase_name="gemma-1.1-7b-it", model_name="gemma-1.1-7b-it"),
      dict(testcase_name="gemma-2-2b", model_name="gemma-2-2b"),
      dict(testcase_name="gemma-2-2b-it", model_name="gemma-2-2b-it"),
      dict(testcase_name="gemma-2-9b", model_name="gemma-2-9b"),
      dict(testcase_name="gemma-2-9b-it", model_name="gemma-2-9b-it"),
      dict(testcase_name="gemma-3-270m", model_name="gemma-3-270m"),
      dict(testcase_name="gemma-3-270m-it", model_name="gemma-3-270m-it"),
      dict(testcase_name="gemma-3-1b-pt", model_name="gemma-3-1b-pt"),
      dict(testcase_name="gemma-3-1b-it", model_name="gemma-3-1b-it"),
      dict(testcase_name="gemma-3-4b-pt", model_name="gemma-3-4b-pt"),
      dict(testcase_name="gemma-3-4b-it", model_name="gemma-3-4b-it"),
      dict(testcase_name="gemma-3-12b-pt", model_name="gemma-3-12b-pt"),
      dict(testcase_name="gemma-3-12b-it", model_name="gemma-3-12b-it"),
      dict(testcase_name="gemma-3-27b-pt", model_name="gemma-3-27b-pt"),
      dict(testcase_name="gemma-3-27b-it", model_name="gemma-3-27b-it"),
      dict(testcase_name="gemma-4-e2b", model_name="gemma-4-e2b"),
      dict(testcase_name="gemma-4-e4b", model_name="gemma-4-e4b"),
      dict(testcase_name="gemma-4-12b", model_name="gemma-4-12b"),
      dict(testcase_name="gemma-4-12b-it", model_name="gemma-4-12b-it"),
      dict(testcase_name="gemma-4-31b", model_name="gemma-4-31b"),
      dict(testcase_name="gemma-4-26b-a4b", model_name="gemma-4-26b-a4b"),
      dict(testcase_name="llama-3-70b", model_name="llama-3-70b"),
      dict(testcase_name="llama-3.1-70b", model_name="llama-3.1-70b"),
      dict(testcase_name="llama-3.1-405b", model_name="llama-3.1-405b"),
      dict(testcase_name="llama-3.1-8b", model_name="llama-3.1-8b"),
      dict(
          testcase_name="llama-3.2-1b-instruct",
          model_name="llama-3.2-1b-instruct",
      ),
      dict(testcase_name="llama-3.2-1b", model_name="llama-3.2-1b"),
      dict(
          testcase_name="llama-3.2-3b-instruct",
          model_name="llama-3.2-3b-instruct",
      ),
      dict(testcase_name="llama-3.2-3b", model_name="llama-3.2-3b"),
      dict(testcase_name="qwen2.5-0.5b", model_name="qwen2.5-0.5b"),
      dict(
          testcase_name="qwen2.5-0.5b-instruct",
          model_name="qwen2.5-0.5b-instruct",
      ),
      dict(
          testcase_name="qwen2.5-coder-0.5b",
          model_name="qwen2.5-coder-0.5b",
      ),
      dict(testcase_name="qwen2.5-1.5b", model_name="qwen2.5-1.5b"),
      dict(
          testcase_name="qwen2.5-1.5b-instruct",
          model_name="qwen2.5-1.5b-instruct",
      ),
      dict(testcase_name="qwen2.5-3b", model_name="qwen2.5-3b"),
      dict(
          testcase_name="qwen2.5-3b-instruct",
          model_name="qwen2.5-3b-instruct",
      ),
      dict(
          testcase_name="qwen2.5-coder-3b",
          model_name="qwen2.5-coder-3b",
      ),
      dict(testcase_name="qwen2.5-7b", model_name="qwen2.5-7b"),
      dict(
          testcase_name="qwen2.5-7b-instruct",
          model_name="qwen2.5-7b-instruct",
      ),
      dict(
          testcase_name="qwen2.5-coder-7b",
          model_name="qwen2.5-coder-7b",
      ),
      dict(testcase_name="qwen2.5-math-1.5b", model_name="qwen2.5-math-1.5b"),
      dict(
          testcase_name="deepseek-r1-distill-qwen-1.5b",
          model_name="deepseek-r1-distill-qwen-1.5b",
      ),
      dict(testcase_name="qwen3-0.6b", model_name="qwen3-0.6b"),
      dict(testcase_name="qwen3-1.7b", model_name="qwen3-1.7b"),
      dict(testcase_name="qwen3-4b", model_name="qwen3-4b"),
      dict(
          testcase_name="qwen3-4b-instruct-2507",
          model_name="qwen3-4b-instruct-2507",
      ),
      dict(
          testcase_name="qwen3-4b-thinking-2507",
          model_name="qwen3-4b-thinking-2507",
      ),
      dict(testcase_name="qwen3-8b", model_name="qwen3-8b"),
      dict(testcase_name="qwen3-14b", model_name="qwen3-14b"),
      dict(testcase_name="qwen3-30b-a3b", model_name="qwen3-30b-a3b"),
      dict(testcase_name="qwen3-32b", model_name="qwen3-32b"),
      dict(testcase_name="Qwen3-32B", model_name="Qwen3-32B"),
      dict(
          testcase_name="qwen3.5-0.8b",
          model_name="qwen3.5-0.8b",
      ),
      dict(
          testcase_name="qwen3.5-2b",
          model_name="qwen3.5-2b",
      ),
      dict(
          testcase_name="qwen3.5-35b-a3b",
          model_name="qwen3.5-35b-a3b",
      ),
      dict(
          testcase_name="qwen3.5-397b-a17b",
          model_name="qwen3.5-397b-a17b",
      ),
  )


class AutoModelTest(parameterized.TestCase):

  @mock.patch(
      "tunix.models.automodel.download_model",
      return_value="gs://my-bucket/my-model",
  )
  def test_from_pretrained_maxtext(self, mock_download):

    m_maxtext = types.ModuleType("maxtext")
    m_pyconfig = mock.MagicMock()
    m_model_creation_utils = mock.MagicMock()

    class MockMaxTextConfig:
      model_fields = {
          "skip_jax_distributed_system": True,
          "hf_access_token": "mock",
      }

    m_maxtext.MaxTextConfig = MockMaxTextConfig
    m_maxtext.pyconfig = m_pyconfig
    m_maxtext.model_creation_utils = m_model_creation_utils

    with mock.patch.dict("sys.modules", {"maxtext": m_maxtext}):
      mock_config = mock.MagicMock()
      m_pyconfig.initialize = mock.MagicMock(
          return_value=mock_config
      )
      m_model_creation_utils.from_pretrained = mock.MagicMock()

      mock_mesh = mock.MagicMock()
      with mock.patch.dict(os.environ, {"HF_TOKEN": "mock_token"}):
        automodel.AutoModel.from_pretrained(
            "qwen2.5-0.5b",
            mesh=mock_mesh,
            model_source=automodel.ModelSource.MAXTEXT,
            use_flash_attention=True,
            tunix_fake_arg_that_should_be_dropped=False,
            skip_jax_distributed_system=False,
            checkpoint_storage_use_ocdbt=False,
            checkpoint_storage_use_zarr3=False,
        )

      m_pyconfig.initialize.assert_called_once()

      called_argv = m_pyconfig.initialize.call_args[0][0]

      self.assertIn("model_name=qwen2.5-0.5b", called_argv)
      has_load_params = any(
          "load_parameters_path" in arg for arg in called_argv
      )
      self.assertFalse(has_load_params)
      self.assertIn("hf_access_token=mock_token", called_argv)

      self.assertIn("skip_jax_distributed_system=false", called_argv)
      self.assertIn("checkpoint_storage_use_ocdbt=false", called_argv)
      self.assertIn("checkpoint_storage_use_zarr3=false", called_argv)

      self.assertNotIn("use_flash_attention=true", called_argv)

      for arg in called_argv:
        self.assertNotIn("tunix_fake_arg_that_should_be_dropped", arg)

      m_model_creation_utils.from_pretrained.assert_called_once_with(
          mock_config, mesh=mock_mesh, wrap_with_tunix_adapter=True
      )

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_obtain_model_params_valid(self, model_name: str):
    automodel.call_model_config(model_name)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_get_params_module_valid(self, model_name: str):
    self.assertEqual(
        tuple(automodel.ModelModule),
        (automodel.ModelModule.MODEL, automodel.ModelModule.PARAMS),
    )
    params_module = automodel.get_model_module(
        model_name, automodel.ModelModule.PARAMS
    )
    for attr in (
        "from_hf_impl",
        "to_hf_impl",
        "build_from_hf_manifest",
        "build_to_hf_manifest",
        "export_to_hf_state_dict",
        "create_model_from_safe_tensors",
    ):
      self.assertTrue(
          hasattr(params_module, attr), msg=f"{model_name}: missing {attr}"
      )
    category = naming.ModelNaming(model_name=model_name).model_config_category
    self.assertEqual(
        hasattr(params_module, "HfVariant"),
        category in ("gemma3", "gemma4", "qwen3p5"),
    )

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_get_model_module_valid(self, model_name: str):
    model_lib_module = automodel.get_model_module(
        model_name, automodel.ModelModule.MODEL
    )
    self.assertTrue(hasattr(model_lib_module, "ModelConfig"))

  def test_get_model_module_invalid(self):
    with self.assertRaisesRegex(
        ValueError, "Could not determine model family for: invalid-model"
    ):
      automodel.get_model_module("invalid-model", automodel.ModelModule.PARAMS)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  @mock.patch.object(automodel, "get_model_module", autospec=True)
  def test_create_model_dynamically(
      self, mock_get_model_module, model_name: str
  ):
    mock_create_fn = mock.Mock()
    # A single-layout family module: no `HfVariant` enum.
    mock_get_model_module.return_value = types.SimpleNamespace(
        create_model_from_safe_tensors=mock_create_fn,
        __name__="mock_params_module",
    )
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    automodel.create_model_from_safe_tensors(
        model_name, "file_dir", "model_config", mesh, "dtype", "mode"
    )
    mock_create_fn.assert_called_once_with(
        file_dir="file_dir",
        config="model_config",
        mesh=mesh,
        dtype="dtype",
        mode="mode",
    )
    mock_get_model_module.assert_called_once_with(
        model_name, automodel.ModelModule.PARAMS
    )

  def test_create_model_infers_hf_variant_from_checkpoint_keys(self):
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    variants = qwen3p5_params.HfVariant
    cfg = qwen3p5_model.ModelConfig(
        num_layers=4,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        rope_theta=10_000,
        norm_eps=1e-6,
        use_tied_embedding=True,
    )
    ckpt_dirs = {}
    for variant, prefix in (
        (variants.CAUSAL_LM, "model."),
        (variants.CONDITIONAL_GENERATION, "model.language_model."),
    ):
      ckpt_dirs[variant] = self.create_tempdir().full_path
      with open(
          os.path.join(ckpt_dirs[variant], "model.safetensors.index.json"),
          "w",
      ) as f:
        json.dump({"weight_map": {f"{prefix}embed_tokens.weight": "a"}}, f)
    empty_dir = self.create_tempdir().full_path
    mock_load = self.enter_context(
        mock.patch.object(
            safetensors_loader, "load_and_create_model", autospec=True
        )
    )
    # 1. `hf_variant=None` resolves from `model.safetensors.index.json` keys
    # and stamps the loaded model.
    for variant, ckpt_dir in ckpt_dirs.items():
      model = automodel.create_model_from_safe_tensors(
          "qwen3.5-0.8b", ckpt_dir, cfg, mesh
      )
      self.assertEqual(
          mock_load.call_args.kwargs["manifest"]["model_type"],
          f"qwen3p5/{variant.value}",
      )
      self.assertIs(_hf_variant_of(model), variant)
    # 2. Explicit `hf_variant` skips key lookup entirely.
    model = automodel.create_model_from_safe_tensors(
        "qwen3.5-0.8b", empty_dir, cfg, mesh, hf_variant=variants.CAUSAL_LM
    )
    self.assertEqual(
        mock_load.call_args.kwargs["manifest"]["model_type"],
        "qwen3p5/causal_lm",
    )
    # 3. No matching anchor key or wrong-type `hf_variant` fails closed.
    with self.assertRaisesRegex(ValueError, "exactly one HF variant anchor"):
      automodel.create_model_from_safe_tensors(
          "qwen3.5-0.8b", empty_dir, cfg, mesh
      )
    with self.assertRaisesRegex(
        ValueError, "Expected `hf_variant` of type HfVariant"
    ):
      automodel.create_model_from_safe_tensors(
          "qwen3.5-0.8b", empty_dir, cfg, mesh, hf_variant="causal_lm"
      )
    # 4. `AutoModel.from_pretrained` loads unregistered `<org>/<repo>` forks
    # from their checkpoint keys, and honors an explicit override.
    with mock.patch.object(
        automodel, "call_model_config", autospec=True, return_value=cfg
    ):
      model, _ = automodel.AutoModel.from_pretrained(
          "my-org/Qwen3.5-0.8B",
          mesh,
          model_path=ckpt_dirs[variants.CAUSAL_LM],
      )
      self.assertIs(_hf_variant_of(model), variants.CAUSAL_LM)
      model, _ = automodel.AutoModel.from_pretrained(
          "my-org/Qwen3.5-0.8B",
          mesh,
          model_path=empty_dir,
          hf_variant=variants.CONDITIONAL_GENERATION,
      )
      self.assertIs(_hf_variant_of(model), variants.CONDITIONAL_GENERATION)
    self.assertEqual(mock_load.call_count, 5)

    # 5. Single-layout families reject `hf_variant`.
    mock_qwen3 = self.enter_context(
        mock.patch.object(
            qwen3_params, "create_model_from_safe_tensors", autospec=True
        )
    )
    automodel.create_model_from_safe_tensors(
        "qwen3-1.7b", empty_dir, "model_config", mesh
    )
    self.assertNotIn("hf_variant", mock_qwen3.call_args.kwargs)
    with self.assertRaisesRegex(
        ValueError, "single HF checkpoint layout and does not accept"
    ):
      automodel.create_model_from_safe_tensors(
          "qwen3-1.7b",
          empty_dir,
          "model_config",
          mesh,
          hf_variant=variants.CAUSAL_LM,
      )

  def test_model_catalog_manifest_and_param_completeness(self):
    self.assertNotEmpty(registry.MODEL_CATALOG)
    eval_cache: dict[tuple[object, ...], set[str]] = {}

    def eval_param_keys(model_cls, cfg, **model_kwargs) -> set[str]:
      key = (model_cls, tuple(sorted(model_kwargs.items())), repr(cfg))
      if key not in eval_cache:
        abs_m = nnx.eval_shape(
            lambda: model_cls(cfg, rngs=nnx.Rngs(params=0), **model_kwargs)
        )
        leaves, _ = jax.tree_util.tree_flatten_with_path(
            nnx.state(abs_m, nnx.Param).to_pure_dict()
        )
        eval_cache[key] = {
            safetensors_loader.path_to_key(p) for p, _ in leaves
        }
      return eval_cache[key]

    for info in registry.MODEL_CATALOG:
      with self.subTest(model_id=info.model_id):
        assert info.model_id is not None and info.model_name is not None
        assert info.model_config_category is not None
        cfg = automodel.call_model_config(info.model_name)
        if info.model_id in _MAXTEXT_ONLY_REPOS:
          with self.assertRaises(NotImplementedError):
            qwen3p5_params.build_from_hf_manifest(
                cfg, qwen3p5_params.HfVariant.CONDITIONAL_GENERATION
            )
          with self.assertRaises(NotImplementedError):
            nnx.eval_shape(
                lambda c=cfg: qwen3p5_model.Qwen3p5(
                    c, rngs=nnx.Rngs(params=0)
                )
            )
          continue

        params_mod = automodel.get_model_module(
            info.model_name, automodel.ModelModule.PARAMS
        )
        model_mod = automodel.get_model_module(
            info.model_name, automodel.ModelModule.MODEL
        )
        model_cls = getattr(
            model_mod, _MODEL_CLASS_BY_CATEGORY[info.model_config_category]
        )

        # 1. Full production preset config builds valid forward & inverse
        # manifests (multi-layout families: `CONDITIONAL_GENERATION` layout).
        layout = (
            (params_mod.HfVariant.CONDITIONAL_GENERATION,)
            if hasattr(params_mod, "HfVariant")
            else ()
        )
        full_from = params_mod.build_from_hf_manifest(cfg, *layout)
        full_to = params_mod.build_to_hf_manifest(cfg, *layout)
        self.assertEqual(full_to, hf_mapping.invert_manifest(full_from))

        # 2. Bijective NNX parameter-leaf coverage on eval_shape model state.
        small_cfg = _shrink_cfg_for_eval_shape(
            info.model_config_category, cfg
        )
        cases: list[tuple[object, dict[str, object]]] = []
        if info.model_config_category == "gemma4":
          text_cfg = dataclasses.replace(
              small_cfg, vision_encoder=None, audio_encoder=None
          )
          cases.append((text_cfg, {"text_only": True}))
          if small_cfg.vision_encoder or small_cfg.audio_encoder:
            cases.append((small_cfg, {"text_only": False}))
        else:
          cases.append((small_cfg, {}))
          if info.model_config_category == "gemma3":
            cfg_fn = getattr(model_mod.ModelConfig, info.model_config_id)
            if "text_only" in inspect.signature(cfg_fn).parameters:
              mm_small = _shrink_cfg_for_eval_shape(
                  "gemma3", cfg_fn(text_only=False)
              )
              cases.append((mm_small, {}))

        for m_cfg, model_kw in cases:
          from_m = params_mod.build_from_hf_manifest(m_cfg, *layout)
          to_m = params_mod.build_to_hf_manifest(m_cfg, *layout)
          eval_cfg = small_cfg if model_kw else m_cfg
          nnx_keys = eval_param_keys(model_cls, eval_cfg, **model_kw)
          self.assertEqual(_manifest_target_keys(from_m), nnx_keys)
          dropped = frozenset(from_m["dropped_keys"])
          tied_dst = {
              dst for _, dst in from_m["tied_keys"] if dst not in dropped
          }
          self.assertEqual(
              set(hf_mapping.active_source_keys(to_m)), nnx_keys - tied_dst
          )

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma-2b",
          model_name="gemma-2b",
          expected_version="2b",
      ),
      dict(
          testcase_name="gemma-2-2b-it",
          model_name="gemma-2-2b-it",
          expected_version="2-2b_it",
      ),
      dict(
          testcase_name="gemma-1.1-2b-it",
          model_name="gemma-1.1-2b-it",
          expected_version="1.1-2b_it",
      ),
  )
  @mock.patch.object(automodel, "get_model_module", autospec=True)
  def test_create_gemma_model_from_params(
      self,
      mock_get_model_module,
      model_name,
      expected_version,
  ):
    mock_params_lib = mock.Mock()
    mock_model_lib = mock.Mock()
    mock_get_model_module.side_effect = [mock_params_lib, mock_model_lib]

    automodel.create_gemma_model_from_params("path", model_name)

    mock_params_lib.load_and_format_params.assert_called_once_with("path")
    mock_model_lib.Gemma.from_params.assert_called_once_with(
        mock_params_lib.load_and_format_params.return_value,
        version=expected_version,
    )

  @mock.patch.object(automodel, "get_model_module", autospec=True)
  def test_create_gemma_model_from_params_shards_with_mesh(
      self, mock_get_model_module
  ):
    env_utils.setup_sharding_environment()

    class TinyGemma(nnx.Module):

      def __init__(self):
        self.config = gemma_model.ModelConfig.gemma2_2b_it()
        self.w = nnx.Param(
            np.ones((4,), dtype=np.float32),
            sharding=("devices",),
        )

    mock_params_lib = mock.create_autospec(gemma_params, spec_set=True)
    mock_params_lib.load_and_format_host_params.return_value = {
        "transformer": {}
    }
    mock_model_lib = mock.create_autospec(gemma_model, spec_set=True)
    tiny_model = TinyGemma()
    mock_model_lib.Gemma.from_params.return_value = tiny_model
    mock_get_model_module.side_effect = [mock_params_lib, mock_model_lib]
    mesh = jax.sharding.Mesh(jax.devices()[:1], ("devices",))

    model, returned_config = automodel.create_gemma_model_from_params(
        "path", "gemma-2-2b-it", mesh=mesh
    )

    mock_params_lib.load_and_format_host_params.assert_called_once_with("path")
    mock_params_lib.load_and_format_params.assert_not_called()
    mock_model_lib.Gemma.from_params.assert_called_once_with(
        {"transformer": {}},
        version="2-2b_it",
    )
    self.assertIsInstance(model.w.value, jax.Array)
    self.assertEqual(
        model.w.value.sharding,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("devices")),
    )
    self.assertIs(returned_config, tiny_model.config)

  def test_gemma_load_and_format_host_params_restores_numpy_arrays(self):
    ckpt_dir = os.path.join(self.create_tempdir().full_path, "ckpt")
    saved_tree = {
        "transformer/embedder": {
            "input_embedding": jax.numpy.arange(8, dtype=jax.numpy.float32)
        },
        "transformer/layer_0/mlp/gating_einsum": {
            "w": jax.numpy.ones((2, 2, 3), dtype=jax.numpy.float32)
        },
    }
    with ocp.StandardCheckpointer() as ckptr:
      ckptr.save(ckpt_dir, saved_tree)
      ckptr.wait_until_finished()

    with mock.patch.object(
        ocp.Checkpointer,
        "restore",
        autospec=True,
        spec_set=True,
        side_effect=ocp.Checkpointer.restore,
    ) as mock_restore:
      host_params = gemma_params.load_and_format_host_params(ckpt_dir)

    self.assertIsInstance(
        mock_restore.call_args.kwargs["args"], ocp.args.PyTreeRestore
    )
    jax_params = gemma_params.load_and_format_params(ckpt_dir)

    host_emb = host_params["transformer"]["embedder"]["input_embedding"]
    host_gate = host_params["transformer"]["layer_0"]["mlp"]["gating_einsum"]
    self.assertIsInstance(host_emb, np.ndarray)
    self.assertIsInstance(host_gate, np.ndarray)
    np.testing.assert_array_equal(host_emb, np.arange(8, dtype=np.float32))
    self.assertEqual(host_gate.shape, (2, 2, 3))
    self.assertIsInstance(
        jax_params["transformer"]["embedder"]["input_embedding"], jax.Array
    )

    zarr_ckpt_dir = os.path.join(self.create_tempdir().full_path, "zarr_ckpt")
    with ocp.Checkpointer(ocp.PyTreeCheckpointHandler(use_ocdbt=False)) as ckpt:
      ckpt.save(zarr_ckpt_dir, saved_tree)
    zarr_host_params = gemma_params.load_and_format_host_params(zarr_ckpt_dir)
    self.assertIsInstance(
        zarr_host_params["transformer"]["embedder"]["input_embedding"],
        np.ndarray,
    )

  def test_gemma_load_and_format_host_params_supports_legacy_checkpoint(self):
    legacy_dir = os.path.join(self.create_tempdir().full_path, "legacy_ckpt")
    os.makedirs(legacy_dir)
    saved_tree = {
        "transformer/embedder": {
            "input_embedding": jax.numpy.arange(4, dtype=jax.numpy.float32)
        },
        "transformer/layer_0/mlp/linear": {
            "w": jax.numpy.ones((2, 3), dtype=jax.numpy.float32)
        },
    }
    with open(os.path.join(legacy_dir, "checkpoint"), "wb") as f:
      f.write(ocp.msgpack_utils.msgpack_serialize(jax.device_get(saved_tree)))
    self.assertFalse(os.path.exists(os.path.join(legacy_dir, "_METADATA")))

    host_params = gemma_params.load_and_format_host_params(legacy_dir)
    emb = host_params["transformer"]["embedder"]["input_embedding"]
    linear = host_params["transformer"]["layer_0"]["mlp"]["linear"]
    self.assertIsInstance(emb, np.ndarray)
    self.assertIsInstance(linear, np.ndarray)
    np.testing.assert_array_equal(emb, np.arange(4, dtype=np.float32))
    self.assertEqual(linear.shape, (2, 3))

    with mock.patch.object(
        ocp.Checkpointer,
        "restore",
        autospec=True,
        spec_set=True,
        return_value=saved_tree,
    ):
      converted_params = gemma_params.load_and_format_host_params(legacy_dir)
    self.assertIsInstance(
        converted_params["transformer"]["embedder"]["input_embedding"],
        np.ndarray,
    )

  def test_gemma_load_and_format_host_params_raises_on_missing_path(self):
    missing_dir = os.path.join(self.create_tempdir().full_path, "missing_ckpt")
    with self.assertRaises(FileNotFoundError):
      gemma_params.load_and_format_host_params(missing_dir)

  @mock.patch.object(automodel, "create_gemma_model_from_params", autospec=True)
  def test_from_pretrained_gemma2_gcs_dispatches_with_mesh(
      self, mock_create_gemma
  ):
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    mock_create_gemma.return_value = (
        mock_model,
        gemma_model.ModelConfig.gemma2_2b_it(),
    )
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    returned_model, resolved_path = automodel.AutoModel.from_pretrained(
        model_id="google/gemma-2-2b-it",
        mesh=mesh,
        model_source=automodel.ModelSource.GCS,
        model_path="gs://gemma-data/checkpoints/gemma2-2b-it",
    )
    self.assertIs(returned_model, mock_model)
    self.assertEqual(resolved_path, "gs://gemma-data/checkpoints/gemma2-2b-it")
    mock_create_gemma.assert_called_once_with(
        params_path="gs://gemma-data/checkpoints/gemma2-2b-it",
        model_name="gemma-2-2b-it",
        mesh=mesh,
    )

  @parameterized.named_parameters(
      dict(testcase_name="blank_token", env={"HF_TOKEN": "   "}, expected=None),
      dict(
          testcase_name="valid_token",
          env={"HF_TOKEN": "hf_secret"},
          expected="hf_secret",
      ),
  )
  def test_from_pretrained_maxtext_handles_hf_token_env(self, env, expected):
    fake_maxtext = types.ModuleType("maxtext")
    observed_argv: list[list[str]] = []
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)

    class FakeMaxTextConfig:
      model_fields = {
          "hf_access_token": str,
          "skip_jax_distributed_system": bool,
      }

    fake_pyconfig = types.SimpleNamespace(
        initialize=lambda argv: observed_argv.append(list(argv)) or "cfg"
    )
    fake_creation = types.SimpleNamespace(
        from_pretrained=lambda cfg, **kw: mock_model
    )
    setattr(fake_maxtext, "MaxTextConfig", FakeMaxTextConfig)
    setattr(fake_maxtext, "pyconfig", fake_pyconfig)
    setattr(fake_maxtext, "model_creation_utils", fake_creation)
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))

    with (
        mock.patch.dict(sys.modules, {"maxtext": fake_maxtext}),
        mock.patch.dict(os.environ, env, clear=True),
    ):
      returned_model, _ = automodel.AutoModel.from_pretrained(
          model_id="Qwen/Qwen2.5-0.5B",
          mesh=mesh,
          model_source=automodel.ModelSource.MAXTEXT,
      )

    self.assertIs(returned_model, mock_model)
    self.assertLen(observed_argv, 1)
    token_args = [
        a for a in observed_argv[0] if a.startswith("hf_access_token=")
    ]
    if expected is None:
      self.assertEmpty(token_args)
    else:
      self.assertEqual(token_args, [f"hf_access_token={expected}"])

  def test_download_model_kaggle_dispatches_to_kaggle_pipeline(self):
    download_dir = self.create_tempdir().full_path
    expected_path = os.path.join(download_dir, "gemma2-2b")
    with mock.patch.object(
        oss_utils,
        "kaggle_pipeline",
        autospec=True,
        spec_set=True,
        return_value=expected_path,
    ) as mock_kaggle_pipeline:
      resolved = automodel.download_model(
          "google/gemma-2/flax/gemma2-2b",
          download_dir,
          automodel.ModelSource.KAGGLE,
      )

    self.assertEqual(resolved, expected_path)
    mock_kaggle_pipeline.assert_called_once_with(
        "google/gemma-2/flax/gemma2-2b", download_dir
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma_defaults",
          model_id="google/gemma-2b",
          model_path="google/gemma/flax/2b",
          expected_model_name="gemma-2b",
          extra_kwargs={},
          expected_intermediate_ckpt_dir=None,
          expected_rng_seed=0,
      ),
      dict(
          testcase_name="gemma1p1_defaults",
          model_id="google/gemma-1.1-2b-it",
          model_path="google/gemma/flax/1.1-2b-it",
          expected_model_name="gemma-1.1-2b-it",
          extra_kwargs={},
          expected_intermediate_ckpt_dir=None,
          expected_rng_seed=0,
      ),
      dict(
          testcase_name="gemma2_with_intermediate_ckpt_and_seed",
          model_id="google/gemma-2-2b-it",
          model_path="google/gemma-2/flax/gemma2-2b-it",
          expected_model_name="gemma-2-2b-it",
          extra_kwargs={
              "intermediate_ckpt_dir": "/tmp/nnx_intermediate",
              "rng_seed": 7,
          },
          expected_intermediate_ckpt_dir="/tmp/nnx_intermediate",
          expected_rng_seed=7,
      ),
  )
  def test_from_pretrained_gemma_kaggle_downloads_and_converts_to_nnx(
      self,
      model_id,
      model_path,
      expected_model_name,
      extra_kwargs,
      expected_intermediate_ckpt_dir,
      expected_rng_seed,
  ):
    download_dir = self.create_tempdir().full_path
    resolved_download = os.path.join(download_dir, "downloaded")
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))

    with (
        mock.patch.object(
            automodel,
            "download_model",
            autospec=True,
            spec_set=True,
            return_value=resolved_download,
        ) as mock_download,
        mock.patch.object(
            automodel,
            "create_gemma_model_with_nnx_conversion",
            autospec=True,
            spec_set=True,
            return_value=(mock_model, {"transformer/embedder": {}}),
        ) as mock_convert,
    ):
      returned_model, resolved_path = automodel.AutoModel.from_pretrained(
          model_id=model_id,
          mesh=mesh,
          model_source=automodel.ModelSource.KAGGLE,
          model_path=model_path,
          model_download_path=download_dir,
          **extra_kwargs,
      )

    self.assertIs(returned_model, mock_model)
    self.assertEqual(resolved_path, resolved_download)
    mock_download.assert_called_once_with(
        model_path, download_dir, automodel.ModelSource.KAGGLE
    )
    mock_convert.assert_called_once_with(
        model_name=expected_model_name,
        ckpt_path=resolved_download,
        intermediate_ckpt_dir=expected_intermediate_ckpt_dir,
        rng_seed=expected_rng_seed,
        mesh=mesh,
        model_path=model_path,
    )

  def test_from_pretrained_gemma_kaggle_uses_local_dir_without_download(self):
    local_dir = self.create_tempdir().full_path
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))

    with (
        mock.patch.object(
            automodel, "download_model", autospec=True, spec_set=True
        ) as mock_download,
        mock.patch.object(
            automodel,
            "create_gemma_model_with_nnx_conversion",
            autospec=True,
            spec_set=True,
            return_value=(mock_model, {"transformer/embedder": {}}),
        ) as mock_convert,
    ):
      returned_model, resolved_path = automodel.AutoModel.from_pretrained(
          model_id="google/gemma-2-2b-it",
          mesh=mesh,
          model_source=automodel.ModelSource.KAGGLE,
          model_path=local_dir,
      )

    self.assertIs(returned_model, mock_model)
    self.assertEqual(resolved_path, local_dir)
    mock_download.assert_not_called()
    mock_convert.assert_called_once_with(
        model_name="gemma-2-2b-it",
        ckpt_path=local_dir,
        intermediate_ckpt_dir=None,
        rng_seed=0,
        mesh=mesh,
        model_path=local_dir,
    )

  def test_from_pretrained_gemma4_kaggle_falls_through_to_safetensors(self):
    download_dir = self.create_tempdir().full_path
    resolved_download = os.path.join(download_dir, "gemma4-e2b")
    mock_model = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))

    with (
        mock.patch.object(
            automodel,
            "download_model",
            autospec=True,
            spec_set=True,
            return_value=resolved_download,
        ) as mock_download,
        mock.patch.object(
            automodel,
            "create_model_from_safe_tensors",
            autospec=True,
            spec_set=True,
            return_value=mock_model,
        ) as mock_safetensors,
    ):
      returned_model, resolved_path = automodel.AutoModel.from_pretrained(
          model_id="google/gemma-4-e2b",
          mesh=mesh,
          model_source=automodel.ModelSource.KAGGLE,
          model_path="google/gemma-4/flax/gemma-4-e2b",
          model_download_path=download_dir,
      )

    self.assertIs(returned_model, mock_model)
    self.assertEqual(resolved_path, resolved_download)
    mock_download.assert_called_once_with(
        "google/gemma-4/flax/gemma-4-e2b",
        download_dir,
        automodel.ModelSource.KAGGLE,
    )
    mock_safetensors.assert_called_once()
    args, kwargs = mock_safetensors.call_args
    self.assertEqual(args[0], "gemma-4-e2b")
    self.assertEqual(args[1], resolved_download)
    self.assertEqual(args[3], mesh)
    self.assertIsNone(kwargs["dtype"])
    self.assertNotIn("repo_id", kwargs)
    self.assertIsNone(kwargs["hf_variant"])

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma3",
          model_id="google/gemma-3-270m",
          error_regex="Gemma 3 models are only supported from GCS or INTERNAL",
      ),
      dict(
          testcase_name="non_gemma_qwen2",
          model_id="Qwen/Qwen2.5-0.5B",
          error_regex="Only Gemma models are supported from KAGGLE or GCS",
      ),
  )
  def test_from_pretrained_kaggle_rejects_unsupported_model_family(
      self, model_id, error_regex
  ):
    local_dir = self.create_tempdir().full_path
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    with self.assertRaisesRegex(NotImplementedError, error_regex):
      automodel.AutoModel.from_pretrained(
          model_id=model_id,
          mesh=mesh,
          model_source=automodel.ModelSource.KAGGLE,
          model_path=local_dir,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma-2b",
          model_name="gemma-2b",
          expected_dir_name="2b",
      ),
      dict(
          testcase_name="gemma-1.1-2b-it",
          model_name="gemma-1.1-2b-it",
          expected_dir_name="gemma-1.1-2b-it",
      ),
      dict(
          testcase_name="gemma-1.1-7b-it",
          model_name="gemma-1.1-7b-it",
          expected_dir_name="gemma-1.1-7b-it",
      ),
      dict(
          testcase_name="gemma-2-9b",
          model_name="gemma-2-9b",
          expected_dir_name="gemma-2-9b",
      ),
  )
  @mock.patch.object(automodel, "create_gemma_model_from_params", autospec=True)
  @mock.patch.object(automodel, "_get_gemma_base_model", autospec=True)
  @mock.patch("tunix.models.automodel.ocp.StandardCheckpointer", autospec=True)
  @mock.patch("os.path.exists", return_value=False)
  def test_create_gemma_model_with_nnx_conversion_dir_name(
      self,
      mock_exists,
      mock_checkpointer,
      mock_get_gemma_base_model,
      mock_create_gemma_model_from_params,
      model_name,
      expected_dir_name,
  ):
    del mock_exists, mock_checkpointer, mock_get_gemma_base_model
    mock_create_gemma_model_from_params.return_value = (
        mock.Mock(),
        mock.Mock(),
    )
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    automodel.create_gemma_model_with_nnx_conversion(
        model_name=model_name,
        ckpt_path="dummy_ckpt_path",
        intermediate_ckpt_dir="dummy_intermediate_ckpt_dir",
        rng_seed=0,
        mesh=mesh,
    )
    mock_create_gemma_model_from_params.assert_called_once()
    calls = mock_create_gemma_model_from_params.call_args_list
    self.assertEqual(
        calls[0][0][0], os.path.join("dummy_ckpt_path", expected_dir_name)
    )

  @parameterized.named_parameters(
      dict(testcase_name="kaggle", model_source=automodel.ModelSource.KAGGLE),
      dict(testcase_name="gcs", model_source=automodel.ModelSource.GCS),
      dict(
          testcase_name="internal", model_source=automodel.ModelSource.INTERNAL
      ),
  )
  def test_from_pretrained_missing_model_path(self, model_source):
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    with self.assertRaisesRegex(
        ValueError,
        f"model_path is required for model_source: {model_source}",
    ):
      automodel.AutoModel.from_pretrained(
          model_id="google/gemma-2b",
          mesh=mesh,
          model_source=model_source,
          model_path=None,
      )

  @mock.patch.object(naming, "ModelNaming", autospec=True)
  @mock.patch.object(automodel, "call_model_config", autospec=True)
  @mock.patch.object(automodel, "download_model", autospec=True)
  @mock.patch.object(automodel, "create_model_from_safe_tensors", autospec=True)
  def test_from_pretrained_with_config_overrides(
      self,
      mock_create_model,
      mock_download_model,
      mock_call_model_config,
      mock_model_naming,
  ):
    @dataclasses.dataclass
    class FakeConfig:
      use_flash_attention: bool = False
      flash_attention_block_size: int = 1024

    mock_naming_info = mock.Mock()
    mock_naming_info.model_family = "qwen2"
    mock_naming_info.model_name = "qwen2.5-0.5b"
    mock_model_naming.return_value = mock_naming_info

    mock_call_model_config.return_value = FakeConfig()
    mock_download_model.return_value = "fake_path"
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))

    # Execution
    automodel.AutoModel.from_pretrained(
        model_id="qwen/Qwen2.5-0.5B",
        mesh=mesh,
        use_flash_attention=True,
        flash_attention_block_size=512,
        invalid_param="ignored",
    )

    # Verification
    # check that create_model_from_safe_tensors was called with the overrides
    self.assertTrue(mock_create_model.called)
    called_config = mock_create_model.call_args[0][2]
    self.assertTrue(called_config.use_flash_attention)
    self.assertEqual(called_config.flash_attention_block_size, 512)
    self.assertFalse(hasattr(called_config, "invalid_param"))

  @mock.patch.object(automodel, "download_model", autospec=True)
  @mock.patch.object(automodel, "create_model_from_safe_tensors", autospec=True)
  def test_from_pretrained_with_local_model_path_and_remat(
      self,
      mock_create_model,
      mock_download_model,
  ):
    local_dir = self.create_tempdir().full_path
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    automodel.AutoModel.from_pretrained(
        model_id="Qwen/Qwen3-1.7B",
        mesh=mesh,
        model_path=local_dir,
        load_dtype="float32",
        remat_config="decoder",
        use_flash_attention=True,
        flash_attention_block_size=256,
    )
    mock_download_model.assert_not_called()
    mock_create_model.assert_called_once()
    args, kwargs = mock_create_model.call_args
    self.assertEqual(args[0], "qwen3-1.7b")
    self.assertEqual(args[1], local_dir)
    self.assertEqual(args[2].remat_config.name, "DECODER")
    self.assertTrue(args[2].use_flash_attention)
    self.assertEqual(args[2].flash_attention_block_size, 256)
    self.assertEqual(kwargs["dtype"], jax.numpy.float32)

    # Bare `model_name` (`dist_rl` nodes) and `ConfigId` resolve the model
    # name and defer `hf_variant` to checkpoint-key inference.
    for selector, expected_name in (
        ("Qwen3.5-0.8B", "qwen3.5-0.8b"),
        ("gemma4_e2b_it", "gemma4_e2b_it"),
    ):
      mock_create_model.reset_mock()
      automodel.AutoModel.from_pretrained(
          model_id=selector,
          mesh=mesh,
          model_path=local_dir,
      )
      mock_create_model.assert_called_once()
      c_args, c_kwargs = mock_create_model.call_args
      self.assertEqual(c_args[0], expected_name)
      self.assertNotIn("repo_id", c_kwargs)
      self.assertIsNone(c_kwargs["hf_variant"])

  def test_gemma_base_and_gemma3_checkpoint_helpers(self):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("fsdp", "tp")
    )
    tiny_cfg = gemma_model.ModelConfig(
        num_layers=1,
        num_embed=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=1,
        final_logit_softcap=None,
        attention_types=(gemma_model.AttentionType.GLOBAL,),
        use_post_attn_norm=False,
        use_post_ffw_norm=False,
    )
    ckpt_dir = self.create_tempdir().full_path
    ref_model = gemma_model.Gemma(tiny_cfg, rngs=nnx.Rngs(0))
    _, ref_state = nnx.split(ref_model)
    with ocp.StandardCheckpointer() as ckptr:
      ckptr.save(os.path.join(ckpt_dir, "state"), ref_state)
      ckptr.wait_until_finished()
    with mock.patch.object(
        automodel, "call_model_config", autospec=True, return_value=tiny_cfg
    ):
      restored_model, cfg_out = (
          automodel.create_gemma_model_with_nnx_conversion(
              "gemma-2b", ckpt_dir, ckpt_dir, 0, mesh
          )
      )
    self.assertIs(cfg_out, tiny_cfg)
    self.assertIsInstance(restored_model, gemma_model.Gemma)

    g3_params_mod = automodel.get_model_module(
        "gemma-3-270m", automodel.ModelModule.PARAMS
    )
    sentinel = mock.create_autospec(nnx.Module, instance=True, spec_set=True)
    with mock.patch.object(
        g3_params_mod,
        "create_model_from_checkpoint",
        autospec=True,
        return_value=sentinel,
    ) as mock_g3_ckpt:
      model_out, g3_cfg = automodel.create_gemma3_model_from_checkpoint(
          "gs://bucket/gemma3", "gemma-3-270m", mesh, sliding_window_size=64
      )
    self.assertIs(model_out, sentinel)
    self.assertEqual(g3_cfg.sliding_window_size, 64)
    mock_g3_ckpt.assert_called_once_with("gs://bucket/gemma3", g3_cfg, mesh)


if __name__ == "__main__":
  absltest.main()
