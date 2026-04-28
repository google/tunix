import dataclasses
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.models import automodel
from tunix.models import naming


def _get_all_models_test_parameters():
  return (
      dict(testcase_name="gemma-2b", model_name="gemma-2b"),
      dict(testcase_name="gemma-2b-it", model_name="gemma-2b-it"),
      dict(testcase_name="gemma-7b", model_name="gemma-7b"),
      dict(testcase_name="gemma-7b-it", model_name="gemma-7b-it"),
      dict(testcase_name="gemma1.1-2b-it", model_name="gemma1.1-2b-it"),
      dict(testcase_name="gemma1.1-7b-it", model_name="gemma1.1-7b-it"),
      dict(testcase_name="gemma-1.1-2b-it", model_name="gemma-1.1-2b-it"),
      dict(testcase_name="gemma-1.1-7b-it", model_name="gemma-1.1-7b-it"),
      dict(testcase_name="gemma2-2b", model_name="gemma2-2b"),
      dict(testcase_name="gemma2-2b-it", model_name="gemma2-2b-it"),
      dict(testcase_name="gemma2-9b", model_name="gemma2-9b"),
      dict(testcase_name="gemma2-9b-it", model_name="gemma2-9b-it"),
      dict(testcase_name="gemma-2-2b", model_name="gemma-2-2b"),
      dict(testcase_name="gemma-2-2b-it", model_name="gemma-2-2b-it"),
      dict(testcase_name="gemma-2-9b", model_name="gemma-2-9b"),
      dict(testcase_name="gemma-2-9b-it", model_name="gemma-2-9b-it"),
      dict(testcase_name="gemma3-270m", model_name="gemma3-270m"),
      dict(testcase_name="gemma3-270m-it", model_name="gemma3-270m-it"),
      dict(testcase_name="gemma3-1b-pt", model_name="gemma3-1b-pt"),
      dict(testcase_name="gemma3-1b-it", model_name="gemma3-1b-it"),
      dict(testcase_name="gemma3-4b-pt", model_name="gemma3-4b-pt"),
      dict(testcase_name="gemma3-4b-it", model_name="gemma3-4b-it"),
      dict(testcase_name="gemma3-12b-pt", model_name="gemma3-12b-pt"),
      dict(testcase_name="gemma3-12b-it", model_name="gemma3-12b-it"),
      dict(testcase_name="gemma3-27b-pt", model_name="gemma3-27b-pt"),
      dict(testcase_name="gemma3-27b-it", model_name="gemma3-27b-it"),
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
      dict(testcase_name="llama3-70b", model_name="llama3-70b"),
      dict(testcase_name="llama-3-70b", model_name="llama-3-70b"),
      dict(testcase_name="llama3.1-70b", model_name="llama3.1-70b"),
      dict(testcase_name="llama-3.1-70b", model_name="llama-3.1-70b"),
      dict(testcase_name="llama3.1-405b", model_name="llama3.1-405b"),
      dict(testcase_name="llama-3.1-405b", model_name="llama-3.1-405b"),
      dict(testcase_name="llama3.1-8b", model_name="llama3.1-8b"),
      dict(testcase_name="llama-3.1-8b", model_name="llama-3.1-8b"),
      dict(testcase_name="llama3.2-1b", model_name="llama3.2-1b"),
      dict(
          testcase_name="llama-3.2-1b-instruct",
          model_name="llama-3.2-1b-instruct",
      ),
      dict(
          testcase_name="llama3.2-1b-instruct",
          model_name="llama3.2-1b-instruct",
      ),
      dict(testcase_name="llama-3.2-1b", model_name="llama-3.2-1b"),
      dict(testcase_name="llama3.2-3b", model_name="llama3.2-3b"),
      dict(
          testcase_name="llama-3.2-3b-instruct",
          model_name="llama-3.2-3b-instruct",
      ),
      dict(
          testcase_name="llama3.2-3b-instruct",
          model_name="llama3.2-3b-instruct",
      ),
      dict(testcase_name="llama-3.2-3b", model_name="llama-3.2-3b"),
      dict(testcase_name="qwen2.5-0.5b", model_name="qwen2.5-0.5b"),
      dict(testcase_name="qwen2.5-1.5b", model_name="qwen2.5-1.5b"),
      dict(testcase_name="qwen2.5-3b", model_name="qwen2.5-3b"),
      dict(testcase_name="qwen2.5-7b", model_name="qwen2.5-7b"),
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
  )


def _get_gemma_models_test_parameters():
  return [
      p
      for p in _get_all_models_test_parameters()
      if p["model_name"].startswith("gemma")
  ]


def _get_non_gemma_models_test_parameters():
  return [
      p
      for p in _get_all_models_test_parameters()
      if not p["model_name"].startswith("gemma")
  ]


class AutoModelTest(parameterized.TestCase):



  @mock.patch("tunix.models.automodel.download_model", return_value="gs://my-bucket/my-model")
  def test_from_pretrained_maxtext(self, mock_download):
    import sys
    import types
    import unittest.mock as mock

    m_maxtext = types.ModuleType("maxtext")
    m_maxtext_configs = types.ModuleType("maxtext.configs")
    m_maxtext_configs_pyconfig = types.ModuleType("maxtext.configs.pyconfig")
    m_maxtext_configs_types = types.ModuleType("maxtext.configs.types")
    m_maxtext_utils = types.ModuleType("maxtext.utils")
    m_maxtext_utils_model_creation_utils = types.ModuleType("maxtext.utils.model_creation_utils")

    with mock.patch.dict("sys.modules", {
        "maxtext": m_maxtext,
        "maxtext.configs": m_maxtext_configs,
        "maxtext.configs.pyconfig": m_maxtext_configs_pyconfig,
        "maxtext.configs.types": m_maxtext_configs_types,
        "maxtext.utils": m_maxtext_utils,
        "maxtext.utils.model_creation_utils": m_maxtext_utils_model_creation_utils
    }):
        setattr(m_maxtext_utils, "model_creation_utils", m_maxtext_utils_model_creation_utils)
        setattr(m_maxtext_configs, "pyconfig", m_maxtext_configs_pyconfig)
        setattr(m_maxtext_configs, "types", m_maxtext_configs_types)
        setattr(m_maxtext, "configs", m_maxtext_configs)
        setattr(m_maxtext, "utils", m_maxtext_utils)

        mock_config = mock.MagicMock()
        m_maxtext_configs_pyconfig.initialize = mock.MagicMock(return_value=mock_config)
        m_maxtext_utils_model_creation_utils.from_pretrained = mock.MagicMock()

        class MockMaxTextConfig:
            model_fields = {"skip_jax_distributed_system": True, "hf_access_token": "mock"}

        m_maxtext_configs_types.MaxTextConfig = MockMaxTextConfig

        mock_mesh = mock.MagicMock()
        import os
        with mock.patch.dict(os.environ, {"HF_TOKEN": "mock_token"}):
            automodel.AutoModel.from_pretrained(
                "qwen2.5-0.5b",
                mesh=mock_mesh,
                model_source=automodel.ModelSource.MAXTEXT,
                use_flash_attention=True,
                tunix_fake_arg_that_should_be_dropped=False,
                skip_jax_distributed_system=False,
            )

        m_maxtext_configs_pyconfig.initialize.assert_called_once()

        called_argv = m_maxtext_configs_pyconfig.initialize.call_args[0][0]

        self.assertIn("model_name=qwen2.5-0.5b", called_argv)
        has_load_params = any("load_parameters_path" in arg for arg in called_argv)
        self.assertFalse(has_load_params)
        self.assertIn("hf_access_token=mock_token", called_argv)

        self.assertIn("skip_jax_distributed_system=false", called_argv)

        self.assertNotIn("use_flash_attention=true", called_argv)

        for arg in called_argv:
            self.assertNotIn("tunix_fake_arg_that_should_be_dropped", arg)

        m_maxtext_utils_model_creation_utils.from_pretrained.assert_called_once_with(
            mock_config, mesh=mock_mesh
        )
  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_obtain_model_params_valid(self, model_name: str):
    automodel.call_model_config(model_name)

  @parameterized.named_parameters(*_get_gemma_models_test_parameters())
  def test_get_params_module_gemma_valid(self, model_name: str):
    params_module = automodel.get_model_module(
        model_name, automodel.ModelModule.PARAMS_SAFETENSORS
    )
    self.assertTrue(hasattr(params_module, "create_model_from_safe_tensors"))

  @parameterized.named_parameters(*_get_non_gemma_models_test_parameters())
  def test_get_params_module_non_gemma_valid(self, model_name: str):
    params_module = automodel.get_model_module(
        model_name, automodel.ModelModule.PARAMS
    )
    self.assertTrue(hasattr(params_module, "create_model_from_safe_tensors"))

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
    mock_params_module = mock.Mock()
    mock_params_module.create_model_from_safe_tensors = mock_create_fn
    mock_params_module.__name__ = "mock_params_module"
    mock_get_model_module.return_value = mock_params_module
    mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
    naming_info = naming.ModelNaming(model_name=model_name)
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

    if naming_info.model_family in ("gemma", "gemma1p1", "gemma2", "gemma3"):
      expected_module_type = automodel.ModelModule.PARAMS_SAFETENSORS
    else:
      expected_module_type = automodel.ModelModule.PARAMS

    mock_get_model_module.assert_called_once_with(
        model_name, expected_module_type
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="gemma-2b",
          model_name="gemma-2b",
          expected_version="2b",
      ),
      dict(
          testcase_name="gemma2-2b-it",
          model_name="gemma2-2b-it",
          expected_version="2-2b_it",
      ),
      dict(
          testcase_name="gemma-2-2b-it",
          model_name="gemma-2-2b-it",
          expected_version="2-2b_it",
      ),
      dict(
          testcase_name="gemma1.1-2b-it",
          model_name="gemma1.1-2b-it",
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


if __name__ == "__main__":
  absltest.main()
