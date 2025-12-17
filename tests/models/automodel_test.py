from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.models import automodel


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
      dict(testcase_name="gemma3-1b", model_name="gemma3-1b"),
      dict(testcase_name="gemma3-1b-it", model_name="gemma3-1b-it"),
      dict(testcase_name="gemma3-4b", model_name="gemma3-4b"),
      dict(testcase_name="gemma3-4b-it", model_name="gemma3-4b-it"),
      dict(testcase_name="gemma3-12b", model_name="gemma3-12b"),
      dict(testcase_name="gemma3-12b-it", model_name="gemma3-12b-it"),
      dict(testcase_name="gemma3-27b", model_name="gemma3-27b"),
      dict(testcase_name="gemma3-27b-it", model_name="gemma3-27b-it"),
      dict(testcase_name="gemma-3-270m", model_name="gemma-3-270m"),
      dict(testcase_name="gemma-3-270m-it", model_name="gemma-3-270m-it"),
      dict(testcase_name="gemma-3-1b", model_name="gemma-3-1b"),
      dict(testcase_name="gemma-3-1b-it", model_name="gemma-3-1b-it"),
      dict(testcase_name="gemma-3-4b", model_name="gemma-3-4b"),
      dict(testcase_name="gemma-3-4b-it", model_name="gemma-3-4b-it"),
      dict(testcase_name="gemma-3-12b", model_name="gemma-3-12b"),
      dict(testcase_name="gemma-3-12b-it", model_name="gemma-3-12b-it"),
      dict(testcase_name="gemma-3-27b", model_name="gemma-3-27b"),
      dict(testcase_name="gemma-3-27b-it", model_name="gemma-3-27b-it"),
      dict(testcase_name="llama3-70b", model_name="llama3-70b"),
      dict(testcase_name="llama-3-70b", model_name="llama-3-70b"),
      dict(testcase_name="llama3.1-405b", model_name="llama3.1-405b"),
      dict(testcase_name="llama-3.1-405b", model_name="llama-3.1-405b"),
      dict(testcase_name="llama3.1-8b", model_name="llama3.1-8b"),
      dict(testcase_name="llama-3.1-8b", model_name="llama-3.1-8b"),
      dict(testcase_name="llama3.2-1b", model_name="llama3.2-1b"),
      dict(testcase_name="llama-3.2-1b", model_name="llama-3.2-1b"),
      dict(testcase_name="llama3.2-3b", model_name="llama3.2-3b"),
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
    automodel.create_model_from_safe_tensors(
        model_name, "file_dir", "model_config", mesh
    )
    mock_create_fn.assert_called_once_with(
        file_dir="file_dir", config="model_config", mesh=mesh
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


if __name__ == "__main__":
  absltest.main()
