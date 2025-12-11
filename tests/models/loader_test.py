from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.models import loader
from tunix.models import naming


def _get_all_models_test_parameters():
  return (
      dict(testcase_name="gemma-2b", model_name="gemma-2b"),
      dict(testcase_name="gemma-2b-it", model_name="gemma-2b-it"),
      dict(testcase_name="gemma-7b", model_name="gemma-7b"),
      dict(testcase_name="gemma-7b-it", model_name="gemma-7b-it"),
      dict(testcase_name="gemma1.1-2b-it", model_name="gemma1.1-2b-it"),
      dict(testcase_name="gemma1.1-7b-it", model_name="gemma1.1-7b-it"),
      dict(testcase_name="gemma2-2b", model_name="gemma2-2b"),
      dict(testcase_name="gemma2-2b-it", model_name="gemma2-2b-it"),
      dict(testcase_name="gemma2-9b", model_name="gemma2-9b"),
      dict(testcase_name="gemma2-9b-it", model_name="gemma2-9b-it"),
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
      dict(testcase_name="llama3-405b", model_name="llama3-405b"),
      dict(testcase_name="llama3.1-8b", model_name="llama3.1-8b"),
      dict(testcase_name="llama3.2-1b", model_name="llama3.2-1b"),
      dict(testcase_name="llama3.2-3b", model_name="llama3.2-3b"),
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
      dict(testcase_name="qwen3-8b", model_name="qwen3-8b"),
      dict(testcase_name="qwen3-14b", model_name="qwen3-14b"),
      dict(testcase_name="qwen3-30b", model_name="qwen3-30b"),
  )


class ModelLoaderTest(parameterized.TestCase):

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_obtain_model_params_valid(self, model_name: str):
    loader.call_model_config(model_name)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_get_model_module_valid(self, model_name: str):
    if model_name.startswith("gemma"):
      params_module = loader.get_model_module(
          model_name, loader.ModelModule.PARAMS_SAFETENSORS
      )
    else:
      params_module = loader.get_model_module(
          model_name, loader.ModelModule.PARAMS
      )
    getattr(params_module, "create_model_from_safe_tensors")

    model_lib_module = loader.get_model_module(
        model_name, loader.ModelModule.MODEL
    )
    getattr(model_lib_module, "ModelConfig")

  def test_get_model_module_invalid(self):
    with self.assertRaises(ValueError):
      loader.get_model_module("invalid-model", loader.ModelModule.PARAMS)

  @parameterized.named_parameters(*_get_all_models_test_parameters())
  def test_create_model_dynamically(self, model_name: str):
    if naming.get_model_config_category(model_name) in ["gemma", "gemma3"]:
      self.skipTest(
          "Gemma models do not support create_model_from_safe_tensors"
      )
    mock_create_fn = mock.Mock()
    mock_params_module = mock.Mock()
    mock_params_module.create_model_from_safe_tensors = mock_create_fn
    mock_params_module.__name__ = "mock_params_module"
    with mock.patch(
        "tunix.models.loader.get_model_module", return_value=mock_params_module
    ):
      mesh = jax.sharding.Mesh(jax.devices(), ("devices",))
      loader.create_model_from_safe_tensors(
          model_name, "file_dir", "model_config", mesh
      )
      mock_create_fn.assert_called_once_with(
          file_dir="file_dir", config="model_config", mesh=mesh
      )


if __name__ == "__main__":
  absltest.main()
