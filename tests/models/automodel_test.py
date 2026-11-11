import dataclasses
import os
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import numpy as np
import orbax.checkpoint as ocp
from tunix.models import automodel
from tunix.models import naming
from tunix.models.gemma import model as gemma_model
from tunix.models.gemma import params as gemma_params
from tunix.oss import utils as oss_utils
from tunix.utils import env_utils


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

    if naming_info.model_family in (
        "gemma",
        "gemma1p1",
        "gemma2",
        "gemma3",
        "gemma4",
    ):
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
  @mock.patch("orbax.checkpoint.StandardCheckpointer", autospec=True)
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


if __name__ == "__main__":
  absltest.main()
