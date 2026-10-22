"""Tests for safetensors_loader."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors import numpy as stnp
from tunix.models import safetensors_loader
from tunix.tests import test_common
from tunix.utils import env_utils


class MockTorchTensor:
  def __init__(self, dtype_str='torch.float16', data=None):
    self.shape = (256, 16)
    self.dtype = type(
        'MockDtype', (), {'__str__': lambda self: dtype_str}
    )()

    if data is None:
      self._data = np.zeros(self.shape, dtype=np.float32)
    else:
      self._data = data

  def float(self):
    new_data = self._data.astype(np.float32)
    return MockTorchTensor(dtype_str='torch.float32', data=new_data)

  def detach(self):
    return self

  def cpu(self):
    return self

  def numpy(self):
    return self._data


def key_mapping(config):
  del config
  return {
      r'^emb\.embedding$': ('emb.embedding', None),
      r'^layers\.(\d+)\.attn\.query\.kernel$': (
          r'layers.\1.attn.query.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.key\.kernel$': (
          r'layers.\1.attn.key.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.value\.kernel$': (
          r'layers.\1.attn.value.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.out\.kernel$': (
          r'layers.\1.attn.out.kernel',
          None,
      ),
      r'^layers\.(\d+)\.w1\.kernel$': (r'layers.\1.w1.kernel', None),
      r'^layers\.(\d+)\.w1\.bias$': (r'layers.\1.w1.bias', None),
      r'^layers\.(\d+)\.w2\.kernel$': (r'layers.\1.w2.kernel', None),
      r'^layers\.(\d+)\.w2\.bias$': (r'layers.\1.w2.bias', None),
      r'^lm_head\.kernel$': ('lm_head.kernel', None),
      r'^lm_head\.bias$': ('lm_head.bias', None),
  }


class SafetensorsLoaderTest(parameterized.TestCase):
  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config = test_common.ModelConfig(num_layers=4, num_kv_heads=4, head_dim=16)
    cls.model = test_common.ToyTransformer(config=config, rngs=nnx.Rngs(0))

    cls.state = nnx.state(cls.model)
    cls.tensors = {
        'emb.embedding': np.array(cls.state['emb']['embedding'].value),
        'lm_head.kernel': np.array(cls.state['lm_head']['kernel'].value),
        'lm_head.bias': np.array(cls.state['lm_head']['bias'].value),
    }
    for i in range(cls.model.config.num_layers):
      layer_state = cls.state['layers'][i]
      cls.tensors[f'layers.{i}.attn.query.kernel'] = np.array(
          layer_state['attn']['query']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.key.kernel'] = np.array(
          layer_state['attn']['key']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.value.kernel'] = np.array(
          layer_state['attn']['value']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.out.kernel'] = np.array(
          layer_state['attn']['out']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.kernel'] = np.array(
          layer_state['w1']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.bias'] = np.array(
          layer_state['w1']['bias'].value
      )
      cls.tensors[f'layers.{i}.w2.kernel'] = np.array(
          layer_state['w2']['kernel'].value
      )
      # Test that nnx.Param are correctly handled.
      cls.tensors[f'layers.{i}.w2.bias'] = nnx.Param(
          np.array(layer_state['w2']['bias'].value),
      )

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.st_dir = self._temp_dir.name
    os.makedirs(self.st_dir, exist_ok=True)

  @parameterized.named_parameters(
      *(([dict(testcase_name='opt_loader_enabled', mode='optimized')]
         if not env_utils.is_internal_env() else []) + [
          dict(testcase_name='absolute_path', path_type='abs'),
          dict(testcase_name='relative_path', path_type='rel'),
          dict(testcase_name='relative_dot_path', path_type='rel_dot'),
          dict(testcase_name='opt_loader_disabled', mode='original'),
      ])
  )
  def test_load_and_create_model(
      self, path_type='abs', mode='auto'
  ):

    origin_dir = os.getcwd()
    self.addCleanup(os.chdir, origin_dir)
    if path_type == 'abs':
      load_dir = self.st_dir
    elif path_type == 'rel':
      os.chdir(os.path.dirname(self.st_dir))
      load_dir = os.path.basename(self.st_dir)
    elif path_type == 'rel_dot':
      os.chdir(os.path.dirname(self.st_dir))
      load_dir = f'./{os.path.basename(self.st_dir)}'
    else:
      raise ValueError(f'Unknown path_type: {path_type}')

    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    loaded_model = safetensors_loader.load_and_create_model(
        load_dir,
        test_common.ToyTransformer,
        self.model.config,
        key_mapping,
        dtype=jnp.float32,
        mode=mode,
    )
    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_from_gcs(self):
    if env_utils.is_internal_env():
      self.skipTest('GCS is not supported in GOOGLE_INTERNAL_PACKAGE_PATH')

    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    with mock.patch.object(
        safetensors_loader, 'load_file_from_gcs'
    ) as mock_load:
      mock_load.return_value = self.st_dir
      loaded_model = safetensors_loader.load_and_create_model(
          'gs://bucket/model',
          test_common.ToyTransformer,
          self.model.config,
          key_mapping,
          dtype=jnp.float32,
      )
      mock_load.assert_called_once_with('gs://bucket/model')

    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_raises_on_duplicate_key(self):
    # Two distinct source keys that both map to the same jax key. In the
    # 'original' loader these are processed on separate threads, so without a
    # lock around the duplicate-key check and write the second write can
    # silently overwrite the first (issue #1259). With the lock the duplicate
    # is always detected.

    tensors = {
        'lm_head.kernel': np.zeros((2, 2), dtype=np.float32),
        'lm_head.weight': np.zeros((2, 2), dtype=np.float32),
    }
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    def duplicate_key_mapping(config):
      del config
      return {
          r'^lm_head\.kernel$': ('lm_head.kernel', None),
          r'^lm_head\.weight$': ('lm_head.kernel', None),
      }

    # The duplicate ValueError is re-raised wrapped as RuntimeError by the
    # loader, so match on RuntimeError. mode='original' is required because the
    # default optimized path is single-threaded and does not run this check.
    with self.assertRaisesRegex(RuntimeError, 'Duplicate key'):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          duplicate_key_mapping,
          dtype=jnp.float32,
          mode='original',
      )

  def test_load_bfloat16_custom_dtype_avoids_ml_dtypes(self):
    bf16_data = np.random.rand(256, 16).astype(ml_dtypes.bfloat16)
    tensors = {'emb.embedding': bf16_data}
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    def simple_mapping(config):
      del config
      return {r'^emb\.embedding$': ('emb.embedding', None)}

    result = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        simple_mapping,
        dtype=jnp.bfloat16,
        mode='original',
    )
    state = nnx.state(result)
    self.assertEqual(state['emb']['embedding'].value.dtype, jnp.bfloat16)
    self.assertEqual(
        state['emb']['embedding'].value.aval.dtype,
        jnp.bfloat16,
    )
    np.testing.assert_allclose(
        state['emb']['embedding'].value,
        bf16_data
    )

  def test_normalize_torch_bfloat16_safetensors(self):
    mock_tensor = MockTorchTensor(dtype_str='torch.bfloat16')
    result = safetensors_loader._normalize_tensor(mock_tensor)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, ml_dtypes.bfloat16)
    self.assertEqual(result.shape, (256, 16))

  def test_normalize_torch_float16_safetensors(self):
    f16_data = np.zeros((256, 16), dtype=np.float16)
    mock_tensor = MockTorchTensor(dtype_str='torch.float16', data=f16_data)

    result = safetensors_loader._normalize_tensor(mock_tensor)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.float16)

  def test_normalize_buffer_explicit_bfloat16(self):
    native_bf16 = np.zeros((256, 16), dtype=ml_dtypes.bfloat16)

    result = safetensors_loader._normalize_tensor(native_bf16)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, ml_dtypes.bfloat16)

  def test_normalize_buffer_unknown_16bit(self):
    native_unknown = np.zeros((256, 16), dtype='V2')

    result = safetensors_loader._normalize_tensor(native_unknown)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, ml_dtypes.bfloat16)

  def test_normalize_buffer_known_float16(self):
    native_f16 = np.zeros((256, 16), dtype=np.float16)

    result = safetensors_loader._normalize_tensor(native_f16)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.float16)

  def test_normalize_buffer_standard_float32(self):
    native_f32 = np.zeros((256, 16), dtype=np.float32)

    result = safetensors_loader._normalize_tensor(native_f32)

    self.assertIsInstance(result, np.ndarray)
    self.assertEqual(result.dtype, np.float32)

if __name__ == '__main__':
  absltest.main()
