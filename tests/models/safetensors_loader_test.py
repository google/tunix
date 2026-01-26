"""Tests for safetensors_loader."""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from safetensors import numpy as stnp
from tunix.models import safetensors_loader
from tunix.tests import test_common


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
      cls.tensors[f'layers.{i}.w2.bias'] = np.array(
          layer_state['w2']['bias'].value
      )

  @parameterized.named_parameters(
      dict(testcase_name='absolute_path', path_type='abs'),
      dict(testcase_name='relative_path', path_type='rel'),
      dict(testcase_name='relative_dot_path', path_type='rel_dot'),
  )
  def test_load_and_create_model(self, path_type):
    try:
      st_dir_abs = self.create_tempdir().full_path
    except Exception:  # pylint: disable=broad-except
      st_dir_abs = tempfile.TemporaryDirectory().name
      os.makedirs(st_dir_abs, exist_ok=True)

    origin_dir = os.getcwd()
    self.addCleanup(os.chdir, origin_dir)
    if path_type == 'abs':
      load_dir = st_dir_abs
    elif path_type == 'rel':
      os.chdir(os.path.dirname(st_dir_abs))
      load_dir = os.path.basename(st_dir_abs)
    elif path_type == 'rel_dot':
      os.chdir(os.path.dirname(st_dir_abs))
      load_dir = f'./{os.path.basename(st_dir_abs)}'
    else:
      raise ValueError(f'Unknown path_type: {path_type}')

    filename = os.path.join(st_dir_abs, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    loaded_model = safetensors_loader.load_and_create_model(
        load_dir,
        test_common.ToyTransformer,
        self.model.config,
        key_mapping,
        dtype=jnp.float32,
    )
    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_from_gcs(self):
    try:
      st_dir_abs = self.create_tempdir().full_path
    except Exception:  # pylint: disable=broad-except
      st_dir_abs = tempfile.TemporaryDirectory().name
      os.makedirs(st_dir_abs, exist_ok=True)

    filename = os.path.join(st_dir_abs, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    with mock.patch.object(
        safetensors_loader, 'load_file_from_gcs'
    ) as mock_load:
      mock_load.return_value = st_dir_abs
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


if __name__ == '__main__':
  absltest.main()
