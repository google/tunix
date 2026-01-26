"""Tests for torch_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from tunix.utils import torch_utils


class TorchUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='simple_case',
          mapping={
              'layer.(\\d+).mlp': (r'jax_layer.\1.jax_mlp', None),
          },
          source_key='layer.12.mlp',
          expected=('jax_layer.12.jax_mlp', None),
      ),
  )
  def test_torch_key_to_jax_key_success(self, mapping, source_key, expected):
    got = torch_utils.torch_key_to_jax_key(mapping, source_key)
    self.assertEqual(expected, got)

  @parameterized.named_parameters(
      dict(
          testcase_name='no_match',
          mapping={
              'layer.(\\d+).mlp': ('jax_layer.{}.jax_mlp', None),
          },
          source_key='other_layer.12.mlp',
      ),
      dict(
          testcase_name='multiple_matches',
          mapping={
              'layer.(\\d+).mlp': ('jax_layer.{}.jax_mlp', None),
              'layer.*': ('jax_layer.all', None),
          },
          source_key='layer.12.mlp',
      ),
  )
  def test_torch_key_to_jax_key_failure(self, mapping, source_key):
    with self.assertRaises(ValueError):
      torch_utils.torch_key_to_jax_key(mapping, source_key)


if __name__ == '__main__':
  absltest.main()
