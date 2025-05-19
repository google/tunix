# Copyright 2025 Google LLC
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

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.distillation.feature_extraction import pooling


class AvgPoolArrayToTargetShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      # VALID padding tests
      dict(
          testcase_name='valid_1d_exact_div',
          input_shape=(10,),
          target_shape=(5,),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_1d_to_one',
          input_shape=(10,),
          target_shape=(1,),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_1d_no_change',
          input_shape=(5,),
          target_shape=(5,),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_1d_general_k4_s2',
          input_shape=(10,),
          target_shape=(4,),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_1d_general_k4_s3',
          input_shape=(10,),
          target_shape=(3,),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_2d_exact_div',
          input_shape=(4, 5),
          target_shape=(2, 5),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_2d_mixed_reduction',
          input_shape=(6, 10),
          target_shape=(3, 5),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='valid_3d_reduction',
          input_shape=(8, 6, 4),
          target_shape=(4, 3, 2),
          padding_mode=pooling.PaddingMode.VALID,
          count_include_pad_for_same_padding=False,
      ),
      # SAME padding tests
      dict(
          testcase_name='same_1d_exact_div',
          input_shape=(10,),
          target_shape=(5,),
          padding_mode=pooling.PaddingMode.SAME,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='same_1d_general',
          input_shape=(10,),
          target_shape=(3,),
          padding_mode=pooling.PaddingMode.SAME,
          count_include_pad_for_same_padding=False,
      ),
      dict(
          testcase_name='same_2d_reduction',
          input_shape=(4, 5),
          target_shape=(2, 3),
          padding_mode=pooling.PaddingMode.SAME,
          count_include_pad_for_same_padding=False,
      ),
  )
  def test_valid_reductions_shape(
      self,
      input_shape: tuple[int, ...],
      target_shape: tuple[int, ...],
      padding_mode: pooling.PaddingMode,
      count_include_pad_for_same_padding: bool,
  ):
    """Tests if the function produces the correct output shape for valid inputs."""
    result = pooling.avg_pool_array_to_target_shape(
        input_array=jnp.arange(np.prod(input_shape), dtype=jnp.float32).reshape(
            input_shape
        ),
        target_shape=target_shape,
        padding_mode=padding_mode,
        count_include_pad_for_same_padding=count_include_pad_for_same_padding,
    )
    self.assertEqual(result.shape, target_shape)
    result.block_until_ready()

  def test_same_padding_count_include_pad_effect_on_values(self):
    input_arr = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32)
    target_shape = (2,)

    result_true = pooling.avg_pool_array_to_target_shape(
        input_array=input_arr,
        target_shape=target_shape,
        padding_mode=pooling.PaddingMode.SAME,
        count_include_pad_for_same_padding=True,
    )
    np.testing.assert_allclose(
        result_true, jnp.array([2.0, 3.0], dtype=jnp.float32), rtol=1e-6
    )

    result_false = pooling.avg_pool_array_to_target_shape(
        input_array=input_arr,
        target_shape=target_shape,
        padding_mode=pooling.PaddingMode.SAME,
        count_include_pad_for_same_padding=False,
    )
    np.testing.assert_allclose(
        result_false, jnp.array([2.0, 4.5], dtype=jnp.float32), rtol=1e-6
    )

  @parameterized.named_parameters(
      dict(
          testcase_name='rank_mismatch',
          input_shape=(10,),
          target_shape=(5, 5),
          error_msg_regex=(
              'Input array rank \\(1\\) must match target_shape rank \\(2\\)'
          ),
      ),
      dict(
          testcase_name='target_dim_too_large',
          input_shape=(2, 5),
          target_shape=(3, 5),
          error_msg_regex=(
              'Target dimension target_shape\\[0\\]=3 is invalid. Must be 1 <='
              ' target_dim <= input_dim \\(2\\).'
          ),
      ),
      dict(
          testcase_name='target_dim_lt_one',
          input_shape=(2, 5),
          target_shape=(2, 0),
          error_msg_regex=(
              'Target dimension target_shape\\[1\\]=0 is invalid. Must be 1 <='
              ' target_dim <= input_dim \\(5\\).'
          ),
      ),
  )
  def test_invalid_input_configurations_raise_value_error(
      self,
      input_shape: tuple[int, ...],
      target_shape: tuple[int, ...],
      error_msg_regex: str,
  ):
    with self.assertRaisesRegex(ValueError, error_msg_regex):
      pooling.avg_pool_array_to_target_shape(
          jnp.arange(np.prod(input_shape), dtype=jnp.float32).reshape(
              input_shape
          ),
          target_shape,
      )


if __name__ == '__main__':
  absltest.main()
