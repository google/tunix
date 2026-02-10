from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.models.gemma3 import token_utils


class TokenUtilsTest(absltest.TestCase):

  def test_add_extra_tokens_for_images_batch_size_1(self):
    text_tokens = jnp.array([
        [0, 0, 1, 2, 3, 4],
    ])

    out = token_utils.add_extra_tokens_for_images(
        tokens=text_tokens,
        max_num_images=1,
        num_tokens_per_image=2,
        start_of_image_token=3,
        end_of_image_token=9,
        soft_token_placeholder=8,
        double_new_line_token=7,
    )

    self.assertEqual(out.shape, (1, 11))

    expected_0 = jnp.array([0, 0, 1, 2, 7, 3, 8, 8, 9, 7, 4])
    np.testing.assert_array_equal(out[0], expected_0)

  def test_add_extra_tokens_for_images_batch_size_2(self):
    text_tokens = jnp.array([
        [0, 0, 1, 2, 3, 4],
        [0, 1, 3, 2, 3, 4],
    ])

    out = token_utils.add_extra_tokens_for_images(
        tokens=text_tokens,
        max_num_images=2,
        num_tokens_per_image=2,
        start_of_image_token=3,
        end_of_image_token=9,
        soft_token_placeholder=8,
        double_new_line_token=7,
    )

    self.assertEqual(out.shape, (2, 16))
    expected = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 1, 2, 7, 3, 8, 8, 9, 7, 4],
            [0, 1, 7, 3, 8, 8, 9, 7, 2, 7, 3, 8, 8, 9, 7, 4],
        ]
    )
    np.testing.assert_array_equal(out, expected)


if __name__ == '__main__':
  absltest.main()
