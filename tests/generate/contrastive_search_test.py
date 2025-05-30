from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
from tunix.generate import contrastive_search as contrastive_search_lib


class ContrastiveSearchTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='alpha_0',
          alpha=0.0,
          expected_id=1,
          expected_hidden_states=[1.0, 0.0, 0.0],
      ),
      dict(
          testcase_name='alpha_1',
          alpha=1.0,
          expected_id=2,
          expected_hidden_states=[0.0, 0.0, 0.0],
      ),
      dict(
          testcase_name='alpha_2',
          alpha=0.6,
          expected_id=2,
          expected_hidden_states=[0.0, 0.0, 0.0],
      ),
  )
  def test_ranking(
      self, alpha: float, expected_id: int, expected_hidden_states: list[int]
  ) -> None:
    mock_context_hidden = jnp.array([
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [100.0, 100.0, 100.0],  # should be masked
        ],
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [100.0, 100.0, 100.0],  # should be masked
        ],
    ])
    mock_next_hidden = jnp.array([
        [
            [1.0, 0.0, 0.0],  # cosine similarity: 1.0
        ],
        [
            [0.0, 0.0, 0.0],  # cosine similarity: 0.0
        ],
    ])
    mock_next_top_k_ids = jnp.array([
        [1],
        [2],
    ])
    mock_next_top_k_probs = jnp.array([
        [0.5],
        [0.5],
    ])
    selected_id, hidden_state = contrastive_search_lib.ranking(
        mock_context_hidden,
        mock_next_hidden,
        mock_next_top_k_ids,
        mock_next_top_k_probs,
        3,
        alpha,
    )
    self.assertEqual(selected_id, expected_id)
    self.assertEqual(hidden_state.shape, (3,))
    self.assertTrue(
        jnp.allclose(hidden_state, jnp.array(expected_hidden_states))
    )


if __name__ == '__main__':
  absltest.main()
