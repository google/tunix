from absl.testing import absltest
import jax
import jax.numpy as jnp
from tunix.generate import beam_search as beam_search_lib
from tunix.generate import sampler as sampler_lib


class BeamSearchTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.batch_size = 2
    self.beam_size = 2
    self.vocab_size = 3
    self.seq_length = 10
    self.pad_token_id = jnp.array([-1])

  def _check_content_equal_per_beam(
      self, beam_content: jnp.ndarray, original_content: jnp.ndarray
  ) -> None:
    self.assertEqual(
        beam_content.shape[0], original_content.shape[0] * self.beam_size
    )
    if len(original_content.shape) > 1:
      self.assertEqual(beam_content.shape[1:], original_content.shape[1:])
      for i in range(0, self.batch_size * self.beam_size, self.beam_size):
        self.assertTrue(
            jnp.allclose(beam_content[i * self.beam_size], original_content[i])
        )

  def test_initialization(self) -> None:
    cache = sampler_lib._init_cache(
        n_layers=2,
        cache_size=16,
        batch_size=self.batch_size,
        num_kv_heads=2,
        head_dim=128,
        dtype=jnp.float32,
    )
    token_buffer = jnp.arange(self.batch_size * self.vocab_size).reshape(
        (self.batch_size, self.vocab_size)
    )
    done = jnp.zeros((self.batch_size), dtype=jnp.bool)
    positions = jnp.arange(self.batch_size * self.seq_length).reshape(
        (self.batch_size, self.seq_length)
    )
    logits = jnp.zeros((self.batch_size, self.vocab_size))
    state, updated_params = beam_search_lib.init_batched_beam_state(
        logits=logits,
        input_token_buffer=token_buffer,
        initial_cache=cache,
        positions=positions,
        done=done,
        logits_buffer=None,
        beam_size=self.beam_size,
    )
    self.assertFalse(state.initialized)
    self.assertEqual(state.scores.shape, (self.batch_size, self.beam_size))
    jax.tree.map(
        self._check_content_equal_per_beam,
        updated_params['cache'],
        cache,
    )
    self._check_content_equal_per_beam(
        updated_params['token_buffer'], token_buffer
    )
    self._check_content_equal_per_beam(updated_params['done'], done)
    self._check_content_equal_per_beam(updated_params['logits'], logits)
    self._check_content_equal_per_beam(
        updated_params['positions'],
        positions,
    )

  def test_beam_search_step_without_stop(self) -> None:
    # b0: [[0.2, 0.35, 0.45]]
    decoding_step = -1
    original_cache = {
        'fake': {
            'dummy0': jnp.arange(self.batch_size).repeat(self.beam_size, axis=0)
        }
    }
    init_state = beam_search_lib._BeamSearchSamplingState(
        scores=jnp.zeros((self.batch_size, self.beam_size), dtype=jnp.float32),
        initialized=False,
    )
    done = jnp.zeros((self.batch_size * self.beam_size), dtype=jnp.bool)
    token_buffer = jnp.full(
        (self.batch_size * self.beam_size, self.seq_length),
        self.pad_token_id,
        dtype=jnp.int32,
    )

    # The first step.
    state, updated_params = beam_search_lib.beam_search_step(
        logits=jnp.array(
            [[0.2, 0.35, 0.45], [0.2, 0.35, 0.45], [1, 2, 1.1], [1, 2, 1.1]]
        ),
        done=done,
        token_buffer=token_buffer,
        cache=original_cache,
        logits_buffer=None,
        state=init_state,
        pad_token_id=self.pad_token_id,
        decoding_step=decoding_step,
    )
    new_scores0, tokens0 = jax.lax.top_k(
        jax.nn.log_softmax(jnp.array([0.2, 0.35, 0.45])), 2
    )
    new_scores1, tokens1 = jax.lax.top_k(
        jax.nn.log_softmax(jnp.array([1, 2, 1.1])), 2
    )
    self.assertAlmostEqual(state.scores[0][0], new_scores0[0], places=6)
    self.assertAlmostEqual(state.scores[0][1], new_scores0[1], places=6)
    self.assertAlmostEqual(state.scores[1][0], new_scores1[0], places=6)
    self.assertAlmostEqual(state.scores[1][1], new_scores1[1], places=6)

    updated_token_buffer = updated_params['token_buffer']
    expected = token_buffer
    expected = expected.at[0, 0].set(tokens0[0])
    expected = expected.at[1, 0].set(tokens0[1])
    expected = expected.at[2, 0].set(tokens1[0])
    expected = expected.at[3, 0].set(tokens1[1])
    self.assertTrue(jnp.allclose(updated_token_buffer, expected))
    self.assertTrue(jnp.allclose(updated_params['done'], done))

    def _check(x, y) -> None:
      self.assertTrue(jnp.allclose(x, y))

    jax.tree.map(
        _check,
        updated_params['cache'],
        original_cache,
    )

    # The second step.
    decoding_step += 1
    state, updated_params = beam_search_lib.beam_search_step(
        logits=jnp.array(
            [[0.2, 0.36, 0.35], [0.2, 0.75, 0.35], [1, 2, 1.1], [1, 2, 1.9]]
        ),
        done=updated_params['done'],
        token_buffer=updated_params['token_buffer'],
        cache=updated_params['cache'],
        logits_buffer=None,
        state=state,
        pad_token_id=self.pad_token_id,
        decoding_step=decoding_step,
    )
    new_scores0, _ = jax.lax.top_k(
        (
            new_scores0[:, None]
            + jax.nn.log_softmax(
                jnp.array([[0.2, 0.36, 0.35], [0.2, 0.75, 0.35]])
            )
        ).ravel(),
        2,
    )
    new_scores1, _ = jax.lax.top_k(
        (
            new_scores1[:, None]
            + jax.nn.log_softmax(jnp.array([[1, 2, 1.1], [1, 2, 1.9]]))
        ).ravel(),
        2,
    )
    self.assertAlmostEqual(state.scores[0][0], new_scores0[0], places=6)
    self.assertAlmostEqual(state.scores[0][1], new_scores0[1], places=6)
    self.assertAlmostEqual(state.scores[1][0], new_scores1[0], places=6)
    self.assertAlmostEqual(state.scores[1][1], new_scores1[1], places=6)
    # before the beam search, the token buffer[:][0] is [2, 1, 1, 2]
    # token_buffer[0] should be [1, 1, -1, ...]
    # token_buffer[1] should be [2, 1, -1, ...]
    # token_buffer[2] should be [1, 1, -1, ...]
    # token_buffer[3] should be [1, 2, -1, ...]
    # after the beam search, the token buffer[:][0] is changes to [1, 2, 1, 1]
    self.assertTrue(
        jnp.allclose(
            updated_params['token_buffer'][:, 0], jnp.array([1, 2, 1, 1])
        )
    )
    self.assertTrue(
        jnp.allclose(
            updated_params['token_buffer'][:, 1], jnp.array([1, 1, 1, 2])
        )
    )
    self.assertTrue(
        jnp.allclose(
            updated_params['token_buffer'][:, 2:],
            jnp.full(
                (self.batch_size * self.beam_size, self.seq_length - 2),
                self.pad_token_id,
            ),
        )
    )
    self.assertTrue(jnp.allclose(updated_params['done'], done))
    original_check_items = original_cache['fake']['dummy0']
    current_check_items = updated_params['cache']['fake']['dummy0']
    # assuming input cache is with order [0, 1, 2, 3], it should change to
    # [1, 0, 2, 2]
    self.assertEqual(current_check_items[0], original_check_items[1])
    self.assertEqual(current_check_items[1], original_check_items[0])
    self.assertEqual(current_check_items[2], original_check_items[2])
    self.assertEqual(current_check_items[3], original_check_items[2])

    final_output = beam_search_lib.finalize_beam_search_state(
        state,
        updated_params['token_buffer'],
        None,
    )['token_buffer']
    self.assertEqual(final_output.shape, (self.batch_size, self.seq_length))
    self.assertTrue(
        jnp.allclose(final_output[0], updated_params['token_buffer'][0])
    )
    self.assertTrue(
        jnp.allclose(
            final_output[1], updated_params['token_buffer'][self.beam_size]
        )
    )

  def test_beam_search_step_with_stop_picked(self) -> None:
    self.batch_size = 1
    initial_scores = jnp.zeros(
        (self.batch_size, self.beam_size), dtype=jnp.float32
    )
    initial_scores = initial_scores.at[0, 0].set(0.2)
    initial_scores = initial_scores.at[0, 1].set(-2)
    done = jnp.array([True, False])  # mark the first beam as done.
    token_buffer = jnp.full(
        (self.batch_size * self.beam_size, self.seq_length),
        self.pad_token_id,
        dtype=jnp.int32,
    )
    token_buffer = token_buffer.at[0, 0].set(0)
    token_buffer = token_buffer.at[1, 0].set(1)
    decoding_step = 0
    cache = {
        'fake': {
            'dummy0': jnp.arange(self.batch_size).repeat(self.beam_size, axis=0)
        }
    }
    state = beam_search_lib._BeamSearchSamplingState(
        scores=initial_scores,
        initialized=True,
    )
    for _ in range(5):
      state, updated_params = beam_search_lib.beam_search_step(
          logits=jnp.array([[0.2, 0.35, 0.45], [1, 2, 1.1]]),
          done=done,
          token_buffer=token_buffer,
          cache=cache,
          logits_buffer=None,
          state=state,
          pad_token_id=self.pad_token_id,
          decoding_step=decoding_step,
      )
      done = updated_params['done']
      token_buffer = updated_params['token_buffer']
      cache = updated_params['cache']
      decoding_step += 1
    self.assertTrue(done[0])
    self.assertEqual(token_buffer[0, 0], 0)
    self.assertTrue(
        jnp.allclose(
            token_buffer[0, 1:],
            jnp.array([self.pad_token_id] * (self.seq_length - 1)),
        )
    )
    self.assertFalse(done[1])
    for i in range(0, decoding_step + 1):
      self.assertNotEqual(token_buffer[1, i], self.pad_token_id)
    for i in range(decoding_step + 1, self.seq_length):
      self.assertEqual(token_buffer[1, i], self.pad_token_id)

  def test_beam_search_step_with_stop_not_picked(self) -> None:
    self.batch_size = 1
    initial_scores = jnp.zeros(
        (self.batch_size, self.beam_size), dtype=jnp.float32
    )
    initial_scores = initial_scores.at[0, 0].set(-100)
    initial_scores = initial_scores.at[0, 1].set(-2)
    done = jnp.array([True, False])  # mark the first beam as done.
    token_buffer = jnp.full(
        (self.batch_size * self.beam_size, self.seq_length),
        self.pad_token_id,
        dtype=jnp.int32,
    )
    token_buffer = token_buffer.at[0, 0].set(0)
    token_buffer = token_buffer.at[1, 0].set(1)
    decoding_step = 0
    cache = {
        'fake': {
            'dummy0': jnp.arange(self.batch_size).repeat(self.beam_size, axis=0)
        }
    }
    state = beam_search_lib._BeamSearchSamplingState(
        scores=initial_scores,
        initialized=True,
    )
    _, updated_params = beam_search_lib.beam_search_step(
        logits=jnp.array([[0.2, 0.35, 0.45], [1, 2, 1.1]]),
        done=done,
        token_buffer=token_buffer,
        cache=cache,
        logits_buffer=None,
        state=state,
        pad_token_id=self.pad_token_id,
        decoding_step=decoding_step,
    )
    done = updated_params['done']
    self.assertFalse(done[0])
    self.assertFalse(done[1])


if __name__ == '__main__':
  absltest.main()
