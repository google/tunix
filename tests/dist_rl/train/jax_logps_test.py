# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for JAX/NNX per-token log-probability helpers."""

import os
from typing import Any

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import abstract_trainer
from tunix.dist_rl.train import jax_logps
from tunix.rl import common as rl_common
from tunix.tests import test_common as tc

# CPU environment setup to simulate multi-device env so pytest collection does
# not initialize JAX with fewer devices than sibling dist_rl tests require.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


class _CallbackTrainer:

  def __init__(self, model: nnx.Module):
    self.model = model
    self._fwd_only_fn = None
    self.fwd_only_calls = []

  def with_fwd_only_fn(self, fn):
    self._fwd_only_fn = fn
    return self

  def fwd_only(self, payload):
    if self._fwd_only_fn is None:
      raise ValueError("with_fwd_only_fn must be called first.")
    self.fwd_only_calls.append(payload)
    return self._fwd_only_fn(self.model, payload)


class _LegacyVariadicTrainer:

  def __init__(self, model: nnx.Module):
    self.model = model
    self.fwd_only_calls = []

  def fwd_only(self, fn, *args, **kwargs):
    self.fwd_only_calls.append((fn, args, kwargs))
    return fn(self.model, *args, **kwargs)


class _UnimplementedLogpsTrainer(abstract_trainer.AbstractTrainer):

  def __init__(self, config: Any = None):
    del config

  def with_loss_fn(self, loss_fn, has_aux=False):
    del loss_fn, has_aux
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn):
    del gen_model_input_fn
    return self

  def compile(self, dummy_data=None):
    del dummy_data

  def fwd_bwd(self, payload):
    del payload

  def update(self):
    return 0

  def eval_step(self, payload):
    del payload

  def with_fwd_only_fn(self, fn):
    del fn
    return self

  def fwd_only(self, payload):
    return payload

  def save_checkpoint(self, metadata=None, *, step=None, force=False):
    del metadata, step, force

  def restore_checkpoint(self, *, step=None):
    del step
    return {}

  def prepare_weight_sync(self, *, sync_request=None):
    del sync_request
    return None

  def get_metrics(self):
    return datatypes.MetricsBuffer(id="dummy")

  def close(self):
    pass


class JaxLogpsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.model = tc.ToyTransformer(
        config=tc.ModelConfig(vocab_size=16), rngs=nnx.Rngs(0)
    )
    self.prompt = np.array([[1, 2], [3, 4]], dtype=np.int32)
    self.completion = np.array([[5, 6, 7], [8, 9, 0]], dtype=np.int32)

  def _expected_logps(
      self,
      *,
      temperature: float = 0.8,
      chunk_size: int = 0,
      segment_ids: np.ndarray | None = None,
      segment_positions: np.ndarray | None = None,
  ) -> np.ndarray:
    graphdef, state = nnx.split(self.model)
    return np.asarray(
        rl_common.compute_per_token_logps(
            graphdef,
            state,
            prompt_tokens=jnp.asarray(self.prompt),
            completion_tokens=jnp.asarray(self.completion),
            pad_id=0,
            eos_id=0,
            stop_gradient=True,
            temperature=temperature,
            chunk_size=chunk_size,
            segment_ids=(
                None if segment_ids is None else jnp.asarray(segment_ids)
            ),
            segment_positions=(
                None
                if segment_positions is None
                else jnp.asarray(segment_positions)
            ),
        ),
        dtype=np.float32,
    )

  def test_compute_per_token_logps_from_payload_matches_reference(self):
    seg_ids = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=np.int32)
    seg_pos = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 0]], dtype=np.int32)
    payload = datatypes.LogprobsScoringPayload(
        prompt_ids=self.prompt,
        completion_ids=self.completion,
        pad_id=0,
        eos_id=0,
        temperature=0.8,
        segment_ids=seg_ids,
        segment_positions=seg_pos,
    )
    out = jax_logps.compute_per_token_logps_from_payload(
        self.model, payload, chunk_size=4
    )
    expected = self._expected_logps(
        temperature=0.8,
        chunk_size=4,
        segment_ids=seg_ids,
        segment_positions=seg_pos,
    )
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-4, atol=1e-4)

  def test_compute_nnx_per_token_logps_converts_numpy_inputs(self):
    out = jax_logps.compute_nnx_per_token_logps(
        self.model,
        self.prompt,
        self.completion,
        pad_id=0,
        eos_id=0,
        temperature=0.8,
        chunk_size=0,
    )
    np.testing.assert_allclose(
        np.asarray(out),
        self._expected_logps(temperature=0.8),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_compute_trainer_per_token_logps_rebinds_after_custom_fwd_only_fn(
      self,
  ):
    trainer = _CallbackTrainer(self.model)
    expected = self._expected_logps(temperature=0.8, chunk_size=0)

    out1 = jax_logps.compute_trainer_per_token_logps(
        trainer,
        self.prompt,
        self.completion,
        pad_id=0,
        eos_id=0,
        temperature=0.8,
        chunk_size=0,
    )
    np.testing.assert_allclose(
        np.asarray(out1), expected, rtol=1e-4, atol=1e-4
    )

    # Overwrite the forward-only callback externally, then verify calling
    # `compute_trainer_per_token_logps` with the SAME `chunk_size=0` detects the
    # callback change and re-binds the log-probability scorer instead of running
    # the custom callback.
    trainer.with_fwd_only_fn(lambda model, payload: jnp.full((2, 3), 99.0))
    custom_out = trainer.fwd_only(trainer.fwd_only_calls[-1])
    np.testing.assert_allclose(np.asarray(custom_out), np.full((2, 3), 99.0))

    out2 = jax_logps.compute_trainer_per_token_logps(
        trainer,
        self.prompt,
        self.completion,
        pad_id=0,
        eos_id=0,
        temperature=0.8,
        chunk_size=0,
    )
    np.testing.assert_allclose(
        np.asarray(out2), expected, rtol=1e-4, atol=1e-4
    )

  def test_compute_trainer_per_token_logps_legacy_fallback(self):
    trainer = _LegacyVariadicTrainer(self.model)
    out = jax_logps.compute_trainer_per_token_logps(
        trainer,
        self.prompt,
        self.completion,
        pad_id=0,
        eos_id=0,
        temperature=0.8,
    )
    self.assertLen(trainer.fwd_only_calls, 1)
    np.testing.assert_allclose(
        np.asarray(out),
        self._expected_logps(temperature=0.8),
        rtol=1e-4,
        atol=1e-4,
    )

  def test_abstract_trainer_default_compute_per_token_logps_raises(self):
    trainer = _UnimplementedLogpsTrainer()
    with self.assertRaisesRegex(
        NotImplementedError, "does not implement compute_per_token_logps"
    ):
      trainer.compute_per_token_logps(
          self.prompt, self.completion, pad_id=0, eos_id=0
      )


if __name__ == "__main__":
  absltest.main()
