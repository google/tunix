# Copyright 2025 Google LLC
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

"""Inference worker for RL."""

from flax import nnx
import jax
from tunix.rl import common


class InferenceWorker:
  """Inference worker hosting critic, reference and reward models."""

  def __init__(self, models: dict[str, nnx.Module]):
    for k in models.keys():
      if k not in ["critic", "reference", "reward"]:
        raise ValueError(
            f"Model role {k} is not supported. Supported models are critic,"
            " reference and reward."
        )
    self._models = models
    # TODO(tsbao): support multiple reward models.

  def compute_rewards(self):
    raise NotImplementedError()

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ):
    ref_model = self._models.get("reference")
    if ref_model is None:
      raise ValueError("Reference model is not available.")
    return common.compute_per_token_logps(
        ref_model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def compute_values(self):
    raise NotImplementedError()
