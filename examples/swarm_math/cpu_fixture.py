# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small trainable categorical policy for CPU correctness, not a math model.

Sampling uses the current shared NNX parameters, with recorded action logps.
The deliberately tiny vocabulary has boxed answers and verifier verdicts so the
same production workflow, grader, credit and trainer can execute end to end.
"""

from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib

VOCAB = (
    "",
    "reason",
    "solve",
    "check",
    r"\boxed{2}",
    r"\boxed{3}",
    "<verify>approve</verify>",
    "<verify>reject</verify>",
)


class TinyPolicy(nnx.Module):

  def __init__(self, logits=None):
    if logits is None:
      logits = np.full((len(VOCAB), len(VOCAB)), -3.0, dtype=np.float32)
      logits[:, 0] = 2.0
      logits[2, :] = -4.0
      logits[2, 4:6] = 1.0
      logits[3, :] = -4.0
      logits[3, 6:8] = 1.0
    self.logits = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))

  def __call__(
      self, tokens, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    del positions, attention_mask, segment_ids
    return self.logits[tokens], cache


def encode(prompt):
  role = (
      3
      if 'You are a "Verifier Agent"' in prompt.rsplit("# Your Role", 1)[-1]
      else 2
  )
  # A real ID-0 in the prompt explicitly exercises attention when pad=eos=0.
  context = [i for i, word in enumerate(VOCAB) if i >= 4 and word in prompt]
  return [1, 0] + context + [role]


class TinySampler:

  def __init__(self, policy):
    self.policy = policy

  async def sample(self, request):
    params = request.sampling_params
    logits = np.asarray(self.policy.logits[...], dtype=np.float64)
    logits /= params.temperature
    shifted = logits - logits.max(axis=-1, keepdims=True)
    logps = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    rng = np.random.default_rng(params.seed)
    last = int(request.prompt[-1])
    tokens, scores = [], []
    for _ in range(params.max_tokens):
      distribution = np.exp(logps[last])
      if params.top_p is not None and params.top_p < 1.0:
        order = np.argsort(-distribution)
        keep = (
            np.cumsum(distribution[order]) - distribution[order] < params.top_p
        )
        distribution[order[~keep]] = 0.0
        distribution /= distribution.sum()
      token = int(rng.choice(len(VOCAB), p=distribution))
      tokens.append(token)
      scores.append(float(logps[last, token]))
      last = token
      if token == 0:
        break
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt, dtype=np.int32),
        token_ids=np.asarray(tokens, dtype=np.int32),
        logprobs=np.asarray(scores, dtype=np.float32),
        text=" ".join(VOCAB[t] for t in tokens if t != 0),
        finish_reason="stop" if last == 0 else "length",
    )
