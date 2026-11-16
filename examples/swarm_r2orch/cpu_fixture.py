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

"""Separate trainable Director and frozen Worker categorical fixtures for CPU.

Prompt state tokens (1..5) are disjoint from generated action/answer tokens
(6..12), so every sampled action token immediately transitions to EOS (token 0).
On canvas state 1 (empty), the Director adds `agent_1` (Solver, layer 0).
On canvas state 2 (`agent_1` ready), the Director mixes prematurely finishing
`agent_1` (wrong draft `0`) vs adding `agent_2` (Verifier, layer 1, fed by
`agent_1`, which produces `42`), giving genuine GRPO group variance.
"""

from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib

VOCAB = (
    "",
    "<state_empty>",
    "<state_agent1_ready>",
    "<state_agent2_ready>",
    "<worker_entry>",
    "<worker_downstream>",
    '{"action":"add_agent","role":"Solver","objective":"Derive initial answer","layer":0,"result_scope":"task_result"}',
    '{"action":"add_agent","role":"Verifier","objective":"Audit upstream derivation","layer":1,"inputs":["agent_1"],"result_scope":"task_result"}',
    '{"action":"finish","target":"agent_1"}',
    '{"action":"finish","target":"agent_2"}',
    '{"action":"invalid"}',
    r"Draft derivation. \boxed{0}",
    r"Verified derivation. \boxed{42}",
)


class CategoricalPolicy(nnx.Module):
  """Minimal bigram logit table compatible with `TrainerWorker`."""

  def __init__(self, logits: np.ndarray):
    self.logits = nnx.Param(jnp.asarray(logits, dtype=jnp.float32))

  def __call__(
      self, tokens, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    del positions, attention_mask, segment_ids
    return self.logits[tokens], cache


def make_director_policy() -> CategoricalPolicy:
  """Trainable Director policy over JSON canvas actions (tokens 6..10)."""
  n = len(VOCAB)
  logits = np.full((n, n), -6.0, dtype=np.float32)
  logits[:, 0] = 6.0  # All action tokens (6..12) immediately emit EOS (0).
  # Prompt state 1 (empty canvas) -> add Layer-0 Solver (token 6).
  logits[1, 0] = -6.0
  logits[1, 6] = 5.0
  # Prompt state 2 (agent_1 ready) -> mix add Verifier (7), early finish (8),
  # and invalid JSON (10) so each group of 4 rollouts has mixed rewards.
  logits[2, 0] = -6.0
  logits[2, 7] = 1.2
  logits[2, 8] = 1.0
  logits[2, 10] = 0.0
  # Prompt state 3 (agent_2 ready) -> finish agent_2 (token 9).
  logits[3, 0] = -6.0
  logits[3, 9] = 5.0
  return CategoricalPolicy(logits)


def make_frozen_worker_policy() -> CategoricalPolicy:
  """Frozen Worker policy: solo Solver errs (`0`), Layer-1 Verifier solves (`42`)."""
  n = len(VOCAB)
  logits = np.full((n, n), -8.0, dtype=np.float32)
  logits[:, 0] = 6.0
  # Prompt state 4 (entry-layer Solver) -> emits wrong draft (token 11).
  logits[4, 0] = -8.0
  logits[4, 11] = 6.0
  # Prompt state 5 (downstream Verifier seeing `[agent_1 output]`) -> emits 42 (token 12).
  logits[5, 0] = -8.0
  logits[5, 12] = 6.0
  return CategoricalPolicy(logits)


def encode_director(prompt: str) -> list[int]:
  """Maps Canvas snapshot state to conditioning token IDs (with pad/eos ID 0)."""
  if '"id":"agent_2"' in prompt:
    state = 3
  elif '"id":"agent_1"' in prompt:
    state = 2
  else:
    state = 1
  return [1, 0, state]


def encode_worker(prompt: str) -> list[int]:
  """Maps Worker upstream context to entry-layer (4) or downstream (5) state."""
  state = 5 if "[agent_1 output]" in prompt else 4
  return [1, 0, state]


class CategoricalSampler:
  """Autoregressive categorical sampler supporting optional logprobs."""

  def __init__(self, policy: CategoricalPolicy):
    self.policy = policy

  async def sample(
      self, request: sampler_lib.SamplingRequest
  ) -> sampler_lib.SamplingResponse:
    params = request.sampling_params
    table = np.asarray(self.policy.logits[...], dtype=np.float64)
    scaled = table / max(float(params.temperature), 1e-4)
    normed = scaled - scaled.max(axis=-1, keepdims=True)
    log_probs = normed - np.log(np.exp(normed).sum(axis=-1, keepdims=True))
    rng = np.random.default_rng(params.seed)
    prev = int(request.prompt[-1])
    chosen_ids: list[int] = []
    chosen_lps: list[float] = []
    for _ in range(params.max_tokens):
      probs = np.exp(log_probs[prev])
      probs /= probs.sum()
      tok = int(rng.choice(len(VOCAB), p=probs))
      chosen_ids.append(tok)
      chosen_lps.append(float(log_probs[prev, tok]))
      prev = tok
      if tok == 0:
        break
    rendered = " ".join(VOCAB[t] for t in chosen_ids if t != 0)
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt, dtype=np.int32),
        token_ids=np.asarray(chosen_ids, dtype=np.int32),
        logprobs=(
            np.asarray(chosen_lps, dtype=np.float32)
            if params.return_logprobs
            else None
        ),
        text=rendered,
        finish_reason="stop" if prev == 0 else "length",
    )
