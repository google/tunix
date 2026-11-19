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

"""Separate trainable Director, Proposer, and frozen Worker fixtures for CPU.

Prompt state tokens (1..5, 13..14) are disjoint from generated action/answer
tokens (6..12, 15), so every sampled action token immediately transitions to EOS
(token 0). Without skill cards (state 2), the Director mixes prematurely
finishing `agent_1` (wrong draft `0`) vs adding `agent_2` (Verifier `42`).
With skill cards (state 13), adding `agent_2` is up-weighted so visible/hidden
arms exhibit positive contrast while preserving mixed group rewards.
"""

from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib

PROPOSER_LABEL_IDS = {"A": 1, "B": 2, "C": 3, "D": 4}

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
    "<state_agent1_scaffolded>",
    "<pats_review_prompt>",
    '{"operations":[{"op":"ADD","evidence_ids":["E1","E2"],"card":{"name":"Peer audit","description":"Cross-check draft before submission.","trigger":"When a single solver draft may fail.","plan":"Add a layer-1 task_result Verifier fed by agent_1 before finish.","pitfall":"Submitting an unverified draft early.","constraint":"Ensure every retained node reaches the target.","kind":"verification"}}]}',
)


class CategoricalPolicy(nnx.Module):
  """Minimal bigram logit table compatible with `TrainerWorker`."""

  def __init__(self, logits: np.ndarray):
    self.logits = nnx.Param(jnp.asarray(logits, dtype=np.float32))

  def __call__(
      self, tokens, positions, cache=None, attention_mask=None, segment_ids=None
  ):
    del positions, attention_mask, segment_ids
    return self.logits[tokens], cache


def make_director_policy() -> CategoricalPolicy:
  """Trainable Director policy over JSON canvas actions (tokens 6..10)."""
  n = len(VOCAB)
  logits = np.full((n, n), -8.0, dtype=np.float32)
  logits[:, 0] = 6.0
  logits[1, 0] = -8.0
  logits[1, 6] = 5.0
  logits[2, 0] = -8.0
  logits[2, 7] = 1.2
  logits[2, 8] = 1.0
  logits[2, 10] = 0.0
  logits[13, 0] = -8.0
  logits[13, 7] = 2.4
  logits[13, 8] = 0.6
  logits[3, 0] = -8.0
  logits[3, 9] = 5.0
  return CategoricalPolicy(logits)


def make_frozen_worker_policy() -> CategoricalPolicy:
  """Frozen Worker policy serving both canvas execution and PATS distillation."""
  n = len(VOCAB)
  logits = np.full((n, n), -8.0, dtype=np.float32)
  logits[:, 0] = 6.0
  logits[4, 0] = -8.0
  logits[4, 11] = 6.0
  logits[5, 0] = -8.0
  logits[5, 12] = 6.0
  logits[14, 0] = -8.0
  logits[14, 15] = 8.0
  return CategoricalPolicy(logits)


def make_proposer_policy() -> CategoricalPolicy:
  """Trainable CPU Proposer policy over candidate labels `A..D` (tokens 1..4)."""
  size = 12
  logits = np.full((size, size), -6.0, dtype=np.float32)
  for state in range(5, size):
    logits[state, 1:5] = np.asarray([1.1, 1.0, 0.9, 0.8], dtype=np.float32)
  return CategoricalPolicy(logits)


def encode_director(prompt: str) -> list[int]:
  """Maps Canvas snapshot and scaffold presence to conditioning token IDs."""
  if '"id":"agent_2"' in prompt:
    state = 3
  elif '"id":"agent_1"' in prompt:
    state = 13 if "Optional Orchestration Knowledge:" in prompt else 2
  else:
    state = 1
  return [1, 0, state]


def encode_worker(prompt: str) -> list[int]:
  """Maps Worker prompt or PATS review payload to conditioning token IDs."""
  if "Review Payload:" in prompt:
    state = 14
  elif "[agent_1 output]" in prompt:
    state = 5
  else:
    state = 4
  return [1, 0, state]


def encode_proposer(prompt_text: str) -> list[int]:
  """Maps Proposer candidate-list JSON to a deterministic state token."""
  bucket = 6 + (sum(ord(ch) for ch in prompt_text) % 6)
  return [5, 0, bucket]


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
