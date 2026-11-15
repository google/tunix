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

"""Tiny categorical graph editor for real optimizer tests; not a language model."""

import json
import numpy as np
from examples.swarm_math.cpu_fixture import TinyPolicy
from tunix.dist_rl.rollout import sampler as sampler_lib

GOOD = {
    "action": "set_prompt",
    "target": "agent_0",
    "role": "solver",
    "objective": "Solve arithmetic accurately",
    "scope": "whole task",
    "expected_output": "boxed answer",
}
BAD = {**GOOD, "objective": "Guess without calculating"}
VOCAB = (
    "",
    "add_state",
    "prompt_state",
    "output_state",
    "finish_state",
    "worker_good",
    "worker_bad",
    json.dumps({"action": "add_agent"}),
    json.dumps(GOOD),
    json.dumps(BAD),
    json.dumps({"action": "set_output", "target": "agent_0"}),
    json.dumps({"action": "finish"}),
    r"\boxed{2}",
    r"\boxed{3}",
)


def make_policy():
  logits = np.full((len(VOCAB), len(VOCAB)), -20.0, dtype=np.float32)
  logits[:, 0] = 0.0
  for state, targets in (
      (1, [7]),
      (2, [8, 9]),
      (3, [10]),
      (4, [11]),
      (5, [12]),
      (6, [13]),
  ):
    logits[state, :] = -20.0
    logits[state, targets] = 0.0
  return TinyPolicy(logits)


def encode(prompt):
  if "\nState:\n" in prompt:
    state = json.loads(
        prompt.split("\nState:\n", 1)[1].split("\nHistory:\n", 1)[0]
    )
    token = (
        1
        if not state["nodes"]
        else 2
        if state["nodes"]["agent_0"]["prompt"] is None
        else 3
        if state["output"] is None
        else 4
    )
  else:
    token = 6 if "Guess without calculating" in prompt else 5
  return [0, token]


class TinySampler:

  def __init__(self, policy):
    self.policy = policy

  async def sample(self, request):
    logits = (
        np.asarray(self.policy.logits[...], np.float64)
        / request.sampling_params.temperature
    )
    logits -= logits.max(axis=-1, keepdims=True)
    logps = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    rng = np.random.default_rng(request.sampling_params.seed)
    current = int(request.prompt[-1])
    tokens, scores = [], []
    for _ in range(request.sampling_params.max_tokens):
      token = int(rng.choice(len(VOCAB), p=np.exp(logps[current])))
      tokens.append(token)
      scores.append(logps[current, token])
      current = token
      if token == 0:
        break
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt, np.int32),
        token_ids=np.asarray(tokens, np.int32),
        logprobs=np.asarray(scores, np.float32),
        text=" ".join(VOCAB[t] for t in tokens if t),
        finish_reason="stop" if current == 0 else "length",
    )
