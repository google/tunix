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

"""Categorical repair actions with exact policy log probabilities for CPU tests.

Action IDs are rendered using the public task context. This finite-action fixture
is not a tokenizer or pretrained language model and is never used by Qwen runs.
"""

import asyncio
import json
import re
import numpy as np
from examples.swarm_math.cpu_fixture import TinyPolicy
from tunix.dist_rl.rollout import sampler as sampler_lib

SIZE = 20


def make_policy():
  logits = np.full((SIZE, SIZE), -30.0, np.float32)
  logits[:, 0] = 0.0
  for state, choices in (
      (1, (10, 11)),
      (2, (11,)),
      (3, (12, 13)),
      (4, (14,)),
      (5, (15,)),
      (6, (16, 17)),
      (7, (18,)),
  ):
    logits[state] = -30.0
    logits[state, list(choices)] = 0.0
  return TinyPolicy(logits)


def observation(prompt):
  return json.loads(prompt.split("\nObservation:\n", 1)[1])


def state(obs):
  assigned = (
      "module_b.json"
      if obs["role"] == "repair:module_b.json"
      else "module_a.json"
  )
  other = "module_a.json" if assigned == "module_b.json" else "module_b.json"
  reads, previous, mode = obs["reads"], obs["last_action"], obs["mode"]
  if obs["turn"] == 0 and obs["role"] == "lead" and mode == "swarm":
    return 1
  if assigned not in reads:
    return 2
  if previous in ("read", "spawn") and other not in reads:
    return 3
  communicating = mode == "fixed" or (
      mode == "swarm" and obs["agents_spawned"] > 0
  )
  if previous == "write" and communicating:
    return 4
  if communicating:
    return 7
  if other not in reads:
    return 5
  if previous == "read":
    return 6
  return 7


def encode(prompt):
  return [0, state(observation(prompt))]


def module_from_public_requirement(text):
  """Fixture-only interpretation of the public specification, never the oracle."""
  if match := re.fullmatch(r"Multiply the input by (-?\d+)\.", text):
    return {"op": "scale", "value": int(match[1])}
  if match := re.fullmatch(r"Add (-?\d+) to the input\.", text):
    return {"op": "shift", "value": int(match[1])}
  operators = {
      "Return the square of the input.": "square",
      "Return the nonnegative magnitude of the input.": "absolute",
      "Return the input unchanged.": "identity",
  }
  return {"op": operators[text], "value": 0}


class TinySampler:

  def __init__(self, policy, delay_seconds=0.001):
    self.policy, self.delay_seconds = policy, delay_seconds
    self.prompts = {}

  def encode(self, prompt):
    ids = encode(prompt)
    # The runtime invokes encode immediately before sample in the same task.
    self.prompts[asyncio.current_task()] = observation(prompt)
    return ids

  async def sample(self, request):
    obs = self.prompts.pop(asyncio.current_task())
    # Yield during a real sampled request to exercise within-team concurrency.
    await asyncio.sleep(self.delay_seconds)
    logits = np.asarray(self.policy.logits[...], np.float64)
    logits -= logits.max(axis=-1, keepdims=True)
    logps = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))
    rng = np.random.default_rng(request.sampling_params.seed)
    token = int(rng.choice(SIZE, p=np.exp(logps[int(request.prompt[-1])])))
    assigned = (
        "module_b.json"
        if obs["role"] == "repair:module_b.json"
        else "module_a.json"
    )
    other = "module_a.json" if assigned == "module_b.json" else "module_b.json"
    if token == 10:
      action = {"action": "spawn", "role": "repair:module_b.json"}
    elif token in (11, 15):
      action = {"action": "read", "path": assigned if token == 11 else other}
    elif token in (12, 13, 16, 17):
      path = assigned if token in (12, 13) else other
      module = (
          module_from_public_requirement(obs["task"]["requirements"][path])
          if token in (12, 16)
          else {"op": "identity", "value": 0}
      )
      action = {"action": "write", "path": path, "module": module}
    elif token == 14:
      action = {
          "action": "send",
          "recipient": "agent-0",
          "message": f"Finished {assigned}.",
      }
    elif token == 18:
      action = {"action": "submit", "continue": False}
    else:
      action = {"action": "invalid"}
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt),
        token_ids=np.array([token, 0], np.int32),
        logprobs=np.array(
            [logps[int(request.prompt[-1]), token], logps[token, 0]], np.float32
        ),
        text=json.dumps(action),
        finish_reason="stop",
    )
