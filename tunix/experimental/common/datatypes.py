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

"""Common data types and DTOs for the Tunix Orchestrator and Workers.

This module centralizes type aliases and dataclasses used for routing data and
commands between the orchestrator and distributed RL workers.

The dataclasses here intentionally start from a small field set. Additional
fields are added as concrete consumers require them, rather than up front.
"""

import dataclasses

from tunix.rl import common
from tunix.rl.agentic.agents import agent_types


@dataclasses.dataclass(kw_only=True)
class SamplingParams:
  """Engine-neutral sampling configuration for a generation request.

  Attributes:
    max_tokens: Maximum number of tokens to generate.
    temperature: Softmax temperature applied while sampling. Kept explicit so it
      can be carried through to any later log-probability scoring, ensuring the
      sampling distribution and the scoring distribution match.
  """

  max_tokens: int
  temperature: float = 1.0


@dataclasses.dataclass(kw_only=True)
class TrajectoryRequest:
  """Request to generate a trajectory from a given prompt.

  Attributes:
    request_id: Unique identifier for this request, echoed back on the
      corresponding result so callers can correlate and de-duplicate responses.
    prompt_id: Identifier of the source prompt (e.g. a dataset row). Provenance
      only; not unique across requests that reuse the same prompt.
    prompt_text: The prompt to generate from.
    sampling_params: Sampling configuration for the generation.
    max_turns: Maximum number of interaction turns for a multi-turn episode.
  """

  request_id: str
  prompt_id: str
  prompt_text: str
  sampling_params: SamplingParams
  max_turns: int = 10


# Worker-internal episode representation produced during rollout. This is the
# in-process type used inside a rollout worker; it is not a serialization format
# and is not meant to cross a process boundary. A dedicated serializable result
# type will be introduced separately when the wire path needs one.
Trajectory = agent_types.Trajectory


# On-device training example consumed by the trainer. This holds device-resident,
# potentially sharded arrays, so it cannot be serialized directly as a
# cross-process payload; a host/numpy-backed wire equivalent will be introduced
# separately when needed.
TrainExample = common.TrainExample
