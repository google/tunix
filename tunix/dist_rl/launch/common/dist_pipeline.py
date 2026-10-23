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

"""`DistPipeline`: the distributed RL config with the `tunix.cli` builders.

Distributed RL node and orchestrator mains build their components directly from
this object (`create_rl_training_config`, `create_optimizer`,
`create_rollout_config`, `_get_tokenizer`, `_create_chat_parser`, ...), so there
is a single conversion from YAML to runtime objects. It loads and finalizes the
config exactly like `resolve_config.DistHyperParameters` (`DIST_INIT_KWARGS`,
then `finalize_dist_config`), which stays separate so the bash launchers can
render exports without importing the training stack.
"""

from __future__ import annotations

from collections.abc import Sequence

from tunix.cli import base_rl_pipeline
from tunix.dist_rl.launch.common import resolve_config


class DistPipeline(base_rl_pipeline.BasePipeline):
  """Distributed RL configuration with the shared `BasePipeline` builders."""

  def __init__(self, argv: Sequence[str]) -> None:
    super().__init__(list(argv), **resolve_config.DIST_INIT_KWARGS)
    resolve_config.finalize_dist_config(self)

  @property
  def _default_training_mode(self) -> str:
    return "agentic_grpo"

  def _run(self, mode: str) -> None:
    raise NotImplementedError(
        "Distributed RL runs as separate trainer/rollout/orchestrator"
        f" processes; there is no single-process `_run` (mode={mode!r})."
    )


def initialize(argv: Sequence[str]) -> DistPipeline:
  """Initializes `DistPipeline` from `argv` (same contract as `resolve_config`)."""
  return DistPipeline(argv)
