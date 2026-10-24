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

It loads and finalizes the config exactly like
`resolve_config.DistHyperParameters` (`DIST_INIT_KWARGS`, then
`finalize_dist_config`) and adds the `tunix.cli.base_rl_pipeline.BasePipeline`
helpers. Most `BasePipeline` builders don't fit distributed RL (e.g.
`create_rl_training_config` assumes prompt-level micro-batches, while the
distributed trainer uses trajectory-level ones); in practice only
`_get_tokenizer` is used. `DistHyperParameters` stays separate so the bash
launchers can render exports without importing the training stack.

TODO(noghabi): Once the launch stack lands, load the tokenizer with
`node_utils.load_hf_tokenizer(hp)` in the orchestrator, use
`DistHyperParameters` everywhere, and delete this module and its test.
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
