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

"""Agent sandbox prewarm for the shared distributed RL orchestrator.

`prewarm_sandboxes` is a `prompt_items_wrapper` for
`tunix.dist_rl.launch.nodes.orchestrator`. When `env_kwargs.use_agent_sandbox`
is enabled, it starts the run's job-scoped `SandboxFleet`, warms the sandbox
images of the next prompt batches (`sandbox_utils.PrewarmDatasetIterator`), and
tears the fleet down when the run ends.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import logging
import math
import os

import grain
from tunix.experimental.examples.deepswe_dist import sandbox_utils


@contextlib.contextmanager
def prewarm_sandboxes(
    prompt_items: grain.MapDataset[dict[str, object]],
    *,
    batch_size: int,
    num_generations: int,
    max_staleness: int,
    env_kwargs: Mapping[str, object],
    max_concurrency: int,
    max_warmpool_replicas: int,
) -> Iterator[
    grain.MapDataset[dict[str, object]] | sandbox_utils.PrewarmDatasetIterator
]:
  """Warms the agent sandboxes of the prompts ahead of their dispatch.

  The rollout workers attach to the same fleet through the same
  `ORCHESTRATOR_ID` (`deepswe.DeepSWEEnv`), so their sandbox claims hit the
  warm pools. When `env_kwargs["use_agent_sandbox"]` is false, yields
  `prompt_items` unchanged without initializing a fleet.

  Args:
    prompt_items: Prompt items; `metadata.env_config.entry` names each task's
      image.
    batch_size: Prompt groups per training step.
    num_generations: Rollouts per prompt group.
    max_staleness: Off-policy steps (`agentic_grpo_config.off_policy_steps`).
    env_kwargs: Run-wide environment settings (`use_agent_sandbox`, `scaffold`).
    max_concurrency: Cap on the fleet's concurrent sandboxes.
    max_warmpool_replicas: Cap on the warm sandboxes of each image.

  Yields:
    An iterator over `prompt_items` that warms each batch's images ahead (or
    `prompt_items` directly when `use_agent_sandbox` is false).

  Raises:
    KeyError: If required `env_kwargs` keys or `ORCHESTRATOR_ID` (when
      `use_agent_sandbox` is true) are missing.
    TypeError: If `env_kwargs["use_agent_sandbox"]` is not a `bool` or
      `env_kwargs["scaffold"]` is not a `str`.
    ValueError: If `batch_size` is not positive or `env_kwargs["scaffold"]` is
      empty.
  """
  if isinstance(batch_size, bool) or batch_size <= 0:
    raise ValueError(f"batch_size must be positive, got {batch_size!r}.")
  use_agent_sandbox = env_kwargs["use_agent_sandbox"]
  if not isinstance(use_agent_sandbox, bool):
    raise TypeError(
        "env_kwargs['use_agent_sandbox'] must be a bool, got"
        f" {type(use_agent_sandbox).__name__}."
    )
  if not use_agent_sandbox:
    yield prompt_items
    return

  scaffold = env_kwargs["scaffold"]
  if not isinstance(scaffold, str):
    raise TypeError(
        f"env_kwargs['scaffold'] must be a str, got {type(scaffold).__name__}."
    )
  if not scaffold.strip():
    raise ValueError("env_kwargs['scaffold'] must be non-empty.")
  job_id = os.environ["ORCHESTRATOR_ID"]
  logging.info(
      "Initializing orchestrator SandboxFleet + PrewarmDatasetIterator"
      " (job_id=%s, batch_size=%d, num_generations=%d, max_concurrency=%d,"
      " max_warmpool_replicas=%d, max_staleness=%d, scaffold=%s).",
      job_id,
      batch_size,
      num_generations,
      max_concurrency,
      max_warmpool_replicas,
      max_staleness,
      scaffold,
  )
  # `prewarm_sandboxes`'s `finally` block tears the fleet down;
  # `teardown_on_exit` would overwrite `managed_cluster_teardown`'s SIGTERM
  # handler.
  fleet = sandbox_utils.init_global_fleet(
      tasks=None,
      max_concurrency=max_concurrency,
      num_generations=num_generations,
      batch_size=batch_size,
      max_warmpool_replicas=max_warmpool_replicas,
      scaffold=scaffold,
      job_id=job_id,
      warm_on_init=False,
      teardown_on_exit=False,
  )
  try:
    prefetch_batches = math.ceil(2 / batch_size)
    prewarm = sandbox_utils.PrewarmDatasetIterator(
        prompt_items,
        fleet=fleet,
        num_generations=num_generations,
        batch_size=batch_size,
        max_warmpool_replicas=max_warmpool_replicas,
        max_staleness=max_staleness + prefetch_batches,
        unwarm_on_exhaustion=True,
        scaffold=scaffold,
    )
    try:
      yield prewarm
    finally:
      prewarm.close()
  finally:
    sandbox_utils.teardown_global_fleet()
