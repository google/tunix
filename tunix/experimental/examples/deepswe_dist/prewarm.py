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

"""MLPerf DeepSWE `data_module`: the DeepSWE records plus a sandbox prewarm.

The experimental DeepSWE rollout environment only acquires sandboxes from warm
pools the orchestrator created. `mlperf_35b_128_v5p.yaml` names this module as
its `data_module`: the orchestrator loads the records with `create_dataset` and
enters `prompt_items_context` around the run, which warms the sandbox pools of
upcoming batches with `prewarm_sandboxes`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import contextlib
import math
import os
import typing

import grain
from tunix.dist_rl.examples.deepswe_dist import deepswe
from tunix.dist_rl.launch.common import resolve_config
from tunix.experimental.examples.deepswe_dist import sandbox_utils

# The records are the DeepSWE task records (`data_config`).
create_dataset = deepswe.create_dataset

# Sandbox creations the orchestrator's fleet may run at once.
_MAX_CONCURRENCY = 1024


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
) -> Iterator[Iterable[dict[str, object]]]:
  """Warms job-scoped `SandboxFleet` pools ahead of dispatch when enabled."""
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
  if not isinstance(scaffold, str) or not scaffold.strip():
    raise ValueError(
        f"env_kwargs['scaffold'] must be a non-empty str, got {scaffold!r}."
    )
  fleet = sandbox_utils.init_global_fleet(
      tasks=None,
      max_concurrency=max_concurrency,
      num_generations=num_generations,
      batch_size=batch_size,
      max_warmpool_replicas=max_warmpool_replicas,
      scaffold=scaffold,
      job_id=os.environ["ORCHESTRATOR_ID"],
      warm_on_init=False,
      teardown_on_exit=False,
  )
  try:
    # `PrewarmDatasetIterator` keeps `max_staleness + 1` previous batches warm.
    # With `trajectory_completion`, `StandardRLProgram` can read up to
    # `2 * max_staleness + 1` batches plus its 2-item read-ahead past a batch
    # that is still rolling out, so the iterator retains
    # `2 * max_staleness + ceil(2 / batch_size)`. Pools are released by
    # `close()` when the run ends, not at dataset EOF, since the final batches
    # may still be rolling out.
    with contextlib.closing(
        sandbox_utils.PrewarmDatasetIterator(
            prompt_items,
            fleet=fleet,
            num_generations=num_generations,
            batch_size=batch_size,
            max_warmpool_replicas=max_warmpool_replicas,
            max_staleness=2 * max_staleness + math.ceil(2 / batch_size),
            unwarm_on_exhaustion=False,
            scaffold=scaffold,
        )
    ) as prewarm:
      # `PrewarmDatasetIterator` yields `prompt_items`' records unchanged.
      yield typing.cast(Iterable[dict[str, object]], prewarm)
  finally:
    sandbox_utils.teardown_global_fleet()


def prompt_items_context(
    hp: resolve_config.DistHyperParameters,
    prompt_items: grain.MapDataset[dict[str, object]],
) -> contextlib.AbstractContextManager[Iterable[dict[str, object]]]:
  """`orchestrator.PromptItemsContext` that prewarms one pool per generation."""
  grpo = hp.config_mapping("agentic_grpo_config")
  num_generations = int(grpo["num_generations"])
  return prewarm_sandboxes(
      prompt_items,
      batch_size=int(hp.config["batch_size"]),
      num_generations=num_generations,
      max_staleness=int(grpo["off_policy_steps"]),
      env_kwargs=hp.config_mapping("env_kwargs"),
      max_concurrency=_MAX_CONCURRENCY,
      max_warmpool_replicas=num_generations,
  )
