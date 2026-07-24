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

"""In-process worker handles backed by an RLCluster.

These are the concrete handles the orchestrator drives when everything runs in a
single process: they satisfy the same handle contracts a remote (RPC) worker
would, but delegate straight to the reused in-process engine. They let the
orchestrator run end-to-end today and give the RPC handles a behavioral
reference to match.
"""

from typing import Any


class InProcessTrainerWorker:
  """Trainer-worker handle that delegates to an in-process ``RLCluster``.

  Runs the actor (and optional critic) trainer in the same process by calling the
  cluster's existing ``update_actor`` / ``update_critic``, exposing the handle
  contract the orchestrator drives:

      train(chunks, eval_ds, skip_jit) -> None
      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array

  The trainer writes its metrics straight to the cluster's shared logger, so this
  handle intentionally exposes no ``drain_metrics`` -- the orchestrator's metrics
  pull is a no-op for it (nothing to move across a process boundary).
  """

  def __init__(self, rl_cluster: Any):
    self._rl_cluster = rl_cluster

  def train(self, chunks: Any, eval_ds: Any, skip_jit: bool) -> None:
    """Runs one trainer pass over the chunked micro-batch (actor + optional critic)."""
    self._rl_cluster.update_actor(chunks, eval_ds, skip_jit)
    if hasattr(self._rl_cluster, "critic_trainer"):
      self._rl_cluster.update_critic(chunks, eval_ds, skip_jit)

  def per_token_logps(
      self, prompt_ids: Any, completion_ids: Any, pad_id: int, eos_id: int
  ) -> Any:
    """Actor-model per-token logprobs over a padded group."""
    return self._rl_cluster.get_actor_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=self._rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
    )
