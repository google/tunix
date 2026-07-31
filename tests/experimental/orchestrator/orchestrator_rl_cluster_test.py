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

"""Routing/delegation tests for OrchestratorRLCluster (no real compute)."""

import types

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import orchestrator_rl_cluster


class _FakeBaseCluster:
  """Records the calls the orchestrator cluster would fall back to."""

  def __init__(self):
    self.global_steps = 0
    self.generate_calls = []
    self.update_actor_calls = []
    self.update_critic_calls = []
    self.sync_calls = 0
    self.ref_calls = []
    self.actor_calls = []
    self.buffered_metrics = []
    # Arbitrary non-primitive surface delegated via __getattr__.
    self.cluster_config = "CLUSTER_CONFIG"
    self.rollout = "ROLLOUT"

  def generate(self, *args):
    self.generate_calls.append(args)
    return "base_generate"

  def update_actor(self, *args):
    self.update_actor_calls.append(args)

  def update_critic(self, *args):
    self.update_critic_calls.append(args)

  def sync_weights(self):
    self.sync_calls += 1

  def get_ref_per_token_logps(self, **kwargs):
    self.ref_calls.append(kwargs)
    return "base_ref"

  def get_actor_per_token_logps(self, **kwargs):
    self.actor_calls.append(kwargs)
    return "base_actor"

  def buffer_metrics(self, metrics, **kwargs):
    self.buffered_metrics.append(metrics)

  def get_rollout_config(self, mode):
    del mode
    return types.SimpleNamespace(temperature=0.7)


class _FakeTrainerWorker:

  def __init__(self, with_logps=False):
    self.train_calls = []
    self.logps_calls = []
    if with_logps:
      self.per_token_logps = self._per_token_logps

  def train(self, chunks, eval_ds, skip_jit):
    self.train_calls.append((chunks, eval_ds, skip_jit))

  def _per_token_logps(self, **kwargs):
    self.logps_calls.append(kwargs)
    return "worker_actor"


class _FakeRolloutWorker:

  def __init__(self):
    self.calls = []

  def generate(self, **kwargs):
    self.calls.append(kwargs)
    return "worker_generate"


class _FakeInferenceWorker:

  def __init__(self, error: datatypes.ErrorInfo | None = None):
    self.calls = []
    self._error = error

  def compute_logps(
      self, request: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResponse:
    self.calls.append(request)
    return datatypes.LogprobsResponse(
        request_id=request.request_id,
        per_token_logps="worker_ref" if self._error is None else None,
        error=self._error,
    )


class _FakeWeightSync:

  def __init__(self):
    self.syncs = 0

  def sync(self):
    self.syncs += 1


class OrchestratorRlClusterTest(absltest.TestCase):

  def test_delegates_unrouted_surface_to_base(self):
    base = _FakeBaseCluster()
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    # Arbitrary attributes delegate through to the base cluster.
    self.assertEqual(cluster.cluster_config, "CLUSTER_CONFIG")
    self.assertEqual(cluster.rollout, "ROLLOUT")
    cluster.buffer_metrics({"loss": (1.0, None)})
    self.assertEqual(base.buffered_metrics, [{"loss": (1.0, None)}])

  def test_global_steps_reads_and_writes_the_base(self):
    base = _FakeBaseCluster()
    cluster = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    self.assertEqual(cluster.global_steps, 0)
    cluster.global_steps += 5
    self.assertEqual(base.global_steps, 5)
    self.assertEqual(cluster.global_steps, 5)

  def test_generate_routes_then_falls_back(self):
    base = _FakeBaseCluster()
    worker = _FakeRolloutWorker()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, rollout_worker=worker
    )
    self.assertEqual(routed.generate(["p"]), "worker_generate")
    self.assertLen(worker.calls, 1)
    self.assertEmpty(base.generate_calls)

    fallback = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    self.assertEqual(fallback.generate(["p"]), "base_generate")
    self.assertLen(base.generate_calls, 1)

  def test_update_actor_routes_then_falls_back(self):
    base = _FakeBaseCluster()
    worker = _FakeTrainerWorker()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=worker
    )
    routed.update_actor(["c"], None, False)
    self.assertEqual(worker.train_calls, [(["c"], None, False)])
    self.assertEmpty(base.update_actor_calls)

    fallback = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    fallback.update_actor(["c"], None, False)
    self.assertEqual(base.update_actor_calls, [(["c"], None, False)])

  def test_sync_weights_routes_then_falls_back(self):
    base = _FakeBaseCluster()
    weight_sync = _FakeWeightSync()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, weight_sync=weight_sync
    )
    routed.sync_weights()
    self.assertEqual(weight_sync.syncs, 1)
    self.assertEqual(base.sync_calls, 0)

    fallback = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    fallback.sync_weights()
    self.assertEqual(base.sync_calls, 1)

  def test_ref_logps_routes_to_inference_then_falls_back(self):
    base = _FakeBaseCluster()
    inference = _FakeInferenceWorker()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, inference_worker=inference
    )
    self.assertEqual(
        routed.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "worker_ref",
    )
    self.assertLen(inference.calls, 1)
    # The temperature the tokens were sampled at travels with the request.
    self.assertEqual(inference.calls[0].temperature, 0.7)
    self.assertEmpty(base.ref_calls)

    fallback = orchestrator_rl_cluster.OrchestratorRLCluster(base)
    self.assertEqual(
        fallback.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "base_ref",
    )
    self.assertLen(base.ref_calls, 1)

  def test_a_remote_trainers_metrics_reach_the_run_logger(self):
    """Otherwise everything the trainer measured is simply absent."""

    class _RemoteTrainer(_FakeTrainerWorker):

      def drain_metrics(self):
        return {"loss": 0.25, "grad_norm": 1.5}

    base = _FakeBaseCluster()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=_RemoteTrainer()
    )

    routed.update_actor(["c"], None, False)

    buffered = base.buffered_metrics[0]
    self.assertIn("trainer/loss", buffered)
    self.assertEqual(buffered["trainer/loss"][0], 0.25)
    self.assertIn("trainer/grad_norm", buffered)

  def test_a_handle_that_shares_the_logger_buffers_nothing_extra(self):
    base = _FakeBaseCluster()
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=_FakeTrainerWorker()
    )

    routed.update_actor(["c"], None, False)

    self.assertEmpty(base.buffered_metrics)

  def test_a_scoring_failure_is_raised_rather_than_returned(self):
    """There is no safe fallback value: a zero array would train on fiction."""
    inference = _FakeInferenceWorker(
        error=datatypes.ErrorInfo(error_type="RuntimeError", message="oom")
    )
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        _FakeBaseCluster(), inference_worker=inference
    )

    with self.assertRaisesRegex(RuntimeError, "oom"):
      routed.get_ref_per_token_logps("p", "c", pad_id=0, eos_id=2)

  def test_actor_logps_routes_when_worker_capable_else_base(self):
    base = _FakeBaseCluster()
    capable = _FakeTrainerWorker(with_logps=True)
    routed = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=capable
    )
    self.assertEqual(
        routed.get_actor_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "worker_actor",
    )
    self.assertLen(capable.logps_calls, 1)
    self.assertEmpty(base.actor_calls)

    # Trainer worker without per_token_logps -> in-process fallback.
    incapable = _FakeTrainerWorker(with_logps=False)
    fallback = orchestrator_rl_cluster.OrchestratorRLCluster(
        base, trainer_worker=incapable
    )
    self.assertEqual(
        fallback.get_actor_per_token_logps("p", "c", pad_id=0, eos_id=2),
        "base_actor",
    )
    self.assertLen(base.actor_calls, 1)


if __name__ == "__main__":
  absltest.main()
