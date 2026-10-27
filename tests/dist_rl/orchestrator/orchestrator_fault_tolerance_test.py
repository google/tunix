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

"""End-to-end distributed fault tolerance test suite for Tunix Orchestrator.

Validates the 3 major distributed fault domains on a single-machine real-gRPC
cluster harness without GPU/TPU hardware:
  1. Discovery, Lifecycle & Worker Restarts.
  2. Rollout Data Plane, Long-Poll Retries & Group Conservation.
  3. WeightSync Cross-Layer Wire Coordination & Poisoning.
Also pins 3 known production edge-case defects via @unittest.expectedFailure.
"""

import asyncio
from collections.abc import Sequence
from typing import Any
import unittest

from absl.testing import absltest
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.testing import fake_cluster
from tunix.dist_rl.testing import fault_injection
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import remote_execution
from tunix.dist_rl.worker import task_session


def _first_rollout_request(
    req: remote_execution.ExecutionRequest,
) -> datatypes.RolloutRequest:
  return req.kwargs["requests"][0]


def _extract_prompt_id(req: remote_execution.ExecutionRequest) -> str:
  return str(_first_rollout_request(req).prompt_id)


def _extract_group_index(req: remote_execution.ExecutionRequest) -> int:
  return int(_first_rollout_request(req).group_index)


class OrchestratorFaultToleranceTest(absltest.TestCase):

  # =========================================================================
  # Domain 1: Discovery, Lifecycle & Restart
  # =========================================================================

  def test_worker_restart_with_override_replaces_grpc_handle(self) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      new_port = harness.restart_worker(
          "rollout_1", same_port=False, override=True, re_register=True
      )
      self.assertEqual(
          harness.orchestrator.registry.handle("rollout_1").target_address,
          f"grpc://localhost:{new_port}",
      )
      harness.orchestrator.bring_up_workers()
      for worker in harness.rollout_workers.values():
        self.assertEqual(worker.state, datatypes.WorkerState.READY)

      program = harness.build_program(
          num_prompts=2,
          max_steps=1,
          num_generations=2,
          mini_batch_size=2,
          sync_weights=True,
      )
      future = harness.start_program(program, bring_up=False)
      future.result(timeout=60.0)

      self.assertEqual(harness.fake_trainer.step_count, 1)
      for sampler in harness.samplers.values():
        self.assertEqual(sampler.policy_version, 1)

  def test_duplicate_worker_registration_without_override_rejected_fail_fast(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      with self.assertRaisesRegex(ValueError, "duplicate worker_id"):
        harness.register_worker_from_discovery(
            "rollout_0",
            "rollout",
            port=harness.servers["rollout_0"].bound_port,
            override=False,
        )
      hb = harness.orchestrator.registry.handle("rollout_0").submit("heartbeat")
      self.assertEqual(hb.state, datatypes.WorkerState.PENDING)

  def test_bring_up_single_worker_compile_failure_aborts_startup(self) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2, compile_error_workers={"trainer_0"}
    ) as harness:
      with self.assertRaises(RuntimeError):
        harness.orchestrator.bring_up_workers()
      self.assertEqual(
          harness.trainer_worker.state, datatypes.WorkerState.ERROR
      )
      self.assertIsNone(harness.orchestrator.engine)

  def test_health_monitor_detects_error_and_unreachable_worker(self) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      harness.orchestrator.bring_up_workers()
      reports = harness.orchestrator.health_monitor.poll()
      self.assertEqual(
          {wid: r.state for wid, r in reports.items()},
          {
              "trainer_0": datatypes.WorkerState.READY,
              "rollout_0": datatypes.WorkerState.READY,
              "rollout_1": datatypes.WorkerState.READY,
          },
      )

      harness.injector.on(
          "rollout_0",
          "heartbeat",
          fault_injection.Return(
              datatypes.HealthReport(
                  state=datatypes.WorkerState.ERROR,
                  last_error="simulated OOM",
              )
          ),
          times=1,
      )
      reports = harness.orchestrator.health_monitor.poll()
      self.assertEqual(reports["rollout_0"].state, datatypes.WorkerState.ERROR)
      self.assertEqual(reports["rollout_0"].last_error, "simulated OOM")

      harness.stop_worker_server("rollout_1", grace=0.0)
      with self.assertRaises(Exception):
        harness.orchestrator.health_monitor.poll()

  # =========================================================================
  # Domain 2: Rollout Data Plane, Long-Poll Retries & Conservation
  # =========================================================================

  def test_rollout_trajectory_error_envelope_preserves_conservation(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      for sampler in harness.samplers.values():
        sampler.fail_request_ids.add("traj_prompt_0_g0")

      program = harness.build_program(
          num_prompts=2,
          max_steps=1,
          num_generations=4,
          mini_batch_size=2,
          batch_size=2,
          sync_weights=True,
      )
      future = harness.start_program(program)
      future.result(timeout=60.0)

      self.assertIsNotNone(program.last_step_result)
      self.assertEqual(program.last_step_result.num_rollouts, 8)
      self.assertEqual(program._in_flight_rollouts, 0)
      frac = program.metrics_logger.get_metric(
          "rollout", "invalid_trajectory_frac", "train"
      )
      self.assertAlmostEqual(frac, 1.0 / 8.0)

  def test_rollout_worker_disconnect_exhausts_retry_budget_and_aborts_program(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        poll_retry_budget_s=0.25,
    ) as harness:
      gate = fault_injection.FaultGate()
      harness.injector.on(
          "rollout_1",
          "generate",
          fault_injection.Block(gate),
          times=1,
      )
      program = harness.build_program(
          num_prompts=4,
          max_steps=1,
          num_generations=2,
          mini_batch_size=2,
          sync_weights=False,
      )
      future = harness.start_program(program)
      self.assertTrue(gate.entered.wait(timeout=30.0))
      harness.stop_worker_server("rollout_1", grace=0.0)
      gate.release()

      with self.assertRaises(
          (task_session.WorkerUnreachableError, RuntimeError)
      ):
        future.result(timeout=30.0)
      self.assertTrue(future.done())

  def test_rollout_worker_disconnect_with_ft_enabled_evicts_and_retries_on_survivor(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        fault_tolerance_config=datatypes.FaultToleranceConfig(
            enabled=True,
            max_task_retries=1,
            poll_retry_budget_s=0.25,
        ),
    ) as harness:
      gate = fault_injection.FaultGate()
      harness.injector.on(
          "rollout_1",
          "generate",
          fault_injection.Block(gate),
          times=1,
      )
      program = harness.build_program(
          num_prompts=4,
          max_steps=1,
          num_generations=2,
          mini_batch_size=2,
          sync_weights=False,
      )
      future = harness.start_program(program)
      self.assertTrue(gate.entered.wait(timeout=30.0))
      harness.stop_worker_server("rollout_1", grace=0.0)
      gate.release()

      future.result(timeout=30.0)
      self.assertEqual(harness.fake_trainer.step_count, 1)
      self.assertEqual(
          [
              info.worker_id
              for info in harness.orchestrator.registry.infos(
                  datatypes.Role.ROLLOUT
              )
          ],
          ["rollout_0"],
      )

  def test_in_flight_straggler_across_weight_sync_filtered_by_staleness(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2, partial_rollout=True
    ) as harness:
      straggler_gate = fault_injection.FaultGate()
      harness.injector.on(
          fault_injection.ANY,
          "generate",
          fault_injection.Block(straggler_gate),
          where=lambda req: (
              _extract_prompt_id(req) == "prompt_0"
              and _extract_group_index(req) == 0
          ),
          times=1,
      )

      def _on_step_end(step: int, unused_res: Any) -> None:
        if step == 1:
          straggler_gate.release()

      program = harness.build_program(
          num_prompts=5,
          max_steps=3,
          num_generations=2,
          mini_batch_size=1,
          batch_size=1,
          max_staleness=1,
          sync_weights=True,
          group_order=rl_program.GroupOrder.TRAJECTORY_COMPLETION,
          on_step_end=_on_step_end,
      )
      future = harness.start_program(program)
      self.assertTrue(straggler_gate.entered.wait(timeout=30.0))
      future.result(timeout=60.0)

      self.assertEqual(harness.fake_trainer.step_count, 3)
      self.assertIn("prompt_0", program.dispatch_window.skipped_prompt_ids)
      self.assertNotIn("prompt_0", program.dispatch_window.committed_prompt_ids)

  def test_bulk_transport_large_payload_rollout_roundtrip(self) -> None:
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=1, enable_bulk_transport=True
    ) as harness:
      prompts = harness.build_prompts(num_prompts=1)
      prompts[0]["metadata"] = {"large_blob": b"x" * (600 * 1024)}
      program = harness.build_program(
          prompts=prompts,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          sync_weights=False,
      )
      future = harness.start_program(program)
      future.result(timeout=60.0)
      self.assertEqual(harness.fake_trainer.step_count, 1)

  # =========================================================================
  # Domain 3: WeightSync Cross-Layer & Wire
  # =========================================================================

  def test_pre_weight_sync_failure_aborts_all_destinations_and_stops_program(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      harness.injector.on(
          "rollout_1",
          "pre_weight_sync",
          fault_injection.Raise(RuntimeError("pre_weight_sync boom")),
          times=1,
      )
      program = harness.build_program(
          num_prompts=2, max_steps=1, sync_weights=True
      )
      future = harness.start_program(program)
      with self.assertRaises(weight_sync_coordinator.WeightSyncError) as ctx:
        future.result(timeout=30.0)
      self.assertEqual(
          ctx.exception.result.state, weight_sync_coordinator.RoundState.ABORTED
      )

      for worker in harness.rollout_workers.values():
        self.assertEqual(worker.state, datatypes.WorkerState.READY)
        self.assertEqual(
            asyncio.run(worker.get_weight_sync_status())["phase"], "aborted"
        )
      for sampler in harness.samplers.values():
        self.assertEqual(sampler.policy_version, 0)
      self.assertEqual(
          harness.fake_trainer.prepare_calls, harness.fake_trainer.release_calls
      )
      self.assertTrue(future.done())

  def test_post_weight_sync_unreachable_poisons_coordinator_as_needs_restart(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      harness.injector.on(
          "rollout_1",
          "post_weight_sync",
          fault_injection.Raise(RuntimeError("post_sync crash")),
          times=1,
      )
      harness.injector.on(
          "rollout_1",
          "get_weight_sync_status",
          fault_injection.Raise(RuntimeError("status unreachable")),
          times=2,
      )
      program = harness.build_program(
          num_prompts=2, max_steps=1, sync_weights=True
      )
      future = harness.start_program(program)
      with self.assertRaises(weight_sync_coordinator.WeightSyncError) as ctx:
        future.result(timeout=30.0)
      self.assertEqual(
          ctx.exception.result.state,
          weight_sync_coordinator.RoundState.FAILED_NEEDS_RESTART,
      )

      with self.assertRaisesRegex(RuntimeError, "poisoned"):
        asyncio.run(program.engine.sync_weights())

  def test_post_weight_sync_partial_commit_poisons_coordinator(self) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      harness.injector.on(
          "rollout_1",
          "post_weight_sync",
          fault_injection.Raise(
              RuntimeError("post_sync rejected before commit")
          ),
          times=2,
      )
      program = harness.build_program(
          num_prompts=2, max_steps=1, sync_weights=True
      )
      future = harness.start_program(program)
      with self.assertRaises(weight_sync_coordinator.WeightSyncError) as ctx:
        future.result(timeout=30.0)
      self.assertEqual(
          ctx.exception.result.state,
          weight_sync_coordinator.RoundState.PARTIALLY_COMMITTED,
      )

      with self.assertRaisesRegex(RuntimeError, "poisoned"):
        asyncio.run(program.engine.sync_weights())

  def test_post_weight_sync_lost_reply_reconciles_via_status_over_grpc(
      self,
  ) -> None:
    with fake_cluster.FakeClusterHarness(num_rollout_workers=2) as harness:
      harness.injector.on(
          "rollout_1",
          "post_weight_sync",
          fault_injection.LoseReply(
              RuntimeError("simulated gRPC stream reset after commit")
          ),
          times=1,
      )
      program = harness.build_program(
          num_prompts=2,
          max_steps=1,
          sync_weights=True,
          precompile_trainer=False,
      )
      future = harness.start_program(program)
      future.result(timeout=60.0)

      self.assertEqual(harness.fake_trainer.step_count, 1)
      for sampler in harness.samplers.values():
        self.assertEqual(sampler.policy_version, 1)
      for worker in harness.rollout_workers.values():
        self.assertEqual(worker.state, datatypes.WorkerState.READY)

  # =========================================================================
  # Bug-Pinning Tests (@unittest.expectedFailure)
  # =========================================================================

  @unittest.expectedFailure
  def test_bug_pin_missing_prompt_id_in_error_response_starves_group(
      self,
  ) -> None:
    """Pins distributed_rl_engine.py:56: missing prompt_id in error metadata falls back to ''."""
    with fake_cluster.FakeClusterHarness(num_rollout_workers=1) as harness:
      harness.injector.on(
          fault_injection.ANY,
          "generate",
          fault_injection.Return(
              datatypes.RolloutResponse(
                  request_id="req_prompt_0_g0_v0",
                  status="ERROR",
                  payload=None,
                  error=datatypes.ErrorInfo(
                      error_type="RuntimeError",
                      message="missing metadata",
                      retryable=False,
                  ),
                  metadata={},
              )
          ),
          where=lambda req: (
              _extract_prompt_id(req) == "prompt_0"
              and _extract_group_index(req) == 0
          ),
          times=1,
      )
      program = harness.build_program(
          num_prompts=1,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          sync_weights=False,
      )
      future = harness.start_program(program)
      future.result(timeout=30.0)
      self.assertEqual(harness.fake_trainer.step_count, 1)

  @unittest.expectedFailure
  def test_bug_pin_critique_stage_exception_swallowed_fails_to_propagate(
      self,
  ) -> None:
    """Pins rl_program.py:672: critique_stage swallows exceptions and breaks."""
    with fake_cluster.FakeClusterHarness(num_rollout_workers=1) as harness:
      gate = fault_injection.FaultGate()
      harness.injector.on(
          "rollout_0",
          "generate",
          fault_injection.Block(gate),
          times=None,
      )
      program = harness.build_program(
          num_prompts=1,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          sync_weights=False,
      )
      asyncio.run(
          program.raw_q.put_exception(RuntimeError("raw_q get_group error"))
      )
      future = harness.start_program(program)
      with self.assertRaisesRegex(RuntimeError, "raw_q get_group error"):
        future.result(timeout=1.5)

  # TODO: b/577214994 - Detect in-place worker restarts and lift expectedFailure.
  @unittest.expectedFailure
  def test_bug_pin_same_port_worker_restart_loses_in_flight_task(self) -> None:
    """Pins task_session.py:486: in-place same-port restart loses in-flight task id."""
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=1,
        poll_retry_budget_s=2.0,
    ) as harness:
      gate = fault_injection.FaultGate()
      harness.injector.on(
          "rollout_0",
          "generate",
          fault_injection.Block(gate),
          times=1,
      )
      program = harness.build_program(
          num_prompts=1,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          sync_weights=False,
      )
      future = harness.start_program(program)
      self.assertTrue(gate.entered.wait(timeout=30.0))
      harness.restart_worker("rollout_0", same_port=True, re_register=False)
      gate.release()
      with self.assertRaises(task_session.WorkerUnreachableError):
        future.result(timeout=2.0)


if __name__ == "__main__":
  absltest.main()
