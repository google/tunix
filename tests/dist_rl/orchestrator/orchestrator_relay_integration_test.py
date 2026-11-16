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

"""End-to-end real-gRPC + Bulk-TCP integration tests for Orchestrator Relay."""

import asyncio
from collections.abc import Sequence
import threading
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.testing import fake_cluster


def _assert_zero_rollout_staging(
    test_case: absltest.TestCase,
    harness: fake_cluster.FakeClusterHarness,
):
  """Asserts every active RolloutWorker has zero used and reserved bytes."""
  engine = harness.orchestrator.engine
  test_case.assertIsNotNone(engine)
  assert engine is not None
  metrics = asyncio.run(engine.get_metrics(role=datatypes.Role.ROLLOUT))
  test_case.assertIsInstance(metrics, Sequence)
  test_case.assertLen(metrics, harness.num_rollout_workers)
  for idx, worker_m in enumerate(metrics):
    test_case.assertEqual(
        worker_m["staging/used_bytes"],
        0,
        f"Worker {idx} leaked used_bytes: {worker_m}",
    )
    test_case.assertEqual(
        worker_m["staging/reserved_bytes"],
        0,
        f"Worker {idx} leaked reserved_bytes: {worker_m}",
    )


class OrchestratorRelayIntegrationTest(parameterized.TestCase):
  """Real-gRPC Bulk-TCP E2E tests for Orchestrator Relay MoE router replay."""

  @parameterized.parameters("packed", "padded")
  def test_relay_e2e_bulk_transport_zero_inline_and_zero_leak_all_exits(
      self, assembler_type: str
  ):
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        enable_bulk_transport=True,
        routed_experts_shape_spec=(4, 8),
        routed_experts_encode_mode="uint8",
        routed_experts_num_experts=64,
        rollout_max_prompt_length=192,
        response_tokens_per_turn=8,
    ) as harness:
      harness.orchestrator.bring_up_workers()
      engine = harness.orchestrator.engine
      assert engine is not None

      # 1. Public blocking `engine.generate()` must keep inline routed_experts
      # without touching staging buffers.
      inline_items = asyncio.run(
          engine.generate(
              harness.build_prompts(num_prompts=2),
              generation_args=datatypes.GenerationArgs(
                  temperature=1.0, return_routed_experts=True
              ),
          )
      )
      self.assertLen(inline_items, 2)
      for item in inline_items:
        self.assertIsNone(item.routed_experts_ref)
        inline_routed = getattr(item, "routed_experts", None)
        self.assertIsNotNone(inline_routed)
        assert inline_routed is not None
        self.assertEqual(inline_routed.shape[1:], (4, 8))
      _assert_zero_rollout_staging(self, harness)

      # 2. Normal `max_steps` exit with extra unconsumed prompts in flight:
      # verifies zero inline arrays in `raw_q` / `scored_q` / `_buffer`, dense
      # `[B, T, 4, 8]` on FakeTrainer, zero staging leaks, and clean cross-run
      # `TaskSession` reuse (`abandon_in_flight()`).
      prog_max_steps = harness.build_program(
          num_prompts=4,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          max_staleness=2,
          assembler_type=assembler_type,
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      orig_raw_put = prog_max_steps.raw_q.put
      orig_scored_put = prog_max_steps.scored_q.put
      seen_refs: list[datatypes.RoutedExpertsRef] = []

      async def _checked_raw_put(item: datatypes.TrajectoryItem):
        self.assertIsNone(getattr(item, "routed_experts", None))
        if isinstance(item.traj, dict):
          self.assertNotIn("routed_experts", item.traj)
        self.assertNotIn("routed_experts", item.metadata)
        self.assertIsNotNone(item.routed_experts_ref)
        assert item.routed_experts_ref is not None
        seen_refs.append(item.routed_experts_ref)
        await orig_raw_put(item)

      async def _checked_scored_put(scored_group: Any):
        for p in scored_group.payloads:
          self.assertIsNone(p.routed_experts)
          self.assertIsNotNone(p.routed_experts_ref)
        await orig_scored_put(scored_group)

      prog_max_steps.raw_q.put = _checked_raw_put  # type: ignore[method-assign]
      prog_max_steps.scored_q.put = (  # type: ignore[method-assign]
          _checked_scored_put
      )
      try:
        harness.orchestrator.run(prog_max_steps, bring_up=False)
        self.assertEqual(prog_max_steps.step, 1)
        self.assertGreaterEqual(len(seen_refs), 2)
        bytes_after_run1 = harness.fake_trainer.total_routed_experts_bytes
        self.assertGreater(bytes_after_run1, 0)
        self.assertEqual(
            harness.fake_trainer.last_routed_experts_shape, (1, 224, 4, 8)
        )
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_max_steps.close()

      # 3. Dataset EOF before `max_steps` on the SAME orchestrator/engine
      # verifies zero Run-A bleedover and zero staging leak on EOF.
      prog_eof = harness.build_program(
          num_prompts=1,
          max_steps=4,
          num_generations=2,
          mini_batch_size=1,
          assembler_type=assembler_type,
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      prog_eof.policy_version = engine._policy_version
      try:
        harness.orchestrator.run(prog_eof, bring_up=False)
        self.assertEqual(prog_eof.step, 1)
        self.assertGreater(
            harness.fake_trainer.total_routed_experts_bytes, bytes_after_run1
        )
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_eof.close()

      # 4. Degenerate `< 2` valid group (`NaN` reward on `g=1` -> zero valid
      # survivors -> critique drops `g=0`'s ref; assembler still emits an
      # all-UNSET_ROUTED_EXPERT batch via `RoutedExpertsSpec`).
      prog_degen = harness.build_program(
          num_prompts=1,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          assembler_type=assembler_type,
          reward_fns=[
              lambda prompts, completions, **_: [1.0, float("nan")][
                  : len(prompts)
              ]
          ],
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      prog_degen.policy_version = engine._policy_version
      try:
        harness.orchestrator.run(prog_degen, bring_up=False)
        self.assertEqual(prog_degen.step, 1)
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_degen.close()

      # 5. Mid-run exception abort cleans up 100% of worker `used_bytes` and
      # `reserved_bytes`.
      def _boom_on_step_end(step: int, result: Any):
        del step, result
        raise RuntimeError("simulated mid-run abort")

      prog_abort = harness.build_program(
          num_prompts=4,
          max_steps=3,
          num_generations=2,
          mini_batch_size=1,
          max_staleness=2,
          assembler_type=assembler_type,
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
          on_step_end=_boom_on_step_end,
      )
      prog_abort.policy_version = engine._policy_version
      try:
        with self.assertRaisesRegex(RuntimeError, "simulated mid-run abort"):
          harness.orchestrator.run(prog_abort, bring_up=False)
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_abort.close()

      # 6. Missing staged ref during `fetch_staged` fails fast and still runs
      # terminal `clear_staged()` across all rollout workers. Match by exact
      # `traj_id` of the group being critiqued so the test never picks an
      # unconsumed off-policy group.
      dropped_once = False

      def _drop_staged_in_critique(prompts, completions, group_index, **kwargs):
        nonlocal dropped_once
        del prompts, completions
        tagged_ids = kwargs.get("test_prompt_id") or ()
        if not dropped_once and tagged_ids:
          target_traj_id = datatypes.format_traj_id(
              tagged_ids[0], group_index[0]
          )
          for rw in harness.rollout_workers.values():
            for ref_id, entry in list(rw._staging_buffer._entries.items()):
              if entry.traj_id == target_traj_id:
                rw.release_staged([ref_id])
                dropped_once = True
                break
            if dropped_once:
              break
        return [1.0 if g % 2 == 0 else -1.0 for g in group_index]

      tagged_prompts = harness.build_prompts(num_prompts=3)
      for idx, p in enumerate(tagged_prompts):
        p["metadata"] = {"test_prompt_id": f"prompt_{idx}"}
      prog_missing = harness.build_program(
          prompts=tagged_prompts,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          max_staleness=2,
          assembler_type=assembler_type,
          reward_fns=[_drop_staged_in_critique],
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      prog_missing.policy_version = engine._policy_version
      try:
        with self.assertRaisesRegex(
            Exception, "Missing or already-released ref_id="
        ):
          harness.orchestrator.run(prog_missing, bring_up=False)
        self.assertTrue(dropped_once)
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_missing.close()

      # 7. Retrying the exact same prompt IDs at the exact same `policy_version`
      # (`engine._policy_version`, unchanged since Run 6 aborted before sync)
      # must NOT dedupe or consume any unpolled worker-queued response from
      # Run 6 (`_e{epoch}` request-id isolation).
      prog_retry = harness.build_program(
          num_prompts=3,
          max_steps=1,
          num_generations=2,
          mini_batch_size=1,
          max_staleness=2,
          assembler_type=assembler_type,
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      prog_retry.policy_version = engine._policy_version
      try:
        harness.orchestrator.run(prog_retry, bring_up=False)
        self.assertEqual(prog_retry.step, 1)
        _assert_zero_rollout_staging(self, harness)
      finally:
        prog_retry.close()

  def test_relay_straggler_2s_retention_bound_and_zombie_reclaim(self):
    """Proves `2*S + span + 2` retains `v=0` at Step 7 and evicts at Step 10."""
    rollout_gate = threading.Event()
    critique_gate = threading.Event()
    fast_peer_staged = threading.Event()
    prompt_0_in_critique = threading.Event()

    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        enable_bulk_transport=True,
        partial_rollout=True,
        routed_experts_shape_spec=(4, 8),
        routed_experts_encode_mode="uint8",
        routed_experts_num_experts=64,
        rollout_max_prompt_length=256,
        response_tokens_per_turn=6,
    ) as harness:
      try:
        harness.orchestrator.bring_up_workers()

        # Plant one leaked unreferenced `v=0` zombie entry on `rollout_0`.
        w0 = harness.rollout_workers["rollout_0"]
        zombie_buf = w0._staging_buffer
        zombie_res = asyncio.run(zombie_buf.reserve(1024))
        zombie_arr = np.ones((8, 4, 8), dtype=np.uint8)
        zombie_ref_id = zombie_buf.commit(
            zombie_res, zombie_arr, traj_id="zombie_v0"
        )
        self.assertGreater(zombie_buf.used_bytes, 0)

        fast_traj_id = datatypes.format_traj_id("prompt_0", 0)
        slow_traj_id = datatypes.format_traj_id("prompt_0", 1)

        # Gate `prompt_0` slow peer (`slow_traj_id`) until `policy_version == 4`
        # while `fast_traj_id` finishes immediately at `v=0`!
        for sampler in harness.samplers.values():
          orig_sample = sampler.sample

          async def _gated_sample(
              sampling_requests=None, orig_fn=orig_sample, **kwargs
          ):
            req_id = getattr(sampling_requests, "request_id", "")
            if req_id == slow_traj_id and not rollout_gate.is_set():
              if not await asyncio.to_thread(rollout_gate.wait, 20.0):
                raise TimeoutError("Timed out waiting for rollout_gate")
            return await orig_fn(sampling_requests, **kwargs)

          sampler.sample = _gated_sample  # type: ignore[method-assign]

        def _wait_fast_peer_committed_at_v0() -> None:
          deadline = time.monotonic() + 10.0
          while time.monotonic() < deadline:
            for rw in harness.rollout_workers.values():
              for entry in list(rw._staging_buffer._entries.values()):
                if entry.traj_id == fast_traj_id:
                  self.assertEqual(entry.staged_policy_version, 0)
                  fast_peer_staged.set()
                  return
            time.sleep(0.01)
          raise TimeoutError(
              f"Timed out waiting for {fast_traj_id} commit at v=0"
          )

        def _gated_reward_fn(prompts, completions, group_index, **kwargs):
          del prompts, completions
          if not fast_peer_staged.is_set():
            _wait_fast_peer_committed_at_v0()
          tagged_ids = kwargs.get("test_prompt_id") or ()
          if any(pid == "prompt_0" for pid in tagged_ids):
            prompt_0_in_critique.set()
            if not critique_gate.wait(timeout=20.0):
              raise TimeoutError("Timed out waiting for critique_gate")
          return [1.0 if g % 2 == 0 else -1.0 for g in group_index]

        step_versions: list[tuple[int, list[str]]] = []
        program_ref: list[Any] = []
        orig_fwd_bwd = harness.fake_trainer.fwd_bwd

        def _gated_fwd_bwd(payload: datatypes.TrainerPayload) -> None:
          call_idx = harness.fake_trainer.fwd_bwd_calls
          orig_fwd_bwd(payload)
          # During Step 4's `train_step` (`call_idx == 4`, `v == 4`), the
          # orchestrator event loop is free awaiting this RPC. Wait until
          # 2 more groups (for Steps 5 & 6) are ready in `scored_q`, then wake
          # `slow_traj_id` so `prompt_0` passes `raw_q` at `v = 4` (`4 <= S`)
          # and parks at `critique_gate` until Step 6 completes!
          if call_idx == 4:
            prog = program_ref[0]
            deadline = time.monotonic() + 10.0
            while (
                prog.scored_q.ready_groups_count < 2
                and time.monotonic() < deadline
            ):
              time.sleep(0.01)
            self.assertGreaterEqual(prog.scored_q.ready_groups_count, 2)
            rollout_gate.set()
            if not prompt_0_in_critique.wait(timeout=10.0):
              raise TimeoutError("Timed out waiting for prompt_0 in critique")

        harness.fake_trainer.fwd_bwd = (  # type: ignore[method-assign]
            _gated_fwd_bwd
        )

        def _on_step_end(step: int, result: Any):
          del result
          if step == 6:
            critique_gate.set()

        prompts = harness.build_prompts(num_prompts=12, min_turns=0)
        for idx, p in enumerate(prompts):
          p["metadata"] = {"test_prompt_id": f"prompt_{idx}"}
        prompts[0]["generation_kwargs"]["min_turns"] = 1
        program = harness.build_program(
            prompts=prompts,
            max_steps=11,
            num_generations=2,
            mini_batch_size=1,
            batch_size=1,
            max_staleness=4,
            assembler_type="packed",
            reward_fns=[_gated_reward_fn],
            generation_args=datatypes.GenerationArgs(
                temperature=1.0, return_routed_experts=True
            ),
            on_step_end=_on_step_end,
        )
        program_ref.append(program)
        orig_commit = program.scored_q.commit

        def _recording_commit(step: int = 0, groups: Any = None):
          pids = [g[0].prompt_id for g in groups or () if g]
          step_versions.append((step, pids))
          # Before Step 10 (`V_new = 11 -> min_retained = 1`), `zombie_v0`
          # (`staged_policy_version = 0`) is still retained; at Step 10's
          # `sync_weights` (which runs right BEFORE `commit`), `zombie_v0`
          # HAS JUST BEEN EVICTED (`0 < 1`)!
          if step < 10:
            self.assertEqual(zombie_buf.zombie_evictions_total, 0)
          elif step == 10:
            self.assertEqual(zombie_buf.zombie_evictions_total, 1)
          orig_commit(step, groups=groups)

        program.scored_q.commit = (  # type: ignore[method-assign]
            _recording_commit
        )
        try:
          harness.orchestrator.run(program, bring_up=False)
          self.assertEqual(program.step, 11)
          consumed_by_step = dict(step_versions)
          self.assertIn("prompt_0", set().union(*consumed_by_step.values()))
          prompt0_step = next(
              s for s, pids in step_versions if "prompt_0" in pids
          )
          # `prompt_0` (fast peer staged at `v=0`) was consumed at Step >= 7
          # (`V_new = 7..8`, up to `2 * S = 8` versions later) without false
          # zombie eviction!
          self.assertGreaterEqual(prompt0_step, 7)
          self.assertEqual(zombie_buf.zombie_evictions_total, 1)
          self.assertNotIn(zombie_ref_id, zombie_buf._entries)
          _assert_zero_rollout_staging(self, harness)
        finally:
          program.close()
      finally:
        rollout_gate.set()
        critique_gate.set()

  def test_relay_aborts_on_mid_run_rollout_eviction_even_under_ft(self):
    """Pins the explicit contract: Relay does not tolerate rollout eviction."""
    with fake_cluster.FakeClusterHarness(
        num_rollout_workers=2,
        enable_bulk_transport=True,
        routed_experts_shape_spec=(4, 8),
        routed_experts_encode_mode="uint8",
        routed_experts_num_experts=64,
        rollout_max_prompt_length=192,
        response_tokens_per_turn=8,
        fault_tolerance_config=datatypes.FaultToleranceConfig(
            enabled=True,
            min_active_rollout_ratio=0.5,
            max_task_retries=1,
        ),
    ) as harness:
      program = harness.build_program(
          prompts=harness.build_prompts(num_prompts=4, min_turns=1),
          max_steps=4,
          mini_batch_size=1,
          batch_size=1,
          num_generations=2,
          assembler_type="packed",
          reward_fns=[
              lambda prompts, completions, group_index, **_: [
                  1.0 if g % 2 == 0 else -1.0 for g in group_index
              ]
          ],
          generation_args=datatypes.GenerationArgs(
              temperature=1.0, return_routed_experts=True
          ),
      )
      orig_fetch = (
          distributed_rl_engine.DistributedRLEngine.fetch_routed_experts
      )
      evicted_once = False

      async def _evict_before_fetch(self_engine, refs):
        nonlocal evicted_once
        if not evicted_once and refs:
          evicted_once = True
          self_engine._registry.unregister(refs[0].worker_id)
        return await orig_fetch(self_engine, refs)

      try:
        with mock.patch.object(
            distributed_rl_engine.DistributedRLEngine,
            "fetch_routed_experts",
            _evict_before_fetch,
        ):
          with self.assertRaisesRegex(
              RuntimeError, "was evicted before staged routed_experts"
          ):
            harness.orchestrator.run(program)
      finally:
        program.close()


if __name__ == "__main__":
  absltest.main()
