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

"""End-to-end CPU + real gRPC Bulk TCP tests for data_plane_benchmark."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import contextlib
import pathlib
import pickle

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.dist_rl.benchmarks.data_plane import benchmark_rewards
from tunix.dist_rl.benchmarks.data_plane import data_plane_benchmark
from tunix.dist_rl.benchmarks.data_plane import mock_rollout
from tunix.dist_rl.benchmarks.data_plane import mock_trainer
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import orchestrator
from tunix.dist_rl.testing import fault_injection

_BENCH_DIR = pathlib.Path(data_plane_benchmark.__file__).parent
_YAML_PATH = _BENCH_DIR / "data_plane_benchmark.yaml"
_TOKENIZER_DIR = _BENCH_DIR / "tokenizer"


class DataPlaneBenchmarkTest(parameterized.TestCase):

  def _resolve(
      self, *extra_overrides: str
  ) -> resolve_config.DistHyperParameters:
    artifact_root = self.create_tempdir().full_path
    return resolve_config.initialize([
        f"override_config_file={_YAML_PATH}",
        f"model_config.model_id={_TOKENIZER_DIR}",
        f"tokenizer_config.tokenizer_path={_TOKENIZER_DIR}",
        f"cluster_config.artifact_root={artifact_root}",
        *extra_overrides,
    ])

  def test_offline_tokenizer_and_dataset_contract(self):
    hp = self._resolve(
        "data_config.num_records=8", "data_config.prompt_tokens=32"
    )
    tokenizer = node_utils.load_hf_tokenizer(hp)
    pad_id, eos_id = node_utils.pad_and_eos_ids(tokenizer)
    self.assertEqual(pad_id, 0)
    self.assertEqual(eos_id, 1)

    items = orchestrator.load_prompt_items(hp)
    first = items[0]
    self.assertIsNotNone(first)
    assert first is not None
    self.assertEqual(first["task_name"], "data_plane_benchmark")

    unshuffled = [
        r["prompt"]
        for r in data_plane_benchmark.create_dataset(
            num_records=12, prompt_tokens=8, shuffle=False, seed=7
        )
    ]
    shuffled = [
        r["prompt"]
        for r in data_plane_benchmark.create_dataset(
            num_records=12, prompt_tokens=8, shuffle=True, seed=7
        )
    ]
    self.assertCountEqual(unshuffled, shuffled)
    self.assertNotEqual(unshuffled, shuffled)
    with self.assertRaisesRegex(ValueError, "prompt_tokens"):
      data_plane_benchmark.create_dataset(num_records=4, prompt_tokens=3)

  def test_mock_env_max_steps_and_group_reward_variance(self):
    hp = self._resolve("data_config.prompt_tokens=32")
    tokenizer = node_utils.load_hf_tokenizer(hp)
    with self.assertRaisesRegex(ValueError, "prompt_tokens"):
      data_plane_benchmark.MockDataPlaneEnv(prompt_tokens=3)

    # Multi-turn (`max_steps=2`) must not terminate on the first non-final step.
    multi_env = data_plane_benchmark.MockDataPlaneEnv(
        seed=1, prompt_tokens=16, max_steps=2, group_index=2
    )
    obs0, _ = multi_env.reset()
    self.assertLen(
        tokenizer.encode(obs0["prompts"], add_special_tokens=False), 16
    )
    obs1, reward1, done1, info1 = multi_env.step("TOOL_CALL: search()")
    self.assertFalse(done1)
    self.assertEqual(reward1, 0.0)
    self.assertEqual(info1["steps"], 1)
    self.assertEqual(obs1["prompts"], "step 1 tok tok")

    obs2, reward2, done2, info2 = multi_env.step("TOOL_CALL: search()")
    self.assertTrue(done2)
    self.assertGreater(reward2, 0.0)
    self.assertEqual(info2["steps"], 2)
    self.assertEqual(obs2["prompts"], "bench done")

    rewards = []
    for group_idx in range(4):
      env = data_plane_benchmark.MockDataPlaneEnv(
          seed=1, prompt_tokens=32, group_index=group_idx
      )
      env.reset()
      _, reward, done, _ = env.step("FINAL_ANSWER: ok")
      self.assertTrue(done)
      rewards.append(reward)
    self.assertGreater(len(set(rewards)), 1)

  @parameterized.named_parameters(
      ("disabled", False, "uint10", None, None),
      ("uint8_200", True, "uint8", 200, 4),
      ("uint10_384", True, "uint10", 384, 5),
      ("int16_2048", True, "int16", 2048, 4),
      ("auto_uint8", True, "auto", 255, 4),
      ("auto_uint10", True, "auto", 256, 5),
      ("auto_int16", True, "auto", 1024, 4),
  )
  def test_routed_experts_spec_modes(
      self,
      return_routed: bool,
      mode: str,
      num_experts: int | None,
      expected_ndim: int | None,
  ):
    num_str = "null" if num_experts is None else str(num_experts)
    hp = self._resolve(
        f"vllm_config.return_routed_experts={str(return_routed).lower()}",
        f"vllm_config.routed_experts_encode_mode={mode}",
        f"vllm_config.routed_experts_num_experts={num_str}",
        "model_config.kwargs.mock_num_layers=6",
        "model_config.kwargs.mock_top_k=8",
    )
    _, fake_trainer = mock_trainer.build_mock_trainer_worker(
        hp, "trainer-0"
    )
    self.assertEqual(fake_trainer.expected_routed_experts_ndim, expected_ndim)

  @parameterized.named_parameters(
      ("missing_num_experts", "auto", "null", "routed_experts_num_experts"),
      ("uint8_overflow", "uint8", "256", "uint8"),
      ("uint10_overflow", "uint10", "1024", "uint10"),
  )
  def test_routed_experts_spec_rejects_invalid_num_experts(
      self, mode: str, num_experts_str: str, err_regex: str
  ):
    hp = self._resolve(
        "vllm_config.return_routed_experts=true",
        f"vllm_config.routed_experts_encode_mode={mode}",
        f"vllm_config.routed_experts_num_experts={num_experts_str}",
    )
    tokenizer = node_utils.load_hf_tokenizer(hp)
    with self.assertRaisesRegex(ValueError, err_regex):
      mock_rollout.build_mock_rollout_worker(hp, "rollout-0", tokenizer)
    with self.assertRaisesRegex(ValueError, err_regex):
      mock_trainer.build_mock_trainer_worker(hp, "trainer-0")

    bad_logps_hp = self._resolve("rollout_config.return_logprobs=false")
    with self.assertRaisesRegex(ValueError, "return_logprobs"):
      mock_rollout.build_mock_rollout_worker(
          bad_logps_hp, "rollout-0", tokenizer
      )
    bad_eos_hp = self._resolve("rollout_config.eos_tokens='tok tok'")
    with self.assertRaisesRegex(ValueError, "eos_tokens"):
      mock_rollout.build_mock_rollout_worker(bad_eos_hp, "rollout-0", tokenizer)

  def test_mock_rollout_worker_uses_word_tokenizer_and_partial_rollout(self):
    # Verifies word-level HF tokenization (no 4x character inflation), stock
    # `RolloutConfig` + `RawTextParser` wiring, `in_flight_weight_updates`, and
    # `max_resp - 1` default response tokens so `env.step` executes.
    hp = self._resolve(
        "data_config.prompt_tokens=32",
        "rollout_config.total_generation_steps=16",
        "model_config.kwargs.mock_num_layers=4",
        "model_config.kwargs.mock_top_k=8",
        "model_config.kwargs.mock_decode_sleep_s=0.0",
        "weight_sync_config.in_flight_weight_updates=true",
    )
    tokenizer = node_utils.load_hf_tokenizer(hp)
    worker = mock_rollout.build_mock_rollout_worker(
        hp, "rollout-check", tokenizer
    )
    try:
      self.assertTrue(worker.manager.partial_rollout)
      resp = asyncio.run(
          worker.generate(
              datatypes.RolloutRequest(
                  prompt="bench-0",
                  prompt_id="p0",
                  group_index=1,
                  max_turns=1,
                  max_response_length=16,
                  generation_kwargs={"return_routed_experts": True},
                  metadata={"env_config": {"seed": 1, "prompt_tokens": 32}},
              )
          )
      )
      assert isinstance(resp, datatypes.RolloutResponse)
      self.assertEqual(resp.status, "COMPLETED")
      assert resp.payload is not None
      self.assertEqual(resp.payload.traj["prompt_length"], 32)
      self.assertEqual(
          resp.payload.traj["status"], datatypes.TrajectoryStatus.SUCCEEDED.name
      )
      self.assertGreater(resp.payload.traj["trajectory_reward"], 0.0)
      self.assertEqual(int(resp.payload.traj["conversation_tokens"][-1]), 1)
      self.assertFalse(resp.payload.metadata["clipped"])
      self.assertEqual(
          resp.payload.traj["routed_experts"].shape, (32 + 15, 4, 2, 5)
      )
    finally:
      worker.stop()
      asyncio.run(worker.sampler.stop())

  def test_multi_turn_rollout_and_trainer_per_token_logps(self):
    hp = self._resolve(
        "data_config.num_records=4",
        "data_config.prompt_tokens=16",
        "rollout_config.total_generation_steps=64",
        "agentic_grpo_config.max_turns=3",
        "model_config.kwargs.mock_num_layers=4",
        "model_config.kwargs.mock_top_k=8",
        "model_config.kwargs.mock_decode_sleep_s=0.0",
    )
    tokenizer = node_utils.load_hf_tokenizer(hp)
    item = orchestrator.load_prompt_items(hp)[0]
    assert item is not None
    item_turns = item["max_turns"]
    item_meta = item["metadata"]
    assert isinstance(item_turns, int)
    assert isinstance(item_meta, Mapping)
    self.assertEqual(item_turns, 3)

    r_worker = mock_rollout.build_mock_rollout_worker(
        hp, "rollout-mt", tokenizer
    )
    t_worker, fake_trainer = mock_trainer.build_mock_trainer_worker(
        hp, "trainer-mt"
    )
    try:
      resp = asyncio.run(
          r_worker.generate(
              datatypes.RolloutRequest(
                  prompt=str(item["prompt"]),
                  prompt_id="mt-0",
                  group_index=2,
                  max_turns=item_turns,
                  max_response_length=64,
                  generation_kwargs={"return_routed_experts": True},
                  metadata=dict(item_meta),
              )
          )
      )
      assert isinstance(resp, datatypes.RolloutResponse)
      assert resp.payload is not None
      traj = resp.payload.traj
      self.assertEqual(
          traj["status"], datatypes.TrajectoryStatus.SUCCEEDED.name
      )
      self.assertEqual(traj["assistant_turn_lengths"], [15, 15, 15])
      self.assertEqual(int(traj["conversation_tokens"][-1]), 1)
      self.assertFalse(resp.payload.metadata["clipped"])
      expected_routed_len = int(traj["prompt_length"]) + int(
          traj["conversation_tokens"].size
      )
      self.assertEqual(
          traj["routed_experts"].shape, (expected_routed_len, 4, 2, 5)
      )

      routed_batch = np.ones((2, 16, 4, 2, 5), dtype=np.uint8)
      logps_resp = t_worker.per_token_logps(
          datatypes.LogprobsRequest(
              prompt_tokens=np.zeros((2, 0), dtype=np.int32),
              completion_tokens=np.ones((2, 16), dtype=np.int32),
              pad_id=0,
              eos_id=1,
              temperature=1.0,
              routed_experts=routed_batch,
          )
      )
      self.assertEqual(np.asarray(logps_resp.per_token_logps).shape, (2, 16))
      self.assertEqual(fake_trainer.fwd_only_calls, 2)
      self.assertEqual(
          fake_trainer.total_routed_experts_bytes, routed_batch.nbytes
      )

      scores = benchmark_rewards.rss_telemetry_reward(["p"], ["c"])
      self.assertEqual(scores, [0.0])
      with self.assertLogs(level="INFO") as cm:
        benchmark_rewards._log_final_orchestrator_summary()
      self.assertTrue(
          any("Final summary:" in msg for msg in cm.output), cm.output
      )
    finally:
      r_worker.stop()
      asyncio.run(r_worker.sampler.stop())
      t_worker.stop()

  def test_multi_worker_bulk_tcp_end_to_end_pipeline(self):
    # Exercises 2 RolloutWorker gRPC+BulkTCP servers -> stock StandardRLProgram
    # (SequencePackedBatchAssembler + uint10 encoding) -> 1 TrainerWorker
    # gRPC+BulkTCP server across 3 off-policy weight-sync steps.
    hp = self._resolve(
        "batch_size=2",
        "rl_training_config.mini_batch_size=2",
        "rl_training_config.train_micro_batch_size=4",
        "rl_training_config.max_steps=3",
        "rl_training_config.max_seq_tokens_per_device=2048",
        "rollout_config.max_prompt_length=256",
        "rollout_config.total_generation_steps=512",
        "data_config.num_records=32",
        "data_config.prompt_tokens=64",
        "model_config.mesh.shape=(4,1)",
        "model_config.kwargs.mock_num_layers=8",
        "model_config.kwargs.mock_top_k=8",
        "model_config.kwargs.mock_decode_sleep_s=0.0",
        "model_config.kwargs.mock_fwd_bwd_sleep_s=0.0",
        "agentic_grpo_config.num_generations=2",
        "agentic_grpo_config.off_policy_steps=2",
        "dist_training_config.precompile_trainer=true",
        # `target=local` enforces `rollout_replicas == 1` in `resolve_config`;
        # `target=k8s` permits `rollout_replicas=2` for in-process multi-worker
        # gRPC + Bulk TCP testing without launching Kubernetes resources.
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=test-image",
        "cluster_config.rollout_replicas=2",
    )
    tokenizer = node_utils.load_hf_tokenizer(hp)
    pad_id, eos_id = node_utils.pad_and_eos_ids(tokenizer)
    phase_timeouts, rpc_timeout_s = orchestrator.resolve_weight_sync_timeouts(
        hp
    )
    cluster = orchestrator.create_cluster(
        hp, weight_sync_timeouts=phase_timeouts
    )
    injector = fault_injection.FaultInjector()
    servers: list[fault_injection.RunningWorkerServer] = []
    try:
      t_worker, fake_trainer = mock_trainer.build_mock_trainer_worker(
          hp, "trainer-0"
      )
      t_server = fault_injection.RunningWorkerServer(
          fault_injection.FaultInjectingServer(
              t_worker,
              worker_id="trainer-0",
              injector=injector,
              enable_bulk_transport=True,
          )
      )
      t_port = t_server.start()
      servers.append(t_server)
      cluster.registry.register_from_hostname(
          "127.0.0.1",
          t_port,
          pickle.dumps({
              "service_type": "trainer",
              "service_port": t_port,
              "worker_id": "trainer-0",
          }),
          rpc_timeout_s=rpc_timeout_s,
      )

      for idx in range(2):
        wid = f"rollout-{idx}"
        r_worker = mock_rollout.build_mock_rollout_worker(
            hp, wid, tokenizer
        )
        r_server = fault_injection.RunningWorkerServer(
            fault_injection.FaultInjectingServer(
                r_worker,
                worker_id=wid,
                injector=injector,
                enable_bulk_transport=True,
            )
        )
        r_port = r_server.start()
        servers.append(r_server)
        cluster.registry.register_from_hostname(
            "127.0.0.1",
            r_port,
            pickle.dumps({
                "service_type": "rollout",
                "service_port": r_port,
                "worker_id": wid,
            }),
            rpc_timeout_s=rpc_timeout_s,
        )

      program = orchestrator.build_program(
          hp,
          algo_config=orchestrator.build_grpo_config(hp),
          reward_fns=hp.obtain_reward_fn(),
          pad_id=pad_id,
          eos_id=eos_id,
          prompt_items=orchestrator.load_prompt_items(hp),
          trajectory_store=cluster.trajectory_store,
          metrics_logging_options=None,
      )
      try:
        cluster.bring_up_workers(dummy_data=None)
        cluster.run(program=program, bring_up=False)
      finally:
        program.close()

      result = program.last_step_result
      self.assertIsNotNone(result)
      assert result is not None
      self.assertEqual(result.step, 2)
      self.assertEqual(result.policy_version, 3)
      self.assertGreater(result.reward_std, 0.0)
      self.assertGreaterEqual(fake_trainer.fwd_bwd_calls, 3)
      self.assertGreater(fake_trainer.total_tokens, 0)
      self.assertLess(
          fake_trainer.total_tokens, fake_trainer.total_token_slots
      )
      self.assertGreater(fake_trainer.total_routed_experts_bytes, 0)
      # 4 packed rows (actor fsdp=4), 2048 seq_len, 8 layers, uint10 (8//4=2, 5)
      self.assertEqual(
          fake_trainer.compiled_routed_experts_shape, (4, 2048, 8, 2, 5)
      )
      self.assertEqual(
          fake_trainer.last_routed_experts_shape, (4, 2048, 8, 2, 5)
      )
    finally:
      with contextlib.suppress(Exception):
        cluster.shutdown()
      with contextlib.suppress(Exception):
        cluster.registry.close()
      for srv in servers:
        with contextlib.suppress(Exception):
          srv.stop(grace=0.1)


if __name__ == "__main__":
  absltest.main()
