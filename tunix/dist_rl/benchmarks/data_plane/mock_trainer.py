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

"""Mock TrainerWorker builder and node entrypoint for data_plane_benchmark."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import logging
from typing import cast

import jax
from tunix.dist_rl.benchmarks.data_plane import data_plane_benchmark
from tunix.dist_rl.common import test_utils
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import trainer_worker

LOCAL_TRAINER_WORKER_ID = "trainer-0"


def build_mock_trainer_worker(
    hp: resolve_config.DistHyperParameters,
    worker_id_str: str,
) -> tuple[trainer_worker.TrainerWorker, test_utils.FakeTrainer]:
  """Constructs a `TrainerWorker` and its underlying `FakeTrainer`."""
  actor_kwargs = cast(
      Mapping[str, object], hp.config_mapping("actor_model_config")["kwargs"]
  )
  rl_train = hp.config_mapping("rl_training_config")
  fwd_bwd_sleep_s = data_plane_benchmark.kwarg_float(
      actor_kwargs, "mock_fwd_bwd_sleep_s", 0.1
  )
  shape_spec, num_experts, expected_ndim = (
      data_plane_benchmark.resolve_routed_experts_spec(
          hp.config_mapping("vllm_config"), actor_kwargs
      )
  )

  fake_trainer = test_utils.FakeTrainer(
      trainer_id=worker_id_str,
      fwd_bwd_sleep_s=fwd_bwd_sleep_s,
      expected_routed_experts_ndim=expected_ndim,
      routed_experts_shape_spec=shape_spec,
      num_experts=num_experts,
      synthetic_fwd_only=True,
  )
  worker_service = trainer_worker.TrainerWorker(
      trainer_factory=lambda: fake_trainer,
      worker_id=worker_id_str,
      logps_chunk_size=int(rl_train["compute_logps_chunk_size"]),
      logps_micro_batch_size=int(rl_train["compute_logps_micro_batch_size"]),
  )
  return worker_service, fake_trainer


def trainer_main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a no-op `FakeTrainer` `TrainerWorker` over gRPC + Bulk TCP."""
  jax.config.update("jax_platforms", "cpu")
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [MockTrainerNode] %(message)s",
      force=True,
  )
  hp = resolve_config.initialize(argv)
  node_utils.apply_library_env(hp, "trainer")
  config_sha256 = node_utils.log_resolved_config(hp, "trainer")
  worker_id_str = node_utils.worker_id(hp, "trainer", LOCAL_TRAINER_WORKER_ID)
  port = node_utils.service_port(hp, "trainer")
  node_utils.setup_otel_metrics(hp, "trainer", worker_id_str)

  worker_service, fake_trainer = build_mock_trainer_worker(hp, worker_id_str)

  def _close() -> None:
    rss_mib, hwm_mib = data_plane_benchmark.read_process_memory_mib()
    logging.info(
        "[DataPlaneBench][Trainer %s] Final summary: fwd_bwd_calls=%d"
        " fwd_only_calls=%d total_tokens=%d total_token_slots=%d"
        " total_routed_GiB=%.3f last_routed_shape=%s VmRSS=%.1f MiB"
        " VmHWM=%.1f MiB",
        worker_id_str,
        fake_trainer.fwd_bwd_calls,
        fake_trainer.fwd_only_calls,
        fake_trainer.total_tokens,
        fake_trainer.total_token_slots,
        fake_trainer.total_routed_experts_bytes / (1024.0**3),
        fake_trainer.last_routed_experts_shape,
        rss_mib,
        hwm_mib,
    )
    worker_service.stop()

  asyncio.run(
      node_utils.serve_worker(
          worker_service,
          process_context=process_context,
          role="trainer",
          worker_id_str=worker_id_str,
          port=port,
          config_sha256=config_sha256,
          close_fn=_close,
      )
  )
