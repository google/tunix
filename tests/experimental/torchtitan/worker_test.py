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

"""Multi-process worker and engine tests using CPU Gloo and real FSDP2."""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import os
import pickle
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from etils import epath
import numpy as np
import portpicker
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp
from torchtitan.experiments.tpu import qwen3 as tpu_qwen3
from torchtitan.experiments.tpu import workarounds as tpu_workarounds
from torchtitan.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.testing import worker_registry_testing
from tunix.dist_rl.worker import remote_execution
from tunix.experimental.torchtitan import grpo_loss
from tunix.experimental.torchtitan.engine import TorchTitanEngine
from tunix.experimental.torchtitan.engine import TorchTitanEngineConfig
from tunix.experimental.torchtitan.worker import run_follower_loop
from tunix.experimental.torchtitan.worker import slice_payload_for_rank
from tunix.experimental.torchtitan.worker import TorchTitanTrainerWorker
from tunix.rl import algorithm_config

import multiprocessing


def _adam_moment_dtypes(optimizer: torch.optim.Optimizer) -> list[str]:
  dtypes: set[str] = set()
  for state in optimizer.state.values():
    for key in ("exp_avg", "exp_avg_sq"):
      val = state.get(key)
      if isinstance(val, torch.Tensor):
        dtypes.add(str(val.dtype))
  return sorted(dtypes)


def _make_payload(seed: int) -> datatypes.RLTrainerPayload:
  """Helper to construct deterministic distinct test payloads."""
  p_ids = np.array(
      [[10 + seed * 4, 11 + seed * 4, 12, 13], [14, 15, 16 + seed, 17]],
      dtype=np.int32,
  )
  p_mask = np.ones_like(p_ids)
  c_ids = np.array([[20 + seed, 21, 2], [22, 23 + seed, 2]], dtype=np.int32)
  c_mask = np.ones_like(c_ids)
  adv = np.array([1.0 + seed * 0.1, -1.0 - seed * 0.1], dtype=np.float32)
  return datatypes.RLTrainerPayload(
      prompt_ids=p_ids,
      prompt_mask=p_mask,
      completion_ids=c_ids,
      completion_mask=c_mask,
      advantages=adv,
  )


def _run_worker_process(
    rank: int,
    world_size: int,
    master_port: int,
    hf_dir: str,
    shared_result: dict[str, Any],
):
  torch.set_num_threads(1)
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)

  dist.init_process_group(
      backend="gloo",
      rank=rank,
      world_size=world_size,
  )

  dp_mesh = init_device_mesh(
      "cpu",
      (world_size,),
      mesh_dim_names=("fsdp",),
  )
  raw_control_pg = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  assert isinstance(raw_control_pg, dist.ProcessGroup)
  control_pg = cast(dist.ProcessGroup, raw_control_pg)
  dp_pg = dp_mesh.get_group()

  cfg = TorchTitanEngineConfig(
      model_name="qwen3",
      flavor="debugmodel",
      hf_assets_path=hf_dir,
      dtype="float32",
      mixed_precision_param="bfloat16",
      learning_rate=1e-3,
      device_type="cpu",
      load_hf_weights=True,
  )
  engine = TorchTitanEngine(
      config=cfg,
      dp_mesh=dp_mesh,
      device=torch.device("cpu"),
  )

  rank_act_dtypes: list[torch.dtype] = []
  rank_hook = engine.model.layers["0"].feed_forward.w1.register_forward_hook(
      lambda _mod, _inp, out: rank_act_dtypes.append(out.dtype)
  )

  if rank == 0:
    ref_engine = TorchTitanEngine(
        config=cfg,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    ref_act_dtypes: list[torch.dtype] = []
    ref_w1 = ref_engine.model.layers["0"].feed_forward.w1
    ref_hook = ref_w1.register_forward_hook(
        lambda _mod, _inp, out: ref_act_dtypes.append(out.dtype)
    )

    worker = TorchTitanTrainerWorker(
        engine=engine,
        control_pg=control_pg,
        dp_pg=dp_pg,
        worker_id="trainer_0",
    )
    stopped_events = []
    worker.set_on_stop(stopped_events.append)
    init_resp = worker.initialize()
    worker_info = worker.info()
    shared_result["init_metadata"] = init_resp.metadata
    shared_result["reinit_metadata"] = worker.initialize().metadata
    shared_result["start_metadata"] = worker.start().metadata
    shared_result["info_resources"] = worker_info.resources
    shared_result["info_instance_id"] = worker_info.instance_id
    shared_result["worker_instance_id"] = worker.instance_id

    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    loss_resp = worker.with_loss_fn(
        "grpo",
        algo_config=algo_cfg,
        pad_id=0,
        eos_id=2,
        fuse_agreement_in_loss=False,
        return_routed_experts=False,
    )
    shared_result["with_loss_fn_metadata"] = loss_resp.metadata
    ref_engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)

    payload1 = _make_payload(1)
    payload2 = _make_payload(2)

    req1 = datatypes.TrainRequest(request_id="req1", payload=payload1)
    req2 = datatypes.TrainRequest(request_id="req2", payload=payload2)

    # Test concurrent fwd_bwd execution under _step_lock
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      f1 = executor.submit(worker.fwd_bwd, req1)
      f2 = executor.submit(worker.fwd_bwd, req2)
      _ = f1.result()
      _ = f2.result()

    _ = ref_engine.fwd_bwd(payload1)
    _ = ref_engine.fwd_bwd(payload2)

    step, metrics = worker.update()
    _, ref_metrics = ref_engine.update()
    rank_hook.remove()
    ref_hook.remove()

    shared_result["fsdp_param_dtypes"] = sorted({
        str((p.to_local() if isinstance(p, DTensor) else p).dtype)
        for p in engine.model.parameters()
    })
    shared_result["ref_param_dtypes"] = sorted(
        {str(p.dtype) for p in ref_engine.model.parameters()}
    )
    shared_result["fsdp_opt_dtypes"] = _adam_moment_dtypes(engine.optimizer)
    shared_result["ref_opt_dtypes"] = _adam_moment_dtypes(ref_engine.optimizer)
    shared_result["fsdp_act_dtypes"] = sorted({str(d) for d in rank_act_dtypes})
    shared_result["ref_act_dtypes"] = sorted({str(d) for d in ref_act_dtypes})

    # Per token logps
    test_p = np.array([[10, 11, 12, 13], [14, 15, 16, 17]], dtype=np.int32)
    test_c = np.array([[20, 21, 2], [22, 23, 2]], dtype=np.int32)
    logp_req = datatypes.LogprobsRequest(
        request_id="logp_req",
        prompt_tokens=test_p,
        completion_tokens=test_c,
        pad_id=0,
        eos_id=2,
        temperature=1.0,
    )
    logp_resp = worker.per_token_logps(logp_req)
    ref_logps = ref_engine.per_token_logps(
        test_p, test_c, pad_id=0, temperature=1.0
    )

    # Test non-divisible batch size (B=3 on world_size=2)
    test_p_3 = np.array(
        [[10, 11, 12, 13], [14, 15, 16, 17], [18, 19, 20, 21]],
        dtype=np.int32,
    )
    test_c_3 = np.array([[20, 21, 2], [22, 23, 2], [24, 25, 2]], dtype=np.int32)
    logp_req_3 = datatypes.LogprobsRequest(
        request_id="logp_req_3",
        prompt_tokens=test_p_3,
        completion_tokens=test_c_3,
        pad_id=0,
        eos_id=2,
        temperature=1.0,
    )
    logp_resp_3 = worker.per_token_logps(logp_req_3)
    ref_logps_3 = ref_engine.per_token_logps(
        test_p_3, test_c_3, pad_id=0, temperature=1.0
    )

    # Weight sync
    hf_weights = worker.prepare_weight_sync()
    worker.release_weight_sync()
    ref_hf_sd = ref_engine.export_hf_state_dict(is_rank_zero=True)

    worker.stop(abort=False)

    shared_result["step"] = step
    shared_result["metrics"] = metrics
    shared_result["ref_metrics"] = ref_metrics
    shared_result["worker_logps"] = logp_resp.per_token_logps
    shared_result["ref_logps"] = ref_logps
    shared_result["worker_logps_3"] = logp_resp_3.per_token_logps
    shared_result["ref_logps_3"] = ref_logps_3
    shared_result["hf_keys"] = list(hf_weights.keys())
    shared_result["ref_hf_keys"] = list(ref_hf_sd.keys())
    shared_result["hf_dtypes"] = sorted(
        {p.dtype for p in hf_weights.values()}
    )
    shared_result["ref_hf_dtypes"] = sorted(
        {str(t.dtype) for t in ref_hf_sd.values()}
    )
    shared_result["weights_match"] = all(
        np.array_equal(
            hf_weights[k].buffer,
            (
                ref_hf_sd[k].view(torch.uint16).numpy()
                if ref_hf_sd[k].dtype == torch.bfloat16
                else ref_hf_sd[k].numpy()
            ),
        )
        for k in hf_weights
    )
    shared_result["stopped_events"] = stopped_events
  else:
    run_follower_loop(engine, control_pg, dp_pg, rank)
    rank_hook.remove()

  dist.destroy_process_group()


def _run_checkpoint_worker_process(
    rank: int,
    world_size: int,
    master_port: int,
    hf_dir: str,
    ckpt_dir: str,
    shared_result: dict[str, Any],
):
  torch.set_num_threads(1)
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)

  dist.init_process_group(
      backend="gloo",
      rank=rank,
      world_size=world_size,
  )

  dp_mesh = init_device_mesh(
      "cpu",
      (world_size,),
      mesh_dim_names=("fsdp",),
  )
  raw_control_pg = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  assert isinstance(raw_control_pg, dist.ProcessGroup)
  control_pg = cast(dist.ProcessGroup, raw_control_pg)
  dp_pg = dp_mesh.get_group()

  cfg = TorchTitanEngineConfig(
      model_name="qwen3",
      flavor="debugmodel",
      hf_assets_path=hf_dir,
      checkpoint_dir=ckpt_dir,
      save_interval_steps=2,
      dtype="float32",
      mixed_precision_param="bfloat16",
      learning_rate=1e-3,
      init_lr=1e-4,
      end_lr=1e-5,
      warmup_steps=2,
      decay_steps=6,
      device_type="cpu",
      load_hf_weights=True,
  )
  engine = TorchTitanEngine(
      config=cfg,
      dp_mesh=dp_mesh,
      device=torch.device("cpu"),
  )

  if rank == 0:
    worker = TorchTitanTrainerWorker(
        engine=engine,
        control_pg=control_pg,
        dp_pg=dp_pg,
        worker_id="trainer_0",
    )
    worker.initialize()
    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    worker.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)

    payloads = [_make_payload(i) for i in range(5)]

    # Continuous Run Steps 1..5
    # Step 1
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step1", payload=payloads[0])
    )
    worker.update()
    resp1 = worker.save_checkpoint(
        metadata={"global_step": 1, "step": 1}, step=1
    )
    shared_result["step_1_saved"] = resp1.metadata["checkpoint_saved"]

    # Step 2
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step2", payload=payloads[1])
    )
    worker.update()
    pv2_req = datatypes.WeightSyncRequest(
        policy_version=2, request_id="ckpt_pv2"
    )
    _ = worker.prepare_weight_sync(pv2_req)
    worker.release_weight_sync(pv2_req)
    resp2 = worker.save_checkpoint(
        metadata={"global_step": 2, "step": 2}, step=2
    )
    shared_result["step_2_saved"] = resp2.metadata["checkpoint_saved"]

    # Step 3
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step3", payload=payloads[2])
    )
    _, m3_cont = worker.update()

    # Step 4
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step4", payload=payloads[3])
    )
    _, m4_cont = worker.update()

    # Step 5
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step5", payload=payloads[4])
    )
    _, m5_cont = worker.update()

    weights_step5_cont = worker.prepare_weight_sync()
    worker.release_weight_sync()

    # Now Restore Checkpoint at Step 2 and Re-run Steps 3..5
    restored_meta = worker.restore_checkpoint(step=2)
    shared_result["restored_meta"] = restored_meta
    shared_result["restored_train_step"] = worker._engine.train_step
    shared_result["restored_policy_version"] = worker._engine.policy_version
    shared_result["restored_param_dtypes"] = sorted({
        str((p.to_local() if isinstance(p, DTensor) else p).dtype)
        for p in engine.model.parameters()
    })
    shared_result["restored_opt_dtypes"] = _adam_moment_dtypes(engine.optimizer)

    # Re-run Step 3
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step3_re", payload=payloads[2])
    )
    _, m3_rest = worker.update()

    # Re-run Step 4
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step4_re", payload=payloads[3])
    )
    _, m4_rest = worker.update()

    # Re-run Step 5
    worker.fwd_bwd(
        datatypes.TrainRequest(request_id="step5_re", payload=payloads[4])
    )
    _, m5_rest = worker.update()

    weights_step5_rest = worker.prepare_weight_sync()
    worker.release_weight_sync()

    shared_result["m3_cont"] = m3_cont
    shared_result["m3_rest"] = m3_rest
    shared_result["m4_cont"] = m4_cont
    shared_result["m4_rest"] = m4_rest
    shared_result["m5_cont"] = m5_cont
    shared_result["m5_rest"] = m5_rest

    shared_result["step5_weights_parity"] = all(
        np.allclose(
            weights_step5_cont[k].buffer,
            weights_step5_rest[k].buffer,
            atol=1e-6,
        )
        for k in weights_step5_cont
    )

    worker.stop(abort=False)
  else:
    run_follower_loop(engine, control_pg, dp_pg, rank)

  dist.destroy_process_group()


def _run_custom_loss_worker_process(
    rank: int,
    world_size: int,
    master_port: int,
    hf_dir: str,
    shared_result: dict[str, Any],
):
  torch.set_num_threads(1)
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(master_port)
  os.environ["RANK"] = str(rank)
  os.environ["WORLD_SIZE"] = str(world_size)

  dist.init_process_group(
      backend="gloo",
      rank=rank,
      world_size=world_size,
  )

  dp_mesh = init_device_mesh(
      "cpu",
      (world_size,),
      mesh_dim_names=("fsdp",),
  )
  raw_control_pg = dist.new_group(ranks=list(range(world_size)), backend="gloo")
  assert isinstance(raw_control_pg, dist.ProcessGroup)
  control_pg = cast(dist.ProcessGroup, raw_control_pg)
  dp_pg = dp_mesh.get_group()

  cfg = TorchTitanEngineConfig(
      model_name="qwen3",
      flavor="debugmodel",
      hf_assets_path=hf_dir,
      dtype="float32",
      mixed_precision_param="float32",
      learning_rate=1e-3,
      device_type="cpu",
      load_hf_weights=True,
  )
  engine = TorchTitanEngine(
      config=cfg,
      dp_mesh=dp_mesh,
      device=torch.device("cpu"),
  )

  if rank == 0:
    ref_engine = TorchTitanEngine(
        config=cfg,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    worker = TorchTitanTrainerWorker(
        engine=engine,
        control_pg=control_pg,
        dp_pg=dp_pg,
        worker_id="trainer_0",
    )
    worker_1r = TorchTitanTrainerWorker(
        engine=ref_engine,
        control_pg=None,
        dp_pg=None,
        worker_id="trainer_ref",
    )
    worker.initialize()
    worker_1r.initialize()

    def _make_weighted_policy_loss(weight: float):
      def _custom_loss(
          inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
          algo_cfg: algorithm_config.AlgorithmConfig,
      ) -> algorithm_config.PolicyLossOutput[torch.Tensor]:
        import torch as local_torch  # pylint: disable=g-import-not-at-top,reimported

        del algo_cfg
        adv = (
            inputs.advantages.unsqueeze(1)
            if inputs.advantages.ndim == 1
            else inputs.advantages
        )
        mask = inputs.completion_mask.to(local_torch.float32)
        loss_sum = -local_torch.sum(
            inputs.per_token_logps * mask * adv * weight
        )
        denom = local_torch.sum(mask).clamp(min=1.0)
        return algorithm_config.PolicyLossOutput(
            loss_sum=loss_sum,
            denominator=denom,
            metrics={
                "custom_metric": local_torch.sum(mask * weight).detach(),
            },
        )

      return _custom_loss

    custom_loss = _make_weighted_policy_loss(0.75)
    try:
      pickle.dumps(custom_loss)
      shared_result["stdlib_pickle_failed"] = False
    except (AttributeError, pickle.PicklingError):
      shared_result["stdlib_pickle_failed"] = True

    algo_cfg = algorithm_config.AlgorithmConfig(
        beta=0.0,
        use_rollout_logps=False,
    )
    resp = worker.with_loss_fn(
        custom_loss, algo_config=algo_cfg, pad_id=0, eos_id=2
    )
    worker_1r.with_loss_fn(
        custom_loss, algo_config=algo_cfg, pad_id=0, eos_id=2
    )
    shared_result["loss_fn_configured"] = resp.metadata["loss_fn_configured"]

    payload1 = _make_payload(1)
    payload2 = _make_payload(2)
    worker.fwd_bwd(datatypes.TrainRequest(request_id="c1", payload=payload1))
    worker.fwd_bwd(datatypes.TrainRequest(request_id="c2", payload=payload2))
    worker_1r.fwd_bwd(datatypes.TrainRequest(request_id="r1", payload=payload1))
    worker_1r.fwd_bwd(datatypes.TrainRequest(request_id="r2", payload=payload2))

    step, metrics = worker.update()
    ref_step, ref_metrics = worker_1r.update()
    worker.stop(abort=False)
    worker_1r.stop(abort=False)

    shared_result["step"] = step
    shared_result["ref_step"] = ref_step
    shared_result["metrics"] = metrics
    shared_result["ref_metrics"] = ref_metrics
  else:
    run_follower_loop(engine, control_pg, dp_pg, rank)

  dist.destroy_process_group()


class TorchTitanWorkerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    # Prepare HF weights
    model_spec = tpu_qwen3.model_registry("debugmodel", attn_backend="sdpa")
    model_spec.model.enable_weight_tying = False
    dummy_model = model_spec.model.build().to(torch.float32)
    torch.manual_seed(42)
    with torch.no_grad():
      for p in dummy_model.parameters():
        p.data.normal_(0.0, 0.02)
    adapter = Qwen3StateDictAdapter(model_spec.model, None)
    hf_sd = adapter.to_hf(dummy_model.state_dict())
    safetensors.torch.save_file(
        hf_sd, os.path.join(self.temp_dir, "model.safetensors")
    )
    with open(os.path.join(self.temp_dir, "config.json"), "w") as f:
      json.dump({"tie_word_embeddings": False}, f)

  def test_slice_payload_for_rank(self):
    p_ids = np.arange(8).reshape(4, 2)
    p_mask = np.ones((4, 2), dtype=np.int32)
    c_ids = np.arange(12).reshape(4, 3)
    c_mask = np.ones((4, 3), dtype=np.int32)
    adv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    payload = datatypes.RLTrainerPayload(
        prompt_ids=p_ids,
        prompt_mask=p_mask,
        completion_ids=c_ids,
        completion_mask=c_mask,
        advantages=adv,
    )

    sliced_0 = slice_payload_for_rank(payload, rank=0, world_size=2)
    sliced_1 = slice_payload_for_rank(payload, rank=1, world_size=2)

    self.assertEqual(sliced_0.prompt_ids.shape, (2, 2))
    self.assertEqual(sliced_1.prompt_ids.shape, (2, 2))
    np.testing.assert_array_equal(sliced_0.advantages, [1.0, 2.0])
    np.testing.assert_array_equal(sliced_1.advantages, [3.0, 4.0])

    with self.assertRaises(ValueError):
      slice_payload_for_rank(payload, rank=0, world_size=3)

  def test_grpo_config_field_coverage(self):
    grpo_fields = {
        f.name for f in dataclasses.fields(algorithm_config.GRPOConfig)
    }
    self.assertTrue(
        grpo_loss._LOSS_FIELDS_IMPLEMENTED.isdisjoint(
            grpo_loss._FIELDS_IGNORED_BY_LOSS
        )
    )
    self.assertEqual(
        grpo_fields,
        grpo_loss._LOSS_FIELDS_IMPLEMENTED | grpo_loss._FIELDS_IGNORED_BY_LOSS,
    )

  def test_with_loss_fn_validation(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )

    # Valid GRPOConfig
    valid_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=valid_cfg, pad_id=0, eos_id=2)
    self.assertIs(engine.algo_config, valid_cfg)
    self.assertEqual(engine.pad_id, 0)
    self.assertEqual(engine.eos_id, 2)

    # Rejection: unknown registered loss name or wrong loss_fn type or role
    with self.assertRaisesRegex(ValueError, "Unknown PyTorch policy loss"):
      engine.with_loss_fn("nonexistent_loss", algo_config=valid_cfg)
    with self.assertRaisesRegex(TypeError, "registered loss name"):
      engine.with_loss_fn(123, algo_config=valid_cfg)  # type: ignore[arg-type]
    with self.assertRaisesRegex(ValueError, "Role.ACTOR"):
      engine.with_loss_fn(
          "grpo", algo_config=valid_cfg, role=datatypes.Role.CRITIC
      )

    # Rejection: unknown GRPOConfig subclass field
    @dataclasses.dataclass(kw_only=True)
    class _ExtendedGRPOConfig(algorithm_config.GRPOConfig):
      new_unhandled_knob: bool = True

    with self.assertRaisesRegex(ValueError, "new_unhandled_knob"):
      engine.with_loss_fn("grpo", algo_config=_ExtendedGRPOConfig())

    # Rejection: invalid policy_loss_fn
    with self.assertRaisesRegex(ValueError, "policy_loss_fn='grpo'"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              policy_loss_fn="ppo",
              loss_algo="grpo",
              loss_agg_mode="token-mean",
          ),
      )

    # Rejection: epsilon_c
    with self.assertRaisesRegex(ValueError, "epsilon_c"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              epsilon_c=0.1,
          ),
      )

    # Rejection: sampler_is
    with self.assertRaisesRegex(ValueError, "sampler_is"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              sampler_is="token",
          ),
      )

    # Rejection: sampler_rs
    with self.assertRaisesRegex(ValueError, "sampler_rs"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              sampler_rs="token",
              sampler_rs_min=0.5,
              sampler_rs_max=1.5,
          ),
      )

    # Rejection: score_centering
    with self.assertRaisesRegex(ValueError, "score_centering"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              score_centering=True,
          ),
      )

    # Rejection: fuse_agreement_in_loss
    with self.assertRaisesRegex(ValueError, "fuse_agreement_in_loss"):
      engine.with_loss_fn(
          "grpo", algo_config=valid_cfg, fuse_agreement_in_loss=True
      )

    # Rejection: return_routed_experts
    with self.assertRaisesRegex(ValueError, "return_routed_experts"):
      engine.with_loss_fn(
          "grpo", algo_config=valid_cfg, return_routed_experts=True
      )

    # Rejection: seq_logprob_error_threshold
    with self.assertRaisesRegex(ValueError, "seq_logprob_error_threshold"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              seq_logprob_error_threshold=2.0,
          ),
      )

    # Rejection: invalid kl_loss_mode
    with self.assertRaisesRegex(ValueError, "kl_loss_mode"):
      engine.with_loss_fn(
          "grpo",
          algo_config=algorithm_config.GRPOConfig(
              loss_algo="grpo",
              loss_agg_mode="token-mean",
              kl_loss_mode="unsupported_kl",
          ),
      )

  def test_fwd_bwd_segment_ids_and_sampler_is_weights_rejected(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)
    payload = _make_payload(1)
    payload_with_segments = datatypes.RLTrainerPayload(
        prompt_ids=payload.prompt_ids,
        prompt_mask=payload.prompt_mask,
        completion_ids=payload.completion_ids,
        completion_mask=payload.completion_mask,
        advantages=payload.advantages,
        segment_ids=np.ones_like(payload.prompt_ids),
    )
    with self.assertRaisesRegex(ValueError, "segment_ids"):
      engine.fwd_bwd(payload_with_segments)

    payload_with_is_weights = datatypes.RLTrainerPayload(
        prompt_ids=payload.prompt_ids,
        prompt_mask=payload.prompt_mask,
        completion_ids=payload.completion_ids,
        completion_mask=payload.completion_mask,
        advantages=payload.advantages,
        sampler_is_weights=np.ones_like(
            payload.completion_mask, dtype=np.float32
        ),
    )
    with self.assertRaisesRegex(ValueError, "sampler_is_weights"):
      engine.fwd_bwd(payload_with_is_weights)

  def test_splash_attention_cpu_and_patch(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        use_splash_attention=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    self.assertIsNotNone(engine.model)
    tpu_workarounds.use_tokamax_splash_attention_patch(engine.model, None)

  def test_multiprocess_worker_loop_and_hf_export(self):
    self.assertTrue(TorchTitanEngineConfig().load_hf_weights)
    world_size = 2
    master_port = portpicker.pick_unused_port()
    with mp.Manager() as manager:  # pyrefly: ignore[missing-attribute]
      managed_result = manager.dict()
      mp.spawn(
          _run_worker_process,
          args=(world_size, master_port, self.temp_dir, managed_result),
          nprocs=world_size,
          join=True,
      )
      shared_result = dict(managed_result)

    # Verify lifecycle flags without legacy capability negotiation keys
    self.assertTrue(shared_result["init_metadata"]["initialized"])
    self.assertTrue(shared_result["reinit_metadata"]["ready"])
    self.assertTrue(shared_result["start_metadata"]["started"])
    self.assertTrue(
        shared_result["with_loss_fn_metadata"]["loss_fn_configured"]
    )
    self.assertNotEmpty(shared_result["worker_instance_id"])
    self.assertEqual(
        shared_result["info_instance_id"],
        shared_result["worker_instance_id"],
    )
    for forbidden_key in (
        "accepts_callable_loss",
        "requires_target_state",
        "needs_aot_compile",
        "supported_losses",
        "trainer_framework",
    ):
      self.assertNotIn(forbidden_key, shared_result["init_metadata"])
      self.assertNotIn(forbidden_key, shared_result["info_resources"])

    self.assertEqual(shared_result["step"], 1)
    metrics = shared_result["metrics"]
    self.assertIn("loss", metrics)
    self.assertIn("grad_norm", metrics)
    self.assertIn("learning_rate", metrics)

    # Parity with 1-rank reference
    ref_metrics = shared_result["ref_metrics"]
    self.assertAlmostEqual(metrics["loss"], ref_metrics["loss"], places=4)
    self.assertAlmostEqual(
        metrics["grad_norm"], ref_metrics["grad_norm"], places=4
    )

    # 2-rank FSDP2 MixedPrecisionPolicy casts embeddings/RMSNorm to bf16 while
    # 1-rank torch.autocast keeps them fp32 (~1.2e-4 logp delta on bf16).
    np.testing.assert_allclose(
        shared_result["worker_logps"],
        shared_result["ref_logps"],
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        shared_result["worker_logps_3"],
        shared_result["ref_logps_3"],
        rtol=1e-3,
        atol=1e-3,
    )

    self.assertEqual(shared_result["fsdp_param_dtypes"], ["torch.float32"])
    self.assertEqual(shared_result["ref_param_dtypes"], ["torch.float32"])
    self.assertEqual(shared_result["fsdp_opt_dtypes"], ["torch.float32"])
    self.assertEqual(shared_result["ref_opt_dtypes"], ["torch.float32"])
    self.assertEqual(shared_result["fsdp_act_dtypes"], ["torch.bfloat16"])
    self.assertEqual(shared_result["ref_act_dtypes"], ["torch.bfloat16"])
    self.assertEqual(shared_result["hf_dtypes"], ["bf16"])
    self.assertEqual(shared_result["ref_hf_dtypes"], ["torch.bfloat16"])

    self.assertEqual(
        set(shared_result["hf_keys"]),
        set(shared_result["ref_hf_keys"]),
    )
    self.assertTrue(shared_result["weights_match"])
    self.assertEqual(shared_result["stopped_events"], [False])

  def test_sequence_bucketing_parity(self):
    cfg_unbucketed = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        seq_bucket_size=0,
    )
    cfg_bucketed = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        seq_bucket_size=16,
    )
    engine_unbucketed = TorchTitanEngine(
        config=cfg_unbucketed,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    engine_bucketed = TorchTitanEngine(
        config=cfg_bucketed,
        dp_mesh=None,
        device=torch.device("cpu"),
    )

    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    engine_unbucketed.with_loss_fn(
        "grpo", algo_config=algo_cfg, pad_id=0, eos_id=2
    )
    engine_bucketed.with_loss_fn(
        "grpo", algo_config=algo_cfg, pad_id=0, eos_id=2
    )

    prompt_ids = np.array(
        [[10, 11, 12, 13, 14], [0, 0, 15, 16, 17]], dtype=np.int32
    )
    prompt_mask = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=np.int32)
    completion_ids = np.array(
        [[20, 21, 22, 23, 24, 25, 2], [26, 27, 28, 2, 0, 0, 0]], dtype=np.int32
    )
    completion_mask = np.array(
        [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]], dtype=np.int32
    )
    advantages = np.array([1.5, -0.5], dtype=np.float32)

    payload = datatypes.RLTrainerPayload(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=advantages,
    )

    logps_unbucketed = engine_unbucketed.per_token_logps(
        prompt_ids, completion_ids, pad_id=0, temperature=1.0
    )
    logps_bucketed = engine_bucketed.per_token_logps(
        prompt_ids, completion_ids, pad_id=0, temperature=1.0
    )
    self.assertEqual(logps_unbucketed.shape, (2, 7))
    self.assertEqual(logps_bucketed.shape, (2, 7))
    np.testing.assert_allclose(logps_unbucketed, logps_bucketed, atol=1e-5)

    _ = engine_unbucketed.fwd_bwd(payload)
    _ = engine_bucketed.fwd_bwd(payload)

    step_u, update_u = engine_unbucketed.update()
    step_b, update_b = engine_bucketed.update()
    self.assertEqual(step_u, step_b)
    self.assertAlmostEqual(update_u["loss"], update_b["loss"], places=4)
    self.assertAlmostEqual(
        update_u["grad_norm"], update_b["grad_norm"], places=4
    )

  def test_sequence_bucketing_256_unaligned_2d_advantages(self):
    cfg_unbucketed = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        seq_bucket_size=0,
    )
    cfg_bucketed = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        seq_bucket_size=256,
    )
    engine_unbucketed = TorchTitanEngine(
        config=cfg_unbucketed,
        dp_mesh=None,
        device=torch.device("cpu"),
    )
    engine_bucketed = TorchTitanEngine(
        config=cfg_bucketed,
        dp_mesh=None,
        device=torch.device("cpu"),
    )

    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    engine_unbucketed.with_loss_fn(
        "grpo", algo_config=algo_cfg, pad_id=0, eos_id=2
    )
    engine_bucketed.with_loss_fn(
        "grpo", algo_config=algo_cfg, pad_id=0, eos_id=2
    )

    prompt_ids = np.array(
        [[10, 11, 12, 13, 14], [0, 0, 15, 16, 17]], dtype=np.int32
    )
    prompt_mask = np.array([[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=np.int32)
    completion_ids = np.full((2, 10), 21, dtype=np.int32)
    completion_ids[0, 9] = 2
    completion_ids[1, 6] = 2
    completion_ids[1, 7:] = 0
    completion_mask = np.array(
        [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]],
        dtype=np.int32,
    )
    advantages_2d = np.tile(
        np.array([[1.5], [-0.5]], dtype=np.float32), (1, 10)
    )

    payload = datatypes.RLTrainerPayload(
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        advantages=advantages_2d,
    )

    _ = engine_unbucketed.fwd_bwd(payload)
    _ = engine_bucketed.fwd_bwd(payload)

    step_u, update_u = engine_unbucketed.update()
    step_b, update_b = engine_bucketed.update()
    self.assertEqual(step_u, step_b)
    self.assertAlmostEqual(update_u["loss"], update_b["loss"], places=4)
    self.assertAlmostEqual(
        update_u["grad_norm"], update_b["grad_norm"], places=4
    )

  def test_multiprocess_checkpoint_save_and_restore(self):
    world_size = 2
    master_port = portpicker.pick_unused_port()
    ckpt_dir = self.create_tempdir("ckpt_dir").full_path
    with mp.Manager() as manager:  # pyrefly: ignore[missing-attribute]
      managed_result = manager.dict()
      mp.spawn(
          _run_checkpoint_worker_process,
          args=(
              world_size,
              master_port,
              self.temp_dir,
              ckpt_dir,
              managed_result,
          ),
          nprocs=world_size,
          join=True,
      )
      shared_result = dict(managed_result)

    self.assertFalse(shared_result["step_1_saved"])
    self.assertTrue(shared_result["step_2_saved"])

    step_2_meta_path = os.path.join(ckpt_dir, "step_2", "metadata.json")
    self.assertTrue(os.path.exists(step_2_meta_path))
    with open(step_2_meta_path, "r") as f:
      meta_json = json.load(f)
    self.assertEqual(meta_json["global_step"], 2)
    self.assertEqual(meta_json["step"], 2)
    self.assertEqual(meta_json["policy_version"], 2)

    self.assertEqual(shared_result["restored_train_step"], 2)
    self.assertEqual(shared_result["restored_policy_version"], 2)
    self.assertEqual(shared_result["restored_param_dtypes"], ["torch.float32"])
    self.assertEqual(shared_result["restored_opt_dtypes"], ["torch.float32"])

    # 5-step continuous vs. step-2 restored numerical parity
    m3_c, m3_r = shared_result["m3_cont"], shared_result["m3_rest"]
    m4_c, m4_r = shared_result["m4_cont"], shared_result["m4_rest"]
    m5_c, m5_r = shared_result["m5_cont"], shared_result["m5_rest"]

    self.assertAlmostEqual(m3_c["loss"], m3_r["loss"], places=5)
    self.assertAlmostEqual(
        m3_c["learning_rate"], m3_r["learning_rate"], places=5
    )

    self.assertAlmostEqual(m4_c["loss"], m4_r["loss"], places=5)
    self.assertAlmostEqual(
        m4_c["learning_rate"], m4_r["learning_rate"], places=5
    )

    self.assertAlmostEqual(m5_c["loss"], m5_r["loss"], places=5)
    self.assertAlmostEqual(
        m5_c["learning_rate"], m5_r["learning_rate"], places=5
    )

    self.assertTrue(
        shared_result["step5_weights_parity"],
        "Final step-5 exported weights must match exactly (atol=1e-6) between"
        " continuous and restored runs.",
    )

  def test_dcp_checkpoint_uses_public_accelerator_synchronize(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    ckpt_dir = self.create_tempdir("accel_sync_ckpt").full_path

    fake_tpu_mod = mock.Mock()
    fake_tpu_mod.synchronize.side_effect = RuntimeError(
        "torch.tpu.synchronize is internal to TorchTPU"
    )
    engine.device = cast(torch.device, mock.Mock(spec=torch.device, type="tpu"))

    with (
        mock.patch.object(torch.accelerator, "is_available", return_value=True),
        mock.patch.object(torch.accelerator, "synchronize") as mock_sync,
        mock.patch.object(torch, "tpu", fake_tpu_mod, create=True),
    ):
      lr_sd = engine.save_dcp_checkpoint(ckpt_dir)
      self.assertEqual(mock_sync.call_count, 1)
      engine.load_dcp_checkpoint(
          ckpt_dir,
          train_step=3,
          policy_version=3,
          lr_scheduler_state=lr_sd,
      )
      self.assertEqual(mock_sync.call_count, 2)
      fake_tpu_mod.synchronize.assert_not_called()
    self.assertEqual(engine.train_step, 3)
    self.assertEqual(engine.policy_version, 3)

  def test_checkpoint_policy_version_override_and_restore(self):
    ckpt_dir = self.create_tempdir("ckpt_pv_override").full_path
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        checkpoint_dir=ckpt_dir,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()

    payload = _make_payload(1)
    _ = worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
    _, _ = worker.update()
    _ = worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
    _, _ = worker.update()
    self.assertEqual(engine.train_step, 2)
    self.assertEqual(engine.policy_version, 0)

    resp = worker.save_checkpoint(
        metadata={"step": 2, "global_step": 1, "policy_version": 1},
        force=True,
    )
    self.assertTrue(resp.metadata["checkpoint_saved"])

    step_2_meta = json.loads(
        (epath.Path(ckpt_dir) / "step_2" / "metadata.json").read_text()
    )
    self.assertEqual(step_2_meta["policy_version"], 1)

    public_meta = worker.restore_checkpoint(step=2)
    self.assertIsNotNone(public_meta)
    self.assertEqual(public_meta["policy_version"], 1)
    self.assertEqual(engine.policy_version, 1)
    self.assertEqual(worker.heartbeat().policy_version, 1)

  def test_prepare_weight_sync_aligns_policy_version(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()
    self.assertEqual(engine.policy_version, 0)

    sync_req = datatypes.WeightSyncRequest(policy_version=5, request_id="r5")
    _ = worker.prepare_weight_sync(sync_req)
    self.assertEqual(engine.policy_version, 5)
    self.assertEqual(worker.heartbeat().policy_version, 5)

  def test_prepare_weight_sync_export_failure_does_not_advance_policy_version(
      self,
  ):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()
    self.assertEqual(engine.policy_version, 0)

    sync_req = datatypes.WeightSyncRequest(policy_version=5, request_id="r5")
    with mock.patch.object(
        engine, "export_hf_state_dict", side_effect=RuntimeError("export boom")
    ):
      with self.assertRaisesRegex(RuntimeError, "export boom"):
        worker.prepare_weight_sync(sync_req)

    self.assertEqual(engine.policy_version, 0)
    self.assertEqual(worker.heartbeat().policy_version, 0)

  def test_save_checkpoint_respects_max_to_keep_and_prunes_stale_uncommitted(
      self,
  ):
    ckpt_dir = self.create_tempdir("ckpt_max_to_keep").full_path
    base = epath.Path(ckpt_dir)
    # Pre-seed a stale uncommitted directory with a dotfile (.metadata only).
    stale = base / "step_1"
    stale.mkdir(parents=True, exist_ok=True)
    (stale / ".metadata").write_text("partial")

    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        checkpoint_dir=ckpt_dir,
        save_interval_steps=2,
        max_to_keep=2,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()

    for step_num in (2, 4, 6):
      resp = worker.save_checkpoint(
          metadata={"step": step_num, "policy_version": step_num // 2},
          step=step_num,
      )
      self.assertTrue(resp.metadata["checkpoint_saved"])

    remaining = sorted(d.name for d in base.iterdir() if d.is_dir())
    self.assertEqual(remaining, ["step_4", "step_6"])

  def test_export_hf_state_dict_offloads_to_cpu_before_to_hf(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        mixed_precision_param="bfloat16",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    observed: list[tuple[bool, bool, str, torch.dtype]] = []
    adapter = engine.state_dict_adapter
    assert adapter is not None
    orig_to_hf = adapter.to_hf

    def _spy_to_hf(sd):
      for v in sd.values():
        observed.append((
            isinstance(v, DTensor),
            isinstance(v, torch.Tensor),
            v.device.type,
            v.dtype,
        ))
      return orig_to_hf(sd)

    with mock.patch.object(adapter, "to_hf", side_effect=_spy_to_hf):
      hf_sd = engine.export_hf_state_dict(is_rank_zero=True)

    self.assertNotEmpty(observed)
    for is_dtensor, is_plain_tensor, dev_type, dtype in observed:
      self.assertFalse(is_dtensor)
      self.assertTrue(is_plain_tensor)
      self.assertEqual(dev_type, "cpu")
      self.assertEqual(dtype, torch.bfloat16)
    self.assertNotEmpty(hf_sd)
    self.assertTrue(
        all(t.dtype == torch.bfloat16 for t in hf_sd.values())
    )
    self.assertTrue(
        all(p.dtype == torch.float32 for p in engine.model.parameters())
    )

  def test_fp32_master_weights_from_bf16_safetensors_and_lr_1e6_update(self):
    bf16_dir = self.create_tempdir("bf16_hf").full_path
    src_sd = safetensors.torch.load_file(
        os.path.join(self.temp_dir, "model.safetensors"), device="cpu"
    )
    bf16_sd = {k: v.to(torch.bfloat16) for k, v in src_sd.items()}
    safetensors.torch.save_file(
        bf16_sd, os.path.join(bf16_dir, "model.safetensors")
    )
    with open(os.path.join(bf16_dir, "config.json"), "w") as f:
      json.dump({"tie_word_embeddings": False}, f)

    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=bf16_dir,
        dtype="float32",
        mixed_precision_param="bfloat16",
        learning_rate=1e-6,
        warmup_steps=0,
        decay_steps=10,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    self.assertTrue(
        all(p.dtype == torch.float32 for p in engine.model.parameters())
    )

    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)

    observed_act_dtypes: list[torch.dtype] = []
    hook = engine.model.layers["0"].feed_forward.w1.register_forward_hook(
        lambda _mod, _inp, out: observed_act_dtypes.append(out.dtype)
    )
    before_params = {
        name: p.detach().clone()
        for name, p in engine.model.named_parameters()
    }
    try:
      engine.fwd_bwd(_make_payload(1))
      _, metrics = engine.update()
    finally:
      hook.remove()

    self.assertIn("loss", metrics)
    self.assertEqual(observed_act_dtypes, [torch.bfloat16])

    self.assertEqual(_adam_moment_dtypes(engine.optimizer), ["torch.float32"])

    total_elems = 0
    changed_elems = 0
    for name, param in engine.model.named_parameters():
      diff = (param.detach() != before_params[name])
      total_elems += diff.numel()
      changed_elems += int(diff.sum().item())
    self.assertGreater(changed_elems / total_elems, 0.9)

    exported = engine.export_hf_state_dict(is_rank_zero=True)
    self.assertNotEmpty(exported)
    self.assertTrue(all(t.dtype == torch.bfloat16 for t in exported.values()))
    self.assertTrue(
        all(p.dtype == torch.float32 for p in engine.model.parameters())
    )

  def test_stop_abort_does_not_block_on_step_lock(self):
    mock_engine = mock.MagicMock()
    mock_engine.policy_version = 0
    worker = TorchTitanTrainerWorker(engine=mock_engine)
    worker.start()

    worker._step_lock.acquire()
    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: worker.stop(abort=True))
        resp = future.result(timeout=2.0)
    finally:
      worker._step_lock.release()

    self.assertTrue(resp.metadata["stopped"])
    self.assertTrue(resp.metadata["abort"])
    self.assertEqual(worker.state, datatypes.WorkerState.STOPPED)
    self.assertTrue(worker._spmd_broken)
    with self.assertRaisesRegex(RuntimeError, "broken or desynchronized"):
      worker.update()

  def test_spmd_broken_poisons_subsequent_calls_after_broadcast_failure(self):
    mock_engine = mock.MagicMock()
    mock_engine.policy_version = 1
    mock_engine.update.side_effect = RuntimeError("boom after broadcast")
    mock_pg = mock.MagicMock()
    worker = TorchTitanTrainerWorker(engine=mock_engine, control_pg=mock_pg)
    worker.start()

    with mock.patch.object(dist, "broadcast_object_list") as mock_bcast:
      with self.assertRaisesRegex(RuntimeError, "boom after broadcast"):
        worker.update()
      self.assertEqual(mock_bcast.call_count, 1)

      self.assertTrue(worker._spmd_broken)
      self.assertEqual(worker.state, datatypes.WorkerState.ERROR)

      mock_engine.update.side_effect = None
      for fn in (
          worker.update,
          worker.compile,
          worker.prepare_weight_sync,
      ):
        with self.assertRaisesRegex(RuntimeError, "broken or desynchronized"):
          fn()
      # Broadcast was never re-invoked after SPMD group was poisoned.
      self.assertEqual(mock_bcast.call_count, 1)

  def test_save_checkpoint_clears_stale_files_in_existing_step_dir(self):
    ckpt_dir = self.create_tempdir("ckpt_stale_overwrite").full_path
    step_dir = epath.Path(ckpt_dir) / "step_2"
    step_dir.mkdir(parents=True, exist_ok=True)
    stale_shard = step_dir / "stale_shard.distcp"
    stale_shard.write_text("stale")
    (step_dir / "metadata.json").write_text('{"step": 2}')

    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
        checkpoint_dir=ckpt_dir,
        save_interval_steps=2,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    algo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="token-mean",
        epsilon=0.2,
    )
    engine.with_loss_fn("grpo", algo_config=algo_cfg, pad_id=0, eos_id=2)
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()

    resp = worker.save_checkpoint(metadata={"step": 2}, step=2, force=True)
    self.assertTrue(resp.metadata["checkpoint_saved"])
    self.assertFalse(stale_shard.exists())
    self.assertTrue((step_dir / "metadata.json").exists())
    self.assertTrue((step_dir / ".metadata").exists())

  def test_rank0_rejects_empty_prompt_before_broadcast_without_poisoning(self):
    mock_engine = mock.MagicMock()
    mock_engine.policy_version = 1
    mock_pg = mock.MagicMock()
    mock_dp_pg = mock.MagicMock()
    worker = TorchTitanTrainerWorker(
        engine=mock_engine, control_pg=mock_pg, dp_pg=mock_dp_pg
    )
    worker.start()

    bad_payload = datatypes.RLTrainerPayload(
        prompt_ids=np.array([[10, 11], [0, 0]], dtype=np.int32),
        prompt_mask=np.array([[1, 1], [0, 0]], dtype=np.int32),
        completion_ids=np.array([[20, 2], [21, 2]], dtype=np.int32),
        completion_mask=np.array([[1, 1], [1, 1]], dtype=np.int32),
        advantages=np.array([1.0, -1.0], dtype=np.float32),
    )
    bad_logps_req = datatypes.LogprobsRequest(
        request_id="lp_empty",
        prompt_tokens=np.array([[10, 11], [0, 0]], dtype=np.int32),
        completion_tokens=np.array([[20, 2], [21, 2]], dtype=np.int32),
        temperature=1.0,
        pad_id=0,
        eos_id=2,
    )

    with (
        mock.patch.object(dist, "get_world_size", return_value=2),
        mock.patch.object(dist, "broadcast_object_list") as mock_bcast,
    ):
      with self.assertRaisesRegex(ValueError, "prompt_len <= 0"):
        worker.fwd_bwd(datatypes.TrainRequest(payload=bad_payload))
      with self.assertRaisesRegex(ValueError, "prompt_len <= 0"):
        worker.per_token_logps(bad_logps_req)
      mock_bcast.assert_not_called()
    self.assertFalse(worker._spmd_broken)
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_release_weight_sync_none_handles_exporter_and_transitions_ready(
      self,
  ):
    mock_engine = mock.MagicMock()
    mock_engine.policy_version = 1
    mock_exporter = mock.MagicMock()
    worker = TorchTitanTrainerWorker(
        engine=mock_engine, weight_sync_exporter=mock_exporter
    )
    worker.state = datatypes.WorkerState.SYNCING

    worker.release_weight_sync(None)

    mock_exporter.release_weight_sync.assert_not_called()
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_prepare_weight_sync_failure_transitions_to_error(self):
    mock_engine = mock.MagicMock()
    mock_engine.policy_version = 1
    mock_engine.export_hf_state_dict.side_effect = RuntimeError("export boom")
    worker = TorchTitanTrainerWorker(engine=mock_engine)
    worker.start()

    with self.assertRaisesRegex(RuntimeError, "export boom"):
      worker.prepare_weight_sync(
          datatypes.WeightSyncRequest(policy_version=2, request_id="r2")
      )
    self.assertEqual(worker.state, datatypes.WorkerState.ERROR)

  def test_custom_policy_loss_closure_multiprocess_fsdp_parity(self):
    world_size = 2
    master_port = portpicker.pick_unused_port()
    with mp.Manager() as manager:  # pyrefly: ignore[missing-attribute]
      managed_result = manager.dict()
      mp.spawn(
          _run_custom_loss_worker_process,
          args=(world_size, master_port, self.temp_dir, managed_result),
          nprocs=world_size,
          join=True,
      )
      shared_result = dict(managed_result)

    self.assertTrue(shared_result["stdlib_pickle_failed"])
    self.assertTrue(shared_result["loss_fn_configured"])
    self.assertEqual(shared_result["step"], 1)
    self.assertEqual(shared_result["ref_step"], 1)

    metrics = shared_result["metrics"]
    ref_metrics = shared_result["ref_metrics"]
    self.assertIn("custom_metric", metrics)
    self.assertAlmostEqual(metrics["loss"], ref_metrics["loss"], places=4)
    self.assertAlmostEqual(
        metrics["grad_norm"], ref_metrics["grad_norm"], places=4
    )
    self.assertAlmostEqual(
        metrics["custom_metric"], ref_metrics["custom_metric"], places=5
    )

  def test_custom_loss_dry_run_rejects_on_rank0_before_broadcast(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    mock_pg = mock.create_autospec(dist.ProcessGroup, instance=True)
    worker = TorchTitanTrainerWorker(engine=engine, control_pg=mock_pg)
    worker.start()

    algo_cfg = algorithm_config.AlgorithmConfig(
        beta=0.0, use_rollout_logps=False
    )

    def _bad_return_type(
        inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
        algo_config_arg: algorithm_config.AlgorithmConfig,
    ) -> Any:
      del inputs, algo_config_arg
      return "not_a_policy_loss_output"

    def _wrong_framework_tensor_loss(
        inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
        algo_config_arg: algorithm_config.AlgorithmConfig,
    ) -> Any:
      del inputs, algo_config_arg
      return algorithm_config.PolicyLossOutput(
          loss_sum=np.array(1.0, dtype=np.float32),
          denominator=torch.tensor(1.0),
      )

    def _detached_loss(
        inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
        algo_config_arg: algorithm_config.AlgorithmConfig,
    ) -> algorithm_config.PolicyLossOutput[torch.Tensor]:
      del algo_config_arg
      detached = (-torch.sum(inputs.per_token_logps)).detach()
      return algorithm_config.PolicyLossOutput(
          loss_sum=detached,
          denominator=torch.tensor(1.0),
      )

    def _bad_metric_loss(
        inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
        algo_config_arg: algorithm_config.AlgorithmConfig,
    ) -> Any:
      del algo_config_arg
      return algorithm_config.PolicyLossOutput(
          loss_sum=-torch.sum(inputs.per_token_logps),
          denominator=torch.tensor(1.0),
          metrics={"bad": 1.0},
      )

    with mock.patch.object(
        dist, "broadcast_object_list", autospec=True
    ) as mock_bcast:
      with self.assertRaisesRegex(TypeError, "PolicyLossOutput"):
        worker.with_loss_fn(_bad_return_type, algo_config=algo_cfg)
      with self.assertRaisesRegex(TypeError, "torch.Tensor"):
        worker.with_loss_fn(_wrong_framework_tensor_loss, algo_config=algo_cfg)
      with self.assertRaisesRegex(ValueError, "requires_grad"):
        worker.with_loss_fn(_detached_loss, algo_config=algo_cfg)
      with self.assertRaisesRegex(TypeError, "torch.Tensor"):
        worker.with_loss_fn(_bad_metric_loss, algo_config=algo_cfg)
      mock_bcast.assert_not_called()

    self.assertFalse(worker._spmd_broken)
    self.assertEqual(worker.state, datatypes.WorkerState.READY)

  def test_set_target_state_returns_configured(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    worker = TorchTitanTrainerWorker(engine=engine)
    worker.start()

    resp = worker.set_target_state(
        {"w": datatypes.TargetParamSpec(shape=(4, 4), dtype="bfloat16")}
    )
    self.assertTrue(resp.metadata["target_state_configured"])

  def test_orchestrator_configure_worker_drives_torchtitan_worker(self):
    cfg = TorchTitanEngineConfig(
        model_name="qwen3",
        flavor="debugmodel",
        hf_assets_path=self.temp_dir,
        dtype="float32",
        learning_rate=1e-3,
        device_type="cpu",
        load_hf_weights=True,
    )
    engine = TorchTitanEngine(
        config=cfg, dp_mesh=None, device=torch.device("cpu")
    )
    worker = TorchTitanTrainerWorker(engine=engine, worker_id="trainer_0")
    handle = remote_execution.InProcessActorHandle(
        remote_execution.InProcessRemoteExecutionServer(worker)
    )
    init_resp = handle.submit("initialize")
    info = worker.info()
    for forbidden_key in (
        "accepts_callable_loss",
        "requires_target_state",
        "needs_aot_compile",
    ):
      self.assertNotIn(forbidden_key, init_resp.metadata)
      self.assertNotIn(forbidden_key, info.resources)

    registry = worker_registry.WorkerRegistry()
    registry.register_handle(
        info.worker_id,
        sorted(info.roles),
        handle,
        instance_id=info.instance_id,
        resources=dict(info.resources),
    )
    worker_registry_testing.set_status(
        registry, info.worker_id, worker_registry.AdmissionStatus.ACTIVE
    )

    rl_engine = distributed_rl_engine.DistributedRLEngine(registry=registry)
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=2,
        max_prompt_length=8,
        max_response_length=8,
        pad_id=0,
        num_generations=1,
        mini_batch_size=2,
    )
    grpo_cfg = algorithm_config.GRPOConfig(
        loss_algo="grpo",
        loss_agg_mode="sequence-mean-token-mean",
        epsilon=0.2,
    )
    rl_engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=grpo_cfg,
        assembler=assembler,
        pad_id=0,
        eos_id=2,
        can_fuse_agreement_in_loss=False,
    )
    self.assertIs(engine.algo_config, grpo_cfg)

    seen_agreement: list[bool] = []

    def _custom_loss(
        inputs: algorithm_config.PolicyLossInputs[torch.Tensor],
        algo_cfg: algorithm_config.AlgorithmConfig,
    ) -> algorithm_config.PolicyLossOutput[torch.Tensor]:
      del algo_cfg
      seen_agreement.append(inputs.sampler_agreement_applied)
      mask = inputs.completion_mask.to(torch.float32)
      loss_sum = -torch.sum(inputs.per_token_logps * mask)
      denom = torch.sum(mask).clamp(min=1.0)
      return algorithm_config.PolicyLossOutput(
          loss_sum=loss_sum,
          denominator=denom,
          metrics={"orch_custom_metric": torch.sum(mask).detach()},
      )

    custom_cfg = algorithm_config.AlgorithmConfig(
        beta=0.0, use_rollout_logps=False
    )
    rl_engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo_config=custom_cfg,
        loss_fn=_custom_loss,
        assembler=assembler,
        pad_id=0,
        eos_id=2,
        can_fuse_agreement_in_loss=False,
    )
    self.assertIs(engine.algo_config, custom_cfg)
    base_payload = _make_payload(1)
    agreed_payload = dataclasses.replace(
        base_payload, sampler_agreement_applied=True
    )
    handle.submit("fwd_bwd", datatypes.TrainRequest(payload=agreed_payload))
    step, metrics = handle.submit("update")
    self.assertEqual(step, 1)
    self.assertIn("orch_custom_metric", metrics)
    self.assertEqual(seen_agreement, [False, True])


if __name__ == "__main__":
  absltest.main()
