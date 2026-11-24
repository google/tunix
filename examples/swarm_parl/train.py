# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Option 5 PARL (Kimi K2.5 §3) TPU trainer, ablation runner & Pareto evaluator."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import time
from typing import Any

from absl import flags
from flax import nnx
import jax
import numpy as np
import optax

from examples.swarm_parl import eval_utils
from examples.swarm_parl import retrieval as parl_ret
from examples.swarm_parl import tasks as parl_tasks
from examples.swarm_parl import workflow as parl_wf
from tunix.common import configs
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import CollectionMode
from tunix.dist_rl.swarm import CriticalStepsBudget
from tunix.dist_rl.swarm import LearnerSpec
from tunix.dist_rl.swarm import ManifestStore
from tunix.dist_rl.swarm import ObjectiveBatch
from tunix.dist_rl.swarm import PolicySpec
from tunix.dist_rl.swarm import Recipe
from tunix.dist_rl.swarm import RoleSamplerAdapter
from tunix.dist_rl.swarm import ServingFidelity
from tunix.dist_rl.swarm import ServingFidelityMode
from tunix.dist_rl.swarm import SnapshotBundle
from tunix.dist_rl.swarm import SnapshotRef
from tunix.dist_rl.swarm import StepContext
from tunix.dist_rl.swarm import group_grpo
from tunix.dist_rl.swarm import parl_grpo
from tunix.dist_rl.swarm import verify_serving_fidelity
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algo_core
from tunix.rl import algorithm_config
from tunix.rl import function_registry


@dataclasses.dataclass(frozen=True)
class ParlTpuPartition:
  train_mesh: jax.sharding.Mesh
  retrieval_devices: list[Any]
  director_mesh: jax.sharding.Mesh
  worker_mesh: jax.sharding.Mesh


def _sort_z_slab(devs: list[Any]) -> list[Any]:
  if devs and hasattr(devs[0], "coords"):
    return sorted(
        devs,
        key=lambda d: (tuple(reversed(tuple(d.coords))), getattr(d, "core_on_chip", 0)),
    )
  return devs


def partition_parl_devices(
    devices: list[Any] | None = None,
    *,
    tp_serve: int = 1,
) -> ParlTpuPartition:
  """Splits TPU chips along contiguous z-slabs into train/ret/director/worker."""
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    pathwaysutils.initialize()
  devs = _sort_z_slab(list(devices) if devices is not None else list(jax.devices()))
  n = len(devs)
  if n >= 128 and tp_serve >= 2:
    n_train, n_ret, n_dir = 32, 16, 32
  elif n >= 128:
    n_train, n_ret, n_dir = 16, 16, 32
  elif n >= 64:
    n_train, n_ret, n_dir = 16, 16, 16
  elif n >= 16:
    n_train, n_ret, n_dir = 4, 4, 4
  else:
    raise ValueError(f"PARL TPU layout requires >= 16 chips, got {n}")
  tp_tr = 4 if n_train >= 4 else 1
  train_mesh = jax.sharding.Mesh(
      np.asarray(devs[:n_train]).reshape(n_train // tp_tr, tp_tr), ("fsdp", "tp")
  )
  ret_devs = devs[n_train : n_train + n_ret]
  d0 = n_train + n_ret
  n_wrk = n - d0 - n_dir
  tp_s = max(1, int(tp_serve))
  dir_mesh = jax.sharding.Mesh(
      np.asarray(devs[d0 : d0 + n_dir]).reshape(n_dir // tp_s, tp_s), ("fsdp", "tp")
  )
  wrk_mesh = jax.sharding.Mesh(
      np.asarray(devs[d0 + n_dir :]).reshape(n_wrk // tp_s, tp_s), ("fsdp", "tp")
  )
  return ParlTpuPartition(train_mesh, ret_devs, dir_mesh, wrk_mesh)


class _VllmParlSamplerWrapper:
  """Adapts InprocessVllmSampler for Option 5 RoleSamplerAdapter & attestation."""

  def __init__(self, engine: Any, tokenizer: Any, eos_ids: list[int], digest: str) -> None:
    self.engine = engine
    self.tokenizer = tokenizer
    self.eos_set = frozenset(int(x) for x in eos_ids)
    self.digest = digest

  async def sample(
      self, request: sampler_lib.SamplingRequest
  ) -> sampler_lib.SamplingResponse:
    params = request.sampling_params
    if params is not None and (params.top_k is None or params.seed is not None):
      request = dataclasses.replace(
          request,
          sampling_params=dataclasses.replace(
              params, top_k=0 if params.top_k is None else params.top_k, seed=None
          ),
      )
    resp = await self.engine.sample(request)
    meta = dict(resp.metadata or {})
    meta["served_digest"] = self.digest
    toks = resp.token_ids
    if len(toks) > 0 and int(toks[-1]) in self.eos_set:
      text = self.tokenizer.decode(toks[:-1].tolist(), skip_special_tokens=True)
    else:
      text = resp.text
    return dataclasses.replace(resp, text=text, metadata=meta)


class Option5PeftDirectorTrainer:
  """Stages Option 5 `ObjectiveBatch` into Tunix `TrainerWorker` (+ optional KL ref)."""

  def __init__(
      self,
      *,
      worker: trainer_worker.TrainerWorker,
      ref_worker: trainer_worker.TrainerWorker | None,
      pad_id: int,
      eos_id: int,
      max_prompt_length: int,
      max_response_length: int,
      micro_batch_size: int,
  ) -> None:
    self.worker = worker
    self.ref_worker = ref_worker
    self.pad_id = int(pad_id)
    self.eos_id = int(eos_id)
    self.max_prompt_length = int(max_prompt_length)
    self.max_response_length = int(max_response_length)
    self.micro_batch_size = int(micro_batch_size)
    self._has_staged = False

  def _examples_to_payloads(
      self, examples: tuple[dict[str, Any], ...]
  ) -> list[datatypes.RLTrainerPayload]:
    payloads: list[datatypes.RLTrainerPayload] = []
    for ex in examples:
      p_ids = np.asarray(ex["prompt_ids"][-self.max_prompt_length :], dtype=np.int32)
      c_ids = np.asarray(ex["completion_ids"][: self.max_response_length], dtype=np.int32)
      c_lps = np.asarray(ex["rollout_logprobs"][: len(c_ids)], dtype=np.float32)
      l_mask = np.asarray(ex["loss_mask"][: len(c_ids)], dtype=np.float32)
      a_mask = np.asarray(ex["attention_mask"][: len(c_ids)], dtype=np.float32)
      if c_ids.size == 0 or float(np.sum(l_mask)) <= 0.0:
        continue
      mask_2d = l_mask[None, :]
      payloads.append(
          datatypes.RLTrainerPayload(
              prompt_ids=p_ids[None, :],
              prompt_mask=np.ones((1, len(p_ids)), dtype=np.float32),
              completion_ids=c_ids[None, :],
              completion_mask=mask_2d,
              completion_attention_mask=a_mask[None, :],
              advantages=mask_2d * float(ex["advantage"]),
              old_per_token_logps=c_lps[None, :],
              metadata={"episode_id": ex["episode_id"], "role": ex["role"]},
          )
      )
    return payloads

  def _attach_ref_logps(
      self, payload: datatypes.RLTrainerPayload
  ) -> datatypes.RLTrainerPayload:
    if self.ref_worker is None:
      return payload
    tok_mask = (
        None
        if payload.completion_attention_mask is None
        else np.concatenate(
            [payload.prompt_mask, payload.completion_attention_mask], axis=1
        )
    )
    ref_resp = self.ref_worker.per_token_logps(
        datatypes.LogprobsRequest(
            request_id="reference_logps",
            prompt_tokens=payload.prompt_ids,
            completion_tokens=payload.completion_ids,
            token_mask=tok_mask,
            pad_id=self.pad_id,
            eos_id=self.eos_id,
            temperature=1.0,
        )
    )
    return batch_assembly.with_ref_per_token_logps(payload, ref_resp)

  def stage_update(self, batch: ObjectiveBatch) -> dict[str, float]:
    payloads = self._examples_to_payloads(batch.examples)
    if not payloads:
      self._has_staged = False
      return {"train/skipped_empty": 1.0}
    assembler = batch_assembly.PaddedBatchAssembler(
        batch_size=self.micro_batch_size,
        max_prompt_length=self.max_prompt_length,
        max_response_length=self.max_response_length,
        pad_id=self.pad_id,
        num_generations=len(payloads),
        mini_batch_size=1,
    )
    mbs = list(assembler.feed(payloads))
    if not mbs:
      self._has_staged = False
      return {"train/skipped_empty": 1.0}
    total_toks = sum(int(np.sum(p.completion_mask)) for p in payloads)
    for mb in mbs:
      payload = self._attach_ref_logps(mb.payload)
      self.worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
    self._has_staged = True
    return {
        "train/n_microbatches": float(len(mbs)),
        "train/active_tokens": float(total_toks),
    }

  def commit_staged(self, version: int) -> None:
    del version
    if self._has_staged:
      self.worker.update()
      self._has_staged = False


def _build_vllm_engine(
    *,
    server_id: str,
    actor: Any,
    tokenizer: Any,
    mesh: jax.sharding.Mesh,
    model_path: str,
    eos_ids: list[int],
    max_prompt_length: int,
    max_response_length: int,
    return_logprobs: bool,
    hbm_utilization: float,
) -> Any:
  from tunix.dist_rl.rollout import inprocess_vllm_sampler  # pylint: disable=g-import-not-at-top
  from tunix.generate import mappings  # pylint: disable=g-import-not-at-top

  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  dp = int(mesh.shape["fsdp"])
  tp = int(mesh.shape["tp"])
  kv_len = max_prompt_length + max_response_length
  rcfg = configs.RolloutConfig(
      max_prompt_length=max_prompt_length,
      max_tokens_to_generate=max_response_length,
      kv_cache_size=kv_len,
      temperature=1.0,
      top_p=1.0,
      top_k=0,
      eos_tokens=eos_ids,
      return_logprobs=return_logprobs,
      tensor_parallel_size=tp,
      data_parallel_size=dp,
      rollout_mapping_config=mappings.MappingConfig.from_model(actor, backend="vllm_jax"),
      rollout_vllm_model_version=model_path,
      rollout_vllm_server_mode=True,
      rollout_vllm_init_with_random_weights=True,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_hbm_utilization=hbm_utilization,
      rollout_vllm_reshard_chunk_size=16,
      rollout_vllm_async_scheduling=False,
      rollout_vllm_max_num_seqs=32,
      rollout_vllm_max_num_batched_tokens=max(8192, kv_len),
      rollout_vllm_kwargs={
          "enable_prefix_caching": False,
          "dtype": "bfloat16",
          "tokenizer": model_path,
          "seed": 42,
      },
  )
  with mesh:
    eng = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id=server_id,
        tokenizer=tokenizer,
        config=rcfg,
        mesh=mesh,
        max_concurrency=32 * dp,
    )
    eng.initialize()
    return eng


def _sync_actor_to_vllm(actor: Any, engine: Any, mesh: jax.sharding.Mesh) -> None:
  params = nnx.state(actor, nnx.Param)
  with mesh:
    engine.update_params(params)


async def _run_parl_fidelity_audit(
    worker: trainer_worker.TrainerWorker,
    dir_wrapper: _VllmParlSamplerWrapper,
    encode: Any,
    pad_id: int,
    eos_id: int,
    fidelity: ServingFidelity,
) -> None:
  prompt_ids = np.asarray(
      encode(
          parl_wf.format_chat_prompt(
              parl_wf.DIRECTOR_SYSTEM_PROMPT,
              "Question: What is the capital of France?",
          )
      ),
      dtype=np.int32,
  )
  req = sampler_lib.SamplingRequest(
      request_id="fidelity-0",
      prompt=prompt_ids,
      sampling_params=sampler_lib.SamplingParams(
          max_tokens=16, temperature=0.0, top_p=1.0, top_k=0, return_logprobs=True
      ),
  )
  resp = await dir_wrapper.sample(req)
  comp = np.asarray(resp.token_ids[:8], dtype=np.int32)
  samp_lps = np.asarray(resp.logprobs[: len(comp)], dtype=np.float64)
  l_resp = worker.per_token_logps(
      datatypes.LogprobsRequest(
          prompt_tokens=prompt_ids[None, :],
          completion_tokens=comp[None, :],
          token_mask=np.ones((1, len(prompt_ids) + len(comp)), dtype=np.int32),
          pad_id=pad_id,
          eos_id=eos_id,
          temperature=1.0,
      )
  )
  learner_lps = np.asarray(l_resp.per_token_logps[0, : len(comp)], dtype=np.float64)
  rep = verify_serving_fidelity(
      learner_top1_ids=comp.tolist(),
      sampler_top1_ids=comp.tolist(),
      learner_logprobs=learner_lps.tolist(),
      sampler_logprobs=samp_lps.tolist(),
      fidelity=fidelity,
      raise_on_fail=True,
  )
  print(
      f"[parl-L3] ServingFidelity PASS mode={rep.mode.value} "
      f"max_lp_diff={rep.max_logprob_abs_diff:.4e} mean_lp_diff={rep.mean_logprob_abs_diff:.4e}",
      flush=True,
  )


def _init_wandb(args: argparse.Namespace) -> Any:
  if not os.environ.get("WANDB_API_KEY"):
    return None
  try:
    import wandb  # pylint: disable=g-import-not-at-top

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "tunix-swarm"),
        entity=os.environ.get("WANDB_ENTITY", "linchai-google"),
        name=args.wandb_name,
        config=vars(args),
        resume="allow",
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"[parl] wandb init skipped: {exc}", flush=True)
    return None


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--mode", choices=("parl", "serial_rl"), default="parl")
  p.add_argument("--model-id", default="Qwen/Qwen3-32B")
  p.add_argument("--model-path", type=Path, required=True)
  p.add_argument("--data-root", type=Path, required=True)
  p.add_argument("--upstream-root", type=Path, required=True)
  p.add_argument("--browsecomp-dir", type=Path, default=None)
  p.add_argument("--run-root", type=Path, required=True)
  p.add_argument("--tp-serve", type=int, default=0)
  p.add_argument("--num-steps", type=int, default=200)
  p.add_argument("--tasks-per-step", type=int, default=16)
  p.add_argument("--attempts-per-task", type=int, default=8)
  p.add_argument("--max-critical-steps", type=int, default=16)
  p.add_argument("--max-waves", type=int, default=6)
  p.add_argument("--max-wave-width", type=int, default=12)
  p.add_argument("--max-worker-steps", type=int, default=4)
  p.add_argument("--dual-mode", action="store_true")
  p.add_argument("--enable-thinking", action="store_true")
  p.add_argument("--wide-min-width", type=int, default=6)
  p.add_argument("--wide-max-width", type=int, default=10)
  p.add_argument("--n-wide-train", type=int, default=1200)
  p.add_argument("--n-multihop-train", type=int, default=600)
  p.add_argument("--n-browsecomp-train", type=int, default=630)
  p.add_argument("--n-wide-eval", type=int, default=96)
  p.add_argument("--n-multihop-eval", type=int, default=64)
  p.add_argument("--n-browsecomp-eval", type=int, default=128)
  p.add_argument("--anneal-steps", type=int, default=60)
  p.add_argument("--lambda-parallel", type=float, default=0.20)
  p.add_argument("--lambda-finish", type=float, default=0.15)
  p.add_argument("--kl-beta", type=float, default=0.01)
  p.add_argument("--epsilon-high", type=float, default=0.28)
  p.add_argument("--learning-rate", type=float, default=1e-6)
  p.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
  p.add_argument("--micro-batch-size", type=int, default=2)
  p.add_argument("--max-prompt-length", type=int, default=6144)
  p.add_argument("--max-response-length", type=int, default=1024)
  p.add_argument("--eval-every", type=int, default=20)
  p.add_argument("--eval-temperature", type=float, default=0.0)
  p.add_argument("--pareto-budgets", default="4,8,12,16,20")
  p.add_argument("--hbm-utilization", type=float, default=0.60)
  p.add_argument("--unlink-staged-weights", action="store_true")
  p.add_argument("--wandb-name", default="haoyugao-parl-v3-qwen3-32b")
  return p.parse_args()


def _resolve_tp_serve(args: argparse.Namespace) -> int:
  if args.tp_serve > 0:
    return int(args.tp_serve)
  low = args.model_id.lower()
  return 2 if ("32b" in low or "30b" in low or "35b" in low) else 1


def _build_ref_worker(
    args: argparse.Namespace,
    mesh: jax.sharding.Mesh,
) -> trainer_worker.TrainerWorker | None:
  if args.kl_beta <= 0.0:
    return None
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  with mesh:
    ref_actor, _ = automodel.AutoModel.from_pretrained(
        args.model_id,
        mesh,
        model_path=str(args.model_path),
        remat_config="NONE",
        dtype="bfloat16",
        load_dtype="bfloat16",
        use_flash_attention=False,
    )
    ref_trainer = peft_trainer_v2.PeftTrainer(
        ref_actor,
        optax.set_to_zero(),
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=args.num_steps + 1,
            max_steps=1,
            gradient_accumulation_steps=1,
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="inprocess_vllm",
    )
    ref_worker = trainer_worker.TrainerWorker(
        lambda: ref_trainer, logps_chunk_size=0, execution_context=mesh
    )
    ref_worker.initialize()
    return ref_worker


def _unlink_safetensors(model_path: Path) -> None:
  removed = 0
  for sf in model_path.glob("*.safetensors"):
    sf.unlink(missing_ok=True)
    removed += 1
  if removed:
    print(f"[parl-boot] freed {removed} staged safetensors shards from tmpfs", flush=True)


def _boot_parl_stack(args: argparse.Namespace) -> dict[str, Any]:
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["swarm_parl"])
  tp_serve = _resolve_tp_serve(args)
  part = partition_parl_devices(tp_serve=tp_serve)
  print(
      f"[parl-boot] mode={args.mode} model={args.model_id} tp_serve={tp_serve} "
      f"chips={len(jax.devices())} train={part.train_mesh.size} "
      f"ret={len(part.retrieval_devices)} dir={part.director_mesh.size} "
      f"wrk={part.worker_mesh.size} kl_beta={args.kl_beta}",
      flush=True,
  )
  ret_server, retriever, bc_ret = parl_ret.start_multi_corpus_retrieval_server(
      args.data_root / "wiki18",
      args.data_root / "models" / "e5-base-v2",
      browsecomp_dir=args.browsecomp_dir,
      port=18610,
      tpu_devices=part.retrieval_devices,
  )
  print(
      f"[parl-boot] Retrieval ready on :18610 wiki18={retriever.total} "
      f"browsecomp={bc_ret.total if bc_ret else 0}",
      flush=True,
  )
  if args.unlink_staged_weights and hasattr(retriever, "release_vectors_file"):
    retriever.release_vectors_file(unlink=True)
    print("[parl-boot] freed staged Wiki-18 vectors (31 GiB) from tmpfs", flush=True)

  tok = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
  eos_ids = [int(tok.eos_token_id)]
  for special in ("<|im_end|>", "<|endoftext|>"):
    tid = tok.convert_tokens_to_ids(special)
    if isinstance(tid, int) and tid >= 0 and tid not in eos_ids:
      eos_ids.append(tid)
  pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else eos_ids[0])

  with part.train_mesh:
    actor, _ = automodel.AutoModel.from_pretrained(
        args.model_id,
        part.train_mesh,
        model_path=str(args.model_path),
        remat_config="BLOCK",
        dtype="float32",
        load_dtype="float32",
        use_flash_attention=False,
    )
    lr_sched: Any = args.learning_rate
    if args.lr_schedule == "cosine":
      lr_sched = optax.warmup_cosine_decay_schedule(
          init_value=0.0,
          peak_value=args.learning_rate,
          warmup_steps=min(10, max(1, args.num_steps // 10)),
          decay_steps=max(args.num_steps, 10),
          end_value=0.1 * args.learning_rate,
      )
    opt = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(lr_sched, b1=0.9, b2=0.999, weight_decay=0.01),
    )
    trainer = peft_trainer_v2.PeftTrainer(
        actor,
        opt,
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=args.num_steps + 1,
            max_steps=args.num_steps,
            gradient_accumulation_steps=1,
            checkpoint_root_directory=str(args.run_root / "checkpoints"),
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="inprocess_vllm",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: trainer, logps_chunk_size=0, execution_context=part.train_mesh
    )
    worker.initialize()
    algo_cfg = algorithm_config.GRPOConfig(
        num_generations=args.attempts_per_task,
        epsilon=0.2,
        epsilon_high=float(args.epsilon_high),
        beta=float(args.kl_beta),
    )
    algo_cfg.temperature = 1.0
    worker.with_loss_fn(
        function_registry.get_policy_loss_fn(algo_cfg.policy_loss_fn), has_aux=True
    )
    worker.with_gen_model_input_fn(
        algo_core.build_gen_model_input_fn(algo_cfg, pad_id=pad_id, eos_id=eos_ids[0])
    )

  ref_worker = _build_ref_worker(args, part.train_mesh)
  dir_engine = _build_vllm_engine(
      server_id="parl_director",
      actor=actor,
      tokenizer=tok,
      mesh=part.director_mesh,
      model_path=str(args.model_path),
      eos_ids=eos_ids,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      return_logprobs=True,
      hbm_utilization=args.hbm_utilization,
  )
  wrk_engine = _build_vllm_engine(
      server_id="parl_worker_frozen",
      actor=actor,
      tokenizer=tok,
      mesh=part.worker_mesh,
      model_path=str(args.model_path),
      eos_ids=eos_ids,
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      return_logprobs=False,
      hbm_utilization=args.hbm_utilization,
  )
  _sync_actor_to_vllm(actor, dir_engine, part.director_mesh)
  _sync_actor_to_vllm(actor, wrk_engine, part.worker_mesh)
  if args.unlink_staged_weights:
    _unlink_safetensors(args.model_path)
  return {
      "actor": actor,
      "worker": worker,
      "ref_worker": ref_worker,
      "tokenizer": tok,
      "eos_ids": eos_ids,
      "pad_id": pad_id,
      "dir_engine": dir_engine,
      "wrk_engine": wrk_engine,
      "dir_mesh": part.director_mesh,
      "ret_server": ret_server,
  }


def _build_role_adapters(
    args: argparse.Namespace,
    dir_wrapper: _VllmParlSamplerWrapper,
    wrk_wrapper: _VllmParlSamplerWrapper,
    encode: Any,
    dir_fidelity: ServingFidelity,
) -> tuple[dict[str, RoleSamplerAdapter], dict[str, RoleSamplerAdapter]]:
  eval_temp = float(args.eval_temperature)
  eval_top_p = 1.0 if eval_temp <= 0.0 else 0.95
  train_adapters = {
      "director": RoleSamplerAdapter(
          sampler=dir_wrapper,
          encode=encode,
          sampling_params=sampler_lib.SamplingParams(
              max_tokens=args.max_response_length, temperature=1.0, top_p=1.0, return_logprobs=True
          ),
          max_prompt_length=args.max_prompt_length,
          fidelity=dir_fidelity,
      ),
      "worker": RoleSamplerAdapter(
          sampler=wrk_wrapper,
          encode=encode,
          sampling_params=sampler_lib.SamplingParams(
              max_tokens=args.max_response_length, temperature=0.3, top_p=0.95, return_logprobs=False
          ),
          max_prompt_length=args.max_prompt_length,
      ),
  }
  eval_adapters = {
      "director": RoleSamplerAdapter(
          sampler=dir_wrapper,
          encode=encode,
          sampling_params=sampler_lib.SamplingParams(
              max_tokens=args.max_response_length, temperature=eval_temp, top_p=eval_top_p, return_logprobs=False
          ),
          max_prompt_length=args.max_prompt_length,
          fidelity=dir_fidelity,
      ),
      "worker": RoleSamplerAdapter(
          sampler=wrk_wrapper,
          encode=encode,
          sampling_params=sampler_lib.SamplingParams(
              max_tokens=args.max_response_length, temperature=eval_temp, top_p=eval_top_p, return_logprobs=False
          ),
          max_prompt_length=args.max_prompt_length,
      ),
  }
  return train_adapters, eval_adapters


def _build_recipe(
    args: argparse.Namespace,
    init_snap: SnapshotRef,
    dir_fidelity: ServingFidelity,
) -> Recipe:
  if args.mode == "serial_rl":
    wf = lambda task, ctx: parl_wf.serial_baseline_workflow(
        task,
        ctx,
        max_steps=args.max_critical_steps,
        role="director",
        enable_thinking=args.enable_thinking,
    )
    step_fn = group_grpo(
        target="director",
        roles=("director",),
        attempts=args.attempts_per_task,
        mode=CollectionMode.STRICT_BUNDLE,
        max_policy_lag=0,
    )
  else:
    wf = lambda task, ctx: parl_wf.parl_episode_workflow(
        task,
        ctx,
        max_waves=args.max_waves,
        max_wave_width=args.max_wave_width,
        max_worker_steps=args.max_worker_steps,
        dual_mode=args.dual_mode,
        enable_thinking=args.enable_thinking,
    )
    step_fn = parl_grpo(
        target="director",
        roles=("director",),
        attempts=args.attempts_per_task,
        lambda_parallel_init=args.lambda_parallel,
        lambda_finish_init=args.lambda_finish,
        anneal_steps=args.anneal_steps,
        mode=CollectionMode.STRICT_BUNDLE,
        max_policy_lag=0,
    )
  return Recipe(
      workflow=wf,
      learners={"director": LearnerSpec.full("director", init_snap, fidelity=dir_fidelity)},
      bindings={
          "director": PolicySpec.follow("director", fidelity=dir_fidelity),
          "worker": PolicySpec.fixed(init_snap),
      },
      grade=parl_tasks.grade_parl_episode,
      train_step=step_fn,
      budget=CriticalStepsBudget(limit=args.max_critical_steps),
  )


def _maybe_update_best_eval(
    best_eval: dict[str, Any],
    eval_metrics: dict[str, float],
    step_id: int,
    run_root: Path,
    tag: str,
) -> dict[str, Any]:
  key = f"eval/{tag}_r_perf"
  if eval_metrics.get(key, -1.0) > best_eval.get(key, -1.0):
    updated = {"step": step_id, **eval_metrics}
    (run_root / "parl_best_eval.json").write_text(
        json.dumps(updated, indent=2) + "\n", encoding="utf-8"
    )
    return updated
  return best_eval


def main() -> int:
  args = parse_args()
  args.run_root.mkdir(parents=True, exist_ok=True)
  stack = _boot_parl_stack(args)
  tok = stack["tokenizer"]
  encode = lambda text: tok.encode(text, add_special_tokens=False)
  train_pub, train_priv, eval_pub, eval_priv = parl_tasks.build_parl_splits(
      args.upstream_root,
      browsecomp_dir=args.browsecomp_dir,
      n_wide_train=args.n_wide_train,
      n_multihop_train=args.n_multihop_train,
      n_browsecomp_train=args.n_browsecomp_train,
      n_wide_eval=args.n_wide_eval,
      n_multihop_eval=args.n_multihop_eval,
      n_browsecomp_eval=args.n_browsecomp_eval,
      wide_min_width=args.wide_min_width,
      wide_max_width=args.wide_max_width,
  )
  print(
      f"[parl-data] train={len(train_pub)} eval={len(eval_pub)} "
      f"wide={args.wide_min_width}..{args.wide_max_width} wave_w={args.max_wave_width} "
      f"dual_mode={args.dual_mode} think={args.enable_thinking}",
      flush=True,
  )
  wiki_env = parl_wf.HttpWikiSearchEnv("http://127.0.0.1:18610/retrieve", top_k=3)
  model_slug = args.model_id.lower().split("/")[-1]
  tag = "serial_rl" if args.mode == "serial_rl" else "parl"

  init_snap = SnapshotRef(model_id=args.model_id, revision=0, digest=f"{model_slug}-v0")
  dir_fidelity = ServingFidelity(
      mode=ServingFidelityMode.FULL_BF16,
      max_top1_mismatch_rate=0.05,
      max_logprob_abs_diff=0.15,
  )
  dir_wrapper = _VllmParlSamplerWrapper(
      stack["dir_engine"], tok, stack["eos_ids"], digest=f"{model_slug}-v0"
  )
  wrk_wrapper = _VllmParlSamplerWrapper(
      stack["wrk_engine"], tok, stack["eos_ids"], digest=f"{model_slug}-v0"
  )
  asyncio.run(
      _run_parl_fidelity_audit(
          stack["worker"], dir_wrapper, encode, stack["pad_id"], stack["eos_ids"][0], dir_fidelity
      )
  )

  recipe = _build_recipe(args, init_snap, dir_fidelity)
  adapters, eval_adapters = _build_role_adapters(
      args, dir_wrapper, wrk_wrapper, encode, dir_fidelity
  )
  dir_trainer = Option5PeftDirectorTrainer(
      worker=stack["worker"],
      ref_worker=stack["ref_worker"],
      pad_id=stack["pad_id"],
      eos_id=stack["eos_ids"][0],
      max_prompt_length=args.max_prompt_length,
      max_response_length=args.max_response_length,
      micro_batch_size=args.micro_batch_size,
  )
  manifest_store = ManifestStore(args.run_root / "option5_manifest")
  live_jsonl = args.run_root / "parl_live_metrics.jsonl"
  wb_run = _init_wandb(args)
  best_eval: dict[str, Any] = {}

  # Evaluate strictly frozen Base Serial ONCE (decoupled from Director updates).
  init_bundle = SnapshotBundle(
      step_id=0,
      role_snapshots={"director": init_snap, "worker": init_snap},
      learner_snapshots={"director": init_snap},
  )
  base_serial = asyncio.run(
      eval_utils.evaluate_frozen_base_serial(
          eval_pub=eval_pub,
          eval_priv=eval_priv,
          recipe=recipe,
          snapshots=init_bundle,
          adapters=eval_adapters,
          wiki_env=wiki_env,
          max_critical_steps=args.max_critical_steps,
          enable_thinking=args.enable_thinking,
      )
  )
  (args.run_root / "base_serial_eval.json").write_text(
      json.dumps(base_serial, indent=2) + "\n", encoding="utf-8"
  )
  print(f"[parl-base-serial] {json.dumps(base_serial)}", flush=True)

  def _publish(versions: dict[str, int]) -> None:
    new_rev = versions["director"]
    _sync_actor_to_vllm(stack["actor"], stack["dir_engine"], stack["dir_mesh"])
    dir_wrapper.digest = f"{model_slug}-v{new_rev}"

  for step_id in range(args.num_steps):
    t0 = time.monotonic()
    cur_snap = SnapshotRef(
        model_id=args.model_id, revision=step_id, digest=dir_wrapper.digest
    )
    bundle = SnapshotBundle(
        step_id=step_id,
        role_snapshots={"director": cur_snap, "worker": init_snap},
        learner_snapshots={"director": cur_snap},
    )
    eval_metrics: dict[str, float] = {}
    if step_id == 0 or (args.eval_every > 0 and step_id % args.eval_every == 0):
      raw_ev = asyncio.run(
          eval_utils.run_policy_eval(
              step_id=step_id,
              eval_pub=eval_pub,
              eval_priv=eval_priv,
              recipe=recipe,
              snapshots=bundle,
              adapters=eval_adapters,
              wiki_env=wiki_env,
              tag=tag,
          )
      )
      eval_metrics = eval_utils.attach_base_deltas(raw_ev, base_serial, tag=tag)
      best_eval = _maybe_update_best_eval(best_eval, eval_metrics, step_id, args.run_root, tag)
    start = (step_id * args.tasks_per_step) % len(train_pub)
    batch_tasks = [
        train_pub[(start + i) % len(train_pub)] for i in range(args.tasks_per_step)
    ]
    step_ctx = StepContext(
        step_id=step_id,
        tasks=batch_tasks,
        private_examples=train_priv,
        recipe=recipe,
        snapshots=bundle,
        adapters=adapters,
        learners={"director": dir_trainer},
        manifest_store=manifest_store,
        cursor=step_id * args.tasks_per_step,
        env_factory=lambda _: wiki_env,
        publish_callback=_publish,
    )
    res = asyncio.run(recipe.train_step(step_ctx))
    asyncio.run(step_ctx.commit(res))
    row = {"step": step_id, "wall_s": time.monotonic() - t0, **res.metrics, **eval_metrics}
    with live_jsonl.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row) + "\n")
    if wb_run is not None:
      wb_run.log(row, step=step_id)
    print(f"[parl-step-{step_id:04d}] {json.dumps(row)}", flush=True)

  final_snap = SnapshotRef(
      model_id=args.model_id, revision=args.num_steps, digest=dir_wrapper.digest
  )
  final_bundle = SnapshotBundle(
      step_id=args.num_steps,
      role_snapshots={"director": final_snap, "worker": init_snap},
      learner_snapshots={"director": final_snap},
  )
  raw_final = asyncio.run(
      eval_utils.run_policy_eval(
          step_id=args.num_steps,
          eval_pub=eval_pub,
          eval_priv=eval_priv,
          recipe=recipe,
          snapshots=final_bundle,
          adapters=eval_adapters,
          wiki_env=wiki_env,
          tag=tag,
      )
  )
  final_eval = eval_utils.attach_base_deltas(raw_final, base_serial, tag=tag)
  best_eval = _maybe_update_best_eval(best_eval, final_eval, args.num_steps, args.run_root, tag)
  (args.run_root / "parl_final_eval.json").write_text(
      json.dumps(final_eval, indent=2) + "\n", encoding="utf-8"
  )
  budgets = tuple(int(x) for x in args.pareto_budgets.split(",") if x.strip())
  if budgets:
    pareto = asyncio.run(
        eval_utils.run_pareto_sweep(
            budgets=budgets,
            step_id=args.num_steps,
            eval_pub=eval_pub,
            eval_priv=eval_priv,
            recipe=recipe,
            snapshots=final_bundle,
            adapters=eval_adapters,
            wiki_env=wiki_env,
            tag=tag,
            enable_thinking=args.enable_thinking,
            include_base_serial=(args.mode == "parl"),
        )
    )
    (args.run_root / "parl_pareto_eval.json").write_text(
        json.dumps(pareto, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[parl-pareto] {json.dumps(pareto)}", flush=True)
  if wb_run is not None:
    wb_run.log(final_eval, step=args.num_steps)
    wb_run.finish()
  print(f"[parl-final] {json.dumps(final_eval)}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
