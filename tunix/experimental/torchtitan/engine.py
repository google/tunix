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

"""TorchTitan Training Engine for distributed policy training."""

from __future__ import annotations

from collections.abc import Mapping
import contextlib
import dataclasses
import math
import os
import tempfile
from typing import Any

from etils import epath
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
from torch.distributed import checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor import DTensor
from torchtitan.components.optimizer.utils import get_flat_optim_state_dict
from torchtitan.components.optimizer.utils import init_optim_state
from torchtitan.components.optimizer.utils import load_flat_optim_state_dict
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.utils import clip_grad_norm_ as titan_clip_grad_norm_
from torchtitan.experiments.tpu.components import unflatten_shim
from torchtitan.models.common.decoder import Decoder
from tunix.dist_rl.common import datatypes
from tunix.experimental.torchtitan import grpo_loss
from tunix.experimental.torchtitan import model_registry
from tunix.rl import algorithm_config


def _validate_custom_torch_policy_loss_fn(
    loss_fn: algorithm_config.PolicyLossFn,
    algo_config: algorithm_config.AlgorithmConfig,
) -> None:
  """Runs a minimal CPU tensor dry-run verifying a custom `PolicyLossFn`."""
  dummy_logps = torch.full(
      (2, 4), -0.5, dtype=torch.float32, device="cpu", requires_grad=True
  )
  dummy_inputs = algorithm_config.PolicyLossInputs(
      per_token_logps=dummy_logps,
      completion_mask=torch.ones((2, 4), dtype=torch.float32, device="cpu"),
      advantages=torch.tensor([1.0, -1.0], dtype=torch.float32, device="cpu"),
      old_per_token_logps=(
          dummy_logps.detach() if algo_config.use_rollout_logps else None
      ),
      ref_per_token_logps=(
          dummy_logps.detach() if algo_config.requires_reference_kl else None
      ),
  )
  out = loss_fn(dummy_inputs, algo_config)
  if not isinstance(out, algorithm_config.PolicyLossOutput):
    raise TypeError(
        "Custom PolicyLossFn must return algorithm_config.PolicyLossOutput,"
        f" got {type(out).__name__}."
    )
  if not isinstance(out.loss_sum, torch.Tensor):
    raise TypeError(
        "PolicyLossOutput.loss_sum must be a torch.Tensor, got"
        f" {type(out.loss_sum).__name__}."
    )
  if not out.loss_sum.requires_grad:
    raise ValueError(
        "PolicyLossOutput.loss_sum must retain the autograd graph"
        " (requires_grad=True) with respect to per_token_logps."
    )
  if not isinstance(out.denominator, torch.Tensor):
    raise TypeError(
        "PolicyLossOutput.denominator must be a torch.Tensor, got"
        f" {type(out.denominator).__name__}."
    )
  if out.token_denom is not None and not isinstance(
      out.token_denom, torch.Tensor
  ):
    raise TypeError(
        "PolicyLossOutput.token_denom must be a torch.Tensor or None, got"
        f" {type(out.token_denom).__name__}."
    )
  for metric_name, metric_val in out.metrics.items():
    if not isinstance(metric_val, torch.Tensor):
      raise TypeError(
          f"PolicyLossOutput.metrics[{metric_name!r}] must be a torch.Tensor,"
          f" got {type(metric_val).__name__}."
      )


def _warmup_cosine_decay_lambda(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    init_lr: float,
    peak_lr: float,
    end_lr: float,
) -> float:
  """Computes learning rate multiplier for linear warmup + cosine decay."""
  if peak_lr <= 0.0:
    return 0.0
  if step < warmup_steps:
    if warmup_steps == 0:
      return 1.0
    alpha = step / float(warmup_steps)
    lr = init_lr + alpha * (peak_lr - init_lr)
    return lr / peak_lr
  else:
    if decay_steps <= warmup_steps:
      return end_lr / peak_lr
    progress = (step - warmup_steps) / float(decay_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    lr = end_lr + (peak_lr - end_lr) * cosine_decay
    return lr / peak_lr


def _parse_torch_dtype(field_name: str, value: str) -> torch.dtype:
  """Parses a TorchTitanEngineConfig dtype string into a `torch.dtype`."""
  if value == "bfloat16":
    return torch.bfloat16
  if value == "float32":
    return torch.float32
  raise ValueError(
      f"TorchTitanEngine supports {field_name} 'bfloat16' or 'float32', got"
      f" {value!r}."
  )


@dataclasses.dataclass(frozen=True, kw_only=True)
class TorchTitanEngineConfig:
  """Configuration for TorchTitanEngine."""

  model_name: str = "qwen3"
  flavor: str = "debugmodel"
  hf_assets_path: str = ""
  checkpoint_dir: str = ""
  save_interval_steps: int = 0
  max_to_keep: int | None = None
  seq_bucket_size: int = 0
  dtype: str = "float32"
  mixed_precision_param: str = "bfloat16"
  learning_rate: float = 1e-5
  init_lr: float = 0.0
  end_lr: float = 0.0
  weight_decay: float = 0.01
  beta1: float = 0.9
  beta2: float = 0.999
  eps: float = 1e-8
  max_grad_norm: float = 1.0
  warmup_steps: int = 10
  decay_steps: int = 1000
  max_context_length: int = 512
  use_splash_attention: bool = False
  device_type: str = "cpu"
  load_hf_weights: bool = True


class TorchTitanEngine:
  """TorchTitan-based training engine for autoregressive language models."""

  def __init__(
      self,
      config: TorchTitanEngineConfig,
      model: torch.nn.Module | None = None,
      device: torch.device | None = None,
      dp_mesh: DeviceMesh | None = None,
  ):
    self.config = config
    self.device = device or torch.device(config.device_type)
    self.dp_mesh = dp_mesh
    self.algo_config: algorithm_config.AlgorithmConfig | None = None
    self._loss_fn: (
        algorithm_config.PolicyLossFn | grpo_loss.TorchPolicyLoss | None
    ) = None
    self.pad_id = 0
    self.eos_id = 0

    self._accum_loss_sum: torch.Tensor | None = None
    self._accum_denom: torch.Tensor | None = None
    self._accum_token_denom: torch.Tensor | None = None
    self._accum_metric_sums: dict[str, torch.Tensor] = {}
    self.microbatch_count = 0
    self.train_step = 0
    self.policy_version = 0

    self.master_dtype = _parse_torch_dtype("dtype", config.dtype)
    self.compute_dtype = _parse_torch_dtype(
        "mixed_precision_param", config.mixed_precision_param
    )

    is_sharded = dp_mesh is not None and dp_mesh.size() > 1
    self._is_sharded = is_sharded

    model_name, flavor = model_registry.resolve_model_name_and_flavor(
        config.model_name, config.flavor
    )
    self.model_spec = model_registry.resolve_model_spec(
        model_name,
        flavor,
        attn_backend="sdpa",
        max_context_length=config.max_context_length,
        hf_assets_path=config.hf_assets_path,
    )

    if model is not None:
      self.model = model
      self.model.to(self.master_dtype)
      self._maybe_apply_tpu_patches(self.model)
      model_cfg = getattr(model, "config", self.model_spec.model)
      if self.model_spec.state_dict_adapter is not None:
        self.state_dict_adapter = self.model_spec.state_dict_adapter(
            model_cfg, config.hf_assets_path or None
        )
      else:
        self.state_dict_adapter = None
      if is_sharded:
        self._parallelize_model(self.model, dp_mesh, self.compute_dtype)
    else:
      if is_sharded:
        with torch.device("meta"):
          built_model = self.model_spec.model.build()
        if not isinstance(built_model, Decoder):
          raise TypeError(
              "Expected Decoder model for FSDP2 sharding, got"
              f" {type(built_model)}"
          )
        self.model = built_model
        self._maybe_apply_tpu_patches(self.model)
        self._parallelize_model(self.model, dp_mesh, self.compute_dtype)
        self.model.to_empty(device=self.device)
        with torch.no_grad():
          self.model.init_states(buffer_device=self.device)
        self.model.to(self.master_dtype)
      else:
        self.model = self.model_spec.model.build().to(
            device=self.device, dtype=self.master_dtype
        )
        self._maybe_apply_tpu_patches(self.model)

      if self.model_spec.state_dict_adapter is not None:
        self.state_dict_adapter = self.model_spec.state_dict_adapter(
            self.model_spec.model, config.hf_assets_path or None
        )
      else:
        self.state_dict_adapter = None

      if config.load_hf_weights and config.hf_assets_path:
        self.load_hf_weights_from_dir(config.hf_assets_path)

    # Use named_parameters() so param_names is populated for FQN-keyed state
    # dicts
    self.optimizer = torch.optim.AdamW(
        self.model.named_parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
        foreach=True,
    )
    self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer,
        lr_lambda=lambda step: _warmup_cosine_decay_lambda(
            step=step,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.decay_steps,
            init_lr=self.config.init_lr,
            peak_lr=self.config.learning_rate,
            end_lr=self.config.end_lr,
        ),
    )

  def _forward_precision_context(
      self,
  ) -> contextlib.AbstractContextManager[None]:
    """Enables autocast on single-device paths where FSDP2 is inactive."""
    if not self._is_sharded and self.compute_dtype == torch.bfloat16:
      return torch.autocast(
          device_type=self.device.type, dtype=self.compute_dtype
      )
    return contextlib.nullcontext()

  def _maybe_apply_tpu_patches(self, model: torch.nn.Module) -> None:
    """Applies TPU-specific SplashAttention monkeypatches if enabled."""
    if self.config.use_splash_attention and self.device.type == "tpu":
      from torchtitan.experiments.tpu import workarounds as tpu_workarounds  # pylint: disable=g-import-not-at-top

      tpu_workarounds.use_tokamax_splash_attention_patch(model, None)

  def _parallelize_model(
      self,
      model: torch.nn.Module,
      dp_mesh: DeviceMesh,
      param_dtype: torch.dtype,
  ) -> None:
    """Applies FSDP2 parallelization to decoder model."""
    first_param = next(model.parameters(), None)
    if (
        first_param is not None
        and not isinstance(first_param, DTensor)
        and isinstance(model, Decoder)
    ):
      apply_fsdp_to_decoder(
          model,
          dp_mesh,
          param_dtype=param_dtype,
          reduce_dtype=torch.float32,
          pp_enabled=False,
          cpu_offload=False,
          reshard_after_forward_policy="default",
      )

  def load_hf_weights_from_dir(self, hf_dir: str) -> None:
    """Loads weights from HuggingFace .safetensors files in `hf_dir` into model."""
    hf_path = epath.Path(hf_dir)
    safetensors_files = sorted(hf_path.glob("*.safetensors"))
    if not safetensors_files:
      raise FileNotFoundError(
          f"No .safetensors files found in hf_dir: {hf_dir}"
      )

    hf_sd = {}
    for st_path in safetensors_files:
      if not os.path.exists(str(st_path)):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as tmp:
          tmp.write(st_path.read_bytes())
          tmp.flush()
          hf_sd.update(safetensors.torch.load_file(tmp.name, device="cpu"))
      else:
        hf_sd.update(safetensors.torch.load_file(str(st_path), device="cpu"))

    if self.state_dict_adapter is None:
      raise RuntimeError(
          "Cannot load HuggingFace weights without a state_dict_adapter"
          " configured."
      )
    titan_sd = self.state_dict_adapter.from_hf(hf_sd)

    # Merge fused QKV projections if any module uses FusedQKVLinear
    for mod_name, module in self.model.named_modules():
      if hasattr(module, "_merge_qkv_on_load"):
        prefix = f"{mod_name}." if mod_name else ""
        module._merge_qkv_on_load(module, titan_sd, prefix)  # pylint: disable=protected-access

    if getattr(self.model, "enable_weight_tying", False):
      if (
          "tok_embeddings.weight" not in titan_sd
          and "lm_head.weight" in titan_sd
      ):
        titan_sd["tok_embeddings.weight"] = titan_sd["lm_head.weight"]
      elif (
          "lm_head.weight" not in titan_sd
          and "tok_embeddings.weight" in titan_sd
      ):
        titan_sd["lm_head.weight"] = titan_sd["tok_embeddings.weight"]

    named_params = dict(self.model.named_parameters())
    missing = set(named_params.keys()) - set(titan_sd.keys())
    if missing:
      raise KeyError(
          f"Missing parameter keys from HuggingFace checkpoint: {missing}"
      )
    with torch.no_grad():
      for name, param in named_params.items():
        val = titan_sd[name].to(dtype=param.dtype)
        if isinstance(param, DTensor):
          sharded = distribute_tensor(
              val.to(device=self.device),
              param.device_mesh,
              param.placements,
              src_data_rank=None,
          )
          param.to_local().copy_(sharded.to_local())
          del sharded
        else:
          param.copy_(val.to(device=self.device))

  def with_loss_fn(
      self,
      loss_fn: str | algorithm_config.PolicyLossFn,
      *,
      algo_config: algorithm_config.AlgorithmConfig,
      role: datatypes.Role = datatypes.Role.ACTOR,
      pad_id: int = 0,
      eos_id: int = 0,
      fuse_agreement_in_loss: bool = False,
      return_routed_experts: bool = False,
  ) -> None:
    """Validates and installs the policy loss kernel and algorithm config."""
    if not isinstance(algo_config, algorithm_config.AlgorithmConfig):
      raise TypeError(
          "TorchTitanEngine.with_loss_fn requires AlgorithmConfig, got"
          f" {type(algo_config).__name__}."
      )
    if role != datatypes.Role.ACTOR:
      raise ValueError(
          f"TorchTitanEngine only supports Role.ACTOR, got {role!r}."
      )
    temperature = (
        1.0
        if algo_config.temperature is None
        else float(algo_config.temperature)
    )
    if temperature <= 0:
      raise ValueError(
          "AlgorithmConfig.temperature must be > 0, got"
          f" {algo_config.temperature!r}."
      )
    if fuse_agreement_in_loss:
      raise ValueError(
          "TorchTitanEngine does not yet support fuse_agreement_in_loss=True."
      )
    if return_routed_experts:
      raise ValueError(
          "TorchTitanEngine does not support return_routed_experts=True."
      )
    if algo_config.score_centering:
      raise ValueError(
          "TorchTitanEngine does not yet support score_centering=True."
      )

    resolved_loss_fn: algorithm_config.PolicyLossFn | grpo_loss.TorchPolicyLoss
    if isinstance(loss_fn, str):
      plugin = grpo_loss.get_torch_policy_loss(loss_fn)
      plugin.validate(
          algo_config, fuse_agreement_in_loss=fuse_agreement_in_loss
      )
      resolved_loss_fn = plugin
    elif callable(loss_fn):
      _validate_custom_torch_policy_loss_fn(loss_fn, algo_config)
      resolved_loss_fn = loss_fn
    else:
      raise TypeError(
          "TorchTitanEngine.with_loss_fn requires a registered loss name (str)"
          f" or PolicyLossFn callable, got {type(loss_fn).__name__}."
      )

    self._loss_fn = resolved_loss_fn
    self.algo_config = algo_config
    self.pad_id = int(pad_id)
    self.eos_id = int(eos_id)

  def compile(
      self, dummy_payload: datatypes.RLTrainerPayload | None = None
  ) -> None:
    """Warmup compilation without mutating weights or optimizer state."""
    del dummy_payload
    return None

  def fwd_bwd(
      self,
      local_payload: datatypes.RLTrainerPayload,
      dp_group: dist.ProcessGroup | None = None,
  ) -> dict[str, float]:
    """Runs forward and backward passes on local slice of RLTrainerPayload."""
    del dp_group
    if self.algo_config is None or self._loss_fn is None:
      raise RuntimeError(
          "TorchTitanEngine.with_loss_fn must be called before fwd_bwd."
      )

    if local_payload.segment_ids is not None:
      raise ValueError(
          "TorchTitanEngine does not support packed payloads (segment_ids must"
          " be None)."
      )
    if local_payload.sampler_is_weights is not None:
      raise ValueError(
          "TorchTitanEngine does not yet support non-None sampler_is_weights."
      )

    self.model.train()

    prompt_ids = local_payload.prompt_ids
    prompt_mask = local_payload.prompt_mask
    completion_ids = local_payload.completion_ids
    completion_mask = local_payload.completion_mask
    advantages = local_payload.advantages
    old_logps = local_payload.old_per_token_logps
    ref_logps = local_payload.ref_per_token_logps

    pad_id = self.pad_id
    if self.config.seq_bucket_size > 0:
      b_size = self.config.seq_bucket_size
      p_len = prompt_ids.shape[1]
      c_len = completion_ids.shape[1]
      p_bucket = ((p_len + b_size - 1) // b_size) * b_size
      c_bucket = ((c_len + b_size - 1) // b_size) * b_size
      c_pad = c_bucket - c_len
      if c_pad > 0:
        completion_ids = np.pad(
            completion_ids, ((0, 0), (0, c_pad)), constant_values=pad_id
        )
        completion_mask = np.pad(
            completion_mask, ((0, 0), (0, c_pad)), constant_values=0
        )
        if advantages.ndim == 2:
          advantages = np.pad(
              advantages, ((0, 0), (0, c_pad)), constant_values=0.0
          )
        if old_logps is not None:
          old_logps = np.pad(
              old_logps, ((0, 0), (0, c_pad)), constant_values=0.0
          )
        if ref_logps is not None:
          ref_logps = np.pad(
              ref_logps, ((0, 0), (0, c_pad)), constant_values=0.0
          )
      total_length = p_bucket + c_bucket
    else:
      total_length = None

    compact_tokens_np, positions_np, prompt_lens_np = (
        grpo_loss.compact_left_padded_batch_np(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            pad_id=pad_id,
            total_length=total_length,
        )
    )

    batch_size = compact_tokens_np.shape[0]
    eff_total = compact_tokens_np.shape[1]
    tokens_1d = torch.from_numpy(compact_tokens_np.reshape(-1)).to(self.device)
    positions = torch.from_numpy(positions_np).to(self.device)
    prompt_lens = torch.from_numpy(prompt_lens_np).to(self.device)
    completion_ids_t = torch.from_numpy(completion_ids).to(self.device)
    completion_mask_t = torch.from_numpy(completion_mask).to(self.device)
    advantages_t = torch.from_numpy(advantages).to(self.device)
    old_logps_t = (
        torch.from_numpy(old_logps).to(self.device)
        if old_logps is not None
        else None
    )
    ref_logps_t = (
        torch.from_numpy(ref_logps).to(self.device)
        if ref_logps is not None
        else None
    )

    with (
        self._forward_precision_context(),
        unflatten_shim.set_unflatten_seq_len(eff_total),
    ):
      out = self.model(tokens_1d, positions=positions)
    logits_flat = out[0] if isinstance(out, (tuple, list)) else out
    logits = logits_flat.reshape(batch_size, eff_total, -1)

    completion_width = completion_ids.shape[1]
    extracted_logits = grpo_loss.extract_completion_logits(
        logits=logits,
        prompt_lens=prompt_lens,
        completion_width=completion_width,
    )

    temperature = (
        1.0
        if self.algo_config.temperature is None
        else float(self.algo_config.temperature)
    )
    per_token_logps = grpo_loss.compute_selective_log_softmax(
        logits=extracted_logits,
        target_ids=completion_ids_t,
        temperature=temperature,
    )

    loss_out = self._loss_fn(
        algorithm_config.PolicyLossInputs(
            per_token_logps=per_token_logps,
            completion_mask=completion_mask_t,
            advantages=advantages_t,
            old_per_token_logps=old_logps_t,
            ref_per_token_logps=ref_logps_t,
            sampler_agreement_applied=(
                local_payload.sampler_agreement_applied
            ),
        ),
        self.algo_config,
    )

    if self.device.type == "tpu":
      try:
        from torch_tpu._internal.sync import synchronize  # pylint: disable=g-import-not-at-top

        synchronize(loss_out.loss_sum, wait=False)
      except ImportError:
        pass

    loss_out.loss_sum.backward()

    loss_sum = loss_out.loss_sum.detach()
    local_denom = loss_out.denominator.detach()
    token_denom = (
        loss_out.token_denom
        if loss_out.token_denom is not None
        else torch.sum(completion_mask_t.to(torch.float32))
    ).detach()

    if self._accum_loss_sum is None:
      self._accum_loss_sum = torch.zeros_like(loss_sum)
      self._accum_denom = torch.zeros_like(local_denom)
      self._accum_token_denom = torch.zeros_like(token_denom)
    self._accum_loss_sum += loss_sum
    self._accum_denom += local_denom
    self._accum_token_denom += token_denom

    for k, v in loss_out.metrics.items():
      v_det = v.detach()
      if k not in self._accum_metric_sums:
        self._accum_metric_sums[k] = torch.zeros_like(v_det)
      self._accum_metric_sums[k] += v_det
    self.microbatch_count += 1

    return {}

  def update(
      self,
      dp_group: dist.ProcessGroup | None = None,
  ) -> tuple[int, dict[str, float]]:
    """Applies accumulated gradients, steps optimizer, and returns step metrics."""
    if self._accum_loss_sum is None:
      return self.train_step, {}

    metric_keys = sorted(self._accum_metric_sums.keys())
    stats_tensors = [
        self._accum_loss_sum,
        self._accum_denom,
        self._accum_token_denom,
        *[self._accum_metric_sums[k] for k in metric_keys],
    ]
    stats = torch.stack(stats_tensors).to(
        dtype=torch.float32, device=self.device
    )

    if dp_group is not None and dist.get_world_size(dp_group) > 1:
      dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=dp_group)

    stats_list = stats.tolist()
    global_loss_sum = stats_list[0]
    global_denom = stats_list[1]
    global_token_denom = stats_list[2]
    global_metric_sums = {k: val for k, val in zip(metric_keys, stats_list[3:])}

    eff_denom = max(global_denom, 1.0)
    for p in self.model.parameters():
      if p.grad is not None:
        p.grad.div_(eff_denom)

    grad_norm = titan_clip_grad_norm_(
        self.model.parameters(), max_norm=self.config.max_grad_norm
    )
    current_lr = self.optimizer.param_groups[0]["lr"]
    self.optimizer.step()
    self.lr_scheduler.step()
    self.optimizer.zero_grad()

    token_d = max(global_token_denom, 1.0)
    metrics_buffer = {
        "loss": global_loss_sum / eff_denom,
        "learning_rate": current_lr,
        "grad_norm": float(
            grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
        ),
        **{k: val / token_d for k, val in global_metric_sums.items()},
    }

    self._accum_loss_sum = None
    self._accum_denom = None
    self._accum_token_denom = None
    self._accum_metric_sums.clear()
    self.microbatch_count = 0

    self.train_step += 1
    return self.train_step, metrics_buffer

  def per_token_logps(
      self,
      prompt_tokens: np.ndarray,
      completion_tokens: np.ndarray,
      pad_id: int,
      temperature: float = 1.0,
  ) -> np.ndarray:
    """Computes per-token log-probabilities under current weights."""
    self.model.eval()
    with torch.no_grad():
      p_mask = prompt_tokens != pad_id
      orig_c_width = completion_tokens.shape[1]
      c_ids = completion_tokens
      if self.config.seq_bucket_size > 0:
        b_size = self.config.seq_bucket_size
        p_len = prompt_tokens.shape[1]
        c_len = c_ids.shape[1]
        p_bucket = ((p_len + b_size - 1) // b_size) * b_size
        c_bucket = ((c_len + b_size - 1) // b_size) * b_size
        c_pad = c_bucket - c_len
        if c_pad > 0:
          c_ids = np.pad(c_ids, ((0, 0), (0, c_pad)), constant_values=pad_id)
        total_length = p_bucket + c_bucket
      else:
        total_length = None

      compact_tokens_np, positions_np, prompt_lens_np = (
          grpo_loss.compact_left_padded_batch_np(
              prompt_ids=prompt_tokens,
              prompt_mask=p_mask,
              completion_ids=c_ids,
              pad_id=pad_id,
              total_length=total_length,
          )
      )

      batch_size = compact_tokens_np.shape[0]
      eff_total = compact_tokens_np.shape[1]
      tokens_1d = torch.from_numpy(compact_tokens_np.reshape(-1)).to(
          self.device
      )
      positions = torch.from_numpy(positions_np).to(self.device)
      prompt_lens = torch.from_numpy(prompt_lens_np).to(self.device)
      c_ids_t = torch.from_numpy(c_ids).to(self.device)

      with (
          self._forward_precision_context(),
          unflatten_shim.set_unflatten_seq_len(eff_total),
      ):
        out = self.model(tokens_1d, positions=positions)
      logits_flat = out[0] if isinstance(out, (tuple, list)) else out
      logits = logits_flat.reshape(batch_size, eff_total, -1)
      extracted_logits = grpo_loss.extract_completion_logits(
          logits=logits,
          prompt_lens=prompt_lens,
          completion_width=c_ids.shape[1],
      )

      logps = grpo_loss.compute_selective_log_softmax(
          logits=extracted_logits,
          target_ids=c_ids_t,
          temperature=temperature,
      )
      return logps.cpu().numpy()[:, :orig_c_width]

  def export_hf_state_dict(
      self,
      *,
      is_rank_zero: bool = True,
  ) -> dict[str, torch.Tensor]:
    """Exports model weights to HF schema CPU tensors one tensor at a time."""
    state_dict = self.model.state_dict()
    unsharded_state_dict: dict[str, torch.Tensor] = {}

    for k in sorted(state_dict.keys()):
      v = state_dict[k]
      if isinstance(v, DTensor):
        full_tensor = v.full_tensor().detach()
      else:
        full_tensor = v.detach()

      if not is_rank_zero:
        del full_tensor
        continue

      cpu_tensor = full_tensor.to(
          dtype=self.compute_dtype, device="cpu"
      ).contiguous()
      del full_tensor
      unsharded_state_dict[k] = cpu_tensor

    if not is_rank_zero:
      return {}

    if self.state_dict_adapter is None:
      raise RuntimeError(
          "Cannot export HuggingFace weights without a state_dict_adapter"
          " configured."
      )
    hf_sd = self.state_dict_adapter.to_hf(unsharded_state_dict)
    del unsharded_state_dict

    return {
        k: v.to(dtype=self.compute_dtype, device="cpu").contiguous()
        for k, v in hf_sd.items()
    }

  def save_dcp_checkpoint(
      self,
      local_dir: str,
      process_group: dist.ProcessGroup | None = None,
  ) -> dict[str, Any]:
    """Flushes devices, initializes optim state, and saves model+optim state using DCP."""
    if self.device.type != "cpu" and torch.accelerator.is_available():
      torch.accelerator.synchronize()
    init_optim_state(self.optimizer)
    state = {
        "model": self.model.state_dict(),
        "optimizer": get_flat_optim_state_dict(self.optimizer),
    }
    dcp.save(
        state,
        checkpoint_id=local_dir,
        process_group=process_group,
        no_dist=(process_group is None),
    )
    return self.lr_scheduler.state_dict()

  def load_dcp_checkpoint(
      self,
      local_dir: str,
      *,
      train_step: int,
      policy_version: int,
      lr_scheduler_state: Mapping[str, Any] | None = None,
      process_group: dist.ProcessGroup | None = None,
  ) -> None:
    """Loads model+optim state from DCP checkpoint and updates counters and lr scheduler."""
    init_optim_state(self.optimizer)
    ckpt_state = {
        "model": self.model.state_dict(),
        "optimizer": get_flat_optim_state_dict(self.optimizer),
    }
    dcp.load(
        ckpt_state,
        checkpoint_id=local_dir,
        process_group=process_group,
        no_dist=(process_group is None),
    )
    self.model.load_state_dict(ckpt_state["model"])
    load_flat_optim_state_dict(self.optimizer, ckpt_state["optimizer"])
    if lr_scheduler_state is not None:
      self.lr_scheduler.load_state_dict(dict(lr_scheduler_state))
    self.train_step = train_step
    self.policy_version = policy_version
    if self.device.type != "cpu" and torch.accelerator.is_available():
      torch.accelerator.synchronize()
