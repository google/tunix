# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optimizer trajectories against Torch AdamW and the original source schedule."""

import copy
import dataclasses
import os
from pathlib import Path
import subprocess

import jax.numpy as jnp
import numpy as np
import optax
import pytest
from examples.r2orch_math import optimizers
from examples.r2orch_math import source_adapter
from examples.r2orch_math.prepare_data import SOURCE_COMMIT


def config(**overrides):
  return {
      "learning_rate": 0.02,
      "optimizer": {
          "weight_decay": 0.01,
          "max_grad_norm": 1.0,
          "warmup_steps": 2,
          "warmup_start_factor": 0.0,
          "learning_rate_schedule": "cosine",
          "total_optimizer_steps": 8,
          **overrides,
      },
  }


def test_adamw_clipping_schedule_and_updates_match_torch():
  torch = pytest.importorskip("torch")
  from transformers import get_cosine_schedule_with_warmup

  cfg = config()
  initial = np.array([2.0, -0.5, 0.3], dtype=np.float32)
  parameter = torch.nn.Parameter(torch.tensor(initial))
  reference = torch.optim.AdamW([parameter], lr=0.02, weight_decay=0.01)
  scheduler = get_cosine_schedule_with_warmup(
      reference, num_warmup_steps=2, num_training_steps=8
  )
  optimizer = optimizers.make_optimizer(cfg, "qwen", num_optimizer_steps=8)
  value = jnp.asarray(initial)
  state = optimizer.init(value)
  schedule = optimizers.learning_rate_schedule(cfg)
  for step in range(8):
    gradients = np.array([20 - step, -3.0, 0.1 + step], dtype=np.float32)
    np.testing.assert_allclose(
        float(schedule(step)), reference.param_groups[0]["lr"], atol=2e-9
    )
    parameter.grad = torch.tensor(gradients)
    torch.nn.utils.clip_grad_norm_([parameter], 1.0)
    reference.step()
    scheduler.step()
    reference.zero_grad(set_to_none=True)
    updates, state = optimizer.update(jnp.asarray(gradients), state, value)
    value = optax.apply_updates(value, updates)
    np.testing.assert_allclose(
        np.asarray(value), parameter.detach().numpy(), atol=4e-7, rtol=4e-7
    )
  assert not np.array_equal(np.asarray(value), initial)


def test_fixed_horizon_and_restored_optimizer_count_preserve_resume():
  cfg = config(total_optimizer_steps=20)
  optimizer = optimizers.make_optimizer(cfg, "qwen", num_optimizer_steps=12)
  value, state = jnp.array([1.0, 2.0]), optimizer.init(jnp.array([1.0, 2.0]))
  for _ in range(3):
    updates, state = optimizer.update(jnp.array([4.0, -3.0]), state, value)
    value = optax.apply_updates(value, updates)
  restored_value, restored_state = copy.deepcopy((value, state))
  # Changing only the stop point does not derive a new decay horizon.
  resumed = optimizers.make_optimizer(cfg, "qwen", num_optimizer_steps=20)
  for _ in range(3):
    update, state = optimizer.update(jnp.array([4.0, -3.0]), state, value)
    value = optax.apply_updates(value, update)
    update, restored_state = resumed.update(
        jnp.array([4.0, -3.0]), restored_state, restored_value
    )
    restored_value = optax.apply_updates(restored_value, update)
  np.testing.assert_array_equal(np.asarray(value), np.asarray(restored_value))
  with pytest.raises(ValueError, match="horizon"):
    optimizers.make_optimizer(cfg, "qwen", num_optimizer_steps=21)


def test_cpu_keeps_plain_sgd_and_constant_warmup_is_supported():
  optimizer = optimizers.make_optimizer({"learning_rate": 0.1}, "cpu")
  value = jnp.array([1.0])
  updates, _ = optimizer.update(jnp.array([3.0]), optimizer.init(value), value)
  np.testing.assert_allclose(
      np.asarray(optax.apply_updates(value, updates)), [0.7]
  )
  schedule = optimizers.learning_rate_schedule(
      config(learning_rate_schedule="constant", warmup_start_factor=0.5)
  )
  np.testing.assert_allclose(
      [float(schedule(i)) for i in range(4)], [0.01, 0.015, 0.02, 0.02]
  )


@pytest.mark.parametrize(
    "settings",
    [
        {"total_optimizer_steps": True},
        {"warmup_steps": 1.5},
        {"weight_decay": -1},
        {"max_grad_norm": 0},
        {"warmup_start_factor": 1.1},
        {"learning_rate_schedule": "linear"},
        {"weight_decay": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_optimizer_settings_are_rejected(settings):
  with pytest.raises(ValueError):
    optimizers.make_optimizer(config(**settings), "qwen")


def test_pinned_source_launcher_and_scheduler_contract():
  source = os.environ.get("TUNIX_R2ORCH_SOURCE")
  if not source:
    pytest.skip(
        "Set TUNIX_R2ORCH_SOURCE to compare the actual source scheduler."
    )
  torch = pytest.importorskip("torch")
  original = source_adapter.load_source_module(source, "training")
  parameters = original.PolicyTrainingConfig(Path("unused"), Path("unused"))
  assert parameters.weight_decay == 0.01 and parameters.max_grad_norm == 1.0
  assert parameters.warmup_steps == 10 and parameters.warmup_start_factor == 0
  assert parameters.learning_rate_schedule == "cosine"
  launcher = subprocess.check_output(
      ["git", "-C", source, "show", f"{SOURCE_COMMIT}:scripts/r2orch/train.sh"],
      text=True,
  )
  assert "--lora-dropout-override 0" in launcher
  assert (
      "--kl-reference-stride 4" in launcher and "--kl-term-cap 10" in launcher
  )
  cfg = config(warmup_steps=10, total_optimizer_steps=400)
  for start in (0.0, 0.2):
    cfg["optimizer"]["warmup_start_factor"] = start
    parameters = dataclasses.replace(
        parameters,
        learning_rate=cfg["learning_rate"],
        warmup_steps=10,
        warmup_start_factor=start,
        total_optimizer_steps=400,
    )
    reference = original._build_learning_rate_scheduler(
        torch.optim.AdamW(
            [torch.nn.Parameter(torch.zeros(1))], lr=cfg["learning_rate"]
        ),
        parameters,
    )
    schedule = optimizers.learning_rate_schedule(cfg)
    for step in (0, 1, 9, 10, 100, 399, 400):
      expected = reference.base_lrs[0] * reference.lr_lambdas[0](step)
      np.testing.assert_allclose(float(schedule(step)), expected, atol=3e-9)
