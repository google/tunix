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

"""The pinned R²Orch launcher's AdamW, clipping and optimizer-step schedule.

The schedule counts actual optimizer updates, not collection rounds. Production
configs store their horizon explicitly so stopping and resuming a run does not
silently change its learning-rate trajectory. The CPU fixture retains SGD.
"""

import math

import jax
import jax.numpy as jnp
import optax


def settings(config, *, num_optimizer_steps=None):
  """Validate the original CLI defaults plus an explicit scheduler horizon."""
  values = {
      "weight_decay": 0.01,
      "max_grad_norm": 1.0,
      "warmup_steps": 10,
      "warmup_start_factor": 0.0,
      "learning_rate_schedule": "cosine",
      **config.get("optimizer", {}),
  }
  unknown = set(values) - {
      "weight_decay",
      "max_grad_norm",
      "warmup_steps",
      "warmup_start_factor",
      "learning_rate_schedule",
      "total_optimizer_steps",
  }
  if unknown:
    raise ValueError(f"Unknown optimizer settings: {sorted(unknown)}")
  horizon = values.get("total_optimizer_steps", num_optimizer_steps)
  if type(horizon) is not int or horizon < 1:
    raise ValueError("An explicit positive total_optimizer_steps is required.")
  values["total_optimizer_steps"] = horizon
  if num_optimizer_steps is not None and (
      type(num_optimizer_steps) is not int
      or not 0 < num_optimizer_steps <= horizon
  ):
    raise ValueError(
        "Requested training exceeds the configured optimizer horizon."
    )
  for name in ("weight_decay", "max_grad_norm", "warmup_start_factor"):
    if not math.isfinite(values[name]) or values[name] < 0:
      raise ValueError(f"Invalid optimizer setting: {name}")
  if values["max_grad_norm"] <= 0 or values["warmup_start_factor"] > 1:
    raise ValueError(
        "Gradient norm must be positive and warmup factor at most one."
    )
  if type(values["warmup_steps"]) is not int or values["warmup_steps"] < 0:
    raise ValueError("warmup_steps must be a nonnegative integer.")
  if values["learning_rate_schedule"] not in ("cosine", "constant"):
    raise ValueError(
        "Only the original cosine and constant schedules are supported."
    )
  return values


def learning_rate_schedule(config, *, num_optimizer_steps=None):
  """Match the source LambdaLR/Transformers schedule, including its step zero."""
  values = settings(config, num_optimizer_steps=num_optimizer_steps)
  learning_rate = float(config["learning_rate"])
  if not math.isfinite(learning_rate) or learning_rate <= 0:
    raise ValueError("learning_rate must be finite and positive.")
  warmup = values["warmup_steps"]
  horizon = values["total_optimizer_steps"]
  start = values["warmup_start_factor"]

  def schedule(step):
    step = jnp.asarray(step, dtype=jnp.float32)
    warmup_factor = start + (1.0 - start) * step / max(1, warmup)
    if values["learning_rate_schedule"] == "constant":
      decay = jnp.ones_like(step)
    else:
      progress = (step - warmup) / max(1, horizon - warmup)
      decay = jnp.maximum(0.0, 0.5 * (1.0 + jnp.cos(jnp.pi * progress)))
    return learning_rate * jnp.where(step < warmup, warmup_factor, decay)

  return schedule


def _clip_grad_norm(max_norm):
  """Torch clip_grad_norm_'s global L2 rule, including the 1e-6 denominator."""

  def update(updates, state, params=None):
    del params
    norm = optax.global_norm(updates)
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree.map(lambda value: value * factor, updates), state

  return optax.GradientTransformation(lambda _: optax.EmptyState(), update)


def make_optimizer(config, backend, *, num_optimizer_steps=None):
  """Create a native optimizer without changing the categorical CPU fixture."""
  if backend == "cpu":
    return optax.sgd(config["learning_rate"])
  if backend != "qwen":
    raise ValueError("Expected the cpu or qwen recipe backend.")
  values = settings(config, num_optimizer_steps=num_optimizer_steps)
  return optax.chain(
      _clip_grad_norm(values["max_grad_norm"]),
      optax.adamw(
          learning_rate_schedule(
              config, num_optimizer_steps=num_optimizer_steps
          ),
          b1=0.9,
          b2=0.999,
          eps=1e-8,
          eps_root=0.0,
          weight_decay=values["weight_decay"],
      ),
  )
