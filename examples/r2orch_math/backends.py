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

"""Pinned Qwen backends with isolated trainable adapters and frozen Workers.

The native trainer discovers nnx.LoRAParam and optimizes only those leaves.
Nonzero LoRA dropout is rejected until sampling, old/reference scoring and
gradient passes share an explicit deterministic/stochastic execution contract.
"""

import math
from pathlib import Path
import re

from flax import nnx
import jax
import jax.numpy as jnp
from examples.swarm_math import train as shared_train
from examples.swarm_math import workflow as math_workflow
from tunix.common import configs

MODEL_IDS = ("Qwen/Qwen3-4B", "Qwen/Qwen3.5-9B")
LORA_TARGETS = (
    ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"
)


def validate_backend_config(config, model_path):
  if config["model_id"] not in MODEL_IDS:
    raise ValueError(f"Supported R²Orch model IDs are {MODEL_IDS}.")
  revision = config.get("model_revision", "")
  if not re.fullmatch(r"[0-9a-f]{40}", revision):
    raise ValueError(
        "Pin model_revision to a full immutable Hugging Face commit."
    )
  if not model_path or not Path(model_path).is_dir():
    raise ValueError("A local pinned model snapshot directory is required.")
  if Path(model_path).resolve().name != revision:
    raise ValueError("The snapshot directory must match model_revision.")
  if config.get("rollout_engine") != "vanilla":
    raise ValueError("These independent adapter backends use native sampling.")
  if config.get("dtype") != "float32" or config.get("use_flash_attention"):
    raise ValueError("R²Orch currently requires float32 reference attention.")
  validate_lora(config.get("lora"))


def validate_lora(lora):
  if lora is None:
    return
  if set(lora) - {"rank", "alpha", "dropout", "module_path"}:
    raise ValueError("Unknown LoRA configuration field.")
  if type(lora.get("rank")) is not int or lora["rank"] < 1:
    raise ValueError("LoRA rank must be a positive integer.")
  if not math.isfinite(lora.get("alpha", 0)) or lora.get("alpha", 0) <= 0:
    raise ValueError("LoRA alpha must be finite and positive.")
  if lora.get("dropout", 0.0) != 0.0:
    raise ValueError(
        "Nonzero LoRA dropout requires an explicit scoring/training RNG"
        " contract."
    )
  if not isinstance(lora.get("module_path", LORA_TARGETS), str):
    raise ValueError("LoRA module_path must be a regular expression string.")


def apply_lora(model, lora, *, mesh=None, seed=0):
  """Install deterministic adapters; reject patterns matching no parameters."""
  validate_lora(lora)
  if lora is None:
    return model
  import qwix  # pylint: disable=g-import-not-at-top

  provider = qwix.LoraProvider(
      module_path=lora.get("module_path", LORA_TARGETS),
      rank=lora["rank"],
      alpha=lora["alpha"],
      dropout=0.0,
  )
  # Initialization needs only enough tokens to visit every layer. Match the
  # activation batch to the configured FSDP mesh rather than hard-coding two.
  batch = max(2, int(mesh.shape.get("fsdp", 1))) if mesh is not None else 2
  length = 4
  inputs = dict(
      input_tokens=jnp.ones((batch, length), jnp.int32),
      positions=jnp.broadcast_to(jnp.arange(length), (batch, length)),
      cache=None,
      attention_mask=jnp.broadcast_to(
          jnp.tril(jnp.ones((length, length), jnp.bool_)),
          (batch, length, length),
      ),
  )
  original_remat = getattr(model.config, "remat_config", None)
  if original_remat is not None:
    model.config.remat_config = type(original_remat).NONE
  try:
    result = qwix.apply_lora_to_model(
        model, provider, rngs=nnx.Rngs(seed), **inputs
    )
  finally:
    if original_remat is not None:
      model.config.remat_config = original_remat
  if original_remat is not None:
    result.config.remat_config = original_remat
  if not jax.tree.leaves(nnx.state(result, nnx.LoRAParam)):
    raise ValueError("LoRA module_path matched no trainable parameters.")
  if mesh is not None:
    from tunix.rl import reshard  # pylint: disable=g-import-not-at-top

    result = reshard.reshard_model_to_mesh(result, mesh)
  return result


def load_base_policy(config, mesh, model_path):
  """Load an independent, unadapted reference/Worker model from pinned bytes."""
  validate_backend_config(config, model_path)
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  model, _ = automodel.AutoModel.from_pretrained(
      config["model_id"],
      mesh,
      model_path=str(model_path),
      remat_config=config["remat_config"],
      dtype=config["dtype"],
      load_dtype=config["load_dtype"],
      use_flash_attention=False,
      flash_attention_block_size=config["flash_attention_block_size"],
  )
  return model


def load_trainable_policy(config, mesh, model_path):
  model = load_base_policy(config, mesh, model_path)
  return apply_lora(model, config.get("lora"), mesh=mesh, seed=config["seed"])


def make_native_samplers(actor, frozen, tokenizer, config, *, eos_tokens):
  """Build isolated actor serving state and immutable Worker/reference state."""
  serving = nnx.clone(actor)
  nnx.update(serving, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))

  def make(
      model, serving_model, temperature, max_tokens, stage, *, prompt_limit=None
  ):
    prompt_limit = (
        config["max_prompt_length"] if prompt_limit is None else prompt_limit
    )
    if type(prompt_limit) is not int or prompt_limit < 1:
      raise ValueError("Each policy prompt limit must be a positive integer.")
    return shared_train.NativeMathSampler(
        model,
        tokenizer,
        configs.RolloutConfig(
            max_prompt_length=prompt_limit,
            max_tokens_to_generate=max_tokens,
            kv_cache_size=prompt_limit + max_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=None,
            eos_tokens=eos_tokens,
            seed=config["seed"],
            return_logprobs=True,
        ),
        stage,
        rollout_model=serving_model,
        batch_size=config["rollout_batch_size"],
    )

  train = make(
      actor,
      serving,
      config["train_temperature"],
      config["max_response_length"],
      "director-train",
  )
  evaluate = make(
      actor,
      serving,
      config["eval_temperature"],
      config["max_response_length"],
      "director-eval",
  )
  worker = make(
      frozen,
      frozen,
      config["worker_temperature"],
      config["worker_max_tokens"],
      "frozen-worker",
      prompt_limit=config.get("worker_max_prompt_length"),
  )

  def sync():
    train.sync()
    evaluate.sampler.transformer_state = train.sampler.transformer_state

  sync()
  return train, evaluate, worker, sync


def create_qwen_backends(config, mesh, model_path):
  """Return the same backend tuple as the existing Director recipe launcher."""
  validate_backend_config(config, model_path)
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tokenizer = AutoTokenizer.from_pretrained(
      str(model_path), local_files_only=True
  )
  actor = load_trainable_policy(config, mesh, model_path)
  frozen = load_base_policy(config, mesh, model_path)
  eos_tokens = shared_train._resolve_eos_tokens(Path(model_path), tokenizer.eos_token_id)  # pylint: disable=protected-access
  ds, es, ws, sync = make_native_samplers(
      actor, frozen, tokenizer, config, eos_tokens=eos_tokens
  )
  return (
      actor,
      frozen,
      ds,
      es,
      ws,
      math_workflow.chat_encoder(tokenizer, config["max_prompt_length"]),
      tokenizer.pad_token_id
      if tokenizer.pad_token_id is not None
      else tokenizer.eos_token_id,
      tokenizer.eos_token_id,
      sync,
  )
