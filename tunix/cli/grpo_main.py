# Copyright 2025 Google LLC
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

"""Main entry point for GRPO training (standard and agentic).

Set ``training_mode: "grpo"`` (default) for standard single-turn GRPO, or
``training_mode: "agentic_grpo"`` for agentic multi-turn GRPO (DeepScaleR,
DeepSWE, etc.).

Usage::

    # Standard GRPO
    python -m tunix.cli.grpo_main examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml

    # Agentic GRPO — DeepScaleR
    bash examples/deepscaler/run_deepscaler_disagg.sh

    # Agentic GRPO — DeepSWE
    python -m tunix.cli.grpo_main examples/deepswe/configs/qwen3_32b.yaml
"""

from collections.abc import MutableMapping
import dataclasses
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.cli.base_rl_main import BasePipeline
from tunix.cli.base_rl_main import setup_jax_pathways
from tunix.cli.base_rl_main import setup_pathways_on_cloud
from tunix.cli.base_rl_main import PATHWAYS_BNS
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

class GrpoPipeline(BasePipeline):
  """Runs standard GRPO or agentic GRPO depending on ``training_mode``.

  ``training_mode: "grpo"`` (default) — standard single-turn GRPO using
  GrpoLearner.  All existing YAML configs continue to work unchanged.

  ``training_mode: "agentic_grpo"`` — multi-turn agentic GRPO using
  GRPOLearner.  Additional config sections are recognised:

  * ``agentic_grpo_config``: GRPOConfig fields (num_generations, beta, …)
    plus ``max_turns``, ``per_turn_timeout_secs``.
  * role-specific ``*_model_config.mesh``: any role with an explicit mesh gets
    its own device slice; omitted meshes share the actor mesh by default.
  * role-specific ``same_mesh_as``: optional mesh sharing like
    ``reference_model_config.same_mesh_as: actor``.
  * ``sglang_jax_config`` / ``vllm_config``: engine-specific rollout params.
  * ``chat_parser_config.type``: ``"default"`` or ``"qwen"``.
  * ``agent_class_path`` / ``env_class_path``: dotted Python paths to load
    agent and env classes dynamically.
  * ``data_module``: dotted module path; the module must expose
    ``create_dataset(**data_config) -> grain.MapDataset`` and optionally a
    ``batch_fn`` used as ``custom_batch_fn`` in post_init_dataset.
  * ``kubernetes_config``: optional Kubernetes env-var and kube-config setup.
  """

  def __init__(self, argv: list[str], **kwargs):
    super().__init__(argv, **kwargs)

  # ------------------------------------------------------------------
  # Rollout config
  # ------------------------------------------------------------------

  def create_rollout_config(
      self,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh] | None = None,
  ) -> base_rollout.RolloutConfig:
    """Build RolloutConfig from YAML.

    Standard mode: pass rollout_config fields through with kv_cache_size =
    max_prompt_length + total_generation_steps + 256.

    Agentic mode: same base. Same kv_cache_size calculation.

    Engine-specific extras (sglang_jax_config, vllm_config) are also applied.

    Args:
      role_to_mesh: Optional mapping from logical role to JAX mesh.

    Returns:
      The constructed RolloutConfig.
    """
    rollout_cfg = self._config_mapping("rollout_config")
    mode = self._config_string("training_mode", "grpo")
    engine = self._config_string("rollout_engine", "vanilla")

    valid_fields = {
        f.name for f in dataclasses.fields(base_rollout.RolloutConfig)
    }

    # Base pass-through (same as original create_rollout_config)
    filtered = {k: v for k, v in rollout_cfg.items() if k in valid_fields}
    if "total_generation_steps" in rollout_cfg:
      filtered["max_tokens_to_generate"] = rollout_cfg["total_generation_steps"]

    max_prompt = rollout_cfg.get("max_prompt_length", 0)
    max_response = rollout_cfg.get("total_generation_steps", 0)

    kv_cache_size = 0
    if mode == "agentic_grpo":
      agentic_cfg = self._config_mapping("agentic_grpo_config")
      kv_cache_size = max_prompt + max_response + 256
      filtered["kv_cache_size"] = kv_cache_size
      logging.info("kv_cache_size: %d", kv_cache_size)

      max_running_requests = agentic_cfg.get("max_concurrency", 16)
    else:
      grpo_cfg = self._config_mapping("grpo_config")
      # Standard: kv_cache_size = max_prompt + max_response + 256
      if max_prompt and max_response:
        kv_cache_size = max_prompt + max_response + 256
        filtered["kv_cache_size"] = kv_cache_size
      # Defaults to global batch size * num_generations to allow full
      # concurrency.
      max_running_requests = self.config.get("batch_size", 1) * grpo_cfg.get(
          "num_generations", 1
      )

    # Engine-specific extras
    extra = self._rollout_engine_extra(
        engine,
        kv_cache_size,
        max_running_requests,
        role_to_mesh=role_to_mesh,
    )
    filtered.update({k: v for k, v in extra.items() if k in valid_fields})
    return base_rollout.RolloutConfig(**filtered)

  # ------------------------------------------------------------------
  # Standard GRPO helpers (unchanged)
  # ------------------------------------------------------------------

  def create_rl_cluster(self, tokenizer):
    role_to_mesh = self.create_role_to_mesh()
    rollout_config = self.create_rollout_config(role_to_mesh=role_to_mesh)
    reference_model_config = self._mutable_config_mapping(
        "reference_model_config"
    )
    actor_model_config = self._mutable_config_mapping("actor_model_config")
    tokenizer_config = self._config_mapping("tokenizer_config")
    # Should not use LoRA for reference model.
    if reference_model_config.get("lora_config"):
      logging.warning(
          "LoRA config is not supported for the reference model. Disabling"
          " LoRA."
      )
      del reference_model_config["lora_config"]
    reference_model, _ = model_lib.create_model(
        dict(reference_model_config),
        tokenizer_config,
        role_to_mesh[rl_cluster_lib.Role.REFERENCE],
    )
    if actor_model_config.get("lora_config", None):
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          role_to_mesh[rl_cluster_lib.Role.ACTOR],
          actor_model_config["lora_config"],
      )
    else:
      graph_def, params = nnx.split(reference_model)
      actor_model = nnx.merge(
          graph_def,
          jax.tree.map(jnp.copy, params),
      )

    cluster_config = self.create_cluster_config(
        role_to_mesh=role_to_mesh,
        rollout_config=rollout_config,
    )
    perf_config = self.create_perf_config(cluster_config)
    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )

  def compute_params(self, dataset):
    rl_training_config = self._mutable_config_mapping("rl_training_config")

    # Return early if max_steps is already specified.
    max_steps = None
    if rl_training_config.get("max_steps"):
      max_steps = rl_training_config.get("max_steps")
    elif not hasattr(dataset, "__len__"):
      raise ValueError(
          "max_steps must be specified since the dataset length cannot be"
          " determined."
      )

    dataset_length = len(dataset)

    batch_size = self.config.get("batch_size", 1)
    num_batches = self.config.get("num_batches")
    if not num_batches:
      num_batches = dataset_length // batch_size
      self.config["num_batches"] = num_batches
      logging.info(
          "Dynamically computed num_batches=%d with batch_size=%d",
          num_batches,
          batch_size,
      )
    self.config["num_batches"] = num_batches
    num_train_epochs = self.config.get("num_train_epochs")
    if not num_train_epochs:
      num_train_epochs = 1

    train_fraction = self.config.get("train_fraction")
    if not train_fraction:
      train_fraction = 0.8
    elif train_fraction <= 0.0 and train_fraction > 1.0:
      logging.warning(
          "train_fraction %.2f out of expected range. Setting to 0.8",
          train_fraction,
      )
      train_fraction = 0.8

    allowed_max_steps = int(num_batches * num_train_epochs * train_fraction)
    if not max_steps:
      max_steps = allowed_max_steps
    elif max_steps > allowed_max_steps:
      raise ValueError(
          f"Maximum allowed value for max_steps is {allowed_max_steps}, but"
          f" {max_steps} is specified."
      )

    rl_training_config["max_steps"] = max_steps
    self._apply_optimizer_step_limits(
        rl_training_config, "actor_optimizer_config", max_steps
    )
    logging.info(
        "Dynamically computed max_steps=%d based on dataset length %d",
        max_steps,
        dataset_length,
    )

  # ------------------------------------------------------------------
  # Agentic GRPO helpers
  # ------------------------------------------------------------------

  def _create_agentic_grpo_config(self):
    """Build GRPOConfig (agentic) from the agentic_grpo_config YAML section."""
    from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig  # pylint: disable=g-import-not-at-top

    cfg = dict(self._config_mapping("agentic_grpo_config"))

    # episode_timeout = per_turn_timeout_secs * max_turns when not explicit
    if "episode_timeout" not in cfg:
      per_turn = cfg.pop("per_turn_timeout_secs", None)
      max_turns = cfg.get("max_turns", 1)
      if per_turn is not None:
        cfg["episode_timeout"] = per_turn * max_turns

    # max_response_length mirrors rollout_config.total_generation_steps
    if "max_response_length" not in cfg:
      cfg["max_response_length"] = self._config_mapping("rollout_config").get(
          "total_generation_steps", 8192
      )

    # Strip helper keys that are not GRPOConfig fields
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    cfg.pop("max_turns", None)
    return GRPOConfig(**{k: v for k, v in cfg.items() if k in valid})

  # ------------------------------------------------------------------
  # Agentic GRPO training
  # ------------------------------------------------------------------

  def _run(self, mode: str = "grpo"):
    """Execute agentic GRPO training (DeepScaleR, DeepSWE, etc.)."""
    self._setup_kubernetes()

    tokenizer = self._get_tokenizer()

    chat_parser = self._create_chat_parser(tokenizer)

    raw_dataset, custom_batch_fn = self._load_raw_dataset(tokenizer)

    self.compute_params(raw_dataset)

    dataset, _ = data_lib.post_init_dataset(
        raw_dataset,
        tokenizer,
        batch_size=self.config.get("batch_size", 1),
        num_batches=self.config.get("num_batches"),
        max_prompt_length=self._config_mapping("rollout_config").get(
            "max_prompt_length"
        ),
        fraction=self.config.get("train_fraction", 1.0),
        num_epochs=self.config.get("num_train_epochs", 1),
        prompt_key=self.config.get("prompt_key", "prompts"),
        custom_batch_fn=custom_batch_fn,
    )

    rl_cluster = self.create_rl_cluster(tokenizer)

    if mode == "grpo":
      from tunix.rl.grpo import grpo_learner  # pylint: disable=g-import-not-at-top

      grpo_trainer = grpo_learner.GrpoLearner(
          rl_cluster=rl_cluster,
          reward_fns=self.obtain_reward_fn(),
          algo_config=grpo_learner.GrpoConfig(
              **self._config_mapping("grpo_config")
          ),
      )
      grpo_trainer.train(dataset)
      return

    # agentic GRPO
    if mode != "agentic_grpo":
      raise ValueError(f"Unsupported training_mode {mode!r}")

    from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner  # pylint: disable=g-import-not-at-top

    algo_config = self._create_agentic_grpo_config()

    reward_fns = (
        self.obtain_reward_fn() if self.config.get("reward_functions") else None
    )

    learner_kwargs: dict[str, Any] = dict(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        chat_parser=chat_parser,
    )

    agent_class_path = self._config_string("agent_class_path")
    if agent_class_path:
      learner_kwargs["agent_class"] = self._load_class_from_path(
          agent_class_path
      )
      learner_kwargs["agent_kwargs"] = dict(
          self.config.get("agent_kwargs") or {}
      )

    env_class_path = self._config_string("env_class_path")
    if env_class_path:
      learner_kwargs["env_class"] = self._load_class_from_path(env_class_path)
      learner_kwargs["env_kwargs"] = dict(self.config.get("env_kwargs") or {})

    logging.info("Starting agentic GRPO training...")
    GRPOLearner(**learner_kwargs).train(dataset)

  # ------------------------------------------------------------------
  # Dispatcher
  # ------------------------------------------------------------------

  def run_grpo_trainer(self):
    """Dispatch to standard or agentic GRPO based on training_mode."""
    mode = self.config.get("training_mode", "grpo")
    self._run(mode=mode)


def main(argv, **kwargs):
  if PATHWAYS_BNS.value:
    setup_jax_pathways(_PATHWAYS_BNS.value)

  if os.getenv("JAX_PLATFORMS") == "proxy":
    setup_pathways_on_cloud()

  pipeline = GrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching GRPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
