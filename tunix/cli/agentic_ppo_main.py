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

"""Main entry point for PPO training (standard and agentic).

Set ``training_mode: "ppo"`` (default) for standard single-turn PPO, or
``training_mode: "agentic_ppo"`` for agentic multi-turn PPO (DeepScaleR,
DeepSWE, etc.).

Usage::

    # Standard PPO
    bash examples/rl/ppo/gsm8k/run_gemma2_2b.sh

"""
import dataclasses
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from tunix.cli import base_rl_pipeline
from tunix.cli.utils import data as data_lib


class PpoPipeline(base_rl_pipeline.BasePipeline):
  """Runs standard PPO or agentic PPO depending on ``training_mode``.

  ``training_mode: "ppo"`` (default) — standard single-turn PPO using
  PpoLearner.  All existing YAML configs continue to work unchanged.

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

  @property
  def _default_training_mode(self):
    return "ppo"

  def _create_agentic_config(self):
    """Build PPOConfig (agentic) from the agentic_ppo_config YAML section."""
    from tunix.rl.agentic.agentic_ppo_learner import PPOConfig # pylint: disable=g-import-not-at-top

    cfg = dict(self._config_mapping("agentic_ppo_config"))

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

    # Strip helper keys that are not PPOConfig fields
    valid = {f.name for f in dataclasses.fields(PPOConfig)}
    cfg.pop("max_turns", None)
    return PPOConfig(**{k: v for k, v in cfg.items() if k in valid})

  def _run(self, mode: str = "ppo"):
    """Execute PPO training (DeepScaleR, DeepSWE, etc.)."""
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

    if mode == "ppo":
      from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top

      ppo_trainer = ppo_learner.PpoLearner(
          rl_cluster=rl_cluster,
          reward_fns=self.obtain_reward_fn(),
          algo_config=ppo_learner.PpoConfig(
              **self._config_mapping("ppo_config")
          ),
      )
      ppo_trainer.train(dataset)
      return
    else: 
      raise ValueError(f"Unsupported training_mode {mode!r}")

    # agentic PPO
    if mode != "agentic_ppo":
      raise ValueError(f"Unsupported training_mode {mode!r}")

    from tunix.rl.agentic.agentic_ppo_learner import PPOLearner  # pylint: disable=g-import-not-at-top

    algo_config = self._create_agentic_config()

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

    logging.info("Starting agentic PPO training...")
    PPOLearner(**learner_kwargs).train(dataset)


def main(argv, **kwargs):
  pathways_bns = flags.FLAGS.pathways_bns
  if pathways_bns:
    base_rl_pipeline.setup_jax_pathways(pathways_bns)

  if os.getenv("JAX_PLATFORMS") == "proxy":
    base_rl_pipeline.setup_pathways_on_cloud()

  pipeline = PpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching PPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_trainer()

if __name__ == "__main__":
  app.run(main)
