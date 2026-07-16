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
    bash examples/rl/grpo/gsm8k/run_gemma2_2b.sh

    # Agentic GRPO — DeepScaleR
    bash examples/deepscaler/run_deepscaler_disagg.sh

    # Agentic GRPO — DeepSWE
    python -m tunix.cli.grpo_main examples/deepswe/configs/qwen3_32b.yaml
"""
import dataclasses
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging
from tunix.cli import base_rl_pipeline
from tunix.cli.utils import data as data_lib


class GrpoPipeline(base_rl_pipeline.BasePipeline):
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

  @property
  def _default_training_mode(self):
    return "grpo"

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


def main(argv, **kwargs):
  pathways_bns = flags.FLAGS.pathways_bns
  if pathways_bns:
    base_rl_pipeline.setup_jax_pathways(pathways_bns)

  if "proxy" in os.getenv("JAX_PLATFORMS", ""):
    base_rl_pipeline.setup_pathways_on_cloud()

  pipeline = GrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching GRPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_trainer()

if __name__ == "__main__":
  app.run(main)
