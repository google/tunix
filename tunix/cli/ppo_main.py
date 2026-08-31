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

Set ``training_mode: "ppo"`` (default) for standard single-turn PPO.

Usage::

    # Standard PPO
    bash examples/rl/ppo/gsm8k/run_gemma2_2b.sh

"""

import os

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

  def _run(self, mode: str = "ppo"):
    """Execute PPO training (DeepScaleR, DeepSWE, etc.)."""
    self._setup_kubernetes()

    tokenizer = self._get_tokenizer()

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

    rl_engine = self.create_rl_engine(tokenizer)

    if mode == "ppo":
      from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top

      ppo_trainer = ppo_learner.PpoLearner(
          rl_engine=rl_engine,
          reward_fns=self.obtain_reward_fn(),
          algo_config=ppo_learner.PpoConfig(
              **self._config_mapping("ppo_config")
          ),
      )
      ppo_trainer.train(dataset)
      return
    elif mode == "agentic_ppo":
      raise ValueError("Agentic PPO is not yet supported.")
    else:
      raise ValueError(f"Unsupported training_mode {mode!r}")


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
  return


if __name__ == "__main__":
  app.run(main)
