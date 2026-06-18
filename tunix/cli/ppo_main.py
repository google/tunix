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

"""Main entry point for PPO training (standard).

Set ``training_mode: "ppo"`` (default) for standard single-turn ppo.

Usage::

    # Standard ppo
    python -m tunix.cli.ppo_main examples/rl/ppo/gsm8k/configs/gemma2_2b.yaml
"""

import dataclasses
import os

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
from tunix.models.gemma import model as gemma_lib
from tunix.perf.experimental import export as perf_export_v2
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

class PpoPipeline(BasePipeline):
  """Runs standard Ppo.

  ``training_mode: "ppo"`` (default) — standard single-turn PPO using
  PpoLearner.  All existing YAML configs continue to work unchanged.

  * role-specific ``*_model_config.mesh``: any role with an explicit mesh gets
    its own device slice; omitted meshes share the actor mesh by default.
  * role-specific ``same_mesh_as``: optional mesh sharing like
    ``reference_model_config.same_mesh_as: actor``.
  * ``sglang_jax_config`` / ``vllm_config``: engine-specific rollout params.
  * ``chat_parser_config.type``: ``"default"`` or ``"qwen"``.
  * ``data_module``: dotted module path; the module must expose
    ``create_dataset(**data_config) -> grain.MapDataset`` and optionally a
    ``batch_fn`` used as ``custom_batch_fn`` in post_init_dataset.
  * ``kubernetes_config``: optional Kubernetes env-var and kube-config setup.
  """

  def __init__(self, argv: list[str], **kwargs):
    super().__init__(argv, **kwargs)
  
  @property
  def _default_training_mode(self):
    return "ppo"

  # ------------------------------------------------------------------

  def _run(self, mode: str = "ppo"):
    """Execute standard PPO training (DeepScaleR, DeepSWE, etc.)."""
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

    rl_cluster = self.create_rl_cluster(tokenizer)

    if mode == "ppo":
      from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top

      ppo_trainer = ppo_learner.PpoLearner(
          rl_cluster=rl_cluster,
          reward_fns=self.obtain_reward_fn(),
          ppo_config=ppo_learner.PpoConfig(
              **self._config_mapping("ppo_config")
          ),
      )
      ppo_trainer.train(dataset)
      return

    else:
      raise ValueError(f"Unsupported training_mode {mode!r}")

def main(argv, **kwargs):
  if PATHWAYS_BNS.value:
    setup_jax_pathways(_PATHWAYS_BNS.value)

  if os.getenv("JAX_PLATFORMS") == "proxy":
    setup_pathways_on_cloud()

  pipeline = PpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching PPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_ppo_trainer()


if __name__ == "__main__":
  app.run(main)
