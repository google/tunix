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

"""Main entry point for GRPO training."""
from absl import app
from absl import flags
import jax
from tunix.cli import config
from tunix.cli.utils import model as model_lib
from tunix.examples.data import math_dataset as data_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.grpo.grpo_learner import GrpoConfig
from tunix.rl.rollout import base_rollout

_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class GrpoPipeline(config.HyperParameters):
  """Class for running the GRPO trainer."""

  def create_rollout_config(self):
    rollout_config = self.config["rollout_config"]
    return base_rollout.RolloutConfig(
        max_tokens_to_generate=rollout_config["total_generation_steps"],
        max_prompt_length=rollout_config["max_prompt_length"],
        kv_cache_size=rollout_config["max_prompt_length"]
        + rollout_config["total_generation_steps"]
        + 256,
        temperature=rollout_config["temperature"],
        top_p=rollout_config["top_p"],
        top_k=rollout_config["top_k"],
    )

  def create_role_to_mesh(self):
    default_mesh = self.create_mesh("actor_model_config")
    actor_mesh = reference_mesh = rollout_mesh = default_mesh
    if "reference_model_config" in self.config:
      reference_mesh = self.create_mesh("reference_model_config")
    if "rollout_model_config" in self.config:
      rollout_mesh = self.create_mesh("rollout_model_config")
    return {
        rl_cluster_lib.Role.ACTOR: actor_mesh,
        rl_cluster_lib.Role.REFERENCE: reference_mesh,
        rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
    }

  def create_cluster_config(self):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh=self.create_role_to_mesh(),
        rollout_engine=self.config["rollout_engine"],
        offload_to_cpu=self.config["offload_to_cpu"],
        training_config=self.create_rl_training_config(),
        rollout_config=self.create_rollout_config(),
    )

  def create_rl_training_config(self):
    base_key = "rl_training_config"
    constructed_rl_training_config = self.obtain_training_config_dict(base_key)

    base_config = self.config[base_key]
    if base_config.get("actor_optimizer_config"):
      constructed_rl_training_config["actor_optimizer"] = self.create_optimizer(
          base_key, "actor_optimizer_config"
      )
    if base_config.get("critic_optimizer_config"):
      constructed_rl_training_config["critic_optimizer"] = (
          self.create_optimizer(base_key, "critic_optimizer_config")
      )

    return rl_cluster_lib.RLTrainingConfig(**constructed_rl_training_config)

  def create_rl_cluster(self):
    reference_model, tokenizer_path = model_lib.create_model(
        self.config["reference_model_config"],
        self.config["tokenizer_config"],
        self.create_mesh("reference_model_config"),
    )
    if self.config["actor_model_config"].get("lora_config", None):
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          self.create_mesh("actor_model_config"),
          self.config["actor_model_config"]["lora_config"],
      )
    else:
      actor_model = reference_model

    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"], tokenizer_path
    )

    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=self.create_cluster_config(),
    )

  def run_grpo_trainer(self):
    grpo_trainer = grpo_learner.GrpoLearner(
        rl_cluster=self.create_rl_cluster(),
        reward_fns=self.obtain_reward_fn(),
        algo_config=GrpoConfig(**self.config["grpo_config"]),
    )

    if self.config["data_source"] == "local":
      dataset = data_lib.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["data_directory"],
          batch_size=self.config["batch_size"],
          num_batches=None,
          tokenizer=grpo_trainer.rl_cluster.tokenizer.tokenizer,
      )
    else:
      dataset = data_lib.create_dataset(
          data_source="tfds",
          dataset=self.config["dataset_name"],
          batch_size=self.config["batch_size"],
          num_batches=self.config["num_batches"],
          tfds_download=self.config["tfds_download"],
      )

    mesh = self.create_mesh("actor_model_config")
    with mesh:
      grpo_trainer.train(dataset)


def _setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)
  pipeline = GrpoPipeline(argv, **kwargs)
  pipeline.run_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
