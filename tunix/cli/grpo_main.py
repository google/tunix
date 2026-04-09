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

import dataclasses
import os
from typing import Any
from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from tunix.cli import config
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.examples.data import math_dataset as example_data
from tunix.perf import export as perf_export
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental import export as perf_export_v2
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout

GrpoConfig = grpo_learner.GrpoConfig

_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class GrpoPipeline(config.HyperParameters):
  """Class for running the GRPO trainer."""

  def create_rollout_config(self):
    rollout_config = self.config["rollout_config"]

    # Get all valid field names from RolloutConfig
    valid_fields = {
        f.name for f in dataclasses.fields(base_rollout.RolloutConfig)
    }

    # Filter rollout_config to only include valid keys
    filtered_config = {
        k: v for k, v in rollout_config.items() if k in valid_fields
    }

    # Apply explicit recomputed/renamed values
    if "total_generation_steps" in rollout_config:
      filtered_config["max_tokens_to_generate"] = rollout_config[
          "total_generation_steps"
      ]
    if (
        "max_prompt_length" in rollout_config
        and "total_generation_steps" in rollout_config
    ):
      filtered_config["kv_cache_size"] = (
          rollout_config["max_prompt_length"]
          + rollout_config["total_generation_steps"]
          + 256
      )

    return base_rollout.RolloutConfig(**filtered_config)

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

  def create_perf_config(self, cluster_config: rl_cluster_lib.ClusterConfig):
    perf_metrics_options = cluster_config.training_config.perf_metrics_options
    if not perf_metrics_options:
      return None

    perf_config = perf_metrics.PerfMetricsConfig()

    if perf_metrics_options.enable_perf_v1:
      custom_export_fn_path = perf_metrics_options.custom_export_fn_path
      if custom_export_fn_path:
        perf_config.custom_export_fn = self._get_function_from_path(
            custom_export_fn_path
        )
        if perf_config.custom_export_fn is None:
          raise ValueError(
              "Could not load custom export function from"
              f" {custom_export_fn_path}"
          )
      else:
        perf_config.custom_export_fn = (
            perf_export.PerfMetricsExport.from_cluster_config(cluster_config)
        )

    if perf_metrics_options.enable_perf_v2:
      custom_export_fn_path_v2 = perf_metrics_options.custom_export_fn_path_v2
      if custom_export_fn_path_v2:
        perf_config.custom_export_fn_v2 = self._get_function_from_path(
            custom_export_fn_path_v2
        )
        if perf_config.custom_export_fn_v2 is None:
          raise ValueError(
              "Could not load custom export function v2 from"
              f" {custom_export_fn_path_v2}"
          )
      else:
        perf_config.custom_export_fn_v2 = (
            perf_export_v2.PerfMetricsExport.from_cluster_config(
                cluster_config=cluster_config,
                enable_trace_writer=perf_metrics_options.enable_trace_writer,
                trace_dir=perf_metrics_options.trace_dir,
            ).export_metrics
        )
    return perf_config

  def create_rl_cluster(self, tokenizer):
    # Should not use LoRA for reference model.
    if self.config["reference_model_config"].get("lora_config"):
      logging.warning(
          "LoRA config is not supported for the reference model. Disabling"
          " LoRA."
      )
      del self.config["reference_model_config"]["lora_config"]
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
      graph_def, params = nnx.split(reference_model)
      actor_model = nnx.merge(
          graph_def,
          jax.tree.map(jnp.copy, params),
      )

    cluster_config = self.create_cluster_config()
    perf_config = self.create_perf_config(cluster_config)
    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )

  def compute_params(self, dataset):
    rl_training_config: dict[str, Any] = self.config.get(
        "rl_training_config", {}
    )

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
      logging.info(
          "Dynamically computed num_batches=%d with batch_size=%d",
          num_batches,
          batch_size,
      )
    num_train_epochs = self.config.get("num_train_epochs")
    if not num_train_epochs:
      num_train_epochs = 1

    train_fraction = self.config.get("train_fraction")
    if not train_fraction:
      train_fraction = 0.8
    elif train_fraction <= 0.0 and train_fraction > 1.0:
      logging.warning(
          f"train_fraction {train_fraction:.2f} out of expected range. Setting"
          " to 0.8"
      )
      train_fraction = 0.8

    allowed_max_steps = int(num_batches * num_train_epochs * train_fraction)
    if not max_steps:
      max_steps = allowed_max_steps
    elif max_steps > allowed_max_steps:
      raise ValueError(
          "Maximum allowed value for max_steps is %d", allowed_max_steps
      )

    rl_training_config["max_steps"] = max_steps
    actor_opt: dict[str, Any] = rl_training_config.get(
        "actor_optimizer_config", {}
    )
    if actor_opt and not actor_opt.get("decay_steps"):
      actor_opt["decay_steps"] = max_steps
    if actor_opt and not actor_opt.get("warmup_steps"):
      warmup_ratio = self.config.get("warmup_ratio", 0.1)
      warmup_steps = self.config.get("warmup_steps", warmup_ratio * max_steps)
      actor_opt["warmup_steps"] = warmup_steps
    logging.info(
        "Dynamically computed max_steps=%d based on dataset length %d",
        max_steps,
        dataset_length,
    )

  def run_grpo_trainer(self):
    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
    )

    if self.config.get("data_module", None):
      dataset = data_lib.get_dataset_from_module(
          self.config["data_module"],
          tokenizer,
      )
    elif self.config["data_source"] == "local":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["data_directory"],
          tokenizer=tokenizer,
      )
    else:
      dataset = example_data.create_dataset(
          data_source="tfds",
          dataset=self.config["dataset_name"],
          tfds_download=self.config["tfds_download"],
      )
    self.compute_params(dataset)
    dataset, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=self.config["batch_size"],
        num_batches=self.config.get("num_batches", None),
        max_prompt_length=self.config["rollout_config"].get(
            "max_prompt_length", None
        ),
    )
    rl_cluster = self.create_rl_cluster(tokenizer)
    grpo_trainer = grpo_learner.GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=self.obtain_reward_fn(),
        algo_config=GrpoConfig(**self.config["grpo_config"]),
    )
    grpo_trainer.train(dataset)


def _setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def _setup_pathways_on_cloud():
  import pathwaysutils  # pylint: disable=g-import-not-at-top

  pathwaysutils.initialize()


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)

  if os.getenv("JAX_PLATFORMS") == "proxy":
    _setup_pathways_on_cloud()

  pipeline = GrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching GRPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
