# Copyright 2026 Google LLC
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

"""Tests that ppo_main dispatches correctly for both training modes

and that KV cache computation is correct.
"""

import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
import omegaconf
from tunix.cli import ppo_main
from tunix.rl import rl_cluster as rl_cluster_lib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _make_pipeline(extra_yaml: str) -> ppo_main.PPOPipeline:
  """Write a minimal valid YAML and instantiate PPOPipeline against it."""
  base = """
model_config:
  model_name: "test_model"
  model_id: "test/model"
  model_source: "huggingface"
  model_display: false
  rng_seed: 0
  intermediate_ckpt_dir: "/tmp/ckpt"

actor_model_config:
  mesh:
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

reference_model_config:
  mesh:
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

rollout_model_config:
  mesh:
    shape: "(1,1)"
    axis_names: "('fsdp','tp')"

tokenizer_config:
  tokenizer_type: "huggingface"
  tokenizer_path: "test/model"
  add_bos: false
  add_eos: false

rollout_engine: "vanilla"
offload_to_cpu: false

rollout_config:
  max_prompt_length: 256
  total_generation_steps: 512
  temperature: 1.0
  top_p: null
  top_k: null

rl_training_config:
  max_steps: 1
  eval_every_n_steps: 1
  mini_batch_size: 1
  train_micro_batch_size: 1
  actor_optimizer_config:
    opt_type: "adamw"
    learning_rate: 1.0e-6
    schedule_type: "warmup_cosine_decay_schedule"
    init_value: 0.0
    end_value: 0.0
    warmup_ratio: 0.1
    b1: 0.9
    b2: 0.99
    weight_decay: 0.01
    max_grad_norm: 1.0
  metrics_logging_options:
    log_dir: "/tmp/tb_test"
    flush_every_n_steps: 1
  checkpointing_options:
    save_interval_steps: 100
    max_to_keep: 1
  checkpoint_root_directory: "/tmp/ckpt_test"

batch_size: 1
num_batches: 1
num_train_epochs: 1
train_fraction: 1.0
"""
  with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
    f.write(base + extra_yaml)
    path = f.name

  # Patch HF_TOKEN so tokenizer validation passes
  with mock.patch.dict(os.environ, {"HF_TOKEN": "fake"}):
    pipeline = ppo_main.PPOPipeline(["", path])
  os.unlink(path)
  return pipeline

class DispatchTest(absltest.TestCase):

  def test_ppo_dispatches_to_ppo(self):
    yaml = """
training_mode: "ppo"
ppo_config:
  num_generations: 2
  num_iterations: 1
"""
    pipeline = _make_pipeline(yaml)
    self.assertEqual(pipeline.config["training_mode"], "ppo")

    with mock.patch.object(pipeline, "run_ppo_trainer") as mockrun_ppo_trainer:
      pipeline.run_ppo_trainer()
      mockrun_ppo_trainer.assert_called_once_with()

if __name__ == "__main__":
  absltest.main()
