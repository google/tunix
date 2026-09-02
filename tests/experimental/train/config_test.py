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

"""Unit tests for experimental TrainingConfig."""

import copy
import dataclasses
from absl.testing import absltest
from tunix.experimental.train import config as train_config

TrainingConfig = train_config.TrainingConfig


class TrainingConfigTest(absltest.TestCase):

  def test_defaults(self):
    cfg = TrainingConfig()
    self.assertEqual(cfg.eval_every_n_steps, 0)
    self.assertIsNone(cfg.max_steps)
    self.assertIsNone(cfg.gradient_accumulation_steps)
    self.assertEqual(cfg.max_inflight_computations, 2)
    self.assertIsNone(cfg.checkpoint_root_directory)
    self.assertIsNone(cfg.checkpointing_options)
    self.assertIsNone(cfg.metrics_logging_options)
    self.assertIsNone(cfg.profiler_options)
    self.assertIsNone(cfg.perf_metrics_options)
    self.assertEqual(cfg.data_sharding_axis, ("fsdp",))
    self.assertEqual(cfg.metrics_prefix, "")
    self.assertEqual(cfg.pbar_description, "Training")
    self.assertIsNone(cfg.max_seq_token_per_tpu)
    self.assertIsNone(cfg.max_segments_per_packed_row)
    self.assertEqual(cfg.engine_kwargs, {})

  def test_standard_fields_initialization(self):
    cfg = TrainingConfig(
        eval_every_n_steps=5,
        max_steps=100,
        gradient_accumulation_steps=4,
        max_inflight_computations=3,
        checkpoint_root_directory="/tmp/ckpts",
        metrics_prefix="actor",
        pbar_description="Actor Training",
        data_sharding_axis=("fsdp", "tp"),
        max_seq_token_per_tpu=2048,
        max_segments_per_packed_row=16,
    )
    self.assertEqual(cfg.eval_every_n_steps, 5)
    self.assertEqual(cfg.max_steps, 100)
    self.assertEqual(cfg.gradient_accumulation_steps, 4)
    self.assertEqual(cfg.max_inflight_computations, 3)
    self.assertEqual(cfg.checkpoint_root_directory, "/tmp/ckpts")
    self.assertEqual(cfg.metrics_prefix, "actor")
    self.assertEqual(cfg.pbar_description, "Actor Training")
    self.assertEqual(cfg.data_sharding_axis, ("fsdp", "tp"))
    self.assertEqual(cfg.max_seq_token_per_tpu, 2048)
    self.assertEqual(cfg.max_segments_per_packed_row, 16)

  def test_extra_kwargs_auto_captured_into_engine_kwargs(self):
    cfg = TrainingConfig(
        eval_every_n_steps=10,
        model_name="llama3.1-8b",
        scan_layers=True,
        logical_axis_rules=(("act", "fsdp"), ("weight", "tp")),
        custom_engine_flag=42,
    )
    self.assertEqual(cfg.eval_every_n_steps, 10)
    self.assertIn("model_name", cfg.engine_kwargs)
    self.assertEqual(cfg.engine_kwargs["model_name"], "llama3.1-8b")
    self.assertEqual(cfg.engine_kwargs["scan_layers"], True)
    self.assertEqual(
        cfg.engine_kwargs["logical_axis_rules"],
        (("act", "fsdp"), ("weight", "tp")),
    )
    self.assertEqual(cfg.engine_kwargs["custom_engine_flag"], 42)

  def test_explicit_engine_kwargs_dict(self):
    cfg = TrainingConfig(
        eval_every_n_steps=10,
        engine_kwargs={
            "model_name": "qwen3-1.7b",
            "weight_dtype": "bfloat16",
        },
        scan_layers=False,  # Should merge into engine_kwargs
    )
    self.assertEqual(cfg.engine_kwargs["model_name"], "qwen3-1.7b")
    self.assertEqual(cfg.engine_kwargs["weight_dtype"], "bfloat16")
    self.assertEqual(cfg.engine_kwargs["scan_layers"], False)

  def test_get_and_get_with_default(self):
    cfg = TrainingConfig(
        eval_every_n_steps=5,
        max_steps=None,
        model_name="qwen3",
    )
    # Standard field with value
    self.assertEqual(cfg.get("eval_every_n_steps"), 5)
    # Standard field with None -> fallback to default
    self.assertEqual(cfg.get("max_steps", default=100), 100)
    self.assertEqual(cfg.get_with_default("max_steps", 100), 100)
    # Engine kwarg
    self.assertEqual(cfg.get("model_name"), "qwen3")
    self.assertEqual(cfg.get_with_default("model_name", "fallback"), "qwen3")
    # Non-existent field
    self.assertIsNone(cfg.get("non_existent"))
    self.assertEqual(cfg.get("non_existent", default="foo"), "foo")
    self.assertEqual(cfg.get_with_default("non_existent", "foo"), "foo")

  def test_attribute_access_for_engine_kwargs(self):
    cfg = TrainingConfig(model_name="gemma-3", scan_layers=True)
    self.assertEqual(cfg.model_name, "gemma-3")
    self.assertTrue(cfg.scan_layers)
    with self.assertRaises(AttributeError):
      _ = cfg.undefined_attr

  def test_to_dict_merging(self):
    cfg = TrainingConfig(
        eval_every_n_steps=10,
        max_steps=50,
        model_name="qwen3",
        learning_rate=1e-4,
    )
    d = cfg.to_dict()
    self.assertEqual(d["eval_every_n_steps"], 10)
    self.assertEqual(d["max_steps"], 50)
    self.assertEqual(d["model_name"], "qwen3")
    self.assertEqual(d["learning_rate"], 1e-4)

  def test_deepcopy_and_dataclass_replace(self):
    cfg = TrainingConfig(
        eval_every_n_steps=10,
        model_name="qwen3",
        extra_list=[1, 2, 3],
    )
    copied = copy.deepcopy(cfg)
    self.assertEqual(copied.eval_every_n_steps, 10)
    self.assertEqual(copied.model_name, "qwen3")
    self.assertEqual(copied.extra_list, [1, 2, 3])

    replaced = dataclasses.replace(
        cfg,
        eval_every_n_steps=20,
        engine_kwargs={"model_name": "qwen3-7b", "extra_list": [1, 2, 3]},
    )
    self.assertEqual(replaced.eval_every_n_steps, 20)
    self.assertEqual(replaced.model_name, "qwen3-7b")


if __name__ == "__main__":
  absltest.main()
