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

import collections
import types

from absl.testing import absltest
from absl.testing import parameterized
from tunix.models import maxtext_parallelism


class MaxTextPipelineConfigTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="pp2_tp4_qwen3_8b",
          pipeline_parallelism=2,
          tensor_parallelism=4,
          layers_per_stage=18,
          microbatches=4,
          expected_shapes=(1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1),
      ),
      dict(
          testcase_name="pp4_tp2_qwen3_8b",
          pipeline_parallelism=4,
          tensor_parallelism=2,
          layers_per_stage=9,
          microbatches=8,
          expected_shapes=(1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 1),
      ),
  )
  def test_hybrid_layout(
      self,
      pipeline_parallelism,
      tensor_parallelism,
      layers_per_stage,
      microbatches,
      expected_shapes,
  ):
    config = maxtext_parallelism.MaxTextPipelineConfig(
        pipeline_parallelism=pipeline_parallelism,
        tensor_parallelism=tensor_parallelism,
        num_layers_per_pipeline_stage=layers_per_stage,
        num_pipeline_microbatches=microbatches,
        pipeline_parallel_layers=36,
    )

    self.assertEqual(config.required_device_count, 8)
    self.assertEqual(config.mesh_axis_shapes, expected_shapes)
    self.assertEqual(
        config.as_maxtext_kwargs()["ici_pipeline_parallelism"],
        pipeline_parallelism,
    )
    self.assertEqual(
        config.as_maxtext_kwargs()["ici_tensor_parallelism"],
        tensor_parallelism,
    )
    config.validate_batch_size(16)

  def test_validate_exact_mesh(self):
    config = maxtext_parallelism.MaxTextPipelineConfig(
        pipeline_parallelism=2,
        tensor_parallelism=4,
        num_layers_per_pipeline_stage=18,
        num_pipeline_microbatches=4,
    )
    mesh = types.SimpleNamespace(
        shape=collections.OrderedDict(
            zip(
                maxtext_parallelism.MAXTEXT_MESH_AXIS_NAMES,
                config.mesh_axis_shapes,
            )
        )
    )

    config.validate_mesh(mesh)

  def test_rejects_tunix_tp_axis_names(self):
    config = maxtext_parallelism.MaxTextPipelineConfig(
        pipeline_parallelism=2,
        tensor_parallelism=4,
    )
    mesh = types.SimpleNamespace(
        shape=collections.OrderedDict((("fsdp", 1), ("tp", 8)))
    )

    with self.assertRaisesRegex(ValueError, "Missing axes"):
      config.validate_mesh(mesh)

  @parameterized.named_parameters(
      dict(
          testcase_name="one_pipeline_stage",
          kwargs={"pipeline_parallelism": 1},
          error="at least 2",
      ),
      dict(
          testcase_name="microbatches_not_divisible",
          kwargs={
              "pipeline_parallelism": 4,
              "num_pipeline_microbatches": 6,
          },
          error="must be divisible",
      ),
      dict(
          testcase_name="layers_not_divisible",
          kwargs={
              "pipeline_parallelism": 4,
              "num_layers_per_pipeline_stage": 3,
              "pipeline_parallel_layers": 28,
          },
          error="must be divisible",
      ),
      dict(
          testcase_name="conflicting_fsdp_modes",
          kwargs={
              "pipeline_parallelism": 2,
              "pipeline_fsdp_ag_once": True,
              "pipeline_fsdp_ag_per_repeat": True,
          },
          error="mutually exclusive",
      ),
  )
  def test_invalid_config(self, kwargs, error):
    with self.assertRaisesRegex(ValueError, error):
      maxtext_parallelism.MaxTextPipelineConfig(**kwargs)

  def test_rejects_incompatible_batch_size(self):
    config = maxtext_parallelism.MaxTextPipelineConfig(
        pipeline_parallelism=4,
        num_pipeline_microbatches=8,
    )

    with self.assertRaisesRegex(ValueError, "global_batch_size"):
      config.validate_batch_size(12)


if __name__ == "__main__":
  absltest.main()
