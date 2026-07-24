# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Kubernetes deployment YAML manifest generator."""

import io
import os
import sys
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.experimental.distributed.deployment import yaml_generator


def _get_template_path(filename: str) -> str:
  # Locate template file relative to yaml_generator module directory
  path = os.path.join(
      os.path.dirname(yaml_generator.__file__),
      "yamls",
      filename,
  )
  return path


class YamlGeneratorTest(parameterized.TestCase):

  def test_generate_cpu_yaml(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-cpu-job",
        "--cpu_machine=n2-standard-64",
        "--worker_container_port=9999",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("test-cpu-job", rendered)
        self.assertIn("9999", rendered)

  @parameterized.named_parameters(
      (
          "tpu7x_small",
          "tpu7x:4x4x4",
          "test-tpu7x",
          "tpu7x",
      ),
      (
          "tpu7x_large",
          "tpu7x:4x4x8",
          "test-tpu7x-large",
          "tpu7x",
      ),
      (
          "tpuv5",
          "tpuv5:2x2x2",
          "test-tpuv5",
          "tpu-v5p-slice",
      ),
      (
          "tpuv5e",
          "tpuv5e:2x4",
          "test-tpuv5e",
          "tpu-v5-lite-podslice",
      ),
      (
          "tpuv6e",
          "tpuv6e:2x4",
          "test-tpuv6e",
          "tpu-v6e-slice",
      ),
      (
          "tpuv6ea",
          "tpuv6ea:2x4",
          "test-tpuv6ea",
          "tpu-v6ea-slice",
      ),
  )
  def test_generate_tpu_slice(
      self, tpu_slice, jobset_name, expected_accelerator
  ):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        f"--tpu_slice={tpu_slice}",
        f"--jobset_name={jobset_name}",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn(expected_accelerator, rendered)
        self.assertIn(jobset_name, rendered)

  @parameterized.named_parameters(
      (
          "unsupported_tpu_type",
          "unknown_tpu:4x4",
          ValueError,
      ),
      (
          "invalid_num_chips",
          "tpu7x:1x2",
          AssertionError,
      ),
  )
  def test_invalid_slice_raises(self, tpu_slice, expected_exception):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        f"--tpu_slice={tpu_slice}",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaises(expected_exception):
        yaml_generator.main()


if __name__ == "__main__":
  absltest.main()
