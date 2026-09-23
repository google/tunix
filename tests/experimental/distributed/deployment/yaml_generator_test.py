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
          f"{tpl_label}_{name}",
          tpl_file,
          tpu_slice,
          jobset_name,
          expected_accelerator,
      )
      for tpl_label, tpl_file in (
          ("pathways", "jobset.pathways.yaml"),
          ("mcjax", "jobset.mcjax.yaml"),
      )
      for name, tpu_slice, jobset_name, expected_accelerator in (
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
              "tpuv5p",
              "tpuv5p:2x2x2",
              "test-tpuv5p",
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
  )
  def test_generate_tpu_slice(
      self, template_name, tpu_slice, jobset_name, expected_accelerator
  ):
    template_file = _get_template_path(template_name)
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
      ("pathways", "jobset.pathways.yaml"),
      ("mcjax", "jobset.mcjax.yaml"),
  )
  def test_generate_worker_container_options(self, template_name):
    template_file = _get_template_path(template_name)
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-job",
        "--tpu_slice=tpuv6e:2x4",
        "--worker_container_port=9999",
        "--worker_container_name=test-worker",
        "--worker_container_image=test-image:latest",
        "--worker_startup_command=echo hello",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("test-job", rendered)
        self.assertIn("9999", rendered)
        self.assertIn("test-worker", rendered)
        self.assertIn("test-image:latest", rendered)
        self.assertIn("echo hello", rendered)

  @parameterized.named_parameters(
      (
          f"{tpl_label}_{name}",
          tpl_file,
          tpu_slice,
          expected_exception,
      )
      for tpl_label, tpl_file in (
          ("pathways", "jobset.pathways.yaml"),
          ("mcjax", "jobset.mcjax.yaml"),
      )
      for name, tpu_slice, expected_exception in (
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
  )
  def test_invalid_slice_raises(
      self, template_name, tpu_slice, expected_exception
  ):
    template_file = _get_template_path(template_name)
    argv = [
        "yaml_generator.py",
        template_file,
        f"--tpu_slice={tpu_slice}",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaises(expected_exception):
        yaml_generator.main()

  def test_generate_yaml_with_queue_name(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-queue-job",
        "--cpu_machine=n2-standard-64",
        "--queue_name=test-local-queue",
        "--namespace=test-namespace",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("kueue.x-k8s.io/queue-name: test-local-queue", rendered)
        self.assertIn("namespace: test-namespace", rendered)

  def test_generate_yaml_default_namespace(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-default-ns-job",
        "--cpu_machine=n2-standard-64",
    ]
    env = dict(os.environ)
    env.pop("K8S_NAMESPACE", None)
    with mock.patch.dict(os.environ, env, clear=True):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()
          self.assertIn("namespace: default", rendered)

  def test_generate_yaml_empty_namespace(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-empty-ns-job",
        "--cpu_machine=n2-standard-64",
    ]
    with mock.patch.dict(os.environ, {"K8S_NAMESPACE": ""}, clear=False):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()
          self.assertIn("namespace: \n", rendered)

  def test_generate_pathways_yaml_default_memory_limits_and_requests(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pathways-job",
        "--tpu_slice=tpuv5e:4x4",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        # Verify default memory limits (100G proxy, 70G user container)
        self.assertIn("memory: 100G", rendered)
        self.assertIn("memory: 70G", rendered)
        # Verify default memory requests (4G rm, 16G proxy, 48G user, 100G worker)
        self.assertIn("memory: 4G", rendered)
        self.assertIn("memory: 16G", rendered)
        self.assertIn("memory: 48G", rendered)
        self.assertIn("memory: 100G", rendered)

  def test_generate_pathways_yaml_custom_memory_limits_and_requests(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pathways-job",
        "--tpu_slice=tpuv5e:4x4",
        "--pathways_proxy_memory_limit=190G",
        "--user_container_memory_limit=160G",
        "--user_container_memory=120G",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("memory: 190G", rendered)
        self.assertIn("memory: 160G", rendered)
        self.assertIn("memory: 120G", rendered)

  def test_397b_sidecar_absent_unless_image_is_set(self):
    """An unset image must render no initContainer at all, not an empty one."""
    template_file = _get_template_path("jobset.pathways.qwen3.5-397b.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-397b",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("COLOCATED_PYTHON_SIDECAR_IMAGE", None)
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()
    self.assertNotIn("initContainers", rendered)
    self.assertNotIn("colocated-python-sidecar", rendered)
    # The worker volumeMount the sidecar block is appended to must survive intact.
    self.assertIn("name: shared-tmp", rendered)

  def test_397b_sidecar_rendered_when_image_is_set(self):
    template_file = _get_template_path("jobset.pathways.qwen3.5-397b.yaml")
    image = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:tag"
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-397b",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with mock.patch.dict(
        os.environ,
        {
            "COLOCATED_PYTHON_SIDECAR_IMAGE": image,
            "COLOCATED_PYTHON_SIDECAR_MEMORY": "24Gi",
        },
        clear=False,
    ):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()

    self.assertIn("initContainers:", rendered)
    self.assertIn("name: colocated-python-sidecar", rendered)
    self.assertIn(f"image: {image}", rendered)
    # Native-sidecar pattern: without this the worker never starts.
    self.assertIn("restartPolicy: Always", rendered)
    self.assertIn("containerPort: 50051", rendered)
    # The scaffold shipped `resources: {}`; on a node budgeted this tightly the
    # sidecar must declare memory, and request == limit pins it to Guaranteed.
    self.assertEqual(rendered.count("memory: 24Gi"), 2)
    self.assertNotIn("resources: {}", rendered)
    # Must be valid YAML with the sidecar attached to the worker pod spec.
    import yaml  # pylint: disable=g-import-not-at-top

    docs = [d for d in yaml.safe_load_all(rendered) if d]
    self.assertLen(docs, 1)
    jobs = docs[0]["spec"]["replicatedJobs"]
    specs = [
        j["template"]["spec"]["template"]["spec"]
        for j in jobs
    ]
    with_sidecar = [
        s for s in specs
        if any(c["name"] == "colocated-python-sidecar" for c in s.get("initContainers", []))
    ]
    self.assertLen(with_sidecar, 1)
    self.assertTrue(
        any(c["name"] == "pathways-worker" for c in with_sidecar[0]["containers"]),
        "sidecar must be attached to the pathways-worker pod, not another job",
    )


if __name__ == "__main__":
  absltest.main()
