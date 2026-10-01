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
    self.assertNotIn("--sidecar_name=external", rendered)
    self.assertNotIn("sidecar-shared-memory", rendered)
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
    self.assertIn("restartPolicy: Always", rendered)
    self.assertIn("containerPort: 50051", rendered)
    self.assertIn("--sidecar_name=external", rendered)
    self.assertIn("--cloud_pathways_sidecar_shm_directory=/tmp/sidecar", rendered)
    self.assertIn("CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY", rendered)
    self.assertIn("medium: Memory", rendered)
    self.assertEqual(rendered.count("memory: 24Gi"), 1)
    self.assertNotIn("resources: {}", rendered)
    self.assertEqual(
        self._get_sidecar_mounts(rendered),
        {"sidecar-shared-memory": "/tmp/sidecar"},
    )

  def _get_sidecar_mounts(self, rendered: str) -> dict[str, str]:
    import yaml  # pylint: disable=g-import-not-at-top

    docs = [d for d in yaml.safe_load_all(rendered) if d]
    self.assertLen(docs, 1)
    specs = [
        j["template"]["spec"]["template"]["spec"]
        for j in docs[0]["spec"]["replicatedJobs"]
    ]
    with_sidecar = [
        s
        for s in specs
        if any(
            c["name"] == "colocated-python-sidecar"
            for c in s.get("initContainers", [])
        )
    ]
    self.assertLen(with_sidecar, 1)
    self.assertTrue(
        any(c["name"] == "pathways-worker" for c in with_sidecar[0]["containers"]),
        "sidecar must be attached to the pathways-worker pod, not another job",
    )
    sidecar_c = next(
        c
        for c in with_sidecar[0]["initContainers"]
        if c["name"] == "colocated-python-sidecar"
    )
    return {m["name"]: m["mountPath"] for m in sidecar_c["volumeMounts"]}

  def test_397b_sidecar_shm_disabled_renders_step2_form(self):
    template_file = _get_template_path("jobset.pathways.qwen3.5-397b.yaml")
    image = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:tag"
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-397b-shm0",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with mock.patch.dict(
        os.environ,
        {
            "COLOCATED_PYTHON_SIDECAR_IMAGE": image,
            "COLOCATED_PYTHON_SIDECAR_SHM": "0",
        },
        clear=False,
    ):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()

    self.assertIn("name: colocated-python-sidecar", rendered)
    self.assertIn("--sidecar_name=external", rendered)
    self.assertNotIn("sidecar-shared-memory", rendered)
    self.assertNotIn("/tmp/sidecar", rendered)
    self.assertNotIn("--cloud_pathways_sidecar_shm_directory", rendered)
    self.assertNotIn("CLOUD_PATHWAYS_SIDECAR_SHM_DIRECTORY", rendered)
    self.assertEqual(
        self._get_sidecar_mounts(rendered), {"shared-tmp": "/tmp"}
    )

  def test_sidecar_guard_raises_when_template_lacks_placeholders(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    with open(template_file, "r") as f:
      mutated = f.read().replace("${COLOCATED_PYTHON_SIDECAR_BLOCK}", "")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-guard",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with (
        mock.patch.dict(
            os.environ,
            {"COLOCATED_PYTHON_SIDECAR_IMAGE": "sidecar:latest"},
            clear=False,
        ),
        mock.patch.object(sys, "argv", argv),
        mock.patch("builtins.open", mock.mock_open(read_data=mutated)),
    ):
      with self.assertRaises(SystemExit) as cm:
        yaml_generator.main()
    self.assertIn(template_file, str(cm.exception))
    self.assertIn("COLOCATED_PYTHON_SIDECAR_BLOCK", str(cm.exception))

  @parameterized.named_parameters(
      (
          "cpu",
          "jobset.cpu.yaml",
          ["--jobset_name=test-cpu", "--cpu_machine=n2-standard-64"],
      ),
      (
          "mcjax_ray",
          "jobset.mcjax.ray.yaml",
          ["--jobset_name=test-ray", "--tpu_slice=tpuv5p:2x2x2"],
      ),
      (
          "tpu",
          "jobset.tpu.yaml",
          ["--jobset_name=test-tpu", "--tpu_slice=tpuv5p:2x2x2"],
      ),
  )
  def test_non_pathways_templates_ignore_colocated_sidecar_image(
      self, template_name, extra_args
  ):
    template_file = _get_template_path(template_name)
    argv = ["yaml_generator.py", template_file] + extra_args
    with mock.patch.dict(
        os.environ,
        {"COLOCATED_PYTHON_SIDECAR_IMAGE": "sidecar:latest"},
        clear=False,
    ):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()
    self.assertNotIn("colocated-python-sidecar", rendered)
    self.assertNotIn("sidecar-shared-memory", rendered)

  def test_pathways_default_template_renders_sidecar(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    image = "us-docker.pkg.dev/cloud-tpu-v2-images/pathways-colocated-python/sidecar:tag"
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pw-sidecar",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with mock.patch.dict(
        os.environ,
        {"COLOCATED_PYTHON_SIDECAR_IMAGE": image},
        clear=False,
    ):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()

    self.assertIn("--sidecar_name=external", rendered)
    self.assertIn("--cloud_pathways_sidecar_shm_directory=/tmp/sidecar", rendered)
    self.assertEqual(
        self._get_sidecar_mounts(rendered),
        {"sidecar-shared-memory": "/tmp/sidecar"},
    )

  def test_397b_termination_grace_period_default_and_override(self):
    template_file = _get_template_path("jobset.pathways.qwen3.5-397b.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-grace",
        "--tpu_slice=tpuv5p:8x16x16",
    ]
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("TERMINATION_GRACE_SECONDS", None)
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered_default = mock_stdout.getvalue()
    self.assertEqual(
        rendered_default.count("terminationGracePeriodSeconds: 360"), 2
    )
    self.assertNotIn("terminationGracePeriodSeconds: 10", rendered_default)

    with mock.patch.object(sys, "argv", argv + ["--termination_grace_seconds=120"]):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered_override = mock_stdout.getvalue()
    self.assertEqual(
        rendered_override.count("terminationGracePeriodSeconds: 120"), 2
    )


if __name__ == "__main__":
  absltest.main()
