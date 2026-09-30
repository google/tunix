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
import subprocess
import sys
from typing import Any
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

  @parameterized.named_parameters(
      ("flag_only", ["--preemptible"], {}, {"scheduling.x-k8s.io/preemptible": "true"}),
      (
          "flag_with_queue",
          ["--queue_name=multislice-queue", "--preemptible"],
          {},
          {
              "kueue.x-k8s.io/queue-name": "multislice-queue",
              "scheduling.x-k8s.io/preemptible": "true",
          },
      ),
      (
          "env_with_queue",
          ["--queue_name=multislice-queue"],
          {"PREEMPTIBLE": "true"},
          {
              "kueue.x-k8s.io/queue-name": "multislice-queue",
              "scheduling.x-k8s.io/preemptible": "true",
          },
      ),
      (
          "no_preemptible_overrides_env",
          ["--queue_name=multislice-queue", "--no-preemptible"],
          {"PREEMPTIBLE": "true"},
          {"kueue.x-k8s.io/queue-name": "multislice-queue"},
      ),
  )
  def test_preemptible_label(self, extra_args, env, expected_labels):
    import yaml  # pylint: disable=g-import-not-at-top

    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-preemptible",
        "--tpu_slice=tpu7x:4x4x8",
        *extra_args,
    ]
    with mock.patch.dict(os.environ, env, clear=False):
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()

    jobset = yaml.safe_load(rendered)
    self.assertEqual(jobset["metadata"]["labels"], expected_labels)


_FAIL_FAST_TEMPLATES = (
    ("tpu", "jobset.tpu.yaml", "tpuv5:2x2x1"),
    ("mcjax_ray", "jobset.mcjax.ray.yaml", "tpuv5p:2x2x4"),
    ("pathways", "jobset.pathways.yaml", "tpuv5:4x4x4"),
    ("pathways_397b", "jobset.pathways.qwen3.5-397b.yaml", "tpuv5p:4x8x8"),
)


def _render(template_name: str, tpu_slice: str, *extra_args: str) -> str:
  argv = [
      "yaml_generator.py",
      _get_template_path(template_name),
      "--jobset_name=run-roll-0",
      f"--tpu_slice={tpu_slice}",
      '--worker_startup_command=eval "$FT_TEST_CMD"',
      *extra_args,
  ]
  with mock.patch.object(sys, "argv", argv):
    with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
      yaml_generator.main()
      return mock_stdout.getvalue()


def _proc_job_spec(rendered: str) -> dict[str, Any]:
  import yaml  # pylint: disable=g-import-not-at-top

  jobset = yaml.safe_load(rendered)
  (proc,) = [j for j in jobset["spec"]["replicatedJobs"] if j["name"] == "proc"]
  return proc["template"]["spec"]


class FailFastRenderTest(parameterized.TestCase):

  @parameterized.named_parameters(*_FAIL_FAST_TEMPLATES)
  def test_fail_fast_off_renders_legacy_policy(self, template_name, tpu_slice):
    rendered = _render(template_name, tpu_slice)
    self.assertNotIn("fail-fast", rendered)
    self.assertNotIn("podFailurePolicy", rendered)
    self.assertNotIn("FT_REGISTERED_MARKER", rendered)
    job = _proc_job_spec(rendered)
    self.assertNotIn("initContainers", job["template"]["spec"])

  @parameterized.named_parameters(*_FAIL_FAST_TEMPLATES)
  def test_fail_fast_on_structure(self, template_name, tpu_slice):
    import yaml  # pylint: disable=g-import-not-at-top

    rendered = _render(
        template_name, tpu_slice, "--fail_fast", "--startup_retries=2"
    )
    jobset = yaml.safe_load(rendered)
    self.assertEqual(
        jobset["spec"]["failurePolicy"],
        {
            "maxRestarts": 2,
            "restartStrategy": "Recreate",
            "rules": [{
                "name": "failJobSetOnPodFailurePolicy",
                "action": "FailJobSet",
                "onJobFailureReasons": ["PodFailurePolicy"],
            }],
        },
    )
    job = _proc_job_spec(rendered)
    self.assertEqual(job["backoffLimit"], 0)
    self.assertEqual(
        job["podFailurePolicy"]["rules"],
        [
            {
                "action": "Count",
                "onExitCodes": {
                    "containerName": "main",
                    "operator": "In",
                    "values": [yaml_generator.STARTUP_RETRY_EXIT_CODE],
                },
            },
            {
                "action": "FailJob",
                "onPodConditions": [{"type": "DisruptionTarget"}],
            },
            {
                "action": "FailJob",
                "onExitCodes": {
                    "containerName": "main",
                    "operator": "NotIn",
                    "values": [0, yaml_generator.STARTUP_RETRY_EXIT_CODE],
                },
            },
        ],
    )
    pod = job["template"]["spec"]
    self.assertEqual(pod["restartPolicy"], "Never")
    self.assertEqual([c["name"] for c in pod["containers"]], ["main"])
    if template_name.startswith("jobset.pathways"):
      # rm/proxy must be native sidecars so the pod ends with the user
      # container; pw-node keeps restarting in place.
      self.assertEqual(
          [c["name"] for c in pod["initContainers"]],
          ["pathways-rm", "pathways-proxy"],
      )
      (pw_node,) = [
          j for j in jobset["spec"]["replicatedJobs"] if j["name"] == "pw-node"
      ]
      self.assertEqual(pw_node["template"]["spec"]["backoffLimit"], 2048000)

  def test_negative_startup_retries_raises(self):
    with self.assertRaises(ValueError):
      _render(
          "jobset.tpu.yaml", "tpuv5:2x2x1", "--fail_fast", "--startup_retries=-1"
      )

  def test_fail_fast_on_unsupported_template_raises(self):
    with self.assertRaisesRegex(ValueError, "not supported by template"):
      _render("jobset.mcjax.yaml", "tpu7x:4x4x4", "--fail_fast")

  @parameterized.named_parameters(
      ("crash_before_register", "exit 1", "0", 75),
      ("exit0_before_register", "exit 0", "0", 75),
      ("crash_after_register", 'touch "$FT_REGISTERED_MARKER"; exit 3', "0", 3),
      ("exit0_after_register", 'touch "$FT_REGISTERED_MARKER"; exit 0', "0", 1),
      (
          "exit75_after_register",
          'touch "$FT_REGISTERED_MARKER"; exit 75',
          "0",
          1,
      ),
  )
  def test_wrapper_exit_code_mapping(self, cmd, pod_index, expected):
    rendered = _render("jobset.tpu.yaml", "tpuv5:2x2x1", "--fail_fast")
    job = _proc_job_spec(rendered)
    (container,) = job["template"]["spec"]["containers"]
    marker = os.path.join(self.create_tempdir().full_path, "registered")
    script = container["command"][2].replace(
        yaml_generator.REGISTERED_MARKER_PATH, marker
    )
    env = dict(os.environ, FT_TEST_CMD=cmd, POD_INDEX=pod_index)
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, timeout=60,
        check=False,
    )
    self.assertEqual(result.returncode, expected, result.stdout)


if __name__ == "__main__":
  absltest.main()
