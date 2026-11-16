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
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from tunix.dist_rl.distributed.deployment import yaml_generator
import yaml

_THRESHOLD_ENV = "IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_THRESHOLD_BYTES"


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
        self.assertIn(
            "node.kubernetes.io/instance-type: n2-standard-64", rendered
        )
        self.assertIn("9999", rendered)
        doc = yaml.safe_load(rendered)
        self.assertEqual(doc["spec"]["failurePolicy"]["maxRestarts"], 0)
        pod_tpl = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]
        self.assertEqual(
            pod_tpl["metadata"]["annotations"][
                "cluster-autoscaler.kubernetes.io/safe-to-evict"
            ],
            "false",
        )
        pod_spec = pod_tpl["spec"]
        self.assertNotIn("priorityClassName", pod_spec)
        self.assertNotIn("serviceAccountName", pod_spec)
        self.assertEqual(pod_spec["terminationGracePeriodSeconds"], 300)
        self.assertNotIn("resources", pod_spec["containers"][0])

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
        self.assertIn(
            "image:"
            " us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/server:latest",
            rendered,
        )
        self.assertIn(
            "image:"
            " us-docker.pkg.dev/cloud-tpu-v2-images/cloud-pathways/proxy_server:latest",
            rendered,
        )

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

  def test_generate_pathways_yaml_large_transfer_directory(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pathways-job",
        "--tpu_slice=tpuv5:2x2x2",
        "--user_container_memory_limit=256G",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        # One memory-backed volume, sized like the user container (its pages
        # are charged there), mounted in pathways-proxy and the user container.
        self.assertEqual(rendered.count("name: ifrt-proxy-large-transfer"), 3)
        self.assertIn("medium: Memory", rendered)
        self.assertIn("sizeLimit: 256G", rendered)
        self.assertEqual(
            rendered.count("mountPath: /ifrt_proxy_large_transfer"), 2
        )
        self.assertEqual(
            rendered.count(
                "name: IFRT_PROXY_GRPC_LARGE_TRANSFER_OPTIMIZATION_DIRECTORY"
            ),
            2,
        )
        # Without the chmod, pathways-proxy cannot delete the files; the
        # container must not start if it fails while the transfers are on.
        self.assertRegex(
            rendered,
            r"if ! chmod 0777 /ifrt_proxy_large_transfer &&\s+"
            rf"\(\( {_THRESHOLD_ENV} < 9223372036854775807 \)\); then",
        )
        # Off by default: the IFRT proxy client's own default threshold.
        self.assertIn('value: "9223372036854775807"', rendered)

  def test_generate_pathways_yaml_large_transfer_threshold(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pathways-job",
        "--tpu_slice=tpuv5:2x2x2",
        "--ifrt_proxy_large_transfer_threshold_bytes=16777216",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertRegex(
            rendered, rf'name: {_THRESHOLD_ENV}\s+value: "16777216"'
        )

  @parameterized.named_parameters(
      ("zero", "0"),
      ("negative", "-1"),
      ("above_int64", str(2**63)),
  )
  def test_invalid_large_transfer_threshold_raises(self, threshold):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--tpu_slice=tpuv5:2x2x2",
        f"--ifrt_proxy_large_transfer_threshold_bytes={threshold}",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stderr", new_callable=io.StringIO):
        with self.assertRaises(SystemExit):
          yaml_generator.main()

  def test_generate_ray_yaml_with_pre_init_commands(self):
    template_file = _get_template_path("leaderworkerset.mcjax.ray.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-ray-job",
        "--tpu_slice=tpuv5e:2x4",
        "--ray_head_pre_init_command=echo head_pre_init",
        "--ray_worker_pre_init_command=echo worker_pre_init",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("echo head_pre_init", rendered)
        self.assertIn("echo worker_pre_init", rendered)

  def test_generate_otel_collector_k8s_yaml(self):
    template_file = _get_template_path("jobset.otel.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-user-otel",
        "--namespace=test-ns",
        "--worker_container_port=4317",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertNotIn("kind: ConfigMap", rendered)
        self.assertIn("kind: Deployment", rendered)
        self.assertIn("kind: Service", rendered)
        self.assertIn("name: test-user-otel", rendered)
        self.assertIn("namespace: test-ns", rendered)
        self.assertIn("otel/opentelemetry-collector-contrib:0.114.0", rendered)
        self.assertIn(
            "--set=receivers::otlp::protocols::grpc::endpoint=0.0.0.0:4317",
            rendered,
        )
        self.assertIn("memory: 512Mi", rendered)

  def test_generate_pathways_yaml_init_containers(self):
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-pathways-job",
        "--tpu_slice=tpuv5e:4x4",
        "--worker_container_name=worker",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        doc = yaml.safe_load(rendered)
        proc_job = doc["spec"]["replicatedJobs"][0]
        proc_spec = proc_job["template"]["spec"]["template"]["spec"]
        init_names = [c["name"] for c in proc_spec["initContainers"]]
        container_names = [c["name"] for c in proc_spec["containers"]]
        self.assertEqual(init_names, ["pathways-rm", "pathways-proxy"])
        self.assertEqual(container_names, ["worker"])
        for init_container in proc_spec["initContainers"]:
          self.assertEqual(init_container["restartPolicy"], "Always")

  @parameterized.parameters(
      "jobset.cpu.yaml",
      "jobset.pathways.yaml",
      "jobset.tpu.yaml",
      "jobset.mcjax.yaml",
      "jobset.mcjax.ray.yaml",
  )
  def test_regular_containers_omit_container_level_restart_policy(
      self, template_name
  ):
    template_file = _get_template_path(template_name)
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-job",
        "--tpu_slice=tpuv5e:4x4",
        "--cpu_machine=n2-standard-64",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
    for rjob in doc["spec"]["replicatedJobs"]:
      pod_spec = rjob["template"]["spec"]["template"]["spec"]
      for container in pod_spec["containers"]:
        self.assertNotIn(
            "restartPolicy",
            container,
            f"Regular container {container['name']!r} in replicatedJob"
            f" {rjob['name']!r} ({template_name}) must not set container-level"
            " restartPolicy.",
        )

  @parameterized.named_parameters(
      (
          "pathways_default",
          "jobset.pathways.yaml",
          False,
          3,
          3,
          0,
          "Never",
      ),
      (
          "pathways_fail_fast_default",
          "jobset.pathways.yaml",
          True,
          3,
          3,
          0,
          "Never",
      ),
      (
          "pathways_fail_fast_custom",
          "jobset.pathways.yaml",
          True,
          5,
          5,
          0,
          "Never",
      ),
      ("tpu_default", "jobset.tpu.yaml", False, 3, 3, 2048000, "OnFailure"),
      (
          "tpu_no_ff_custom_retries",
          "jobset.tpu.yaml",
          False,
          0,
          0,
          2048000,
          "OnFailure",
      ),
      ("tpu_fail_fast_default", "jobset.tpu.yaml", True, 3, 3, 0, "Never"),
      ("tpu_fail_fast_custom", "jobset.tpu.yaml", True, 5, 5, 0, "Never"),
      ("mcjax_default", "jobset.mcjax.yaml", False, 3, 3, 2048000, "OnFailure"),
      ("mcjax_fail_fast_default", "jobset.mcjax.yaml", True, 3, 3, 0, "Never"),
      ("mcjax_fail_fast_custom", "jobset.mcjax.yaml", True, 5, 5, 0, "Never"),
      (
          "mcjax_ray_default",
          "jobset.mcjax.ray.yaml",
          False,
          3,
          3,
          2048000,
          "OnFailure",
      ),
      (
          "mcjax_ray_fail_fast_default",
          "jobset.mcjax.ray.yaml",
          True,
          3,
          3,
          0,
          "Never",
      ),
      (
          "mcjax_ray_fail_fast_custom",
          "jobset.mcjax.ray.yaml",
          True,
          5,
          5,
          0,
          "Never",
      ),
  )
  def test_fail_fast_modes(
      self,
      template_name,
      fail_fast,
      startup_retries,
      expected_restarts,
      expected_backoff,
      expected_restart_policy,
  ):
    template_file = _get_template_path(template_name)
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-job",
        "--tpu_slice=tpuv5e:4x4",
        f"--startup_retries={startup_retries}",
        "--worker_container_name=worker",
    ]
    if fail_fast:
      argv.append("--fail_fast")
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        doc = yaml.safe_load(rendered)
        self.assertIsNotNone(doc)
        self.assertEqual(
            int(doc["spec"]["failurePolicy"]["maxRestarts"]), expected_restarts
        )
        proc_job = doc["spec"]["replicatedJobs"][0]
        self.assertEqual(
            int(proc_job["template"]["spec"]["backoffLimit"]), expected_backoff
        )
        proc_spec = proc_job["template"]["spec"]["template"]["spec"]
        self.assertEqual(proc_spec["restartPolicy"], expected_restart_policy)
        if template_name == "jobset.pathways.yaml":
          # pw-node keeps in-place restarts regardless of --fail_fast.
          worker_job = doc["spec"]["replicatedJobs"][1]
          self.assertEqual(
              int(worker_job["template"]["spec"]["backoffLimit"]), 2048000
          )
          worker_spec = worker_job["template"]["spec"]["template"]["spec"]
          self.assertEqual(worker_spec["restartPolicy"], "OnFailure")

        # safe-to-evict annotation
        pod_metadata = proc_job["template"]["spec"]["template"]["metadata"]
        self.assertEqual(
            pod_metadata["annotations"][
                "cluster-autoscaler.kubernetes.io/safe-to-evict"
            ],
            "false",
        )

        # Worker non-zero exit guard
        if fail_fast:
          self.assertIn(yaml_generator._WORKER_NONZERO_EXIT_GUARD, rendered)
        else:
          self.assertNotIn("Worker exited with code 0 unexpectedly", rendered)

        # initContainers vs containers check
        if template_name == "jobset.pathways.yaml":
          init_names = [c["name"] for c in proc_spec["initContainers"]]
          container_names = [c["name"] for c in proc_spec["containers"]]
          self.assertEqual(init_names, ["pathways-rm", "pathways-proxy"])
          self.assertEqual(container_names, ["worker"])
        else:
          container_names = [c["name"] for c in proc_spec["containers"]]
          self.assertEqual(container_names, ["worker"])

  def test_negative_startup_retries_raises(self):
    template_file = _get_template_path("jobset.tpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--tpu_slice=tpuv5e:4x4",
        "--startup_retries=-1",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaisesRegex(ValueError, "--startup_retries must be >= 0"):
        yaml_generator.main()

  def test_gang_id_valid_and_invalid(self):
    rendered = yaml_generator.render_gang_labels("test-gang-123.foo_bar")
    self.assertIn(
        'scheduling.x-k8s.io/gang-id: "test-gang-123.foo_bar"', rendered
    )

    self.assertEqual(yaml_generator.render_gang_labels(None), "")
    self.assertEqual(yaml_generator.render_gang_labels(""), "")

    for invalid_id in [
        "-invalid-start",
        "invalid-end-",
        "invalid space",
        "invalid@char",
        "abc\n",
        "a" * 64,
    ]:
      with self.assertRaises(ValueError):
        yaml_generator.render_gang_labels(invalid_id)
      with self.assertRaises(ValueError):
        yaml_generator.render_teardown_on_exit(True, invalid_id, "ns", "job")

  def test_teardown_gang_on_exit(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    base_argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-cpu-job",
        "--cpu_machine=n2-standard-64",
        "--namespace=test-ns",
    ]

    # Enabled with gang_id -> includes kubectl delete
    argv_enabled = base_argv + [
        "--gang_id=my-gang",
        "--teardown_gang_on_exit",
    ]
    with mock.patch.object(sys, "argv", argv_enabled):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        doc = yaml.safe_load(rendered)
        self.assertIsNotNone(doc)
        self.assertEqual(
            doc["metadata"]["labels"]["scheduling.x-k8s.io/gang-id"], "my-gang"
        )
        self.assertIn("kubectl delete jobsets.jobset.x-k8s.io", rendered)
        self.assertIn('-n "test-ns"', rendered)
        self.assertIn('-l "scheduling.x-k8s.io/gang-id=my-gang"', rendered)
        self.assertIn(
            '--field-selector="metadata.name!=test-cpu-job"', rendered
        )

    # False teardown flag with gang_id -> omitted
    argv_disabled = base_argv + ["--gang_id=my-gang"]
    with mock.patch.object(sys, "argv", argv_disabled):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        doc = yaml.safe_load(rendered)
        self.assertIsNotNone(doc)
        self.assertEqual(
            doc["metadata"]["labels"]["scheduling.x-k8s.io/gang-id"], "my-gang"
        )
        self.assertNotIn("kubectl delete jobsets.jobset.x-k8s.io", rendered)

  def test_teardown_on_exit_warns_and_keeps_exit_code_when_kubectl_fails(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    with open(template_file, "r") as f:
      template_text = f.read()
    args = yaml_generator.parse_args([
        template_file,
        "--jobset_name=orch",
        "--cpu_machine=n2-standard-64",
        "--gang_id=gang-1",
        "--teardown_gang_on_exit",
        "--worker_startup_command=exit 7",
    ])
    doc = yaml.safe_load(yaml_generator.render_template(template_text, args))
    script = doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]["containers"][0]["command"][2]
    with tempfile.TemporaryDirectory() as tmpdir:
      kubectl = os.path.join(tmpdir, "kubectl")
      with open(kubectl, "w") as f:
        f.write("#!/bin/sh\nexit 1\n")
      os.chmod(kubectl, 0o755)
      res = subprocess.run(
          ["bash", "-c", script],
          env=dict(os.environ, PATH=f"{tmpdir}:{os.environ.get('PATH', '')}"),
          capture_output=True,
          text=True,
          check=False,
          timeout=10,
      )
    self.assertEqual(res.returncode, 7)
    self.assertIn(
        "WARNING: Failed to tear down sibling JobSets in gang gang-1",
        res.stderr,
    )

  @parameterized.parameters(
      "jobset.cpu.yaml",
      "jobset.pathways.yaml",
      "jobset.tpu.yaml",
      "jobset.mcjax.yaml",
      "jobset.mcjax.ray.yaml",
  )
  def test_generate_yaml_with_gang_id_top_level_labels(self, template_name):
    template_file = _get_template_path(template_name)
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-gang-job",
        "--tpu_slice=tpuv5e:4x4",
        "--cpu_machine=n2-standard-64",
        "--gang_id=my-gang-1",
        "--queue_name=my-queue",
        "--namespace=my-ns",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        doc = yaml.safe_load(rendered)
        self.assertIsNotNone(doc)
        self.assertEqual(
            doc["metadata"]["labels"]["scheduling.x-k8s.io/gang-id"],
            "my-gang-1",
        )
        self.assertEqual(
            doc["metadata"]["labels"]["kueue.x-k8s.io/queue-name"],
            "my-queue",
        )

  def test_teardown_gang_on_exit_without_gang_id_raises(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-cpu-job",
        "--cpu_machine=n2-standard-64",
        "--teardown_gang_on_exit",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaisesRegex(
          ValueError, "--teardown_gang_on_exit requires --gang_id"
      ):
        yaml_generator.main()

  def test_template_compatibility_checks(self):
    # Explicit --fail_fast with jobset.cpu.yaml raises ValueError
    cpu_template = _get_template_path("jobset.cpu.yaml")
    cpu_argv = [
        "yaml_generator.py",
        cpu_template,
        "--jobset_name=test-cpu",
        "--cpu_machine=n2-standard-64",
        "--fail_fast",
    ]
    with mock.patch.object(sys, "argv", cpu_argv):
      with self.assertRaisesRegex(
          ValueError, "--fail_fast is not supported by template"
      ):
        yaml_generator.main()

    # Explicit --teardown_gang_on_exit with jobset.tpu.yaml raises ValueError
    tpu_template = _get_template_path("jobset.tpu.yaml")
    tpu_argv = [
        "yaml_generator.py",
        tpu_template,
        "--jobset_name=test-tpu",
        "--tpu_slice=tpuv5e:4x4",
        "--gang_id=my-gang",
        "--teardown_gang_on_exit",
    ]
    with mock.patch.object(sys, "argv", tpu_argv):
      with self.assertRaisesRegex(
          ValueError, "--teardown_gang_on_exit is not supported by template"
      ):
        yaml_generator.main()

    # Gang-wide env vars FAIL_FAST=1 and TEARDOWN_GANG_ON_EXIT=1 apply only to
    # templates that support them without raising on other templates in the
    # gang.
    with open(cpu_template, "r") as f:
      cpu_text = f.read()
    with open(tpu_template, "r") as f:
      tpu_text = f.read()
    with mock.patch.dict(
        os.environ,
        {
            "FAIL_FAST": "1",
            "GANG_ID": "gang-env",
            "TEARDOWN_GANG_ON_EXIT": "1",
        },
    ):
      cpu_args = yaml_generator.parse_args([
          cpu_template,
          "--jobset_name=orch",
          "--cpu_machine=n2-standard-64",
      ])
      tpu_args = yaml_generator.parse_args([
          tpu_template,
          "--jobset_name=worker",
          "--tpu_slice=tpuv5e:4x4",
      ])
      cpu_rendered = yaml_generator.render_template(cpu_text, cpu_args)
      self.assertIn("kubectl delete jobsets.jobset.x-k8s.io", cpu_rendered)
      tpu_rendered = yaml_generator.render_template(tpu_text, tpu_args)
      self.assertIn("Worker exited with code 0 unexpectedly", tpu_rendered)
      # Each env var is ignored by the template that lacks its placeholder.
      cpu_subs = yaml_generator.build_substitutions(
          cpu_args, template_text=cpu_text
      )
      self.assertEqual(cpu_subs["WORKER_NONZERO_EXIT_GUARD"], "")
      tpu_subs = yaml_generator.build_substitutions(
          tpu_args, template_text=tpu_text
      )
      self.assertEqual(tpu_subs["TEARDOWN_ON_EXIT"], "")

  def test_startup_retries_env_default(self):
    # STARTUP_RETRIES sets the default; unset or empty falls back to 3.
    with mock.patch.dict(os.environ, {"STARTUP_RETRIES": "5"}):
      args = yaml_generator.parse_args(["jobset.tpu.yaml"])
      self.assertEqual(args.startup_retries, 5)

    with mock.patch.dict(os.environ, {"STARTUP_RETRIES": ""}):
      args = yaml_generator.parse_args(["jobset.tpu.yaml"])
      self.assertEqual(args.startup_retries, 3)

  def _extract_tpu_worker_script(
      self, fail_fast: bool, startup_cmd: str
  ) -> str:
    template_file = _get_template_path("jobset.tpu.yaml")
    with open(template_file, "r") as f:
      template_text = f.read()
    argv = [
        template_file,
        "--tpu_slice=tpuv5e:4x4",
        f"--worker_startup_command={startup_cmd}",
    ]
    if fail_fast:
      argv.append("--fail_fast")
    args = yaml_generator.parse_args(argv)
    rendered = yaml_generator.render_template(template_text, args)
    doc = yaml.safe_load(rendered)
    container = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]["containers"][0]
    return container["command"][2]

  def test_tpu_worker_pid1_script_exit_codes(self):
    # Non-zero exit code propagated under fail_fast
    script_42 = self._extract_tpu_worker_script(
        fail_fast=True, startup_cmd="exit 42"
    )
    res_42 = subprocess.run(
        ["bash", "-c", script_42],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(res_42.returncode, 42)

    # Exit code 0 mapped to 1 under fail_fast without SIGTERM
    script_0_ff = self._extract_tpu_worker_script(
        fail_fast=True, startup_cmd="exit 0"
    )
    res_0_ff = subprocess.run(
        ["bash", "-c", script_0_ff],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(res_0_ff.returncode, 1)
    self.assertIn(
        "Worker exited with code 0 unexpectedly; forcing exit code 1.",
        res_0_ff.stderr,
    )

    # Exit code 0 preserved when fail_fast is disabled
    script_0_no_ff = self._extract_tpu_worker_script(
        fail_fast=False, startup_cmd="exit 0"
    )
    res_0_no_ff = subprocess.run(
        ["bash", "-c", script_0_no_ff],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertEqual(res_0_no_ff.returncode, 0)

  def test_tpu_worker_pid1_script_sigterm_forwarding(self):
    # Non-exec-optimized compound command where a direct child drains on SIGTERM
    # before writing a marker; wrapper must wait for the drain and exit 143.
    # Redirect stdout/stderr to files and use proc.wait() (not communicate(),
    # which blocks on pipe EOF from orphaned grandchildren) so this test fails
    # if the subshell exits before its child finishes draining.
    with tempfile.TemporaryDirectory() as tmpdir:
      ready_file = os.path.join(tmpdir, "ready")
      drain_marker = os.path.join(tmpdir, "drained")
      stdout_path = os.path.join(tmpdir, "stdout.log")
      stderr_path = os.path.join(tmpdir, "stderr.log")
      child_sh = (
          "sleep 30 >/dev/null 2>&1 & SPID=$!; "
          "trap 'kill $SPID 2>/dev/null; sleep 0.4; "
          f"echo drained > {shlex.quote(drain_marker)}; exit 0' SIGTERM; "
          f"echo 1 > {shlex.quote(ready_file)}; "
          "wait $SPID"
      )
      startup_cmd = (
          f"bash -c {shlex.quote(child_sh)}; echo trailing_step_should_not_run"
      )
      script = self._extract_tpu_worker_script(
          fail_fast=True, startup_cmd=startup_cmd
      )
      with (
          open(stdout_path, "wb") as stdout_f,
          open(stderr_path, "wb") as stderr_f,
      ):
        proc = subprocess.Popen(
            ["bash", "-c", script],
            stdout=stdout_f,
            stderr=stderr_f,
        )
        deadline = time.monotonic() + 5.0
        while not os.path.exists(ready_file):
          self.assertLess(time.monotonic(), deadline)
          time.sleep(0.02)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10)
      self.assertTrue(os.path.exists(drain_marker))
      self.assertEqual(rc, 143)
      with open(stdout_path, "r") as f:
        stdout = f.read()
      with open(stderr_path, "r") as f:
        stderr = f.read()
      self.assertNotIn("trailing_step_should_not_run", stdout)
      self.assertNotIn("Worker exited with code 0 unexpectedly", stderr)

  def test_preemptible_flag_and_env(self):
    template_file = _get_template_path("jobset.tpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-job",
        "--tpu_slice=tpuv5e:4x4",
        "--preemptible",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        self.assertEqual(
            doc["metadata"]["labels"]["scheduling.x-k8s.io/preemptible"],
            "true",
        )

    # Test via environment variable PREEMPTIBLE=true
    with mock.patch.dict(os.environ, {"PREEMPTIBLE": "true"}):
      argv_env = [
          "yaml_generator.py",
          template_file,
          "--jobset_name=test-job-env",
          "--tpu_slice=tpuv5e:4x4",
      ]
      with mock.patch.object(sys, "argv", argv_env):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          doc = yaml.safe_load(mock_stdout.getvalue())
          self.assertEqual(
              doc["metadata"]["labels"]["scheduling.x-k8s.io/preemptible"],
              "true",
          )

  def test_hardware_and_scheduling_knobs(self):
    cpu_template = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        cpu_template,
        "--jobset_name=test-cpu",
        "--cpu_machine=n2-standard-64",
        "--priority_class=high-priority",
        "--service_account=custom-sa",
        "--cpu_nodepool=my-cpu-pool",
        "--cpu_memory=64Gi",
        "--termination_grace_seconds=45",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertEqual(pod_spec["priorityClassName"], "high-priority")
        self.assertEqual(pod_spec["serviceAccountName"], "custom-sa")
        self.assertEqual(pod_spec["terminationGracePeriodSeconds"], 45)
        self.assertEqual(
            pod_spec["nodeSelector"]["cloud.google.com/gke-nodepool"],
            "my-cpu-pool",
        )
        self.assertEqual(
            pod_spec["containers"][0]["resources"]["requests"]["memory"],
            "64Gi",
        )

    tpu_template = _get_template_path("jobset.tpu.yaml")
    with mock.patch.dict(
        os.environ,
        {"KUEUE_PRIORITY_CLASS": "fallback-prio", "TPU_RESERVATION": "my-res"},
    ):
      argv = [
          "yaml_generator.py",
          tpu_template,
          "--jobset_name=test-tpu",
          "--tpu_slice=tpuv5e:4x4",
          "--termination_grace_seconds=60",
      ]
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          doc = yaml.safe_load(mock_stdout.getvalue())
          pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
              "template"
          ]["spec"]
          self.assertEqual(pod_spec["priorityClassName"], "fallback-prio")
          self.assertEqual(pod_spec["terminationGracePeriodSeconds"], 60)
          self.assertEqual(
              pod_spec["nodeSelector"]["cloud.google.com/reservation-name"],
              "my-res",
          )
          self.assertIn(
              {"name": "dshm", "emptyDir": {"medium": "Memory"}},
              pod_spec["volumes"],
          )
          self.assertIn(
              {"name": "dshm", "mountPath": "/dev/shm"},
              pod_spec["containers"][0]["volumeMounts"],
          )
          env_map = {
              e["name"]: e["value"]
              for e in pod_spec["containers"][0]["env"]
              if "value" in e
          }
          self.assertEqual(env_map["ENABLE_MULTI_NUMA"], "0")
          self.assertEqual(env_map["TPU_RAIDEN_DATA_NICS"], "eth0")

    mcjax_template = _get_template_path("jobset.mcjax.yaml")
    argv_mcjax = [
        "yaml_generator.py",
        mcjax_template,
        "--jobset_name=test-mcjax",
        "--tpu_slice=tpuv5e:4x4",
    ]
    with mock.patch.object(sys, "argv", argv_mcjax):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        pod_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertIn(
            {"name": "dshm", "emptyDir": {"medium": "Memory"}},
            pod_spec["volumes"],
        )
        self.assertIn(
            {"name": "dshm", "mountPath": "/dev/shm"},
            pod_spec["containers"][0]["volumeMounts"],
        )

  def test_dynamic_slicing(self):
    pw_template = _get_template_path("jobset.pathways.yaml")
    # --head_nodepool without --use_dynamic_slicing raises ValueError
    with mock.patch.object(
        sys,
        "argv",
        [
            "yaml_generator.py",
            pw_template,
            "--jobset_name=test-pw",
            "--tpu_slice=tpu7x:4x4x8",
            "--head_nodepool=head-pool",
        ],
    ):
      with self.assertRaisesRegex(
          ValueError, "--head_nodepool.*requires --use_dynamic_slicing"
      ):
        yaml_generator.main()

    # With --use_dynamic_slicing and --head_nodepool
    argv = [
        "yaml_generator.py",
        pw_template,
        "--jobset_name=test-pw",
        "--tpu_slice=tpu7x:4x4x8",
        "--use_dynamic_slicing",
        "--head_nodepool=head-pool",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        proc_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertEqual(
            proc_spec["nodeSelector"]["cloud.google.com/gke-nodepool"],
            "head-pool",
        )
        pw_spec = doc["spec"]["replicatedJobs"][1]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertIn("affinity", pw_spec)
        self.assertIn("nodeAffinity", pw_spec["affinity"])

    # With --use_dynamic_slicing and --cpu_machine (without --head_nodepool)
    argv_machine = [
        "yaml_generator.py",
        pw_template,
        "--jobset_name=test-pw-machine",
        "--tpu_slice=tpu7x:4x4x4",
        "--use_dynamic_slicing",
        "--cpu_machine=n2d-standard-64",
    ]
    with mock.patch.object(sys, "argv", argv_machine):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        proc_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertEqual(
            proc_spec["nodeSelector"]["node.kubernetes.io/instance-type"],
            "n2d-standard-64",
        )
        self.assertEqual(proc_spec["tolerations"][0]["operator"], "Exists")

    # With --use_dynamic_slicing and neither --head_nodepool nor --cpu_machine
    argv_default_pool = [
        "yaml_generator.py",
        pw_template,
        "--jobset_name=test-pw-default",
        "--tpu_slice=tpu7x:4x4x4",
        "--use_dynamic_slicing",
    ]
    with mock.patch.object(sys, "argv", argv_default_pool):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        doc = yaml.safe_load(mock_stdout.getvalue())
        proc_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        self.assertEqual(
            proc_spec["nodeSelector"]["cloud.google.com/gke-nodepool"],
            "cpu-np",
        )
        self.assertEqual(proc_spec["tolerations"][0]["operator"], "Equal")
        self.assertEqual(proc_spec["tolerations"][0]["value"], "cpu-np")

  def _render_pathways_head_spec(
      self, extra_argv: list[str], env: dict[str, str]
  ) -> dict[str, Any]:
    template_file = _get_template_path("jobset.pathways.yaml")
    with open(template_file, "r") as f:
      template_text = f.read()
    with mock.patch.dict(os.environ, env, clear=True):
      args = yaml_generator.parse_args([
          template_file,
          "--jobset_name=test-pw",
          "--tpu_slice=tpuv5:4x4x4",
          *extra_argv,
      ])
      doc = yaml.safe_load(yaml_generator.render_template(template_text, args))
    return doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]

  def test_colocated_head_has_no_node_affinity_by_default(self):
    proc_spec = self._render_pathways_head_spec([], {})
    self.assertEqual(list(proc_spec["affinity"]), ["podAffinity"])

  @parameterized.named_parameters(
      ("without_reservation", {}, {}),
      (
          "with_reservation",
          {"TPU_RESERVATION": "my-res"},
          {"cloud.google.com/reservation-name": "my-res"},
      ),
  )
  def test_colocated_head_excludes_given_nodepools(
      self, env: dict[str, str], extra_selector: dict[str, str]
  ):
    proc_spec = self._render_pathways_head_spec(
        ["--head_excluded_nodepools=cpu-np,sandbox-pool"], env
    )
    self.assertEqual(
        proc_spec["nodeSelector"],
        {
            "cloud.google.com/gke-tpu-accelerator": "tpu-v5p-slice",
            "cloud.google.com/gke-tpu-topology": "4x4x4",
            **extra_selector,
        },
    )
    self.assertEqual(
        proc_spec["affinity"]["nodeAffinity"],
        {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [{
                    "matchExpressions": [{
                        "key": "cloud.google.com/gke-nodepool",
                        "operator": "NotIn",
                        "values": ["cpu-np", "sandbox-pool"],
                    }]
                }]
            }
        },
    )
    (pod_term,) = proc_spec["affinity"]["podAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]
    self.assertEqual(pod_term["topologyKey"], "kubernetes.io/hostname")

  @parameterized.named_parameters(
      ("empty_entry", ["--head_excluded_nodepools=cpu-np,"], "Invalid"),
      ("bad_chars", ["--head_excluded_nodepools=cpu np"], "Invalid"),
      ("repeated", ["--head_excluded_nodepools=a,a"], "repeats"),
      (
          "with_dynamic_slicing",
          ["--head_excluded_nodepools=cpu-np", "--use_dynamic_slicing"],
          "only without --use_dynamic_slicing",
      ),
  )
  def test_invalid_head_excluded_nodepools_raise(
      self, extra_argv: list[str], error: str
  ):
    with self.assertRaisesRegex(ValueError, error):
      self._render_pathways_head_spec(extra_argv, {})

  def test_colocated_python_sidecar(self):
    pw_template = _get_template_path("jobset.pathways.yaml")
    with mock.patch.dict(
        os.environ,
        {
            "COLOCATED_PYTHON_SIDECAR_IMAGE": "test-sidecar-img:v1",
            "COLOCATED_PYTHON_SIDECAR_SHM": "1",
        },
    ):
      argv = [
          "yaml_generator.py",
          pw_template,
          "--jobset_name=test-pw-sidecar",
          "--tpu_slice=tpuv5e:4x4",
      ]
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          rendered = mock_stdout.getvalue()
          doc = yaml.safe_load(rendered)
          pw_spec = doc["spec"]["replicatedJobs"][1]["template"]["spec"][
              "template"
          ]["spec"]
          worker_container = pw_spec["containers"][0]
          self.assertIn(
              "--cloud_pathways_sidecar_shm_directory=/tmp/sidecar",
              worker_container["args"],
          )
          # check colocated sidecar container in initContainers
          sidecar = next(
              c
              for c in pw_spec["initContainers"]
              if c["name"] == "colocated-python-sidecar"
          )
          self.assertEqual(sidecar["image"], "test-sidecar-img:v1")
          # check sidecar-shared-memory in volumes
          self.assertTrue(
              any(
                  v["name"] == "sidecar-shared-memory"
                  for v in pw_spec["volumes"]
              )
          )

  def test_pathways_extra_env_and_proxy_args(self):
    pw_template = _get_template_path("jobset.pathways.yaml")
    with mock.patch.dict(
        os.environ,
        {
            "PATHWAYS_WORKER_EXTRA_ENV": 'MY_CUSTOM_VAR="val1"\nOTHER_VAR=2',
            "PATHWAYS_PROXY_EXTRA_ARGS": "foo=1 bar=2",
            "RAIDEN_BROADCAST_K": "128",
        },
    ):
      argv = [
          "yaml_generator.py",
          pw_template,
          "--jobset_name=test-pw-extra",
          "--tpu_slice=tpuv5e:4x4",
      ]
      with mock.patch.object(sys, "argv", argv):
        with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
          yaml_generator.main()
          doc = yaml.safe_load(mock_stdout.getvalue())
          proxy = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
              "template"
          ]["spec"]["initContainers"][1]
          self.assertIn("foo=1", proxy["args"])
          self.assertIn("bar=2", proxy["args"])
          worker = doc["spec"]["replicatedJobs"][1]["template"]["spec"][
              "template"
          ]["spec"]["containers"][0]
          env_map = {
              e["name"]: e.get("value") for e in worker["env"] if "value" in e
          }
          self.assertEqual(env_map["MY_CUSTOM_VAR"], "val1")
          self.assertEqual(env_map["OTHER_VAR"], "2")
          self.assertEqual(env_map["RAIDEN_BROADCAST_K"], "128")

    # Conflicting or malformed PATHWAYS_WORKER_EXTRA_ENV raises ValueError
    for bad_extra_env in (
        "RAIDEN_BROADCAST_K=128",
        "TPU_MIN_LOG_LEVEL=2",
        "MALFORMED_LINE_WITHOUT_EQUALS",
        "=empty_key",
    ):
      with mock.patch.dict(
          os.environ, {"PATHWAYS_WORKER_EXTRA_ENV": bad_extra_env}
      ):
        with mock.patch.object(
            sys,
            "argv",
            [
                "yaml_generator.py",
                pw_template,
                "--jobset_name=test-pw-extra",
                "--tpu_slice=tpuv5e:4x4",
            ],
        ):
          with self.assertRaises(ValueError):
            yaml_generator.main()

  def test_generate_mcjax_ray_yaml(self):
    ray_template = _get_template_path("jobset.mcjax.ray.yaml")
    argv = [
        "yaml_generator.py",
        ray_template,
        "--jobset_name=test-ray-job",
        "--tpu_slice=tpuv5e:4x4",
        "--ray_head_pre_init_command=echo 'head pre-init'",
        "--ray_worker_pre_init_command=echo 'worker pre-init'",
        "--fail_fast",
        "--startup_retries=5",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("echo 'head pre-init'", rendered)
        self.assertIn("echo 'worker pre-init'", rendered)
        self.assertIn('grep -E -q "/16(\\.0)? TPU"', rendered)
        self.assertIn("ERROR: Worker exited with code 0 unexpectedly", rendered)
        doc = yaml.safe_load(rendered)
        self.assertEqual(doc["spec"]["failurePolicy"]["maxRestarts"], 5)
        job_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"]
        self.assertEqual(job_spec["backoffLimit"], 0)
        pod_spec = job_spec["template"]["spec"]
        self.assertEqual(pod_spec["restartPolicy"], "Never")
        self.assertTrue(any(v["name"] == "dshm" for v in pod_spec["volumes"]))

  def _render_mcjax_ray_script(
      self,
      *,
      head_pre_init: str = "",
      worker_pre_init: str = "",
      startup_cmd: str = "sleep infinity",
  ) -> str:
    ray_template = _get_template_path("jobset.mcjax.ray.yaml")
    with open(ray_template, "r") as f:
      template_text = f.read()
    args = yaml_generator.parse_args([
        ray_template,
        "--jobset_name=roll",
        "--tpu_slice=tpuv5e:4x4",
        "--fail_fast",
        f"--ray_head_pre_init_command={head_pre_init}",
        f"--ray_worker_pre_init_command={worker_pre_init}",
        f"--worker_startup_command={startup_cmd}",
    ])
    rendered = yaml_generator.render_template(template_text, args)
    doc = yaml.safe_load(rendered)
    return doc["spec"]["replicatedJobs"][0]["template"]["spec"]["template"][
        "spec"
    ]["containers"][0]["command"][2]

  def test_mcjax_ray_worker_pid1_script_fails_fast_on_crash(self):
    pre_init_cmd = (
        "STATUS_CALLS=0; "
        "START_CALLS=0; "
        "sleep() { :; }; "
        "ray() { "
        'if [ "$1" = "status" ]; then '
        "STATUS_CALLS=$((STATUS_CALLS + 1)); "
        '[ "$STATUS_CALLS" -ge 2 ]; return $?; '
        'elif [ "$1" = "start" ]; then '
        "START_CALLS=$((START_CALLS + 1)); "
        'echo "RAY_START_CALL=$START_CALLS args=$*"; '
        "return 1; "
        "fi; "
        "}"
    )
    script = self._render_mcjax_ray_script(worker_pre_init=pre_init_cmd)
    env = dict(
        os.environ,
        JOBSET_NAME="roll",
        REPLICATED_JOB_NAME="worker",
        JOB_INDEX="0",
        POD_INDEX="1",
    )
    res = subprocess.run(
        ["bash", "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(res.returncode, 1)
    self.assertIn(
        "Ray head roll-worker-0-0.roll not ready, retrying in 5s...",
        res.stdout,
    )
    self.assertIn(
        "RAY_START_CALL=1 args=start --address=roll-worker-0-0.roll:6379"
        " --block",
        res.stdout,
    )
    self.assertNotIn("RAY_START_CALL=2", res.stdout)

  def test_mcjax_ray_head_and_worker_startup_failures_and_timeouts(self):
    # 1. Head pod fails fast when ray start --head fails
    head_fail_script = self._render_mcjax_ray_script(
        head_pre_init="ray() { return 1; }",
        startup_cmd="echo WORKLOAD_RAN",
    )
    res_head_fail = subprocess.run(
        ["bash", "-c", head_fail_script],
        env=dict(
            os.environ,
            JOBSET_NAME="roll",
            REPLICATED_JOB_NAME="worker",
            JOB_INDEX="0",
            POD_INDEX="0",
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(res_head_fail.returncode, 1)
    self.assertNotIn("WORKLOAD_RAN", res_head_fail.stdout)

    # 2. Head pod times out when ray status never reaches NUM_CHIPS
    head_timeout_script = self._render_mcjax_ray_script(
        head_pre_init=(
            'ray() { if [ "$1" = "start" ]; then return 0; fi; '
            'echo "0.0/8.0 TPU"; return 0; }'
        ),
        startup_cmd="echo WORKLOAD_RAN",
    )
    res_head_timeout = subprocess.run(
        ["bash", "-c", head_timeout_script],
        env=dict(
            os.environ,
            JOBSET_NAME="roll",
            REPLICATED_JOB_NAME="worker",
            JOB_INDEX="0",
            POD_INDEX="0",
            RAY_CLUSTER_WAIT_TIMEOUT_S="0",
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(res_head_timeout.returncode, 1)
    self.assertIn(
        "Timed out after 0s waiting for 16 TPUs", res_head_timeout.stderr
    )
    self.assertNotIn("WORKLOAD_RAN", res_head_timeout.stdout)

    # 3. Worker pod times out when Ray head never becomes reachable
    worker_timeout_script = self._render_mcjax_ray_script(
        worker_pre_init=(
            'ray() { if [ "$1" = "status" ]; then return 1; fi; '
            'echo "RAY_START_RAN"; return 0; }'
        ),
    )
    res_worker_timeout = subprocess.run(
        ["bash", "-c", worker_timeout_script],
        env=dict(
            os.environ,
            JOBSET_NAME="roll",
            REPLICATED_JOB_NAME="worker",
            JOB_INDEX="0",
            POD_INDEX="1",
            RAY_CLUSTER_WAIT_TIMEOUT_S="0",
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    self.assertEqual(res_worker_timeout.returncode, 1)
    self.assertIn(
        "Timed out after 0s waiting for Ray head", res_worker_timeout.stderr
    )
    self.assertNotIn("RAY_START_RAN", res_worker_timeout.stdout)

  def test_mcjax_ray_sigterm_drains_child_and_exits_143(self):
    # Redirect stdout/stderr to files and use proc.wait() (not communicate(),
    # which blocks on pipe EOF from orphaned grandchildren) so this test fails
    # if the subshell exits before its child finishes draining.
    with tempfile.TemporaryDirectory() as tmpdir:
      ready_file = os.path.join(tmpdir, "ready")
      drain_marker = os.path.join(tmpdir, "drained")
      stdout_path = os.path.join(tmpdir, "stdout.log")
      stderr_path = os.path.join(tmpdir, "stderr.log")
      child_sh = (
          "sleep 30 >/dev/null 2>&1 & SPID=$!; "
          "trap 'kill $SPID 2>/dev/null; sleep 0.4; "
          f"echo drained > {shlex.quote(drain_marker)}; exit 0' SIGTERM; "
          f"echo 1 > {shlex.quote(ready_file)}; "
          "wait $SPID"
      )
      script = self._render_mcjax_ray_script(
          worker_pre_init=(
              'ray() { if [ "$1" = "status" ]; then return 0; fi; '
              f"bash -c {shlex.quote(child_sh)}; "
              "}"
          ),
      )
      with (
          open(stdout_path, "wb") as stdout_f,
          open(stderr_path, "wb") as stderr_f,
      ):
        proc = subprocess.Popen(
            ["bash", "-c", script],
            env=dict(
                os.environ,
                JOBSET_NAME="roll",
                REPLICATED_JOB_NAME="worker",
                JOB_INDEX="0",
                POD_INDEX="1",
            ),
            stdout=stdout_f,
            stderr=stderr_f,
        )
        deadline = time.monotonic() + 5.0
        while not os.path.exists(ready_file):
          self.assertLess(time.monotonic(), deadline)
          time.sleep(0.02)
        proc.send_signal(signal.SIGTERM)
        rc = proc.wait(timeout=10)
      self.assertTrue(os.path.exists(drain_marker))
      self.assertEqual(rc, 143)
      with open(stderr_path, "r") as f:
        stderr = f.read()
      self.assertNotIn("Worker exited with code 0 unexpectedly", stderr)

  def test_standalone_placeholder_formatting_and_multiline_commands(self):
    # 1. Empty optional standalone placeholders leave no blank lines
    cpu_template = _get_template_path("jobset.cpu.yaml")
    with open(cpu_template, "r") as f:
      cpu_text = f.read()
    cpu_rendered = yaml_generator.render_template(
        cpu_text,
        yaml_generator.parse_args([
            cpu_template,
            "--jobset_name=test-cpu",
            "--cpu_machine=n2-standard-64",
            "--worker_startup_command=echo line1\necho line2",
        ]),
    )
    self.assertIn("  namespace: default\nspec:\n", cpu_rendered)
    self.assertIn(
        "          spec:\n            dnsPolicy: ClusterFirstWithHostNet\n",
        cpu_rendered,
    )
    self.assertIn(
        "              securityContext:\n                privileged: true\n"
        "              command:\n",
        cpu_rendered,
    )
    self.assertIn(
        "                    echo line1\n                    echo line2\n",
        cpu_rendered,
    )
    doc = yaml.safe_load(cpu_rendered)
    self.assertIsNotNone(doc)

  def test_pathways_head_user_container_mounts_memory_dshm(self):
    pathways_template = _get_template_path("jobset.pathways.yaml")
    with open(pathways_template, "r") as f:
      pathways_text = f.read()
    rendered = yaml_generator.render_template(
        pathways_text,
        yaml_generator.parse_args([
            pathways_template,
            "--jobset_name=test-pathways",
            "--tpu_slice=tpuv5:4x4x4",
        ]),
    )
    doc = yaml.safe_load(rendered)
    proc_spec = doc["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    self.assertIn(
        {"name": "dshm", "emptyDir": {"medium": "Memory"}},
        proc_spec["volumes"],
    )
    user_mounts = proc_spec["containers"][0]["volumeMounts"]
    self.assertIn({"name": "dshm", "mountPath": "/dev/shm"}, user_mounts)


if __name__ == "__main__":
  absltest.main()
