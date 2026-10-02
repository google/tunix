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
        self.assertIn("cloud.google.com/gke-nodepool: cpu-np", rendered)
        self.assertIn("memory: 240G", rendered)

  def test_generate_cpu_yaml_custom_nodepool_and_memory(self):
    template_file = _get_template_path("jobset.cpu.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-cpu-highmem-job",
        "--cpu_nodepool=cpu-highmem-np",
        "--cpu_memory=300Gi",
    ]
    with mock.patch.object(sys, "argv", argv):
      with mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        yaml_generator.main()
        rendered = mock_stdout.getvalue()
        self.assertIn("test-cpu-highmem-job", rendered)
        self.assertIn("cloud.google.com/gke-nodepool: cpu-highmem-np", rendered)
        self.assertIn("memory: 300Gi", rendered)

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
    for template_name in ("jobset.pathways.yaml", "jobset.pathways.qwen3.5-397b.yaml"):
      with self.subTest(template_name=template_name):
        template_file = _get_template_path(template_name)
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
      (
          "gang_id_flag",
          ["--gang_id=atwigg-256-prof"],
          {},
          {"scheduling.x-k8s.io/gang-id": "atwigg-256-prof"},
      ),
      (
          "gang_id_env_with_preemptible_and_queue",
          ["--queue_name=multislice-queue"],
          {"PREEMPTIBLE": "true", "GANG_ID": "atwigg-256-prof"},
          {
              "kueue.x-k8s.io/queue-name": "multislice-queue",
              "scheduling.x-k8s.io/preemptible": "true",
              "scheduling.x-k8s.io/gang-id": "atwigg-256-prof",
          },
      ),
  )
  def test_preemptible_label(
      self,
      extra_args: list[str],
      env: dict[str, str],
      expected_labels: dict[str, str],
  ) -> None:
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

  def test_k8s_launcher_attaches_gang_id_to_all_jobsets(self) -> None:
    import yaml  # pylint: disable=g-import-not-at-top

    launcher_path = os.path.abspath(
        os.path.join(
            os.path.dirname(yaml_generator.__file__),
            "..",
            "..",
            "examples",
            "deepswe_dist",
            "k8s_launcher.sh",
        )
    )
    env = dict(
        os.environ,
        JOB_PREFIX="atwigg-256-prof",
        ROLLOUT_REPLICAS="2",
        KUEUE_QUEUE="multislice-queue",
        PREEMPTIBLE="true",
        USE_AGENT_SANDBOX="0",
        PYTHON_BIN=sys.executable,
    )
    env.pop("GANG_ID", None)
    result = subprocess.run(
        ["bash", launcher_path, "start", "--dry-run"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    docs = [
        d
        for d in yaml.safe_load_all(result.stdout)
        if isinstance(d, dict) and d.get("kind") == "JobSet"
    ]
    self.assertEqual(
        [d["metadata"]["name"] for d in docs],
        [
            "atwigg-256-prof-orch",
            "atwigg-256-prof-train",
            "atwigg-256-prof-roll-0",
            "atwigg-256-prof-roll-1",
        ],
    )
    for doc in docs:
      self.assertEqual(
          doc["metadata"]["labels"],
          {
              "kueue.x-k8s.io/queue-name": "multislice-queue",
              "scheduling.x-k8s.io/preemptible": "true",
              "scheduling.x-k8s.io/gang-id": "atwigg-256-prof",
          },
      )

  @parameterized.named_parameters(
      ("contains_at", "user@example.com"),
      ("starts_with_hyphen", "-invalid"),
      ("ends_with_hyphen", "invalid-"),
      ("too_long", "a" * 64),
  )
  def test_invalid_gang_id_raises(self, invalid_gang_id: str) -> None:
    template_file = _get_template_path("jobset.pathways.yaml")
    argv = [
        "yaml_generator.py",
        template_file,
        "--jobset_name=test-invalid-gang-id",
        "--tpu_slice=tpu7x:4x4x8",
        f"--gang_id={invalid_gang_id}",
    ]
    with mock.patch.object(sys, "argv", argv):
      with self.assertRaisesRegex(ValueError, "Invalid gang_id"):
        yaml_generator.main()


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
