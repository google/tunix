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

from __future__ import annotations

import os
from unittest import mock

from absl.testing import absltest
from examples.deepswe import template
from tunix.experimental.examples.deepswe_dist import prepull_sandbox_images


class PrepullSandboxImagesTest(absltest.TestCase):

  def test_collect_static_sandbox_images_is_order_independent(self):
    tasks_order_a = [
        {"docker_image": "namanjain12/sympy_final:img3"},
        {"docker_image": "namanjain12/sympy_final:img1"},
        {"docker_image": "namanjain12/sympy_final:img2"},
        {"docker_image": "namanjain12/sympy_final:img1"},
    ]
    tasks_order_b = list(reversed(tasks_order_a))

    with mock.patch.dict(
        os.environ,
        {
            "IMAGE_REWRITE_PREFIX": "us-central1-docker.pkg.dev/proj/tunix/",
            "OPENHANDS_SERVER_IMAGE": "us-central1-docker.pkg.dev/proj/oh:v1",
        },
        clear=False,
    ):
      imgs_a = prepull_sandbox_images.collect_static_sandbox_images(
          tasks_order_a, scaffold="openhands"
      )
      imgs_b = prepull_sandbox_images.collect_static_sandbox_images(
          tasks_order_b, scaffold="openhands"
      )

    self.assertEqual(imgs_a, imgs_b)
    self.assertEqual(
        imgs_a,
        [
            "us-central1-docker.pkg.dev/proj/oh:v1",
            "us-central1-docker.pkg.dev/proj/tunix/sympy_final:img1",
            "us-central1-docker.pkg.dev/proj/tunix/sympy_final:img2",
            "us-central1-docker.pkg.dev/proj/tunix/sympy_final:img3",
        ],
    )

  def test_build_prepull_and_drop_caches_manifests(self):
    ds = prepull_sandbox_images.build_prepull_daemonset_manifest(
        name="prepull-w0",
        namespace="trellis",
        images=["img-a:1", "img-b:2"],
        run_id="run-123",
        node_selector={"workload": "sandbox"},
        tolerations=[{"key": "workload", "value": "sandbox"}],
    )
    pod_spec = ds["spec"]["template"]["spec"]
    self.assertEqual(pod_spec["nodeSelector"], {"workload": "sandbox"})
    self.assertLen(pod_spec["initContainers"], 2)
    for c in pod_spec["initContainers"]:
      self.assertEqual(c["imagePullPolicy"], "IfNotPresent")
    self.assertEqual(
        pod_spec["containers"][0]["imagePullPolicy"], "IfNotPresent"
    )

    drop_ds = prepull_sandbox_images.build_drop_caches_daemonset_manifest(
        name="prepull-drop",
        namespace="trellis",
        image="img-a:1",
        run_id="run-123",
    )
    init_c = drop_ds["spec"]["template"]["spec"]["initContainers"][0]
    self.assertEqual(init_c["imagePullPolicy"], "IfNotPresent")
    self.assertTrue(init_c["securityContext"]["privileged"])
    self.assertIn("vm.drop_caches=3", init_c["command"][2])

  def test_prepull_sandbox_images_stages_waves_drops_caches_and_cleans_up(self):
    created_ds = []
    deleted_ds = []

    apps_api = mock.MagicMock()
    core_api = mock.MagicMock()

    def _create_ds(namespace, body):
      del namespace
      created_ds.append(body["metadata"]["name"])

    def _read_ds(name, namespace):
      del name, namespace
      ds = mock.MagicMock()
      ds.status.desired_number_scheduled = 4
      ds.status.number_ready = 4
      return ds

    def _delete_ds(name, namespace, **kwargs):
      del namespace, kwargs
      deleted_ds.append(name)

    apps_api.create_namespaced_daemon_set.side_effect = _create_ds
    apps_api.read_namespaced_daemon_set.side_effect = _read_ds
    apps_api.delete_namespaced_daemon_set.side_effect = _delete_ds
    core_api.list_namespaced_pod.return_value = mock.MagicMock(items=[])

    count = prepull_sandbox_images.prepull_sandbox_images(
        ["img-3", "img-1", "img-2"],
        namespace="trellis",
        run_id="testrun",
        images_per_pod=2,
        max_parallel_daemonsets=1,
        drop_caches=True,
        apps_api=apps_api,
        core_api=core_api,
        sleep=lambda _: None,
    )
    self.assertEqual(count, 3)
    self.assertEqual(
        created_ds,
        ["testrun-w0", "testrun-w1", "testrun-drop-caches"],
    )
    self.assertEqual(
        deleted_ds,
        ["testrun-w0", "testrun-w1", "testrun-drop-caches"],
    )
    self.assertGreaterEqual(
        apps_api.delete_collection_namespaced_daemon_set.call_count, 2
    )
    self.assertGreaterEqual(
        core_api.delete_collection_namespaced_pod.call_count, 2
    )

  def test_unblock_kueue_gated_pods_patches_only_gated_pods(self):
    gated_pod = mock.MagicMock()
    gated_pod.metadata.name = "prepull-w0-abc"
    gated_pod.spec.scheduling_gates = [mock.MagicMock(name="kueue.x-k8s.io/admission")]

    ungated_pod = mock.MagicMock()
    ungated_pod.metadata.name = "prepull-w0-def"
    ungated_pod.spec.scheduling_gates = None

    core_api = mock.MagicMock()
    core_api.list_namespaced_pod.return_value = mock.MagicMock(
        items=[gated_pod, ungated_pod]
    )

    unblocked = prepull_sandbox_images._unblock_kueue_gated_pods(
        core_api, namespace="priority-dev", run_id="testrun"
    )
    self.assertEqual(unblocked, 1)
    core_api.patch_namespaced_pod.assert_called_once_with(
        name="prepull-w0-abc",
        namespace="priority-dev",
        body={
            "metadata": {
                "labels": {"kueue.x-k8s.io/managed": None},
                "finalizers": None,
            },
            "spec": {"schedulingGates": None},
        },
    )

  def test_template_uses_if_not_present_pull_policy_by_default(self):
    self.assertEqual(template.get_image_pull_policy(), "IfNotPresent")

  def test_load_images_from_manifest_filters_and_rewrites(self):
    manifest_content = (
        "# Header comment\n"
        "namanjain12/sympy_final:img2\n"
        "\n"
        "namanjain12/sympy_final:img1\n"
        "namanjain12/sympy_final:img2\n"
        "  # Inline indented comment\n"
    )
    manifest_file = self.create_tempfile("images.txt", content=manifest_content)

    with mock.patch.dict(
        os.environ,
        {
            "IMAGE_REWRITE_PREFIX": "us-central1-docker.pkg.dev/proj/tunix/",
            "OPENHANDS_SERVER_IMAGE": "us-central1-docker.pkg.dev/proj/oh:v1",
        },
        clear=False,
    ):
      images = prepull_sandbox_images.load_images_from_manifest(
          manifest_file.full_path, scaffold="openhands"
      )

    self.assertEqual(
        images,
        [
            "us-central1-docker.pkg.dev/proj/oh:v1",
            "us-central1-docker.pkg.dev/proj/tunix/sympy_final:img1",
            "us-central1-docker.pkg.dev/proj/tunix/sympy_final:img2",
        ],
    )

  @mock.patch.object(prepull_sandbox_images, "prepull_sandbox_images")
  def test_prepull_from_manifest_calls_prepull_sandbox_images(
      self, mock_prepull
  ):
    mock_prepull.return_value = 2
    manifest_content = "namanjain12/aiohttp_final:v1\nnamanjain12/sympy_final:v2\n"
    manifest_file = self.create_tempfile("images.txt", content=manifest_content)

    num_staged = prepull_sandbox_images.prepull_from_manifest(
        manifest_file.full_path,
        scaffold="r2egym",
        namespace="test-ns",
        images_per_pod=10,
        max_parallel_daemonsets=2,
    )

    self.assertEqual(num_staged, 2)
    mock_prepull.assert_called_once()
    call_args, call_kwargs = mock_prepull.call_args
    self.assertEqual(
        call_args[0],
        ["namanjain12/aiohttp_final:v1", "namanjain12/sympy_final:v2"],
    )
    self.assertEqual(call_kwargs["namespace"], "test-ns")
    self.assertEqual(call_kwargs["images_per_pod"], 10)
    self.assertEqual(call_kwargs["max_parallel_daemonsets"], 2)


if __name__ == "__main__":
  absltest.main()
