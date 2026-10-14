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

"""End-to-end test exercising DeepSWE dataset loading and sandbox lifecycle on Kubernetes.

Verifies the full pipeline:
  1. Dataset loading (DeepSWE R2E-Gym dataset)
  2. Fleet initialization
  3. Lookahead dynamic pre-warming with PrewarmDatasetIterator
  4. Sandbox checkout / acquire via SWEEnv(use_agent_sandbox=True)
  5. Live command execution inside the warm sandbox pod
  6. Sandbox release back to fleet on env.close()
  7. Dynamic retirement / unwarming of finished batch pools as iterator advances
  8. Clean fleet teardown
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import random
import sys
import threading
import time
from typing import Any

from absl.testing import absltest

# pylint: disable=g-import-not-at-top
try:
  from tunix.experimental.examples.deepswe_dist import deepswe
except ImportError:
  try:
    import deepswe
  except ImportError:
    deepswe = None

try:
  from examples.deepswe import swe_env
except ImportError:
  try:
    from examples.deepswe import swe_env
  except ImportError:
    import swe_env

try:
  from examples.deepswe import sandbox_utils
except ImportError:
  try:
    from examples.deepswe import sandbox_utils
  except ImportError:
    import sandbox_utils


class FakeFleet:
  """Mock SandboxFleet tracking calls for local verification without a live GKE cluster."""

  def __init__(self):
    self._lock = threading.Lock()
    self.warm_calls: list[tuple[str, int | None, bool]] = []
    self.set_replicas_calls: list[tuple[str, int]] = []
    self.unwarm_calls: list[str] = []
    self.active_pools: dict[str, int] = {}
    self.acquired_tasks: list[Any] = []
    self.released_handles: list[Any] = []

  def warm_image(
      self, image: str, replicas_override: int | None = None, wait: bool = False
  ) -> None:
    with self._lock:
      self.warm_calls.append((image, replicas_override, wait))
      self.active_pools[image] = replicas_override or 1

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    with self._lock:
      self.set_replicas_calls.append((image, replicas))
      self.active_pools[image] = replicas

  def unwarm_image(self, image: str) -> None:
    with self._lock:
      self.unwarm_calls.append(image)
      self.active_pools.pop(image, None)

  def acquire(self, task: Any) -> Any:
    with self._lock:
      self.acquired_tasks.append(task)

    class FakeHandle:

      def __init__(self, t):
        self.task = t
        self.pod_name = f"pod-{getattr(t, 'image', 'default')}-claim"

    return FakeHandle(task)

  def release(self, handle: Any) -> None:
    with self._lock:
      self.released_handles.append(handle)

  def teardown(self) -> None:
    with self._lock:
      self.active_pools.clear()


def create_synthetic_dataset(
    num_samples: int = 12,
    default_image: str = "numpy_final:141d3a954b955b7f0821574e03b693ec4078640b",
) -> list[dict[str, Any]]:
  """Creates synthetic DeepSWE task records for testing."""
  test_images = [
      default_image,
      "pandas_final:6b575b4644bd1808808ce0270413c6e75ff7427c",
      "pyramid_final:f3bffdfc35a5ecbb45b5f63bdb08bdc41553b63d",
      "sympy_final:b39e65839b1a5ef8c8db182239d22c95e54d7e97",
  ]
  samples = []
  synthetic_commit = (
      '{"file_diffs": [], "old_commit_hash": "0000000", "new_commit_hash":'
      ' "1111111", "commit_message": "synthetic test", "commit_date":'
      ' "2026-01-01T00:00:00"}'
  )
  for i in range(num_samples):
    img = test_images[i % len(test_images)]
    samples.append({
        "instance_id": f"task-sample-{i}",
        "docker_image": img,
        "repo_name": img.split("_")[0],
        "prompt": f"Fix issue #{i} in repository",
        "parsed_commit_content": synthetic_commit,
        "expected_output_json": "{}",
        "metadata": {
            "env_config": {
                "entry": {
                    "docker_image": img,
                }
            }
        },
    })
  return samples


def _verify_openhands_harness_in_sandbox(env: Any) -> None:
  """Verifies OpenHands tools (editor, bash heredocs/comments, LD_LIBRARY_PATH, think, task_tracker, finish, reward) on a live sandbox."""

  def _obs(step_res: Any) -> str:
    if isinstance(step_res, tuple):
      return str(step_res[0])
    return str(getattr(step_res, "observation", step_res))

  img = str(getattr(env, "entry", {}).get("docker_image", ""))

  # 1. execute_bash: heredoc, python heredoc, trailing comment, clean LD_LIBRARY_PATH, and R2E test hiding
  bash_obs = _obs(
      env.step(
          "<function=execute_bash>\n"
          "<parameter=command>\n"
          "cat << 'EOF'\n"
          "heredoc_ok\n"
          "EOF\n"
          "python << 'PYEOF'\n"
          "import sys\n"
          "print('py_heredoc_ok', sys.version_info[:2])\n"
          "PYEOF\n"
          'echo "ld=${LD_LIBRARY_PATH-unset}"\n'
          'if [ -e /r2e_tests ] || [ -e /root/run_tests.sh ]; then echo "r2e_leaked"; else echo "r2e_hidden"; fi  # trailing comment\n'
          "</parameter>\n"
          "</function>"
      )
  )
  assert "heredoc_ok" in bash_obs, f"Heredoc failed: {bash_obs!r}"
  assert "py_heredoc_ok" in bash_obs, f"Python heredoc failed: {bash_obs!r}"
  assert "/tmp/_MEI" not in bash_obs, f"PyInstaller LD_LIBRARY_PATH leaked: {bash_obs!r}"
  assert "/oh/glibc236" not in bash_obs, f"glibc236 LD_LIBRARY_PATH leaked: {bash_obs!r}"
  assert "r2e_hidden" in bash_obs, f"R2E grading tests not hidden during rollout: {bash_obs!r}"
  assert "[Command finished with exit code 0]" in bash_obs, f"Unexpected bash suffix: {bash_obs!r}"

  # 2. execute_bash: working directory persistence across turns & repo import (C/C++ extensions)
  _ = env.step(
      "<function=execute_bash>\n"
      "<parameter=command>cd /tmp</parameter>\n"
      "</function>"
  )
  cwd_obs = _obs(
      env.step(
          "<function=execute_bash>\n"
          "<parameter=command>pwd && cd /testbed</parameter>\n"
          "</function>"
      )
  )
  assert cwd_obs.startswith("/tmp\n"), f"CWD did not persist across steps: {cwd_obs!r}"
  assert "[Current working directory: /testbed]" in cwd_obs, f"CWD metadata mismatch: {cwd_obs!r}"

  for pkg in ("pandas", "numpy", "pyramid", "sympy", "tornado", "PIL"):
    if pkg.lower() in img.lower() or (pkg == "PIL" and "pillow" in img.lower()):
      imp_obs = _obs(
          env.step(
              "<function=execute_bash>\n"
              f"<parameter=command>python -c 'import {pkg}; print(\"{pkg}_import_ok\")' && pytest --version</parameter>\n"
              "</function>"
          )
      )
      assert f"{pkg}_import_ok" in imp_obs, (
          f"Failed to import {pkg} in sandbox ({img}): {imp_obs!r}"
      )
      ipy_obs = _obs(
          env.step(
              "<function=execute_ipython_cell>\n"
              f"<parameter=code>import {pkg}; print('{pkg}_ipython_ok')</parameter>\n"
              "</function>"
          )
      )
      assert f"{pkg}_ipython_ok" in ipy_obs, (
          f"Failed to import {pkg} via execute_ipython_cell ({img}): {ipy_obs!r}"
      )

  # 3. str_replace_editor: create, view, str_replace (fuzzy), insert, undo_edit, binary view
  probe_path = "/tmp/_e2e_oh_editor_probe.py"
  pyc_path = "/tmp/_e2e_oh_editor_probe.pyc"
  create_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>create</parameter>\n"
          f"<parameter=path>{probe_path}</parameter>\n"
          "<parameter=file_text>def calc(x):\n    return x + 1\n</parameter>\n"
          "</function>"
      )
  )
  assert "File created successfully" in create_obs, f"Editor create failed: {create_obs!r}"

  view_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>view</parameter>\n"
          f"<parameter=path>{probe_path}</parameter>\n"
          "<parameter=view_range>[1, 2]</parameter>\n"
          "</function>"
      )
  )
  assert "return x + 1" in view_obs, f"Editor view failed: {view_obs!r}"

  replace_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>str_replace</parameter>\n"
          f"<parameter=path>{probe_path}</parameter>\n"
          "<parameter=old_str>  return x + 1</parameter>\n"
          "<parameter=new_str>  return x + 2</parameter>\n"
          "</function>"
      )
  )
  assert "has been edited" in replace_obs and "return x + 2" in replace_obs, (
      f"Editor fuzzy str_replace failed: {replace_obs!r}"
  )

  insert_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>insert</parameter>\n"
          f"<parameter=path>{probe_path}</parameter>\n"
          "<parameter=insert_line>1</parameter>\n"
          "<parameter=new_str>    # inserted line</parameter>\n"
          "</function>"
      )
  )
  assert "has been edited" in insert_obs and "# inserted line" in insert_obs, (
      f"Editor insert failed: {insert_obs!r}"
  )

  undo_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>undo_edit</parameter>\n"
          f"<parameter=path>{probe_path}</parameter>\n"
          "</function>"
      )
  )
  assert "undone successfully" in undo_obs and "# inserted line" not in undo_obs, (
      f"Editor undo_edit failed: {undo_obs!r}"
  )

  _ = env.step(
      "<function=execute_bash>\n"
      f"<parameter=command>cp {probe_path} {pyc_path} && rm -f {probe_path}</parameter>\n"
      "</function>"
  )
  bin_obs = _obs(
      env.step(
          "<function=str_replace_editor>\n"
          "<parameter=command>view</parameter>\n"
          f"<parameter=path>{pyc_path}</parameter>\n"
          "</function>"
      )
  )
  assert bin_obs.startswith("ERROR_BINARY_FILE"), f"Binary view check failed: {bin_obs!r}"
  _ = env.step(
      "<function=execute_bash>\n"
      f"<parameter=command>rm -f {pyc_path}</parameter>\n"
      "</function>"
  )

  # 4. think + task_tracker + finish
  think_obs = _obs(
      env.step(
          "<function=think>\n"
          "<parameter=thought>Verifying think tool</parameter>\n"
          "</function>"
      )
  )
  assert think_obs == "Your thought has been logged.", f"Think failed: {think_obs!r}"

  plan_obs = _obs(
      env.step(
          "<function=task_tracker>\n"
          "<parameter=command>plan</parameter>\n"
          '<parameter=task_list>[{"title": "Verify harness", "status": "done", "notes": "ok"}]</parameter>\n'
          "</function>"
      )
  )
  assert "Task list has been updated with 1 items." in plan_obs, (
      f"task_tracker plan failed: {plan_obs!r}"
  )

  finish_res = env.step(
      "<function=finish>\n"
      "<parameter=message>Verification complete</parameter>\n"
      "</function>"
  )
  finish_done = (
      finish_res[2]
      if isinstance(finish_res, tuple)
      else getattr(finish_res, "done", False)
  )
  assert finish_done is True, f"finish tool did not mark done=True: {finish_res!r}"

  # 5. Verify fresh-container reward computation, pod security, and rollout/eval container isolation
  try:
    from examples.deepswe import openhands_utils
  except ImportError:
    import openhands_utils

  # 5a. Inspect live K8s Pod spec if handle exposes pod_name
  pod_name = getattr(getattr(env, "handle", None), "pod_name", None)
  if pod_name:
    try:
      from kubernetes import client, config

      try:
        config.load_incluster_config()
      except Exception:
        config.load_kube_config()
      ns = os.environ.get("NAMESPACE", "trellis")
      pod = client.CoreV1Api().read_namespaced_pod(pod_name, ns)
      assert pod.spec.automount_service_account_token is False, (
          f"Expected automountServiceAccountToken=False on {pod_name}, got"
          f" {pod.spec.automount_service_account_token!r}"
      )
      assert pod.spec.share_process_namespace is False, (
          f"Expected shareProcessNamespace=False on {pod_name}, got"
          f" {pod.spec.share_process_namespace!r}"
      )
      container_names = [c.name for c in pod.spec.containers]
      assert container_names == ["agent-runtime", "eval"], (
          f"Expected containers ['agent-runtime', 'eval'] on {pod_name}, got"
          f" {container_names!r}"
      )
      eval_c = pod.spec.containers[1]
      assert list(eval_c.command or []) == ["sleep", "infinity"], (
          f"Expected eval container command ['sleep', 'infinity'], got"
          f" {eval_c.command!r}"
      )
      assert not eval_c.volume_mounts, (
          f"Expected eval container to have no volumeMounts, got"
          f" {eval_c.volume_mounts!r}"
      )
      logging.info(
          "      [OK] Pod %s spec verified: containers=%s,"
          " automountServiceAccountToken=False, shareProcessNamespace=False",
          pod_name,
          container_names,
      )
    except ImportError:
      pass

  # 5b. Verify R2E tests, grading stash, and K8s SA token are absent in rollout container
  rollout_check_obs = _obs(
      env.step(
          "<function=execute_bash>\n"
          "<parameter=command>\n"
          "for p in /r2e_tests /root/r2e_tests /testbed/r2e_tests"
          " /root/run_tests.sh /testbed/run_tests.sh"
          " /testbed/expected_test_output.json /root/expected_test_output.json"
          " /var/tmp/.r2e_grading_stash"
          " /var/run/secrets/kubernetes.io/serviceaccount; do\n"
          '  if [ -e "$p" ] || [ -L "$p" ]; then echo "leaked:$p"; fi\n'
          "done\n"
          'echo "isolation_check_done"\n'
          "</parameter>\n"
          "</function>"
      )
  )
  assert "leaked:" not in rollout_check_obs, (
      f"Sensitive path leaked into rollout container: {rollout_check_obs!r}"
  )
  assert "isolation_check_done" in rollout_check_obs

  # 5c. Verify unmodified workspace extracts empty patch "" (ignoring bash_events) and scores 0.0
  base_commit = openhands_utils.resolve_base_commit(env.entry)
  empty_patch = openhands_utils.extract_agent_patch(
      env.workspace or env.env,
      base_commit=base_commit,
      workspace_path="/testbed",
  )
  assert empty_patch == "", (
      f"Expected empty patch on unmodified repo, got:\n{empty_patch[:500]!r}"
  )
  if getattr(env, "final_reward_fn", None) is not None:
    reward_empty = env.final_reward_fn()
    assert reward_empty == 0.0, (
        f"Expected 0.0 reward on empty patch, got {reward_empty!r}"
    )
    if getattr(env, "env", None) is not None and getattr(env.env, "runtime", None) is not None:
      assert getattr(env.env.runtime, "_needs_deferred_setup_env", None) is True, (
          "Expected eval container setup_env() to remain deferred after empty patch"
      )
    logging.info("      [OK] Unmodified workspace extracted empty patch and scored 0.0")

  # 5d. Mutate rollout container (.venv, gitignored .so, binary file, nested .git, background process)
  # and apply gold patch (if available) to verify clean extraction and fresh eval container grading
  gold_patch = ""
  commit_obj = getattr(getattr(getattr(env, "env", None), "runtime", None), "commit", None)
  if commit_obj is not None and hasattr(commit_obj, "get_patch"):
    try:
      gold_patch = commit_obj.get_patch(test_file=False, non_test_file=True) or ""
    except Exception as e:
      logging.warning("Could not extract gold_patch from commit: %s", e)

  import base64

  gold_patch_b64 = base64.b64encode(gold_patch.encode("utf-8")).decode("ascii")
  mutate_obs = _obs(
      env.step(
          "<function=execute_bash>\n"
          "<parameter=command>\n"
          "mkdir -p /testbed/.venv && echo 'POISON = 1' > /testbed/.venv/_mutated_venv_marker.py\n"
          "echo '*.so' >> /testbed/.git/info/exclude\n"
          "echo 'fake_shared_object' > /testbed/_gitignored_ext.so\n"
          "python -c \"open('/testbed/_untracked_binary.bin', 'wb').write(b'\\x00\\x01\\x02\\x03binary')\"\n"
          "mkdir -p /testbed/_nested_git/.git && echo 'ref: refs/heads/main' > /testbed/_nested_git/.git/HEAD\n"
          "echo 'NESTED_OK = True' > /testbed/_nested_git/nested_code.py\n"
          "nohup sleep 31337 >/dev/null 2>&1 &\n"
          "echo $! > /tmp/_bg_sleep.pid\n"
          "sleep 0.2\n"
          '_bg_state=$(awk \'/^State:/ {print $2}\' "/proc/$(cat /tmp/_bg_sleep.pid)/status" 2>/dev/null || echo "gone")\n'
          'if [ "$_bg_state" != "gone" ] && [ "$_bg_state" != "Z" ]; then echo "bg_proc_running:$_bg_state"; fi\n'
          f"if [ -n '{gold_patch_b64}' ]; then\n"
          f"  echo '{gold_patch_b64}' | base64 -d > /tmp/_gold.patch\n"
          "  git -C /testbed apply /tmp/_gold.patch && echo 'gold_applied_ok'\n"
          "  rm -f /tmp/_gold.patch\n"
          "fi\n"
          "</parameter>\n"
          "</function>"
      )
  )
  assert "bg_proc_running" in mutate_obs, (
      f"Background process did not start in rollout container: {mutate_obs!r}"
  )
  if gold_patch.strip():
    assert "gold_applied_ok" in mutate_obs, (
        f"Failed to apply gold patch in rollout container: {mutate_obs!r}"
    )

  extracted_patch = openhands_utils.extract_agent_patch(
      env.workspace or env.env,
      base_commit=base_commit,
      workspace_path="/testbed",
  )
  assert "_mutated_venv_marker.py" not in extracted_patch, (
      "Mutated .venv leaked into extracted patch!"
  )
  assert "_gitignored_ext.so" not in extracted_patch, (
      "Gitignored .so file leaked into extracted patch!"
  )
  assert "_untracked_binary.bin" not in extracted_patch, (
      "Binary file leaked into extracted patch!"
  )
  assert "_nested_git/.git" not in extracted_patch, (
      "Nested .git directory leaked into extracted patch!"
  )
  assert "bash_events" not in extracted_patch, (
      "OpenHands bash_events leaked into extracted patch!"
  )
  assert "_nested_git/nested_code.py" in extracted_patch, (
      "Expected tracked/untracked text file inside _nested_git (after nested .git removal) in patch!"
  )

  if getattr(env, "final_reward_fn", None) is not None and env.entry.get("parsed_commit_content"):
    reward = env.final_reward_fn()
    if gold_patch.strip():
      assert reward == 1.0, (
          f"Expected reward=1.0 for gold patch in fresh eval container on {img}, got {reward!r}"
      )
      logging.info("      [OK] Gold patch evaluated in fresh eval container scored 1.0 on %s", img)
    else:
      assert reward in (0.0, 1.0), f"Unexpected reward value: {reward!r}"

  # Verify background process was killed in rollout container while agent-server stayed alive
  # Note: When openhands-agent-server runs as PID 1 without tini, a killed orphan transitions to State=Z (zombie)
  # where kill -0 still returns 0 even though the process has terminated.
  bg_after_obs = _obs(
      env.step(
          "<function=execute_bash>\n"
          '<parameter=command>_bg_state=$(awk \'/^State:/ {print $2}\' "/proc/$(cat /tmp/_bg_sleep.pid)/status" 2>/dev/null || echo "gone"); '
          'if [ "$_bg_state" = "gone" ] || [ "$_bg_state" = "Z" ]; then echo "bg_killed:$_bg_state"; else echo "bg_still_alive:$_bg_state"; fi</parameter>\n'
          "</function>"
      )
  )
  assert "bg_killed" in bg_after_obs, (
      f"Background process was not killed in rollout container: {bg_after_obs!r}"
  )

  # Verify eval container has R2E tests, ran deferred setup_env(), and has none of rollout's unextracted mutations
  if getattr(env, "env", None) is not None and getattr(env.env, "runtime", None) is not None:
    assert getattr(env.env.runtime, "_needs_deferred_setup_env", None) is False, (
        "Expected eval container _needs_deferred_setup_env=False after non-empty patch evaluation"
    )
    eval_res = openhands_utils._exec_in_sandbox(
        env.env,
        "if [ -e /r2e_tests ] || [ -e /root/r2e_tests ] || [ -e /testbed/r2e_tests ]; then echo 'eval_tests_present'; else echo 'eval_tests_missing'; fi; "
        "if [ -e /testbed/.venv/_mutated_venv_marker.py ] || [ -e /testbed/_gitignored_ext.so ] || [ -e /testbed/_untracked_binary.bin ]; then echo 'eval_polluted'; else echo 'eval_clean'; fi; "
        "if [ -e /testbed/_nested_git/nested_code.py ]; then echo 'patch_applied_in_eval'; fi",
    )
    eval_out, _ = openhands_utils._unpack_exec_output(eval_res)
    assert "eval_tests_present" in str(eval_out), (
        f"R2E grading tests missing in eval container: {eval_out!r}"
    )
    assert "eval_clean" in str(eval_out), (
        f"Rollout container mutations polluted eval container: {eval_out!r}"
    )
    assert "patch_applied_in_eval" in str(eval_out), (
        f"Extracted patch was not applied in eval container: {eval_out!r}"
    )

  logging.info("      [OK] OpenHands full tool suite & fresh-container sandboxing verified on %s", img)


def run_pipeline_e2e(
    dataset: list[Any],
    fleet: Any,
    batch_size: int = 1,
    num_generations: int = 2,
    max_steps: int = 3,
    is_mock: bool = False,
    scaffold: str = "r2egym",
    minimum_step_time_in_second: float = 0.0,
    sampling_rate: float = 1.0,
) -> dict[str, Any]:
  """Runs the complete pre-warm, acquire, execute, release, and unwarm cycle."""
  logging.info("=== [DeepSWE E2E Test] Starting Sandbox Lifecycle Run ===")
  logging.info(
      "Parameters: batch_size=%d, num_generations=%d, max_steps=%d,"
      " is_mock=%s, scaffold=%s, minimum_step_time_in_second=%.2f,"
      " sampling_rate=%.2f",
      batch_size,
      num_generations,
      max_steps,
      is_mock,
      scaffold,
      minimum_step_time_in_second,
      sampling_rate,
  )

  # 1. Initialize PrewarmDatasetIterator
  logging.info("1. Initializing PrewarmDatasetIterator...")
  iterator = sandbox_utils.PrewarmDatasetIterator(
      dataset,
      fleet=fleet,
      num_generations=num_generations,
      batch_size=batch_size,
      unwarm_on_exhaustion=True,
      scaffold=scaffold,
  )
  logging.info(
      "   [OK] Priming complete. Active warm pools: %s",
      getattr(fleet, "active_pools", {}),
  )

  stats = {
      "steps_executed": 0,
      "batches_sampled": 0,
      "sandboxes_acquired": 0,
      "sandboxes_released": 0,
      "pools_unwarmed": 0,
      "avg_acquire_time_sec": 0.0,
      "avg_exec_time_sec": 0.0,
  }

  acquire_durations: list[float] = []
  exec_durations: list[float] = []
  print_lock = threading.Lock()

  def _execute_single_rollout(
      step: int,
      item_idx: int,
      batch_item: Any,
      gen_idx: int,
      total_items: int,
      n_gens: int,
  ) -> dict[str, Any]:
    try:
      if is_mock:
        task = sandbox_utils.normalize_tasks_for_fleet(
            [batch_item], scaffold=scaffold
        )[0]
        t_acq_start = time.perf_counter()
        handle = fleet.acquire(task)
        t_acq = time.perf_counter() - t_acq_start

        t_exec_start = time.perf_counter()
        step_res = (
            f"AUTHORS.rst LICENSE README.rst setup.py (gen={gen_idx})",
            0.0,
            True,
            {"max_steps": 1},
        )
        t_exec = time.perf_counter() - t_exec_start

        fleet.release(handle)
      else:
        t_acq_start = time.perf_counter()
        env = swe_env.SWEEnv(
            batch_item,
            fleet=fleet,
            use_agent_sandbox=True,
            scaffold=scaffold,
            max_steps=30,
        )
        _ = env.reset()
        t_acq = time.perf_counter() - t_acq_start

        t_exec_start = time.perf_counter()
        param_name = "command" if scaffold == "openhands" else "cmd"
        action = (
            "<function=execute_bash>\n"
            f"<parameter={param_name}>ls</parameter>\n"
            "</function>"
        )
        step_res = env.step(action)
        t_exec = time.perf_counter() - t_exec_start

        if scaffold == "openhands" and gen_idx == 0:
          _verify_openhands_harness_in_sandbox(env)

        env.close()

      # Print out immediately for the first generation of each item
      if gen_idx == 0:
        obs_text = (
            step_res[0]
            if isinstance(step_res, tuple)
            else getattr(step_res, "observation", str(step_res))
        )
        with print_lock:
          logging.info(
              "      [Batch Item %d/%d, Gen 0/%d] Sandbox acquired in %.3fs,"
              " command executed in %.3fs",
              item_idx + 1,
              total_items,
              n_gens,
              t_acq,
              t_exec,
          )
          logging.info(
              "      [OK] step_res (first 128 chars): %s",
              str(step_res)[:128],
          )
          print(
              f"      >>> [Batch Item {item_idx + 1}/{total_items},"
              f" Gen 0/{n_gens}] acq={t_acq:.3f}s, exec={t_exec:.3f}s",
              flush=True,
          )
          print(
              "      >>> [OK] step_res (first 128 chars):"
              f" {str(step_res)[:128]}",
              flush=True,
          )
          print(
              "      >>> [OK] observation (first 128 chars):"
              f" {repr(str(obs_text)[:128])}",
              flush=True,
          )

      return {
          "item_idx": item_idx,
          "gen_idx": gen_idx,
          "t_acq": t_acq,
          "t_exec": t_exec,
          "step_res": step_res,
      }
    except Exception as e:
      with print_lock:
        logging.error(
            "      [FAIL] Step %d Item %d/%d Gen %d/%d rollout failed: %s",
            step,
            item_idx + 1,
            total_items,
            gen_idx,
            n_gens,
            e,
        )
        print(
            f"      [FAIL] Step {step} Item {item_idx + 1}/{total_items} Gen"
            f" {gen_idx}/{n_gens} rollout failed: {e}",
            flush=True,
        )
      raise

  for step in range(max_steps):
    step_start_time = time.perf_counter()
    logging.info("--- [Step %d] Fetching next batch from iterator ---", step)
    batch_items = []
    for b_idx in range(batch_size):
      try:
        batch_items.append(next(iterator))
      except StopIteration:
        logging.info(
            "   Dataset iterator exhausted at step %d, item %d", step, b_idx
        )
        break

    if not batch_items:
      logging.info("   Dataset iterator exhausted at step %d", step)
      break

    stats["steps_executed"] += 1
    logging.info(
        "   [OK] Step %d fetched batch of %d items", step, len(batch_items)
    )

    should_test_batch = (sampling_rate >= 1.0) or (
        sampling_rate > 0.0 and random.random() < sampling_rate
    )

    if not should_test_batch:
      logging.info(
          "   [Step %d] Skipping sandbox acquire & execution"
          " (sampling_rate=%.2f).",
          step,
          sampling_rate,
      )
      print(
          f"   [Step {step}] Skipping sandbox acquire & execution"
          f" (sampling_rate={sampling_rate:.2f})",
          flush=True,
      )
    else:
      stats["batches_sampled"] += 1
      rollout_jobs = []
      for item_idx, batch_item in enumerate(batch_items):
        for gen_idx in range(num_generations):
          rollout_jobs.append((item_idx, batch_item, gen_idx))

      logging.info(
          "   [Step %d] Fanning out %d parallel sandbox acquire & execute tasks"
          " (%d items * %d generations)...",
          step,
          len(rollout_jobs),
          len(batch_items),
          num_generations,
      )
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=max(1, len(rollout_jobs))
      ) as executor:
        futures = [
            executor.submit(
                _execute_single_rollout,
                step,
                item_idx,
                b_item,
                gen_idx,
                len(batch_items),
                num_generations,
            )
            for item_idx, b_item, gen_idx in rollout_jobs
        ]
        results = [f.result() for f in futures]

      results.sort(key=lambda r: (r["item_idx"], r["gen_idx"]))
      step_acq_times = []
      step_exec_times = []
      for res in results:
        t_acq = res["t_acq"]
        t_exec = res["t_exec"]

        acquire_durations.append(t_acq)
        exec_durations.append(t_exec)
        step_acq_times.append(t_acq)
        step_exec_times.append(t_exec)
        stats["sandboxes_acquired"] += 1
        stats["sandboxes_released"] += 1

      step_avg_acq = (
          sum(step_acq_times) / len(step_acq_times) if step_acq_times else 0.0
      )
      step_avg_exec = (
          sum(step_exec_times) / len(step_exec_times)
          if step_exec_times
          else 0.0
      )
      logging.info(
          "   [Step %d Thread Summary] %d parallel rollouts: avg_acquire=%.3fs,"
          " avg_exec=%.3fs",
          step,
          len(results),
          step_avg_acq,
          step_avg_exec,
      )
      print(
          f"   [Step {step} Thread Summary] {len(results)} parallel rollouts:"
          f" avg_acquire={step_avg_acq:.3f}s, avg_exec={step_avg_exec:.3f}s",
          flush=True,
      )

    logging.info(
        "   - Active warm pools after step %d: %s",
        step,
        getattr(fleet, "active_pools", {}),
    )

    step_duration = time.perf_counter() - step_start_time
    if (
        minimum_step_time_in_second > 0
        and step < max_steps - 1
        and step_duration < minimum_step_time_in_second
    ):
      sleep_secs = minimum_step_time_in_second - step_duration
      logging.info(
          "   [Step %d] Finished in %.2fs (< %.2fs"
          " minimum_step_time_in_second). Waiting %.2fs before processing the"
          " next step...",
          step,
          step_duration,
          minimum_step_time_in_second,
          sleep_secs,
      )
      print(
          f"   [Step {step}] Finished in {step_duration:.2f}s (<"
          f" {minimum_step_time_in_second:.2f}s). Waiting {sleep_secs:.2f}s"
          " before processing the next step...",
          flush=True,
      )
      time.sleep(sleep_secs)

  # 5. Teardown and cleanup
  logging.info("5. Tearing down iterator and cleaning up fleet...")
  iterator.close()
  if not is_mock:
    sandbox_utils.teardown_global_fleet()
  else:
    fleet.teardown()

  stats["pools_unwarmed"] = len(getattr(iterator, "unwarm_calls", [])) or len(
      getattr(fleet, "unwarm_calls", [])
  )
  stats["avg_acquire_time_sec"] = (
      sum(acquire_durations) / len(acquire_durations)
      if acquire_durations
      else 0.0
  )
  stats["avg_exec_time_sec"] = (
      sum(exec_durations) / len(exec_durations) if exec_durations else 0.0
  )
  logging.info(
      "   [OK] Timing: avg_acquire_time=%.3fs, avg_exec_time=%.3fs",
      stats["avg_acquire_time_sec"],
      stats["avg_exec_time_sec"],
  )
  print(
      "\n=== Performance Timers ==="
      f"\n  Average Sandbox Acquire Time: {stats['avg_acquire_time_sec']:.3f}s"
      f"\n  Average Command Exec Time:    {stats['avg_exec_time_sec']:.3f}s\n",
      flush=True,
  )
  logging.info(
      "   [OK] Teardown complete. Active pools: %s",
      getattr(fleet, "active_pools", {}),
  )
  logging.info("=== [DeepSWE E2E Test] Pipeline Run Complete: %s ===", stats)
  return stats


def str_to_bool(v: Any) -> bool:
  """Converts string representations of boolean values to bool."""
  if isinstance(v, bool):
    return v
  if str(v).lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif str(v).lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def main(argv: list[str]) -> None:
  parser = argparse.ArgumentParser(description="DeepSWE Sandbox K8s E2E Test.")
  parser.add_argument(
      "--dataset_name", type=str, default="R2E-Gym/R2E-Gym-Subset"
  )
  parser.add_argument("--dataset_split", type=str, default="train")
  parser.add_argument("--dataset_path", type=str, default="")
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--num_generations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=3)
  parser.add_argument("--namespace", type=str, default="rl-tunix-swebench")
  parser.add_argument(
      "--scaffold",
      type=str,
      default="r2egym",
      choices=["r2egym", "sweagent", "openhands"],
      help="Scaffold harness to test ('r2egym', 'sweagent', 'openhands').",
  )
  parser.add_argument(
      "--dry_run", action="store_true", help="Run with mock fleet."
  )
  parser.add_argument(
      "--run_as_job", action="store_true", help="Run as live K8s Job."
  )
  parser.add_argument(
      "--synthetic_dataset",
      action="store_true",
      help="Use synthetic samples.",
  )
  parser.add_argument(
      "--node_selector_key",
      type=str,
      default=os.environ.get(
          "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
      ),
      help="Node selector key for sandbox placement.",
  )
  parser.add_argument(
      "--node_selector_val",
      type=str,
      default=os.environ.get("NODE_SELECTOR_VAL", "sandbox-cpu-pool"),
      help="Node selector value (nodepool name) for sandbox placement.",
  )
  parser.add_argument(
      "--minimum_step_time_in_second",
      type=float,
      default=0.0,
      help=(
          "Minimum duration in seconds for each step. If a step finishes"
          " faster than this, sleeps the difference before processing the"
          " next step."
      ),
  )
  parser.add_argument(
      "--sampling_rate",
      type=float,
      default=1.0,
      help=(
          "Sampling rate in [0.0, 1.0] for randomly choosing batches to test"
          " sandbox acquire and execution (default: 1.0). Once a batch is"
          " selected, all generations in the group are tested."
      ),
  )
  parser.add_argument(
      "--shuffle",
      type=str_to_bool,
      nargs="?",
      const=True,
      default=True,
      help="Whether to shuffle the dataset (default: True).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for shuffling (default: 42).",
  )
  args, _ = parser.parse_known_args(argv[1:])

  if args.seed is not None:
    random.seed(args.seed)

  os.environ["OPENHANDS_SUPPRESS_BANNER"] = "1"

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] [DeepSWEE2E] %(message)s",
      force=True,
  )

  dataset = None
  if not args.synthetic_dataset and not args.dry_run and deepswe is not None:
    try:
      logging.info(
          "Loading dataset %s (split=%s, shuffle=%s, seed=%d)...",
          args.dataset_name,
          args.dataset_split,
          args.shuffle,
          args.seed,
      )
      dataset = deepswe.load_deepswe_dataset(
          dataset_name=args.dataset_name,
          dataset_split=args.dataset_split,
          dataset_path=args.dataset_path or None,
          shuffle=args.shuffle,
          seed=args.seed,
      )
      logging.info("Loaded %d dataset samples.", len(dataset))
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning(
          "Dataset load note (%s), falling back to synthetic dataset.", e
      )
      dataset = None

  if dataset is None:
    logging.info("Creating synthetic dataset samples for E2E test...")
    dataset = create_synthetic_dataset(
        num_samples=max(6, args.max_steps * args.batch_size * 2)
    )
    if args.shuffle:
      random.Random(args.seed).shuffle(dataset)

  if args.dry_run:
    fleet = FakeFleet()
    is_mock = True
  else:
    node_sel = None
    if args.node_selector_key and args.node_selector_val:
      node_sel = {args.node_selector_key: args.node_selector_val}
    logging.info(
        "Initializing global fleet in namespace '%s' targeting node"
        " selector: %s (scaffold=%s)",
        args.namespace,
        node_sel,
        args.scaffold,
    )
    fleet = sandbox_utils.init_global_fleet(
        tasks=dataset,
        max_concurrency=128,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        namespace=args.namespace,
        scaffold=args.scaffold,
        node_selector=node_sel,
    )
    is_mock = False

  stats = run_pipeline_e2e(
      dataset=dataset,
      fleet=fleet,
      batch_size=args.batch_size,
      num_generations=args.num_generations,
      max_steps=args.max_steps,
      is_mock=is_mock,
      scaffold=args.scaffold,
      minimum_step_time_in_second=args.minimum_step_time_in_second,
      sampling_rate=args.sampling_rate,
  )

  expected_sandboxes = args.max_steps * args.batch_size * args.num_generations
  assert (
      stats["steps_executed"] == args.max_steps
  ), f"Executed {stats['steps_executed']} steps, expected {args.max_steps}"
  if args.sampling_rate >= 1.0:
    assert stats["sandboxes_acquired"] == expected_sandboxes, (
        f"Acquired {stats['sandboxes_acquired']} sandboxes, expected"
        f" {expected_sandboxes}"
    )
  else:
    assert stats["sandboxes_acquired"] <= expected_sandboxes, (
        f"Acquired {stats['sandboxes_acquired']} sandboxes, expected <="
        f" {expected_sandboxes}"
    )
  assert stats["sandboxes_acquired"] == stats["sandboxes_released"], (
      f"Acquired {stats['sandboxes_acquired']} but released"
      f" {stats['sandboxes_released']}!"
  )
  assert stats["pools_unwarmed"] > 0, "No warm pools were unwarmed!"
  print(
      "\nPerformance Summary:"
      f"\n  Average Sandbox Acquire Time: {stats['avg_acquire_time_sec']:.3f}s"
      f"\n  Average Command Exec Time:    {stats['avg_exec_time_sec']:.3f}s"
  )
  print("\nALL CHECKS PASSED: DeepSWE Agent Sandbox E2E Verified Successfully!")


class DeepSWESandboxE2ETest(absltest.TestCase):
  """Unit test case enabling test execution."""

  def test_e2e_pipeline_lifecycle_mock(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=12)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        scaffold="r2egym",
    )
    expected_sandboxes = max_steps * batch_size * num_generations
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["sandboxes_acquired"], expected_sandboxes)
    self.assertEqual(stats["sandboxes_released"], expected_sandboxes)
    self.assertGreater(stats["pools_unwarmed"], 0)
    self.assertIn("avg_acquire_time_sec", stats)
    self.assertIn("avg_exec_time_sec", stats)
    self.assertEqual(fleet.active_pools, {})

  def test_e2e_pipeline_lifecycle_mock_openhands(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=12)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        scaffold="openhands",
    )
    expected_sandboxes = max_steps * batch_size * num_generations
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["sandboxes_acquired"], expected_sandboxes)
    self.assertEqual(stats["sandboxes_released"], expected_sandboxes)
    self.assertGreater(stats["pools_unwarmed"], 0)
    self.assertIn("avg_acquire_time_sec", stats)
    self.assertIn("avg_exec_time_sec", stats)
    self.assertEqual(fleet.active_pools, {})

  def test_e2e_pipeline_minimum_step_time(self):
    batch_size = 1
    num_generations = 1
    max_steps = 2
    dataset = create_synthetic_dataset(num_samples=6)
    fleet = FakeFleet()
    t_start = time.perf_counter()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        minimum_step_time_in_second=0.1,
    )
    elapsed = time.perf_counter() - t_start
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertGreaterEqual(elapsed, 0.1)

  def test_e2e_pipeline_sampling_rate_zero(self):
    batch_size = 1
    num_generations = 2
    max_steps = 3
    dataset = create_synthetic_dataset(num_samples=6)
    fleet = FakeFleet()
    stats = run_pipeline_e2e(
        dataset=dataset,
        fleet=fleet,
        batch_size=batch_size,
        num_generations=num_generations,
        max_steps=max_steps,
        is_mock=True,
        sampling_rate=0.0,
    )
    self.assertEqual(stats["steps_executed"], max_steps)
    self.assertEqual(stats["batches_sampled"], 0)
    self.assertEqual(stats["sandboxes_acquired"], 0)
    self.assertEqual(stats["sandboxes_released"], 0)
    self.assertGreater(stats["pools_unwarmed"], 0)


if __name__ == "__main__":
  if (
      "--dry_run" in sys.argv
      or "--batch_size" in sys.argv
      or "--run_as_job" in sys.argv
  ):
    main(sys.argv)
  else:
    absltest.main()
