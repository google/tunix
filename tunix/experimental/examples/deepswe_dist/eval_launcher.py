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

"""Supervise separate distributed eval controller and rollout processes."""

import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import eval_deepswe

SCRIPT = str(Path(__file__).with_name("eval_deepswe.py"))


def wait_for_pathways_workers(timeout):
  """Wait for the expected JobSet TPU hosts before initializing the proxy."""
  job = os.environ.get("EVAL_JOBSET_NAME")
  if not job:
    return
  from kubernetes import client, config

  config.load_incluster_config()
  api = client.CoreV1Api()
  count = int(os.environ.get("EVAL_TPU_HOSTS", "16"))
  selector = (
      f"jobset.sigs.k8s.io/jobset-name={job},"
      "jobset.sigs.k8s.io/replicatedjob-name=worker"
  )
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    pods = api.list_namespaced_pod(
        os.environ.get("NAMESPACE", "trellis"),
        label_selector=selector,
        field_selector="status.phase=Running",
        _request_timeout=30,
    ).items
    if len(pods) == count:
      logging.info("All %d Pathways TPU hosts are Running", count)
      return
    logging.info("Waiting for Pathways TPU hosts: %d/%d", len(pods), count)
    time.sleep(5)
  raise TimeoutError("Pathways TPU pods did not become Running")


def terminate(process):
  if process is None:
    return
  # Kill the process group even if its parent already exited: inference
  # subprocesses may still be alive.
  try:
    os.killpg(process.pid, signal.SIGTERM)
  except ProcessLookupError:
    return
  try:
    process.wait(timeout=120)
  except subprocess.TimeoutExpired:
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def main(argv=None):
  argv = list(sys.argv[1:] if argv is None else argv)
  args = eval_deepswe.parse_args(argv)
  if args.worker_addresses != ["localhost:20001"] or args.port != 20001:
    raise ValueError(
        "Bundled launcher uses localhost:20001; run roles separately for remote"
        " workers"
    )
  logging.basicConfig(level=logging.INFO)
  worker = controller = None

  def stop(_sig, _frame):
    raise KeyboardInterrupt()

  signal.signal(signal.SIGTERM, stop)
  signal.signal(signal.SIGINT, stop)
  try:
    wait_for_pathways_workers(args.startup_timeout)
    worker = subprocess.Popen(
        [sys.executable, SCRIPT, *argv, "--role", "worker"],
        start_new_session=True,
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    controller = subprocess.Popen(
        [sys.executable, SCRIPT, *argv, "--role", "controller"],
        env=env,
        start_new_session=True,
    )
    while controller.poll() is None:
      if worker.poll() is not None:
        raise RuntimeError(
            f"Rollout worker exited early: code={worker.returncode}"
        )
      time.sleep(1)
    return controller.returncode
  finally:
    terminate(controller)
    terminate(worker)


if __name__ == "__main__":
  sys.exit(main())
