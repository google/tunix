#!/usr/bin/env python3
"""Render one production-shaped, Kueue-admitted P58 CPU sandbox probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml


_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_QUEUE = "multislice-queue"
_SANDBOX_NODEPOOL = "deepswe-cpu-pool-2"


def _label(value: str, *, field: str) -> str:
  if not value or len(value) > 63 or _DNS_LABEL.fullmatch(value) is None:
    raise ValueError(f"{field} must be an exact Kubernetes DNS label")
  return value


def render(
    *,
    run_id: str,
    task_image: str,
    namespace: str = "default",
    queue_name: str = _QUEUE,
    sandbox_nodepool: str = _SANDBOX_NODEPOOL,
    image_pull_secret: str = "dockerhub-pro",
) -> dict:
  """Return a single-Pod admission probe; this does not execute R2E."""
  run_id = _label(run_id, field="run_id")
  namespace = _label(namespace, field="namespace")
  queue_name = _label(queue_name, field="queue_name")
  sandbox_nodepool = _label(
      sandbox_nodepool, field="sandbox_nodepool"
  )
  image_pull_secret = _label(
      image_pull_secret, field="image_pull_secret"
  )
  if queue_name != _QUEUE:
    raise ValueError(f"P58 sandbox probe requires queue {_QUEUE!r}")
  if sandbox_nodepool != _SANDBOX_NODEPOOL:
    raise ValueError(
        f"P58 sandbox probe requires node pool {_SANDBOX_NODEPOOL!r}"
    )
  if not task_image or any(char.isspace() for char in task_image):
    raise ValueError("task_image must be a non-empty container image reference")
  name = f"canon-p58-sandbox-probe-{run_id}"
  _label(name, field="metadata.name")
  return {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {
          "name": name,
          "namespace": namespace,
          "labels": {
              "app.kubernetes.io/name": "r2egym",
              "app.kubernetes.io/managed-by": "tunix-deepswe",
              "canon.zero-tim/run-id": run_id,
              "canon.zero-tim/probe": "sandbox-capacity",
              "kueue.x-k8s.io/queue-name": queue_name,
          },
      },
      "spec": {
          "restartPolicy": "Never",
          "activeDeadlineSeconds": 900,
          "terminationGracePeriodSeconds": 10,
          "automountServiceAccountToken": False,
          "nodeSelector": {
              "cloud.google.com/gke-nodepool": sandbox_nodepool,
          },
          "imagePullSecrets": [{"name": image_pull_secret}],
          "containers": [{
              "name": "sandbox-capacity-probe",
              "image": task_image,
              "imagePullPolicy": "IfNotPresent",
              "command": ["/bin/sh", "-c"],
              "args": ["trap 'exit 0' TERM INT; sleep 600 & wait"],
              "resources": {
                  "requests": {"cpu": "2", "memory": "4Gi"},
                  "limits": {"cpu": "4", "memory": "8Gi"},
              },
          }],
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--task-image", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--namespace", default="default")
  parser.add_argument("--queue-name", default=_QUEUE)
  parser.add_argument("--sandbox-nodepool", default=_SANDBOX_NODEPOOL)
  parser.add_argument("--image-pull-secret", default="dockerhub-pro")
  args = parser.parse_args()
  output = Path(args.output)
  if not output.is_absolute():
    raise ValueError("--output must be an absolute path")
  document = render(
      run_id=args.run_id,
      task_image=args.task_image,
      namespace=args.namespace,
      queue_name=args.queue_name,
      sandbox_nodepool=args.sandbox_nodepool,
      image_pull_secret=args.image_pull_secret,
  )
  with output.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(document, stream, sort_keys=False)
  print(
      "P58_SANDBOX_PROBE_RENDER_PASS "
      f"pod={document['metadata']['name']} output={output}",
      flush=True,
  )


if __name__ == "__main__":
  main()
