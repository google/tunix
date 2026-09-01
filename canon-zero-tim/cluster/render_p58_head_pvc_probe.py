#!/usr/bin/env python3
"""Render one bounded P58 head-pool PVC mount probe."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import yaml


_DNS_LABEL = re.compile(r"^[a-z0-9](?:[-a-z0-9]*[a-z0-9])?$")
_DIGEST_IMAGE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
_QUEUE = "multislice-queue"
_HEAD_NODEPOOL = "canon-cpu-pool"
_MODEL_PVC = "haoyugao-cpu-np-pvc"
_MOUNT_PATH = "/mnt/disks/linchai_data"
_REQUIRED_PATH = f"{_MOUNT_PATH}/models/Qwen3-4B-Instruct-2507"


def _label(value: str, *, field: str) -> str:
  if not value or len(value) > 63 or _DNS_LABEL.fullmatch(value) is None:
    raise ValueError(f"{field} must be an exact Kubernetes DNS label")
  return value


def render(
    *,
    run_id: str,
    client_image: str,
    namespace: str = "default",
    queue_name: str = _QUEUE,
    head_nodepool: str = _HEAD_NODEPOOL,
    model_pvc: str = _MODEL_PVC,
    required_path: str = _REQUIRED_PATH,
) -> dict:
  """Returns a read-only mount probe; this never launches TPU work."""
  run_id = _label(run_id, field="run_id")
  namespace = _label(namespace, field="namespace")
  queue_name = _label(queue_name, field="queue_name")
  head_nodepool = _label(head_nodepool, field="head_nodepool")
  model_pvc = _label(model_pvc, field="model_pvc")
  if queue_name != _QUEUE:
    raise ValueError(f"P58 PVC probe requires queue {_QUEUE!r}")
  if head_nodepool != _HEAD_NODEPOOL:
    raise ValueError(f"P58 PVC probe requires head pool {_HEAD_NODEPOOL!r}")
  if model_pvc != _MODEL_PVC:
    raise ValueError(f"P58 PVC probe requires model PVC {_MODEL_PVC!r}")
  if _DIGEST_IMAGE.fullmatch(client_image) is None:
    raise ValueError("P58 PVC probe client image must be digest-pinned")
  required = Path(required_path)
  if not required.is_absolute() or required_path != str(required):
    raise ValueError("P58 PVC probe required path must be normalized and absolute")
  if required_path != _REQUIRED_PATH:
    raise ValueError(f"P58 PVC probe requires model path {_REQUIRED_PATH!r}")

  name = f"canon-p58-head-pvc-probe-{run_id}"
  _label(name, field="metadata.name")
  return {
      "apiVersion": "v1",
      "kind": "Pod",
      "metadata": {
          "name": name,
          "namespace": namespace,
          "labels": {
              "app.kubernetes.io/name": "p58-head-pvc-probe",
              "app.kubernetes.io/managed-by": "canon-zero-tim",
              "canon.zero-tim/run-id": run_id,
              "canon.zero-tim/probe": "head-pvc-mount",
              "kueue.x-k8s.io/queue-name": queue_name,
          },
      },
      "spec": {
          "restartPolicy": "Never",
          "activeDeadlineSeconds": 600,
          "terminationGracePeriodSeconds": 10,
          "automountServiceAccountToken": False,
          "priorityClassName": "very-high",
          "hostNetwork": True,
          "dnsPolicy": "ClusterFirstWithHostNet",
          "nodeSelector": {
              "cloud.google.com/gke-nodepool": head_nodepool,
          },
          "containers": [{
              "name": "head-pvc-probe",
              "image": client_image,
              "imagePullPolicy": "IfNotPresent",
              "command": ["python3", "-c"],
              "args": [
                  "import os,sys; path=sys.argv[1]; "
                  "assert os.path.isdir(path), path; "
                  "assert os.access(path, os.R_OK), path; "
                  "print('P58_HEAD_PVC_MOUNT_PASS path=' + path, flush=True)",
                  required_path,
              ],
              "resources": {
                  "requests": {"cpu": "1", "memory": "1Gi"},
                  "limits": {"cpu": "2", "memory": "4Gi"},
              },
              "volumeMounts": [{
                  "name": "model-pvc",
                  "mountPath": _MOUNT_PATH,
                  "readOnly": True,
              }],
          }],
          "volumes": [{
              "name": "model-pvc",
              "persistentVolumeClaim": {
                  "claimName": model_pvc,
                  "readOnly": True,
              },
          }],
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--client-image", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--namespace", default="default")
  parser.add_argument("--queue-name", default=_QUEUE)
  parser.add_argument("--head-nodepool", default=_HEAD_NODEPOOL)
  parser.add_argument("--model-pvc", default=_MODEL_PVC)
  parser.add_argument("--required-path", default=_REQUIRED_PATH)
  args = parser.parse_args()
  output = Path(args.output)
  if not output.is_absolute():
    raise ValueError("--output must be an absolute path")
  document = render(
      run_id=args.run_id,
      client_image=args.client_image,
      namespace=args.namespace,
      queue_name=args.queue_name,
      head_nodepool=args.head_nodepool,
      model_pvc=args.model_pvc,
      required_path=args.required_path,
  )
  with output.open("x", encoding="utf-8") as stream:
    yaml.safe_dump(document, stream, sort_keys=False)
  print(
      "P58_HEAD_PVC_PROBE_RENDER_PASS "
      f"pod={document['metadata']['name']} output={output}",
      flush=True,
  )


if __name__ == "__main__":
  main()
