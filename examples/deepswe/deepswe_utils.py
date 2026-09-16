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

"""Utility functions for DeepSWE."""

import argparse
import os
import sys
from typing import Any, Optional


def str2bool(v: str | bool) -> bool:
  """Parses common string representations into boolean values."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  elif v.lower() in ("no", "false", "f", "n", "0"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_runtime_environment(workdir: Optional[str] = None) -> None:
  """Sets up sys.path and initializes Pathways before JAX initialization."""
  workdir = workdir or os.getcwd()
  for root in [
      workdir,
      os.path.join(workdir, "pathways-utils"),
      os.path.join(workdir, "r2egym"),
      "/app",
  ]:
    if os.path.exists(root) and root not in sys.path:
      sys.path.insert(0, root)

  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")

  if "proxy" in os.getenv("JAX_PLATFORMS", ""):
    try:
      import pathwaysutils  # pytype: disable=import-error

      pathwaysutils.initialize()
      print("Pathways initialized successfully before JAX import.", flush=True)
    except ImportError:
      pass


def configure_orbax_ocdbt_handler(model_path: str, logger: Any = None) -> None:
  """Configures the standard Orbax ArrayHandler if the checkpoint is OCDBT.

  pathwaysutils registers CloudPathwaysArrayHandler on init, which reads
  checkpoint shards on the Pathways workers. It does not support OCDBT yet
  (b/365549911), so an OCDBT checkpoint has to fall back to the standard
  ArrayHandler.
  """

  def _log(msg: str, *args: Any, level: str = "info") -> None:
    if logger is not None:
      getattr(logger, level)(msg, *args)
    else:
      print(msg % args if args else msg, flush=True)

  try:
    from etils import epath  # pytype: disable=import-error
    import jax
    from orbax.checkpoint._src.serialization import jax_array_handlers  # pytype: disable=import-error
    from orbax.checkpoint._src.serialization import type_handler_registry  # pytype: disable=import-error

    if (epath.Path(model_path) / "manifest.ocdbt").exists():
      type_handler_registry.register_type_handler(
          jax.Array, jax_array_handlers.ArrayHandler(), override=True
      )
      _log(
          "Checkpoint is OCDBT; registered standard ArrayHandler: %s",
          model_path,
      )
    else:
      _log(
          "Checkpoint is not OCDBT; keeping the registered handler so reads"
          " stay on the Pathways workers: %s",
          model_path,
      )
  except Exception as e:
    _log("Could not inspect checkpoint storage format: %s", e, level="warning")


def setup_kubernetes_config(
    node_selector_val: str = "cpu-np",
    node_selector_key: str = "cloud.google.com/gke-nodepool",
    kubeconfig: str = "~/.kube/config",
    logger: Any = None,
) -> Any:
  """Sets Kubernetes environment variables and loads in-cluster or kubeconfig."""
  from kubernetes import client, config as k8s_config  # pytype: disable=import-error

  os.environ.setdefault("KUBECONFIG", kubeconfig)
  os.environ.setdefault("NODE_SELECTOR_KEY", node_selector_key)
  os.environ["NODE_SELECTOR_VAL"] = node_selector_val

  try:
    if os.getenv("KUBERNETES_SERVICE_HOST"):
      k8s_config.load_incluster_config()
    else:
      k8s_config.load_kube_config()
    k8s_client = client.CoreV1Api()
    if logger is not None:
      logger.info("Kubernetes connection verified.")
    else:
      print("Kubernetes connection verified.", flush=True)
    return k8s_client
  except Exception as e:
    if logger is not None:
      logger.warning("Kubernetes config loading failed: %s", e)
    else:
      print(f"Warning: Kubernetes config loading failed: {e}", flush=True)
    return None
