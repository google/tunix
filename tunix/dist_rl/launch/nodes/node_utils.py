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

"""Helpers shared by the distributed RL trainer/rollout/inference node mains.

Every helper reads from the validated `resolve_config.DistHyperParameters` (or
takes explicit typed values), so the nodes keep a single YAML -> runtime
conversion without importing the heavy agentic `BasePipeline`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import logging
import pathlib
import pickle
import signal
import sys
from typing import Literal, TypeAlias, cast

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import transformers
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.worker import remote_execution

NodeRole = Literal["trainer", "rollout", "inference"]

# Axis names of the 2D device mesh the tunix trainer, the vanilla/in-process
# vLLM rollout and the reference inference node build from `mesh_axes`.
MESH_AXIS_NAMES = ("fsdp", "tp")


# The Hugging Face tokenizer the nodes load: the type the rollout samplers
# (`VanillaSampler`, `InprocessVllmSampler`) take.
HfTokenizer: TypeAlias = transformers.PreTrainedTokenizerBase


def ensure_repo_root_on_path() -> None:
  """Ensures `resolve_config.REPO_ROOT` is on `sys.path` for dynamic imports."""
  repo_root = str(resolve_config.REPO_ROOT)
  if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
  logging.info("Repo root inserted into sys.path: %s", repo_root)


# TODO(noghabi): Reuse the agentic tokenizer loading
# (`tunix.cli.utils.model.create_tokenizer`, unwrapping `.tokenizer`) so
# `tokenizer_type`, `add_bos`/`add_eos`, and `HF_TOKEN` are handled in one place.
# Kept as in the previous `run_*_node.py` mains for now.
def load_hf_tokenizer(hp: resolve_config.DistHyperParameters) -> HfTokenizer:
  """Loads the Hugging Face tokenizer from `tokenizer_config.tokenizer_path`.

  `BasePipeline._get_tokenizer()` is not used: it returns the Tunix `Tokenizer`
  wrapper, while the samplers, `TokenizerAdapter`, chat parsers and the
  pad/eos id lookups take the raw Hugging Face tokenizer.

  Args:
    hp: Validated distributed RL configuration.

  Returns:
    The tokenizer, with `pad_token` defaulted to `eos_token` when unset.

  Raises:
    ValueError: If no tokenizer could be loaded from the path.
  """
  tokenizer_path = str(hp.config_mapping("tokenizer_config")["tokenizer_path"])
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  loaded = transformers.AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if loaded is None:
    raise ValueError(f"No tokenizer could be loaded from {tokenizer_path!r}.")
  tokenizer = cast(HfTokenizer, loaded)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer


def pad_and_eos_ids(tokenizer: HfTokenizer) -> tuple[int, int]:
  """Returns `(pad_id, eos_id)`: pad falls back to 0 and eos to pad."""
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = (
      tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
  )
  return int(pad_id), int(eos_id)


def trajectory_store_config(
    hp: resolve_config.DistHyperParameters,
) -> dict[str, str] | None:
  """Builds the TrajectoryStore config, or None when the store is disabled."""
  store = hp.config_mapping("trajectory_store_config")
  if not store["enable"]:
    return None
  config = {"backend": str(store["backend"])}
  if store["dir"]:
    config["root_dir"] = str(store["dir"])
  run_id = str(hp.config_mapping("cluster_config")["run_id"])
  if run_id:
    config["run_id"] = run_id
  return config


def require_discovery_context(
    context: context_lib.ProcessContext | None,
) -> context_lib.ProcessContext:
  """Returns `context`, or raises if it cannot register with discovery.

  Args:
    context: Process context handed to the node `main` by the runtime.

  Returns:
    The same context, narrowed to non-None.

  Raises:
    RuntimeError: If there is no context or it has no discovery API.
  """
  if context and context.ipc and context.ipc.discovery:
    return context
  raise RuntimeError(
      "Require discovery API, but process context doesn't support."
  )


def worker_id(
    hp: resolve_config.DistHyperParameters,
    role: NodeRole,
    local_worker_id: str,
) -> str:
  """Returns the worker id this node registers with.

  On `cluster_config.target: k8s` the id is `cluster_config.k8s.<role>_id`
  (always resolved by `DistHyperParameters`); on `target: local` it is the
  fixed `local_worker_id`, so local runs keep stable ids across users.

  Args:
    hp: Validated distributed RL configuration.
    role: Node role, which selects the `cluster_config.k8s.<role>_id` key.
    local_worker_id: Worker id used on `target: local`.

  Returns:
    The worker id.
  """
  cluster = hp.config_mapping("cluster_config")
  if cluster["target"] == "k8s":
    return str(cluster["k8s"][f"{role}_id"])
  return local_worker_id


def service_port(hp: resolve_config.DistHyperParameters, role: NodeRole) -> int:
  """Returns `cluster_config.ports.<role>`, the node's gRPC port."""
  return int(hp.config_mapping("cluster_config")["ports"][role])


def optional_bool(mapping: Mapping[str, object], key: str) -> bool | None:
  """Returns `mapping[key]` as bool, or None when the key is absent or null.

  For optional keys of open-ended config sections (e.g.
  `rollout_model_config.kwargs`), where an unset key keeps the consumer's own
  default.

  Args:
    mapping: Open-ended config section.
    key: Optional key to read.
  """
  if key in mapping and mapping[key] is not None:
    return bool(mapping[key])
  return None


async def serve_worker(
    worker_service: object,
    *,
    process_context: context_lib.ProcessContext,
    role: NodeRole,
    worker_id_str: str,
    port: int,
    extra_metadata: Mapping[str, object] | None = None,
    drain_fn: Callable[[], Awaitable[None] | None] | None = None,
) -> None:
  """Starts the gRPC server, registers with discovery, and drains on signal.

  Args:
    worker_service: Worker RPC service exposed by `GrpcRemoteExecutionServer`.
    process_context: Validated discovery-capable process context.
    role: Node role ("trainer", "rollout", "inference").
    worker_id_str: Registered worker id.
    port: gRPC port to listen on.
    extra_metadata: Optional extra fields merged into the discovery metadata.
    drain_fn: Optional sync or async drain callback run before `stop_serving`.
  """
  server = remote_execution.GrpcRemoteExecutionServer(worker_service)
  await server.start_serving_async(port)
  logging.info("Serving %s worker on port %d.", role, port)
  # Everything after `start_serving_async` is inside the `try`, so the server is
  # always drained and stopped, even if discovery registration fails.
  try:
    metadata: dict[str, object] = {
        "service_type": role,
        "service_port": port,
        "worker_id": worker_id_str,
    }
    if extra_metadata:
      metadata.update(extra_metadata)
    assert process_context.ipc is not None
    assert process_context.ipc.discovery is not None
    process_context.ipc.discovery.register(metadata=pickle.dumps(metadata))
    logging.info("%s worker is registered.", role.capitalize())

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.add_signal_handler(sig, stop_event.set)
      except NotImplementedError:
        pass

    await stop_event.wait()
  except asyncio.CancelledError:
    pass
  finally:
    if drain_fn is not None:
      logging.info("Draining %s worker...", role)
      try:
        result = drain_fn()
        if result is not None:
          await result
        logging.info("%s worker drained.", role.capitalize())
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception("Failed to drain %s worker cleanly.", role)
    await server.stop_serving()


# TODO(noghabi): Delegate to `tunix.utils.mesh.create_mesh` with
# `hp.parse_mesh_config(model_key)`, as the agentic `BasePipeline` does, keeping
# the exact device-count check below (`mesh_lib.create_mesh(devices=None)` only
# rejects meshes larger than the device count). Kept as in the previous
# `run_*_node.py` mains for now.
def create_mesh(
    hp: resolve_config.DistHyperParameters, model_key: str, label: str
) -> Mesh:
  """Builds the `(fsdp, tp)` device mesh for `model_key` over all devices.

  Args:
    hp: Validated distributed RL configuration.
    model_key: Model section whose `mesh` gives the fsdp/tp sizes, e.g.
      `actor_model_config`.
    label: Human-readable node name used in the error message.

  Returns:
    The JAX mesh with axis names `MESH_AXIS_NAMES`.

  Raises:
    ValueError: If `dp` or `expert` is not 1, or if `fsdp * tp` does not equal
      the visible device count.
  """
  axes = hp.mesh_axes(model_key)
  if axes["dp"] != 1 or axes["expert"] != 1:
    raise ValueError(
        f"{label} mesh only supports 2D ('fsdp', 'tp') sharding (dp=1,"
        f" expert=1), got {model_key}.mesh axes={axes}."
    )
  shape = (axes["fsdp"], axes["tp"])
  if shape[0] * shape[1] != jax.device_count():
    raise ValueError(
        f"{label} mesh dimensions must multiply to visible JAX device count. "
        f"Got {model_key}.mesh shape={shape}, devices={jax.device_count()}."
    )
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=MESH_AXIS_NAMES)


# TODO(noghabi): Drop `ensure_local_safetensors` and its helpers and let
# `AutoModel.from_pretrained` download via `model_id`, `model_source`, and
# `model_download_path` (as `tunix.cli.utils.model.create_model` does). Kept as
# in the previous `run_*_node.py` mains for now.
def has_direct_safetensors(model_path: pathlib.Path) -> bool:
  """Returns whether `model_path` directly contains `*.safetensors` shards."""
  return any(model_path.glob("*.safetensors"))


def _download_hf_model(model_id: str, model_dir: str) -> None:
  from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

  oss_utils.hf_pipeline(model_id, model_dir)


def ensure_local_safetensors(
    model_dir: str, model_id: str, *, model_key: str
) -> str:
  """Returns a local directory with direct safetensors, downloading if needed.

  Args:
    model_dir: `<model_key>.model_download_path`.
    model_id: `<model_key>.model_id`, downloaded from Hugging Face when
      `model_dir` has no direct `*.safetensors`.
    model_key: Config section name, used in error messages.

  Returns:
    The expanded `model_dir`.

  Raises:
    ValueError: If `model_dir` is empty, is a file, or still holds no direct
      safetensors after the download.
  """
  if not model_dir:
    raise ValueError(
        f"{model_key}.model_download_path is required for JAX model weights."
    )
  model_path = pathlib.Path(model_dir).expanduser()
  if model_path.exists() and not model_path.is_dir():
    raise ValueError(
        f"{model_key}.model_download_path must point to a local directory."
        f" Got: {model_dir}"
    )
  if has_direct_safetensors(model_path):
    return str(model_path)

  logging.info(
      "No direct safetensors found in %s. Downloading %s.", model_path, model_id
  )
  model_path.mkdir(parents=True, exist_ok=True)
  _download_hf_model(model_id, str(model_path))
  if has_direct_safetensors(model_path):
    return str(model_path)
  raise ValueError(
      "Download completed, but no '*.safetensors' files were found directly"
      f" in {model_key}.model_download_path: {model_path}"
  )
