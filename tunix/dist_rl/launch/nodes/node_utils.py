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
import contextlib
import json
import logging
import os
import pathlib
import pickle
import signal
import sys
import types
from typing import Final, Literal, TYPE_CHECKING, TypeAlias, TypedDict, cast

import transformers
from tunix.dist_rl.common import otel_setup
from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.worker import abstract_worker
from tunix.dist_rl.worker import remote_execution
from tunix.oss import utils as oss_utils

if TYPE_CHECKING:
  from jax.sharding import Mesh

NodeRole = Literal["trainer", "rollout", "inference"]
# Every process of a run: the orchestrator and the worker nodes.
ProcessRole = Literal["orchestrator", "trainer", "rollout", "inference"]

# Environment variable that `k8s.sh` or an external launcher
# (`cluster_config.target: external`) sets to each rollout replica's index
# (0-based) when `cluster_config.rollout_replicas > 1`, instead of giving each
# replica its own config.
ROLLOUT_REPLICA_ENV = "DIST_RL_ROLLOUT_REPLICA"
# Environment variable holding the worker's gRPC port on
# `cluster_config.target: external`, where the launcher assigns each process
# its own port (e.g. one a job scheduler allocates), instead of giving each
# worker its own config.
WORKER_PORT_ENV = "DIST_RL_WORKER_PORT"

# Marker that starts the resolved-config log line, so the record can be found in
# mixed logs (e.g. `sed -n 's/.*DIST_RL_RESOLVED_CONFIG //p' trainer.log | jq`).
RESOLVED_CONFIG_LOG_MARKER = "DIST_RL_RESOLVED_CONFIG"
# Discovery registration metadata key holding the worker's config
# fingerprint; the orchestrator rejects a worker whose value differs from its
# own.
CONFIG_SHA256_METADATA_KEY = "config_sha256"
# Environment variables recorded with the resolved config: runtime settings
# only, never credentials. The exact names are the toggles the launchers and
# nodes set; the prefixes are whole namespaces.
_LOGGED_ENV_VARS = frozenset({
    ROLLOUT_REPLICA_ENV,
    WORKER_PORT_ENV,
    "ALLOW_MULTIPLE_LIBTPU_LOAD",
    "DISABLE_JAX_CACHE",
    "DISABLE_MOSAIC_ATTN",
    "ENABLE_PATHWAYS_PERSISTENCE",
    "LOCAL_JAX_CACHE_DIR",
    "NEW_MODEL_DESIGN",
    "ROLLOUT_JAX_CACHE_GCS_DIR",
    "SAVE_JAX_CACHE",
    "SKIP_JAX_PRECOMPILE",
    "TFDS_DATA_DIR",
    "TRAINER_PATHWAYS_LOCAL_INIT",
    "USE_RAIDEN",
    "VERIFY_WEIGHTS",
})
_LOGGED_ENV_PREFIXES = (
    "JAX_",
    "LIBTPU_",
    "RAIDEN_",
    "TPU_",
    "TUNIX_",
    "VLLM_",
    "XLA_",
)
# Credentials can live under those prefixes (e.g. vLLM's `VLLM_API_KEY`); the
# suffixes are matched in any case.
_CREDENTIAL_ENV_SUFFIXES = (
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_KEY",
    "_PASSWORD",
    "_SECRET",
    "_TOKEN",
)

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


# Environment variables `apply_library_env` sets from the config, and the
# config key each comes from. Launchers reject them in user-supplied
# environments so a value can only come from the config.
LIBRARY_ENV_CONFIG_KEYS: Final[Mapping[str, str]] = types.MappingProxyType({
    "SKIP_JAX_PRECOMPILE": "vllm_config.skip_jax_precompile",
    "TPU_RAIDEN_DATA_NICS": "weight_sync_config.raiden_data_nics",
    "VERIFY_WEIGHTS": "weight_sync_config.verify_weights",
})


def _library_env(
    hp: resolve_config.DistHyperParameters, role: Literal["trainer", "rollout"]
) -> dict[str, tuple[str | None, str]]:
  """Returns `{variable: (value, config key)}`; a None value leaves it unset."""
  ws = hp.config_mapping("weight_sync_config")
  values: dict[str, str | None] = {
      "VERIFY_WEIGHTS": "true" if ws["verify_weights"] else "false",
      "TPU_RAIDEN_DATA_NICS": str(ws["raiden_data_nics"]) or None,
  }
  if role == "rollout":
    skip = hp.config_mapping("vllm_config")["skip_jax_precompile"]
    values["SKIP_JAX_PRECOMPILE"] = "1" if skip else "0"
  return {
      name: (value, LIBRARY_ENV_CONFIG_KEYS[name])
      for name, value in values.items()
  }


def apply_library_env(
    hp: resolve_config.DistHyperParameters, role: Literal["trainer", "rollout"]
) -> None:
  """Sets the environment variables that libraries read, from the config.

  MaxText, tpu_inference and Raiden take these settings only from the
  environment, so the node sets them itself and the launchers pass config only.
  Call it right after resolving the config: before `log_resolved_config`, which
  records them, and before JAX or an engine starts.

  * `VERIFY_WEIGHTS` (trainer, rollout): `weight_sync_config.verify_weights`
    as "true" or "false": whether weight syncs log checksums and Raiden
    metrics.
  * `TPU_RAIDEN_DATA_NICS` (trainer, rollout):
    `weight_sync_config.raiden_data_nics`; unset when empty, so Raiden picks
    its NICs itself.
  * `SKIP_JAX_PRECOMPILE` (rollout): `vllm_config.skip_jax_precompile` as "1"
    or "0".

  Args:
    hp: Validated distributed RL configuration.
    role: Worker node role, which selects the variables.

  Raises:
    ValueError: If the environment already holds one of the variables with
      another value than the config gives, or holds one the config leaves
      unset.
  """
  env = _library_env(hp, role)
  for name, (value, key) in env.items():
    if name in os.environ and os.environ[name] != value:
      wanted = "leaves it unset" if value is None else f"sets it to {value!r}"
      raise ValueError(
          f"{name}={os.environ[name]!r} is set in the environment, but {key}"
          f" {wanted}: the node sets {name} from the config, so unset it and"
          f" set {key} instead."
      )
  for name, (value, _) in env.items():
    if value is not None:
      os.environ[name] = value


class ResolvedConfigRecord(TypedDict):
  """The JSON record `log_resolved_config` logs after the marker.

  Attributes:
    role: Process that resolved the config.
    config_sha256: `resolve_config.config_fingerprint` of the config.
    env: The runtime environment variables `_logged_env` selects.
    config: `resolve_config.plain_config` of the config.
  """

  role: ProcessRole
  config_sha256: str
  env: dict[str, str]
  config: dict[str, object]


def is_credential_env_var(name: str) -> bool:
  """Returns whether environment variable `name` may hold a credential."""
  return name.upper().endswith(_CREDENTIAL_ENV_SUFFIXES)


def _logged_env() -> dict[str, str]:
  """Returns the runtime environment variables to record, without credentials."""
  return {
      name: value
      for name, value in sorted(os.environ.items())
      if (name in _LOGGED_ENV_VARS or name.startswith(_LOGGED_ENV_PREFIXES))
      and not is_credential_env_var(name)
  }


def log_resolved_config(
    hp: resolve_config.DistHyperParameters, role: ProcessRole
) -> str:
  """Logs the resolved config as one JSON record and returns its fingerprint.

  The record is a single line, `RESOLVED_CONFIG_LOG_MARKER {json}`, holding a
  `ResolvedConfigRecord`: the process `role`, the `config_sha256` fingerprint
  (compare it across the processes of a run), the runtime environment
  variables, and the full resolved `config`.

  Args:
    hp: Validated distributed RL configuration.
    role: Process that resolved `hp`.

  Returns:
    `resolve_config.config_fingerprint(hp)`.
  """
  fingerprint = resolve_config.config_fingerprint(hp)
  record: ResolvedConfigRecord = {
      "role": role,
      "config_sha256": fingerprint,
      "env": _logged_env(),
      "config": resolve_config.plain_config(hp),
  }
  logging.info(
      "%s %s", RESOLVED_CONFIG_LOG_MARKER, json.dumps(record, sort_keys=True)
  )
  return fingerprint


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
  (always resolved by `DistHyperParameters`; `k8s.sh` pins it to the
  launcher's value). On `target: local` and `target: external` it is the fixed
  `local_worker_id`, which keeps the ids stable across users.

  With several rollout replicas (`cluster_config.rollout_replicas > 1`, which
  `target: local` rejects), each rollout replica takes its index from
  `ROLLOUT_REPLICA_ENV`, so the replicas share one config: on k8s it appends
  `-<index>` to its id, matching its JobSet name; on external its id is
  `rollout-<index>`.

  Args:
    hp: Validated distributed RL configuration.
    role: Node role, which selects the `cluster_config.k8s.<role>_id` key.
    local_worker_id: Worker id used on `target: local` and `target: external`.

  Returns:
    The worker id.

  Raises:
    ValueError: If `ROLLOUT_REPLICA_ENV` is unset (when `rollout_replicas > 1`),
      not a non-negative integer, or out of `[0, rollout_replicas)` on a rollout
      worker.
  """
  cluster = hp.config_mapping("cluster_config")
  target = cluster["target"]
  role_id = (
      str(cluster["k8s"][f"{role}_id"]) if target == "k8s" else local_worker_id
  )
  if role != "rollout":
    return role_id
  replicas = int(cluster["rollout_replicas"])
  if replicas == 1 and ROLLOUT_REPLICA_ENV not in os.environ:
    return role_id
  if ROLLOUT_REPLICA_ENV not in os.environ:
    raise ValueError(
        f"{ROLLOUT_REPLICA_ENV} must be set on each rollout replica when"
        f" cluster_config.rollout_replicas={replicas} (k8s.sh or the external"
        " launcher sets it)."
    )
  raw_replica = os.environ[ROLLOUT_REPLICA_ENV]
  if not raw_replica.isdecimal():
    raise ValueError(
        f"{ROLLOUT_REPLICA_ENV} must be a non-negative integer, got"
        f" {raw_replica!r}."
    )
  replica = int(raw_replica)
  if not 0 <= replica < replicas:
    raise ValueError(
        f"{ROLLOUT_REPLICA_ENV}={replica} is out of range for"
        f" cluster_config.rollout_replicas={replicas}."
    )
  if replicas == 1:
    return role_id
  # `rollout-<i>` matches `rollout.LOCAL_WORKER_ID = "rollout-0"`.
  return f"{role_id}-{replica}" if target == "k8s" else f"{role}-{replica}"


def service_port(hp: resolve_config.DistHyperParameters, role: NodeRole) -> int:
  """Returns the node's gRPC port.

  On `cluster_config.target: external` the launcher assigns each worker its
  port in `WORKER_PORT_ENV`, since every process gets the same config. On the
  other targets the port is `cluster_config.ports.<role>`.

  Args:
    hp: Validated distributed RL configuration.
    role: Node role, which selects the `cluster_config.ports.<role>` key.

  Returns:
    The port.

  Raises:
    ValueError: If `WORKER_PORT_ENV` is unset or not a port number on
      `target: external`, or set on another target.
  """
  cluster = hp.config_mapping("cluster_config")
  target = cluster["target"]
  if target != "external":
    if WORKER_PORT_ENV in os.environ:
      raise ValueError(
          f"{WORKER_PORT_ENV} is only for cluster_config.target=external; on"
          f" target {target!r} the node listens on cluster_config.ports.{role}."
          f" Unset {WORKER_PORT_ENV}."
      )
    return int(cluster["ports"][role])
  if WORKER_PORT_ENV not in os.environ:
    raise ValueError(
        f"{WORKER_PORT_ENV} must be set on cluster_config.target=external,"
        " where the launcher assigns each worker its gRPC port."
    )
  port = os.environ[WORKER_PORT_ENV]
  if not port.isdecimal() or not 0 < int(port) <= 65535:
    raise ValueError(
        f"{WORKER_PORT_ENV} must be a port number in [1, 65535], got {port!r}."
    )
  return int(port)


def setup_otel_metrics(
    hp: resolve_config.DistHyperParameters,
    role: ProcessRole,
    worker_id_str: str | None,
) -> bool:
  """Exports this process's metrics to OpenTelemetry when it is configured.

  `otel_setup.setup_metrics` installs an exporter when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set, and fails on a malformed OpenTelemetry
  setting. Every process names its service `tunix-<role>` and tags its metrics
  with its role, the run id, and its worker id.

  Args:
    hp: Validated distributed RL configuration.
    role: Process role.
    worker_id_str: The id the worker registers with; None for the orchestrator,
      which has none.

  Returns:
    Whether an exporter was installed.
  """
  attributes = {
      "tunix.role": role,
      "tunix.run_id": str(hp.config_mapping("cluster_config")["run_id"]),
  }
  if worker_id_str is not None:
    attributes["tunix.worker_id"] = worker_id_str
  return otel_setup.setup_metrics(
      default_service_name=f"tunix-{role}", resource_attributes=attributes
  )


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


# Grace period for in-flight RPCs when a worker stops serving; matches
# `remote_execution.serve_worker`.
_STOP_SERVING_GRACE_S: Final[float] = 5.0


async def serve_worker(
    worker_service: abstract_worker.Worker,
    *,
    process_context: context_lib.ProcessContext,
    role: NodeRole,
    worker_id_str: str,
    port: int,
    config_sha256: str,
    extra_metadata: Mapping[str, object] | None = None,
    drain_fn: Callable[[], Awaitable[None] | None] | None = None,
    close_fn: Callable[[], None] | None = None,
) -> None:
  """Starts the gRPC server, registers with discovery, and drains on stop.

  Wakes on SIGINT/SIGTERM, cancellation, or when `worker_service` invokes its
  `set_on_stop` callback (from an incoming `stop(abort=...)` RPC), then shuts
  down in this order: `drain_fn`, `server.stop_serving()`, then `close_fn`.

  Args:
    worker_service: Worker RPC service exposed by `GrpcRemoteExecutionServer`.
    process_context: Validated discovery-capable process context.
    role: Node role ("trainer", "rollout", "inference").
    worker_id_str: Registered worker id.
    port: gRPC port to listen on.
    config_sha256: This worker's config fingerprint (`log_resolved_config`
      returns it), sent with the registration so that the orchestrator can
      reject a worker that resolved a different config.
    extra_metadata: Optional extra fields merged into the discovery metadata.
    drain_fn: Optional sync or async callback run before `stop_serving`, e.g. to
      cancel long-running work so in-flight RPCs return.
    close_fn: Optional sync callback run after `stop_serving` has cancelled the
      queued RPCs and waited for the running one, e.g. to release resources
      those RPCs use. Runs even if `stop_serving` fails. Its own failure
      propagates, unless the worker was stopped with an abort: it is then
      logged and the process exits 1.

  Raises:
    SystemExit: With code 1 when stopped with abort=True (or when worker stop
      escalates a failure to an abort).
  """
  stop_event = asyncio.Event()
  loop = asyncio.get_running_loop()
  aborted: list[bool] = []

  def _on_stop(abort: bool = False) -> None:
    aborted.append(bool(abort))
    loop.call_soon_threadsafe(stop_event.set)

  worker_service.set_on_stop(_on_stop)
  try:
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(port)
  except BaseException:
    worker_service.set_on_stop(None)
    raise
  logging.info("Serving %s worker on port %d.", role, port)
  # Everything after `start_serving_async` is inside the `try`, so the server is
  # always drained, stopped, and closed, even if discovery registration fails.
  try:
    metadata: dict[str, object] = {
        "service_type": role,
        "service_port": port,
        "worker_id": worker_id_str,
        CONFIG_SHA256_METADATA_KEY: config_sha256,
    }
    if extra_metadata:
      metadata.update(extra_metadata)
    assert process_context.ipc is not None
    assert process_context.ipc.discovery is not None
    process_context.ipc.discovery.register(metadata=pickle.dumps(metadata))
    logging.info("%s worker is registered.", role.capitalize())

    installed_signals: list[signal.Signals] = []
    try:
      for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
          loop.add_signal_handler(sig, stop_event.set)
          installed_signals.append(sig)

      with contextlib.suppress(asyncio.CancelledError):
        await stop_event.wait()
    finally:
      # Restore the default handlers, so a second SIGINT/SIGTERM can still
      # interrupt a teardown that hangs (e.g. a blocked `close_fn`).
      for sig in installed_signals:
        loop.remove_signal_handler(sig)
  finally:
    await asyncio.sleep(0)
    # TODO(noghabi): Let `drain_fn` failures propagate too, once `stop_serving`
    # and `close_fn` have run (check the rollout's drain first).
    if drain_fn is not None:
      logging.info("Draining %s worker...", role)
      try:
        result = drain_fn()
        if result is not None:
          await result
        logging.info("%s worker drained.", role.capitalize())
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception("Failed to drain %s worker cleanly.", role)
    try:
      # The grace lets the `stop` RPC that woke this loop return its response
      # to the orchestrator before the server goes away.
      await server.stop_serving(grace=_STOP_SERVING_GRACE_S)
    finally:
      try:
        if close_fn is not None:
          logging.info("Closing %s worker...", role)
          close_fn()
          logging.info("%s worker closed.", role.capitalize())
      except Exception:  # pylint: disable=broad-exception-caught
        # After an abort (e.g. the worker's own `stop()` failed and now
        # re-raises that failure), exit 1 below rather than with this error.
        if not any(aborted):
          raise
        logging.exception("Failed to close %s worker cleanly.", role)
      finally:
        worker_service.set_on_stop(None)
  if any(aborted):
    raise SystemExit(1)


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
  import jax  # pylint: disable=g-import-not-at-top
  from jax import sharding as jax_sharding  # pylint: disable=g-import-not-at-top
  from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top

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
  return jax_sharding.Mesh(devices, axis_names=MESH_AXIS_NAMES)


# TODO(noghabi): Drop `ensure_local_safetensors` and its helpers and let
# `AutoModel.from_pretrained` download via `model_id`, `model_source`, and
# `model_download_path` (as `tunix.cli.utils.model.create_model` does). Kept as
# in the previous `run_*_node.py` mains for now.
def has_direct_safetensors(model_path: pathlib.Path) -> bool:
  """Returns whether `model_path` directly contains `*.safetensors` shards."""
  return any(model_path.glob("*.safetensors"))


def _download_hf_model(model_id: str, model_dir: str) -> None:
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
