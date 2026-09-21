# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""File/GCS-based weight synchronization implementation (`GCSWeightSync` & `GCSWeightSyncHandler`).

Implements the unified `WeightSynchronizer` worker-side interface and
`WeightSyncHandler` coordinator-side interface using Orbax sharded checkpoints.
"""

from __future__ import annotations

from concurrent import futures
import dataclasses
import logging
import os
import shutil
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from etils import epath
import jax
from jax.experimental import multihost_utils
import orbax.checkpoint as ocp
from tunix.experimental.weight_sync import raiden_synchronizer
from tunix.experimental.weight_sync import weight_sync

logger = logging.getLogger(__name__)


def _is_verify_weights_enabled() -> bool:
  return weight_sync.is_verify_weights_enabled()


def _resolve_staging_dir(
    explicit_dir: Optional[str] = None,
    sync_request: Any = None,
) -> str:
  """Resolves the directory used for temporary weight-sync checkpoints."""
  if explicit_dir:
    return explicit_dir
  if sync_request is not None:
    extra = getattr(sync_request, "extra_config", None)
    if isinstance(extra, Mapping):
      for key in ("staging_dir", "weight_sync_gcs_dir"):
        if extra.get(key):
          return str(extra[key])
  for env_var in (
      "WEIGHT_SYNC_GCS_DIR",
      "WEIGHT_SYNC_STAGING_DIR",
  ):
    val = os.environ.get(env_var, "").strip()
    if val:
      return val
  for fallback_root_env in (
      "MAXTEXT_OUTPUT_DIR",
      "CHECKPOINT_ROOT_DIRECTORY",
      "ARTIFACT_ROOT",
  ):
    root = os.environ.get(fallback_root_env, "").strip()
    if root:
      return os.path.join(root, "weight_sync_staging")
  raise ValueError(
      "GCSWeightSync requires a staging directory. Set WEIGHT_SYNC_GCS_DIR "
      "or pass staging_dir to GCSWeightSync / create_weight_synchronizer."
  )


def _remove_path(path_str: str) -> None:
  """Safely removes a local or GCS directory tree."""
  if not path_str:
    return
  try:
    p = epath.Path(path_str)
    if p.exists():
      p.rmtree()
  except Exception as exc:  # pylint: disable=broad-exception-caught
    try:
      if os.path.exists(path_str):
        shutil.rmtree(path_str, ignore_errors=True)
    except Exception:  # pylint: disable=broad-exception-caught
      logger.warning(
          "Failed to clean up staging checkpoint %s: %s", path_str, exc
      )


def _cleanup_stale_checkpoints(base_dir: str) -> None:
  """Purges leftover step_* or tmp checkpoints in base_dir from prior runs."""
  if not base_dir:
    return
  try:
    base_path = epath.Path(base_dir)
    if not base_path.exists():
      return
    for child in base_path.iterdir():
      name = child.name
      if name.startswith("step_") or ".orbax-checkpoint-tmp" in name:
        logger.info(
            "GCSWeightSync startup cleanup removing leftover checkpoint: %s",
            child,
        )
        _remove_path(str(child))
  except Exception as exc:  # pylint: disable=broad-exception-caught
    logger.warning(
        "GCSWeightSync startup cleanup on %s encountered non-fatal error: %s",
        base_dir,
        exc,
    )


class GCSWeightSync(
    raiden_synchronizer.RaidenSynchronizer, weight_sync.WeightSynchronizer
):
  """Worker-side file/GCS weight synchronizer implementing `WeightSynchronizer`.

  Shares PyTree flattening, canonical parameter naming (`_param_key`),
  `TensorMetadata` generation, `apply_to_runner`, and `checksums` with
  `RaidenWeightSync`, while executing D2H/H2D via sharded Orbax checkpoints.
  """

  def __init__(
      self,
      job_name: str,
      state: Any = None,
      *,
      worker_index: int = 0,
      staging_dir: Optional[str] = None,
      max_to_keep: Optional[int] = None,
      use_ocdbt: bool = True,
      use_zarr3: bool = False,
      use_compression: Optional[bool] = None,
      ocdbt_target_data_file_size: Optional[int] = None,
      chunk_byte_size: Optional[int] = None,
      cleanup_on_start: Optional[bool] = None,
      **kwargs: Any,
  ):
    self.staging_dir = staging_dir
    if max_to_keep is None:
      max_to_keep = int(os.environ.get("WEIGHT_SYNC_GCS_MAX_TO_KEEP", "1"))
    self.max_to_keep = max_to_keep
    self._use_ocdbt = use_ocdbt
    self._use_zarr3 = use_zarr3
    if use_compression is None:
      use_compression = os.environ.get(
          "WEIGHT_SYNC_GCS_USE_COMPRESSION", "1"
      ).strip().lower() in ("1", "true", "yes", "y", "t")
    self._use_compression = use_compression
    if ocdbt_target_data_file_size is None:
      ocdbt_target_data_file_size = int(
          os.environ.get(
              "WEIGHT_SYNC_GCS_OCDBT_TARGET_BYTES", str(512 * 1024 * 1024)
          )
      )
    self._ocdbt_target_data_file_size = ocdbt_target_data_file_size
    if chunk_byte_size is None:
      chunk_byte_size = int(
          os.environ.get("WEIGHT_SYNC_GCS_CHUNK_BYTES", str(64 * 1024 * 1024))
      )
    self._chunk_byte_size = chunk_byte_size
    if cleanup_on_start is None:
      cleanup_on_start = os.environ.get(
          "WEIGHT_SYNC_GCS_CLEANUP_ON_START", "1"
      ).strip().lower() in ("1", "true", "yes", "y", "t")
    self._cleanup_on_start = cleanup_on_start
    self._cleaned_base_dirs: set[str] = set()
    self._lock = threading.RLock()
    self._checkpointer: Optional[ocp.Checkpointer] = None
    self._cleanup_executor: Optional[futures.ThreadPoolExecutor] = None
    self._cleanup_futures: dict[str, futures.Future[Any]] = {}
    self._artifact_uri: Optional[str] = None
    self._saved_uris: list[str] = []
    self._step_counter = 0
    self._gcs_metrics: dict[str, Any] = {}
    super().__init__(
        job_name=job_name,
        state=state,
        worker_index=worker_index,
        **kwargs,
    )

  def _get_checkpointer(self) -> ocp.Checkpointer:
    if self._checkpointer is None:
      handler = ocp.PyTreeCheckpointHandler(
          save_concurrent_gb=96,
          restore_concurrent_gb=96,
          use_ocdbt=self._use_ocdbt,
          use_zarr3=self._use_zarr3,
          use_compression=self._use_compression,
      )
      self._checkpointer = ocp.Checkpointer(handler)
    return self._checkpointer

  def _wait_pending_cleanup(self, uri: Optional[str] = None) -> None:
    """Waits for background cleanup tasks (either a specific URI or all)."""
    if uri is not None:
      fut = self._cleanup_futures.pop(uri, None)
      if fut is not None:
        fut.result()
      return
    pending = list(self._cleanup_futures.values())
    self._cleanup_futures.clear()
    for fut in pending:
      fut.result()

  @property
  def active(self) -> bool:
    return self.bound

  def bind(self, state: Any) -> None:
    """Binds or rebinds a model state PyTree for GCS weight synchronization."""
    with self._lock:
      self.names = []
      self.arrays = []
      self.names, self.arrays = raiden_synchronizer._filter_bindable(  # pylint: disable=protected-access
          *raiden_synchronizer.flatten_weights(state),
          allow_proxy=self._is_proxy,
      )
      del state
      array_mesh = None
      for arr in self.arrays:
        array_mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
        if array_mesh is not None:
          break
      self._host_subgrid = raiden_synchronizer._compute_host_subgrid(array_mesh)  # pylint: disable=protected-access
      logger.info(
          "%s GCSWeightSync bound %d arrays.", self.job_name, len(self.arrays)
      )

  def d2h(self, sync_request: Any = None) -> None:
    """Saves bound weights as a sharded Orbax checkpoint in staging_dir."""
    with self._lock:
      if not self.bound:
        raise RuntimeError(f"{self.job_name}: bind() must run before d2h()")

      t0 = time.monotonic()
      base_dir = _resolve_staging_dir(self.staging_dir, sync_request)
      policy_version = getattr(sync_request, "policy_version", None)
      extra = getattr(sync_request, "extra_config", None) or {}
      req_id = (
          (extra.get("req_id") if isinstance(extra, Mapping) else None)
          or getattr(sync_request, "req_id", None)
      )
      if policy_version is None:
        policy_version = self._step_counter
      if not req_id:
        req_id = f"r{self._step_counter}"
      self._step_counter += 1

      # Sanitize req_id for filesystem/GCS path safety
      safe_req_id = "".join(
          c if (c.isalnum() or c in ("-", "_")) else "_" for c in str(req_id)
      )
      ckpt_dir = os.path.join(base_dir, f"step_{policy_version}_{safe_req_id}")

      # Ensure no background delete is still running on ckpt_dir
      self._wait_pending_cleanup(ckpt_dir)

      if jax.process_index() == 0:
        if self._cleanup_on_start and base_dir not in self._cleaned_base_dirs:
          self._wait_pending_cleanup()
          _cleanup_stale_checkpoints(base_dir)
          self._cleaned_base_dirs.add(base_dir)
        if not ckpt_dir.startswith("gs://"):
          _remove_path(ckpt_dir)
          epath.Path(base_dir).mkdir(parents=True, exist_ok=True)
        elif ckpt_dir in self._saved_uris:
          _remove_path(ckpt_dir)

      # Multi-host barrier so process_0's directory cleanup never races with
      # non-zero processes writing their OCDBT shards.
      if jax.process_count() > 1:
        multihost_utils.sync_global_devices(
            f"gcs_weight_sync_pre_save_{policy_version}_{safe_req_id}"
        )

      flat_weights: dict[str, Any] = {}
      for name, arr in zip(self.names, self.arrays):
        canon_key = raiden_synchronizer._param_key(name)  # pylint: disable=protected-access
        if not canon_key:
          raise ValueError(
              f"{self.job_name}: parameter name {name!r} canonicalized to empty key."
          )
        flat_weights[canon_key] = arr

      save_args = {
          k: ocp.type_handlers.SaveArgs(chunk_byte_size=self._chunk_byte_size)
          for k in flat_weights
      }
      checkpointer = self._get_checkpointer()
      checkpointer.save(
          ckpt_dir,
          force=True,
          args=ocp.args.PyTreeSave(
              item=flat_weights,
              save_args=save_args,
              ocdbt_target_data_file_size=(
                  self._ocdbt_target_data_file_size if self._use_ocdbt else None
              ),
          ),
      )
      if hasattr(checkpointer, "wait_until_finished"):
        checkpointer.wait_until_finished()

      elapsed = time.monotonic() - t0
      self._artifact_uri = ckpt_dir
      if ckpt_dir not in self._saved_uris:
        self._saved_uris.append(ckpt_dir)
      self._gcs_metrics = {
          "d2h_elapsed_s": elapsed,
          "artifact_uri": ckpt_dir,
          "num_tensors": len(flat_weights),
      }
      logger.info(
          "%s GCSWeightSync.d2h saved %d tensors to %s in %.2fs",
          self.job_name,
          len(flat_weights),
          ckpt_dir,
          elapsed,
      )

  def h2d(
      self,
      sync_request: Any = None,
      *,
      checkpoint_path: Optional[str] = None,
      **kwargs: Any,
  ) -> None:
    """Restores sharded weights from Orbax checkpoint directly into TPU HBM."""
    del kwargs
    with self._lock:
      if not self.bound:
        raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")

      resolved_path = checkpoint_path
      if not resolved_path and sync_request is not None:
        extra = getattr(sync_request, "extra_config", None)
        if isinstance(extra, Mapping):
          resolved_path = extra.get("checkpoint_path") or extra.get(
              "artifact_uri"
          )
        if not resolved_path:
          src_metas = getattr(sync_request, "source_metadata", None) or ()
          for m in src_metas:
            uri = (
                m.get("artifact_uri")
                if isinstance(m, Mapping)
                else getattr(m, "artifact_uri", None)
            )
            if uri:
              resolved_path = str(uri)
              break
      if not resolved_path:
        resolved_path = self._artifact_uri
      if not resolved_path:
        raise ValueError(
            f"{self.job_name}: GCSWeightSync.h2d requires checkpoint_path or "
            "sync_request with extra_config['checkpoint_path']."
        )

      self._wait_pending_cleanup(resolved_path)

      t0 = time.monotonic()
      target_specs: dict[str, jax.ShapeDtypeStruct] = {}
      restore_args: dict[str, ocp.type_handlers.ArrayRestoreArgs] = {}
      key_order: list[str] = []
      for name, arr in zip(self.names, self.arrays):
        canon_key = raiden_synchronizer._param_key(name)  # pylint: disable=protected-access
        key_order.append(canon_key)
        sharding = getattr(arr, "sharding", None)
        target_specs[canon_key] = jax.ShapeDtypeStruct(
            shape=tuple(arr.shape),
            dtype=arr.dtype,
            sharding=sharding,
        )
        restore_args[canon_key] = ocp.type_handlers.ArrayRestoreArgs(
            restore_type=jax.Array,
            dtype=arr.dtype,
            sharding=sharding,
            global_shape=tuple(arr.shape),
        )

      checkpointer = self._get_checkpointer()
      restored = checkpointer.restore(
          resolved_path,
          args=ocp.args.PyTreeRestore(
              item=target_specs,
              restore_args=restore_args,
          ),
      )

      if not isinstance(restored, Mapping):
        raise RuntimeError(
            f"{self.job_name}: Expected mapping from Orbax restore at "
            f"{resolved_path}, got {type(restored).__name__}"
        )

      missing = [k for k in key_order if k not in restored]
      if missing:
        raise RuntimeError(
            f"{self.job_name}: Checkpoint at {resolved_path} is missing "
            f"{len(missing)} of {len(key_order)} expected tensors, e.g. "
            f"{missing[:10]}"
        )

      old_arrays = list(self.arrays)
      self.arrays = [restored[k] for k in key_order]
      jax.block_until_ready(self.arrays)
      self._stale_arrays = old_arrays
      elapsed = time.monotonic() - t0
      self._artifact_uri = resolved_path
      self._gcs_metrics = {
          "h2d_elapsed_s": elapsed,
          "checkpoint_path": resolved_path,
          "num_tensors": len(self.arrays),
      }
      logger.info(
          "%s GCSWeightSync.h2d restored %d sharded tensors from %s in %.2fs",
          self.job_name,
          len(self.arrays),
          resolved_path,
          elapsed,
      )

  def apply_to_runner(self, runner: Any) -> None:
    """Applies restored GCS arrays to runner and frees replaced HBM buffers."""
    old_leaves = list(getattr(runner, "state_leaves", ()) or ())
    super().apply_to_runner(runner)
    new_ids = {id(a) for a in self.arrays}
    for old_arr in old_leaves + getattr(self, "_stale_arrays", []):
      raw = getattr(old_arr, "value", old_arr)
      if id(raw) not in new_ids and hasattr(raw, "delete"):
        try:
          if not getattr(raw, "is_deleted", lambda: False)():
            raw.delete()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    self._stale_arrays = []

  def work_unit_metadata(self) -> weight_sync.WorkUnitMetadata:
    with self._lock:
      mesh = None
      for arr in self.arrays:
        mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
        if mesh is not None:
          break
      if mesh is None:
        mesh_axes, mesh_shape = ("fsdp",), (1,)
      else:
        mesh_axes = tuple(mesh.axis_names)
        mesh_shape = tuple(int(mesh.shape[a]) for a in mesh.axis_names)

      variables = tuple(
          raiden_synchronizer._tensor_metadata(  # pylint: disable=protected-access
              raiden_synchronizer._param_key(name), arr, idx  # pylint: disable=protected-access
          )
          for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
      )
      unit = weight_sync.WorkUnitId(
          job_name=self.job_name,
          job_replica_id=str(self.worker_index) if self.worker_index else "",
      )
      checksums = (
          self.checksums(sample=None) if _is_verify_weights_enabled() else None
      )
      return weight_sync.WorkUnitMetadata(
          unit=unit,
          shards=(),
          control_plane_rpc_address="",
          mesh_shape=mesh_shape,
          variables=variables,
          mesh_axes=mesh_axes or None,
          transport_mode="gcs",
          use_ffi=False,
          host_subgrid=self._host_subgrid,
          artifact_uri=self._artifact_uri,
          checksums=checksums,
      )

  def metrics(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._gcs_metrics)

  def release(self, sync_request: Any = None) -> None:
    """Prunes temporary staging checkpoints according to max_to_keep."""
    del sync_request
    with self._lock:
      if jax.process_index() != 0:
        return
      keep = max(0, self.max_to_keep)
      while len(self._saved_uris) > keep:
        old_uri = self._saved_uris.pop(0)
        logger.info(
            "%s GCSWeightSync.release removing temporary staging checkpoint %s",
            self.job_name,
            old_uri,
        )
        if old_uri.startswith("gs://"):
          if self._cleanup_executor is None:
            self._cleanup_executor = futures.ThreadPoolExecutor(max_workers=1)
          self._wait_pending_cleanup(old_uri)
          self._cleanup_futures[old_uri] = self._cleanup_executor.submit(
              _remove_path, old_uri
          )
        else:
          _remove_path(old_uri)

  def close(self) -> None:
    """Closes persistent checkpointer and waits for background cleanup."""
    with self._lock:
      self._wait_pending_cleanup()
      if self._cleanup_executor is not None:
        self._cleanup_executor.shutdown(wait=True)
        self._cleanup_executor = None
      if self._checkpointer is not None:
        try:
          self._checkpointer.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        self._checkpointer = None
      if hasattr(super(), "close"):
        super().close()


GcsWeightSynchronizer = GCSWeightSync


class GCSWeightSyncHandler(weight_sync.WeightSyncHandler):
  """Coordinator-side `WeightSyncHandler` for file/GCS Orbax checkpoints."""

  def __init__(self) -> None:
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}
    self._latest_artifact_uri: Optional[str] = None
    self._latest_src_checksums: Optional[dict[str, float]] = None

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    self._units[metadata.unit] = metadata
    if metadata.artifact_uri:
      self._latest_artifact_uri = metadata.artifact_uri
    if metadata.checksums:
      self._latest_src_checksums = dict(metadata.checksums)

  def build_extra_config(
      self, src_metadata: Sequence[weight_sync.WorkUnitMetadata]
  ) -> dict[str, Any]:
    extra: dict[str, Any] = {"weight_sync_mode": "gcs"}
    for m in src_metadata:
      if m.artifact_uri:
        extra["checkpoint_path"] = m.artifact_uri
        break
    else:
      if self._latest_artifact_uri:
        extra["checkpoint_path"] = self._latest_artifact_uri

    for m in src_metadata:
      if m.checksums:
        extra["source_checksums"] = dict(m.checksums)
        break
    else:
      if self._latest_src_checksums:
        extra["source_checksums"] = dict(self._latest_src_checksums)
    return extra

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: Optional[str] = None,
      generation: Optional[int] = None,
  ) -> weight_sync.TransferResult:
    del generation
    rid = req_id or ""
    if not src_units:
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message="No source work units provided for GCS weight sync.",
      )
    if not dst_units:
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message="No destination work units provided for GCS weight sync.",
      )

    missing_src = [u for u in src_units if u not in self._units]
    if missing_src:
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message=f"Unregistered source unit(s): {missing_src}",
      )
    missing_dst = [u for u in dst_units if u not in self._units]
    if missing_dst:
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message=f"Unregistered destination unit(s): {missing_dst}",
      )

    artifact_uri = None
    for u in src_units:
      meta = self._units[u]
      if meta.artifact_uri:
        artifact_uri = meta.artifact_uri
        break
    if not artifact_uri:
      artifact_uri = self._latest_artifact_uri
    if not artifact_uri:
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message="Source work unit metadata did not report an artifact_uri.",
      )

    path_obj = epath.Path(artifact_uri)
    if not path_obj.exists():
      return weight_sync.TransferResult(
          req_id=rid,
          success=False,
          message=f"Staged checkpoint path does not exist: {artifact_uri}",
      )

    return weight_sync.TransferResult(
        req_id=rid,
        success=True,
        message=f"Verified staged checkpoint at {artifact_uri}",
    )

  def close(self) -> None:
    self._units.clear()


GcsSyncHandler = GCSWeightSyncHandler


# ---------------------------------------------------------------------------
# Subprocess-safe vLLM TPUWorker RPC callables (invoked via collective_rpc)
# ---------------------------------------------------------------------------


def tpu_worker_bind_gcs_sync(
    worker: Any,
    worker_index: int = 0,
    job_name: str = "rollout",
    staging_dir: Optional[str] = None,
) -> dict[str, Any]:
  """Binds live TPUWorker weights to GCSWeightSync inside the worker process."""
  state = worker.get_weights_state()
  sync = getattr(worker, "_gcs_rl_weight_sync", None)
  if sync is None:
    sync = GCSWeightSync(
        job_name=job_name,
        worker_index=worker_index,
        staging_dir=staging_dir,
    )
    worker._gcs_rl_weight_sync = sync  # pylint: disable=protected-access
  else:
    sync.job_name = job_name
    sync.worker_index = worker_index
    if staging_dir:
      sync.staging_dir = staging_dir
  sync.bind(state)
  return dataclasses.asdict(sync.work_unit_metadata())


def tpu_worker_get_gcs_metadata(worker: Any) -> dict[str, Any]:
  """Returns wire-safe WorkUnitMetadata dict for a bound TPUWorker."""
  sync = getattr(worker, "_gcs_rl_weight_sync", None)
  if sync is None or not sync.bound:
    raise RuntimeError(
        "TPUWorker._gcs_rl_weight_sync is not bound; call "
        "tpu_worker_bind_gcs_sync first."
    )
  return dataclasses.asdict(sync.work_unit_metadata())


def tpu_worker_gcs_h2d(
    worker: Any,
    checkpoint_path: str,
    source_checksums: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
  """Restores sharded Orbax weights into TPUWorker's model_runner and verifies parity."""
  sync: Optional[GCSWeightSync] = getattr(worker, "_gcs_rl_weight_sync", None)
  if sync is None or not sync.bound:
    state = worker.get_weights_state()
    sync = GCSWeightSync(job_name="rollout", state=state)
    worker._gcs_rl_weight_sync = sync  # pylint: disable=protected-access

  sync.h2d(checkpoint_path=checkpoint_path)
  sync.apply_to_runner(worker.model_runner)
  if hasattr(worker, "refresh_model_state_leaves"):
    worker.refresh_model_state_leaves()

  if _is_verify_weights_enabled():
    # Re-bind to worker.get_weights_state() so checksums inspect the live
    # runner state buffers actually served by vLLM after apply_to_runner().
    sync.bind(worker.get_weights_state())
    dst_checksums = sync.checksums(sample=None)
    if source_checksums:
      weight_sync.verify_weight_checksums(source_checksums, dst_checksums)
    return dst_checksums
  return {}


def tpu_worker_gcs_metrics(worker: Any) -> dict[str, Any]:
  """Returns GCS weight sync metrics from the TPUWorker."""
  sync = getattr(worker, "_gcs_rl_weight_sync", None)
  return sync.metrics() if sync is not None else {}


def patch_tpu_worker_gcs_sync() -> None:
  """Attaches GCS weight sync RPC methods onto tpu_inference.worker.tpu_worker.TPUWorker."""
  if os.environ.get("JAX_PLATFORMS") == "cpu":
    return
  tw_mod = raiden_synchronizer._lazy_import_module(  # pylint: disable=protected-access
      "tpu_inference.worker.tpu_worker"
  )
  if tw_mod is None or not hasattr(tw_mod, "TPUWorker"):
    return
  tpu_worker_cls = tw_mod.TPUWorker
  tpu_worker_cls.bind_gcs_sync = tpu_worker_bind_gcs_sync
  tpu_worker_cls.get_gcs_metadata = tpu_worker_get_gcs_metadata
  tpu_worker_cls.gcs_h2d = tpu_worker_gcs_h2d
  tpu_worker_cls.gcs_metrics = tpu_worker_gcs_metrics

