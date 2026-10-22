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

"""GCS utility for persisting and restoring JAX compilation cache artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent import futures
import dataclasses
import hashlib
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_CACHE_DIR = pathlib.Path("/tmp/jax_cache")
DEFAULT_SYNC_TIMEOUT_S = 180.0
DEFAULT_CLI_TIMEOUT_S = 300.0


def _env_str(env: Mapping[str, str], key: str, default: str = "") -> str:
  """Returns a stripped string value from `env` if present and non-empty."""
  if key in env and env[key]:
    return env[key].strip()
  return default


def _parse_bool_str(value: str, *, default: bool) -> bool:
  """Parses a boolean string representation."""
  normalized = value.strip().lower()
  if not normalized:
    return default
  return normalized in ("1", "true", "yes", "y", "t")


def is_jax_cache_disabled(env: Mapping[str, str] | None = None) -> bool:
  """Returns True when DISABLE_JAX_CACHE is truthy in `env`."""
  source_env = env if env is not None else os.environ
  return _parse_bool_str(
      _env_str(source_env, "DISABLE_JAX_CACHE"), default=False
  )


def classify_tpu_hardware(tpu_slice: str) -> str:
  """Maps a TPU slice or accelerator identifier to its canonical hardware family."""
  raw_hw = tpu_slice.split(":", 1)[0].strip().lower()
  # Check v5e / v5litepod before generic tpuv5 so tpuv5e does not get
  # misclassified as v5p.
  if raw_hw.startswith(("tpuv5e", "tpu-v5-lite", "v5litepod", "v5e")):
    return "v5e"
  if raw_hw.startswith(("tpuv5p", "tpuv5", "tpu-v5p", "v5p")):
    return "v5p"
  if raw_hw.startswith(("tpuv6e", "tpu-v6e", "v6e")):
    return "v6e"
  if raw_hw.startswith(("tpu7x", "tpu-v7x", "v7x")):
    return "v7x"
  sanitized = re.sub(r"[^a-z0-9._-]+", "-", raw_hw).strip("-")
  return sanitized or "tpu"


@dataclasses.dataclass(frozen=True)
class RolloutCacheKeyParams:
  """Compilation-affecting parameters used to derive a rollout GCS cache URI."""

  tpu_slice: str = "tpuv5e:4x4"
  model_name: str = "model"
  sampler: str = "vllm"
  max_prompt_length: int = 1024
  max_response_length: int = 1024
  vllm_max_model_len: int | None = None
  mesh_tp: int = 1
  mesh_fsdp: int = 1
  mesh_expert: int = 1
  vllm_data_parallel_size: int | None = None
  use_lora: bool = False
  lora_rank: int = 64
  enable_prefix_caching: bool = False
  vllm_server_mode: bool = True
  vllm_async_scheduling: bool = False
  rollout_fp8: str = ""
  rollout_quantization: str = ""
  float32_logits: str = ""
  libtpu_init_args: str = ""
  vllm_moe_chunk_size: str = ""
  onehot_moe_permute_threshold: str = ""
  attn_custom_num_reqs_buckets: str = ""
  use_batched_rpa_kernel: str = ""

  @classmethod
  def from_env(
      cls, env: Mapping[str, str] | None = None
  ) -> RolloutCacheKeyParams:
    """Constructs `RolloutCacheKeyParams` from environment variables."""
    source_env = env if env is not None else os.environ
    tpu_slice = (
        _env_str(source_env, "ROLLOUT_TPU_SLICE")
        or _env_str(source_env, "TPU_SLICE")
        or "tpuv5e:4x4"
    )
    model_name = (
        _env_str(source_env, "MODEL_NAME")
        or _env_str(source_env, "MAXTEXT_MODEL_NAME")
        or _env_str(source_env, "MODEL_ID")
        or "model"
    )
    vllm_max_len_raw = _env_str(source_env, "VLLM_MAX_MODEL_LEN")
    vllm_dp_raw = _env_str(source_env, "VLLM_DATA_PARALLEL_SIZE")
    return cls(
        tpu_slice=tpu_slice,
        model_name=model_name,
        sampler=_env_str(source_env, "SAMPLER", "vllm"),
        max_prompt_length=int(
            _env_str(source_env, "MAX_PROMPT_LENGTH", "1024")
        ),
        max_response_length=int(
            _env_str(source_env, "MAX_RESPONSE_LENGTH", "1024")
        ),
        vllm_max_model_len=(
            int(vllm_max_len_raw) if vllm_max_len_raw else None
        ),
        mesh_tp=int(_env_str(source_env, "ROLLOUT_MESH_TP", "1")),
        mesh_fsdp=int(_env_str(source_env, "ROLLOUT_MESH_FSDP", "1")),
        mesh_expert=int(_env_str(source_env, "ROLLOUT_MESH_EXPERT", "1")),
        vllm_data_parallel_size=(int(vllm_dp_raw) if vllm_dp_raw else None),
        use_lora=_parse_bool_str(
            _env_str(source_env, "USE_LORA"), default=False
        ),
        lora_rank=int(_env_str(source_env, "LORA_RANK", "64")),
        enable_prefix_caching=_parse_bool_str(
            _env_str(source_env, "ENABLE_PREFIX_CACHING"), default=False
        ),
        vllm_server_mode=_parse_bool_str(
            _env_str(source_env, "VLLM_SERVER_MODE"), default=True
        ),
        vllm_async_scheduling=_parse_bool_str(
            _env_str(source_env, "VLLM_ASYNC_SCHEDULING"), default=False
        ),
        rollout_fp8=_env_str(source_env, "ROLLOUT_FP8"),
        rollout_quantization=_env_str(source_env, "ROLLOUT_QUANTIZATION"),
        float32_logits=_env_str(source_env, "FLOAT32_LOGITS"),
        libtpu_init_args=_env_str(source_env, "LIBTPU_INIT_ARGS"),
        vllm_moe_chunk_size=_env_str(source_env, "VLLM_MOE_CHUNK_SIZE"),
        onehot_moe_permute_threshold=_env_str(
            source_env, "ONEHOT_MOE_PERMUTE_THRESHOLD"
        ),
        attn_custom_num_reqs_buckets=_env_str(
            source_env, "ATTN_CUSTOM_NUM_REQS_BUCKETS"
        ),
        use_batched_rpa_kernel=(
            _env_str(source_env, "USE_BATCHED_RPA_KERNEL")
            or _env_str(source_env, "ROLLOUT_USE_BATCHED_RPA")
        ),
    )

  def cache_fingerprint(self) -> str:
    """Computes a 10-character SHA-256 hex digest of compilation-affecting parameters."""
    effective_max_len = self.vllm_max_model_len or (
        self.max_prompt_length + self.max_response_length
    )
    effective_dp = (
        self.vllm_data_parallel_size
        if self.vllm_data_parallel_size is not None
        else self.mesh_fsdp
    )
    effective_server_mode = (
        True if self.vllm_server_mode is None else bool(self.vllm_server_mode)
    )
    parts = (
        f"model={self.model_name}",
        f"sampler={self.sampler}",
        f"prompt={self.max_prompt_length}",
        f"resp={self.max_response_length}",
        f"maxlen={effective_max_len}",
        f"slice={self.tpu_slice}",
        f"tp={self.mesh_tp}",
        f"fsdp={self.mesh_fsdp}",
        f"ep={self.mesh_expert}",
        f"dp={effective_dp}",
        f"lora={int(self.use_lora)}",
        f"lora_rank={self.lora_rank if self.use_lora else 0}",
        f"prefix_cache={int(self.enable_prefix_caching)}",
        f"server_mode={effective_server_mode}",
        f"async_sched={int(self.vllm_async_scheduling)}",
        f"fp8={self.rollout_fp8}",
        f"quant={self.rollout_quantization}",
        f"f32logits={self.float32_logits}",
        f"libtpu={self.libtpu_init_args}",
        f"moe_chunk={self.vllm_moe_chunk_size}",
        f"moe_perm={self.onehot_moe_permute_threshold}",
        f"attn_buckets={self.attn_custom_num_reqs_buckets}",
        f"batched_rpa={self.use_batched_rpa_kernel}",
    )
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()[:10]


def _resolve_cache_base_uri(
    bucket: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
  """Resolves the base GCS cache prefix from explicit `bucket` or environment."""
  source_env = env if env is not None else os.environ
  explicit_bucket = bucket.strip() if bucket else ""
  if not explicit_bucket:
    explicit_base_dir = _env_str(source_env, "JAX_CACHE_GCS_DIR")
    if explicit_base_dir:
      cleaned_base = explicit_base_dir
      if cleaned_base.startswith("gs://"):
        cleaned_base = cleaned_base[len("gs://") :]
      cleaned_base = cleaned_base.strip("/")
      return f"gs://{cleaned_base}" if cleaned_base else None

  raw = (
      explicit_bucket
      or _env_str(source_env, "JAX_CACHE_BUCKET")
      or _env_str(source_env, "BUCKET")
  )
  if not raw:
    maxtext_out = _env_str(source_env, "MAXTEXT_OUTPUT_DIR")
    if maxtext_out.startswith("gs://"):
      raw = maxtext_out
  if not raw:
    return None
  if raw.startswith("gs://"):
    raw = raw[len("gs://") :]
  cleaned = raw.strip("/")
  if not cleaned:
    return None
  return f"gs://{cleaned}/jax_cache"


def is_primary_rollout_worker(worker_id: str) -> bool:
  """Returns True when `worker_id` identifies the primary rollout replica.

  In multi-replica deployments (`ROLLOUT_REPLICAS > 1`), replicas are named
  `<prefix>-0`, `<prefix>-1`, ..., `<prefix>-N`; only replica 0 (`-0`) is the
  primary writer. Single-replica workers without a `-{index}` suffix (e.g.
  `linchai-roll` or `rollout_worker_42`) are also treated as primary.
  """
  match = re.search(r"-(\d+)$", worker_id.strip())
  if match is None:
    return True
  return int(match.group(1)) == 0


def derive_rollout_cache_uri(
    bucket: str | None = None,
    params: RolloutCacheKeyParams | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
  """Resolves or derives the rollout GCS compilation cache URI in pure Python."""
  source_env = env if env is not None else os.environ
  if is_jax_cache_disabled(source_env):
    return None

  explicit_rollout_uri = _env_str(
      source_env, "ROLLOUT_JAX_CACHE_GCS_DIR"
  ) or _env_str(source_env, "VLLM_JAX_CACHE_GCS_DIR")
  if explicit_rollout_uri:
    return explicit_rollout_uri.rstrip("/")

  base_uri = _resolve_cache_base_uri(bucket=bucket, env=source_env)
  if not base_uri:
    return None

  resolved_params = (
      params
      if params is not None
      else RolloutCacheKeyParams.from_env(source_env)
  )
  hw_type = classify_tpu_hardware(resolved_params.tpu_slice)
  topo = (
      resolved_params.tpu_slice.split(":", 1)[1].strip()
      if ":" in resolved_params.tpu_slice
      else "default"
  )
  topo = re.sub(r"[^a-zA-Z0-9._-]+", "-", topo).strip("-") or "default"
  model_slug = (
      re.sub(r"[^a-z0-9._-]+", "-", resolved_params.model_name.lower()).strip(
          "-"
      )
      or "model"
  )
  digest = resolved_params.cache_fingerprint()
  return (
      f"{base_uri}/{hw_type}/{model_slug}/"
      f"rollout_{topo}_ep{resolved_params.mesh_expert}_tp{resolved_params.mesh_tp}_{digest}"
  )


def resolve_local_cache_dir(
    local_dir: str | pathlib.Path | None = None,
    env: Mapping[str, str] | None = None,
) -> pathlib.Path:
  """Resolves the local compilation cache directory from argument or environment."""
  if local_dir is not None:
    return pathlib.Path(local_dir)
  source_env = env if env is not None else os.environ
  for key in (
      "LOCAL_JAX_CACHE_DIR",
      "VLLM_LOCAL_JAX_CACHE_DIR",
      "JAX_CACHE_DIR",
      "JAX_COMPILATION_CACHE_DIR",
      "VLLM_XLA_CACHE_PATH",
  ):
    val = _env_str(source_env, key)
    if val:
      return pathlib.Path(val)
  return DEFAULT_LOCAL_CACHE_DIR


def get_active_gcs_uri(env: Mapping[str, str] | None = None) -> str | None:
  """Returns the active rollout GCS cache URI for this process, if configured."""
  source_env = env if env is not None else os.environ
  if is_jax_cache_disabled(source_env):
    return None
  explicit_rollout_uri = _env_str(
      source_env, "ROLLOUT_JAX_CACHE_GCS_DIR"
  ) or _env_str(source_env, "VLLM_JAX_CACHE_GCS_DIR")
  if explicit_rollout_uri:
    return explicit_rollout_uri.rstrip("/")
  if _env_str(source_env, "JAX_CACHE_GCS_DIR"):
    return derive_rollout_cache_uri(env=source_env)
  return None


@dataclasses.dataclass(frozen=True)
class JaxCacheConfig:
  """Typed configuration for JAX/vLLM GCS compilation cache synchronization."""

  save_jax_cache: bool = True
  rollout_jax_cache_gcs_dir: str | None = None
  jax_cache_gcs_dir: str | None = None
  local_dir: pathlib.Path = DEFAULT_LOCAL_CACHE_DIR
  sync_timeout_s: float = DEFAULT_SYNC_TIMEOUT_S

  @property
  def resolved_gcs_uri(self) -> str | None:
    """Returns the effective GCS cache URI for rollout workers, if configured."""
    if self.rollout_jax_cache_gcs_dir:
      return self.rollout_jax_cache_gcs_dir
    if self.jax_cache_gcs_dir:
      env = dict(os.environ)
      env["JAX_CACHE_GCS_DIR"] = self.jax_cache_gcs_dir
      env.pop("ROLLOUT_JAX_CACHE_GCS_DIR", None)
      env.pop("VLLM_JAX_CACHE_GCS_DIR", None)
      return derive_rollout_cache_uri(env=env)
    return None

  @classmethod
  def from_env(
      cls,
      env: Mapping[str, str] | None = None,
      *,
      derive_from_bucket: bool = False,
      params: RolloutCacheKeyParams | None = None,
  ) -> JaxCacheConfig:
    """Constructs `JaxCacheConfig` from environment variables."""
    source_env = env if env is not None else os.environ
    disabled = is_jax_cache_disabled(source_env)
    save_raw = _env_str(source_env, "SAVE_JAX_CACHE") or _env_str(
        source_env, "VLLM_SAVE_JAX_CACHE"
    )
    save_enabled = not disabled and _parse_bool_str(save_raw, default=True)
    rollout_uri = (
        None
        if disabled
        else (
            _env_str(source_env, "ROLLOUT_JAX_CACHE_GCS_DIR")
            or _env_str(source_env, "VLLM_JAX_CACHE_GCS_DIR")
            or None
        )
    )
    base_uri = (
        None
        if disabled
        else (_env_str(source_env, "JAX_CACHE_GCS_DIR") or None)
    )
    if not disabled and rollout_uri is None and derive_from_bucket:
      rollout_uri = derive_rollout_cache_uri(params=params, env=source_env)
    timeout_raw = _env_str(source_env, "JAX_CACHE_SYNC_TIMEOUT_S")
    sync_timeout_s = (
        float(timeout_raw) if timeout_raw else DEFAULT_SYNC_TIMEOUT_S
    )
    return cls(
        save_jax_cache=save_enabled,
        rollout_jax_cache_gcs_dir=rollout_uri,
        jax_cache_gcs_dir=base_uri,
        local_dir=resolve_local_cache_dir(env=source_env),
        sync_timeout_s=sync_timeout_s,
    )


def _parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
  """Parses a gs://bucket/prefix URI into (bucket_name, prefix)."""
  m = re.match(r"^gs://([^/]+)(?:/(.*))?$", gcs_uri)
  if not m:
    raise ValueError(f"Invalid GCS URI: {gcs_uri}")
  bucket_name = m.group(1)
  prefix = m.group(2) or ""
  if prefix and not prefix.endswith("/"):
    prefix += "/"
  return bucket_name, prefix


def _log_local_cache_status(
    local_path: pathlib.Path, context: str = "after download"
) -> int:
  """Logs the count of cache objects present in local_path."""
  local_files = [p for p in local_path.rglob("*") if p.is_file()]
  if not local_files:
    logger.warning(
        "[jax_cache] 0 cache objects detected in %s %s.",
        local_path,
        context,
    )
  else:
    logger.info(
        "[jax_cache] Cache download completed successfully (%d cache objects"
        " detected in %s).",
        len(local_files),
        local_path,
    )
  return len(local_files)


def download_cache(
    local_dir: str | pathlib.Path,
    gcs_uri: str,
    max_workers: int = 8,
    timeout_s: float = DEFAULT_SYNC_TIMEOUT_S,
) -> bool:
  """Downloads cached compilation artifacts from GCS into local_dir."""
  local_path = pathlib.Path(local_dir)
  try:
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
  except ValueError as e:
    logger.error("Skipping download: %s", e)
    return False

  # Try python google-cloud-storage transfer_manager first
  try:
    from google.cloud import storage  # pylint: disable=g-import-not-at-top
    from google.cloud.storage import transfer_manager  # pylint: disable=g-import-not-at-top

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT")
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=prefix, timeout=timeout_s))
    blob_names = [
        b.name[len(prefix) :]
        for b in blobs
        if not b.name.endswith("/") and len(b.name) > len(prefix)
    ]
    if not blob_names:
      logger.warning(
          "[jax_cache] 0 cache objects detected in %s (cold cache start). JAX"
          " will compile on the fly.",
          gcs_uri,
      )
      return True

    logger.info(
        "[jax_cache] Detected %d cache objects in %s.",
        len(blob_names),
        gcs_uri,
    )

    local_path.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[jax_cache] Downloading %d artifacts from %s to %s...",
        len(blob_names),
        gcs_uri,
        local_path,
    )
    results = transfer_manager.download_many_to_path(
        bucket,
        blob_names,
        destination_directory=str(local_path),
        blob_name_prefix=prefix,
        # Never fork: the caller holds live TPU/gRPC state, and forked children
        # of such a process can segfault or deadlock (default is PROCESS).
        worker_type=transfer_manager.THREAD,
        max_workers=max_workers,
        deadline=int(timeout_s),
        download_kwargs={"timeout": timeout_s},
    )
    any_failed = False
    for name, result in zip(blob_names, results):
      if isinstance(result, Exception):
        logger.warning("[jax_cache] Failed to download %s: %s", name, result)
        any_failed = True
    if any_failed:
      raise RuntimeError("Some artifacts failed to download.")
    _log_local_cache_status(local_path)
    return True
  except ImportError:
    pass
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("[jax_cache] transfer_manager download failed: %s", e)

  # Fallback to gsutil or gcloud CLI if available
  if shutil.which("gsutil"):
    if _run_cli_rsync(
        ["gsutil", "-m", "rsync", "-r", gcs_uri, str(local_path)]
    ):
      _log_local_cache_status(local_path)
      return True
    return False
  if shutil.which("gcloud"):
    if _run_cli_rsync(
        ["gcloud", "storage", "rsync", "-r", gcs_uri, str(local_path)]
    ):
      _log_local_cache_status(local_path)
      return True
    return False

  logger.warning(
      "[jax_cache] No supported GCS sync backend available for download."
  )
  return False


def _run_cli_rsync(
    cmd: list[str], timeout_s: float = DEFAULT_CLI_TIMEOUT_S
) -> bool:
  """Runs a gsutil/gcloud rsync command with a bounded timeout."""
  try:
    res = subprocess.run(cmd, check=False, timeout=timeout_s)
    return res.returncode == 0
  except subprocess.TimeoutExpired:
    logger.warning(
        "[jax_cache] CLI sync command %s timed out after %.0fs.",
        cmd[0],
        timeout_s,
    )
    return False


def upload_cache(
    local_dir: str | pathlib.Path,
    gcs_uri: str,
    max_workers: int = 8,
    timeout_s: float = DEFAULT_SYNC_TIMEOUT_S,
) -> bool:
  """Uploads cached compilation artifacts from local_dir to GCS."""
  local_path = pathlib.Path(local_dir)
  if not local_path.is_dir():
    logger.warning(
        "[jax_cache] 0 cache objects detected (local cache directory %s does"
        " not exist); skipping upload.",
        local_path,
    )
    return True

  files = [
      p.relative_to(local_path).as_posix()
      for p in local_path.rglob("*")
      if p.is_file()
  ]
  if not files:
    logger.warning(
        "[jax_cache] 0 cache objects detected in local cache directory %s;"
        " skipping upload.",
        local_path,
    )
    return True

  logger.info(
      "[jax_cache] Detected %d cache objects in local directory %s to upload"
      " to %s.",
      len(files),
      local_path,
      gcs_uri,
  )

  try:
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
  except ValueError as e:
    logger.error("Skipping upload: %s", e)
    return False

  # Try python google-cloud-storage transfer_manager first
  try:
    from google.api_core import exceptions as google_exceptions  # pylint: disable=g-import-not-at-top
    from google.cloud import storage  # pylint: disable=g-import-not-at-top
    from google.cloud.storage import transfer_manager  # pylint: disable=g-import-not-at-top

    project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("PROJECT")
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    existing = {
        b.name[len(prefix) :]
        for b in client.list_blobs(bucket, prefix=prefix, timeout=timeout_s)
    }
    missing = [f for f in files if f not in existing]
    already_in_gcs = len(files) - len(missing)
    if not missing:
      logger.info(
          "[jax_cache] All %d local cache objects already in %s; nothing to"
          " upload.",
          already_in_gcs,
          gcs_uri,
      )
      return True

    logger.info(
        "[jax_cache] Uploading %d artifacts from %s to %s (%d already in"
        " GCS)...",
        len(missing),
        local_path,
        gcs_uri,
        already_in_gcs,
    )
    results = transfer_manager.upload_many_from_filenames(
        bucket,
        missing,
        source_directory=str(local_path),
        blob_name_prefix=prefix,
        # Guards against concurrent writers racing on the same object.
        skip_if_exists=True,
        # Never fork: the caller holds live TPU/gRPC state, and forked children
        # of such a process can segfault or deadlock (default is PROCESS).
        worker_type=transfer_manager.THREAD,
        max_workers=max_workers,
        deadline=int(timeout_s),
        upload_kwargs={"timeout": timeout_s},
    )
    any_failed = False
    skipped = already_in_gcs
    uploaded = 0
    for name, result in zip(missing, results):
      if isinstance(result, Exception):
        is_precondition_failed = isinstance(
            result, google_exceptions.PreconditionFailed
        ) or (
            isinstance(result, google_exceptions.GoogleAPICallError)
            and result.code == 412
        )
        if is_precondition_failed:
          skipped += 1
          continue
        logger.warning("[jax_cache] Failed to upload %s: %s", name, result)
        any_failed = True
      else:
        uploaded += 1
    if any_failed:
      raise RuntimeError("Some artifacts failed to upload.")
    logger.info(
        "[jax_cache] Cache upload completed successfully (%d uploaded, %d"
        " skipped already in GCS).",
        uploaded,
        skipped,
    )
    return True
  except ImportError:
    pass
  except Exception as e:  # pylint: disable=broad-except
    logger.warning("[jax_cache] transfer_manager upload failed: %s", e)

  # Fallback to gsutil or gcloud CLI if available
  if shutil.which("gsutil"):
    if _run_cli_rsync(
        ["gsutil", "-m", "rsync", "-r", str(local_path), gcs_uri]
    ):
      logger.info(
          "[jax_cache] Cache upload completed successfully (%d cache objects"
          " uploaded to %s).",
          len(files),
          gcs_uri,
      )
      return True
    return False
  if shutil.which("gcloud"):
    if _run_cli_rsync(
        ["gcloud", "storage", "rsync", "-r", str(local_path), gcs_uri]
    ):
      logger.info(
          "[jax_cache] Cache upload completed successfully (%d cache objects"
          " uploaded to %s).",
          len(files),
          gcs_uri,
      )
      return True
    return False

  logger.warning(
      "[jax_cache] No supported GCS sync backend available for upload."
  )
  return False


def ensure_jax_cache_env(
    local_dir: str | pathlib.Path | None = None,
) -> pathlib.Path:
  """Sets JAX and vLLM compilation cache env vars to local_dir."""
  path = resolve_local_cache_dir(local_dir)
  path.mkdir(parents=True, exist_ok=True)
  path_str = str(path)
  os.environ["JAX_COMPILATION_CACHE_DIR"] = path_str
  os.environ["VLLM_XLA_CACHE_PATH"] = path_str
  os.environ["VLLM_LOCAL_JAX_CACHE_DIR"] = path_str
  if "jax" in sys.modules:
    import jax  # pylint: disable=g-import-not-at-top

    jax.config.update("jax_compilation_cache_dir", path_str)
  return path


def restore_jax_cache(
    gcs_uri: str | None = None,
    local_dir: str | pathlib.Path | None = None,
    params: RolloutCacheKeyParams | None = None,
) -> bool:
  """Resolves GCS URI and restores compilation cache to local disk before JAX compilation."""
  if is_jax_cache_disabled():
    return False
  path = ensure_jax_cache_env(local_dir)

  resolved_uri = gcs_uri or derive_rollout_cache_uri(params=params)
  if not resolved_uri:
    return False

  if not os.getenv("ROLLOUT_JAX_CACHE_GCS_DIR"):
    os.environ["ROLLOUT_JAX_CACHE_GCS_DIR"] = resolved_uri
  if not os.getenv("VLLM_JAX_CACHE_GCS_DIR"):
    os.environ["VLLM_JAX_CACHE_GCS_DIR"] = resolved_uri

  logger.info(
      "[jax_cache] Restoring compilation cache from %s to %s...",
      resolved_uri,
      path,
  )
  return download_cache(path, resolved_uri)


def save_jax_cache(
    gcs_uri: str | None = None,
    local_dir: str | pathlib.Path | None = None,
    params: RolloutCacheKeyParams | None = None,
) -> bool:
  """Resolves GCS URI and uploads compilation cache to GCS if SAVE_JAX_CACHE is enabled."""
  if is_jax_cache_disabled():
    return False

  save_raw = (
      os.getenv("SAVE_JAX_CACHE") or os.getenv("VLLM_SAVE_JAX_CACHE") or "true"
  )
  save_enabled = _parse_bool_str(save_raw, default=True)
  if not save_enabled:
    return False

  resolved_uri = gcs_uri or derive_rollout_cache_uri(params=params)
  if not resolved_uri:
    return False

  path = resolve_local_cache_dir(local_dir)
  if not path.is_dir():
    return False

  logger.info(
      "[jax_cache] Uploading compilation cache from %s to %s...",
      path,
      resolved_uri,
  )
  return upload_cache(path, resolved_uri)


class JaxCacheSyncer:
  """Manages background JAX compilation cache uploads for a rollout worker."""

  def __init__(
      self,
      worker_id: str,
      config: JaxCacheConfig | None = None,
  ) -> None:
    self.worker_id = worker_id
    self.config: JaxCacheConfig = (
        config if config is not None else JaxCacheConfig.from_env()
    )
    self._pending_sync: tuple[futures.Future[bool], float] | None = None
    self._first_rollout_synced: bool = False
    self._lock = threading.Lock()

  def _resolve_gcs_uri(self) -> str | None:
    """Resolves the target rollout GCS compilation cache URI, if configured."""
    return self.config.resolved_gcs_uri or get_active_gcs_uri()

  def _await_upload(self, pending: tuple[futures.Future[bool], float]) -> None:
    """Waits for a specific JAX cache upload future and logs its outcome."""
    outcome, sync_timeout_s = pending
    try:
      uploaded = outcome.result(timeout=sync_timeout_s)
    except futures.TimeoutError:
      logger.warning(
          "[jax_cache] Upload on rollout worker %s timed out after %.0fs;"
          " abandoning it.",
          self.worker_id,
          sync_timeout_s,
      )
      return
    except Exception as err:  # pylint: disable=broad-except
      logger.warning(
          "[jax_cache] Failed to sync cache on rollout worker %s: %r",
          self.worker_id,
          err,
      )
      return
    if not uploaded:
      logger.warning(
          "[jax_cache] Rollout worker %s reported a failed cache upload.",
          self.worker_id,
      )
      return
    logger.info(
        "[jax_cache] Rollout worker %s cache upload finished.", self.worker_id
    )

  def wait(self) -> None:
    """Waits for any in-flight background JAX cache upload to finish."""
    with self._lock:
      pending = self._pending_sync
      self._pending_sync = None
    if pending is not None:
      self._await_upload(pending)

  def sync(self, *, wait: bool = False) -> futures.Future[bool] | None:
    """Synchronizes local JAX compilation cache to GCS on the primary replica."""
    if (
        not self.config.save_jax_cache
        or is_jax_cache_disabled()
        or not is_primary_rollout_worker(self.worker_id)
    ):
      return None
    gcs_uri = self._resolve_gcs_uri()
    if not gcs_uri:
      return None

    local_dir = self.config.local_dir
    sync_timeout_s = self.config.sync_timeout_s
    logger.info(
        "[jax_cache] Triggering compilation cache upload to GCS (%s) on"
        " rollout worker %s (wait=%s)...",
        gcs_uri,
        self.worker_id,
        wait,
    )

    outcome: futures.Future[bool] = futures.Future()
    with self._lock:
      prior_pending = self._pending_sync
      self._pending_sync = (outcome, sync_timeout_s)

    def _run() -> None:
      if prior_pending is not None:
        self._await_upload(prior_pending)
      try:
        outcome.set_result(save_jax_cache(gcs_uri=gcs_uri, local_dir=local_dir))
      except Exception as err:  # pylint: disable=broad-except
        outcome.set_exception(err)

    threading.Thread(
        target=_run,
        name=f"jax-cache-upload-{self.worker_id}",
        daemon=True,
    ).start()
    if wait:
      self.wait()
    return outcome

  def sync_after_first_completed(
      self, status: str
  ) -> futures.Future[bool] | None:
    """Triggers a one-time background cache upload upon the first COMPLETED rollout."""
    if status != "COMPLETED" or self._first_rollout_synced:
      return None
    with self._lock:
      if self._first_rollout_synced:
        return None
      self._first_rollout_synced = True
    return self.sync(wait=False)


def main(argv: list[str] | None = None) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parser = argparse.ArgumentParser(
      description="JAX compilation cache GCS sync utility"
  )
  parser.add_argument(
      "action",
      choices=["download", "upload", "resolve-uri"],
      help="Action to perform",
  )
  parser.add_argument(
      "local_dir",
      nargs="?",
      default=None,
      help="Local compilation cache directory",
  )
  parser.add_argument(
      "gcs_uri",
      nargs="?",
      default=None,
      help="GCS URI (e.g. gs://bucket/path); derived from env if omitted",
  )
  parser.add_argument(
      "--max-workers",
      type=int,
      default=8,
      help="Number of concurrent workers",
  )

  args = parser.parse_args(argv)
  if args.action == "resolve-uri":
    resolved = derive_rollout_cache_uri()
    if resolved:
      print(resolved)
      sys.exit(0)
    sys.exit(1)

  local_dir = resolve_local_cache_dir(args.local_dir)
  resolved_uri = (args.gcs_uri or "").strip() or derive_rollout_cache_uri()
  if not resolved_uri:
    logger.warning("[jax_cache] No GCS cache URI configured or derived.")
    sys.exit(1)

  success = False
  if args.action == "download":
    success = download_cache(
        local_dir, resolved_uri, max_workers=args.max_workers
    )
  elif args.action == "upload":
    success = upload_cache(
        local_dir, resolved_uri, max_workers=args.max_workers
    )
  sys.exit(0 if success else 1)


if __name__ == "__main__":
  main()
