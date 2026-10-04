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
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


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


def _log_local_cache_status(local_path: Path, context: str = "after download") -> int:
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
        "[jax_cache] Cache download completed successfully (%d cache objects detected in %s).",
        len(local_files),
        local_path,
    )
  return len(local_files)


def download_cache(local_dir: str | Path, gcs_uri: str, max_workers: int = 8) -> bool:
  """Downloads cached compilation artifacts from GCS into local_dir."""
  local_path = Path(local_dir)
  try:
    bucket_name, prefix = _parse_gcs_uri(gcs_uri)
  except ValueError as e:
    logger.error("Skipping download: %s", e)
    return False

  # Try python google-cloud-storage transfer_manager first
  try:
    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT")
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=prefix))
    blob_names = [
        b.name[len(prefix):]
        for b in blobs
        if not b.name.endswith("/") and len(b.name) > len(prefix)
    ]
    if not blob_names:
      logger.warning(
          "[jax_cache] 0 cache objects detected in %s (cold cache start). JAX will compile on the fly.",
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
        max_workers=max_workers,
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
  except Exception as e:
    logger.warning("[jax_cache] transfer_manager download failed: %s", e)

  # Fallback to gsutil or gcloud CLI if available
  if shutil.which("gsutil"):
    res = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", gcs_uri, str(local_path)], check=False
    )
    if res.returncode == 0:
      _log_local_cache_status(local_path)
      return True
    return False
  if shutil.which("gcloud"):
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "-r", gcs_uri, str(local_path)],
        check=False,
    )
    if res.returncode == 0:
      _log_local_cache_status(local_path)
      return True
    return False

  logger.warning("[jax_cache] No supported GCS sync backend available for download.")
  return False


def upload_cache(local_dir: str | Path, gcs_uri: str, max_workers: int = 8) -> bool:
  """Uploads cached compilation artifacts from local_dir to GCS."""
  local_path = Path(local_dir)
  if not local_path.is_dir():
    logger.warning(
        "[jax_cache] 0 cache objects detected (local cache directory %s does not exist); skipping upload.",
        local_path,
    )
    return True

  files = [p.relative_to(local_path).as_posix() for p in local_path.rglob("*") if p.is_file()]
  if not files:
    logger.warning(
        "[jax_cache] 0 cache objects detected in local cache directory %s; skipping upload.",
        local_path,
    )
    return True

  logger.info(
      "[jax_cache] Detected %d cache objects in local directory %s to upload to %s.",
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
    from google.api_core import exceptions as google_exceptions
    from google.cloud import storage
    from google.cloud.storage import transfer_manager

    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("PROJECT")
    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)

    logger.info(
        "[jax_cache] Uploading %d artifacts from %s to %s...",
        len(files),
        local_path,
        gcs_uri,
    )
    results = transfer_manager.upload_many_from_filenames(
        bucket,
        files,
        source_directory=str(local_path),
        blob_name_prefix=prefix,
        skip_if_exists=True,
        max_workers=max_workers,
    )
    any_failed = False
    skipped = 0
    uploaded = 0
    for name, result in zip(files, results):
      if isinstance(result, Exception):
        is_precondition_failed = (
            getattr(result, "code", None) == 412
            or type(result).__name__ == "PreconditionFailed"
            or (
                isinstance(getattr(google_exceptions, "PreconditionFailed", None), type)
                and isinstance(result, google_exceptions.PreconditionFailed)
            )
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
        "[jax_cache] Cache upload completed successfully (%d uploaded, %d skipped already in GCS).",
        uploaded,
        skipped,
    )
    return True
  except ImportError:
    pass
  except Exception as e:
    logger.warning("[jax_cache] transfer_manager upload failed: %s", e)

  # Fallback to gsutil or gcloud CLI if available
  if shutil.which("gsutil"):
    res = subprocess.run(
        ["gsutil", "-m", "rsync", "-r", str(local_path), gcs_uri], check=False
    )
    if res.returncode == 0:
      logger.info(
          "[jax_cache] Cache upload completed successfully (%d cache objects uploaded to %s).",
          len(files),
          gcs_uri,
      )
      return True
    return False
  if shutil.which("gcloud"):
    res = subprocess.run(
        ["gcloud", "storage", "rsync", "-r", str(local_path), gcs_uri],
        check=False,
    )
    if res.returncode == 0:
      logger.info(
          "[jax_cache] Cache upload completed successfully (%d cache objects uploaded to %s).",
          len(files),
          gcs_uri,
      )
      return True
    return False

  logger.warning("[jax_cache] No supported GCS sync backend available for upload.")
  return False


def ensure_jax_cache_env(local_dir: str | Path | None = None) -> Path:
  """Sets JAX and vLLM compilation cache env vars to local_dir."""
  path = Path(
      local_dir
      or os.getenv("LOCAL_JAX_CACHE_DIR")
      or os.getenv("JAX_CACHE_DIR")
      or os.getenv("JAX_COMPILATION_CACHE_DIR")
      or "/tmp/jax_cache"
  )
  path.mkdir(parents=True, exist_ok=True)
  path_str = str(path)
  os.environ["JAX_COMPILATION_CACHE_DIR"] = path_str
  os.environ["VLLM_XLA_CACHE_PATH"] = path_str
  if "jax" in sys.modules:
    import jax  # pylint: disable=g-import-not-at-top
    jax.config.update("jax_compilation_cache_dir", path_str)
  return path


def restore_jax_cache(
    gcs_uri: str | None = None,
    local_dir: str | Path | None = None,
    role: str | None = None,
) -> bool:
  """Resolves GCS URI and restores compilation cache to local disk before JAX compilation."""
  if not gcs_uri:
    gcs_uri = os.getenv("ROLLOUT_JAX_CACHE_GCS_DIR") or os.getenv("JAX_CACHE_GCS_DIR")

  path = ensure_jax_cache_env(local_dir)
  if not gcs_uri:
    return False

  logger.info("[jax_cache] Restoring compilation cache from %s to %s...", gcs_uri, path)
  return download_cache(path, gcs_uri)


def save_jax_cache(
    gcs_uri: str | None = None,
    local_dir: str | Path | None = None,
    role: str | None = None,
) -> bool:
  """Resolves GCS URI and uploads compilation cache to GCS if SAVE_JAX_CACHE is enabled."""
  save_enabled = os.getenv("SAVE_JAX_CACHE", "true").strip().lower() in ("1", "true", "yes")
  if not save_enabled:
    return False

  if not gcs_uri:
    gcs_uri = os.getenv("ROLLOUT_JAX_CACHE_GCS_DIR") or os.getenv("JAX_CACHE_GCS_DIR")

  if not gcs_uri:
    return False

  path = Path(
      local_dir
      or os.getenv("LOCAL_JAX_CACHE_DIR")
      or os.getenv("JAX_CACHE_DIR")
      or os.getenv("JAX_COMPILATION_CACHE_DIR")
      or "/tmp/jax_cache"
  )
  if not path.is_dir():
    return False

  logger.info("[jax_cache] Uploading compilation cache from %s to %s...", path, gcs_uri)
  return upload_cache(path, gcs_uri)



def main(argv: list[str] | None = None) -> None:
  logging.basicConfig(level=logging.INFO, format="%(message)s")
  parser = argparse.ArgumentParser(
      description="JAX compilation cache GCS sync utility"
  )
  parser.add_argument(
      "action", choices=["download", "upload"], help="Action to perform"
  )
  parser.add_argument("local_dir", help="Local compilation cache directory")
  parser.add_argument("gcs_uri", help="GCS URI (e.g. gs://bucket/path)")
  parser.add_argument(
      "--max-workers",
      type=int,
      default=8,
      help="Number of concurrent workers",
  )

  args = parser.parse_args(argv)
  success = False
  if args.action == "download":
    success = download_cache(args.local_dir, args.gcs_uri, max_workers=args.max_workers)
  elif args.action == "upload":
    success = upload_cache(args.local_dir, args.gcs_uri, max_workers=args.max_workers)
  sys.exit(0 if success else 1)


if __name__ == "__main__":
  main()
