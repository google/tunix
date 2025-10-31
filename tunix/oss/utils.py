# Copyright 2025 Google LLC
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
"""Utils for OSS code."""

import os


def pathways_available() -> bool:
  if "proxy" not in os.getenv("JAX_PLATFORMS", ""):
    return False
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top, unused-import

    return True
  except ImportError:
    return False


def load_file_from_gcs(file_dir: str):
  """Load file from GCS."""
  if file_dir.startswith("/"):
    return file_dir
  bucket_name, file_name = file_dir[5:].split("/", 1)
  try:
    from google.cloud import storage  # pylint: disable=g-import-not-at-top
    import tempfile  # pylint: disable=g-import-not-at-top
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    temp_dir = tempfile.TemporaryDirectory().name
    local_file_path = os.path.join(temp_dir, file_name)
    blob.download_to_filename(local_file_path)
    return local_file_path
  except ImportError as e:
    raise ImportError(
        "Please install google-cloud-storage to load model from GCS."
    ) from e

