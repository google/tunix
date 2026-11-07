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

from collections.abc import Callable
import os
import shutil
import sys
import tempfile

from absl import logging
import fsspec
import huggingface_hub as hf
from huggingface_hub import errors as hf_errors

_HF_AUTH_EXCEPTIONS = (
    hf_errors.HfHubHTTPError,
    hf_errors.LocalTokenNotFoundError,
)


def pathways_available() -> bool:
  if 'proxy' not in os.getenv('JAX_PLATFORMS', ''):
    return False
  try:
    import pathwaysutils  # pylint: disable=g-import-not-at-top, unused-import  # pyrefly: ignore[missing-import]

    return True
  except ImportError:
    return False


def load_file_from_gcs(file_dir: str, target_dir: str | None = None) -> str:
  """Load file from GCS."""
  if not file_dir.startswith('gs://'):
    raise ValueError(f'Invalid GCS path: {file_dir}')

  _, prefix = file_dir[5:].split('/', 1)
  try:
    if target_dir is None:
      target_dir = tempfile.gettempdir()
    local_dir = os.path.join(target_dir, prefix)

    fsspec_fs = fsspec.filesystem('gs')
    fsspec_fs.get(file_dir, local_dir, recursive=True)

    return local_dir
  except ImportError as e:
    raise ImportError(
        'Please install google-cloud-storage to load model from GCS.'
    ) from e


def _can_prompt_interactively() -> bool:
  """Returns True when interactive credential prompts will not hang."""
  if os.getenv('CI', '').strip().lower() in ('1', 'true', 'yes'):
    return False
  if 'IPython' in sys.modules and sys.modules['IPython'] is not None:
    shell = sys.modules['IPython'].get_ipython()
    if shell is not None and any(
        cls.__name__ == 'ZMQInteractiveShell' for cls in type(shell).__mro__
    ):
      return shell.kernel is not None and bool(
          shell.kernel._allow_stdin  # pylint: disable=protected-access
      )
  return sys.stdin is not None and sys.stdin.isatty()


def _prompt_interactive_login(
    login_fn: Callable[[], None],
    err: Exception,
    *,
    provider: str,
    skip_login: bool = False,
) -> None:
  """Runs `login_fn()` when interactive and allowed, else re-raises `err`."""
  if skip_login or not _can_prompt_interactively():
    raise err
  try:
    login_fn()
  except ImportError as login_err:
    logging.warning(
        'Interactive %s login unavailable (%s); re-raising auth error.',
        provider,
        login_err,
    )
    raise err from login_err


def _prompt_hf_login(err: Exception, *, logged_in: bool = False) -> None:
  """Prompts `huggingface_hub.login()` once on interactive HF auth errors."""
  has_env_token = bool(
      os.getenv('HF_TOKEN', '').strip()
      or os.getenv('HUGGING_FACE_HUB_TOKEN', '').strip()
  )
  is_auth_err = not isinstance(err, hf_errors.DisabledRepoError) and (
      isinstance(
          err, (hf_errors.LocalTokenNotFoundError, hf_errors.GatedRepoError)
      )
      or (
          isinstance(err, hf_errors.HfHubHTTPError)
          and err.response is not None
          and err.response.status_code in (401, 403)
      )
  )
  _prompt_interactive_login(
      hf.login,
      err,
      provider='Hugging Face',
      skip_login=logged_in or has_env_token or not is_auth_err,
  )


def _backfill_kagglesdk_web_endpoint() -> None:
  """Backfills `kagglesdk.kaggle_env.get_web_endpoint` before `kagglehub`."""
  try:
    import kagglesdk.kaggle_env  # pylint: disable=g-import-not-at-top
  except ImportError:
    return
  if not hasattr(kagglesdk.kaggle_env, 'get_web_endpoint') and hasattr(
      kagglesdk.kaggle_env, 'get_endpoint'
  ):
    kagglesdk.kaggle_env.get_web_endpoint = kagglesdk.kaggle_env.get_endpoint


def _link_or_copy_cached_file(cached_path: str, dst: str) -> None:
  """Hard-links `cached_path` to `dst`, falling back to copy across mounts."""
  os.makedirs(os.path.dirname(dst), exist_ok=True)
  try:
    os.remove(dst)
  except FileNotFoundError:
    pass
  try:
    os.link(os.path.realpath(cached_path), dst)
  except FileExistsError:
    pass  # Another process already linked it successfully.
  except OSError:
    shutil.copy2(cached_path, dst)


def kaggle_pipeline(
    model_id: str, model_download_path: str | None = None
) -> str:
  """Downloads a model from Kaggle, prompting login only if interactive."""
  # `kagglehub` imports `get_web_endpoint` from `kagglesdk.kaggle_env` at
  # import time, so backfill older `kagglesdk` releases before importing it.
  _backfill_kagglesdk_web_endpoint()
  import kagglehub  # pylint: disable=g-import-not-at-top
  from kagglehub import exceptions as kaggle_exceptions  # pylint: disable=g-import-not-at-top

  if model_download_path is not None:
    os.environ['KAGGLEHUB_CACHE'] = model_download_path
  kaggle_auth_errors = (
      kaggle_exceptions.UnauthenticatedError,
      kaggle_exceptions.CredentialError,
      kaggle_exceptions.KaggleApiHTTPError,
      kaggle_exceptions.ColabHTTPError,
  )
  try:
    return kagglehub.model_download(model_id)
  except kaggle_auth_errors as err:
    has_env_creds = bool(os.getenv('KAGGLE_API_TOKEN', '').strip()) or bool(
        os.getenv('KAGGLE_USERNAME', '').strip()
        and os.getenv('KAGGLE_KEY', '').strip()
    )
    is_non_auth_http = isinstance(
        err,
        (
            kaggle_exceptions.KaggleApiHTTPError,
            kaggle_exceptions.ColabHTTPError,
        ),
    ) and (err.response is None or err.response.status_code not in (401, 403))
    _prompt_interactive_login(
        kagglehub.login,
        err,
        provider='Kaggle',
        skip_login=has_env_creds or is_non_auth_http,
    )
  return kagglehub.model_download(model_id)


def hf_pipeline(model_id: str, model_download_path: str | None = None) -> str:
  """Downloads a model from Hugging Face, prompting login if interactive."""
  if model_download_path is None:
    model_download_path = os.path.join(
        tempfile.gettempdir(), 'models', model_id
    )
  logged_in = False
  try:
    all_files = hf.list_repo_files(model_id)
  except _HF_AUTH_EXCEPTIONS as err:
    _prompt_hf_login(err)
    logged_in = True
    all_files = hf.list_repo_files(model_id)

  filtered_files = [f for f in all_files if not f.startswith('original/')]
  for filename in filtered_files:
    dst = os.path.join(model_download_path, filename)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
      continue
    try:
      cached_path = hf.hf_hub_download(repo_id=model_id, filename=filename)
    except _HF_AUTH_EXCEPTIONS as err:
      _prompt_hf_login(err, logged_in=logged_in)
      logged_in = True
      cached_path = hf.hf_hub_download(repo_id=model_id, filename=filename)
    _link_or_copy_cached_file(cached_path, dst)

  os.makedirs(model_download_path, exist_ok=True)
  with open(os.path.join(model_download_path, '.hf_complete'), 'w') as f:
    f.write(model_id)
  logging.info(
      'Downloaded %s to: %s',
      filtered_files,
      model_download_path,
  )
  return model_download_path
