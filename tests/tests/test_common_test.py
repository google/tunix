# Copyright 2025 Google LLC
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

import importlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
from typing import Any, cast
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import huggingface_hub
from huggingface_hub import errors as hf_errors
import kagglehub
from kagglehub import exceptions as kaggle_exceptions
import requests
from tunix.oss import utils as oss_utils
from tunix.tests import test_common

flags.FLAGS.mark_as_parsed()


def _make_http_response(status_code: int) -> requests.Response:
  resp = requests.Response()
  resp.status_code = status_code
  resp.url = "https://example.com/repo"
  resp.request = requests.Request("GET", resp.url).prepare()
  return resp


def _make_hf_http_error(
    status_code: int | None,
    cls: type[hf_errors.HfHubHTTPError] = hf_errors.HfHubHTTPError,
) -> hf_errors.HfHubHTTPError:
  resp = cast(Any, _make_http_response(status_code or 500))
  err = cls(f"HTTP {status_code}", response=resp)
  if status_code is None:
    err.response = cast(Any, None)
  return err


def _make_tty_stdin(is_tty: bool) -> io.TextIOBase:
  stdin = mock.create_autospec(io.TextIOBase, instance=True, spec_set=True)
  stdin.isatty.return_value = is_tty
  return stdin


def _make_zmq_ipython_module(
    *, allow_stdin: bool = True, has_kernel: bool = True
) -> types.ModuleType:
  FakeKernel = type("FakeKernel", (), {"_allow_stdin": allow_stdin})
  kernel = FakeKernel() if has_kernel else None
  ZMQInteractiveShell = type("ZMQInteractiveShell", (), {"kernel": kernel})
  ColabShell = type("Shell", (ZMQInteractiveShell,), {})
  fake_ipython = types.ModuleType("IPython")
  setattr(fake_ipython, "get_ipython", ColabShell)
  return fake_ipython


def _make_empty_ipython_module() -> types.ModuleType:
  fake_ipython = types.ModuleType("IPython")
  setattr(fake_ipython, "get_ipython", lambda: None)
  return fake_ipython


class SafeDownloadTest(parameterized.TestCase):

  def test_safe_download_returns_existing_nonempty_file(self):
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    dst = os.path.join(local_dir, "model.safetensors")
    with open(dst, "w") as f:
      f.write("cached-weights")

    with mock.patch.object(
        huggingface_hub, "hf_hub_download", autospec=True, spec_set=True
    ) as mock_hf_download:
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, dst)
    mock_hf_download.assert_not_called()

  def test_safe_download_links_from_hf_cache(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with mock.patch.object(
        huggingface_hub,
        "hf_hub_download",
        autospec=True,
        spec_set=True,
        return_value=cached_file,
    ) as mock_hf_download:
      result = test_common.safe_download(
          "org/repo", "sub/model.safetensors", local_dir
      )

    expected_dst = os.path.join(local_dir, "sub/model.safetensors")
    self.assertEqual(result, expected_dst)
    mock_hf_download.assert_called_once_with(
        repo_id="org/repo", filename="sub/model.safetensors"
    )
    self.assertTrue(os.path.samefile(cached_file, expected_dst))

  def test_safe_download_handles_file_exists_race_without_copying(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with (
        mock.patch.object(
            huggingface_hub,
            "hf_hub_download",
            autospec=True,
            spec_set=True,
            return_value=cached_file,
        ),
        mock.patch.object(
            os,
            "link",
            autospec=True,
            spec_set=True,
            side_effect=FileExistsError,
        ),
        mock.patch.object(
            shutil, "copy2", autospec=True, spec_set=True
        ) as mock_copy2,
    ):
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, os.path.join(local_dir, "model.safetensors"))
    mock_copy2.assert_not_called()

  def test_safe_download_falls_back_to_copy_on_oserror(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with (
        mock.patch.object(
            huggingface_hub,
            "hf_hub_download",
            autospec=True,
            spec_set=True,
            return_value=cached_file,
        ),
        mock.patch.object(
            os,
            "link",
            autospec=True,
            spec_set=True,
            side_effect=OSError("EXDEV"),
        ),
    ):
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, os.path.join(local_dir, "model.safetensors"))
    with open(result) as f:
      self.assertEqual(f.read(), "downloaded-weights")

  def test_hf_pipeline_links_and_skips_existing_files(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_456")
    with open(cached_file, "w") as f:
      f.write("config-content")

    existing_file = os.path.join(local_dir, "already_there.json")
    with open(existing_file, "w") as f:
      f.write("existing")
    empty_file = os.path.join(local_dir, "empty.json")
    with open(empty_file, "w"):
      pass

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            return_value=[
                "already_there.json",
                "empty.json",
                "config.json",
                "original/ignored.bin",
            ],
        ),
        mock.patch.object(
            huggingface_hub,
            "hf_hub_download",
            autospec=True,
            spec_set=True,
            return_value=cached_file,
        ) as mock_hf_download,
        mock.patch.object(
            huggingface_hub, "login", autospec=True, spec_set=True
        ) as mock_login,
    ):
      result = oss_utils.hf_pipeline("org/repo", local_dir)

    self.assertEqual(result, local_dir)
    mock_login.assert_not_called()
    self.assertEqual(
        mock_hf_download.call_args_list,
        [
            mock.call(repo_id="org/repo", filename="empty.json"),
            mock.call(repo_id="org/repo", filename="config.json"),
        ],
    )
    self.assertTrue(
        os.path.samefile(cached_file, os.path.join(local_dir, "config.json"))
    )

  def test_hf_pipeline_defaults_download_path_when_none(self):
    temp_root = self.create_tempdir().full_path
    with (
        mock.patch.object(
            tempfile,
            "gettempdir",
            autospec=True,
            spec_set=True,
            return_value=temp_root,
        ),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            return_value=[],
        ),
    ):
      result = oss_utils.hf_pipeline("org/repo", None)

    self.assertEqual(result, os.path.join(temp_root, "models", "org/repo"))

  @parameterized.named_parameters(
      dict(
          testcase_name="http_401_missing_token",
          env={"CI": "false"},
          make_error=lambda: _make_hf_http_error(401),
      ),
      dict(
          testcase_name="http_403_blank_token",
          env={"HF_TOKEN": "   ", "HUGGING_FACE_HUB_TOKEN": ""},
          make_error=lambda: _make_hf_http_error(403),
      ),
      dict(
          testcase_name="gated_repo_error",
          env={},
          make_error=lambda: _make_hf_http_error(403, hf_errors.GatedRepoError),
      ),
      dict(
          testcase_name="local_token_not_found",
          env={},
          make_error=lambda: hf_errors.LocalTokenNotFoundError("Token missing"),
      ),
  )
  def test_hf_pipeline_prompts_login_on_list_files_auth_error_when_interactive(
      self, env, make_error
  ):
    cache_dir = self.create_tempdir().full_path
    local_dir = self.create_tempdir().full_path
    cached_file = os.path.join(cache_dir, "blob_789")
    with open(cached_file, "w") as f:
      f.write("config-content")

    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict(sys.modules, {"IPython": None}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(True)),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            side_effect=[make_error(), ["config.json"]],
        ) as mock_list_files,
        mock.patch.object(
            huggingface_hub, "login", autospec=True, spec_set=True
        ) as mock_login,
        mock.patch.object(
            huggingface_hub,
            "hf_hub_download",
            autospec=True,
            spec_set=True,
            return_value=cached_file,
        ),
    ):
      result = oss_utils.hf_pipeline("gated/repo", local_dir)

    self.assertEqual(result, local_dir)
    mock_login.assert_called_once_with()
    self.assertEqual(mock_list_files.call_count, 2)

  def test_hf_pipeline_prompts_login_on_gated_file_download_and_retries_once(
      self,
  ):
    cache_dir = self.create_tempdir().full_path
    local_dir = self.create_tempdir().full_path
    cached_file = os.path.join(cache_dir, "blob_gated")
    with open(cached_file, "w") as f:
      f.write("weights")

    with (
        mock.patch.dict(os.environ, {"HF_TOKEN": ""}, clear=True),
        mock.patch.dict(sys.modules, {"IPython": _make_zmq_ipython_module()}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(False)),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            return_value=["config.json", "model.safetensors"],
        ),
        mock.patch.object(
            huggingface_hub, "login", autospec=True, spec_set=True
        ) as mock_login,
        mock.patch.object(
            huggingface_hub,
            "hf_hub_download",
            autospec=True,
            spec_set=True,
            side_effect=[
                _make_hf_http_error(401, hf_errors.GatedRepoError),
                cached_file,
                _make_hf_http_error(403, hf_errors.GatedRepoError),
            ],
        ) as mock_download,
    ):
      with self.assertRaises(hf_errors.GatedRepoError):
        oss_utils.hf_pipeline("meta-llama/Llama-3.2-1B", local_dir)

    mock_login.assert_called_once_with()
    self.assertEqual(mock_download.call_count, 3)

  @parameterized.named_parameters(
      dict(
          testcase_name="hf_token_set",
          env={"HF_TOKEN": "bad_token"},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(401),
      ),
      dict(
          testcase_name="legacy_hf_hub_token_set",
          env={"HUGGING_FACE_HUB_TOKEN": "bad_token"},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(403),
      ),
      dict(
          testcase_name="non_tty_headless",
          env={},
          is_tty=False,
          ipython_factory=None,
          error_cls=hf_errors.GatedRepoError,
          make_error=lambda: _make_hf_http_error(401, hf_errors.GatedRepoError),
      ),
      dict(
          testcase_name="ci_true_blocks_tty_and_notebook",
          env={"CI": " true "},
          is_tty=True,
          ipython_factory=_make_zmq_ipython_module,
          error_cls=hf_errors.LocalTokenNotFoundError,
          make_error=lambda: hf_errors.LocalTokenNotFoundError("No token"),
      ),
      dict(
          testcase_name="ci_1_blocks_prompt",
          env={"CI": "1"},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(401),
      ),
      dict(
          testcase_name="headless_notebook_allow_stdin_false",
          env={},
          is_tty=True,
          ipython_factory=lambda: _make_zmq_ipython_module(allow_stdin=False),
          error_cls=hf_errors.GatedRepoError,
          make_error=lambda: _make_hf_http_error(401, hf_errors.GatedRepoError),
      ),
      dict(
          testcase_name="unattached_notebook_shell_without_kernel",
          env={},
          is_tty=True,
          ipython_factory=lambda: _make_zmq_ipython_module(has_kernel=False),
          error_cls=hf_errors.GatedRepoError,
          make_error=lambda: _make_hf_http_error(401, hf_errors.GatedRepoError),
      ),
      dict(
          testcase_name="ipython_loaded_without_active_shell",
          env={},
          is_tty=False,
          ipython_factory=_make_empty_ipython_module,
          error_cls=hf_errors.LocalTokenNotFoundError,
          make_error=lambda: hf_errors.LocalTokenNotFoundError("No token"),
      ),
      dict(
          testcase_name="http_404_public_not_found",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.RepositoryNotFoundError,
          make_error=lambda: _make_hf_http_error(
              404, hf_errors.RepositoryNotFoundError
          ),
      ),
      dict(
          testcase_name="http_429_rate_limit",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(429),
      ),
      dict(
          testcase_name="http_503_unavailable",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(503),
      ),
      dict(
          testcase_name="http_error_without_response",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.HfHubHTTPError,
          make_error=lambda: _make_hf_http_error(None),
      ),
      dict(
          testcase_name="disabled_repo_403",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=hf_errors.DisabledRepoError,
          make_error=lambda: _make_hf_http_error(
              403, hf_errors.DisabledRepoError
          ),
      ),
  )
  def test_hf_pipeline_reraises_without_login(
      self, env, is_tty, ipython_factory, error_cls, make_error
  ):
    local_dir = self.create_tempdir().full_path
    ipython_entry = ipython_factory() if ipython_factory is not None else None
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict(sys.modules, {"IPython": ipython_entry}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(is_tty)),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            side_effect=make_error(),
        ),
        mock.patch.object(
            huggingface_hub, "login", autospec=True, spec_set=True
        ) as mock_login,
    ):
      with self.assertRaises(error_cls):
        oss_utils.hf_pipeline("org/repo", local_dir)

    mock_login.assert_not_called()

  def test_hf_pipeline_reraises_auth_error_when_login_raises_import_error(self):
    local_dir = self.create_tempdir().full_path
    missing_widgets = ImportError("No module named 'ipywidgets'")
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.dict(sys.modules, {"IPython": _make_zmq_ipython_module()}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(False)),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            side_effect=_make_hf_http_error(403, hf_errors.GatedRepoError),
        ) as mock_list_files,
        mock.patch.object(
            huggingface_hub,
            "login",
            autospec=True,
            spec_set=True,
            side_effect=missing_widgets,
        ) as mock_login,
    ):
      with self.assertRaises(hf_errors.GatedRepoError) as ctx:
        oss_utils.hf_pipeline("meta-llama/Llama-3.2-1B", local_dir)

    self.assertIs(ctx.exception.__cause__, missing_widgets)
    mock_login.assert_called_once_with()
    mock_list_files.assert_called_once_with("meta-llama/Llama-3.2-1B")

  def test_kaggle_pipeline_downloads_directly_without_env_credentials(self):
    download_dir = self.create_tempdir().full_path
    expected_path = os.path.join(download_dir, "gemma2-2b")
    observed_cache_dirs: list[str] = []

    def fake_model_download(handle: str, *args, **kwargs) -> str:
      del handle, args, kwargs
      observed_cache_dirs.append(os.environ["KAGGLEHUB_CACHE"])
      return expected_path

    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(
            kagglehub, "login", autospec=True, spec_set=True
        ) as mock_login,
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            side_effect=fake_model_download,
        ) as mock_download,
    ):
      result = oss_utils.kaggle_pipeline(
          "google/gemma-2/flax/gemma2-2b", download_dir
      )

    self.assertEqual(result, expected_path)
    self.assertEqual(observed_cache_dirs, [download_dir])
    mock_login.assert_not_called()
    mock_download.assert_called_once_with("google/gemma-2/flax/gemma2-2b")

  def test_kaggle_pipeline_leaves_default_cache_when_download_path_is_none(
      self,
  ):
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            return_value="/default/cache/model",
        ),
    ):
      result = oss_utils.kaggle_pipeline("google/gemma-2/flax/gemma2-2b", None)
      cache_set = "KAGGLEHUB_CACHE" in os.environ

    self.assertEqual(result, "/default/cache/model")
    self.assertFalse(cache_set)

  @parameterized.named_parameters(
      dict(
          testcase_name="unauthenticated_missing_both",
          env={},
          make_error=kaggle_exceptions.UnauthenticatedError,
      ),
      dict(
          testcase_name="credential_error_missing_username",
          env={"KAGGLE_KEY": "kaggle_secret"},
          make_error=lambda: kaggle_exceptions.CredentialError("bad creds"),
      ),
      dict(
          testcase_name="http_401_missing_key",
          env={"KAGGLE_USERNAME": "kaggle_user", "KAGGLE_API_TOKEN": "  "},
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "401 Client Error", response=_make_http_response(401)
          ),
      ),
      dict(
          testcase_name="colab_http_403_blank_creds",
          env={"KAGGLE_USERNAME": " ", "KAGGLE_KEY": ""},
          make_error=lambda: kaggle_exceptions.ColabHTTPError(
              "403 Client Error", response=_make_http_response(403)
          ),
      ),
  )
  def test_kaggle_pipeline_prompts_login_on_auth_error_when_interactive(
      self, env, make_error
  ):
    download_dir = self.create_tempdir().full_path
    expected_path = os.path.join(download_dir, "gemma2-2b")
    call_order: list[str] = []
    attempts = 0

    def record_login(*args, **kwargs):
      del args, kwargs
      call_order.append("login")

    def record_download(handle: str, *args, **kwargs) -> str:
      nonlocal attempts
      del handle, args, kwargs
      attempts += 1
      call_order.append(f"download:{os.environ['KAGGLEHUB_CACHE']}")
      if attempts == 1:
        raise make_error()
      return expected_path

    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict(sys.modules, {"IPython": None}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(True)),
        mock.patch.object(
            kagglehub,
            "login",
            autospec=True,
            spec_set=True,
            side_effect=record_login,
        ) as mock_login,
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            side_effect=record_download,
        ) as mock_download,
    ):
      result = oss_utils.kaggle_pipeline(
          "google/gemma-2/flax/gemma2-2b", download_dir
      )

    self.assertEqual(result, expected_path)
    self.assertEqual(
        call_order,
        [
            f"download:{download_dir}",
            "login",
            f"download:{download_dir}",
        ],
    )
    mock_login.assert_called_once_with()
    self.assertEqual(mock_download.call_count, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name="username_and_key_set",
          env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"},
          is_tty=True,
          error_cls=kaggle_exceptions.KaggleApiHTTPError,
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "403", response=_make_http_response(403)
          ),
      ),
      dict(
          testcase_name="api_token_set",
          env={"KAGGLE_API_TOKEN": "kgat_token"},
          is_tty=True,
          error_cls=kaggle_exceptions.UnauthenticatedError,
          make_error=kaggle_exceptions.UnauthenticatedError,
      ),
      dict(
          testcase_name="headless_non_tty",
          env={},
          is_tty=False,
          error_cls=kaggle_exceptions.UnauthenticatedError,
          make_error=kaggle_exceptions.UnauthenticatedError,
      ),
      dict(
          testcase_name="ci_yes_blocks_prompt",
          env={"CI": "YES"},
          is_tty=True,
          error_cls=kaggle_exceptions.KaggleApiHTTPError,
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "401", response=_make_http_response(401)
          ),
      ),
      dict(
          testcase_name="http_404_not_found",
          env={},
          is_tty=True,
          error_cls=kaggle_exceptions.KaggleApiHTTPError,
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "404", response=_make_http_response(404)
          ),
      ),
      dict(
          testcase_name="http_500_server_error",
          env={},
          is_tty=True,
          error_cls=kaggle_exceptions.KaggleApiHTTPError,
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "500", response=_make_http_response(500)
          ),
      ),
      dict(
          testcase_name="headless_notebook_allow_stdin_false",
          env={},
          is_tty=True,
          ipython_factory=lambda: _make_zmq_ipython_module(allow_stdin=False),
          error_cls=kaggle_exceptions.UnauthenticatedError,
          make_error=kaggle_exceptions.UnauthenticatedError,
      ),
      dict(
          testcase_name="http_error_without_response",
          env={},
          is_tty=True,
          ipython_factory=None,
          error_cls=kaggle_exceptions.KaggleApiHTTPError,
          make_error=lambda: kaggle_exceptions.KaggleApiHTTPError(
              "no response", response=None
          ),
      ),
  )
  def test_kaggle_pipeline_reraises_without_login(
      self, env, is_tty, error_cls, make_error, ipython_factory=None
  ):
    download_dir = self.create_tempdir().full_path
    ipython_entry = ipython_factory() if ipython_factory is not None else None
    with (
        mock.patch.dict(os.environ, env, clear=True),
        mock.patch.dict(sys.modules, {"IPython": ipython_entry}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(is_tty)),
        mock.patch.object(
            kagglehub, "login", autospec=True, spec_set=True
        ) as mock_login,
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            side_effect=make_error(),
        ),
    ):
      with self.assertRaises(error_cls):
        oss_utils.kaggle_pipeline("google/gemma-2/flax/gemma2-2b", download_dir)

    mock_login.assert_not_called()

  def test_kaggle_pipeline_reraises_auth_error_when_login_raises_import_error(
      self,
  ):
    download_dir = self.create_tempdir().full_path
    missing_widgets = ImportError(
        "You need the `ipywidgets` module: `pip install ipywidgets`."
    )
    with (
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.dict(sys.modules, {"IPython": _make_zmq_ipython_module()}),
        mock.patch.object(sys, "stdin", new=_make_tty_stdin(False)),
        mock.patch.object(
            kagglehub,
            "login",
            autospec=True,
            spec_set=True,
            side_effect=missing_widgets,
        ) as mock_login,
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            side_effect=kaggle_exceptions.UnauthenticatedError(),
        ) as mock_download,
    ):
      with self.assertRaises(kaggle_exceptions.UnauthenticatedError) as ctx:
        oss_utils.kaggle_pipeline("google/gemma-2/flax/gemma2-2b", download_dir)

    self.assertIs(ctx.exception.__cause__, missing_widgets)
    mock_login.assert_called_once_with()
    mock_download.assert_called_once_with("google/gemma-2/flax/gemma2-2b")

  def test_import_oss_utils_succeeds_without_kagglehub_installed(self):
    local_dir = self.create_tempdir().full_path
    spec = importlib.util.spec_from_file_location(
        "_isolated_tunix_oss_utils", oss_utils.__file__
    )
    self.assertIsNotNone(spec)
    assert spec is not None and spec.loader is not None
    isolated_mod = importlib.util.module_from_spec(spec)
    blocked = {
        "kagglehub": None,
        "kagglehub.exceptions": None,
        "kagglesdk": None,
        "kagglesdk.kaggle_env": None,
    }
    with (
        mock.patch.dict(sys.modules, blocked),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            autospec=True,
            spec_set=True,
            return_value=[],
        ),
    ):
      spec.loader.exec_module(isolated_mod)
      result = isolated_mod.hf_pipeline("org/repo", local_dir)

    self.assertEqual(result, local_dir)

  def test_kaggle_pipeline_backfills_missing_kagglesdk_web_endpoint(self):
    download_dir = self.create_tempdir().full_path
    fake_site_dir = self.create_tempdir().full_path
    fake_kagglehub_dir = os.path.join(fake_site_dir, "kagglehub")
    os.makedirs(fake_kagglehub_dir, exist_ok=True)
    with open(os.path.join(fake_kagglehub_dir, "exceptions.py"), "w") as f:
      f.write(
          "class UnauthenticatedError(Exception):\n  pass\n"
          "class CredentialError(Exception):\n  pass\n"
          "class KaggleApiHTTPError(Exception):\n  response = None\n"
          "class ColabHTTPError(Exception):\n  response = None\n"
      )
    with open(os.path.join(fake_kagglehub_dir, "__init__.py"), "w") as f:
      f.write(
          "import os\n"
          "from kagglesdk.kaggle_env import get_web_endpoint\n"
          "from kagglehub import exceptions\n"
          "IMPORTED_WEB_ENDPOINT = get_web_endpoint\n"
          "def login():\n  raise AssertionError('Unexpected login()')\n"
          "def model_download(handle: str) -> str:\n"
          "  return os.path.join(\n"
          "      os.environ['KAGGLEHUB_CACHE'], handle.rsplit('/', 1)[-1]\n"
          "  )\n"
      )
    importlib.invalidate_caches()

    fake_pkg = types.ModuleType("kagglesdk")
    fake_pkg.__path__ = []
    fake_kaggle_env = types.ModuleType("kagglesdk.kaggle_env")
    api_endpoint = lambda: "https://www.kaggle.com"
    setattr(fake_kaggle_env, "get_endpoint", api_endpoint)
    setattr(fake_pkg, "kaggle_env", fake_kaggle_env)

    saved_kagglehub = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "kagglehub" or name.startswith("kagglehub.")
    }
    try:
      for name in saved_kagglehub:
        sys.modules.pop(name, None)
      with (
          mock.patch.dict(
              sys.modules,
              {"kagglesdk": fake_pkg, "kagglesdk.kaggle_env": fake_kaggle_env},
          ),
          mock.patch.object(sys, "path", [fake_site_dir, *sys.path]),
          mock.patch.dict(os.environ, {}, clear=True),
      ):
        result = oss_utils.kaggle_pipeline("google/gemma/flax/2b", download_dir)
        imported_kagglehub = sys.modules["kagglehub"]
        imported_endpoint = getattr(imported_kagglehub, "IMPORTED_WEB_ENDPOINT")
    finally:
      for name in list(sys.modules):
        if name == "kagglehub" or name.startswith("kagglehub."):
          sys.modules.pop(name, None)
      sys.modules.update(saved_kagglehub)

    self.assertEqual(result, os.path.join(download_dir, "2b"))
    self.assertIs(getattr(fake_kaggle_env, "get_web_endpoint"), api_endpoint)
    self.assertIs(imported_endpoint, api_endpoint)

  def test_kaggle_pipeline_preserves_existing_kagglesdk_web_endpoint(self):
    download_dir = self.create_tempdir().full_path
    fake_pkg = types.ModuleType("kagglesdk")
    fake_pkg.__path__ = []
    fake_kaggle_env = types.ModuleType("kagglesdk.kaggle_env")
    api_endpoint = lambda: "https://api.kaggle.com"
    web_endpoint = lambda: "https://www.kaggle.com"
    setattr(fake_kaggle_env, "get_endpoint", api_endpoint)
    setattr(fake_kaggle_env, "get_web_endpoint", web_endpoint)
    setattr(fake_pkg, "kaggle_env", fake_kaggle_env)

    with (
        mock.patch.dict(
            sys.modules,
            {"kagglesdk": fake_pkg, "kagglesdk.kaggle_env": fake_kaggle_env},
        ),
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(kagglehub, "login", autospec=True, spec_set=True),
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            return_value=download_dir,
        ),
    ):
      oss_utils.kaggle_pipeline("google/gemma/flax/2b", download_dir)

    self.assertIs(getattr(fake_kaggle_env, "get_web_endpoint"), web_endpoint)

  def test_kaggle_pipeline_skips_backfill_when_get_endpoint_missing(self):
    download_dir = self.create_tempdir().full_path
    fake_pkg = types.ModuleType("kagglesdk")
    fake_pkg.__path__ = []
    fake_kaggle_env = types.ModuleType("kagglesdk.kaggle_env")
    setattr(fake_pkg, "kaggle_env", fake_kaggle_env)

    with (
        mock.patch.dict(
            sys.modules,
            {"kagglesdk": fake_pkg, "kagglesdk.kaggle_env": fake_kaggle_env},
        ),
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(kagglehub, "login", autospec=True, spec_set=True),
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            return_value=download_dir,
        ),
    ):
      result = oss_utils.kaggle_pipeline("google/gemma/flax/2b", download_dir)

    self.assertEqual(result, download_dir)
    self.assertFalse(hasattr(fake_kaggle_env, "get_web_endpoint"))

  def test_kaggle_pipeline_tolerates_missing_kagglesdk_module(self):
    download_dir = self.create_tempdir().full_path
    with (
        mock.patch.dict(
            sys.modules, {"kagglesdk": None, "kagglesdk.kaggle_env": None}
        ),
        mock.patch.dict(os.environ, {}, clear=True),
        mock.patch.object(kagglehub, "login", autospec=True, spec_set=True),
        mock.patch.object(
            kagglehub,
            "model_download",
            autospec=True,
            spec_set=True,
            return_value=download_dir,
        ) as mock_download,
    ):
      result = oss_utils.kaggle_pipeline("google/gemma/flax/2b", download_dir)

    self.assertEqual(result, download_dir)
    mock_download.assert_called_once_with("google/gemma/flax/2b")


if __name__ == "__main__":
  absltest.main()
