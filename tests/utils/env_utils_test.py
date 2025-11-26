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

"""Tests for env_utils."""

import sys
from unittest import mock

import fsspec

from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
from GOOGLE_INTERNAL_PACKAGE_PATH.testing.pybase import googletest
from tunix.utils import env_utils


class EnvUtilsTest(googletest.TestCase):

  @mock.patch.object(gfile, 'Open', autospec=True)
  def test_fs_open_gfile(self, mock_gfile_open):
    env_utils.fs_open('test_file')
    mock_gfile_open.assert_called_once_with('test_file', 'rb')

  @mock.patch.object(fsspec, 'open', autospec=True)
  def test_fs_open_fsspec(self, mock_fsspec_open):
    with mock.patch.dict(sys.modules, {'GOOGLE_INTERNAL_PACKAGE_PATH.pyglib.gfile': None}):
      env_utils.fs_open('test_file')
      mock_fsspec_open.assert_called_once_with('test_file', 'rb')


if __name__ == '__main__':
  googletest.main()
