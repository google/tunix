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

"""Environment utils."""

import flax


def setup_sharding_environment():
  """Sets up the sharding environment."""
  if hasattr(flax.config, 'flax_always_shard_variable'):
    flax.config.update('flax_always_shard_variable', False)


def is_internal_env():
  """Checks if the code is running within the internal environment."""
  try:
    from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile  # noqa: F401

    return True
  except ImportError:
    return False
