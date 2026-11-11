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

"""Sticky routing of requests to workers."""

import hashlib


def stable_route_hash(route_key: int | str) -> int:
  """Maps a sticky routing key to a process-stable non-negative integer.

  Builtin `hash()` salts `str` per process, so routing by it would not survive a
  restart or agree across hosts.

  Args:
    route_key: The sticky routing key. Non-negative integers are explicit shard
      indices and are returned as is; strings are hashed with BLAKE2b.

  Returns:
    A non-negative integer suitable for `% num_shards` bucketing.

  Raises:
    ValueError: If `route_key` is a negative integer or a bool.
  """
  if isinstance(route_key, bool):
    raise ValueError(f"route_key must be an int or a str, got {route_key}.")
  if isinstance(route_key, int):
    if route_key < 0:
      raise ValueError(
          "An integer route_key is an explicit shard index and must be"
          f" non-negative, got {route_key}."
      )
    return route_key
  digest = hashlib.blake2b(route_key.encode("utf-8"), digest_size=8)
  return int.from_bytes(digest.digest(), "big")
