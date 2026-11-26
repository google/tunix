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

from absl.testing import absltest
from absl.testing import parameterized
from tunix.common import routing


class StableRouteHashTest(parameterized.TestCase):

  def test_passes_non_negative_ints_through(self):
    self.assertEqual(routing.stable_route_hash(0), 0)
    self.assertEqual(routing.stable_route_hash(7), 7)

  def test_hashes_strs_stably(self):
    # Fixed values: the hash must not change across processes or releases.
    self.assertEqual(routing.stable_route_hash("prompt_7"), 7682816066691324195)
    self.assertEqual(
        routing.stable_route_hash("route_key_abc"), 8273245243606849686
    )

  def test_separates_different_strs(self):
    self.assertNotEqual(
        routing.stable_route_hash("g3_p1"), routing.stable_route_hash("g3_p2")
    )

  @parameterized.parameters(-1, True, False)
  def test_rejects_negative_ints_and_bools(self, route_key):
    with self.assertRaises(ValueError):
      routing.stable_route_hash(route_key)


if __name__ == "__main__":
  absltest.main()
