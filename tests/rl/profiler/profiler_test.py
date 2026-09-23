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

"""Tests for RL Profiler steps bounding mapping."""

from absl.testing import absltest
from tunix.common import configs
from tunix.rl.profiler import profiler


class RLProfilerTest(absltest.TestCase):

  def test_parse_steps(self):
    steps_str = "trainer:5,10;sampler:2"
    parsed = profiler._parse_steps(steps_str)
    self.assertEqual(parsed, {"trainer": [5, 10], "sampler": [2]})

  def test_should_profile(self):
    config = configs.RLProfileConfig(
        profile_steps="trainer:5,10;sampler:2", output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    self.assertTrue(p.should_profile(5, "trainer"))
    self.assertTrue(p.should_profile(10, "trainer"))
    self.assertFalse(p.should_profile(2, "trainer"))

    self.assertTrue(p.should_profile(2, "sampler"))
    self.assertFalse(p.should_profile(5, "sampler"))


if __name__ == "__main__":
  absltest.main()
