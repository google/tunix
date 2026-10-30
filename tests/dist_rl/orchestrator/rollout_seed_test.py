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

"""Stable per-request seeds for grouped distributed rollouts."""

import unittest

from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import worker_registry


class RolloutSeedTest(unittest.TestCase):

  def setUp(self):
    self.engine = distributed_rl_engine.DistributedRLEngine(
        worker_registry.WorkerRegistry()
    )

  def _requests(self, prompts, seed=42):
    return self.engine._build_rollout_requests(
        prompts,
        num_generations=4,
        generation_args=datatypes.GenerationArgs(seed=seed),
    )

  def test_siblings_are_distinct_and_independent_of_dispatch_order(self):
    prompts = [dict(prompt="same", prompt_id=p) for p in ("first", "second")]
    first = self._requests(prompts)
    reordered = self._requests(list(reversed(prompts)))
    seeds = {
        (r.prompt_id, r.group_index): r.generation_kwargs["seed"] for r in first
    }
    self.assertEqual(
        seeds,
        {
            (r.prompt_id, r.group_index): r.generation_kwargs["seed"]
            for r in reordered
        },
    )
    self.assertEqual(len(set(seeds.values())), 8)
    self.assertTrue(all(0 <= seed < 2**32 for seed in seeds.values()))

  def test_base_seed_and_prompt_override_change_request_seeds(self):
    prompt = dict(prompt="question", prompt_id="p")
    original = self._requests([prompt])
    changed = self._requests([prompt], seed=43)
    overridden = self._requests([{**prompt, "generation_kwargs": {"seed": 43}}])
    self.assertNotEqual(
        original[0].generation_kwargs["seed"],
        changed[0].generation_kwargs["seed"],
    )
    self.assertEqual(
        [r.generation_kwargs["seed"] for r in changed],
        [r.generation_kwargs["seed"] for r in overridden],
    )

  def test_none_keeps_the_unseeded_path(self):
    requests = self._requests([dict(prompt="q", prompt_id="p")], seed=None)
    self.assertTrue(all("seed" not in r.generation_kwargs for r in requests))

  def test_preformed_request_inherits_seed_once_and_keeps_explicit_seed(self):
    inherited = datatypes.RolloutRequest(
        prompt="q", prompt_id="p", group_index=2
    )
    explicit = datatypes.RolloutRequest(
        prompt="q", prompt_id="p", group_index=3, generation_kwargs={"seed": 17}
    )
    self._requests([inherited, explicit])
    derived = inherited.generation_kwargs["seed"]
    self._requests([inherited, explicit])
    self.assertEqual(inherited.generation_kwargs["seed"], derived)
    self.assertEqual(explicit.generation_kwargs["seed"], 17)


if __name__ == "__main__":
  unittest.main()
