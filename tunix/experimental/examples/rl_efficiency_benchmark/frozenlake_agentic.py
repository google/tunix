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

"""FrozenLake agentic entry point with tracing and matched input data."""

from absl import app
from tunix.cli import grpo_main
from tunix.experimental.examples.rl_efficiency_benchmark import runtime


def create_dataset(size=10000, seed=42, limit=None, **unused_kwargs):
  """Shares exact ordered environment configurations with the dist run."""
  import grain
  from tunix.experimental.examples.frozenlake_dist import frozenlake

  rows = frozenlake.create_dataset(
      size=size, seed=seed, shuffle_seed=seed, limit=limit
  )
  runtime.record_dataset(rows)
  return grain.MapDataset.source([{**row, "prompts": ""} for row in rows])


if __name__ == "__main__":
  runtime.install("agentic")
  app.run(grpo_main.main)
