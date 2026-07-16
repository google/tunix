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

"""Regression: the standard GrpoLearner logs both pg-loss reductions.

`grpo_loss_fn` produces aux `reduced_pg_loss` (mean-of-means) and
`unreduced_pg_loss` (global sum(S)/sum(d)), but the standard GrpoLearner used by
`tunix.cli.grpo_main` previously registered only `kl` and `pg_clipfrac` for
logging, so those two curves never showed up in standard (non-agentic) GRPO
runs. This pins the registration and keeps its reducers consistent with the
agentic learner.

The test reads the source files directly (AST) so it needs no heavy imports.
"""

import ast
import os

from absl.testing import absltest

_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..')
)
_STANDARD = 'tunix/rl/grpo/grpo_learner.py'
_AGENTIC = 'tunix/rl/agentic/agentic_grpo_learner.py'


def _registered_metrics(rel_path):
  """{metric_name: reducer_source} from `with_rl_metrics_to_log({...})` calls."""
  with open(os.path.join(_ROOT, rel_path), encoding='utf-8') as f:
    src = f.read()
  found = {}
  for node in ast.walk(ast.parse(src)):
    if (
        isinstance(node, ast.Call)
        and getattr(node.func, 'attr', '') == 'with_rl_metrics_to_log'
        and node.args
        and isinstance(node.args[0], ast.Dict)
    ):
      for key, val in zip(node.args[0].keys, node.args[0].values):
        if isinstance(key, ast.Constant):
          found[key.value] = ast.unparse(val)
  return found


class GrpoPgLossMetricsTest(absltest.TestCase):

  def test_standard_learner_registers_both_pg_loss_reductions(self):
    reg = _registered_metrics(_STANDARD)
    self.assertIn('reduced_pg_loss', reg)
    self.assertIn('unreduced_pg_loss', reg)
    self.assertEqual(reg['reduced_pg_loss'], 'common.mean_of_means')
    self.assertEqual(reg['unreduced_pg_loss'], 'common.global_weighted_mean')

  def test_reducers_match_agentic_learner(self):
    std = _registered_metrics(_STANDARD)
    agentic = _registered_metrics(_AGENTIC)
    for metric in ('reduced_pg_loss', 'unreduced_pg_loss'):
      self.assertEqual(
          std[metric],
          agentic[metric],
          f'{metric} reducer differs between standard and agentic learners',
      )


if __name__ == '__main__':
  absltest.main()
