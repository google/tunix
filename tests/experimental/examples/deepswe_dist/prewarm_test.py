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

"""Unit tests for tunix.experimental.examples.deepswe_dist.prewarm."""

from __future__ import annotations

import math
import os
import unittest
from unittest import mock

from absl.testing import absltest

# `r2egym` is an optional dependency in OSS (`swe_agent` requires it at import
# time); match `deepswe_test.py` so OSS CPU CI skips cleanly when absent.
try:
  import r2egym  # pylint: disable=unused-import,g-import-not-at-top
except ImportError as exc:
  raise unittest.SkipTest("r2egym is not installed") from exc

# pylint: disable=g-import-not-at-top
import grain
from tunix.dist_rl.examples.deepswe_dist import deepswe
from tunix.dist_rl.launch.common import resolve_config
from tunix.experimental.examples.deepswe_dist import prewarm
from tunix.experimental.examples.deepswe_dist import sandbox_utils

# pylint: enable=g-import-not-at-top

_JOB_ID = "tsbao-mlperf-35b-orch"
_REAL_PREWARM_ITER = sandbox_utils.PrewarmDatasetIterator


class PrewarmSandboxesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {"ORCHESTRATOR_ID": _JOB_ID})
    )
    self.init_fleet = self.enter_context(
        mock.patch.object(sandbox_utils, "init_global_fleet", autospec=True)
    )
    self.iter_cls = self.enter_context(
        mock.patch.object(
            sandbox_utils, "PrewarmDatasetIterator", autospec=True
        )
    )
    self.teardown_fleet = self.enter_context(
        mock.patch.object(sandbox_utils, "teardown_global_fleet", autospec=True)
    )
    self.cleanup_calls: list[str] = []
    self.iter_cls.return_value.close.side_effect = (
        lambda: self.cleanup_calls.append("close")
    )
    self.teardown_fleet.side_effect = lambda: self.cleanup_calls.append(
        "teardown"
    )
    self.prompt_items: grain.MapDataset[dict[str, object]] = (
        grain.MapDataset.source([{"prompt": "p0"}])
    )

  def _wrap(self, *, use_agent_sandbox: bool = True):
    return prewarm.prewarm_sandboxes(
        self.prompt_items,
        batch_size=16,
        num_generations=8,
        max_staleness=1,
        env_kwargs={
            "use_agent_sandbox": use_agent_sandbox,
            "scaffold": "openhands",
        },
        max_concurrency=1024,
        max_warmpool_replicas=4,
    )

  def test_data_module_loads_deepswe_records_and_prewarms_from_config(self):
    self.assertIs(prewarm.create_dataset, deepswe.create_dataset)
    hp = resolve_config.initialize([
        "batch_size=16",
        "agentic_grpo_config.num_generations=8",
        "agentic_grpo_config.off_policy_steps=1",
    ])
    with mock.patch.object(
        prewarm, "prewarm_sandboxes", autospec=True
    ) as prewarm_fn:
      self.assertIs(
          prewarm.prompt_items_context(hp, self.prompt_items),
          prewarm_fn.return_value,
      )
    prewarm_fn.assert_called_once_with(
        self.prompt_items,
        batch_size=16,
        num_generations=8,
        max_staleness=1,
        env_kwargs=hp.config_mapping("env_kwargs"),
        max_concurrency=1024,
        max_warmpool_replicas=8,
    )

  def test_noop_when_agent_sandbox_disabled(self):
    del os.environ["ORCHESTRATOR_ID"]
    with self._wrap(use_agent_sandbox=False) as items:
      self.assertIs(items, self.prompt_items)
    self.init_fleet.assert_not_called()

  def test_lifecycle_and_error_unwinding(self):
    with self._wrap() as items:
      self.assertIs(items, self.iter_cls.return_value)
      self.assertEqual(self.cleanup_calls, [])

    self.init_fleet.assert_called_once_with(
        tasks=None,
        max_concurrency=1024,
        num_generations=8,
        batch_size=16,
        max_warmpool_replicas=4,
        scaffold="openhands",
        job_id=_JOB_ID,
        warm_on_init=False,
        teardown_on_exit=False,
    )
    self.iter_cls.assert_called_once_with(
        self.prompt_items,
        fleet=self.init_fleet.return_value,
        num_generations=8,
        batch_size=16,
        max_warmpool_replicas=4,
        max_staleness=3,
        unwarm_on_exhaustion=False,
        scaffold="openhands",
    )
    self.assertEqual(self.cleanup_calls, ["close", "teardown"])

    self.cleanup_calls.clear()
    with self.assertRaisesRegex(RuntimeError, "boom"), self._wrap():
      raise RuntimeError("boom")
    self.assertEqual(self.cleanup_calls, ["close", "teardown"])

    self.cleanup_calls.clear()
    self.iter_cls.side_effect = RuntimeError("iterator init failed")
    with self.assertRaisesRegex(RuntimeError, "iterator init failed"):
      with self._wrap():
        pass
    self.assertEqual(self.cleanup_calls, ["teardown"])

    del os.environ["ORCHESTRATOR_ID"]
    with self.assertRaisesRegex(KeyError, "ORCHESTRATOR_ID"), self._wrap():
      pass

  def test_retains_in_flight_batches_until_close(self):
    fleet = mock.create_autospec(
        sandbox_utils.SandboxFleetLike, instance=True, tasks=[], plan_=None
    )
    self.init_fleet.return_value = fleet
    self.iter_cls.side_effect = _REAL_PREWARM_ITER
    for batch_size, max_staleness in ((1, 0), (1, 1), (2, 1), (16, 1)):
      retained = 2 * max_staleness + math.ceil(2 / batch_size)
      num_batches = retained + 4
      items: grain.MapDataset[dict[str, object]] = grain.MapDataset.source([
          {
              "prompt": f"p{i}",
              "metadata": {
                  "env_config": {
                      "entry": {"docker_image": f"img_{i // batch_size}"}
                  }
              },
          }
          for i in range(num_batches * batch_size)
      ])
      with prewarm.prewarm_sandboxes(
          items,
          batch_size=batch_size,
          num_generations=2,
          max_staleness=max_staleness,
          env_kwargs={"use_agent_sandbox": True, "scaffold": "r2egym"},
          max_concurrency=64,
          max_warmpool_replicas=4,
      ) as prewarm_iter:
        assert isinstance(prewarm_iter, _REAL_PREWARM_ITER)
        it = iter(prewarm_iter)
        # With `trajectory_completion`, `StandardRLProgram` can read
        # `2 * max_staleness + 1` batches plus 2 items past batch 0 while
        # batch 0 is still rolling out.
        in_flight_reads = (2 * max_staleness + 2) * batch_size + 2
        for _ in range(in_flight_reads):
          next(it)
        self.assertNotIn("img_0", prewarm_iter.unwarm_calls)
        evicting_reads = (retained + 2) * batch_size + 1
        for _ in range(evicting_reads - in_flight_reads):
          next(it)
        self.assertIn("img_0", prewarm_iter.unwarm_calls)

        last_image = f"img_{num_batches - 1}"
        self.assertLen(list(it), num_batches * batch_size - evicting_reads)
        self.assertNotIn(last_image, prewarm_iter.unwarm_calls)
      self.assertIn(last_image, prewarm_iter.unwarm_calls)


if __name__ == "__main__":
  absltest.main()
