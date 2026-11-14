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

from collections.abc import Mapping
import contextlib
import os
from unittest import mock

from absl.testing import absltest
import grain
from tunix.experimental.examples.deepswe_dist import prewarm
from tunix.experimental.examples.deepswe_dist import sandbox_utils

_JOB_ID = "tsbao-mlperf-35b-orch"


class PrewarmSandboxesTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(os.environ, {"ORCHESTRATOR_ID": _JOB_ID})
    )
    self.init_global_fleet = self.enter_context(
        mock.patch.object(sandbox_utils, "init_global_fleet", autospec=True)
    )
    self.prewarm_iterator_cls = self.enter_context(
        mock.patch.object(
            sandbox_utils, "PrewarmDatasetIterator", autospec=True
        )
    )
    self.teardown_global_fleet = self.enter_context(
        mock.patch.object(sandbox_utils, "teardown_global_fleet", autospec=True)
    )
    self.cleanup_calls: list[str] = []
    self.prewarm_iterator_cls.return_value.close.side_effect = (
        lambda: self.cleanup_calls.append("close")
    )
    self.teardown_global_fleet.side_effect = lambda: self.cleanup_calls.append(
        "teardown"
    )
    self.prompt_items: grain.MapDataset[dict[str, object]] = (
        grain.MapDataset.source([{
            "prompt": "p0",
            "metadata": {"env_config": {"entry": {"docker_image": "img_a"}}},
        }])
    )

  def _prewarm_sandboxes(
      self,
      env_kwargs: Mapping[str, object] | None = None,
  ) -> contextlib.AbstractContextManager[
      grain.MapDataset[dict[str, object]] | sandbox_utils.PrewarmDatasetIterator
  ]:
    resolved_env = (
        {"use_agent_sandbox": True, "scaffold": "openhands"}
        if env_kwargs is None
        else env_kwargs
    )
    return prewarm.prewarm_sandboxes(
        self.prompt_items,
        batch_size=16,
        num_generations=8,
        max_staleness=1,
        env_kwargs=resolved_env,
        max_concurrency=1024,
        max_warmpool_replicas=4,
    )

  def test_yields_prompt_items_directly_when_agent_sandbox_disabled(self):
    del os.environ["ORCHESTRATOR_ID"]
    with self._prewarm_sandboxes(
        env_kwargs={"use_agent_sandbox": False, "scaffold": "openhands"}
    ) as items:
      self.assertIs(items, self.prompt_items)

    self.init_global_fleet.assert_not_called()
    self.prewarm_iterator_cls.assert_not_called()
    self.teardown_global_fleet.assert_not_called()

  def test_yields_prewarm_iterator_then_closes_it_and_tears_down_fleet(self):
    with self._prewarm_sandboxes() as items:
      self.assertIs(items, self.prewarm_iterator_cls.return_value)
      self.assertEqual(self.cleanup_calls, [])

    self.init_global_fleet.assert_called_once_with(
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
    self.prewarm_iterator_cls.assert_called_once_with(
        self.prompt_items,
        fleet=self.init_global_fleet.return_value,
        num_generations=8,
        batch_size=16,
        max_warmpool_replicas=4,
        max_staleness=2,
        unwarm_on_exhaustion=True,
        scaffold="openhands",
    )
    self.assertEqual(self.cleanup_calls, ["close", "teardown"])

  def test_closes_iterator_then_tears_down_fleet_when_body_raises(self):
    with self.assertRaisesRegex(RuntimeError, "run failed"):
      with self._prewarm_sandboxes():
        raise RuntimeError("run failed")

    self.assertEqual(self.cleanup_calls, ["close", "teardown"])

  def test_tears_down_fleet_when_iterator_init_raises(self):
    self.prewarm_iterator_cls.side_effect = RuntimeError("warm failed")

    with self.assertRaisesRegex(RuntimeError, "warm failed"):
      with self._prewarm_sandboxes():
        self.fail("The body must not run.")

    self.assertEqual(self.cleanup_calls, ["teardown"])

  def test_raises_without_orchestrator_id(self):
    del os.environ["ORCHESTRATOR_ID"]

    with self.assertRaisesRegex(KeyError, "ORCHESTRATOR_ID"):
      with self._prewarm_sandboxes():
        self.fail("The body must not run.")

    self.init_global_fleet.assert_not_called()
    self.teardown_global_fleet.assert_not_called()

  def test_rejects_invalid_env_kwargs_or_batch_size(self):
    with self.assertRaisesRegex(TypeError, "use_agent_sandbox"):
      with self._prewarm_sandboxes(
          env_kwargs={"use_agent_sandbox": 1, "scaffold": "openhands"}
      ):
        pass
    with self.assertRaisesRegex(ValueError, "scaffold"):
      with self._prewarm_sandboxes(
          env_kwargs={"use_agent_sandbox": True, "scaffold": "  "}
      ):
        pass
    with self.assertRaisesRegex(ValueError, "batch_size"):
      with prewarm.prewarm_sandboxes(
          self.prompt_items,
          batch_size=0,
          num_generations=8,
          max_staleness=1,
          env_kwargs={"use_agent_sandbox": False, "scaffold": "openhands"},
          max_concurrency=1024,
          max_warmpool_replicas=4,
      ):
        pass


if __name__ == "__main__":
  absltest.main()
