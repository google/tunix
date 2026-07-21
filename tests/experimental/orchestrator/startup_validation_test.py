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

"""Tests for orchestrator startup validation."""

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_registry
from tunix.experimental.worker import abstract_worker


class _DescribedWorker(abstract_worker.Worker):
  """A worker with a fully specified WorkerInfo, for validation tests."""

  def __init__(self, worker_id, roles, resources=None):
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id,
        roles=frozenset(roles),
        resources=dict(resources or {}),
    )

  def initialize(self) -> None:
    pass

  def compile(self, shape_config) -> None:
    del shape_config

  def start(self) -> None:
    pass

  def stop(self) -> None:
    pass

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state="READY")

  def info(self) -> datatypes.WorkerInfo:
    return self._info


_SHAPE = datatypes.ShapeConfig(
    max_prompt_length=64, max_response_tokens=128, micro_batch_sizes=[8]
)


def _registry(*workers) -> worker_registry.WorkerRegistry:
  registry = worker_registry.WorkerRegistry()
  for worker in workers:
    registry.register(worker)
  return registry


class StartupValidationTest(absltest.TestCase):

  def test_valid_configuration_passes(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"fsdp_size": 4}),
        _DescribedWorker("i0", {"inference"}),
        _DescribedWorker("r0", {"rollout"}),
    )
    startup_validation.validate_startup(
        registry, _SHAPE, group_size=2, global_batch_size=8
    )

  def test_no_workers_raises(self):
    with self.assertRaises(startup_validation.StartupValidationError):
      startup_validation.validate_startup(
          worker_registry.WorkerRegistry(),
          _SHAPE,
          group_size=2,
          global_batch_size=8,
      )

  def test_multi_process_worker_rejected(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"process_count": 2})
    )
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          registry, _SHAPE, group_size=2, global_batch_size=8
      )
    self.assertTrue(any("process_count" in e for e in ctx.exception.errors))

  def test_multi_process_allowed_when_guard_disabled(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"process_count": 2})
    )
    startup_validation.validate_startup(
        registry,
        _SHAPE,
        group_size=2,
        global_batch_size=8,
        require_single_process=False,
    )

  def test_tokenizer_hash_mismatch_rejected(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"tokenizer_hash": "aaa"}),
        _DescribedWorker("i0", {"inference"}, {"tokenizer_hash": "bbb"}),
    )
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          registry, _SHAPE, group_size=2, global_batch_size=8
      )
    self.assertTrue(any("tokenizer" in e for e in ctx.exception.errors))

  def test_batch_not_divisible_by_group_rejected(self):
    registry = _registry(_DescribedWorker("t0", {"trainer"}))
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          registry, _SHAPE, group_size=3, global_batch_size=8
      )
    self.assertTrue(any("divisible" in e for e in ctx.exception.errors))

  def test_micro_batch_not_divisible_by_fsdp_rejected(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"fsdp_size": 4})
    )
    shape = datatypes.ShapeConfig(
        max_prompt_length=64, max_response_tokens=128, micro_batch_sizes=[6]
    )
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          registry, shape, group_size=2, global_batch_size=8
      )
    self.assertTrue(any("fsdp_size" in e for e in ctx.exception.errors))

  def test_multiple_failures_are_aggregated(self):
    registry = _registry(
        _DescribedWorker("t0", {"trainer"}, {"process_count": 2}),
    )
    with self.assertRaises(startup_validation.StartupValidationError) as ctx:
      startup_validation.validate_startup(
          registry, _SHAPE, group_size=3, global_batch_size=8
      )
    # Both the process_count guard and the batch/group lattice failed.
    self.assertGreaterEqual(len(ctx.exception.errors), 2)


if __name__ == "__main__":
  absltest.main()
