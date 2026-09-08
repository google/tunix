# Copyright 2026 Google LLC
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

"""Unit tests for the weight sync coordinator.

Workers and the transport handler are fakes, so this needs no TPU. Under test
is the round protocol, not byte movement: ordering across workers, concurrency,
metadata collected exactly once, the staging/publish contract, and every
failure branch ending in the state it claims.

The fakes carry a tiny numeric payload (plain lists) end to end: the source
stages a per-round pattern, the fake handler "transfers" it into each
destination's host staging, `weight_sync` copies it to the device staging copy,
and `post_weight_sync` publishes it. Tests then assert on the serving copy,
which is the only observable that matters.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
import unittest
from typing import Any, Mapping, Optional, Sequence

from absl.testing import absltest

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry

try:
  from tunix.experimental.weight_sync import weight_sync
  from tunix.experimental.weight_sync import weight_sync_coordinator
except ImportError:
  raise unittest.SkipTest("tpu_raiden is required")


WorkUnitId = weight_sync.WorkUnitId
RoundState = weight_sync_coordinator.RoundState
WeightSyncError = weight_sync_coordinator.WeightSyncError
PhaseTimeouts = weight_sync_coordinator.PhaseTimeouts

FAST_TIMEOUTS = PhaseTimeouts(
    bind=5, metadata=5, source_prepare=5, pre=5, transfer=5, h2d=5, post=5,
    abort=5, status=5,
)


class Wire:
  """The fake data plane: whatever the source staged last."""

  def __init__(self):
    self.pattern: Optional[list[float]] = None


class FakeSource:
  """Trainer side. Synchronizer state is per round: fresh ports, fresh data."""

  def __init__(
      self,
      worker_id: str,
      wire: Wire,
      log: list[str],
      hosts: int = 1,
      variables: Sequence[weight_sync.TensorMetadata] = (),
      prepare_delay: float = 0.0,
  ):
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset({datatypes.Role.ACTOR.value})
    )
    self._wire = wire
    self._log = log
    self._hosts = hosts
    self._variables = tuple(variables)
    self.release_gate = None
    self.release_entered = None
    self._prepare_delay = prepare_delay
    self.port = 20000
    self.prepare_calls = 0
    self.release_calls = 0

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  async def prepare_weight_sync(self, sync_request: Any = None, **kwargs):
    del kwargs
    self.prepare_calls += 1
    if self._prepare_delay:
      await asyncio.sleep(self._prepare_delay)
    self.port += 1  # rebind: a snapshot from last round would be stale
    self._wire.pattern = [
        sync_request.policy_version * 1000.0 + i for i in range(4)
    ]
    self._log.append(f"{self._info.worker_id}:prepare")
    return [
        weight_sync.WorkUnitMetadata(
            unit=WorkUnitId(
                job_name=self._info.worker_id, job_replica_id=str(host)
            ),
            shards=(f"10.0.0.1:{self.port + host}",),
            control_plane_rpc_address=f"10.0.0.1:{self.port + host + 500}",
            global_shape=(4,),
            mesh_shape=(1,),
            layout=(0,),
            item_size=4,
            variables=self._variables,
        )
        for host in range(self._hosts)
    ]

  async def release_weight_sync(self, sync_request: Any = None, **kwargs):
    del sync_request, kwargs
    self._log.append(f"{self._info.worker_id}:release-start")
    if self.release_gate is not None:
      self.release_entered.set()
      await self.release_gate.wait()
    self.release_calls += 1
    self._log.append(f"{self._info.worker_id}:release")


class FakeDestination:
  """Sampler side, with real staging/serving double-buffer semantics and a
  real `WorkerRoundTracker` behind every phase.

  `fail_on` makes the named phase raise AFTER its partial work, which is the
  dangerous variant: a pre that has already gated admission, an abort that has
  not yet restored serving. `crash_after_publish` raises once between
  publishing and recording -- the crash-consistency window the tracker's
  admit/complete split exists for. `raise_after_complete_once` completes the
  phase fully and then raises -- a lost reply.
  """

  def __init__(
      self,
      worker_id: str,
      log: list[str],
      fail_on: Optional[str] = None,
      variables: Sequence[weight_sync.TensorMetadata] = (),
      global_shape: tuple[int, ...] = (4,),
      item_size: int = 4,
      fail_persistently: bool = False,
      pre_gate: Optional[asyncio.Event] = None,
      pre_await: Optional[asyncio.Event] = None,
      delay_after_phase: Optional[str] = None,
      round_report_override: Optional[Mapping[str, Any]] = None,
      round_report_sequence: Optional[list] = None,
      crash_after_publish: bool = False,
      raise_after_complete_once: Optional[str] = None,
      status_unreachable: bool = False,
  ):
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset({datatypes.Role.ROLLOUT.value})
    )
    self._log = log
    self._variables = tuple(variables)
    self._global_shape = global_shape
    self._item_size = item_size
    self._fail_on = fail_on
    self._fail_persistently = fail_persistently
    self._failed_once: set[str] = set()
    self._pre_gate = pre_gate
    self._pre_await = pre_await
    self._delay_after_phase = delay_after_phase
    self._round_report_override = round_report_override
    # Consumed one per status query, last entry repeats: models a worker
    # whose answer changes between the coordinator's polls (e.g. a post
    # still running at the first poll, committed by the second).
    self._round_report_sequence = list(round_report_sequence or [])
    self._crash_after_publish = crash_after_publish
    self._crashed_once = False
    self._raise_after_complete_once = raise_after_complete_once
    self._raised_after_complete: set[str] = set()
    self._status_unreachable = status_unreachable

    self.unit = WorkUnitId(job_name=worker_id, job_replica_id="0")
    self.serving: list[float] = [0.0] * 4
    self.staging: Optional[list[float]] = None
    self.host_staging: Optional[list[float]] = None
    self.admitting = True
    self.kv_cache = True
    self.bound = False
    self.port = 0
    self.bind_calls = 0
    self._port_base = 30000
    self.tracker = weight_sync_coordinator.WorkerRoundTracker()

  def info(self) -> datatypes.WorkerInfo:
    return self._info

  def restart(self) -> None:
    """Simulates a worker restart: ports gone, round state gone."""
    self.bound = False
    self._port_base += 100
    self.host_staging = None
    self.staging = None
    self.tracker = weight_sync_coordinator.WorkerRoundTracker()

  def receive(self, pattern: list[float]) -> None:
    """Called by the fake handler: bytes landing in host staging."""
    self.host_staging = list(pattern)

  def _maybe_fail(self, phase: str) -> None:
    if self._fail_on == phase:
      if self._fail_persistently or phase not in self._failed_once:
        self._failed_once.add(phase)
        raise RuntimeError(f"{self._info.worker_id} failed at {phase}")

  def _maybe_raise_after_complete(self, phase: str) -> None:
    if (
        self._raise_after_complete_once == phase
        and phase not in self._raised_after_complete
    ):
      self._raised_after_complete.add(phase)
      raise ConnectionResetError(
          f"{self._info.worker_id}: reply lost after {phase}"
      )

  async def _maybe_delay(self, phase: str) -> None:
    if self._delay_after_phase == phase:
      await asyncio.sleep(60)

  async def bind_weight_sync(self):
    if not self.bound:
      self.bind_calls += 1
      self.port = self._port_base + self.bind_calls
      self.bound = True
    self._log.append(f"{self._info.worker_id}:bind")

  async def get_weight_sync_metadata(self):
    self._log.append(f"{self._info.worker_id}:metadata")
    if self._variables:
      return [
          weight_sync.WorkUnitMetadata(
              unit=self.unit,
              shards=(f"10.0.0.2:{self.port}",),
              control_plane_rpc_address=f"10.0.0.2:{self.port + 500}",
              variables=self._variables,
          )
      ]
    return [
        weight_sync.WorkUnitMetadata(
            unit=self.unit,
            shards=(f"10.0.0.2:{self.port}",),
            control_plane_rpc_address=f"10.0.0.2:{self.port + 500}",
            global_shape=self._global_shape,
            mesh_shape=(1,),
            layout=(0,),
            item_size=self._item_size,
        )
    ]

  async def pre_weight_sync(self, sync_request: Any = None, **kwargs):
    del kwargs
    self._log.append(f"{self._info.worker_id}:pre")
    if self._pre_gate is not None:
      self._pre_gate.set()
    if not self.tracker.admit(sync_request, "prepared"):
      return  # duplicate delivery of a phase this round already passed
    # Quiesce first, then maybe fail: the dangerous partial state.
    self.admitting = False
    self.kv_cache = False
    self.last_request = sync_request
    self._maybe_fail("pre")
    if self._pre_await is not None:
      await asyncio.wait_for(self._pre_await.wait(), 5)
    self.tracker.complete(sync_request, "prepared")
    await self._maybe_delay("pre")

  async def weight_sync(self, sync_request: Any = None, **kwargs):
    del kwargs
    self._log.append(f"{self._info.worker_id}:sync")
    if not self.tracker.admit(sync_request, "h2d_done"):
      return
    self._maybe_fail("weight_sync")
    assert self.host_staging is not None, "weight_sync before bytes arrived"
    # Staging copy only. The serving copy must survive an abort after this.
    self.staging = list(self.host_staging)
    self.tracker.complete(sync_request, "h2d_done")

  async def post_weight_sync(self, sync_request: Any = None, **kwargs):
    del kwargs
    self._log.append(f"{self._info.worker_id}:post")
    if not self.tracker.admit(sync_request, "committed"):
      return  # already committed this round: duplicate, no-op
    self._maybe_fail("post")
    if self.staging is not None:
      self.serving = self.staging  # atomic publish
      self.staging = None
      self.host_staging = None
    if self._crash_after_publish and not self._crashed_once:
      # Published but crashed before recording: the tracker still says
      # h2d_done, so a retry re-enters here and must converge.
      self._crashed_once = True
      raise RuntimeError(
          f"{self._info.worker_id}: crashed after publish, before recording"
      )
    self.kv_cache = True
    self.admitting = True
    self.tracker.complete(sync_request, "committed")
    self._maybe_raise_after_complete("post")
    await self._maybe_delay("post")

  async def abort_weight_sync(self, sync_request: Any = None, **kwargs):
    del kwargs
    self._log.append(f"{self._info.worker_id}:abort")
    self._maybe_fail("abort")
    if not self.tracker.admit(sync_request, "aborted"):
      return
    self.staging = None
    self.host_staging = None
    self.kv_cache = True
    self.admitting = True
    self.tracker.complete(sync_request, "aborted")

  async def get_weight_sync_status(self):
    if self._status_unreachable:
      raise ConnectionResetError(f"{self._info.worker_id}: status unreachable")
    if self._round_report_sequence:
      if len(self._round_report_sequence) > 1:
        return dict(self._round_report_sequence.pop(0))
      return dict(self._round_report_sequence[0])
    if self._round_report_override is not None:
      return dict(self._round_report_override)
    return self.tracker.report()


class FakeHandler(weight_sync.WeightSyncHandler):
  """Records registrations and transfers; moves the wire pattern on transfer."""

  def __init__(self, wire: Wire, log: list[str]):
    self._wire = wire
    self._log = log
    self._destinations: list[FakeDestination] = []
    self.registered: list[weight_sync.WorkUnitMetadata] = []
    self.transfers: list[dict[str, Any]] = []
    self.result_success = True
    self.result_message = ""
    self.raise_on_transfer: Optional[Exception] = None
    self.transfer_delay = 0.0  # blocks the executor thread, like a hung RPC

  def attach(self, *destinations: FakeDestination) -> None:
    self._destinations.extend(destinations)

  def register_work_unit(self, metadata) -> None:
    self.registered.append(metadata)
    self._log.append(f"register:{metadata.unit.job_name}")

  def transfer(
      self, src_units, dst_units, req_id=None, generation=None, **kwargs
  ):
    self._log.append("transfer")
    if self.transfer_delay:
      time.sleep(self.transfer_delay)
    self.transfers.append(
        dict(
            src=list(src_units),
            dst=list(dst_units),
            req_id=req_id,
            generation=generation,
            backend_kwargs=dict(kwargs),
        )
    )
    if self.raise_on_transfer is not None:
      raise self.raise_on_transfer
    if self.result_success:
      assert self._wire.pattern is not None
      for destination in self._destinations:
        if destination.unit in dst_units:
          destination.receive(self._wire.pattern)
    return weight_sync.TransferResult(
        req_id=req_id or "",
        success=self.result_success,
        message=self.result_message,
    )


def expected_pattern(policy_version: int) -> list[float]:
  return [policy_version * 1000.0 + i for i in range(4)]


class CoordinatorTestBase(absltest.TestCase):

  def make(
      self,
      *destinations: FakeDestination,
      sources=None,
      timeouts=None,
  ):
    self.log: list[str] = []
    self.wire = Wire()
    self.handler = FakeHandler(self.wire, self.log)
    self.handler.attach(*destinations)
    self.registry = worker_registry.WorkerRegistry()
    if sources is None:
      sources = [FakeSource("trainer", self.wire, self.log)]
    else:
      for s in sources:
        s._wire = self.wire  # pylint: disable=protected-access
        s._log = self.log  # pylint: disable=protected-access
    self.sources = sources
    for source in sources:
      self.registry.register(source)
    for destination in destinations:
      destination._log = self.log  # pylint: disable=protected-access
      self.registry.register(destination)
    self.coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=self.registry,
        handler=self.handler,
        controller_id="test-controller",
        timeouts=timeouts or FAST_TIMEOUTS,
    )
    return self.coordinator

  def sync(self, policy_version=1, **kwargs):
    return asyncio.run(self.coordinator.sync(policy_version, **kwargs))

  def phases(self, worker_id: str) -> list[str]:
    prefix = f"{worker_id}:"
    return [e[len(prefix):] for e in self.log if e.startswith(prefix)]


class SuccessPathTest(CoordinatorTestBase):

  def test_committed_round_delivers_the_bytes(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    result = self.sync(policy_version=3)

    self.assertTrue(result.success)
    self.assertIs(result.state, RoundState.COMMITTED)
    self.assertEqual(dest.serving, expected_pattern(3))
    self.assertTrue(dest.kv_cache)
    self.assertTrue(dest.admitting)

  def test_public_result_exposes_only_neutral_work_unit_ids(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    result = self.sync()

    self.assertIsInstance(result.source_units[0], weight_sync.WorkUnitId)
    self.assertIsInstance(result.destination_units[0], weight_sync.WorkUnitId)
    self.assertNotIn("Raiden", type(result.source_units[0]).__name__)

  def test_coordinator_passes_no_backend_specific_transfer_options(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync()

    transfer = self.handler.transfers[0]
    self.assertEmpty(transfer["backend_kwargs"])
    self.assertEqual(transfer["generation"], 1)

  def test_phase_order_within_a_round(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync()

    self.assertEqual(
        self.phases("sampler"),
        ["bind", "metadata", "pre", "sync", "post"],
    )
    # The KV cache must be gone before bytes arrive, the device copy after.
    self.assertLess(self.log.index("sampler:pre"), self.log.index("transfer"))
    self.assertLess(self.log.index("transfer"), self.log.index("sampler:sync"))

  def test_registration_costs_no_downtime(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync()

    # Both sides register while the destination is still serving.
    register_indices = [
        i for i, e in enumerate(self.log) if e.startswith("register:")
    ]
    self.assertLen(register_indices, 2)
    self.assertLess(max(register_indices), self.log.index("sampler:pre"))

  def test_two_rounds_change_the_weights_twice(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync(policy_version=1)
    first = list(dest.serving)
    self.sync(policy_version=2)

    self.assertEqual(first, expected_pattern(1))
    self.assertEqual(dest.serving, expected_pattern(2))

  def test_destination_ports_survive_across_rounds(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync(policy_version=1)
    self.sync(policy_version=2)

    self.assertEqual(dest.bind_calls, 1)
    dst_addrs = {
        m.shards[0] for m in self.handler.registered
        if m.unit.job_name == "sampler"
    }
    self.assertLen(dst_addrs, 1)

  def test_source_metadata_collected_exactly_once_per_round(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync()

    self.assertEqual(self.sources[0].prepare_calls, 1)
    # The registered endpoints and the ones carried in the request are the
    # same collection, not two snapshots of a rebinding synchronizer.
    registered = [
        m.shards[0] for m in self.handler.registered
        if m.unit.job_name == "trainer"
    ]
    in_request = [
        m.shards[0] for m in dest.last_request.source_metadata
    ]
    self.assertEqual(registered, in_request)

  def test_source_release_runs_on_success(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync()

    self.assertEqual(self.sources[0].release_calls, 1)

  def test_fresh_req_id_and_uuid_even_for_the_same_policy_version(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    r1 = self.sync(policy_version=5)
    r2 = self.sync(policy_version=5)

    self.assertNotEqual(r1.req_id, r2.req_id)
    self.assertNotEqual(r1.uuid, r2.uuid)
    transfers = self.handler.transfers
    self.assertEqual(
        [t["generation"] for t in transfers], [r1.uuid, r2.uuid]
    )

  def test_destination_phases_run_concurrently(self):
    # Each destination's pre waits for the other's to have started; serial
    # execution deadlocks here, concurrent execution sails through. The
    # events are created inside the running loop (3.9 binds them to one).
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination("s2", [])
    coordinator = self.make(d1, d2)

    async def scenario():
      started_1, started_2 = asyncio.Event(), asyncio.Event()
      d1._pre_gate, d1._pre_await = started_1, started_2  # pylint: disable=protected-access
      d2._pre_gate, d2._pre_await = started_2, started_1  # pylint: disable=protected-access
      return await coordinator.sync(1)

    result = asyncio.run(scenario())

    self.assertTrue(result.success)


class MultiHostAndVariablesTest(CoordinatorTestBase):

  def test_multi_host_source_registers_one_unit_per_host(self):
    dest = FakeDestination("sampler", [])
    wire = Wire()
    source = FakeSource("trainer", wire, [], hosts=3)
    self.make(dest, sources=[source])

    result = self.sync()

    self.assertLen(result.source_units, 3)
    self.assertEqual(
        sorted(u.job_replica_id for u in result.source_units),
        ["0", "1", "2"],
    )
    self.assertLen(self.handler.transfers[0]["src"], 3)
    trainer_regs = [
        m for m in self.handler.registered if m.unit.job_name == "trainer"
    ]
    self.assertLen(trainer_regs, 3)

  def test_variables_manifest_travels_to_registration(self):
    variables = (
        weight_sync.TensorMetadata(
            name="w", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
            item_size=4, layer_idx=0,
        ),
        weight_sync.TensorMetadata(
            name="b", shape=(4,), mesh_shape=(1,), layout=(0,),
            item_size=2, layer_idx=1,
        ),
    )
    # Both sides carry the SAME manifest: the preflight enforces exactly
    # the symmetry the controller's name pairing depends on.
    dest = FakeDestination("sampler", [], variables=variables)
    wire = Wire()
    source = FakeSource("trainer", wire, [], variables=variables)
    self.make(dest, sources=[source])

    self.sync()

    trainer_reg = next(
        m for m in self.handler.registered if m.unit.job_name == "trainer"
    )
    self.assertEqual(
        [(v.name, v.shape, v.item_size) for v in trainer_reg.variables],
        [("w", (8, 4), 4), ("b", (4,), 2)],
    )


class ManifestPreflightTest(CoordinatorTestBase):
  """The controller pairs variables by exact name and silently skips
  mismatches; the preflight turns that silent loss into a loud, zero-cost
  failure BEFORE any destination is quiesced."""

  def test_name_mismatch_fails_before_any_downtime(self):
    variables = (
        weight_sync.TensorMetadata(
            name="w", shape=(4,), mesh_shape=(1,), layout=(0,), item_size=4
        ),
    )
    dest = FakeDestination("sampler", [], variables=variables)
    self.make(dest)  # source registers single-tensor under its data_name

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertTrue(
        any("preflight" in f for f in ctx.exception.result.failures)
    )
    # No quiesce happened, so nothing to roll back and no poison.
    self.assertNotIn("pre", self.phases("sampler"))
    self.assertNotIn("abort", self.phases("sampler"))
    self.assertIsNone(self.coordinator.poisoned)

  def test_failure_log_shows_both_manifests(self):
    """Verifies failure logs sample both manifests on mismatch.

    WeightSyncError renders only its outer message, and the problem list is
    name-sorted, so a naming-convention mismatch shows one side unless the log
    samples both.
    """

    def var(name):
      return weight_sync.TensorMetadata(
          name=name, shape=(4,), mesh_shape=(1,), layout=(0,), item_size=4
      )

    dest = FakeDestination("sampler", [], variables=(var("rollout_w"),))
    source = FakeSource("trainer", Wire(), [], variables=(var("trainer_w"),))
    self.make(dest, sources=[source])

    with self.assertLogs(level="ERROR") as logs:
      with self.assertRaises(WeightSyncError):
        self.sync()

    blob = "\n".join(logs.output)
    self.assertIn("manifest preflight failed", blob)
    self.assertIn("trainer_w", blob)
    self.assertIn("rollout_w", blob)

  def test_shape_mismatch_fails_before_any_downtime(self):
    dest = FakeDestination("sampler", [], global_shape=(8,))
    self.make(dest)  # source's global shape is (4,)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertTrue(
        any("shape" in f for f in ctx.exception.result.failures)
    )
    self.assertNotIn("pre", self.phases("sampler"))
    # Source staging is still released on this exit path.
    self.assertEqual(self.sources[0].release_calls, 1)


class FailurePathTest(CoordinatorTestBase):

  def assert_serving_old_weights(self, dest: FakeDestination):
    self.assertEqual(dest.serving, [0.0] * 4)
    self.assertTrue(dest.kv_cache)
    self.assertTrue(dest.admitting)
    self.assertIsNone(dest.staging)

  def test_failed_transfer_aborts_and_restores_serving(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.result_success = False
    self.handler.result_message = "planner produced no chunks"

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)
    self.assertIn("abort", self.phases("sampler"))
    self.assertNotIn("sync", self.phases("sampler"))
    self.assert_serving_old_weights(dest)

  def test_transfer_raising_still_rolls_back(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.raise_on_transfer = RuntimeError("controller unreachable")

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)
    self.assert_serving_old_weights(dest)

  def test_partial_pre_failure_aborts_every_attempted_destination(self):
    # s2 raises AFTER gating admission: the dangerous partial state. Both
    # destinations were attempted, so both must be rolled back.
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination("s2", [], fail_on="pre", fail_persistently=True)
    self.make(d1, d2)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)
    self.assertIn("abort", self.phases("s1"))
    self.assertIn("abort", self.phases("s2"))
    self.assertTrue(d2.admitting)  # the abort un-stranded it

  def test_h2d_failure_aborts_all_and_serving_survives(self):
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination("s2", [])
    self.make(d1, d2)
    self.sync(policy_version=1)  # a committed baseline
    baseline = list(d1.serving)
    d2._fail_on = "weight_sync"  # pylint: disable=protected-access
    d2._fail_persistently = True  # pylint: disable=protected-access

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=2)

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)
    # d1's H2D succeeded into staging, but staging-only means rollback is
    # still clean: serving never moved.
    self.assertEqual(d1.serving, baseline)
    self.assertEqual(d2.serving, baseline)

  def test_post_partial_failure_is_partially_committed_not_aborted(self):
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination("s2", [], fail_on="post", fail_persistently=True)
    self.make(d1, d2)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=4)

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.PARTIALLY_COMMITTED)
    # Version consistency: d1 serves the new version, d2 the old one.
    self.assertEqual(d1.serving, expected_pattern(4))
    self.assertEqual(d2.serving, [0.0] * 4)
    # Health is a separate axis and must not be papered over: d2 is not
    # "serving the old version", it is DOWN -- post failed before restoring
    # admission and the KV cache.
    self.assertFalse(d2.admitting)
    self.assertFalse(d2.kv_cache)
    d2_report = next(w for w in result.workers if w.worker_id == "s2")
    self.assertTrue(d2_report.needs_restart)
    # A mixed-version fleet must poison the coordinator: the next round could
    # compound the damage.
    self.assertIsNotNone(self.coordinator.poisoned)
    with self.assertRaisesRegex(RuntimeError, "poisoned"):
      self.sync(policy_version=5)
    self.coordinator.reset_after_recovery()
    self.assertIsNone(self.coordinator.poisoned)

  def test_post_transient_failure_is_retried_to_commit(self):
    dest = FakeDestination("sampler", [], fail_on="post",
                           fail_persistently=False)
    self.make(dest)

    result = self.sync(policy_version=4)

    self.assertTrue(result.success)
    self.assertEqual(dest.serving, expected_pattern(4))

  def test_post_failing_everywhere_rolls_back_uniformly(self):
    d1 = FakeDestination("s1", [], fail_on="post", fail_persistently=True)
    d2 = FakeDestination("s2", [], fail_on="post", fail_persistently=True)
    self.make(d1, d2)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)
    self.assert_serving_old_weights(d1)
    self.assert_serving_old_weights(d2)

  def test_abort_failure_is_not_reported_as_a_clean_abort(self):
    dest = FakeDestination("sampler", [], fail_on="abort",
                           fail_persistently=True)
    self.make(dest)
    self.handler.result_success = False

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(
        ctx.exception.result.state, RoundState.FAILED_NEEDS_RESTART
    )

  def test_source_release_runs_on_the_failure_path_too(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.result_success = False

    with self.assertRaises(WeightSyncError):
      self.sync()

    self.assertEqual(self.sources[0].release_calls, 1)


class RoundIdentityTest(CoordinatorTestBase):

  def test_policy_version_must_not_regress(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.sync(policy_version=5)

    with self.assertRaisesRegex(ValueError, "regresses"):
      self.sync(policy_version=4)

  def test_retrying_the_same_version_is_allowed(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.sync(policy_version=5)

    result = self.sync(policy_version=5)

    self.assertTrue(result.success)

  def test_rounds_are_single_flight(self):
    dest = FakeDestination("sampler", [])
    coordinator = self.make(dest)

    async def scenario():
      gate = asyncio.Event()
      dest._pre_await = gate  # pylint: disable=protected-access
      task = asyncio.ensure_future(
          coordinator.sync(1)
      )
      for _ in range(50):
        await asyncio.sleep(0)
        if "sampler:pre" in self.log:
          break
      with self.assertRaisesRegex(RuntimeError, "already in flight"):
        await coordinator.sync(2)
      gate.set()
      return await task

    result = asyncio.run(scenario())
    self.assertTrue(result.success)

  def test_worker_restart_gets_fresh_ports_registered(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.sync(policy_version=1)
    old_addr = next(
        m.shards[0] for m in self.handler.registered
        if m.unit.job_name == "sampler"
    )

    dest.restart()
    self.sync(policy_version=2)

    new_addr = [
        m.shards[0] for m in self.handler.registered
        if m.unit.job_name == "sampler"
    ][-1]
    self.assertNotEqual(old_addr, new_addr)
    self.assertEqual(dest.bind_calls, 2)
    self.assertEqual(dest.serving, expected_pattern(2))


class TimeoutReconciliationTest(CoordinatorTestBase):

  def test_lost_reply_with_completed_work_counts_as_success(self):
    # pre records its round state and then never returns; the coordinator's
    # deadline fires, it asks the worker, the worker says "prepared" for this
    # (req_id, uuid), and the round proceeds.
    dest = FakeDestination("sampler", [], delay_after_phase="pre")
    self.make(
        dest,
        timeouts=dataclasses.replace(FAST_TIMEOUTS, pre=0.05),
    )

    result = self.sync()

    self.assertTrue(result.success)

  def test_stale_round_report_does_not_count(self):
    # The worker answers the status query with a different round's identity;
    # that must not be mistaken for this round's completion.
    dest = FakeDestination(
        "sampler",
        [],
        delay_after_phase="pre",
        round_report_override={
            "req_id": "someone-elses-round", "uuid": 999,
            "phase": "prepared", "policy_version": 0,
        },
    )
    self.make(
        dest,
        timeouts=dataclasses.replace(FAST_TIMEOUTS, pre=0.05),
    )

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)


class ProtocolRejectionTest(CoordinatorTestBase):

  def test_source_without_the_protocol_is_rejected(self):
    class Bare:
      def info(self):
        return datatypes.WorkerInfo(
            worker_id="bare", roles=frozenset({datatypes.Role.ACTOR.value})
        )

    dest = FakeDestination("sampler", [])
    registry = worker_registry.WorkerRegistry()
    registry.register(Bare())
    registry.register(dest)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry, handler=FakeHandler(Wire(), []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(TypeError, "prepare_weight_sync"):
      asyncio.run(coordinator.sync(1))

  def test_destination_missing_the_lifecycle_is_rejected(self):
    class MetadataOnly:
      def info(self):
        return datatypes.WorkerInfo(
            worker_id="half", roles=frozenset({datatypes.Role.ROLLOUT.value})
        )

      async def get_weight_sync_metadata(self):
        return []

    wire = Wire()
    registry = worker_registry.WorkerRegistry()
    registry.register(FakeSource("trainer", wire, []))
    registry.register(MetadataOnly())
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry, handler=FakeHandler(wire, []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(TypeError, "destination protocol"):
      asyncio.run(coordinator.sync(1))

  def test_missing_role_is_reported(self):
    wire = Wire()
    registry = worker_registry.WorkerRegistry()
    registry.register(FakeSource("trainer", wire, []))
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry, handler=FakeHandler(wire, []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(ValueError, "no workers registered"):
      asyncio.run(coordinator.sync(1))


def make_request(uuid: int, req_id: str = "r", policy_version: int = 1):
  return datatypes.WeightSyncRequest(
      policy_version=policy_version,
      extra_config={"req_id": req_id, "uuid": uuid},
  )


class WorkerRoundTrackerTest(absltest.TestCase):
  """The worker-side state machine, tested directly."""

  def setUp(self):
    super().setUp()
    self.tracker = weight_sync_coordinator.WorkerRoundTracker()

  def test_duplicate_phase_is_a_no_op(self):
    r = make_request(uuid=1)
    self.assertTrue(self.tracker.admit(r, "prepared"))
    self.tracker.complete(r, "prepared")
    self.assertFalse(self.tracker.admit(r, "prepared"))

  def test_stale_round_is_rejected(self):
    self.tracker.admit(make_request(uuid=2), "prepared")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.admit(make_request(uuid=1), "prepared")

  def test_newer_round_supersedes(self):
    r1, r2 = make_request(uuid=1), make_request(uuid=2)
    self.tracker.admit(r1, "prepared")
    self.tracker.complete(r1, "prepared")
    self.assertTrue(self.tracker.admit(r2, "prepared"))
    self.assertEqual(self.tracker.report()["uuid"], 2)

  def test_future_uuid_arriving_as_h2d_done_is_rejected(self):
    # A stray weight_sync (a duplicate, a late post from a round the
    # coordinator abandoned) must not become this worker's round: it would
    # raise the high-water mark past rounds still legitimately coming.
    r = make_request(uuid=5, req_id="round-5", policy_version=2)
    self.tracker.admit(r, "prepared")
    self.tracker.complete(r, "prepared")
    before = self.tracker.report()

    with self.assertRaisesRegex(
        weight_sync_coordinator.StaleRoundError, "cannot open a round"
    ):
      self.tracker.admit(
          make_request(uuid=9, req_id="stray", policy_version=4), "h2d_done"
      )

    self.assertEqual(self.tracker.report(), before)

  def test_future_uuid_arriving_as_committed_is_rejected(self):
    r = make_request(uuid=5, req_id="round-5", policy_version=2)
    self.tracker.admit(r, "prepared")
    self.tracker.complete(r, "prepared")
    before = self.tracker.report()

    with self.assertRaisesRegex(
        weight_sync_coordinator.StaleRoundError, "cannot open a round"
    ):
      self.tracker.admit(
          make_request(uuid=9, req_id="stray", policy_version=4), "committed"
      )

    self.assertEqual(self.tracker.report(), before)

  def test_smaller_real_round_survives_a_future_uuid_stray(self):
    # The damage the rejection prevents: had the stray been admitted, uuid 6
    # would be stale on this worker forever and no round could reach it again.
    r5 = make_request(uuid=5, req_id="round-5", policy_version=2)
    self.tracker.admit(r5, "prepared")
    self.tracker.complete(r5, "prepared")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.admit(make_request(uuid=9, req_id="stray"), "h2d_done")

    r6 = make_request(uuid=6, req_id="round-6", policy_version=3)
    for phase in ("prepared", "h2d_done", "committed"):
      self.assertTrue(self.tracker.admit(r6, phase))
      self.tracker.complete(r6, phase)

    self.assertEqual(
        self.tracker.report(),
        {
            "req_id": "round-6",
            "uuid": 6,
            "phase": "committed",
            "policy_version": 3,
        },
    )

  def test_new_round_may_open_as_aborted(self):
    # A cancellation can beat pre to the worker; refusing it would strand a
    # round nobody can close.
    r = make_request(uuid=4, req_id="cancelled")
    self.assertTrue(self.tracker.admit(r, "aborted"))
    self.tracker.complete(r, "aborted")
    self.assertEqual(self.tracker.report()["uuid"], 4)
    self.assertEqual(self.tracker.report()["phase"], "aborted")

  def test_sentinel_uuid_cannot_open_a_round_either(self):
    # A fresh tracker sits at uuid -1, and the staleness and key-reuse guards
    # are both conditioned on a round existing, so an incoming uuid of -1
    # would otherwise slip past every check -- publishing whatever is in
    # staging and reporting a committed round with no req_id at all. A
    # proto default or a high-water mark read from an empty checkpoint is
    # all it takes to produce one.
    r = make_request(uuid=-1, req_id="")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.admit(r, "committed")
    report = self.tracker.report()
    self.assertEqual(report["phase"], "idle")
    self.assertIsNone(report["req_id"])
    # And the real round that follows still opens normally.
    real = make_request(uuid=1, req_id="round-1")
    self.assertTrue(self.tracker.admit(real, "prepared"))

  def test_complete_for_a_different_round_is_rejected(self):
    self.tracker.admit(make_request(uuid=3), "prepared")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.complete(make_request(uuid=2), "prepared")

  def test_same_uuid_different_req_id_is_rejected(self):
    # A round key must never be reused: the same uuid arriving under another
    # req_id is a protocol violation, not a duplicate delivery.
    self.tracker.admit(make_request(uuid=7, req_id="round-a"), "prepared")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.admit(make_request(uuid=7, req_id="round-b"), "prepared")

  def test_commit_after_abort_is_refused(self):
    # The staging this round would publish was already discarded; committing
    # it must fail loudly, not no-op into a phantom publish.
    r = make_request(uuid=1)
    self.assertTrue(self.tracker.admit(r, "aborted"))
    self.tracker.complete(r, "aborted")
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      self.tracker.admit(r, "committed")

  def test_abort_after_commit_is_a_no_op(self):
    # The publish stands; a late abort must not undo it.
    r = make_request(uuid=1)
    self.tracker.admit(r, "prepared")  # only a pre opens the round
    self.tracker.complete(r, "prepared")
    self.assertTrue(self.tracker.admit(r, "committed"))
    self.tracker.complete(r, "committed")
    self.assertFalse(self.tracker.admit(r, "aborted"))
    self.assertEqual(self.tracker.report()["phase"], "committed")

  def test_crash_window_reports_the_previous_phase(self):
    # admit/complete are split so a crash between work and record is visible:
    # the report still says the previous phase, which is what tells the
    # coordinator to retry.
    r = make_request(uuid=1)
    self.tracker.admit(r, "prepared")
    self.tracker.complete(r, "prepared")
    self.assertTrue(self.tracker.admit(r, "h2d_done"))
    # crash here: no complete
    self.assertEqual(self.tracker.report()["phase"], "prepared")
    self.assertTrue(self.tracker.admit(r, "h2d_done"))  # retry re-enters


class WorkerIdempotencyTest(absltest.TestCase):
  """The fakes carry the tracker, so duplicates and stale RPCs behave."""

  def test_duplicate_post_after_commit_is_a_no_op(self):
    dest = FakeDestination("s", [])
    r = make_request(uuid=1, policy_version=3)

    async def scenario():
      await dest.pre_weight_sync(r)
      dest.receive([3000.0, 3001.0, 3002.0, 3003.0])
      await dest.weight_sync(r)
      await dest.post_weight_sync(r)
      before = list(dest.serving)
      await dest.post_weight_sync(r)  # duplicate delivery
      return before

    before = asyncio.run(scenario())
    self.assertEqual(dest.serving, before)
    self.assertTrue(dest.admitting)

  def test_worker_rejects_a_stale_round_rpc(self):
    dest = FakeDestination("s", [])

    async def scenario():
      await dest.pre_weight_sync(make_request(uuid=5))
      with self.assertRaises(weight_sync_coordinator.StaleRoundError):
        await dest.pre_weight_sync(make_request(uuid=4))

    asyncio.run(scenario())


class PostReconciliationTest(CoordinatorTestBase):
  """A failed post RPC is not a failed publish until the worker says so."""

  def test_post_reply_lost_to_timeout_still_commits(self):
    dest = FakeDestination("sampler", [], delay_after_phase="post")
    self.make(
        dest, timeouts=dataclasses.replace(FAST_TIMEOUTS, post=0.05)
    )

    result = self.sync(policy_version=7)

    self.assertTrue(result.success)
    self.assertEqual(dest.serving, expected_pattern(7))

  def test_post_reply_lost_to_connection_error_still_commits(self):
    # Not a timeout: the RPC layer raised after the work was done. The
    # reconciliation must run for ANY failure, not only TimeoutError.
    dest = FakeDestination("sampler", [], raise_after_complete_once="post")
    self.make(dest)

    result = self.sync(policy_version=7)

    self.assertTrue(result.success)
    self.assertEqual(dest.serving, expected_pattern(7))

  def test_crash_between_publish_and_record_converges_on_retry(self):
    # The worker publishes, crashes before recording, and reports h2d_done.
    # The retry must re-enter an idempotent post and converge to committed
    # without double-applying.
    dest = FakeDestination("sampler", [], crash_after_publish=True)
    self.make(dest)

    result = self.sync(policy_version=9)

    self.assertTrue(result.success)
    self.assertEqual(dest.serving, expected_pattern(9))
    self.assertTrue(dest.admitting)
    self.assertTrue(dest.kv_cache)

  def test_unknown_post_state_is_failed_needs_restart_and_never_aborted(self):
    # s2's post fails AND its status query is unreachable. Nothing can be
    # inferred -- including that it did NOT publish -- so aborting it could
    # roll back a publish that happened. It gets needs_restart, not an abort.
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination(
        "s2", [], fail_on="post", fail_persistently=True,
        status_unreachable=True,
    )
    self.make(d1, d2)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=4)

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.FAILED_NEEDS_RESTART)
    self.assertNotIn("abort", self.phases("s2"))
    d2_report = next(w for w in result.workers if w.worker_id == "s2")
    self.assertTrue(d2_report.needs_restart)
    self.assertEqual(d2_report.phase, "unknown")
    self.assertIsNotNone(self.coordinator.poisoned)

  def test_post_retry_unknown_state_is_needs_restart_not_aborted(self):
    # The worker was POSITIVELY h2d_done before the retry, but its
    # post-retry classification query answers with a different round's
    # identity. Nothing can be inferred anymore -- including that it did not
    # publish in between -- so it must join the unknown bucket
    # (needs_restart, never aborted), not still_failed, whose
    # committed_count==0 path would abort it.
    key = {"req_id": "wsync-v1-r0", "uuid": 1, "policy_version": 1}
    dest = FakeDestination(
        "sampler",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_sequence=[
            dict(key, phase="h2d_done"),  # _call_phase reconcile, 1st post
            dict(key, phase="h2d_done"),  # classification -> retryable
            dict(key, phase="h2d_done"),  # _call_phase reconcile, retry
            {"req_id": "someone-elses-round", "uuid": 999,
             "phase": "prepared", "policy_version": 0},  # retry classify
        ],
    )
    self.make(dest)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=1)

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.FAILED_NEEDS_RESTART)
    self.assertNotIn("abort", self.phases("sampler"))
    report = next(w for w in result.workers if w.worker_id == "sampler")
    self.assertTrue(report.needs_restart)
    self.assertIsNotNone(self.coordinator.poisoned)


class TransferTimeoutTest(CoordinatorTestBase):

  def test_timed_out_transfer_parks_unknown_and_touches_nothing(self):
    dest = FakeDestination("sampler", [])
    self.make(
        dest,
        timeouts=dataclasses.replace(FAST_TIMEOUTS, transfer=0.05),
    )
    self.handler.transfer_delay = 0.4  # the executor thread outlives the wait

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(
        ctx.exception.result.state, RoundState.UNKNOWN_TRANSFER_STATE
    )
    # The thread may still be writing: aborting would discard buffers in use,
    # releasing source staging would pull memory out from under it.
    self.assertNotIn("abort", self.phases("sampler"))
    self.assertEqual(self.sources[0].release_calls, 0)
    self.assertIsNotNone(self.coordinator.poisoned)

    # Recovery is explicit: reset, then a fresh round works.
    self.handler.transfer_delay = 0.0
    self.coordinator.reset_after_recovery()
    result = self.sync(policy_version=2)
    self.assertTrue(result.success)


class TransferOutcomeUnknownTest(CoordinatorTestBase):

  def test_outcome_unknown_parks_unknown_and_touches_nothing(self):
    # The transport's RPC timed out client-side: the reply is lost and the
    # controller may still be executing the transfer. Same treatment as the
    # coordinator's own transfer deadline -- never the rollback branch.
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.raise_on_transfer = weight_sync.TransferOutcomeUnknownError(
        "coordinate_transfer timed out client-side"
    )

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(
        ctx.exception.result.state, RoundState.UNKNOWN_TRANSFER_STATE
    )
    self.assertNotIn("abort", self.phases("sampler"))
    self.assertEqual(self.sources[0].release_calls, 0)
    self.assertIsNotNone(self.coordinator.poisoned)


class CancellationTest(CoordinatorTestBase):

  def test_cancelled_round_still_rolls_destinations_back(self):
    dest = FakeDestination("sampler", [])
    coordinator = self.make(dest)

    async def scenario():
      started = asyncio.Event()  # set at pre entry
      gate = asyncio.Event()  # never set: pre blocks until cancelled
      dest._pre_gate = started  # pylint: disable=protected-access
      dest._pre_await = gate  # pylint: disable=protected-access
      task = asyncio.ensure_future(
          coordinator.sync(1)
      )
      # Event-driven, not a capped yield loop: registration runs through the
      # executor, and on a slow machine a fixed number of bare yields can
      # expire before pre begins -- cancelling pre-quiesce, where there is
      # nothing to roll back and the abort assertion has no subject.
      await asyncio.wait_for(started.wait(), 10)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    # The quiesced destination must not be stranded: the rollback is shielded
    # from the cancellation and actually runs.
    self.assertIn("abort", self.phases("sampler"))
    self.assertTrue(dest.admitting)
    self.assertTrue(dest.kv_cache)

  def test_cancel_during_release_waits_for_it_to_finish(self):
    # The release is shielded so a cancel cannot kill it -- but shielding
    # alone lets the AWAITING coroutine leave immediately, so the release
    # runs on behind a round the caller already considers over. It would
    # then overlap the next round's prepare, or a close() that frees the
    # staging it is still reading.
    dest = FakeDestination("sampler", [])
    coordinator = self.make(dest)
    source = self.sources[0]
    source.release_gate = None  # set inside the loop below

    async def scenario():
      source.release_gate = asyncio.Event()
      source.release_entered = asyncio.Event()
      task = asyncio.ensure_future(
          coordinator.sync(1)
      )
      await asyncio.wait_for(source.release_entered.wait(), 10)
      task.cancel()
      # The cancellation must NOT come back while the release is still open.
      with self.assertRaises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), 0.3)
      source.release_gate.set()
      with self.assertRaises(asyncio.CancelledError):
        await task
      # And it finished before the caller was let go.
      self.assertEqual(source.release_calls, 1)
      self.assertIn("trainer:release", self.log)

    asyncio.run(scenario())

  def test_cancelled_round_with_failed_rollback_poisons(self):
    # The shielded rollback's OUTCOME must not be dropped: an abort failure
    # during a cancelled round leaves a possibly-unserving worker, and the
    # coordinator must poison rather than report poisoned=None.
    dest = FakeDestination(
        "sampler", [], fail_on="abort", fail_persistently=True
    )
    coordinator = self.make(dest)

    async def scenario():
      started = asyncio.Event()
      gate = asyncio.Event()  # never set: pre blocks until cancelled
      dest._pre_gate = started  # pylint: disable=protected-access
      dest._pre_await = gate  # pylint: disable=protected-access
      task = asyncio.ensure_future(
          coordinator.sync(1)
      )
      await asyncio.wait_for(started.wait(), 10)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertIsNotNone(self.coordinator.poisoned)

  def test_cancel_during_transfer_parks_unknown_and_touches_nothing(self):
    # Cancellation cannot reach the executor thread running the blocking
    # transfer. Treating a mid-transfer cancel like a normal mid-round cancel
    # would abort destinations and release source staging while the thread
    # may still be writing into them; it must get the timeout treatment.
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.transfer_delay = 2.0

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1)
      )
      while "transfer" not in self.log:
        await asyncio.sleep(0.01)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertNotIn("abort", self.phases("sampler"))
    self.assertEqual(self.sources[0].release_calls, 0)
    self.assertIsNotNone(self.coordinator.poisoned)

  def test_cancel_during_post_with_split_fleet_poisons_partial(self):
    # Cancel lands while post RPCs are in flight: one worker already
    # committed (its RPC reply is just slow), one did not. The SAME policy
    # as the normal path applies: with a commit present, the h2d_done
    # worker is NOT aborted -- reopening it on the old policy would serve
    # mixed versions next to the committed worker. It stays quiesced
    # (fail closed) and the coordinator poisons.
    dest_a = FakeDestination("sampler-a", [], delay_after_phase="post")
    dest_b = FakeDestination(
        "sampler-b", [], fail_on="post", fail_persistently=True
    )
    self.make(dest_a, dest_b)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1)
      )
      while not (
          "sampler-a:post" in self.log and "sampler-b:post" in self.log
      ):
        await asyncio.sleep(0.01)
      await asyncio.sleep(0.1)  # let B's failure reconcile; A still hangs
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertIsNotNone(self.coordinator.poisoned)
    self.assertNotIn("abort", self.phases("sampler-a"))  # publish stands
    self.assertNotIn("abort", self.phases("sampler-b"))  # fail closed: DOWN
    self.assertEqual(dest_a.serving, expected_pattern(1))
    self.assertFalse(dest_b.admitting)  # quiesced, not serving old weights

  def test_cancel_during_post_with_unknown_worker_never_aborts_it(self):
    # Same cancellation window, but one worker's status answers with a
    # different round's identity: unknown. Coarse bucketing would abort
    # everything not reporting committed -- including unknown, which could
    # roll back a publish that actually happened. Unknown is never aborted,
    # on this path like every other.
    dest_a = FakeDestination("sampler-a", [], delay_after_phase="post")
    dest_b = FakeDestination(
        "sampler-b",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_override={
            "req_id": "someone-elses-round", "uuid": 999,
            "phase": "prepared", "policy_version": 0,
        },
    )
    self.make(dest_a, dest_b)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1)
      )
      while not (
          "sampler-a:post" in self.log and "sampler-b:post" in self.log
      ):
        await asyncio.sleep(0.01)
      await asyncio.sleep(0.1)  # let B's failure reconcile; A still hangs
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertIsNotNone(self.coordinator.poisoned)
    self.assertNotIn("abort", self.phases("sampler-b"))  # unknown: hands off
    self.assertNotIn("abort", self.phases("sampler-a"))  # publish stands
    self.assertEqual(dest_a.serving, expected_pattern(1))
    # Same precedence as the normal path: an unknown worker outranks the
    # version split, so this is needs-restart and not partially_committed.
    self.assertIn("failed_needs_restart", self.coordinator.poisoned)

  def test_cancel_during_registration_waits_for_all_registrations(self):
    # A cancel cannot stop the registration executor threads, only orphan
    # them -- and an orphaned registration landing AFTER the next round
    # re-registers the same unit would overwrite fresh ports/layout with
    # this round's stale ones. The shield must hold the cancellation until
    # every registration has actually settled.
    dest = FakeDestination("sampler", [])
    self.make(dest)
    entered = threading.Event()
    release = threading.Event()
    settled: list[str] = []
    original = self.handler.register_work_unit

    def slow_register(metadata):
      entered.set()
      release.wait(timeout=10)
      original(metadata)
      settled.append(metadata.unit.job_name)

    self.handler.register_work_unit = slow_register

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1)
      )
      while not entered.is_set():
        await asyncio.sleep(0.01)
      task.cancel()
      # The cancellation must NOT resolve while registration threads run.
      with self.assertRaises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), 0.3)
      release.set()
      with self.assertRaises(asyncio.CancelledError):
        await task
      self.assertLen(settled, 2)  # source + destination both settled

    asyncio.run(scenario())

  def test_cancel_recovery_rereads_after_abort_and_records_late_commit(self):
    # Classification sees h2d_done, so the worker is aborted -- but its post
    # completed between the classification query and the abort, so the abort
    # is the tracker's no-op over a committed round and the RPC returns
    # cleanly. Without re-reading the report, the recovery would end with an
    # actually-committed fleet: version unrecorded and nothing poisoned.
    key = {"req_id": "wsync-v1-r0", "uuid": 1, "policy_version": 1}
    dest = FakeDestination(
        "sampler",
        [],
        delay_after_phase="post",  # post body completes; the reply hangs
        round_report_sequence=[
            dict(key, phase="h2d_done"),  # classification: abort candidate
            dict(key, phase="committed"),  # re-read after the clean abort
        ],
    )
    self.make(dest)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1)
      )
      while "sampler:post" not in self.log:
        await asyncio.sleep(0.01)
      await asyncio.sleep(0.05)  # let the post body complete; reply hangs
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    # The publish survived (the abort no-opped) and the re-read saw it:
    # the committed version must be recorded exactly like the direct
    # all-committed case.
    self.assertEqual(dest.serving, expected_pattern(1))
    self.assertEqual(self.coordinator.last_committed_version, 1)
    self.assertIsNone(self.coordinator.poisoned)

  def test_cancel_during_post_when_all_committed_records_the_version(self):
    # Same window, but every worker committed: the cancel lost the result,
    # not the round. Nothing is aborted, nothing poisons, and the committed
    # version is recorded so the next round's regression guard is truthful.
    dest = FakeDestination("sampler", [], delay_after_phase="post")
    self.make(dest)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(3)
      )
      while "sampler:post" not in self.log:
        await asyncio.sleep(0.01)
      await asyncio.sleep(0.05)  # let the post body complete; reply hangs
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertIsNone(self.coordinator.poisoned)
    self.assertNotIn("abort", self.phases("sampler"))
    self.assertEqual(dest.serving, expected_pattern(3))
    self.assertEqual(self.coordinator.last_committed_version, 3)


class TerminalReconciliationTest(CoordinatorTestBase):
  """The two terminal phases prove opposite things and must not be conflated.

  Both regressions here pass under a rank-based reconciliation ("committed"
  and "aborted" share the terminal rank) and are exactly why the coordinator
  reconciles against explicit per-call accept sets instead.
  """

  def test_post_failure_with_aborted_report_is_not_success(self):
    # The post RPC fails and the worker reports "aborted" for this round: the
    # publish did NOT happen. Mistaking that for a successful post would
    # declare the round committed while the worker serves the old weights.
    dest = FakeDestination(
        "sampler",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_override={
            "req_id": "wsync-v1-r0", "uuid": 1,
            "phase": "aborted", "policy_version": 0,
        },
    )
    self.make(dest)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.ABORTED)
    report = {w.worker_id: w for w in result.workers}["sampler"]
    self.assertEqual(report.phase, "aborted")
    # Alive and consistent on the old weights: a version outcome, not a
    # health problem.
    self.assertFalse(report.needs_restart)

  def test_abort_failure_with_committed_report_is_not_rollback_success(self):
    # Rollback's abort RPC fails and the worker reports "committed": the
    # publish stands and rollback did NOT happen. Mistaking that for a
    # successful abort would report a clean ABORTED round while one worker
    # serves the new weights.
    dest = FakeDestination(
        "sampler",
        [],
        fail_on="abort",
        fail_persistently=True,
        round_report_override={
            "req_id": "wsync-v1-r0", "uuid": 1,
            "phase": "committed", "policy_version": 1,
        },
    )
    self.make(dest)
    self.handler.result_success = False
    self.handler.result_message = "transfer exploded"

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(
        ctx.exception.result.state, RoundState.FAILED_NEEDS_RESTART
    )
    self.assertIsNotNone(self.coordinator.poisoned)

  def test_rollback_over_late_commit_is_partial_not_aborted(self):
    # committed_count==0 -> uniform rollback. One worker's post actually
    # published late: its abort no-ops on the committed round and the RPC
    # returns cleanly. A clean abort RPC is not proof of a rollback -- the
    # re-read report says committed, so the round is PARTIALLY_COMMITTED
    # (mixed versions), never a clean ABORTED.
    key = {"req_id": "wsync-v1-r0", "uuid": 1, "policy_version": 1}
    dest_a = FakeDestination(
        "sampler-a",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_sequence=[
            dict(key, phase="h2d_done"),  # _call_phase reconcile, 1st post
            dict(key, phase="h2d_done"),  # classification -> retryable
            dict(key, phase="h2d_done"),  # _call_phase reconcile, retry
            dict(key, phase="h2d_done"),  # retry classify -> still_failed
            dict(key, phase="committed"),  # rollback confirm: published!
        ],
    )
    dest_b = FakeDestination(
        "sampler-b", [], fail_on="post", fail_persistently=True
    )
    self.make(dest_a, dest_b)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=1)

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.PARTIALLY_COMMITTED)
    self.assertTrue(any("publish stands" in f for f in result.failures))
    self.assertIsNotNone(self.coordinator.poisoned)

  def test_rollback_unconfirmed_after_clean_abort_is_needs_restart(self):
    # Post-path rollback (a commit was possible): the abort RPC returns
    # cleanly but the re-read cannot confirm "aborted" -- it answers with a
    # different round's identity. A clean abort here can be a no-op over a
    # completed publish, so this must NOT be certified ABORTED. (Contrast
    # test_stale_round_report_does_not_count: same flaky report on the PRE
    # path, where no post ever ran, a commit is impossible, and the clean
    # abort alone certifies the rollback.)
    key = {"req_id": "wsync-v1-r0", "uuid": 1, "policy_version": 1}
    dest_a = FakeDestination(
        "sampler-a",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_sequence=[
            dict(key, phase="h2d_done"),  # _call_phase reconcile, 1st post
            dict(key, phase="h2d_done"),  # classification -> retryable
            dict(key, phase="h2d_done"),  # _call_phase reconcile, retry
            dict(key, phase="h2d_done"),  # retry classify -> still_failed
            {"req_id": "someone-elses-round", "uuid": 999,
             "phase": "prepared", "policy_version": 0},  # rollback re-read
        ],
    )
    dest_b = FakeDestination(
        "sampler-b", [], fail_on="post", fail_persistently=True
    )
    self.make(dest_a, dest_b)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync(policy_version=1)

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.FAILED_NEEDS_RESTART)
    self.assertTrue(any("unconfirmed" in f for f in result.failures))
    self.assertIsNotNone(self.coordinator.poisoned)

  def test_post_failure_with_late_committed_report_is_success(self):
    # A timed-out post can still be executing at _call_phase's first
    # reconciliation poll (worker says h2d_done) and finish before
    # _resolve_post_failures re-polls (worker says committed). That is a
    # successful publish, not an unknown worker: the round must commit, not
    # end FAILED_NEEDS_RESTART with a poisoned coordinator.
    key = {"req_id": "wsync-v1-r0", "uuid": 1, "policy_version": 1}
    dest = FakeDestination(
        "sampler",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_sequence=[
            dict(key, phase="h2d_done"),
            dict(key, phase="committed"),
        ],
    )
    self.make(dest)

    result = self.sync()

    self.assertTrue(result.success)
    self.assertIsNone(self.coordinator.poisoned)

  def test_aborted_rollback_corrects_worker_reports(self):
    # committed_count==0 -> rollback -> ABORTED. The interim per-worker
    # records (h2d_done, needs_restart=True) must be rewritten once every
    # abort confirmed, or the result contradicts its own state and directs
    # restarts of healthy workers.
    dest_a = FakeDestination(
        "sampler-a",
        [],
        fail_on="post",
        fail_persistently=True,
        round_report_override={
            "req_id": "wsync-v1-r0", "uuid": 1,
            "phase": "aborted", "policy_version": 0,
        },
    )
    dest_b = FakeDestination(
        "sampler-b", [], fail_on="post", fail_persistently=True
    )
    self.make(dest_a, dest_b)

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    result = ctx.exception.result
    self.assertIs(result.state, RoundState.ABORTED)
    self.assertLen(result.workers, 2)
    for report in result.workers:
      self.assertEqual(report.phase, "aborted", report)
      self.assertFalse(report.needs_restart, report)


class SourcePrepareTimeoutTest(CoordinatorTestBase):

  def test_source_prepare_timeout_fails_before_any_downtime(self):
    dest = FakeDestination("sampler", [])
    wire = Wire()
    slow_source = FakeSource("trainer", wire, [], prepare_delay=60)
    self.make(
        dest,
        sources=[slow_source],
        timeouts=dataclasses.replace(FAST_TIMEOUTS, source_prepare=0.05),
    )

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIn("pre-quiesce", "".join(ctx.exception.result.failures))
    # The destinations were never quiesced, so there is nothing to roll back
    # and the failure is clean: no abort, no poison.
    self.assertNotIn("pre", self.phases("sampler"))
    self.assertNotIn("abort", self.phases("sampler"))
    self.assertIsNone(self.coordinator.poisoned)
    # Release still runs; workers must make it safe against an in-flight
    # prepare for the same round.
    self.assertEqual(slow_source.release_calls, 1)


if __name__ == "__main__":
  absltest.main()
