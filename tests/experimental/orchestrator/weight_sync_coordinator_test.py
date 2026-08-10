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
import time
from typing import Any, Mapping, Optional, Sequence

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import weight_sync
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.orchestrator import worker_registry

RaidenId = weight_sync.RaidenId
RoundState = weight_sync_coordinator.RoundState
WeightSyncError = weight_sync_coordinator.WeightSyncError
PhaseTimeouts = weight_sync_coordinator.PhaseTimeouts

FAST_TIMEOUTS = PhaseTimeouts(
    bind=5,
    metadata=5,
    source_prepare=5,
    pre=5,
    transfer=5,
    h2d=5,
    post=5,
    abort=5,
    status=5,
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
      variables: Sequence[weight_sync.VariableMetadata] = (),
      prepare_delay: float = 0.0,
  ):
    self._info = datatypes.WorkerInfo(
        worker_id=worker_id, roles=frozenset({"trainer"})
    )
    self._wire = wire
    self._log = log
    self._hosts = hosts
    self._variables = tuple(variables)
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
        weight_sync.RaidenWorkUnitMetadata(
            unit=RaidenId(
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
    self.release_calls += 1
    self._log.append(f"{self._info.worker_id}:release")


class FakeDestination:
  """Sampler side, with real staging/serving double-buffer semantics and a

  real `WorkerRoundTracker` behind every phase.

  `fail_on` makes the named phase raise AFTER its partial work, which is the
  dangerous variant: a pre that has already gated admission, an abort that has
  not yet restored serving. `crash_after_publish` raises once between
  publishing and recording — the crash-consistency window the tracker's
  admit/complete split exists for. `raise_after_complete_once` completes the
  phase fully and then raises — a lost reply.
  """

  def __init__(
      self,
      worker_id: str,
      log: list[str],
      fail_on: Optional[str] = None,
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
        worker_id=worker_id, roles=frozenset({"rollout"})
    )
    self._log = log
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

    self.unit = RaidenId(job_name=worker_id, job_replica_id="0")
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
    return [
        weight_sync.RaidenWorkUnitMetadata(
            unit=self.unit,
            shards=(f"10.0.0.2:{self.port}",),
            control_plane_rpc_address=f"10.0.0.2:{self.port + 500}",
            global_shape=(4,),
            mesh_shape=(1,),
            layout=(0,),
            item_size=4,
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

  async def get_weight_sync_round(self):
    if self._status_unreachable:
      raise ConnectionResetError(f"{self._info.worker_id}: status unreachable")
    if self._round_report_sequence:
      if len(self._round_report_sequence) > 1:
        return dict(self._round_report_sequence.pop(0))
      return dict(self._round_report_sequence[0])
    if self._round_report_override is not None:
      return dict(self._round_report_override)
    return self.tracker.report()


class FakeActorHandle:
  """Equivalent of a GrpcRemoteActorHandle: method-name dispatch only."""

  def __init__(self, target: Any):
    self._target = target
    self.calls: list[str] = []

  def submit(self, method_name: Optional[str] = None, *args, **kwargs):
    self.calls.append(method_name)
    return getattr(self._target, method_name)(*args, **kwargs)

  async def asubmit(self, method_name: Optional[str] = None, *args, **kwargs):
    self.calls.append(method_name)
    result = getattr(self._target, method_name)(*args, **kwargs)
    if asyncio.iscoroutine(result):
      return await result
    return result


class FakeHandler(weight_sync.WeightSyncHandler):
  """Records registrations and transfers; moves the wire pattern on transfer."""

  def __init__(self, wire: Wire, log: list[str]):
    self._wire = wire
    self._log = log
    self._destinations: list[FakeDestination] = []
    self.registered: list[weight_sync.RaidenWorkUnitMetadata] = []
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

  def transfer(self, src_units, dst_units, req_id=None, **kwargs):
    self._log.append("transfer")
    if self.transfer_delay:
      time.sleep(self.transfer_delay)
    self.transfers.append(
        dict(
            src=list(src_units),
            dst=list(dst_units),
            req_id=req_id,
            uuid=kwargs.get("uuid"),
            expected_block_count=kwargs.get("expected_block_count"),
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

  def make(self, *destinations: FakeDestination, sources=None, timeouts=None):
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

  def sync(self, policy_version=1, expected_block_count=4, **kwargs):
    return asyncio.run(
        self.coordinator.sync(
            policy_version,
            expected_block_count=expected_block_count,
            **kwargs,
        )
    )

  def phases(self, worker_id: str) -> list[str]:
    prefix = f"{worker_id}:"
    return [e[len(prefix) :] for e in self.log if e.startswith(prefix)]


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
        m.shards[0]
        for m in self.handler.registered
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
        m.shards[0]
        for m in self.handler.registered
        if m.unit.job_name == "trainer"
    ]
    in_request = [m.shards[0] for m in dest.last_request.source_metadata]
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
    self.assertEqual([t["uuid"] for t in transfers], [r1.uuid, r2.uuid])

  def test_expected_block_count_reaches_the_handler(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    self.sync(expected_block_count=8)

    self.assertEqual(self.handler.transfers[0]["expected_block_count"], 8)

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
      return await coordinator.sync(1, expected_block_count=4)

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
        weight_sync.VariableMetadata(
            name="w",
            shape=(8, 4),
            mesh_shape=(1, 1),
            layout=(1, 0),
            item_size=4,
            layer_idx=0,
        ),
        weight_sync.VariableMetadata(
            name="b",
            shape=(4,),
            mesh_shape=(1,),
            layout=(0,),
            item_size=2,
            layer_idx=1,
        ),
    )
    dest = FakeDestination("sampler", [])
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
    # "serving the old version", it is DOWN — post failed before restoring
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
    dest = FakeDestination(
        "sampler", [], fail_on="post", fail_persistently=False
    )
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
    dest = FakeDestination(
        "sampler", [], fail_on="abort", fail_persistently=True
    )
    self.make(dest)
    self.handler.result_success = False

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.FAILED_NEEDS_RESTART)

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
      task = asyncio.ensure_future(coordinator.sync(1, expected_block_count=4))
      for _ in range(50):
        await asyncio.sleep(0)
        if "sampler:pre" in self.log:
          break
      with self.assertRaisesRegex(RuntimeError, "already in flight"):
        await coordinator.sync(2, expected_block_count=4)
      gate.set()
      return await task

    result = asyncio.run(scenario())
    self.assertTrue(result.success)

  def test_worker_restart_gets_fresh_ports_registered(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.sync(policy_version=1)
    old_addr = next(
        m.shards[0]
        for m in self.handler.registered
        if m.unit.job_name == "sampler"
    )

    dest.restart()
    self.sync(policy_version=2)

    new_addr = [
        m.shards[0]
        for m in self.handler.registered
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
            "req_id": "someone-elses-round",
            "uuid": 999,
            "phase": "prepared",
            "policy_version": 0,
        },
    )
    self.make(
        dest,
        timeouts=dataclasses.replace(FAST_TIMEOUTS, pre=0.05),
    )

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.ABORTED)


class RemoteProxyTest(CoordinatorTestBase):

  def test_full_round_through_method_name_dispatch(self):
    # The registry stores Worker-shaped objects; a remote worker is reachable
    # only through submit/asubmit. The proxy has to be a full stand-in.
    log: list[str] = []
    wire = Wire()
    real_source = FakeSource("trainer", wire, log)
    real_dest = FakeDestination("sampler", log)

    source_handle = FakeActorHandle(real_source)
    dest_handle = FakeActorHandle(real_dest)
    source_proxy = weight_sync_coordinator.RemoteParticipantProxy(
        source_handle,
        datatypes.WorkerInfo(worker_id="trainer", roles=frozenset({"trainer"})),
    )
    dest_proxy = weight_sync_coordinator.RemoteParticipantProxy(
        dest_handle,
        datatypes.WorkerInfo(worker_id="sampler", roles=frozenset({"rollout"})),
    )

    self.log = log
    self.wire = wire
    self.handler = FakeHandler(wire, log)
    self.handler.attach(real_dest)
    self.registry = worker_registry.WorkerRegistry()
    self.registry.register(source_proxy)
    self.registry.register(dest_proxy)
    self.sources = [real_source]
    self.coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=self.registry,
        handler=self.handler,
        timeouts=FAST_TIMEOUTS,
    )

    result = self.sync(policy_version=6)

    self.assertTrue(result.success)
    self.assertEqual(real_dest.serving, expected_pattern(6))
    # Everything went through name dispatch, nothing through direct attribute
    # access on the real workers.
    self.assertIn("pre_weight_sync", dest_handle.calls)
    self.assertIn("prepare_weight_sync", source_handle.calls)


class ProtocolRejectionTest(CoordinatorTestBase):

  def test_source_without_the_protocol_is_rejected(self):
    class Bare:

      def info(self):
        return datatypes.WorkerInfo(
            worker_id="bare", roles=frozenset({"trainer"})
        )

    dest = FakeDestination("sampler", [])
    registry = worker_registry.WorkerRegistry()
    registry.register(Bare())
    registry.register(dest)
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=FakeHandler(Wire(), []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(TypeError, "prepare_weight_sync"):
      asyncio.run(coordinator.sync(1, expected_block_count=4))

  def test_destination_missing_the_lifecycle_is_rejected(self):
    class MetadataOnly:

      def info(self):
        return datatypes.WorkerInfo(
            worker_id="half", roles=frozenset({"rollout"})
        )

      async def get_weight_sync_metadata(self):
        return []

    wire = Wire()
    registry = worker_registry.WorkerRegistry()
    registry.register(FakeSource("trainer", wire, []))
    registry.register(MetadataOnly())
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=FakeHandler(wire, []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(TypeError, "destination protocol"):
      asyncio.run(coordinator.sync(1, expected_block_count=4))

  def test_missing_role_is_reported(self):
    wire = Wire()
    registry = worker_registry.WorkerRegistry()
    registry.register(FakeSource("trainer", wire, []))
    coordinator = weight_sync_coordinator.WeightSyncCoordinator(
        registry=registry,
        handler=FakeHandler(wire, []),
        timeouts=FAST_TIMEOUTS,
    )

    with self.assertRaisesRegex(ValueError, "no workers registered"):
      asyncio.run(coordinator.sync(1, expected_block_count=4))


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
    self.make(dest, timeouts=dataclasses.replace(FAST_TIMEOUTS, post=0.05))

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
    # inferred — including that it did NOT publish — so aborting it could
    # roll back a publish that happened. It gets needs_restart, not an abort.
    d1 = FakeDestination("s1", [])
    d2 = FakeDestination(
        "s2",
        [],
        fail_on="post",
        fail_persistently=True,
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

    self.assertIs(ctx.exception.result.state, RoundState.UNKNOWN_TRANSFER_STATE)
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
    # coordinator's own transfer deadline — never the rollback branch.
    dest = FakeDestination("sampler", [])
    self.make(dest)
    self.handler.raise_on_transfer = weight_sync.TransferOutcomeUnknownError(
        "coordinate_transfer timed out client-side"
    )

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.UNKNOWN_TRANSFER_STATE)
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
      task = asyncio.ensure_future(coordinator.sync(1, expected_block_count=4))
      # Event-driven, not a capped yield loop: registration runs through the
      # executor, and on a slow machine a fixed number of bare yields can
      # expire before pre begins — cancelling pre-quiesce, where there is
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
          self.coordinator.sync(1, expected_block_count=4)
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
    # committed (its RPC reply is just slow), one did not. A blanket
    # rollback would no-op on the committed worker and roll back the other —
    # mixed versions with nobody told. The handler must sort by each
    # worker's own report, abort only the unpublished, and poison.
    dest_a = FakeDestination("sampler-a", [], delay_after_phase="post")
    dest_b = FakeDestination(
        "sampler-b", [], fail_on="post", fail_persistently=True
    )
    self.make(dest_a, dest_b)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(1, expected_block_count=4)
      )
      while not ("sampler-a:post" in self.log and "sampler-b:post" in self.log):
        await asyncio.sleep(0.01)
      await asyncio.sleep(0.1)  # let B's failure reconcile; A still hangs
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(scenario())

    self.assertIsNotNone(self.coordinator.poisoned)
    self.assertNotIn("abort", self.phases("sampler-a"))  # publish stands
    self.assertIn("abort", self.phases("sampler-b"))
    self.assertEqual(dest_a.serving, expected_pattern(1))
    self.assertTrue(dest_b.admitting)

  def test_cancel_during_post_when_all_committed_records_the_version(self):
    # Same window, but every worker committed: the cancel lost the result,
    # not the round. Nothing is aborted, nothing poisons, and the committed
    # version is recorded so the next round's regression guard is truthful.
    dest = FakeDestination("sampler", [], delay_after_phase="post")
    self.make(dest)

    async def scenario():
      task = asyncio.ensure_future(
          self.coordinator.sync(3, expected_block_count=4)
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
            "req_id": "wsync-v1-r0",
            "uuid": 1,
            "phase": "aborted",
            "policy_version": 0,
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
            "req_id": "wsync-v1-r0",
            "uuid": 1,
            "phase": "committed",
            "policy_version": 1,
        },
    )
    self.make(dest)
    self.handler.result_success = False
    self.handler.result_message = "transfer exploded"

    with self.assertRaises(WeightSyncError) as ctx:
      self.sync()

    self.assertIs(ctx.exception.result.state, RoundState.FAILED_NEEDS_RESTART)
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
            "req_id": "wsync-v1-r0",
            "uuid": 1,
            "phase": "aborted",
            "policy_version": 0,
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


class ExpectedBlockCountTest(CoordinatorTestBase):

  def test_zero_expected_block_count_defers_to_the_transport(self):
    # 0 means "the transport's own schedule-derived count"; the round runs
    # and the 0 reaches the handler untouched.
    dest = FakeDestination("sampler", [])
    self.make(dest)

    result = self.sync(expected_block_count=0)

    self.assertTrue(result.success)
    self.assertEqual(self.handler.transfers[0]["expected_block_count"], 0)

  def test_negative_expected_block_count_is_rejected_upfront(self):
    dest = FakeDestination("sampler", [])
    self.make(dest)

    with self.assertRaisesRegex(ValueError, "expected_block_count"):
      self.sync(expected_block_count=-1)

    self.assertEqual(self.log, [])  # nothing was touched


if __name__ == "__main__":
  absltest.main()
