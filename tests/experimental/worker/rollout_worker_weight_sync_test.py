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

"""RolloutWorker weight-sync phase tests.

Silent failures this file locks out:
  - A quiesced, cache-less worker reported READY (healthy) between pre and
    post/abort, when it cannot actually serve.
  - post_weight_sync publishing staging whose H2D never ran (phase ORDER,
    not just round identity) and then recording the round as committed.
  - A stop() racing a phase call getting recorded as a committed or aborted
    round: the tracker must stay at the last truly-reached phase so the
    coordinator reconciles needs-restart, never a false terminal.
  - An abort that the sampler did not positively confirm (mismatched round
    key, an already-published round, an unknown report format, or a failed
    status RPC) being recorded as "aborted".
  - Admission reopening before the round reached a serving terminal: new
    requests admitted while the rollback is still unconfirmed, or over a
    publish that turned out not to be serving.
  - Duplicate phase deliveries re-running work, and stale round keys from a
    previous round touching current state.
  - get_weight_sync_round answering from the sampler's internal sub-state
    instead of the worker's authoritative tracker.
"""

import asyncio
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import weight_sync_coordinator
from tunix.experimental.rollout import manager as manager_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.worker import rollout_worker

WorkerState = datatypes.WorkerState


class FakeSampler:
  """All 16 Sampler protocol methods; phase methods track a .round dict.

  Test knobs:
    gates: phase name -> asyncio.Event the phase awaits before finishing,
      so a test can hold a phase at a known await point. "round" gates the
      round report, i.e. the abort's confirmation query.
    started: phase name -> asyncio.Event set when the phase is entered.
    round_override: if set, get_weight_sync_round returns it verbatim,
      simulating a sampler sub-state that disagrees with the worker.
    round_error: if set, get_weight_sync_round raises it (status RPC
      failure).
  """

  def __init__(self):
    self.calls = []
    self.round = {"req_id": None, "uuid": None, "phase": "idle"}
    self.gates = {}
    self.started = {}
    self.round_override = None
    self.round_error = None
    self.round_returns_none = False

  async def _enter(self, name):
    self.calls.append(name)
    if name in self.started:
      self.started[name].set()
    if name in self.gates:
      await self.gates[name].wait()

  def _key(self, sync_request):
    extra = getattr(sync_request, "extra_config", None) or {}
    return extra.get("req_id"), extra.get("uuid")

  async def start(self, **kw):
    pass

  async def stop(self, **kw):
    pass

  async def pause(self, **kw):
    pass

  async def resume(self, **kw):
    pass

  async def get_mesh(self, **kw):
    return None

  async def sample(self, sampling_requests, **kw):
    return []

  async def get_weight_sync_metadata(self, **kw):
    return [{"host": 0}]

  async def bind_weight_sync(self, **kw):
    return None

  async def pre_weight_sync(self, sync_request=None, **kw):
    await self._enter("pre")
    req_id, uuid = self._key(sync_request)
    self.round = {"req_id": req_id, "uuid": uuid, "phase": "prepared"}

  async def weight_sync(self, sync_request=None, **kw):
    await self._enter("weight")
    req_id, uuid = self._key(sync_request)
    self.round = {"req_id": req_id, "uuid": uuid, "phase": "h2d_done"}

  async def post_weight_sync(self, sync_request=None, **kw):
    await self._enter("post")
    req_id, uuid = self._key(sync_request)
    self.round = {"req_id": req_id, "uuid": uuid, "phase": "committed"}

  async def abort_weight_sync(self, sync_request=None, **kw):
    await self._enter("abort")
    req_id, uuid = self._key(sync_request)
    self.round = {"req_id": req_id, "uuid": uuid, "phase": "aborted"}

  async def get_weight_sync_round(self, **kw):
    await self._enter("round")
    if self.round_returns_none:
      return None
    if self.round_error is not None:
      raise self.round_error
    if self.round_override is not None:
      return self.round_override
    return dict(self.round)

  async def get_transfer_status(self, req_id, **kw):
    return "DONE"

  async def migrate_kv_cache(
      self, source_server_id, target_server_id, token_ids, **kw
  ):
    return True

  async def get_load_info(self, **kw):
    return None


class FakeCollector:
  """Matches TrajectoryCollectorEngine's constructor and surface."""

  def __init__(
      self,
      *,
      traj_id,
      request,
      sampler,
      env_client,
      agent,
      tokenizer,
      chat_parser,
  ):
    self.traj_id = traj_id
    self.request = request
    self.env = env_client
    self.is_done = False
    self.is_paused = False

  async def run_episode(self):
    self.is_done = True
    # Trajectory requires a nested `agent`, which these tests never read.
    return trajectory_lib.Trajectory.model_construct(trajectory_id=self.traj_id)

  def pause(self):
    self.is_paused = True

  def resume(self):
    self.is_paused = False

  def cancel(self):
    pass


def ws_req(uuid, req_id="r1", version=1):
  return sampler_lib.WeightSyncRequest(
      policy_version=version,
      extra_config={"req_id": req_id, "uuid": uuid},
  )


class RolloutWorkerWeightSyncTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A task created inside a `with` block can reach its collector
    # construction after the block exits, so the patch has to outlive any
    # single block. The real engine rejects the None sampler/env/agent these
    # tests run without.
    patcher = mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine", FakeCollector
    )
    patcher.start()
    self.addCleanup(patcher.stop)

  def _make_worker(self, sampler):
    worker = rollout_worker.RolloutWorker(
        "w0", sampler=sampler, tokenizer=object(), chat_parser=object()
    )
    worker.initialize()
    return worker

  async def _generate(self, worker, prompt_id):
    """Runs one trivial episode through the worker's own manager.

    Admission is checked where the gate lives, so these tests see exactly
    what a client request would: either a trajectory or the closed-gate
    rejection.
    """
    with mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine", FakeCollector
    ):
      return await worker.manager.generate(
          datatypes.RolloutRequest(prompt_id=prompt_id)
      )

  # ------------------------------------------------------- happy-path gating

  def test_pre_leaves_worker_syncing_not_ready(self):
    asyncio.run(
        asyncio.wait_for(self._pre_leaves_worker_syncing_not_ready(), 30)
    )

  async def _pre_leaves_worker_syncing_not_ready(self):
    worker = self._make_worker(FakeSampler())
    await worker.pre_weight_sync(ws_req(uuid=1))
    # Restoring READY in a finally would report a drained,
    # cache-less worker as healthy while it could not serve.
    self.assertEqual(worker.state, WorkerState.SYNCING)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  def test_full_round_commits_and_restores_ready(self):
    asyncio.run(
        asyncio.wait_for(self._full_round_commits_and_restores_ready(), 30)
    )

  async def _full_round_commits_and_restores_ready(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=7, req_id="r7")
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    self.assertEqual(worker.state, WorkerState.READY)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "committed")
    self.assertEqual(report["uuid"], 7)
    self.assertEqual(report["req_id"], "r7")
    self.assertEqual(sampler.calls, ["pre", "weight", "post"])

  # ------------------------------------------------------ phase-order gating

  def test_post_before_h2d_rejected(self):
    asyncio.run(asyncio.wait_for(self._post_before_h2d_rejected(), 30))

  async def _post_before_h2d_rejected(self):
    worker = self._make_worker(FakeSampler())
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    # Skipping weight_sync would publish staging no H2D ever wrote.
    with self.assertRaisesRegex(RuntimeError, r"weight_sync \(H2D\)"):
      await worker.post_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")
    self.assertEqual(worker.state, WorkerState.SYNCING)

  def test_post_on_fresh_round_rejected(self):
    asyncio.run(asyncio.wait_for(self._post_on_fresh_round_rejected(), 30))

  async def _post_on_fresh_round_rejected(self):
    # The tracker refuses it before the worker's own phase-order check ever
    # runs: only a pre or an abort may open a round, so a post for a round
    # this worker has never seen cannot advance its high-water mark.
    worker = self._make_worker(FakeSampler())
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      await worker.post_weight_sync(ws_req(uuid=1))
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "idle")
    self.assertEqual(worker.state, WorkerState.READY)

  def test_weight_on_fresh_round_rejected(self):
    asyncio.run(asyncio.wait_for(self._weight_on_fresh_round_rejected(), 30))

  async def _weight_on_fresh_round_rejected(self):
    # Same rule for H2D. The SYNCING state check is
    # only reachable once a round is genuinely open, which is what
    # test_stop_during_pre_then_weight_rejected covers.
    worker = self._make_worker(FakeSampler())
    self.assertEqual(worker.state, WorkerState.READY)
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      await worker.weight_sync(ws_req(uuid=1))
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "idle")

  # -------------------------------------------------------- stop() vs phases

  def test_stop_during_post_wins(self):
    asyncio.run(asyncio.wait_for(self._stop_during_post_wins(), 30))

  async def _stop_during_post_wins(self):
    sampler = FakeSampler()
    sampler.gates["post"] = asyncio.Event()
    sampler.started["post"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    task = asyncio.create_task(worker.post_weight_sync(req))
    await sampler.started["post"].wait()
    worker.stop()
    sampler.gates["post"].set()
    # The publish happened but is not serving; recording "committed" here
    # would hide a stopped worker behind a green round.
    with self.assertRaisesRegex(RuntimeError, "not serving"):
      await task
    self.assertEqual(worker.state, WorkerState.STOPPED)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "h2d_done")

  def test_stop_during_weight_then_post_rejected(self):
    asyncio.run(
        asyncio.wait_for(self._stop_during_weight_then_post_rejected(), 30)
    )

  async def _stop_during_weight_then_post_rejected(self):
    sampler = FakeSampler()
    sampler.gates["weight"] = asyncio.Event()
    sampler.started["weight"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    task = asyncio.create_task(worker.weight_sync(req))
    await sampler.started["weight"].wait()
    worker.stop()
    sampler.gates["weight"].set()
    await task  # H2D into staging completes even though the worker stopped
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "h2d_done")
    with self.assertRaisesRegex(RuntimeError, "not serving"):
      await worker.post_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "h2d_done")

  def test_stop_during_pre_then_weight_rejected(self):
    asyncio.run(
        asyncio.wait_for(self._stop_during_pre_then_weight_rejected(), 30)
    )

  async def _stop_during_pre_then_weight_rejected(self):
    sampler = FakeSampler()
    sampler.gates["pre"] = asyncio.Event()
    sampler.started["pre"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    task = asyncio.create_task(worker.pre_weight_sync(req))
    await sampler.started["pre"].wait()
    worker.stop()
    sampler.gates["pre"].set()
    await task  # pre itself returns; the round is "prepared"
    # STOPPED is not SYNCING, so H2D must be refused via the state check.
    with self.assertRaisesRegex(RuntimeError, "pre_weight_sync must run"):
      await worker.weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  def test_stop_during_abort_reconciles_needs_restart(self):
    asyncio.run(
        asyncio.wait_for(self._stop_during_abort_reconciles_needs_restart(), 30)
    )

  async def _stop_during_abort_reconciles_needs_restart(self):
    sampler = FakeSampler()
    sampler.gates["abort"] = asyncio.Event()
    sampler.started["abort"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    task = asyncio.create_task(worker.abort_weight_sync(req))
    await sampler.started["abort"].wait()
    worker.stop()
    sampler.gates["abort"].set()
    # The sampler rolled back (its own report confirms "aborted"), but the
    # worker is STOPPED, not serving: recording "aborted" would tell the
    # coordinator this worker is consistent on the old weights and healthy.
    with self.assertRaisesRegex(RuntimeError, "needs-restart"):
      await task
    self.assertEqual(worker.state, WorkerState.STOPPED)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  # ------------------------------------------------- idempotency + staleness

  def test_duplicate_post_is_noop(self):
    asyncio.run(asyncio.wait_for(self._duplicate_post_is_noop(), 30))

  async def _duplicate_post_is_noop(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    self.assertIsNone(await worker.post_weight_sync(req))
    self.assertEqual(sampler.calls.count("post"), 1)
    self.assertEqual(worker.state, WorkerState.READY)

  def test_duplicate_pre_is_noop(self):
    asyncio.run(asyncio.wait_for(self._duplicate_pre_is_noop(), 30))

  async def _duplicate_pre_is_noop(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    # A retried pre RPC must not re-drain and re-quiesce a sampler that is
    # already quiesced for this round.
    self.assertIsNone(await worker.pre_weight_sync(req))
    self.assertEqual(sampler.calls.count("pre"), 1)
    self.assertEqual(worker.state, WorkerState.SYNCING)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  def test_duplicate_weight_is_noop(self):
    asyncio.run(asyncio.wait_for(self._duplicate_weight_is_noop(), 30))

  async def _duplicate_weight_is_noop(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    self.assertIsNone(await worker.weight_sync(req))  # no second H2D
    self.assertEqual(sampler.calls.count("weight"), 1)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "h2d_done")

  def test_duplicate_abort_is_noop(self):
    asyncio.run(asyncio.wait_for(self._duplicate_abort_is_noop(), 30))

  async def _duplicate_abort_is_noop(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.abort_weight_sync(req)
    self.assertIsNone(await worker.abort_weight_sync(req))
    self.assertEqual(sampler.calls.count("abort"), 1)
    self.assertEqual(worker.state, WorkerState.READY)

  def test_abort_after_commit_is_noop(self):
    asyncio.run(asyncio.wait_for(self._abort_after_commit_is_noop(), 30))

  async def _abort_after_commit_is_noop(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    # A late abort of a committed round is a no-op: the publish stands, and
    # rolling back a publish needs a NEW round, not a late abort.
    self.assertIsNone(await worker.abort_weight_sync(req))
    self.assertNotIn("abort", sampler.calls)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "committed")

  def test_same_uuid_different_req_id_raises(self):
    asyncio.run(asyncio.wait_for(self._same_uuid_different_req_id_raises(), 30))

  async def _same_uuid_different_req_id_raises(self):
    worker = self._make_worker(FakeSampler())
    await worker.pre_weight_sync(ws_req(uuid=1, req_id="r1"))
    # A round key must never be reused: uuid 1 under a different req_id is
    # a retried coordinator restart stepping on the current round.
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      await worker.weight_sync(ws_req(uuid=1, req_id="r2"))
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["req_id"], "r1")
    self.assertEqual(report["phase"], "prepared")

  def test_stale_round_raises(self):
    asyncio.run(asyncio.wait_for(self._stale_round_raises(), 30))

  async def _stale_round_raises(self):
    worker = self._make_worker(FakeSampler())
    req = ws_req(uuid=2)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    with self.assertRaises(weight_sync_coordinator.StaleRoundError):
      await worker.pre_weight_sync(ws_req(uuid=1))
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["uuid"], 2)
    self.assertEqual(report["phase"], "committed")

  # ------------------------------------------------- abort terminal (closed)

  def test_abort_confirmed_records_aborted(self):
    asyncio.run(asyncio.wait_for(self._abort_confirmed_records_aborted(), 30))

  async def _abort_confirmed_records_aborted(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.abort_weight_sync(req)
    self.assertEqual(worker.state, WorkerState.READY)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "aborted")
    # Admission must actually reopen on the old weights.
    with mock.patch.object(
        manager_lib.collector_lib, "TrajectoryCollectorEngine", FakeCollector
    ):
      res = await worker.manager.generate(
          datatypes.RolloutRequest(prompt_id="t1")
      )
    self.assertIsInstance(res, trajectory_lib.Trajectory)

  def test_abort_mismatched_uuid_not_confirmed(self):
    asyncio.run(
        asyncio.wait_for(self._abort_mismatched_uuid_not_confirmed(), 30)
    )

  async def _abort_mismatched_uuid_not_confirmed(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    sampler.round_override = {"req_id": "r1", "uuid": 999, "phase": "aborted"}
    with self.assertRaisesRegex(RuntimeError, "not positively confirmed"):
      await worker.abort_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertNotEqual(report["phase"], "aborted")
    self.assertEqual(report["phase"], "prepared")

  def test_abort_published_phase_not_confirmed(self):
    asyncio.run(
        asyncio.wait_for(self._abort_published_phase_not_confirmed(), 30)
    )

  async def _abort_published_phase_not_confirmed(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    # The sampler says this round PUBLISHED: a publish must never be
    # recorded as an abort, whatever the round key says.
    sampler.round_override = {"req_id": "r1", "uuid": 1, "phase": "published"}
    with self.assertRaisesRegex(RuntimeError, "not positively confirmed"):
      await worker.abort_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  def test_abort_unknown_report_format_not_confirmed(self):
    asyncio.run(
        asyncio.wait_for(self._abort_unknown_report_format_not_confirmed(), 30)
    )

  async def _abort_unknown_report_format_not_confirmed(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    sampler.round_override = "rolled back ok"  # not a dict: fail closed
    with self.assertRaisesRegex(RuntimeError, "not positively confirmed"):
      await worker.abort_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")

  def test_abort_none_report_is_not_confirmed(self):
    asyncio.run(
        asyncio.wait_for(self._abort_none_report_is_not_confirmed(), 30)
    )

  async def _abort_none_report_is_not_confirmed(self):
    # A status endpoint that answers nothing has told us nothing. Skipping
    # the confirmation on a None report would flip the worker to READY and
    # reopen admission over a rollback nobody confirmed -- and the
    # samplerless case that might excuse the skip cannot happen, because
    # RolloutManager's constructor rejects a sampler of None.
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    sampler.round_returns_none = True
    with self.assertRaisesRegex(RuntimeError, "not positively confirmed"):
      await worker.abort_weight_sync(req)
    self.assertEqual(worker.state, WorkerState.SYNCING)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "prepared")
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t1")

  def test_abort_status_rpc_failure_propagates(self):
    asyncio.run(
        asyncio.wait_for(self._abort_status_rpc_failure_propagates(), 30)
    )

  async def _abort_status_rpc_failure_propagates(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    sampler.round_error = RuntimeError("status RPC failed")
    with self.assertRaisesRegex(RuntimeError, "status RPC failed"):
      await worker.abort_weight_sync(req)
    report = await worker.get_weight_sync_round()
    self.assertNotEqual(report["phase"], "aborted")
    # The sampler's state is UNKNOWN after the failed confirmation RPC:
    # flipping to READY here would report a healthy worker over it.
    self.assertEqual(worker.state, WorkerState.SYNCING)

  # ------------------------------------------------- admission gate timing

  def test_admission_stays_closed_while_abort_confirmation_pending(self):
    asyncio.run(
        asyncio.wait_for(
            self._admission_stays_closed_while_abort_confirmation_pending(), 30
        )
    )

  async def _admission_stays_closed_while_abort_confirmation_pending(self):
    sampler = FakeSampler()
    sampler.gates["round"] = asyncio.Event()
    sampler.started["round"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    task = asyncio.create_task(worker.abort_weight_sync(req))
    await asyncio.wait_for(sampler.started["round"].wait(), 5)
    # The sampler's abort returned, but its rollback is not confirmed yet:
    # every unconfirmed outcome (mismatched key, published round, status RPC
    # failure) ends SYNCING + needs-restart, so admitting here would run
    # requests against a sampler that may never come back.
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t1")

    sampler.gates["round"].set()
    await asyncio.wait_for(task, 5)
    self.assertEqual(worker.state, WorkerState.READY)
    res = await self._generate(worker, "t2")
    self.assertIsInstance(res, trajectory_lib.Trajectory)

  def test_admission_stays_closed_when_abort_not_confirmed(self):
    asyncio.run(
        asyncio.wait_for(
            self._admission_stays_closed_when_abort_not_confirmed(), 30
        )
    )

  async def _admission_stays_closed_when_abort_not_confirmed(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    sampler.round_override = {"req_id": "r1", "uuid": 999, "phase": "aborted"}
    with self.assertRaisesRegex(RuntimeError, "not positively confirmed"):
      await worker.abort_weight_sync(req)
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t1")
    self.assertEqual(worker.state, WorkerState.SYNCING)

  def test_admission_stays_closed_when_post_fails_state_check(self):
    asyncio.run(
        asyncio.wait_for(
            self._admission_stays_closed_when_post_fails_state_check(), 30
        )
    )

  async def _admission_stays_closed_when_post_fails_state_check(self):
    sampler = FakeSampler()
    sampler.gates["post"] = asyncio.Event()
    sampler.started["post"] = asyncio.Event()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    task = asyncio.create_task(worker.post_weight_sync(req))
    await asyncio.wait_for(sampler.started["post"].wait(), 5)
    worker.stop()
    sampler.gates["post"].set()
    with self.assertRaisesRegex(RuntimeError, "not serving"):
      await task
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t1")

  def test_admission_stays_closed_when_publish_is_not_serving(self):
    asyncio.run(
        asyncio.wait_for(
            self._admission_stays_closed_when_publish_is_not_serving(), 30
        )
    )

  async def _admission_stays_closed_when_publish_is_not_serving(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)

    publish = worker.manager.post_weight_sync

    async def publish_then_leave_state_unserving(sync_request=None, **kw):
      res = await publish(sync_request, **kw)
      # A transition landing between the publish and the state check. Unlike
      # stop() it leaves the manager itself reopenable, so this pins WHERE
      # the reopen happens rather than the manager's permanent close.
      # ERROR, not PENDING: upstream only allows SYNCING -> READY / STOPPED
      # / ERROR, and the local stub does not validate transitions, so a
      # forbidden one here would pass locally and fail in GOOGLE_INTERNAL_PACKAGE_PATH.
      worker.state = WorkerState.ERROR
      return res

    worker.manager.post_weight_sync = publish_then_leave_state_unserving
    with self.assertRaisesRegex(RuntimeError, "not serving"):
      await worker.post_weight_sync(req)
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t1")
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "h2d_done")

  def test_admission_reopens_after_confirmed_abort(self):
    asyncio.run(
        asyncio.wait_for(self._admission_reopens_after_confirmed_abort(), 30)
    )

  async def _admission_reopens_after_confirmed_abort(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t0")
    await worker.abort_weight_sync(req)
    self.assertEqual(worker.state, WorkerState.READY)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "aborted")
    res = await self._generate(worker, "t1")
    self.assertIsInstance(res, trajectory_lib.Trajectory)

  def test_admission_reopens_after_committed_post(self):
    asyncio.run(
        asyncio.wait_for(self._admission_reopens_after_committed_post(), 30)
    )

  async def _admission_reopens_after_committed_post(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=1)
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    with self.assertRaisesRegex(RuntimeError, "admission is closed"):
      await self._generate(worker, "t0")
    await worker.post_weight_sync(req)
    self.assertEqual(worker.state, WorkerState.READY)
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "committed")
    res = await self._generate(worker, "t1")
    self.assertIsInstance(res, trajectory_lib.Trajectory)

  # ----------------------------------------------------- report authorities

  def test_get_round_is_worker_authoritative(self):
    asyncio.run(asyncio.wait_for(self._get_round_is_worker_authoritative(), 30))

  async def _get_round_is_worker_authoritative(self):
    sampler = FakeSampler()
    worker = self._make_worker(sampler)
    req = ws_req(uuid=3, req_id="r3")
    await worker.pre_weight_sync(req)
    await worker.weight_sync(req)
    await worker.post_weight_sync(req)
    # Garbage in the sampler's sub-state must not leak into the report the
    # coordinator reconciles failed RPCs against.
    sampler.round = "garbage"
    sampler.round_override = "garbage"
    report = await worker.get_weight_sync_round()
    self.assertEqual(report["phase"], "committed")
    self.assertEqual(report["uuid"], 3)
    self.assertEqual(report["req_id"], "r3")


if __name__ == "__main__":
  absltest.main()
