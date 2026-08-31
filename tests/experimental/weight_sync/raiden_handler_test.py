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

"""Unit tests for the Raiden weight sync handler.

The Raiden controller and its listener are mocked out, so this needs no TPU
and no network. Under test is the handler's contract: what it forwards to the
controller, what it advertises, how it drives a transfer future, and how it
reports transfer outcomes.

The controller mock is `autospec`ed, which is load-bearing rather than
stylistic: the handler calls the controller directly now, so a kwarg the real
`RaidenController` would reject has to fail here too.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest import mock

from absl.testing import absltest

from tunix.experimental.weight_sync import raiden_handler
from tunix.experimental.weight_sync import weight_sync


WorkUnitId = weight_sync.WorkUnitId

SRC = WorkUnitId(
    job_name="trainer", job_replica_id="0", data_name="weights",
    data_replica_idx=3,
)
DST = WorkUnitId(
    job_name="sampler", job_replica_id="0", data_name="weights",
    data_replica_idx=4,
)
RAIDEN_SRC = raiden_handler.raiden_controller.RaidenId(
    "trainer", "0", "weights", 3
)
RAIDEN_DST = raiden_handler.raiden_controller.RaidenId(
    "sampler", "0", "weights", 4
)


def make_metadata(
    unit: WorkUnitId,
    shards: tuple[str, ...] = ("10.0.0.1:20000",),
    variables: tuple[weight_sync.TensorMetadata, ...] = (),
    **overrides: Any,
) -> weight_sync.WorkUnitMetadata:
  fields: dict[str, Any] = dict(
      unit=unit,
      shards=shards,
      control_plane_rpc_address="10.0.0.1:20001",
      global_shape=(1024, 1024),
      mesh_shape=(4, 1),
      layout=(1, 0),
      item_size=4,
      variables=variables,
  )
  if variables:
    fields["mesh_axes"] = ("x", "y")
  fields.update(overrides)
  return weight_sync.WorkUnitMetadata(**fields)


class _FakeFuture:
  """Stands in for `RaidenFuture` and records how it was driven.

  A real `start_transfer` only builds the coroutine; whoever wins `try_start`
  has to run it, and `wait_threadsafe` alone would block forever because the
  event it waits on is set inside `wait`. Recording which path ran is how the
  tests pin that down.
  """

  def __init__(self, already_started: bool = False, exc: Any = None):
    self._started = already_started
    self._exc = exc
    self.waited_async = False
    self.waited_threadsafe = False

  def try_start(self) -> bool:
    if self._started:
      return False
    self._started = True
    return True

  async def wait(self) -> None:
    self.waited_async = True
    if self._exc:
      raise self._exc

  def wait_threadsafe(self, timeout: Optional[float] = None) -> None:
    self.waited_threadsafe = True
    if self._exc:
      raise self._exc


class RaidenHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    controller_patch = mock.patch.object(
        raiden_handler.raiden_controller, "RaidenController", autospec=True
    )
    server_patch = mock.patch.object(
        raiden_handler.raiden_controller, "RaidenControllerServer", autospec=True
    )
    rpc_client_patch = mock.patch.object(
        raiden_handler.raiden_controller,
        "WeightSyncWorkerRpcClient",
        autospec=True,
    )
    self.controller_cls = self.enter_context(controller_patch)
    self.server_cls = self.enter_context(server_patch)
    self.rpc_client_cls = self.enter_context(rpc_client_patch)

    self.server_cls.return_value.start.return_value = 15000
    self.controller = self.controller_cls.return_value

    # A fresh future per call, kept so a test can assert which way it was
    # driven. `start_transfer` returning a bare Mock would not do: the
    # handler awaits `wait()`, and a Mock is not awaitable.
    self.futures: list[_FakeFuture] = []

    def _new_future(*args: Any, **kwargs: Any) -> _FakeFuture:
      del args, kwargs
      future = _FakeFuture()
      self.futures.append(future)
      return future

    self.controller.start_transfer.side_effect = _new_future
    # `transfer` verifies completion after the future resolves; without this
    # the mock's default return compares unequal and success is False.
    self.controller.get_transfer_status.return_value = (
        raiden_handler.controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
    )

    self.handler = raiden_handler.RaidenHandler(port=0)

  # ------------------------------------------------------------- addresses

  def test_loopback_address_uses_ipv6(self):
    # 127.0.0.1 would be wrong even though nothing dials this: the listener
    # can come up IPv6-only (its IPV6_V6ONLY setsockopt is wrapped in a
    # swallowing try/except), so the v4 spelling is not an alias for this
    # controller and would not be recognized as a self-addressed peer.
    self.assertEqual(self.handler.loopback_address, "[::1]:15000")

  def test_advertised_address_defaults_to_the_loopback_address(self):
    self.assertEqual(self.handler.advertised_address, "[::1]:15000")

  def test_explicit_advertised_address_is_kept_separate(self):
    handler = raiden_handler.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    # Remote controller callers use the advertised name; the loopback spelling
    # stays as a second alias for the self-peer check.
    self.assertEqual(
        handler.advertised_address, "controller-a.example:10019"
    )
    self.assertEqual(handler.loopback_address, "[::1]:15000")

  def test_no_client_stub_is_constructed(self):
    # The handler owns the controller object, so reaching it through
    # RaidenControllerClientFacade would serialize a proto and open a TCP
    # connection to our own listener to call something already in hand.
    with mock.patch.object(
        raiden_handler.raiden_controller,
        "RaidenControllerClientFacade",
        autospec=True,
    ) as facade_cls:
      raiden_handler.RaidenHandler(port=0)
    facade_cls.assert_not_called()

  def test_transfer_never_defaults_controller_peer_addresses(self):
    # The controller treats a non-empty destination controller address as a
    # REMOTE peer and synchronously RPCs it; defaulting it to our own
    # address self-deadlocks coordinate_transfer (same req_id, circular
    # future wait, 300s facade timeout). Single-controller transfers must
    # carry NO peer addresses.
    handler = raiden_handler.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(src_units=[SRC], dst_units=[DST], expected_block_count=4)

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertNotIn("src_controller_address", kwargs)
    self.assertNotIn("dst_controller_address", kwargs)

  def test_transfer_rejects_self_addressed_controller_peer(self):
    handler = raiden_handler.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    with self.assertRaisesRegex(ValueError, "deadlock"):
      handler.transfer(
          src_units=[SRC],
          dst_units=[DST],
          expected_block_count=4,
          dst_controller_address="controller-a.example:10019",
      )
    self.controller.start_transfer.assert_not_called()

  def test_transfer_passes_a_genuinely_remote_controller_peer_through(self):
    # Dual-controller topologies pass the OTHER controller's address; that
    # must flow to the wire untouched.
    handler = raiden_handler.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(
        src_units=[SRC],
        dst_units=[DST],
        expected_block_count=4,
        dst_controller_address="controller-b.example:10019",
    )

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertEqual(
        kwargs["dst_controller_address"],
        "controller-b.example:10019",
    )

  def test_start_transfer_timeout_before_future_is_known_failure(self):
    # Direct start_transfer only plans and returns a lazy future. Before that
    # future exists no worker push has started, so this is a known failure.
    self._register_both()
    self.controller.start_transfer.side_effect = TimeoutError("recv timeout")

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertFalse(result.success)
    self.assertIn("recv timeout", result.message)

  def test_resolver_reaches_the_controllers_outbound_client(self):
    # The facade resolving names is not enough: the controller itself makes
    # outbound calls to worker control-plane addresses through its
    # WeightSyncWorkerRpcClient, and that client needs the resolver too.
    resolver = object()
    raiden_handler.RaidenHandler(port=0, name_resolver=resolver)

    self.rpc_client_cls.assert_called_once_with(name_resolver=resolver)
    self.assertIs(
        self.controller_cls.call_args.kwargs["worker_rpc_client"],
        self.rpc_client_cls.return_value,
    )

  def test_no_resolver_means_no_worker_rpc_client_override(self):
    self.assertIsNone(
        self.controller_cls.call_args.kwargs["worker_rpc_client"]
    )

  # ---------------------------------------------------------- registration

  def test_facade_hides_the_raiden_native_surface(self):
    self.assertIsInstance(self.handler, weight_sync.WeightSyncHandler)
    self.assertFalse(hasattr(self.handler, "_controller"))
    self.assertFalse(hasattr(self.handler, "registered_metadata"))
    self.assertFalse(hasattr(self.handler, "transfer_status"))

  def test_register_forwards_every_proto_field(self):
    self.handler.register_work_unit(make_metadata(SRC))

    self.controller.register_work_unit.assert_called_once()
    kwargs = self.controller.register_work_unit.call_args.kwargs
    self.assertEqual(kwargs["unit"], RAIDEN_SRC)
    self.assertEqual(kwargs["shards"], ["10.0.0.1:20000"])
    self.assertEqual(kwargs["control_plane_rpc_address"], "10.0.0.1:20001")
    self.assertEqual(kwargs["global_shape"], (1024, 1024))
    self.assertEqual(kwargs["mesh_shape"], (4, 1))
    self.assertEqual(kwargs["layout"], (1, 0))
    self.assertEqual(kwargs["itemsize"], 4)
    self.assertIn(SRC, self.handler.registered_units)

  def test_register_rejects_a_unit_without_a_data_address(self):
    # The synchronizer assigns ports on construction; registering beforehand
    # would publish an address that does not exist yet.
    with self.assertRaisesRegex(ValueError, "data-plane address"):
      self.handler.register_work_unit(make_metadata(SRC, shards=()))
    self.controller.register_work_unit.assert_not_called()

  def test_reregistering_a_unit_replaces_rather_than_accumulates(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.handler.register_work_unit(
        make_metadata(SRC, shards=("10.0.0.1:20099",))
    )
    self.assertEqual(self.handler.registered_units, frozenset({SRC}))

  def test_variables_manifest_is_converted_to_protos(self):
    variables = (
        weight_sync.TensorMetadata(
            name="layer_0", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
            item_size=4, layer_idx=0, sharding_spec=("", ""),
        ),
        weight_sync.TensorMetadata(
            name="layer_1", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
            item_size=4, layer_idx=1, sharding_spec=("", ""),
        ),
    )
    self.handler.register_work_unit(make_metadata(SRC, variables=variables))

    sent = self.controller.register_work_unit.call_args.kwargs["variables"]
    self.assertEqual([v.name for v in sent], ["layer_0", "layer_1"])
    self.assertEqual([v.layer_idx for v in sent], [0, 1])

  def test_mesh_axes_are_forwarded(self):
    # Without mesh_axes (and a variable's sharding_spec) the controller falls
    # back to inferring the host axis from mesh dimension SIZES, which is
    # ambiguous whenever two dimensions match and reports the miss as a log
    # warning rather than an error.
    self.handler.register_work_unit(
        make_metadata(SRC, mesh_axes=("fsdp", "tp"))
    )

    kwargs = self.controller.register_work_unit.call_args.kwargs
    self.assertEqual(kwargs["mesh_axes"], ["fsdp", "tp"])

  def test_absent_mesh_axes_send_none_rather_than_an_empty_list(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.assertIsNone(
        self.controller.register_work_unit.call_args.kwargs["mesh_axes"]
    )

  def test_pool_reshard_fields_are_never_sent(self):
    # The controller requires pool_manifest, layout_fingerprint, page_tokens,
    # transfer_parallelism and transfer_rank together or not at all. Weight
    # sync does byte-span reshard, so sending any of them can only produce a
    # registration that raises.
    self.handler.register_work_unit(make_metadata(SRC))

    kwargs = self.controller.register_work_unit.call_args.kwargs
    for field in (
        "pool_manifest",
        "layout_fingerprint",
        "page_tokens",
        "transfer_parallelism",
        "transfer_rank",
    ):
      self.assertNotIn(field, kwargs)

  def test_sharding_spec_reaches_the_variable_proto(self):
    variables = (
        weight_sync.TensorMetadata(
            # Raiden uses -1 for a replicated layout dimension; this is a
            # supported partial layout, not an invalid permutation.
            name="w", shape=(8, 4), mesh_shape=(1, 1), layout=(-1, 0),
            item_size=4, sharding_spec=("", "y"),
        ),
    )
    self.handler.register_work_unit(make_metadata(SRC, variables=variables))

    sent = self.controller.register_work_unit.call_args.kwargs["variables"]
    self.assertEqual(sent[0].sharding_spec, ["", "y"])
    self.assertEqual(sent[0].layout, [-1, 0])

  def test_variables_require_explicit_mesh_axes_and_sharding_spec(self):
    variable = weight_sync.TensorMetadata(
        name="w", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
        item_size=4,
    )

    with self.assertRaisesRegex(ValueError, "mesh_shape and mesh_axes"):
      self.handler.register_work_unit(
          make_metadata(SRC, variables=(variable,), mesh_axes=None)
      )
    with self.assertRaisesRegex(ValueError, "provide sharding_spec"):
      self.handler.register_work_unit(
          make_metadata(SRC, variables=(variable,))
      )

  def test_variable_axis_sizes_must_match_physical_mesh(self):
    variable = weight_sync.TensorMetadata(
        name="w", shape=(8, 4), mesh_shape=(1, 2), layout=(1, 0),
        item_size=4, sharding_spec=("", "y"),
    )

    with self.assertRaisesRegex(ValueError, "physical mesh has size 1"):
      self.handler.register_work_unit(
          make_metadata(SRC, variables=(variable,))
      )

  def test_no_variables_sends_none_rather_than_an_empty_list(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.assertIsNone(
        self.controller.register_work_unit.call_args.kwargs["variables"]
    )

  # -------------------------------------------------------------- transfer

  def _register_both(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.handler.register_work_unit(make_metadata(DST))

  def test_transfer_passes_units_and_expected_block_count(self):
    self._register_both()

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertTrue(result.success)
    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertEqual(kwargs["src_units"], [RAIDEN_SRC])
    self.assertEqual(kwargs["dst_units"], [RAIDEN_DST])
    self.assertEqual(kwargs["expected_block_count"], 4)

  def test_caller_uuid_overrides_the_handler_default(self):
    self._register_both()

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4, generation=77
    )

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertEqual(kwargs["uuid"], 77)

  def test_status_lookup_is_keyed_on_req_id_alone(self):
    # The controller's active-task table is keyed on req_id, so a uuid
    # argument here would imply a scoping that does not exist.
    self._register_both()

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4, generation=77
    )

    self.controller.get_transfer_status.assert_called_once_with(result.req_id)

  def test_zero_expected_block_count_defers_to_the_controller(self):
    # 0 is not an error: the controller's direct-schedule path derives the
    # per-destination push count from its own schedule, which is the only
    # authoritative source. The 0 must reach the controller untouched.
    self._register_both()

    self.handler.transfer(src_units=[SRC], dst_units=[DST])

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertEqual(kwargs["expected_block_count"], 0)

  def test_negative_expected_block_count_is_refused(self):
    self._register_both()

    with self.assertRaisesRegex(ValueError, "expected_block_count"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=-1
      )
    self.controller.start_transfer.assert_not_called()

  def test_transfer_refuses_unregistered_units(self):
    self.handler.register_work_unit(make_metadata(SRC))
    with self.assertRaisesRegex(ValueError, "unregistered"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )
    self.controller.start_transfer.assert_not_called()

  def test_transfer_ids_are_unique_per_call(self):
    self._register_both()

    first = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )
    second = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertNotEqual(first.req_id, second.req_id)

  def test_caller_supplied_transfer_id_is_honoured(self):
    self._register_both()

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], req_id="step-7",
        expected_block_count=4,
    )

    self.assertEqual(result.req_id, "step-7")

  def test_req_id_cannot_be_reused_with_a_different_uuid(self):
    self._register_both()
    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], req_id="step-7", generation=7,
        expected_block_count=4,
    )

    with self.assertRaisesRegex(ValueError, "already bound to uuid 7"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], req_id="step-7", generation=8,
          expected_block_count=4,
      )

  def test_auto_req_id_skips_a_caller_reserved_name(self):
    self._register_both()
    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], req_id="wsync-1",
        expected_block_count=4,
    )

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertEqual(result.req_id, "wsync-2")

  def test_transport_failure_is_reported_not_raised(self):
    self._register_both()
    self.controller.start_transfer.side_effect = RuntimeError(
        "Timeout (300.0s) failed to connect"
    )

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertFalse(result.success)
    self.assertIn("Timeout", result.message)

  def test_the_transfer_future_is_actually_driven(self):
    # `start_transfer` only builds the coroutine. Whoever wins try_start has
    # to run it; waiting without running would block forever, because the
    # event wait_threadsafe blocks on is set inside wait().
    self._register_both()

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertLen(self.futures, 1)
    self.assertTrue(self.futures[0].waited_async)
    self.assertFalse(self.futures[0].waited_threadsafe)

  def test_a_future_someone_else_started_is_waited_on_not_driven(self):
    # Driving a coroutine a second time would re-run the transfer.
    self._register_both()
    already_running = _FakeFuture(already_started=True)
    self.controller.start_transfer.side_effect = None
    self.controller.start_transfer.return_value = already_running

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertTrue(already_running.waited_threadsafe)
    self.assertFalse(already_running.waited_async)

  def test_exception_driving_a_created_future_is_outcome_unknown(self):
    self._register_both()
    future = _FakeFuture(exc=RuntimeError("worker reset after push"))
    self.controller.start_transfer.side_effect = None
    self.controller.start_transfer.return_value = future

    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )

  def test_exception_waiting_on_another_driver_is_outcome_unknown(self):
    self._register_both()
    future = _FakeFuture(
        already_started=True, exc=RuntimeError("driver disappeared")
    )
    self.controller.start_transfer.side_effect = None
    self.controller.start_transfer.return_value = future

    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )

  def test_final_status_query_failure_is_outcome_unknown(self):
    self._register_both()
    self.controller.get_transfer_status.side_effect = RuntimeError(
        "status channel reset"
    )

    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )

  def test_loop_creation_failure_happens_before_future_is_requested(self):
    self._register_both()

    with mock.patch.object(
        raiden_handler.asyncio,
        "new_event_loop",
        side_effect=RuntimeError("cannot allocate loop"),
    ):
      with self.assertRaisesRegex(RuntimeError, "cannot allocate loop"):
        self.handler.transfer(
            src_units=[SRC], dst_units=[DST], expected_block_count=4
        )

    self.controller.start_transfer.assert_not_called()

  def test_parallelism_is_passed_to_the_transfer(self):
    # It belongs on start_transfer, not on the work-unit registration, where
    # it would be read as pool-reshard metadata.
    handler = raiden_handler.RaidenHandler(port=0, transfer_parallelism=8)
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertEqual(
        self.controller.start_transfer.call_args.kwargs["parallelism"], 8
    )

  def test_handler_owned_options_reach_raiden_without_the_coordinator(self):
    handler = raiden_handler.RaidenHandler(
        port=0,
        transfer_options=raiden_handler.RaidenTransferOptions(
            parallelism=3,
            expected_block_count=7,
            skip_d2h=True,
            skip_tiling={0: True},
            group_size=2,
        ),
    )
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(src_units=[SRC], dst_units=[DST])

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertEqual(kwargs["parallelism"], 3)
    self.assertEqual(kwargs["expected_block_count"], 7)
    self.assertTrue(kwargs["skip_d2h"])
    self.assertEqual(kwargs["skip_tiling"], {0: True})
    self.assertEqual(kwargs["group_size"], 2)

  def test_per_call_parallelism_overrides_handler_default(self):
    handler = raiden_handler.RaidenHandler(port=0, transfer_parallelism=8)
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4,
        parallelism=3,
    )

    self.assertEqual(
        self.controller.start_transfer.call_args.kwargs["parallelism"], 3
    )

  def test_d2h_and_tiling_knobs_default_to_controller_inference(self):
    # skip_tiling=None asks the controller to infer the map; it is not the
    # same as an explicitly empty map. group_size 1 gives every layer its own
    # routing key.
    self._register_both()

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    kwargs = self.controller.start_transfer.call_args.kwargs
    self.assertFalse(kwargs["skip_d2h"])
    self.assertIsNone(kwargs["skip_tiling"])
    self.assertEqual(kwargs["group_size"], 1)

  def test_a_source_with_a_staging_contract_can_skip_controller_d2h(self):
    # This is safe only when source staging and tiling format are explicit.
    self._register_both()

    self.handler.transfer(
        src_units=[SRC],
        dst_units=[DST],
        expected_block_count=4,
        skip_d2h=True,
        skip_tiling={0: True},
    )

    self.assertTrue(
        self.controller.start_transfer.call_args.kwargs["skip_d2h"]
    )
    self.assertEqual(
        self.controller.start_transfer.call_args.kwargs["skip_tiling"],
        {0: True},
    )

  def test_skip_d2h_without_an_explicit_tiling_contract_is_rejected(self):
    self._register_both()

    with self.assertRaisesRegex(ValueError, "explicit skip_tiling"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4,
          skip_d2h=True,
      )

    self.controller.start_transfer.assert_not_called()

  def test_explicit_empty_skip_tiling_map_is_forwarded(self):
    self._register_both()

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4,
        skip_tiling={},
    )

    self.assertEqual(
        self.controller.start_transfer.call_args.kwargs["skip_tiling"], {}
    )

  def test_incomplete_status_is_outcome_unknown(self):
    self._register_both()
    self.controller.get_transfer_status.return_value = (
        raiden_handler.controller_service_pb2.GetTransferStatusResponse.STATUS_IN_PROGRESS
    )

    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )

  def test_failed_status_is_known_failure(self):
    self._register_both()
    self.controller.get_transfer_status.return_value = (
        raiden_handler.controller_service_pb2.GetTransferStatusResponse.STATUS_FAILED
    )

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertFalse(result.success)
    self.assertIn("STATUS_FAILED", result.message)


if __name__ == "__main__":
  absltest.main()