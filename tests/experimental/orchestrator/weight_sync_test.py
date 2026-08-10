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

The Raiden controller, server, and client facade are mocked out, so this needs
no TPU and no network. Under test is the handler's contract: what it forwards
to the controller, which addresses it dials versus advertises, and how it
reports transfer outcomes.
"""

from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest import mock

from absl.testing import absltest

try:
  from tunix.experimental.orchestrator import weight_sync
except ImportError:
  raise unittest.SkipTest("tpu_raiden is required")


RaidenId = weight_sync.RaidenId

SRC = RaidenId(job_name="trainer", job_replica_id="0")
DST = RaidenId(job_name="sampler", job_replica_id="0")


def make_metadata(
    unit: RaidenId,
    shards: tuple[str, ...] = ("10.0.0.1:20000",),
    variables: tuple[weight_sync.VariableMetadata, ...] = (),
    **overrides: Any,
) -> weight_sync.RaidenWorkUnitMetadata:
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
  fields.update(overrides)
  return weight_sync.RaidenWorkUnitMetadata(**fields)


class RaidenHandlerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    controller_patch = mock.patch.object(
        weight_sync.raiden_controller, "RaidenController", autospec=True
    )
    server_patch = mock.patch.object(
        weight_sync.raiden_controller, "RaidenControllerServer", autospec=True
    )
    client_patch = mock.patch.object(
        weight_sync.raiden_controller,
        "RaidenControllerClientFacade",
        autospec=True,
    )
    rpc_client_patch = mock.patch.object(
        weight_sync.raiden_controller,
        "WeightSyncWorkerRpcClient",
        autospec=True,
    )
    self.addCleanup(mock.patch.stopall)
    self.controller_cls = controller_patch.start()
    self.server_cls = server_patch.start()
    self.client_cls = client_patch.start()
    self.rpc_client_cls = rpc_client_patch.start()

    self.server_cls.return_value.start.return_value = 15000
    self.client = self.client_cls.return_value
    # `transfer` verifies completion after the blocking call returns; without
    # this the mock's default return compares unequal and success is False.
    self.client.get_transfer_status.return_value = (
        weight_sync.controller_service_pb2.GetTransferStatusResponse.STATUS_COMPLETED
    )

    self.handler = weight_sync.RaidenHandler(port=0)

  # ------------------------------------------------------------- addresses

  def test_self_dial_uses_ipv6_loopback(self):
    # The controller's listener can come up IPv6-only (its IPV6_V6ONLY
    # setsockopt is wrapped in a swallowing try/except); dialing 127.0.0.1
    # then times out instead of failing fast. Probe A hit exactly this.
    self.assertEqual(self.handler.dial_address, "[::1]:15000")
    self.assertEqual(self.client_cls.call_args.args[0], "[::1]:15000")

  def test_advertised_address_defaults_to_dial_address(self):
    self.assertEqual(self.handler.advertised_address, "[::1]:15000")

  def test_explicit_advertised_address_is_kept_separate_from_dial(self):
    self.client_cls.reset_mock()
    handler = weight_sync.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    # Workers are told the advertised name; the handler still dials loopback.
    self.assertEqual(
        handler.advertised_address, "controller-a.example:10019"
    )
    self.assertEqual(handler.dial_address, "[::1]:15000")
    self.assertEqual(self.client_cls.call_args.args[0], "[::1]:15000")

  def test_transfer_never_defaults_controller_peer_addresses(self):
    # The controller treats a non-empty destination controller address as a
    # REMOTE peer and synchronously RPCs it; defaulting it to our own
    # address self-deadlocks coordinate_transfer (same req_id, circular
    # future wait, 300s facade timeout). Single-controller transfers must
    # carry NO peer addresses.
    handler = weight_sync.RaidenHandler(
        port=0, advertised_address="controller-a.example:10019"
    )
    handler.register_work_unit(make_metadata(SRC))
    handler.register_work_unit(make_metadata(DST))

    handler.transfer(src_units=[SRC], dst_units=[DST], expected_block_count=4)

    kwargs = self.client.coordinate_transfer.call_args.kwargs
    self.assertNotIn("src_controller_address", kwargs)
    self.assertNotIn("dst_controller_address", kwargs)

  def test_transfer_rejects_self_addressed_controller_peer(self):
    handler = weight_sync.RaidenHandler(
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
    self.client.coordinate_transfer.assert_not_called()

  def test_transfer_passes_a_genuinely_remote_controller_peer_through(self):
    # Dual-controller topologies pass the OTHER controller's address; that
    # must flow to the wire untouched.
    handler = weight_sync.RaidenHandler(
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

    kwargs = self.client.coordinate_transfer.call_args.kwargs
    self.assertEqual(
        kwargs["dst_controller_address"],
        "controller-b.example:10019",
    )

  def test_transfer_rpc_timeout_raises_outcome_unknown(self):
    # A client-side RPC timeout means the reply is lost, not that the
    # transfer failed: the controller may still be executing it. Reporting
    # failure would invite a rollback into buffers a live transfer may be
    # writing.
    self._register_both()
    self.client.coordinate_transfer.side_effect = TimeoutError("recv timeout")

    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )

  def test_resolver_reaches_the_controllers_outbound_client(self):
    # The facade resolving names is not enough: the controller itself makes
    # outbound calls to worker control-plane addresses through its
    # WeightSyncWorkerRpcClient, and that client needs the resolver too.
    resolver = object()
    weight_sync.RaidenHandler(port=0, name_resolver=resolver)

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

  def test_register_forwards_every_proto_field(self):
    self.handler.register_work_unit(make_metadata(SRC))

    self.client.register_work_unit.assert_called_once()
    kwargs = self.client.register_work_unit.call_args.kwargs
    self.assertEqual(kwargs["unit"], SRC)
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
    self.client.register_work_unit.assert_not_called()

  def test_reregistering_a_unit_replaces_rather_than_accumulates(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.handler.register_work_unit(
        make_metadata(SRC, shards=("10.0.0.1:20099",))
    )
    self.assertEqual(self.handler.registered_units, frozenset({SRC}))

  def test_variables_manifest_is_converted_to_protos(self):
    variables = (
        weight_sync.VariableMetadata(
            name="layer_0", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
            item_size=4, layer_idx=0,
        ),
        weight_sync.VariableMetadata(
            name="layer_1", shape=(8, 4), mesh_shape=(1, 1), layout=(1, 0),
            item_size=4, layer_idx=1,
        ),
    )
    self.handler.register_work_unit(make_metadata(SRC, variables=variables))

    sent = self.client.register_work_unit.call_args.kwargs["variables"]
    self.assertEqual([v.name for v in sent], ["layer_0", "layer_1"])
    self.assertEqual([v.layer_idx for v in sent], [0, 1])

  def test_no_variables_sends_none_rather_than_an_empty_list(self):
    self.handler.register_work_unit(make_metadata(SRC))
    self.assertIsNone(
        self.client.register_work_unit.call_args.kwargs["variables"]
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
    kwargs = self.client.coordinate_transfer.call_args.kwargs
    self.assertEqual(kwargs["src_units"], [SRC])
    self.assertEqual(kwargs["dst_units"], [DST])
    self.assertEqual(kwargs["expected_block_count"], 4)

  def test_caller_uuid_overrides_the_handler_default(self):
    self._register_both()

    self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4, uuid=77
    )

    kwargs = self.client.coordinate_transfer.call_args.kwargs
    self.assertEqual(kwargs["uuid"], 77)
    self.client.get_transfer_status.assert_called_with(mock.ANY, uuid=77)

  def test_zero_expected_block_count_defers_to_the_controller(self):
    # 0 is not an error: the controller's direct-schedule path derives the
    # per-destination push count from its own schedule, which is the only
    # authoritative source. The 0 must reach the controller untouched.
    self._register_both()

    self.handler.transfer(src_units=[SRC], dst_units=[DST])

    kwargs = self.client.coordinate_transfer.call_args.kwargs
    self.assertEqual(kwargs["expected_block_count"], 0)

  def test_negative_expected_block_count_is_refused(self):
    self._register_both()

    with self.assertRaisesRegex(ValueError, "expected_block_count"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=-1
      )
    self.client.coordinate_transfer.assert_not_called()

  def test_transfer_refuses_unregistered_units(self):
    self.handler.register_work_unit(make_metadata(SRC))
    with self.assertRaisesRegex(ValueError, "unregistered"):
      self.handler.transfer(
          src_units=[SRC], dst_units=[DST], expected_block_count=4
      )
    self.client.coordinate_transfer.assert_not_called()

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

  def test_transport_failure_is_reported_not_raised(self):
    self._register_both()
    self.client.coordinate_transfer.side_effect = RuntimeError(
        "Timeout (300.0s) failed to connect"
    )

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertFalse(result.success)
    self.assertIn("Timeout", result.message)

  def test_incomplete_status_is_a_failure(self):
    self._register_both()
    self.client.get_transfer_status.return_value = 1  # in progress

    result = self.handler.transfer(
        src_units=[SRC], dst_units=[DST], expected_block_count=4
    )

    self.assertFalse(result.success)
    self.assertIn("status 1", result.message)


if __name__ == "__main__":
  absltest.main()
