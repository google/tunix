#!/usr/bin/env python3

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

"""Direct Raiden Pathways repro without Tunix wrappers.

This script mirrors the attached internal flow at the API level:

1. Separate controller and worker roles.
2. Source uses FFI `init_weight_synchronizer_and_d2h`.
3. Destination uses FFI `init_weight_synchronizer` and `multi_h2d`.
4. Work-unit metadata is registered directly via `RaidenControllerClientFacade`.
5. Transfer is launched via `RaidenController.start_transfer`.

It intentionally avoids `RaidenSynchronizer` and `RaidenHandler` so a failure here
isolates the lower-level Raiden/FFI path rather than the Tunix wrappers.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import os
import socket
import sys
import time

if "--FLAGS_pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
  sys.argv.append("--FLAGS_pathways_enforce_subset_devices_form_subslice=false")

from absl import logging
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from tpu_sync.frameworks.jax import weight_synchronizer_ffi as raiden_ffi
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2

P = jax.sharding.PartitionSpec


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--role",
      required=True,
      choices=("source", "destination", "controller_src", "controller_dst"),
  )
  parser.add_argument("--controller_address", default="127.0.0.1:10019")
  parser.add_argument("--dst_controller_address")
  parser.add_argument("--pathways_target")
  parser.add_argument("--parallelism", type=int, default=4)
  parser.add_argument("--group_size", type=int, default=1)
  parser.add_argument("--num_src_hosts", type=int, default=4)
  parser.add_argument("--num_dst_hosts", type=int, default=4)
  parser.add_argument("--req_id", default="pathways_direct_api_repro")
  parser.add_argument("--log_level", default="INFO")
  args, _ = parser.parse_known_args(argv)
  return args


def _initialize_pathways_runtime(args: argparse.Namespace) -> None:
  if args.pathways_target:
    os_environ = {
        "JAX_PLATFORMS": "proxy,cpu",
        "JAX_BACKEND_TARGET": args.pathways_target,
    }
    for key, value in os_environ.items():
      if key not in __import__("os").environ:
        __import__("os").environ[key] = value
  os_mod = __import__("os")
  if "proxy" in os_mod.environ.get("JAX_PLATFORMS", "") and os_mod.environ.get(
      "JAX_BACKEND_TARGET"
  ):
    import pathwaysutils

    logging.info(
        "Initializing Pathways runtime via pathwaysutils for %s",
        os_mod.environ["JAX_BACKEND_TARGET"],
    )
    pathwaysutils.initialize()


def _specs() -> list[tuple[tuple[int, ...], jax.sharding.PartitionSpec, str]]:
  return [
      ((8, 8), P("fsdp", "tp"), "block.w_in"),
      ((8, 8), P("tp", "fsdp"), "block.w_out"),
      ((8,), P("fsdp"), "block.norm"),
  ]


def _resolve_local_ip() -> str:
  ip = "127.0.0.1"
  for family, probe in (
      (socket.AF_INET, ("10.255.255.255", 1)),
      (socket.AF_INET6, ("2001:4860:4860::8888", 1)),
  ):
    try:
      sock = socket.socket(family, socket.SOCK_DGRAM)
      try:
        sock.connect(probe)
        ip = sock.getsockname()[0]
      finally:
        sock.close()
      break
    except OSError:
      continue
  return f"[{ip}]" if ":" in ip else ip


def _unpack_ip(row) -> str:
  raw_bytes = b"".join(
      int(x).to_bytes(4, byteorder="little", signed=True) for x in row[:4]
  )
  if raw_bytes[:10] == b"\x00" * 10 and raw_bytes[10:12] == b"\xff\xff":
    return str(ipaddress.IPv4Address(raw_bytes[12:16]))
  addr_str = str(ipaddress.IPv6Address(raw_bytes))
  return f"[{addr_str}]" if ":" in addr_str else addr_str


def _ping_port(addr: str) -> bool:
  ip, port = addr.rsplit(":", 1)
  if ip.startswith("[") and ip.endswith("]"):
    ip = ip[1:-1]
  try:
    sock = socket.create_connection((ip, int(port)), timeout=1)
    sock.close()
    return True
  except OSError:
    return False


def _devices_per_host(devices: np.ndarray) -> int:
  flat = list(devices.flatten())
  num_processes = len(set(getattr(d, "process_index", 0) for d in flat))
  return len(flat) // max(1, num_processes)


def _format_mesh_devices(mesh: jax.sharding.Mesh) -> list[dict[str, object]]:
  formatted = []
  for device in mesh.devices.flatten():
    formatted.append(
        {
            "id": int(device.id),
            "process_index": int(getattr(device, "process_index", -1)),
            "coords": tuple(getattr(device, "coords", ())),
            "slice_index": int(getattr(device, "slice_index", 0)),
        }
    )
  return formatted


def _format_ws_info_rows(gathered_ws_info: np.ndarray) -> list[dict[str, object]]:
  rows = []
  for idx, row in enumerate(gathered_ws_info.tolist()):
    rows.append(
        {
            "row_idx": idx,
            "raw": row,
            "worker_address": f"{_unpack_ip(row)}:{int(row[4])}",
            "listener_address": f"{_unpack_ip(row)}:{int(row[5])}",
        }
    )
  return rows


def _format_variable_metadata(
    protos: list[raiden_service_pb2.VariableMetadataProto],
) -> list[dict[str, object]]:
  formatted = []
  for proto in protos:
    formatted.append(
        {
            "name": proto.name,
            "shape": list(proto.shape),
            "mesh_shape": list(proto.mesh_shape),
            "layout": list(proto.layout),
            "item_size": int(proto.item_size),
            "layer_idx": int(proto.layer_idx),
            "sharding_spec": list(proto.sharding_spec),
        }
    )
  return formatted


def _log_variable_protos(
    role: str,
    protos: list[raiden_service_pb2.VariableMetadataProto],
) -> None:
  for idx, proto in enumerate(protos):
    logging.info("%s variable proto[%d]: %s", role, idx, proto)


def _log_runtime_metadata(
    role: str,
    arrays: list[jax.Array],
    mesh: jax.sharding.Mesh,
    shard_idx: jax.Array,
    slice_byte_sizes: jax.Array,
    gathered_ws_info: np.ndarray,
) -> None:
  variable_protos = _variable_protos(arrays)
  logging.info("%s mesh shape: %s", role, dict(mesh.shape))
  logging.info("%s mesh devices: %s", role, _format_mesh_devices(mesh))
  logging.info("%s shard_idx: %s", role, np.array(shard_idx).tolist())
  logging.info(
      "%s slice_byte_sizes: %s", role, np.array(slice_byte_sizes).tolist()
  )
  logging.info("%s ws_info rows: %s", role, _format_ws_info_rows(gathered_ws_info))
  logging.info(
      "%s variable metadata: %s",
      role,
      _format_variable_metadata(variable_protos),
  )
  _log_variable_protos(role, variable_protos)


def _build_role_arrays(role: str) -> tuple[list[jax.Array], jax.sharding.Mesh]:
  devices = np.array(jax.devices())
  specs = _specs()
  arrays: list[jax.Array] = []
  if len(devices) < 16:
    raise ValueError(
        f"Whole-slice direct API repro requires 16 visible JAX devices, got {len(devices)}"
    )
  mesh_devices = devices[:16].reshape((4, 4))
  mesh = jax.sharding.Mesh(mesh_devices, ("fsdp", "tp"))
  if role == "source":
    for idx, (shape, pspec, _) in enumerate(specs):
      sharding = jax.sharding.NamedSharding(mesh, pspec)
      arrays.append(
          jax.jit(
              lambda s=shape, v=float(idx + 1): jnp.full(
                  s, fill_value=v, dtype=jnp.float32
              ),
              out_shardings=sharding,
          )()
      )
  else:
    for shape, pspec, _ in specs:
      sharding = jax.sharding.NamedSharding(mesh, pspec)
      arrays.append(
          jax.jit(
              lambda s=shape: jnp.zeros(s, dtype=jnp.float32),
              out_shardings=sharding,
          )()
      )
  for arr in arrays:
    arr.block_until_ready()
  return arrays, mesh


def _slice_byte_sizes(arrays: list[jax.Array], mesh: jax.sharding.Mesh) -> jax.Array:
  slice_byte_sizes = [
      int(np.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize)
      for arr in arrays
  ]
  sizes_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  return jax.device_put(jnp.array(slice_byte_sizes, dtype=jnp.int32), sizes_sharding)


def _shard_idx(mesh: jax.sharding.Mesh) -> jax.Array:
  task_mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
  global_ids = jnp.array([d.id for d in mesh.devices.flatten()], dtype=jnp.int32)
  global_ids = global_ids.reshape(task_mesh_shape)
  return jax.device_put(
      global_ids,
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*mesh.axis_names)),
  )


def _variable_protos(
    arrays: list[jax.Array],
) -> list[raiden_service_pb2.VariableMetadataProto]:
  protos = []
  for idx, ((global_shape, _, name), arr) in enumerate(zip(_specs(), arrays)):
    local_shard_shape = arr.sharding.shard_shape(global_shape)
    sharding_shape = [g // l for g, l in zip(global_shape, local_shard_shape)]
    spec_axes = []
    for axis in arr.sharding.spec:
      if axis is None:
        spec_axes.append("")
      elif isinstance(axis, str):
        spec_axes.append(axis)
      else:
        spec_axes.append(",".join(axis))
    protos.append(
        raiden_service_pb2.VariableMetadataProto(
            name=name,
            shape=global_shape,
            mesh_shape=sharding_shape,
            layout=tuple(range(len(global_shape) - 1, -1, -1)),
            item_size=arr.dtype.itemsize,
            layer_idx=idx,
            sharding_spec=spec_axes,
        )
    )
  return protos


def _ffi_ws_info(
    role: str,
    arrays: list[jax.Array],
    mesh: jax.sharding.Mesh,
    shard_idx: jax.Array,
    slice_byte_sizes: jax.Array,
    parallelism: int,
    num_dst_hosts: int,
):
  if role == "source":
    multihost_utils.sync_global_devices("direct_api_d2h_start")
    ws_info = raiden_ffi.init_weight_synchronizer_and_d2h(
        device_arrays=arrays,
        shard_idx=shard_idx,
        mesh=mesh,
        slice_byte_sizes=slice_byte_sizes,
        parallelism=parallelism,
        num_layers=len(arrays),
        listener_port=0,
        num_shards=_devices_per_host(np.array(mesh.devices)),
    )
    multihost_utils.sync_global_devices("direct_api_d2h_done")
    return ws_info
  del num_dst_hosts
  return raiden_ffi.init_weight_synchronizer(
      device_array=arrays[0],
      shard_idx=shard_idx,
      mesh=mesh,
      slice_byte_sizes=slice_byte_sizes,
      parallelism=parallelism,
      num_layers=len(arrays),
      listener_port=0,
      num_shards=_devices_per_host(np.array(mesh.devices)),
  )


def _gather_ws_info(ws_info, mesh: jax.sharding.Mesh) -> np.ndarray:
  local_ws_info = multihost_utils.global_array_to_host_local_array(
      ws_info,
      mesh,
      jax.sharding.PartitionSpec(*mesh.axis_names, None),
  )
  return multihost_utils.process_allgather(local_ws_info).reshape(-1, 6)


def _listener_groups(gathered_ws_info: np.ndarray) -> tuple[list[str], list[str]]:
  ips, listeners = [], []
  for row in gathered_ws_info:
    ip = _unpack_ip(row)
    ips.append(f"{ip}:{int(row[4])}")
    listeners.append(f"{ip}:{int(row[5])}")
  unique_listeners = []
  for listener in listeners:
    if listener not in unique_listeners:
      unique_listeners.append(listener)
  return ips, unique_listeners


def _register_worker_units(
    role: str,
    controller_address: str,
    arrays: list[jax.Array],
    mesh: jax.sharding.Mesh,
    ips: list[str],
    gathered_ws_info: np.ndarray,
) -> list[str]:
  ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_address)
  listeners = [f"{_unpack_ip(row)}:{int(row[5])}" for row in gathered_ws_info]
  unique_listeners = []
  for listener in listeners:
    if listener not in unique_listeners:
      unique_listeners.append(listener)
  unit_prefix = "pathways_trainer" if role == "source" else "pathways_sampler"
  variable_protos = _variable_protos(arrays)
  mesh_shape = [mesh.shape[a] for a in mesh.axis_names]
  mesh_axes = list(mesh.axis_names)
  logging.info(
      "%s registration summary: %d ws_info rows, %d unique listeners, mesh_shape=%s mesh_axes=%s",
      role,
      len(gathered_ws_info),
      len(unique_listeners),
      mesh_shape,
      mesh_axes,
  )
  for task_idx, listener in enumerate(unique_listeners):
    shards = [ips[i] for i, item in enumerate(listeners) if item == listener]
    ctrl_client.register_work_unit(
        unit=raiden_controller.RaidenId(
            unit_prefix, str(task_idx), "direct_api_repro_weights"
        ),
        shards=shards,
        control_plane_rpc_address=listener,
        mesh_shape=mesh_shape,
        variables=variable_protos,
        mesh_axes=mesh_axes,
    )
    logging.info(
        "Registered %s task %d listener=%s shards=%s",
        role,
        task_idx,
        listener,
        shards,
    )
  return unique_listeners


def _format_all_shards(arr: jax.Array) -> list[dict[str, object]]:
  shards = []
  for shard in arr.addressable_shards:
    shards.append(
        {
            "device": str(shard.device),
            "index": str(shard.index),
            "data": np.array(shard.data).tolist(),
        }
    )
  return shards


def _log_array_snapshots(role: str, arrays: list[jax.Array]) -> None:
  for arr, (_, _, name) in zip(arrays, _specs()):
    arr_np = np.array(arr)
    logging.info("%s %s full array: %s", role, name, arr_np.tolist())
    logging.info("%s %s shards: %s", role, name, _format_all_shards(arr))


def _verify_destination(arrays: list[jax.Array]) -> None:
  mismatches = []
  for idx, (arr, (_, _, name)) in enumerate(zip(arrays, _specs())):
    expected = float(idx + 1)
    if not bool(jnp.allclose(arr, expected, rtol=1e-5, atol=1e-5)):
      arr_np = np.array(arr)
      mismatches.append(
          (
              name,
              float(np.mean(arr_np)),
              float(np.max(np.abs(arr_np - expected))),
              arr_np.tolist(),
              _format_all_shards(arr),
          )
      )
  if mismatches:
    raise AssertionError(f"Destination mismatch after multi_h2d: {mismatches}")


def _run_controller_src(args: argparse.Namespace) -> None:
  os.environ.setdefault("RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS", "1")
  os.environ.setdefault("RAIDEN_LOG_DETAILED_SLICE_PLANS", "1")
  self_ip = _resolve_local_ip()
  src_port = int(args.controller_address.rsplit(":", 1)[1])
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
  controller = raiden_controller.RaidenController(
      port=src_port,
      worker_rpc_client=worker_rpc_client,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  server.start()
  src_units = [
      raiden_controller.RaidenId(
        "pathways_trainer", str(i), "direct_api_repro_weights"
      )
      for i in range(args.num_src_hosts)
  ]
  dst_units = [
      raiden_controller.RaidenId(
          "pathways_sampler", str(i), "direct_api_repro_weights"
      )
      for i in range(args.num_dst_hosts)
  ]
  deadline = time.time() + 1800.0
  while True:
    registered = set(controller._registered_shards.keys())
    if all(unit in registered for unit in src_units):
      break
    if time.time() > deadline:
      raise RuntimeError("Timeout waiting for source workers to register")
    time.sleep(2)
  logging.info(
      "Source controller registered units: %s",
      sorted(str(unit) for unit in controller._registered_shards.keys()),
  )
  time.sleep(5)
  dst_addr = args.dst_controller_address or f"{self_ip}:{src_port + 1}"
  dst_facade = raiden_controller.RaidenControllerClientFacade(dst_addr)
  while True:
    try:
      metadata_list = dst_facade.get_metadata()
      registered = {
          raiden_controller.RaidenId(
              m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
          )
          for m in metadata_list
      }
      if all(unit in registered for unit in dst_units):
        break
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning("Failed to query destination metadata: %s", exc)
    if time.time() > deadline:
      raise RuntimeError("Timeout waiting for destination workers to register")
    time.sleep(2)
  logging.info(
      "Destination metadata units: %s",
      sorted(str(unit) for unit in registered),
  )
  logging.info(
      "Destination metadata summary: %s",
      [
          {
              "unit": str(
                  raiden_controller.RaidenId(
                      m.unit.job_name, m.unit.job_replica_id, m.unit.data_name
                  )
              ),
              "num_shards": len(m.shards),
              "shards": list(m.shards),
              "control_plane_rpc_address": m.control_plane_rpc_address,
              "variables": _format_variable_metadata(list(m.variables)),
              "mesh_shape": list(m.mesh_shape),
              "mesh_axes": list(m.mesh_axes),
          }
          for m in metadata_list
      ],
  )
  num_variables = len(metadata_list[0].variables)
  total_devices = sum(len(m.shards) for m in metadata_list)
  expected_block_count = num_variables * total_devices
  logging.info(
      "Starting transfer with num_variables=%d total_devices=%d expected_block_count=%d group_size=%d",
      num_variables,
      total_devices,
      expected_block_count,
      args.group_size,
  )
  future = controller.start_transfer(
      src_units=src_units,
      dst_units=dst_units,
      dst_mem_type=raiden_controller.RaidenMemoryType.DRAM,
      use_block_chunks=True,
      is_sender=True,
      dst_controller_address=dst_addr,
      uuid=123456,
      req_id=args.req_id,
      expected_block_count=expected_block_count,
      group_size=args.group_size,
  )
  loop = asyncio.new_event_loop()
  try:
    loop.run_until_complete(future.wait())
  finally:
    loop.close()
  logging.info("Direct API transfer complete")
  try:
    dst_facade.shutdown()
  finally:
    close_loop = asyncio.new_event_loop()
    try:
      close_loop.run_until_complete(worker_rpc_client.shutdown_workers())
    finally:
      close_loop.close()
      server.stop()


def _run_controller_dst(args: argparse.Namespace) -> None:
  os.environ.setdefault("RAIDEN_FAIL_ON_IDENTICAL_SLICE_PLANS", "1")
  os.environ.setdefault("RAIDEN_LOG_DETAILED_SLICE_PLANS", "1")
  dst_port = int(args.controller_address.rsplit(":", 1)[1])
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
  controller = raiden_controller.RaidenController(
      port=dst_port,
      worker_rpc_client=worker_rpc_client,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  server.start()
  dst_units = [
      raiden_controller.RaidenId(
          "pathways_sampler", str(i), "direct_api_repro_weights"
      )
      for i in range(args.num_dst_hosts)
  ]
  deadline = time.time() + 1800.0
  while True:
    registered = set(controller._registered_shards.keys())
    if all(unit in registered for unit in dst_units):
      break
    if time.time() > deadline:
      raise RuntimeError("Timeout waiting for destination workers to register")
    time.sleep(2)
  logging.info(
      "Destination controller registered units: %s",
      sorted(str(unit) for unit in controller._registered_shards.keys()),
  )
  while not server._stopped:
    time.sleep(2)


def _run_worker(args: argparse.Namespace) -> None:
  _initialize_pathways_runtime(args)
  logging.info("Visible JAX devices=%d", jax.device_count())
  arrays, mesh = _build_role_arrays(args.role)
  _log_array_snapshots(f"{args.role}:before_transfer", arrays)
  shard_idx = _shard_idx(mesh)
  slice_byte_sizes = _slice_byte_sizes(arrays, mesh)
  ws_info = _ffi_ws_info(
      args.role,
      arrays,
      mesh,
      shard_idx,
      slice_byte_sizes,
      args.parallelism,
      args.num_dst_hosts,
  )
  gathered_ws_info = _gather_ws_info(ws_info, mesh)
  _log_runtime_metadata(
      args.role,
      arrays,
      mesh,
      shard_idx,
      slice_byte_sizes,
      gathered_ws_info,
  )
  ips, _ = _listener_groups(gathered_ws_info)
  unique_listeners = _register_worker_units(
      args.role,
      args.controller_address,
      arrays,
      mesh,
      ips,
      gathered_ws_info,
  )
  for listener in unique_listeners:
    while not _ping_port(listener):
      time.sleep(0.5)
  for listener in unique_listeners:
    while _ping_port(listener):
      time.sleep(1)
  if args.role == "source":
    raiden_ffi.destroy_weight_synchronizer()
    logging.info("Source finished cleanly")
    return
  multihost_utils.sync_global_devices("direct_api_h2d_start")
  arrays = list(raiden_ffi.multi_h2d(arrays, shard_idx, mesh))
  for arr in arrays:
    arr.block_until_ready()
  multihost_utils.sync_global_devices("direct_api_h2d_done")
  _log_array_snapshots("destination:after_h2d", arrays)
  _verify_destination(arrays)
  raiden_ffi.destroy_weight_synchronizer()
  logging.info("Destination verification succeeded")


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv or sys.argv[1:])
  logging.set_verbosity(args.log_level)
  logging.use_absl_handler()
  logging.info("Starting direct API repro as role=%s", args.role)
  if args.role == "controller_src":
    _run_controller_src(args)
    return
  if args.role == "controller_dst":
    _run_controller_dst(args)
    return
  _run_worker(args)


if __name__ == "__main__":
  main()