#!/usr/bin/env python3

"""Pathways-only direct Raiden repro with two controllers and FFI on both sides."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import math
import os
import socket
import sys
import time

if "--FLAGS_pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
  sys.argv.append("--FLAGS_pathways_enforce_subset_devices_form_subslice=false")

from absl import logging
import jax
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from tpu_sync.frameworks.jax import weight_synchronizer_ffi as raiden_ffi
from tpu_sync.rpc import raiden_controller
from tpu_sync.rpc import raiden_service_pb2
from tunix.experimental.weight_sync.raiden_synchronizer import _ensure_ffi_compute_on_compat

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
    os.environ.setdefault("JAX_PLATFORMS", "proxy,cpu")
    os.environ.setdefault("JAX_BACKEND_TARGET", args.pathways_target)
  if "proxy" in os.environ.get("JAX_PLATFORMS", "") and os.environ.get(
      "JAX_BACKEND_TARGET"
  ):
    import pathwaysutils

    logging.info(
        "Initializing Pathways runtime via pathwaysutils for %s",
        os.environ["JAX_BACKEND_TARGET"],
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


def _devices_per_host(devices: np.ndarray, num_hosts: int = 4) -> int:
  flat = list(devices.flatten())
  num_processes = len(set(getattr(d, "process_index", 0) for d in flat))
  if num_processes > 1:
    return len(flat) // num_processes
  return len(flat) // max(1, num_hosts)


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


def _build_role_arrays(role: str) -> tuple[list[jax.Array], jax.sharding.Mesh]:
  devices = np.array(jax.devices())
  mesh_shape = (len(devices) // 4, 4)
  mesh_devices = mesh_utils.create_device_mesh(mesh_shape, devices)
  mesh = jax.sharding.Mesh(mesh_devices, ("fsdp", "tp"))
  arrays = []
  for idx, (shape, pspec, _) in enumerate(_specs()):
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    if role == "source":
      arrays.append(
          jax.jit(
              lambda s=shape, v=float(idx + 1): jnp.full(
                  s, fill_value=v, dtype=jnp.float32
              ),
              out_shardings=sharding,
          )()
      )
    else:
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
  sizes = [
      int(np.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize)
      for arr in arrays
  ]
  sizes_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None))
  return jax.device_put(jnp.array(sizes, dtype=jnp.int32), sizes_sharding)


def _host_subgrid(mesh_shape: tuple[int, ...], devices_per_host: int) -> tuple[int, ...]:
  if len(mesh_shape) == 2:
    if devices_per_host == 4 and mesh_shape[0] % 2 == 0 and mesh_shape[1] % 2 == 0:
      return (2, 2)
  elif len(mesh_shape) == 3:
    if (
        devices_per_host == 4
        and mesh_shape[1] % 2 == 0
        and mesh_shape[2] % 2 == 0
    ):
      return (1, 2, 2)

  subgrid = [1] * len(mesh_shape)
  remaining = devices_per_host
  for axis in range(len(mesh_shape) - 1, -1, -1):
    factor = math.gcd(mesh_shape[axis], remaining)
    subgrid[axis] = factor
    remaining //= factor
  if remaining != 1:
    subgrid = [1] * len(mesh_shape)
    subgrid[-1] = devices_per_host
  return tuple(subgrid)


def _ffi_shard_idx(mesh: jax.sharding.Mesh) -> jax.Array:
  task_mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
  devices_per_host = _devices_per_host(np.array(mesh.devices))
  host_subgrid = _host_subgrid(task_mesh_shape, devices_per_host)
  host_grid = tuple(
      mesh_dim // host_dim for mesh_dim, host_dim in zip(task_mesh_shape, host_subgrid)
  )
  shard_ids = np.zeros(task_mesh_shape, dtype=np.int32)

  for coords in np.ndindex(task_mesh_shape):
    host_coords = tuple(coord // size for coord, size in zip(coords, host_subgrid))
    local_coords = tuple(coord % size for coord, size in zip(coords, host_subgrid))

    host_index = 0
    for coord, dim in zip(host_coords, host_grid):
      host_index = host_index * dim + coord

    local_index = 0
    for coord, dim in zip(local_coords, host_subgrid):
      local_index = local_index * dim + coord

    shard_ids[coords] = host_index * devices_per_host + local_index

  return jax.device_put(
      jnp.array(shard_ids, dtype=jnp.int32),
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


def _register_proxy_worker_units(
    role: str,
    controller_address: str,
    arrays: list[jax.Array],
    mesh: jax.sharding.Mesh,
    gathered_ws_info: np.ndarray,
) -> list[str]:
  ctrl_client = raiden_controller.RaidenControllerClientFacade(controller_address)
  ips, unique_listeners = _listener_groups(gathered_ws_info)
  listeners = [f"{_unpack_ip(row)}:{int(row[5])}" for row in gathered_ws_info]
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
    unit_id = raiden_controller.RaidenId(
        unit_prefix, str(task_idx), "direct_api_repro_weights"
    )
    while True:
      try:
        ctrl_client.register_work_unit(
            unit=unit_id,
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
        break
      except Exception as exc:
        logging.warning(
            "Waiting to register %s task %d with controller at %s: %s",
            role,
            task_idx,
            controller_address,
            exc,
        )
        time.sleep(1)
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


def _array_to_numpy(arr: jax.Array) -> np.ndarray:
  try:
    return np.array(arr)
  except RuntimeError:
    return np.array(multihost_utils.process_allgather(arr, tiled=True))


def _log_array_snapshots(role: str, arrays: list[jax.Array]) -> None:
  for arr, (_, _, name) in zip(arrays, _specs()):
    logging.info("%s %s full array: %s", role, name, _array_to_numpy(arr).tolist())
    logging.info("%s %s shards: %s", role, name, _format_all_shards(arr))


def _verify_destination(arrays: list[jax.Array]) -> None:
  mismatches = []
  for idx, (arr, (_, _, name)) in enumerate(zip(arrays, _specs())):
    expected = float(idx + 1)
    arr_np = _array_to_numpy(arr)
    if not bool(np.allclose(arr_np, expected, rtol=1e-5, atol=1e-5)):
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


def _wait_for_listeners(unique_listeners: list[str]) -> None:
  for listener in unique_listeners:
    while not _ping_port(listener):
      time.sleep(0.5)
  while any(_ping_port(listener) for listener in unique_listeners):
    time.sleep(1.0)


def _sorted_units(
    units: set[raiden_controller.RaidenId] | list[raiden_controller.RaidenId],
) -> list[raiden_controller.RaidenId]:
  return sorted(
      units,
      key=lambda unit: (
          unit.job_name,
          int(unit.job_replica_id),
          unit.data_name,
          unit.data_replica_idx,
      ),
  )


def _wait_for_stable_registered_units(
    controller: raiden_controller.RaidenController,
    unit_prefix: str,
    min_count: int,
    deadline: float,
    description: str,
) -> list[raiden_controller.RaidenId]:
  previous_units: list[raiden_controller.RaidenId] | None = None
  while True:
    registered_units = {
        unit
        for unit in controller._registered_shards.keys()
        if unit.job_name == unit_prefix
        and unit.data_name == "direct_api_repro_weights"
    }
    current_units = _sorted_units(registered_units)
    if len(current_units) >= min_count and current_units == previous_units:
      return current_units
    if time.time() > deadline:
      raise RuntimeError(
          f"Timeout waiting for stable {description} registration: {current_units}"
      )
    previous_units = current_units
    time.sleep(2)


def _wait_for_stable_metadata(
    facade: raiden_controller.RaidenControllerClientFacade,
    unit_prefix: str,
    min_count: int,
    deadline: float,
) -> tuple[list[object], list[raiden_controller.RaidenId]]:
  previous_units: list[raiden_controller.RaidenId] | None = None
  while True:
    try:
      metadata_list = list(facade.get_metadata())
      registered_units = {
          raiden_controller.RaidenId(
              metadata.unit.job_name,
              metadata.unit.job_replica_id,
              metadata.unit.data_name,
          )
          for metadata in metadata_list
          if metadata.unit.job_name == unit_prefix
          and metadata.unit.data_name == "direct_api_repro_weights"
      }
      current_units = _sorted_units(registered_units)
      if len(current_units) >= min_count and current_units == previous_units:
        return metadata_list, current_units
      previous_units = current_units
    except Exception as exc:
      logging.warning("Failed to query %s metadata: %s", unit_prefix, exc)
    if time.time() > deadline:
      raise RuntimeError(f"Timeout waiting for stable {unit_prefix} metadata")
    time.sleep(2)


def _run_destination_worker(args: argparse.Namespace) -> None:
  _initialize_pathways_runtime(args)
  logging.info("Visible JAX devices=%d", jax.device_count())
  arrays, mesh = _build_role_arrays("destination")
  _log_array_snapshots("destination:before_transfer", arrays)
  _ensure_ffi_compute_on_compat()
  shard_idx = _ffi_shard_idx(mesh)
  slice_byte_sizes = _slice_byte_sizes(arrays, mesh)
  ws_info = raiden_ffi.init_weight_synchronizer(
      device_array=arrays[0],
      shard_idx=shard_idx,
      mesh=mesh,
      slice_byte_sizes=slice_byte_sizes,
      parallelism=args.parallelism,
      num_layers=len(arrays),
      listener_port=0,
      num_shards=_devices_per_host(np.array(mesh.devices), num_hosts=args.num_dst_hosts),
  )
  gathered_ws_info = _gather_ws_info(ws_info, mesh)
  _log_runtime_metadata(
      "destination", arrays, mesh, shard_idx, slice_byte_sizes, gathered_ws_info
  )
  unique_listeners = _register_proxy_worker_units(
      "destination",
      args.controller_address,
      arrays,
      mesh,
      gathered_ws_info,
  )
  logging.info("Destination waiting for transfer completion...")
  _wait_for_listeners(unique_listeners)
  multihost_utils.sync_global_devices("pw_ffi_h2d_start")
  arrays = list(raiden_ffi.multi_h2d(arrays, shard_idx, mesh))
  for arr in arrays:
    arr.block_until_ready()
  multihost_utils.sync_global_devices("pw_ffi_h2d_done")
  _log_array_snapshots("destination:after_h2d", arrays)
  _verify_destination(arrays)
  logging.info("Destination verification succeeded")
  raiden_ffi.destroy_weight_synchronizer()


def _run_source_worker(args: argparse.Namespace) -> None:
  _initialize_pathways_runtime(args)
  logging.info("Visible JAX devices=%d", jax.device_count())
  arrays, mesh = _build_role_arrays("source")
  _log_array_snapshots("source:before_transfer", arrays)
  _ensure_ffi_compute_on_compat()
  shard_idx = _ffi_shard_idx(mesh)
  slice_byte_sizes = _slice_byte_sizes(arrays, mesh)
  multihost_utils.sync_global_devices("pw_ffi_d2h_start")
  ws_info = raiden_ffi.init_weight_synchronizer_and_d2h(
      device_arrays=arrays,
      shard_idx=shard_idx,
      mesh=mesh,
      slice_byte_sizes=slice_byte_sizes,
      parallelism=args.parallelism,
      num_layers=len(arrays),
      listener_port=0,
      num_shards=_devices_per_host(np.array(mesh.devices), num_hosts=args.num_src_hosts),
  )
  multihost_utils.sync_global_devices("pw_ffi_d2h_done")
  gathered_ws_info = _gather_ws_info(ws_info, mesh)
  _log_runtime_metadata(
      "source", arrays, mesh, shard_idx, slice_byte_sizes, gathered_ws_info
  )
  unique_listeners = _register_proxy_worker_units(
      "source",
      args.controller_address,
      arrays,
      mesh,
      gathered_ws_info,
  )
  logging.info("Source worker ready and registered; standing by.")
  _wait_for_listeners(unique_listeners)
  raiden_ffi.destroy_weight_synchronizer()


def _run_controller_src(args: argparse.Namespace) -> None:
  self_ip = _resolve_local_ip()
  src_port = int(args.controller_address.rsplit(":", 1)[1])
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
  controller = raiden_controller.RaidenController(
      port=src_port,
      worker_rpc_client=worker_rpc_client,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  server.start()
  deadline = time.time() + 1800.0
  src_units = _wait_for_stable_registered_units(
      controller=controller,
      unit_prefix="pathways_trainer",
      min_count=args.num_src_hosts,
      deadline=deadline,
      description="source worker",
  )
  logging.info(
      "Source controller registered units: %s",
      [str(unit) for unit in src_units],
  )
  dst_addr = args.dst_controller_address or f"{self_ip}:{src_port + 1}"
  dst_facade = raiden_controller.RaidenControllerClientFacade(dst_addr)
  metadata_list, dst_units = _wait_for_stable_metadata(
      facade=dst_facade,
      unit_prefix="pathways_sampler",
      min_count=args.num_dst_hosts,
      deadline=deadline,
  )
  logging.info(
      "Destination metadata units: %s",
      [str(unit) for unit in dst_units],
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
  skip_tiling = {i: False for i in range(num_variables)}
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
      skip_tiling=skip_tiling,
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
    if hasattr(worker_rpc_client, "shutdown_workers"):
      loop = asyncio.new_event_loop()
      try:
        loop.run_until_complete(worker_rpc_client.shutdown_workers())
      finally:
        loop.close()
    server.stop()


def _run_controller_dst(args: argparse.Namespace) -> None:
  dst_port = int(args.controller_address.rsplit(":", 1)[1])
  worker_rpc_client = raiden_controller.WeightSyncWorkerRpcClient(name_resolver=None)
  controller = raiden_controller.RaidenController(
      port=dst_port,
      worker_rpc_client=worker_rpc_client,
  )
  server = raiden_controller.RaidenControllerServer(controller)
  server.start()
  deadline = time.time() + 1800.0
  dst_units = _wait_for_stable_registered_units(
      controller=controller,
      unit_prefix="pathways_sampler",
      min_count=args.num_dst_hosts,
      deadline=deadline,
      description="destination worker",
  )
  logging.info(
      "Destination controller registered units: %s",
      [str(unit) for unit in dst_units],
  )
  while not server._stopped:
    time.sleep(2)


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
  if args.role == "source":
    _run_source_worker(args)
    return
  _run_destination_worker(args)


if __name__ == "__main__":
  main(sys.argv[1:])