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

"""One process of a multi-process CPU JAX world, for `ici_transport_test`.

Run as `<binary> <scenario json>`. Each process joins the world over Gloo CPU
collectives, plays its role in one transfer, verifies what it received, and
prints a single `RESULT <json>` line.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from absl import app
import numpy as np
from tunix.dist_rl.testing import ici_testing


def _expected(shape: tuple[int, ...], name: str, version: int) -> np.ndarray:
  size = int(np.prod(shape))
  base = sum(map(ord, name)) + 1000 * version
  return (np.arange(size) + base).reshape(shape)


def _verify_source_intact(
    leaves: Sequence[Any], source: Mapping[str, Any], version: int
) -> bool:
  for leaf in leaves:
    want = _expected(leaf.shape, leaf.name, version).astype(leaf.dtype)
    for shard in source[leaf.name].addressable_shards:
      if not np.array_equal(np.asarray(shard.data), want[shard.index]):
        return False
  return True


def _verify_destination(
    leaves: Sequence[Any], sink: Any, version: int
) -> list[str]:
  mismatched: list[str] = []
  for leaf in leaves:
    got = sink.state[leaf.name]
    if got.sharding != sink.target_sharding(leaf.name):
      mismatched.append(leaf.name + ":sharding")
    want = _expected(leaf.shape, leaf.name, version).astype(leaf.dtype)
    if not np.array_equal(np.asarray(got), want):
      mismatched.append(leaf.name + f":value@v{version}")
  return mismatched


def _run_scenario(cfg: dict[str, Any]) -> None:
  """Initializes CPU Gloo JAX and executes the transport scenario."""
  jax = ici_testing.init_cpu_gloo_world(
      width=cfg["width"],
      jax_port=cfg["port"],
      processes=cfg["processes"],
      rank=cfg["rank"],
  )
  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp
  from tunix.dist_rl.weight_sync import ici_transport
  # pylint: enable=g-import-not-at-top

  leaves = (
      ici_transport.LeafSpec("embed", (8, 6), "float32"),
      ici_transport.LeafSpec("odd", (3, 5), "float32"),
      ici_transport.LeafSpec("late_axis", (3, 4), "bfloat16"),
      ici_transport.LeafSpec("bias", (4,), "float32"),
      ici_transport.LeafSpec("ints", (2, 2, 2), "int32"),
  )
  leaves_by_name = {leaf.name: leaf for leaf in leaves}
  reverse_mesh = bool(cfg.get("reverse_device_order", False))

  class _Sink:
    """Stand-in for a sampler holding leaves on a non-canonical sharding."""

    def __init__(self):
      devices = list(ici_transport.local_devices())
      if reverse_mesh:
        devices = list(reversed(devices))
      self.mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
      self.state: dict[str, Any] = {}
      self.released: list[str] = []
      for leaf in leaves:
        self.state[leaf.name] = jax.device_put(
            jnp.zeros(leaf.shape, leaf.dtype), self.target_sharding(leaf.name)
        )

    def target_sharding(self, name: str) -> Any:
      leaf = leaves_by_name[name]
      spec: list[str | None] = [None] * len(leaf.shape)
      for axis in reversed(range(len(leaf.shape))):
        if leaf.shape[axis] % len(self.mesh.devices) == 0:
          spec[axis] = "tp"
          break
      return jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec(*spec)
      )

    def release(self, names: Sequence[str]) -> None:
      self.released.extend(names)
      for name in names:
        self.state.pop(name).delete()

    def install(self, arrays: Mapping[str, Any]) -> None:
      self.state.update(arrays)

  def _source_arrays(version: int) -> dict[str, Any]:
    sources = cfg["sources"]
    rows = [
        list(reversed(ici_transport.local_devices(p)))
        if reverse_mesh
        else list(ici_transport.local_devices(p))
        for p in sources
    ]
    mesh = jax.sharding.Mesh(np.asarray(rows), ("fsdp", "tp"))
    arrays: dict[str, Any] = {}
    for leaf in leaves:
      spec: list[str | None] = [None] * len(leaf.shape)
      if len(sources) > 1 and leaf.shape[0] % len(sources) == 0:
        spec[0] = "fsdp"
      sharding = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*spec)
      )
      val = _expected(leaf.shape, leaf.name, version)
      arrays[leaf.name] = jax.make_array_from_callback(
          leaf.shape,
          sharding,
          lambda index, val=val, leaf=leaf: val[index].astype(leaf.dtype),
      )
    return arrays

  rank = cfg["rank"]
  result: dict[str, Any] = {"rank": rank, "role": "bystander"}
  plan = ici_transport.build_plan(
      leaves,
      source_processes=cfg["sources"],
      destination_processes=cfg["destinations"],
      width=cfg["width"],
      world_processes=range(cfg["processes"]),
      chunk_bytes=cfg["chunk_bytes"],
  )
  result["chunks"] = len(plan.chunks)
  result["rounds"] = len(plan.rounds)
  sink = _Sink() if rank in cfg["destinations"] else None
  for version in range(1, cfg["syncs"] + 1):
    if rank in cfg["sources"]:
      result["role"] = "source"
      source = _source_arrays(version)
      ici_transport.execute(plan, source=source)
      result["source_intact"] = _verify_source_intact(leaves, source, version)
    elif sink is not None:
      result["role"] = "destination"
      timings = ici_transport.execute(plan, sink=sink)
      result["bytes"] = timings["bytes"]
      result.setdefault("mismatched", []).extend(
          _verify_destination(leaves, sink, version)
      )
      result["released"] = len(sink.released)
    else:
      ici_transport.execute(plan)
  print("RESULT " + json.dumps(result), flush=True)


def main(argv: Sequence[str]) -> None:
  """Entrypoint parsing the JSON scenario config from `argv[1]`."""
  _run_scenario(json.loads(argv[1]))


if __name__ == "__main__":
  app.run(main)
