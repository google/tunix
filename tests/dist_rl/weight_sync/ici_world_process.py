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

"""One process of a multi-process CPU JAX world, for `ici_transport_test`.

Run as `python ici_world_process.py <scenario json>`. Each process joins the
world over gloo CPU collectives, plays its role in one transfer, verifies what
it received, and prints a single `RESULT <json>` line.
"""

import json
import os
import sys

_CFG = json.loads(sys.argv[1])
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={_CFG['width']}"
)
os.environ["JAX_PLATFORMS"] = "cpu"

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax

jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(
    coordinator_address=f"127.0.0.1:{_CFG['port']}",
    num_processes=_CFG["processes"],
    process_id=_CFG["rank"],
)

import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.weight_sync import ici_transport

LEAVES = (
    ici_transport.LeafSpec("embed", (8, 6), "float32"),
    ici_transport.LeafSpec("odd", (3, 5), "float32"),  # nothing divides
    ici_transport.LeafSpec("late_axis", (3, 4), "bfloat16"),  # splits axis 1
    ici_transport.LeafSpec("bias", (4,), "float32"),
    ici_transport.LeafSpec("ints", (2, 2, 2), "int32"),
)


def expected(leaf: ici_transport.LeafSpec, version: int) -> np.ndarray:
  size = int(np.prod(leaf.shape))
  base = sum(map(ord, leaf.name)) + 1000 * version
  return (np.arange(size) + base).reshape(leaf.shape)


class _Sink:
  """Stand-in for a sampler: holds leaves on a non-canonical sharding."""

  def __init__(self):
    devices = ici_transport.local_devices()
    self.mesh = jax.sharding.Mesh(np.asarray(devices), ("tp",))
    self.state = {}
    self.released = []
    for leaf in LEAVES:
      self.state[leaf.name] = jax.device_put(
          jnp.zeros(leaf.shape, leaf.dtype), self.target_sharding(leaf.name)
      )

  def target_sharding(self, name):
    leaf = next(l for l in LEAVES if l.name == name)
    # Shard the LAST axis the device count divides: deliberately not the
    # canonical (first such axis) layout, so install has to reshard.
    spec = [None] * len(leaf.shape)
    for axis in reversed(range(len(leaf.shape))):
      if leaf.shape[axis] % len(self.mesh.devices) == 0:
        spec[axis] = "tp"
        break
    return jax.sharding.NamedSharding(
        self.mesh, jax.sharding.PartitionSpec(*spec)
    )

  def release(self, names):
    self.released.extend(names)
    for name in names:
      self.state.pop(name).delete()

  def install(self, arrays):
    self.state.update(arrays)


def _source_arrays(version: int):
  """Trainer-side leaves, sharded the way a trainer would, not canonically."""
  sources = _CFG["sources"]
  rows = [ici_transport.local_devices(p) for p in sources]
  mesh = jax.sharding.Mesh(np.asarray(rows), ("fsdp", "tp"))
  arrays = {}
  for leaf in LEAVES:
    spec = [None] * len(leaf.shape)
    if len(sources) > 1 and leaf.shape[0] % len(sources) == 0:
      spec[0] = "fsdp"  # crosses processes: not fully addressable
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec(*spec)
    )
    value = expected(leaf, version)
    arrays[leaf.name] = jax.make_array_from_callback(
        leaf.shape,
        sharding,
        lambda index, value=value, leaf=leaf: value[index].astype(leaf.dtype),
    )
  return arrays


def main() -> None:
  rank = _CFG["rank"]
  result = {"rank": rank, "role": "bystander"}
  plan = ici_transport.build_plan(
      LEAVES,
      source_processes=_CFG["sources"],
      destination_processes=_CFG["destinations"],
      width=_CFG["width"],
      world_processes=range(_CFG["processes"]),
      chunk_bytes=_CFG["chunk_bytes"],
  )
  result["chunks"] = len(plan.chunks)
  result["rounds"] = len(plan.rounds)
  sink = _Sink() if rank in _CFG["destinations"] else None
  for version in range(1, _CFG["syncs"] + 1):
    if rank in _CFG["sources"]:
      result["role"] = "source"
      source = _source_arrays(version)
      ici_transport.execute(plan, source=source)
      # The source's own arrays must survive the transfer untouched.
      result["source_intact"] = all(
          np.array_equal(
              np.asarray(shard.data),
              expected(leaf, version).astype(leaf.dtype)[shard.index],
          )
          for leaf in LEAVES
          for shard in source[leaf.name].addressable_shards
      )
    elif sink is not None:
      result["role"] = "destination"
      timings = ici_transport.execute(plan, sink=sink)
      result["bytes"] = timings["bytes"]
      mismatched = []
      for leaf in LEAVES:
        got = sink.state[leaf.name]
        if got.sharding != sink.target_sharding(leaf.name):
          mismatched.append(leaf.name + ":sharding")
        if not np.array_equal(
            np.asarray(got), expected(leaf, version).astype(leaf.dtype)
        ):
          mismatched.append(leaf.name + f":value@v{version}")
      result.setdefault("mismatched", []).extend(mismatched)
      result["released"] = len(sink.released)
    else:
      # In the world but neither sending nor receiving: still has to enter
      # every collective, or the others block.
      ici_transport.execute(plan)
  print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
  main()
