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

"""One worker of a multi-process CPU JAX world, for `ici_end_to_end_test`.

Run as `python ici_worker_process.py <config json>`. The process joins the
world, builds either the trainer-side `IciSourceSynchronizer` or the
rollout-side `IciWeightSyncDelegate` around a fake sampler, then serves
`(method, argument)` commands from the test over a
`multiprocessing.connection`, the stand-in for a worker's gRPC server.
"""

import asyncio
import inspect
import json
from multiprocessing import connection
import os
import sys
import traceback

_CFG = json.loads(sys.argv[1])
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={_CFG['width']}"
)
os.environ["JAX_PLATFORMS"] = "cpu"

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax

jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(
    coordinator_address=f"127.0.0.1:{_CFG['jax_port']}",
    num_processes=_CFG["processes"],
    process_id=_CFG["rank"],
)
# The backend only comes up once every process of the world has asked for it,
# so ask now: a trainer that stayed lazy until its first command would leave
# the rollouts blocked here.
jax.devices()

import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.weight_sync import ici_transport
from tunix.dist_rl.weight_sync import ici_weight_sync_delegate

# name -> (shape, dtype). Small, but covers a leaf no axis of which divides
# the device count and a non-float32 dtype.
WEIGHTS = {
    "embed": ((8, 6), "float32"),
    "block.kernel": ((4, 4), "bfloat16"),
    "block.bias": ((3,), "float32"),
    "head.kernel": ((6, 2), "float32"),
}


def value(name: str, version: int) -> np.ndarray:
  shape, _ = WEIGHTS[name]
  base = sum(map(ord, name)) % 97 + 100 * version
  return (np.arange(int(np.prod(shape))) + base).reshape(shape)


def _nested(flat):
  tree = {}
  for name, leaf in flat.items():
    node = tree
    *parents, last = name.split(".")
    for part in parents:
      node = node.setdefault(part, {})
    node[last] = leaf
  return tree


def _mesh():
  return jax.sharding.Mesh(np.asarray(ici_transport.local_devices()), ("tp",))


def _put(name: str, version: int, axis_from_end: bool):
  """A weight on this process's devices, sharded on one divisible axis."""
  shape, dtype = WEIGHTS[name]
  mesh = _mesh()
  spec = [None] * len(shape)
  axes = range(len(shape))
  for axis in reversed(axes) if axis_from_end else axes:
    if shape[axis] % mesh.size == 0:
      spec[axis] = "tp"
      break
  return jax.device_put(
      jnp.asarray(value(name, version), dtype),
      jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)),
  )


class _FakeSampler:
  """The sampler surface the delegate uses; weights in a nested dict."""

  model_runner = None
  config = None

  def __init__(self):
    # A flat `{dotted name: array}` dict, where the trainer publishes a nested
    # tree rooted under "base"; `param_key` gives both the same leaf names.
    self._flat = {n: _put(n, 0, axis_from_end=True) for n in WEIGHTS}
    self.calls = []

  @property
  def transformer_state(self):
    return self._flat

  def delete_cache(self):
    self.calls.append("delete_cache")

  def reset_prefix_cache(self):
    self.calls.append("reset_prefix_cache")

  def reinitialize_cache(self):
    self.calls.append("reinitialize_cache")

  def refresh_state_leaves(self):
    self.calls.append("refresh_state_leaves")


class _Request:
  """The fields of `WeightSyncRequest` the delegate reads."""

  def __init__(self, fields):
    self.policy_version = fields["policy_version"]
    self.extra_config = {"req_id": fields["req_id"], "uuid": fields["uuid"]}


class _Rollout:

  def __init__(self):
    self.sampler = _FakeSampler()
    self.delegate = ici_weight_sync_delegate.IciWeightSyncDelegate(
        server_id=f"rollout-{_CFG['rank']}",
        drop_old_weights=_CFG["drop_old"],
    )

  def bind(self, _):
    return asyncio.run(self.delegate.bind_weight_sync(sampler=self.sampler))

  def metadata(self, _):
    return asyncio.run(self.delegate.get_weight_sync_metadata())

  def pre(self, fields):
    return asyncio.run(self.delegate.pre_weight_sync(_Request(fields)))

  def ici_check(self, request):
    return self.delegate.check(request)

  def ici_transfer(self, request):
    return self.delegate.transfer(request)

  def weight_sync(self, fields):
    return asyncio.run(self.delegate.weight_sync(_Request(fields)))

  def post(self, fields):
    return asyncio.run(self.delegate.post_weight_sync(_Request(fields)))

  def abort(self, fields):
    return asyncio.run(self.delegate.abort_weight_sync(_Request(fields)))

  def status(self, _):
    return dict(self.delegate.get_weight_sync_status())

  def snapshot(self, _):
    """What the sampler would serve from, as host arrays."""
    out = {}
    for name, array in self.sampler.transformer_state.items():
      if array.is_deleted():
        out[name] = None
      else:
        out[name] = np.asarray(array).astype(np.float64)
    return {"weights": out, "calls": list(self.sampler.calls)}


class _Trainer:

  def __init__(self):
    self.sync = ici_weight_sync_delegate.IciSourceSynchronizer("trainer")

  def prepare(self, version):
    state = {
        "base": _nested(
            {n: _put(n, version, axis_from_end=False) for n in WEIGHTS}
        )
    }
    self.sync.bind(state)
    self.sync.d2h()
    return [self.sync.work_unit_metadata()]

  def ici_check(self, request):
    return self.sync.check(request)

  def ici_transfer(self, request):
    return self.sync.transfer(request)

  def release(self, _):
    self.sync.release()
    return self.sync.bound


def main() -> None:
  worker = _Trainer() if _CFG["role"] == "trainer" else _Rollout()
  conn = connection.Client(("127.0.0.1", _CFG["control_port"]))
  conn.send(_CFG["rank"])
  while True:
    try:
      method, argument = conn.recv()
    except EOFError:
      return
    if method == "exit":
      return
    try:
      result = getattr(worker, method)(argument)
      if inspect.isawaitable(result):
        result = asyncio.run(result)
      conn.send(("ok", result))
    except Exception as e:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      conn.send(("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
  main()
