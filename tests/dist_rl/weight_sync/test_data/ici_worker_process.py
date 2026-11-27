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

"""One worker of a multi-process CPU JAX world, for `ici_end_to_end_test`.

Run as `<binary> <config json>`. The process joins the world, builds either
the trainer-side `IciSourceSynchronizer` or the rollout-side
`IciWeightSyncDelegate` around a fake sampler, then serves `(method, argument)`
commands from the test over a `multiprocessing.connection`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
from typing import Any

from absl import app
import numpy as np
from tunix.dist_rl.testing import ici_testing

# name -> (shape, dtype). Small, but covers a leaf no axis of which divides
# the device count and a non-float32 dtype.
WEIGHTS: dict[str, tuple[tuple[int, ...], str]] = {
    "embed": ((8, 6), "float32"),
    "block.kernel": ((4, 4), "bfloat16"),
    "block.bias": ((3,), "float32"),
    "head.kernel": ((6, 2), "float32"),
}


def expected_value(name: str, version: int) -> np.ndarray:
  """Returns the deterministic expected weight tensor for `(name, version)`."""
  shape, _ = WEIGHTS[name]
  base = sum(map(ord, name)) % 97 + 100 * version
  return (np.arange(int(np.prod(shape))) + base).reshape(shape)


def _nested(flat: dict[str, Any]) -> dict[str, Any]:
  tree: dict[str, Any] = {}
  for name, leaf in flat.items():
    node = tree
    *parents, last = name.split(".")
    for part in parents:
      node = node.setdefault(part, {})
    node[last] = leaf
  return tree


def _serve(cfg: dict[str, Any]) -> None:
  """Initializes CPU Gloo JAX and serves worker commands until exit."""
  jax = ici_testing.init_cpu_gloo_world(
      width=cfg["width"],
      jax_port=cfg["jax_port"],
      processes=cfg["processes"],
      rank=cfg["rank"],
  )
  # pylint: disable=g-import-not-at-top
  import jax.numpy as jnp
  from tunix.dist_rl.weight_sync import ici_transport
  from tunix.dist_rl.weight_sync import ici_weight_sync_delegate
  # pylint: enable=g-import-not-at-top

  def _mesh() -> Any:
    return jax.sharding.Mesh(
        np.asarray(ici_transport.local_devices()), ("tp",)
    )

  def _put(name: str, version: int, axis_from_end: bool) -> Any:
    shape, dtype = WEIGHTS[name]
    mesh = _mesh()
    spec: list[str | None] = [None] * len(shape)
    axes = range(len(shape))
    for axis in reversed(axes) if axis_from_end else axes:
      if shape[axis] % mesh.size == 0:
        spec[axis] = "tp"
        break
    return jax.device_put(
        jnp.asarray(expected_value(name, version), dtype),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec)),
    )

  class _FakeSampler:
    """The sampler surface the delegate uses; weights in a flat dict."""

    model_runner = None
    config = None

    def __init__(self):
      self._flat = {n: _put(n, 0, axis_from_end=True) for n in WEIGHTS}
      self.calls: list[str] = []

    @property
    def transformer_state(self) -> dict[str, Any]:
      return self._flat

    def delete_cache(self) -> None:
      self.calls.append("delete_cache")

    def reset_prefix_cache(self) -> None:
      self.calls.append("reset_prefix_cache")

    def reinitialize_cache(self) -> None:
      self.calls.append("reinitialize_cache")

    def refresh_state_leaves(self) -> None:
      self.calls.append("refresh_state_leaves")

  class _Rollout:
    """Rollout command handler wrapping `IciWeightSyncDelegate`."""

    def __init__(self):
      self.sampler = _FakeSampler()
      self.delegate = ici_weight_sync_delegate.IciWeightSyncDelegate(
          server_id=f"rollout-{cfg['rank']}",
      )

    def bind(self, _: Any) -> bool:
      return asyncio.run(self.delegate.bind_weight_sync(sampler=self.sampler))

    def metadata(self, _: Any) -> Any:
      return asyncio.run(self.delegate.get_weight_sync_metadata())

    def pre(self, fields: dict[str, Any]) -> bool:
      req = ici_testing.FakeSyncRequest(fields)
      return asyncio.run(self.delegate.pre_weight_sync(req))

    def ici_check(self, request: dict[str, Any]) -> Any:
      return self.delegate.check(request)

    def ici_transfer(self, request: dict[str, Any]) -> Any:
      return self.delegate.transfer(request)

    def weight_sync(self, fields: dict[str, Any]) -> int:
      req = ici_testing.FakeSyncRequest(fields)
      return asyncio.run(self.delegate.weight_sync(req))

    def post(self, fields: dict[str, Any]) -> bool:
      req = ici_testing.FakeSyncRequest(fields)
      return asyncio.run(self.delegate.post_weight_sync(req))

    def abort(self, fields: dict[str, Any]) -> bool:
      req = ici_testing.FakeSyncRequest(fields)
      return asyncio.run(self.delegate.abort_weight_sync(req))

    def status(self, _: Any) -> dict[str, Any]:
      return dict(self.delegate.get_weight_sync_status())

    def snapshot(self, _: Any) -> dict[str, Any]:
      out: dict[str, Any] = {}
      for name, array in self.sampler.transformer_state.items():
        if array.is_deleted():
          out[name] = None
        else:
          out[name] = np.asarray(array).astype(np.float64)
      return {"weights": out, "calls": list(self.sampler.calls)}

  class _Trainer:
    """Trainer command handler wrapping `IciSourceSynchronizer`."""

    def __init__(self):
      self.sync = ici_weight_sync_delegate.IciSourceSynchronizer("trainer")

    def prepare(self, version: int) -> list[Any]:
      state = {
          "base": _nested(
              {n: _put(n, version, axis_from_end=False) for n in WEIGHTS}
          )
      }
      self.sync.bind(state)
      self.sync.d2h()
      return [self.sync.work_unit_metadata()]

    def ici_check(self, request: dict[str, Any]) -> Any:
      return self.sync.check(request)

    def ici_transfer(self, request: dict[str, Any]) -> Any:
      return self.sync.transfer(request)

    def release(self, _: Any) -> bool:
      self.sync.release()
      return self.sync.bound

  worker = _Trainer() if cfg["role"] == "trainer" else _Rollout()
  ici_testing.serve_worker_ipc(cfg["control_port"], cfg["rank"], worker)


def main(argv: Sequence[str]) -> None:
  """Entrypoint parsing the JSON worker config from `argv[1]`."""
  _serve(json.loads(argv[1]))


if __name__ == "__main__":
  app.run(main)
