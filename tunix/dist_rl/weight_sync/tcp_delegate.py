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

"""Destination-side weight sync delegate for TCP weight sync."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import logging
import threading
import time
from typing import Any, Protocol, runtime_checkable

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import tcp_handler
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.models import hf_mapping
from tunix.models import hf_mapping_jax


@runtime_checkable
class HfWeightUpdatableSampler(Protocol):
  """Protocol for samplers supporting declarative HF weight updates."""

  def from_hf_manifest(self) -> hf_mapping.WeightMappingSpec:
    ...

  def update_params_from_hf(
      self, hf_weights: Mapping[str, jax.Array]
  ) -> None:
    ...


def derive_hf_variables_from_manifest(
    dst_spec: hf_mapping.WeightMappingSpec,
    dst_state: nnx.State | Mapping[Any, Any],
) -> tuple[weight_sync.TensorMetadata, ...]:
  """Derives canonical HF TensorMetadata from a manifest and target state."""
  if isinstance(dst_state, nnx.State):
    flat_by_path = {
        ".".join(str(k) for k in path): var.value
        for path, var in dst_state.flat_state()
        if isinstance(var, nnx.Variable)
    }
  elif isinstance(dst_state, Mapping):
    flat_by_path = {
        (k if isinstance(k, str) else ".".join(str(p) for p in k)): (
            v.value if isinstance(v, nnx.Variable) else v
        )
        for k, v in dst_state.items()
    }
  else:
    raise TypeError(
        f"dst_state must be an nnx.State or Mapping, got {type(dst_state)}"
    )

  dropped = set(dst_spec.dropped_keys)
  required_dst_keys: set[str] = set()
  for r in dst_spec.rules:
    if not dropped.intersection((*r.source_keys, *r.target_keys)):
      required_dst_keys.update(r.target_keys)
  missing_dst = sorted(required_dst_keys - flat_by_path.keys())
  if missing_dst:
    raise KeyError(
        "Destination state missing target keys required by HF manifest:"
        f" {missing_dst}"
    )

  abstract_dst = {
      k: jax.ShapeDtypeStruct(
          shape=tuple(flat_by_path[k].shape), dtype=flat_by_path[k].dtype
      )
      for k in required_dst_keys
  }
  inv_manifest = hf_mapping.invert_manifest(dst_spec)
  abstract_hf = jax.eval_shape(
      functools.partial(hf_mapping_jax.to_hf_impl, inv_manifest),
      abstract_dst,
  )
  return tuple(
      weight_sync.TensorMetadata(
          name=hf_key,
          shape=tuple(struct.shape),
          mesh_shape=(1,) * len(struct.shape),
          layout=tuple(range(len(struct.shape) - 1, -1, -1)),
          item_size=int(np.dtype(struct.dtype).itemsize),
      )
      for hf_key, struct in sorted(abstract_hf.items())
  )


def _extract_round_id(
    sync_request: datatypes.WeightSyncRequest | None,
) -> tuple[str, int]:
  """Extracts and strictly validates req_id and generation from sync_request."""
  if sync_request is None:
    raise ValueError("WeightSyncRequest cannot be None")
  if not isinstance(sync_request, datatypes.WeightSyncRequest):
    raise TypeError(f"Expected WeightSyncRequest, got {type(sync_request)}")
  extra = sync_request.extra_config
  if not extra or "req_id" not in extra or not extra["req_id"]:
    raise ValueError(
        f"WeightSyncRequest.extra_config missing required 'req_id': {extra}"
    )
  if "uuid" not in extra or extra["uuid"] is None:
    raise ValueError(
        f"WeightSyncRequest.extra_config missing required 'uuid': {extra}"
    )
  generation = int(extra["uuid"])
  if generation <= 0:
    raise ValueError(
        f"WeightSyncRequest uuid must be positive, got {generation}"
    )
  return str(extra["req_id"]), generation


class TcpDestinationDelegate:
  """Manages host-staged TCP weight synchronization for rollout samplers.

  Incoming weights from trainer rank 0 are streamed directly into host RAM
  buffers managed by `TcpReceiverServer`. Model weights on TPU/GPU are
  updated only during the `weight_sync()` phase.
  """

  def __init__(
      self,
      worker_index: int = 0,
      server_id: str = "rollout",
      partial_rollout: bool = False,
      bind_port: int | None = None,
      advertised_host: str | None = None,
      expected_variables: Sequence[weight_sync.TensorMetadata] | None = None,
  ):
    self._server_id = server_id
    self._worker_index = worker_index
    self.partial_rollout: bool = bool(partial_rollout)
    self._bind_port = bind_port
    self._advertised_host = advertised_host
    self._variables: tuple[weight_sync.TensorMetadata, ...] = tuple(
        expected_variables or ()
    )

    self._sampler: HfWeightUpdatableSampler | None = None
    self._version = 0
    self._tracker = weight_sync_coordinator.WorkerRoundTracker()
    self._phase_lock = threading.Lock()
    self._receiver: tcp_handler.TcpReceiverServer | None = None
    self._serving_corrupted = False
    self._bound = False

  @property
  def serving_corrupted(self) -> bool:
    return self._serving_corrupted

  def is_bounded(self) -> bool:
    """Returns whether destination receiver is bound and listening."""
    return self._bound

  async def bind_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      state: Any = None,
      sampler: Any = None,
      **kwargs: Any,
  ) -> Any:
    """Binds destination-side transport receiver for weight sync."""
    del sync_request
    with self._phase_lock:
      if self._bound:
        return True

      if sampler is None or not isinstance(
          sampler, HfWeightUpdatableSampler
      ):
        raise TypeError(
            f"Sampler {type(sampler)} does not implement"
            " HfWeightUpdatableSampler"
        )
      self._sampler = sampler

      if (
          "expected_variables" in kwargs
          and kwargs["expected_variables"] is not None
      ):
        self._variables = tuple(kwargs["expected_variables"])
      elif not self._variables:
        if state is None:
          raise ValueError(
              "state is required to derive HF TensorMetadata when"
              " expected_variables is not pre-configured."
          )
        self._variables = derive_hf_variables_from_manifest(
            self._sampler.from_hf_manifest(), state
        )

      if self._receiver is None:
        self._receiver = tcp_handler.TcpReceiverServer(
            bind_port=self._bind_port,
            advertised_host=self._advertised_host,
        )

      self._bound = True
      return True

  async def get_weight_sync_metadata(
      self, **kwargs: Any
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Retrieves destination worker metadata for the sync coordinator."""
    del kwargs
    if not self._bound or self._receiver is None:
      raise RuntimeError(
          f"TcpDestinationDelegate [{self._server_id}] is not bound."
      )
    return [
        weight_sync.WorkUnitMetadata(
            unit=weight_sync.WorkUnitId(
                job_name=self._server_id,
                data_replica_idx=self._worker_index,
            ),
            control_plane_rpc_address=self._receiver.address,
            variables=self._variables,
            transport_mode="tcp",
            use_ffi=False,
        )
    ]

  async def pre_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Pre-sync phase hook executed before weight transfer begins."""
    del kwargs
    with self._phase_lock:
      if self._serving_corrupted:
        raise RuntimeError(
            "Serving weights are corrupted from a previous failed H2D; cannot"
            " admit new round."
        )

      _extract_round_id(sync_request)
      if not self._tracker.admit(sync_request, "prepared"):
        return True
      self._tracker.complete(sync_request, "prepared")

      return True

  async def weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Executes weight installation on device from host staging buffer."""
    del kwargs
    with self._phase_lock:
      if self._serving_corrupted:
        raise RuntimeError(
            "Serving weights are corrupted from a previous failed H2D; cannot"
            " admit weight_sync."
        )

      req_id, generation = _extract_round_id(sync_request)
      assert sync_request is not None
      if not self._tracker.admit(sync_request, "h2d_done"):
        return self._version

      if self._receiver is None:
        raise RuntimeError(
            "TcpDestinationDelegate receiver is not initialized."
        )

      round_weights = self._receiver.pop_round(req_id, generation)

      cpu_device = jax.devices("cpu")[0]
      jax_weights: dict[str, jax.Array] = {}
      with jax.default_device(cpu_device):
        for name, payload in round_weights.items():
          if payload.dtype in ("bf16", "bfloat16"):
            jax_weights[name] = jax.lax.bitcast_convert_type(
                jnp.asarray(payload.buffer), jnp.bfloat16
            )
          else:
            jax_weights[name] = jnp.asarray(payload.buffer)

      if self._sampler is None:
        raise RuntimeError("Sampler is not available for weight sync")

      if not isinstance(self._sampler, HfWeightUpdatableSampler):
        raise TypeError(
            f"Sampler {type(self._sampler)} does not implement"
            " HfWeightUpdatableSampler (missing update_params_from_hf)"
        )

      t_install_start = time.perf_counter()
      try:
        self._sampler.update_params_from_hf(jax_weights)
      except Exception as exc:
        self._serving_corrupted = True
        logging.error(
            "TcpDestinationDelegate: failed during H2D update_params_from_hf:"
            " %s",
            exc,
        )
        raise

      version = sync_request.policy_version
      self._version = (
          int(version) if version is not None else self._version + 1
      )

      install_elapsed = time.perf_counter() - t_install_start
      logging.info(
          "[TCP] installed %d tensors on device (policy_version=%d) in %.3fs",
          len(round_weights),
          self._version,
          install_elapsed,
      )

      self._tracker.complete(sync_request, "h2d_done")

      return self._version

  async def post_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Post-sync phase hook executed after weight installation completes."""
    del kwargs
    with self._phase_lock:
      _extract_round_id(sync_request)
      if not self._tracker.admit(sync_request, "committed"):
        return True
      self._tracker.complete(sync_request, "committed")

      return True

  async def abort_weight_sync(
      self,
      sync_request: datatypes.WeightSyncRequest | None = None,
      **kwargs: Any,
  ) -> Any:
    """Safely handles abort of weight sync round."""
    del kwargs
    with self._phase_lock:
      if self._serving_corrupted:
        raise RuntimeError(
            "Cannot abort weight sync after partial H2D failure; serving"
            " weights are corrupted."
        )
      req_id, generation = _extract_round_id(sync_request)
      if not self._tracker.admit(sync_request, "aborted"):
        return False
      if self._receiver is not None:
        self._receiver.discard_round(req_id, generation)
      self._tracker.complete(sync_request, "aborted")
      return True

  def get_weight_sync_status(self) -> Mapping[str, Any]:
    """Reports worker-side round status for coordinator recovery checks."""
    return self._tracker.report()

  def close(self) -> None:
    """Shuts down receiver server and marks delegate unbound."""
    with self._phase_lock:
      if self._receiver is not None:
        self._receiver.close()
        self._receiver = None
      self._bound = False
