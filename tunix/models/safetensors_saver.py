# Copyright 2025 Google LLC
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

"""Safetensors writers: sharded state export and LoRA-merged export."""

from collections.abc import Mapping, Sequence
import contextlib
import dataclasses
import json
import os
import shutil
import struct
from typing import Any, BinaryIO, Callable, cast

from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
import safetensors.flax as safe_flax
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.utils import sharding_utils

# Soft cap on bytes per shard file (2.5 GiB), in line with typical multi-GiB
# Hugging Face checkpoint shards; a larger single tensor gets its own shard.
DEFAULT_MAX_SHARD_BYTES = 2560 * 1024 * 1024


def join_path(path) -> str:
  return '.'.join([str(field) for field in path])


def save_lora_merged_model_as_safetensors(
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
    state_key_transform_fn: Callable[[str], str],
    custom_layer_extractor_fn: (
        Callable[[dict[str, list[Any]]], dict[str, list[Any]]] | None
    ) = None,
    transpose_rules: dict[str, tuple[int, ...]] | None = None,
):
  """Saves a model with LoRA weights merged in safetensors format.

  This is a generic function that can be used for any model architecture.
  Model-specific logic is provided via callback functions.

  Args:
    local_model_path: Path to the base model safetensors checkpoint directory.
    output_dir: Directory where the merged model will be saved.
    lora_model: Model instance with LoRA weights.
    rank: LoRA rank used during training.
    alpha: LoRA alpha used during training.
    state_key_transform_fn: Function that transforms model layer paths to
      safetensors state dict keys.
    custom_layer_extractor_fn: Optional function that updates the extracted LoRA
      layer dictionary; it should accept the current layer dict and return a
      dict of the new/updated LoRA layers' names as strings to a tuple of the
      corresponding lora pair.
  """

  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
  os.makedirs(output_dir)

  lora_layers: dict[str, list[Any]] = {}

  for path, value in nnx.iter_graph(lora_model):
    if isinstance(value, nnx.LoRAParam):
      path_str = join_path(path[:-1])
      if path_str in lora_layers:
        assert (
            'lora_b' in path[-1]  # pyrefly: ignore[not-iterable]
        ), f'Expect second LoRAParam to be lora_b, got {path[-1]}'
        lora_layers[path_str].append(value)
      else:
        assert (
            'lora_a' in path[-1]  # pyrefly: ignore[not-iterable]
        ), f'Expect first LoRAParam to be lora_a, got {path[-1]}'
        lora_layers[path_str] = [value]

  if custom_layer_extractor_fn:
    lora_layers = custom_layer_extractor_fn(lora_layers)

  # Load base model state
  base_state = safe_flax.load_file(local_model_path + '/model.safetensors')

  # Apply LoRA deltas
  for path, (lora_a, lora_b) in lora_layers.items():
    state_key = state_key_transform_fn(path)
    assert state_key in base_state, (
        f'LoRA layer {path} not found in base model state dict'
        f' {base_state.keys()}'
    )

    lora_a_val = jnp.asarray(getattr(lora_a, 'value', lora_a))
    lora_b_val = jnp.asarray(getattr(lora_b, 'value', lora_b))

    # Reshape 3D tensors to 2D if necessary
    if lora_a_val.ndim == 3:
      d0, d1, d2 = lora_a_val.shape
      lora_a_val = lora_a_val.reshape(d0 * d1, d2)
    if lora_b_val.ndim == 3:
      d0, d1, d2 = lora_b_val.shape
      lora_b_val = lora_b_val.reshape(d0, d1 * d2)

    # Compute and apply LoRA delta
    combined_lora = (lora_a_val @ lora_b_val) * (alpha / rank)
    if transpose_rules:
      for t_key, rule in transpose_rules.items():
        if t_key in state_key:
          combined_lora = combined_lora.transpose(rule)
          break

    base_state[state_key] += combined_lora.astype(base_state[state_key].dtype)

  # Save merged model
  safetensors_path = os.path.join(output_dir, 'model.safetensors')
  safe_flax.save_file(base_state, safetensors_path)

  # Copy non-safetensors files (config, tokenizer, etc.)
  for filename in os.listdir(local_model_path):
    if not filename.endswith('.safetensors'):
      src = os.path.join(local_model_path, filename)
      if os.path.isfile(src):
        dst = os.path.join(output_dir, filename)
        shutil.copy(src, dst)


# Truncation limit for the host-0 error message that is serialized into the
# JSON payload broadcast to the other processes.
_MAX_BROADCAST_ERROR_CHARS = 4096
# Tensor payloads are written as memoryview slices of this size so backends
# that copy bytes-like inputs never copy a whole tensor; 16 MiB amortizes
# per-write overhead.
_WRITE_CHUNK_BYTES = 16 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class SafetensorsSaveResult:
  """Outcome of a committed `save_safetensors` call, identical on all processes.

  Attributes:
    paths: Committed files; the last entry is the commit marker
      (`model.safetensors.index.json` when sharded, else `model.safetensors`).
    checksums: `checksum_fn` output over the written tensors, else `None`.
  """

  paths: tuple[str, ...]
  checksums: dict[str, float | int] | None


def _safetensors_dtype(key: str, arr: np.ndarray) -> str:
  """Returns the safetensors dtype tag for `arr`; raises if unsupported."""
  np_dt = np.dtype(arr.dtype)
  if np_dt not in safetensors_io.NP_TO_SAFETENSORS_DTYPE:
    raise ValueError(f'Unsupported safetensors dtype {np_dt!r} for {key!r}.')
  return safetensors_io.NP_TO_SAFETENSORS_DTYPE[np_dt]


def encode_safetensors_header(
    shard_items: Sequence[tuple[str, np.ndarray]],
) -> tuple[bytes, list[np.ndarray], int]:
  """Encodes the safetensors wire header and contiguous byte views for a shard.

  Args:
    shard_items: Ordered `(tensor_name, array)` pairs for one shard.

  Returns:
    A `(header_bytes, payloads, total_bytes)` tuple where `header_bytes` is the
    8-byte little-endian header length followed by the 8-byte-aligned UTF-8
    JSON header, `payloads` are contiguous `uint8` views of each tensor in
    header order, and `total_bytes` is the sum of tensor payload byte sizes.
  """
  header: dict[str, Any] = {}
  payloads: list[np.ndarray] = []
  offset = 0
  for key, arr in shard_items:
    dtype = _safetensors_dtype(key, arr)
    c_arr = np.ascontiguousarray(arr) if arr.ndim > 0 else np.asarray(arr)
    end = offset + int(c_arr.nbytes)
    header[key] = {
        'dtype': dtype,
        'shape': [int(d) for d in c_arr.shape],
        'data_offsets': [offset, end],
    }
    payloads.append(c_arr.reshape(-1).view(np.uint8))
    offset = end

  raw_header = json.dumps(header, separators=(',', ':')).encode('utf-8')
  padded_header = raw_header + (b' ' * ((-len(raw_header)) % 8))
  header_bytes = struct.pack('<Q', len(padded_header)) + padded_header
  return header_bytes, payloads, offset


def write_safetensors_shard(
    shard_path: epath.Path,
    shard_items: Sequence[tuple[str, np.ndarray]],
) -> int:
  """Writes one `.safetensors` file via `.tmp` + `replace()`; returns bytes."""
  header_bytes, payloads, total_bytes = encode_safetensors_header(shard_items)
  tmp_path = shard_path.parent / f'{shard_path.name}.tmp'
  try:
    with cast(BinaryIO, tmp_path.open('wb')) as fh:
      fh.write(header_bytes)
      for raw_bytes in payloads:
        # Chunk memoryview writes to keep local POSIX zero-copy while bounding
        # internal C++/remote-stream buffer copies to <= _WRITE_CHUNK_BYTES.
        mv = memoryview(raw_bytes).cast('B')
        for off in range(0, len(mv), _WRITE_CHUNK_BYTES):
          fh.write(mv[off : off + _WRITE_CHUNK_BYTES])
    tmp_path.replace(shard_path)
  except BaseException:
    with contextlib.suppress(Exception):
      tmp_path.unlink(missing_ok=True)
    raise
  return total_bytes


def partition_safetensors_shards(
    host_tensors: Mapping[str, np.ndarray],
    max_shard_bytes: int,
) -> list[list[tuple[str, np.ndarray]]]:
  """Groups `(key, array)` pairs greedily within `max_shard_bytes`.

  A single tensor whose byte size exceeds `max_shard_bytes` is placed in its
  own 1-tensor shard.

  Args:
    host_tensors: Mapping of tensor names to host NumPy arrays.
    max_shard_bytes: Positive soft upper bound on bytes per shard.

  Returns:
    Non-empty list of shards, each a list of `(key, array)` pairs.
  """
  if (
      isinstance(max_shard_bytes, bool)
      or not isinstance(max_shard_bytes, int)
      or max_shard_bytes <= 0
  ):
    raise ValueError(
        f'max_shard_bytes must be a positive int, got {max_shard_bytes!r}.'
    )
  shards: list[list[tuple[str, np.ndarray]]] = []
  cur_shard: list[tuple[str, np.ndarray]] = []
  cur_bytes = 0
  for key, arr in host_tensors.items():
    nbytes = int(arr.nbytes)
    if cur_shard and cur_bytes + nbytes > max_shard_bytes:
      shards.append(cur_shard)
      cur_shard = []
      cur_bytes = 0
    cur_shard.append((key, arr))
    cur_bytes += nbytes
  if cur_shard or not shards:
    shards.append(cur_shard)
  return shards


def remove_tree(path: epath.Path) -> None:
  """Deletes `path` and everything under it on local and remote stores.

  Some object-store backends (e.g., `gs://`) create a `dir/` placeholder object
  on `mkdir`, and `rmtree` may remove only that placeholder while leaving child
  objects behind. Deleting children bottom-up before calling `rmtree` ensures
  complete cleanup across local filesystems and remote object stores.

  Args:
    path: File or directory to delete; a missing path is a no-op.
  """
  if not path.is_dir():
    path.unlink(missing_ok=True)
    return
  for child in path.iterdir():
    remove_tree(child)
  path.rmtree(missing_ok=True)


def write_safetensors_index(
    output_dir: epath.Path,
    weight_map: Mapping[str, str],
    total_bytes: int,
) -> epath.Path:
  """Atomically writes `model.safetensors.index.json` as the commit marker."""
  index_path = output_dir / 'model.safetensors.index.json'
  tmp_index = output_dir / f'{index_path.name}.tmp'
  try:
    tmp_index.write_text(
        json.dumps(
            {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map},
            indent=2,
            sort_keys=True,
        )
    )
    tmp_index.replace(index_path)
  except BaseException:
    with contextlib.suppress(Exception):
      tmp_index.unlink(missing_ok=True)
    raise
  return index_path


def publish_safetensors_shards(
    output_dir: epath.Path,
    shards: Sequence[Sequence[tuple[str, np.ndarray]]],
) -> list[str]:
  """Replaces `output_dir` with shard files, writing the commit marker last."""
  # Validate before deleting so a bad dtype cannot destroy a prior export.
  for items in shards:
    for key, arr in items:
      _safetensors_dtype(key, arr)
  (output_dir / 'model.safetensors.index.json').unlink(missing_ok=True)
  remove_tree(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  if len(shards) == 1:
    single = output_dir / 'model.safetensors'
    write_safetensors_shard(single, shards[0])
    return [str(single)]
  paths: list[str] = []
  total_bytes = 0
  weight_map: dict[str, str] = {}
  for idx, items in enumerate(shards, start=1):
    fname = f'model-{idx:05d}-of-{len(shards):05d}.safetensors'
    total_bytes += write_safetensors_shard(output_dir / fname, items)
    weight_map.update((key, fname) for key, _ in items)
    paths.append(str(output_dir / fname))
  index_path = write_safetensors_index(output_dir, weight_map, total_bytes)
  return [*paths, str(index_path)]


def _commit_host_tensors(
    host_tensors: Mapping[str, np.ndarray],
    output_dir: epath.Path,
    *,
    parsed_manifest: Mapping[str, Any] | None,
    max_shard_bytes: int,
    checksum_fn: (
        Callable[[Mapping[str, np.ndarray]], dict[str, float | int]] | None
    ),
) -> dict[str, Any]:
  """Transforms, partitions, and publishes gathered host tensors on host 0."""
  if parsed_manifest is not None:
    host_tensors = hf_mapping.apply_host_manifest(
        parsed_manifest, host_tensors, drop_tied_targets=True
    )
  checksums = checksum_fn(host_tensors) if checksum_fn is not None else None
  if checksums is not None:
    json.dumps(checksums)
  shards = partition_safetensors_shards(host_tensors, max_shard_bytes)
  paths = publish_safetensors_shards(output_dir, shards)
  return {'paths': paths, 'checksums': checksums}


def save_safetensors(
    tensors: Mapping[str, Any],
    output_dir: str | os.PathLike[str],
    *,
    to_hf_manifest: hf_mapping.RuleSpec | None = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    checksum_fn: (
        Callable[[Mapping[str, np.ndarray]], dict[str, float | int]] | None
    ) = None,
) -> SafetensorsSaveResult:
  """Gathers `tensors` to process 0 and commits them as safetensors shards.

  Collective: every process must call it with the same arguments. Process 0
  replaces `output_dir` with shards of at most `max_shard_bytes` each and
  writes the commit marker last (`model.safetensors.index.json` when sharded,
  else the single `model.safetensors`); readers must treat a directory without
  its marker as uncommitted. A cross-process barrier then keeps every process
  from returning before the commit, and any host-0 exception is broadcast to
  non-zero processes so multi-host runs fail cleanly instead of deadlocking.

  Args:
    tensors: Global `jax.Array` or host arrays keyed by source tensor name.
    output_dir: Destination directory (local path or `gs://` URI).
    to_hf_manifest: Optional `hf_mapping` manifest converting source names and
      layouts to the written ones; tied targets are dropped.
    max_shard_bytes: Soft upper bound on bytes per shard; a single tensor larger
      than this limit is placed in its own 1-tensor shard.
    checksum_fn: Optional checksum over the written (post-manifest) tensors.

  Returns:
    Committed paths and checksums, shared from process 0 to every process.

  Raises:
    ValueError: If `output_dir`, `max_shard_bytes`, `to_hf_manifest`, or any
      tensor dtype is invalid.
    RuntimeError: On non-zero processes if process 0 fails during commit.
    Exception: Re-raises the original exception on process 0 if commit fails.
  """
  if not os.fspath(output_dir):
    raise ValueError('save_safetensors output_dir must not be empty.')
  if (
      isinstance(max_shard_bytes, bool)
      or not isinstance(max_shard_bytes, int)
      or max_shard_bytes <= 0
  ):
    raise ValueError(
        f'max_shard_bytes must be a positive int, got {max_shard_bytes!r}.'
    )
  parsed_manifest = (
      hf_mapping.parse_manifest(to_hf_manifest)
      if to_hf_manifest is not None
      else None
  )
  multihost = jax.process_count() > 1
  is_host0 = jax.process_index() == 0
  host_tensors = sharding_utils.gather_tensors_to_host0(
      tensors, is_host0=is_host0
  )

  out_path = epath.Path(output_dir)
  if not multihost:
    committed = _commit_host_tensors(
        host_tensors,
        out_path,
        parsed_manifest=parsed_manifest,
        max_shard_bytes=max_shard_bytes,
        checksum_fn=checksum_fn,
    )
    shared = sharding_utils.broadcast_host0_json(committed)
    return SafetensorsSaveResult(
        paths=tuple(shared['paths']), checksums=shared['checksums']
    )

  committed: dict[str, Any] = {}
  host0_exc: Exception | None = None
  if is_host0:
    try:
      committed = _commit_host_tensors(
          host_tensors,
          out_path,
          parsed_manifest=parsed_manifest,
          max_shard_bytes=max_shard_bytes,
          checksum_fn=checksum_fn,
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      host0_exc = exc
      committed = {
          'error': f'{type(exc).__name__}: {exc}'[:_MAX_BROADCAST_ERROR_CHARS]
      }
  multihost_utils.sync_global_devices(
      f'tunix:save_safetensors:{os.fspath(output_dir)}'
  )
  shared = sharding_utils.broadcast_host0_json(committed)
  if host0_exc is not None:
    raise host0_exc
  if 'error' in shared:
    raise RuntimeError(
        f'save_safetensors failed on process 0: {shared["error"]}'
    )
  return SafetensorsSaveResult(
      paths=tuple(shared['paths']), checksums=shared['checksums']
  )
