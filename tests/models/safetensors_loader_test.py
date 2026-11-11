"""Tests for safetensors_io and safetensors_loader."""

from __future__ import annotations

import io
import itertools
import json
import math
import os
import pathlib
import re
import struct
import tempfile  # pylint: disable=unused-import
import threading
import types
from typing import Any, cast
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors import numpy as stnp
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.models import safetensors_loader
from tunix.tests import test_common
from tunix.utils import env_utils

jax.config.update('jax_num_cpu_devices', 64)


def build_toy_manifest(config: test_common.ModelConfig) -> dict[str, object]:
  return hf_mapping.generate_manifest(
      model_type='toy',
      template_rules=[
          {'source_pattern': [k], 'target_pattern': [k], 'ops': []}
          for k in (
              'emb.embedding',
              'layers.{i}.attn.query.kernel',
              'layers.{i}.attn.key.kernel',
              'layers.{i}.attn.value.kernel',
              'layers.{i}.attn.out.kernel',
              'layers.{i}.w1.kernel',
              'layers.{i}.w1.bias',
              'layers.{i}.w2.kernel',
              'layers.{i}.w2.bias',
              'lm_head.kernel',
              'lm_head.bias',
          )
      ],
      axes={'i': config.num_layers},
  )


class _TrackedBinaryFile(io.BufferedIOBase):
  """Real-file wrapper recording `seek + readinto` ranges and fault modes."""

  def __init__(
      self,
      path: str,
      *,
      fault_mode: str | None = None,
      fault_trigger_call: int = 2,
      counter_lock: threading.Lock | None = None,
      shared_call_count: list[int] | None = None,
  ):
    super().__init__()
    self._fh = open(path, 'rb')
    self._fault_mode = fault_mode
    self._fault_trigger_call = fault_trigger_call
    self._counter_lock = counter_lock
    self._shared_call_count = shared_call_count
    self.read_ranges: list[tuple[int, int]] = []
    self.closed_flag = False

  def read(self, size: int | None = -1) -> bytes:
    return self._fh.read(-1 if size is None else size)

  def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
    return self._fh.seek(offset, whence)

  def readinto(self, b) -> int:
    view = memoryview(b).cast('B')
    pos = self._fh.tell()
    call_idx = 0
    if self._counter_lock is not None and self._shared_call_count is not None:
      with self._counter_lock:
        self._shared_call_count[0] += 1
        call_idx = self._shared_call_count[0]
    if self._fault_mode is not None and call_idx == self._fault_trigger_call:
      if self._fault_mode == 'minus_one':
        return -1
      if self._fault_mode == 'short_eof':
        half = max(1, len(view) // 2)
        n = self._fh.readinto(view[:half])
        return 0 if len(view) == 1 else n
      if self._fault_mode == 'raise_oserror':
        raise OSError('simulated CNS stripe read failure')
    n = self._fh.readinto(view)
    self.read_ranges.append((pos, pos + n))
    return n

  def close(self) -> None:
    self.closed_flag = True
    self._fh.close()
    super().close()


class _EdgeCaseModel(nnx.Module):
  """Minimal NNX model with scalar, empty, integer, and transposed weights."""

  def __init__(self, config, *, rngs: nnx.Rngs):
    del config, rngs
    self.scalar = nnx.Param(jnp.zeros((), dtype=jnp.bfloat16))
    self.empty = nnx.Param(jnp.zeros((0, 4), dtype=jnp.float32))
    self.codes = nnx.Param(jnp.zeros((3,), dtype=jnp.uint8))
    self.unaligned_f32 = nnx.Param(jnp.zeros((4,), dtype=jnp.float32))
    self.f16_proj = nnx.Param(jnp.zeros((3, 2), dtype=jnp.float16))


class SafetensorsLoaderTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    config = test_common.ModelConfig(num_layers=4, num_kv_heads=4, head_dim=16)
    cls.model = test_common.ToyTransformer(config=config, rngs=nnx.Rngs(0))
    cls.manifest = build_toy_manifest(config)

    cls.state = nnx.state(cls.model)
    cls.tensors = {
        'emb.embedding': np.array(cls.state['emb']['embedding'].value),
        'lm_head.kernel': np.array(cls.state['lm_head']['kernel'].value),
        'lm_head.bias': np.array(cls.state['lm_head']['bias'].value),
    }
    for i in range(cls.model.config.num_layers):
      layer_state = cls.state['layers'][i]
      cls.tensors[f'layers.{i}.attn.query.kernel'] = np.array(
          layer_state['attn']['query']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.key.kernel'] = np.array(
          layer_state['attn']['key']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.value.kernel'] = np.array(
          layer_state['attn']['value']['kernel'].value
      )
      cls.tensors[f'layers.{i}.attn.out.kernel'] = np.array(
          layer_state['attn']['out']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.kernel'] = np.array(
          layer_state['w1']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w1.bias'] = np.array(
          layer_state['w1']['bias'].value
      )
      cls.tensors[f'layers.{i}.w2.kernel'] = np.array(
          layer_state['w2']['kernel'].value
      )
      cls.tensors[f'layers.{i}.w2.bias'] = nnx.Param(
          np.array(layer_state['w2']['bias'].value),
      )

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.st_dir = self._temp_dir.name
    os.makedirs(self.st_dir, exist_ok=True)

  @parameterized.named_parameters(*(
      (
          [dict(testcase_name='opt_loader_enabled', mode='optimized')]
          if not env_utils.is_internal_env()
          else []
      )
      + [
          dict(testcase_name='absolute_path', path_type='abs'),
          dict(testcase_name='relative_path', path_type='rel'),
          dict(testcase_name='relative_dot_path', path_type='rel_dot'),
          dict(testcase_name='opt_loader_disabled', mode='original'),
      ]
  ))
  def test_load_and_create_model(self, path_type='abs', mode='auto'):
    origin_dir = os.getcwd()
    self.addCleanup(os.chdir, origin_dir)
    if path_type == 'abs':
      load_dir = self.st_dir
    elif path_type == 'rel':
      os.chdir(os.path.dirname(self.st_dir))
      load_dir = os.path.basename(self.st_dir)
    elif path_type == 'rel_dot':
      os.chdir(os.path.dirname(self.st_dir))
      load_dir = f'./{os.path.basename(self.st_dir)}'
    else:
      raise ValueError(f'Unknown path_type: {path_type}')

    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    loaded_model = safetensors_loader.load_and_create_model(
        load_dir,
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.float32,
        mode=mode,
    )
    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_from_gcs(self):
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    real_path = epath.Path
    mock_load = self.enter_context(
        mock.patch.object(
            safetensors_loader,
            'load_file_from_gcs',
            autospec=True,
            spec_set=True,
        )
    )
    self.enter_context(
        mock.patch.object(
            epath,
            'Path',
            side_effect=lambda p: (
                real_path(self.st_dir)
                if str(p) == 'gs://bucket/model'
                else real_path(p)
            ),
        )
    )
    loaded_model = safetensors_loader.load_and_create_model(
        'gs://bucket/model',
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.float32,
        mode='original',
    )
    mock_load.assert_not_called()
    loaded_state = nnx.state(loaded_model)
    jax.tree.map(
        np.testing.assert_array_equal,
        self.state,
        loaded_state,
    )

  def test_load_and_create_model_raises_on_duplicate_key(self):
    tensors = {
        'lm_head.kernel': np.zeros((2, 2), dtype=np.float32),
        'lm_head.weight': np.zeros((2, 2), dtype=np.float32),
    }
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    duplicate_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {
                'source_keys': ['lm_head.kernel'],
                'target_keys': ['lm_head.kernel'],
                'ops': [],
            },
            {
                'source_keys': ['lm_head.weight'],
                'target_keys': ['lm_head.kernel'],
                'ops': [],
            },
        ],
    }

    with self.assertRaisesRegex(ValueError, 'Duplicate target_keys'):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          duplicate_manifest,
          dtype=jnp.float32,
          mode='original',
      )

  def test_load_bfloat16_custom_dtype_avoids_ml_dtypes(self):
    bf16_data = np.random.rand(256, 16).astype(ml_dtypes.bfloat16)
    tensors = {
        k: np.asarray(
            v.value if isinstance(v, nnx.Param) else v, dtype=ml_dtypes.bfloat16
        )
        for k, v in self.tensors.items()
    }
    tensors['emb.embedding'] = bf16_data
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    result = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.bfloat16,
        mode='original',
    )
    state = nnx.state(result)
    self.assertEqual(state['emb']['embedding'].value.dtype, jnp.bfloat16)
    self.assertEqual(
        state['emb']['embedding'].value.aval.dtype,
        jnp.bfloat16,
    )
    np.testing.assert_allclose(state['emb']['embedding'].value, bf16_data)

  def test_load_mixed_dtype_safetensors_offsets(self):
    f32_data = np.ones((16,), dtype=np.float32) * 1.5
    bf16_data = np.full((256, 16), 2.0, dtype=ml_dtypes.bfloat16)
    tensors = {
        'a_log': f32_data,
        'emb.embedding': bf16_data,
    }
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    contig, metadata, mm, fh = safetensors_loader.load_safetensors_with_offsets(
        filename
    )
    try:
      by_name = {m['name']: m for m in metadata}
      a_slice = contig[
          by_name['a_log']['start_byte'] : by_name['a_log']['end_byte']
      ]
      a_loaded = a_slice.view(np.float32).reshape(by_name['a_log']['shape'])
      np.testing.assert_allclose(a_loaded, f32_data)

      emb_slice = contig[
          by_name['emb.embedding']['start_byte'] : by_name['emb.embedding'][
              'end_byte'
          ]
      ]
      emb_loaded = emb_slice.view(ml_dtypes.bfloat16).reshape(
          by_name['emb.embedding']['shape']
      )
      np.testing.assert_allclose(emb_loaded, bf16_data)
    finally:
      del contig, a_slice, a_loaded, emb_slice, emb_loaded
      mm.close()
      fh.close()

  def test_original_loader_casts_on_host_and_batches_device_put_per_shard(self):
    bf16_items = sorted(
        (
            k,
            np.asarray(v.value if isinstance(v, nnx.Param) else v).astype(
                ml_dtypes.bfloat16
            ),
        )
        for k, v in self.tensors.items()
    )
    cut1 = len(bf16_items) // 3
    cut2 = (2 * len(bf16_items)) // 3
    stnp.save_file(
        dict(bf16_items[:cut1]),
        os.path.join(self.st_dir, 'model-00001-of-00004.safetensors'),
    )
    stnp.save_file(
        dict(bf16_items[cut1:cut2]),
        os.path.join(self.st_dir, 'model-00002-of-00004.safetensors'),
    )
    stnp.save_file(
        dict(bf16_items[cut2:]),
        os.path.join(self.st_dir, 'model-00003-of-00004.safetensors'),
    )
    stnp.save_file(
        {
            'vision.patch_embed.weight': np.ones(
                (8, 8), dtype=ml_dtypes.bfloat16
            )
        },
        os.path.join(self.st_dir, 'model-00004-of-00004.safetensors'),
    )

    host_checks: list[tuple[bool, bool, bool, bool]] = []
    orig_load_file_tensors = safetensors_loader._load_file_tensors

    def check_host_tensors(*args, **kwargs):
      out = orig_load_file_tensors(*args, **kwargs)
      for arr in out.values():
        host_checks.append((
            isinstance(arr, np.ndarray),
            arr.dtype == np.float32,
            arr.flags.c_contiguous and arr.flags.aligned,
            arr.flags.writeable,
        ))
      return out

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )
    self.enter_context(
        mock.patch.object(
            safetensors_loader,
            '_load_file_tensors',
            side_effect=check_host_tensors,
        )
    )
    mock_device_put = self.enter_context(
        mock.patch.object(jax, 'device_put', wraps=jax.device_put)
    )

    loaded_model = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        mesh=mesh,
        dtype=jnp.float32,
        mode='original',
    )

    self.assertLen(host_checks, len(self.tensors))
    self.assertTrue(all(all(flags) for flags in host_checks))
    self.assertEqual(mock_device_put.call_count, 3)
    for call_args in mock_device_put.call_args_list:
      placed_inputs = call_args.args[0]
      self.assertTrue(all(isinstance(x, np.ndarray) for x in placed_inputs))

    loaded_state = nnx.state(loaded_model)
    expected_from_bf16 = jax.tree.map(
        lambda x: np.asarray(x).astype(ml_dtypes.bfloat16).astype(np.float32),
        self.state,
    )
    jax.tree.map(np.testing.assert_allclose, expected_from_bf16, loaded_state)

  def test_original_loader_stripes_large_tensors_skips_unmapped_and_closes_handles(
      self,
  ):
    filename = os.path.join(self.st_dir, 'model.safetensors')
    raw_tensors = {
        k: np.asarray(
            v.value if isinstance(v, nnx.Param) else v, dtype=np.float32
        )
        for k, v in self.tensors.items()
    }
    raw_tensors['m_unmapped.vision.tensor'] = np.ones((256,), dtype=np.float32)
    stnp.save_file(raw_tensors, filename)

    with open(filename, 'rb') as fh:
      header_size = struct.unpack('<Q', fh.read(8))[0]
      header = json.loads(fh.read(header_size))
    unmapped_start, unmapped_end = header['m_unmapped.vision.tensor'][
        'data_offsets'
    ]
    unmapped_abs_lo = 8 + header_size + unmapped_start
    unmapped_abs_hi = 8 + header_size + unmapped_end

    opened_paths: list[str] = []
    tracked_handles: list[_TrackedBinaryFile] = []
    handles_lock = threading.Lock()
    path_cls = type(epath.Path(filename))

    def open_tracked(path_self, mode: str = 'r', *args, **kwargs):
      del args, kwargs
      self.assertEqual(mode, 'rb')
      path_str = os.fspath(path_self)
      handle = _TrackedBinaryFile(path_str)
      with handles_lock:
        opened_paths.append(path_str)
        tracked_handles.append(handle)
      return handle

    self.enter_context(
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 128)
    )
    self.enter_context(
        mock.patch.object(safetensors_loader, '_READ_WORKERS', 4)
    )
    self.enter_context(
        mock.patch.object(
            path_cls, 'open', autospec=True, side_effect=open_tracked
        )
    )

    loaded_model = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.float32,
        mode='original',
    )

    self.assertTrue(all('/readahead/' not in p for p in opened_paths))
    self.assertTrue(all(h.closed_flag for h in tracked_handles))
    all_ranges = list(
        itertools.chain.from_iterable(h.read_ranges for h in tracked_handles)
    )
    self.assertGreater(len(all_ranges), len(self.tensors) + 50)
    for lo, hi in all_ranges:
      self.assertTrue(hi <= unmapped_abs_lo or lo >= unmapped_abs_hi)

    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(loaded_model)
    )

  @parameterized.named_parameters(
      dict(testcase_name='minus_one_read_error', fault_mode='minus_one'),
      dict(testcase_name='truncated_short_eof', fault_mode='short_eof'),
      dict(testcase_name='worker_oserror', fault_mode='raise_oserror'),
  )
  def test_original_loader_propagates_io_errors_and_closes_handles(
      self, fault_mode: str
  ):
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    tracked_handles: list[_TrackedBinaryFile] = []
    counter_lock = threading.Lock()
    shared_calls = [0]
    path_cls = type(epath.Path(filename))

    def open_faulty(path_self, mode: str = 'r', *args, **kwargs):
      del mode, args, kwargs
      handle = _TrackedBinaryFile(
          os.fspath(path_self),
          fault_mode=fault_mode,
          fault_trigger_call=2,
          counter_lock=counter_lock,
          shared_call_count=shared_calls,
      )
      with counter_lock:
        tracked_handles.append(handle)
      return handle

    self.enter_context(
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 128)
    )
    self.enter_context(
        mock.patch.object(safetensors_loader, '_READ_WORKERS', 4)
    )
    self.enter_context(
        mock.patch.object(
            path_cls, 'open', autospec=True, side_effect=open_faulty
        )
    )

    with self.assertRaises(OSError):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          dtype=jnp.float32,
          mode='original',
      )

    self.assertNotEmpty(tracked_handles)
    self.assertTrue(all(h.closed_flag for h in tracked_handles))

  def test_original_loader_supports_scalar_empty_unaligned_and_transformed_tensors(
      self,
  ):
    scalar_val = np.array(3.5, dtype=ml_dtypes.bfloat16)
    empty_val = np.zeros((0, 4), dtype=np.float32)
    codes_val = np.array([7, 8, 9], dtype=np.uint8)
    unaligned_val = np.array([1.25, -2.5, 3.75, 4.0], dtype=np.float32)
    proj_stored = np.arange(6, dtype=np.float16).reshape(2, 3)

    ordered = [
        ('a_codes', 'U8', [3], codes_val.tobytes()),
        ('b_unaligned', 'F32', [4], unaligned_val.tobytes()),
        ('c_empty', 'F32', [0, 4], empty_val.tobytes()),
        ('d_scalar', 'BF16', [], scalar_val.tobytes()),
        ('e_proj', 'F16', [2, 3], proj_stored.tobytes()),
    ]
    header: dict[str, object] = {'__metadata__': {'format': 'pt'}}
    payload = bytearray()
    for name, dtype_name, shape, raw_bytes in ordered:
      start = len(payload)
      payload.extend(raw_bytes)
      header[name] = {
          'dtype': dtype_name,
          'shape': shape,
          'data_offsets': [start, len(payload)],
      }
    header_bytes = json.dumps(header).encode('utf-8')
    filepath = os.path.join(self.st_dir, 'edge_cases.safetensors')
    with open(filepath, 'wb') as fh:
      fh.write(struct.pack('<Q', len(header_bytes)))
      fh.write(header_bytes)
      fh.write(payload)

    edge_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {'source_keys': ['a_codes'], 'target_keys': ['codes'], 'ops': []},
            {
                'source_keys': ['b_unaligned'],
                'target_keys': ['unaligned_f32'],
                'ops': [],
            },
            {'source_keys': ['c_empty'], 'target_keys': ['empty'], 'ops': []},
            {'source_keys': ['d_scalar'], 'target_keys': ['scalar'], 'ops': []},
            {
                'source_keys': ['e_proj'],
                'target_keys': ['f16_proj'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'reshape': {'src_shape': [3, 2], 'dst_shape': [3, 2]}},
                ],
            },
        ],
    }

    loaded = safetensors_loader.load_and_create_model(
        self.st_dir,
        _EdgeCaseModel,
        config=None,
        manifest=edge_manifest,
        dtype=None,
        mode='original',
    )

    np.testing.assert_array_equal(loaded.codes.value, codes_val)
    np.testing.assert_array_equal(loaded.unaligned_f32.value, unaligned_val)
    self.assertEqual(loaded.empty.value.shape, (0, 4))
    self.assertEqual(loaded.scalar.value.shape, ())
    self.assertEqual(loaded.scalar.value.dtype, jnp.bfloat16)
    np.testing.assert_allclose(loaded.scalar.value, scalar_val)
    np.testing.assert_array_equal(loaded.f16_proj.value, proj_stored.T)

  def test_original_loader_raises_on_shape_mismatch_and_empty_dir(self):
    empty_dir = os.path.join(self.st_dir, 'empty')
    os.makedirs(empty_dir, exist_ok=True)
    with self.assertRaisesRegex(ValueError, 'No safetensors found'):
      safetensors_loader.load_and_create_model(
          empty_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mode='original',
      )
    with (
        mock.patch.object(env_utils, 'is_internal_env', return_value=False),
        mock.patch.object(
            env_utils, 'is_pathways_initialized', return_value=True
        ),
        self.assertRaisesRegex(ValueError, 'Optimized mode is not supported'),
    ):
      safetensors_loader.load_and_create_model(
          empty_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mode='optimized',
      )

    stnp.save_file(
        {'emb.embedding': np.zeros((4, 4), dtype=np.float32)},
        os.path.join(self.st_dir, 'bad_shape.safetensors'),
    )
    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mode='original',
      )

  @parameterized.named_parameters(
      dict(testcase_name='short_size_prefix', raw=b'short'),
      dict(
          testcase_name='truncated_metadata',
          raw=struct.pack('<Q', 64) + b'{"a": 1}',
      ),
  )
  def test_original_loader_raises_on_truncated_header(self, raw: bytes):
    with open(os.path.join(self.st_dir, 'truncated.safetensors'), 'wb') as fh:
      fh.write(raw)
    with self.assertRaisesRegex(OSError, 'Truncated safetensors'):
      safetensors_loader.read_safetensors_headers(self.st_dir)
    with self.assertRaisesRegex(OSError, 'Truncated safetensors'):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mode='original',
      )

  def test_original_loader_supports_multihost_sharding_and_local_device_default(
      self,
  ):
    stnp.save_file(self.tensors, os.path.join(self.st_dir, 'model.safetensors'))
    local_dev = jax.local_devices()[0]
    remote_dev = mock.create_autospec(jax.Device, instance=True)
    orig_devices = jax.devices

    with (
        mock.patch.object(
            jax,
            'devices',
            side_effect=lambda backend=None: (
                [remote_dev, local_dev]
                if backend is None
                else orig_devices(backend)
            ),
        ),
        mock.patch.object(jax, 'local_devices', return_value=[local_dev]),
    ):
      local_loaded = safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mesh=None,
          dtype=jnp.float32,
          mode='original',
      )
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(local_loaded)
    )

    mesh = jax.sharding.Mesh(
        np.array([local_dev]).reshape(1, 1), ('fsdp', 'tp')
    )
    orig_get_named_sharding = nnx.get_named_sharding

    def fake_multihost_named_sharding(state, target_mesh):
      real = orig_get_named_sharding(state, target_mesh)

      def to_non_addressable(leaf_sharding):
        s = mock.create_autospec(jax.sharding.NamedSharding, instance=True)
        s.is_fully_addressable = False
        s.addressable_devices_indices_map.side_effect = (
            leaf_sharding.addressable_devices_indices_map
        )
        return s

      return jax.tree.map(to_non_addressable, real)

    with (
        mock.patch.object(
            nnx, 'get_named_sharding', side_effect=fake_multihost_named_sharding
        ),
        mock.patch.object(
            jax, 'device_put', wraps=jax.device_put
        ) as mock_device_put,
        mock.patch.object(
            jax,
            'make_array_from_single_device_arrays',
            side_effect=lambda shape, s, arrays: arrays[0],
        ) as mock_from_single_dev,
    ):
      mh_loaded = safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mesh=mesh,
          dtype=jnp.float32,
          mode='original',
      )

    mock_device_put.assert_called_once()
    self.assertLen(mock_from_single_dev.call_args_list, len(self.tensors))
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(mh_loaded)
    )

  def test_host_sharded_reader_slices_transforms_and_skips_remote_moe_experts(
      self,
  ):
    class _HybridMoEModel(nnx.Module):

      def __init__(self, config, *, rngs: nnx.Rngs):
        del config, rngs
        self.q_proj = nnx.Param(
            jnp.zeros((8, 4, 2), dtype=jnp.float32),
            sharding=('fsdp', 'tp', None),
        )
        self.o_proj = nnx.Param(
            jnp.zeros((4, 2, 8), dtype=jnp.float32),
            sharding=('tp', None, 'fsdp'),
        )
        self.same_rank = nnx.Param(
            jnp.zeros((8, 4), dtype=jnp.float32),
            sharding=('fsdp', 'tp'),
        )
        self.fallback = nnx.Param(
            jnp.zeros((2, 8), dtype=jnp.float32),
            sharding=(None, 'fsdp'),
        )
        self.layers = nnx.List([
            nnx.Dict({
                'mlp': nnx.Dict({
                    'gate_proj': nnx.Param(
                        jnp.zeros((4, 8, 6), dtype=jnp.float32),
                        sharding=('fsdp', None, 'tp'),
                    ),
                })
            })
        ])

    q_src = np.arange(64, dtype=np.float32).reshape(8, 8)
    o_src = np.arange(64, dtype=np.float32).reshape(8, 8) + 100.0
    same_src = np.arange(32, dtype=np.float32).reshape(4, 8) + 200.0
    fallback_src = np.arange(16, dtype=np.float32).reshape(4, 4) + 300.0
    experts_src = [
        np.arange(48, dtype=np.float32).reshape(6, 8) + (e + 1) * 1000.0
        for e in range(4)
    ]
    raw_tensors_0 = {
        'a_exp0': experts_src[0],
        'b_exp1': experts_src[1],
        'c_exp2': experts_src[2],
        'e_q_proj': q_src,
        'f_o_proj': o_src,
    }
    raw_tensors_1 = {
        'd_exp3': experts_src[3],
        'g_same_rank': same_src,
        'h_fallback': fallback_src,
    }
    filepath = os.path.join(self.st_dir, 'hybrid_moe_0.safetensors')
    stnp.save_file(raw_tensors_0, filepath)
    stnp.save_file(
        raw_tensors_1, os.path.join(self.st_dir, 'hybrid_moe_1.safetensors')
    )

    with open(filepath, 'rb') as fh:
      header_size = struct.unpack('<Q', fh.read(8))[0]
      header = json.loads(fh.read(header_size))
    exp0_lo, exp0_hi = (
        8 + header_size + x for x in header['a_exp0']['data_offsets']
    )
    exp1_lo, exp1_hi = (
        8 + header_size + x for x in header['b_exp1']['data_offsets']
    )

    hybrid_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {
                'source_keys': ['e_q_proj'],
                'target_keys': ['q_proj'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'reshape': {'src_shape': [8, 8], 'dst_shape': [8, 4, 2]}},
                ],
            },
            {
                'source_keys': ['f_o_proj'],
                'target_keys': ['o_proj'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'reshape': {'src_shape': [8, 8], 'dst_shape': [4, 2, 8]}},
                ],
            },
            {
                'source_keys': ['g_same_rank'],
                'target_keys': ['same_rank'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'reshape': {'src_shape': [8, 4], 'dst_shape': [8, 4]}},
                ],
            },
            {
                'source_keys': ['h_fallback'],
                'target_keys': ['fallback'],
                'ops': [
                    {'reshape': {'src_shape': [4, 4], 'dst_shape': [2, 8]}}
                ],
            },
            {
                'source_keys': ['a_exp0', 'b_exp1', 'c_exp2', 'd_exp3'],
                'target_keys': ['layers.0.mlp.gate_proj'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'stack': {'axis': 0}},
                ],
            },
        ],
    }

    local_dev = jax.local_devices()[0]
    dev_tp0 = mock.create_autospec(jax.Device, instance=True)
    dev_tp1 = mock.create_autospec(jax.Device, instance=True)

    def make_host1_sharding(spec_names):
      s = mock.create_autospec(jax.sharding.NamedSharding, instance=True)

      def addr_map(shape):
        slices_tp0 = []
        slices_tp1 = []
        for dim, axis_name in zip(shape, spec_names):
          if axis_name == 'fsdp':
            slices_tp0.append(slice(dim // 2, dim))
            slices_tp1.append(slice(dim // 2, dim))
          elif axis_name == 'tp':
            slices_tp0.append(slice(0, dim // 2))
            slices_tp1.append(slice(dim // 2, dim))
          else:
            slices_tp0.append(slice(0, dim))
            slices_tp1.append(slice(0, dim))
        return {dev_tp0: tuple(slices_tp0), dev_tp1: tuple(slices_tp1)}

      s.addressable_devices_indices_map.side_effect = addr_map
      return s

    orig_get_named_sharding = nnx.get_named_sharding

    def fake_host1_shardings(state, target_mesh):
      real = orig_get_named_sharding(state, target_mesh)
      return jax.tree.map(
          lambda s: make_host1_sharding(tuple(s.spec)),
          real,
      )

    tracked_handles: list[_TrackedBinaryFile] = []
    path_cls = type(epath.Path(filepath))

    def open_tracked(path_self, mode: str = 'r', *args, **kwargs):
      del mode, args, kwargs
      handle = _TrackedBinaryFile(os.fspath(path_self))
      tracked_handles.append(handle)
      return handle

    placed_by_key: dict[tuple[int, ...], list[np.ndarray]] = {}

    def fake_device_put(shards, devices):
      self.assertTrue(all(d in (dev_tp0, dev_tp1) for d in devices))
      return [jnp.asarray(arr) for arr in shards]

    def fake_make_array(shape, sharding, dev_arrays):
      del sharding
      placed_by_key[shape] = [np.asarray(a) for a in dev_arrays]
      return jnp.concatenate(dev_arrays, axis=0)

    mesh = jax.sharding.Mesh(
        np.array([local_dev]).reshape(1, 1), ('fsdp', 'tp')
    )
    with (
        mock.patch.object(
            nnx, 'get_named_sharding', side_effect=fake_host1_shardings
        ),
        mock.patch.object(
            path_cls, 'open', autospec=True, side_effect=open_tracked
        ),
        mock.patch.object(jax, 'device_put', side_effect=fake_device_put),
        mock.patch.object(
            jax,
            'make_array_from_single_device_arrays',
            side_effect=fake_make_array,
        ),
    ):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          _HybridMoEModel,
          config=None,
          manifest=hybrid_manifest,
          mesh=mesh,
          dtype=jnp.float32,
          mode='original',
      )

    all_ranges = list(
        itertools.chain.from_iterable(
            h.read_ranges for h in tracked_handles if h._fh.name == filepath
        )
    )
    for lo, hi in all_ranges:
      self.assertTrue(hi <= exp0_lo or lo >= exp0_hi)
      self.assertTrue(hi <= exp1_lo or lo >= exp1_hi)

    q_full = q_src.T.reshape(8, 4, 2)
    np.testing.assert_array_equal(
        placed_by_key[(8, 4, 2)][0], q_full[4:8, 0:2, :]
    )
    np.testing.assert_array_equal(
        placed_by_key[(8, 4, 2)][1], q_full[4:8, 2:4, :]
    )

    o_full = o_src.T.reshape(4, 2, 8)
    np.testing.assert_array_equal(
        placed_by_key[(4, 2, 8)][0], o_full[0:2, :, 4:8]
    )
    np.testing.assert_array_equal(
        placed_by_key[(4, 2, 8)][1], o_full[2:4, :, 4:8]
    )

    same_full = same_src.T.reshape(8, 4)
    np.testing.assert_array_equal(placed_by_key[(8, 4)][0], same_full[4:8, 0:2])
    np.testing.assert_array_equal(placed_by_key[(8, 4)][1], same_full[4:8, 2:4])

    fb_full = fallback_src.reshape(2, 8)
    np.testing.assert_array_equal(placed_by_key[(2, 8)][0], fb_full[:, 4:8])

    exp_full = np.stack([e.T for e in experts_src], axis=0)
    np.testing.assert_array_equal(
        placed_by_key[(4, 8, 6)][0], exp_full[2:4, :, 0:3]
    )
    np.testing.assert_array_equal(
        placed_by_key[(4, 8, 6)][1], exp_full[2:4, :, 3:6]
    )

    self.assertEqual(
        safetensors_loader._resolve_target_slices(
            safetensors_loader._SourcePlan('q_proj'), (8, 4, 2), None
        ),
        (True, None),
    )

  def test_host_sharded_reader_scale_profiles(self):
    profiles = (
        ('0.6B', 28, 1024, 16, 8, 128, 3072, 0, 0, 2, 4, 1),
        ('30B_MoE', 48, 2048, 32, 4, 128, 0, 128, 768, 8, 4, 4),
        ('397B_Hybrid_MoE', 60, 4096, 32, 2, 256, 1024, 512, 1024, 16, 8, 8),
        ('2.8T_MoE', 80, 8192, 64, 8, 128, 2048, 512, 2048, 64, 8, 16),
    )
    for (
        name,
        layers,
        d_full,
        nq_full,
        nkv_full,
        h,
        fd_full,
        e_cnt,
        fm_full,
        hosts,
        tp,
        scale,
    ) in profiles:
      del layers
      d = d_full // scale
      nq = max(tp, nq_full // scale)
      nkv = max(1, nkv_full // scale)
      fd = fd_full // scale if fd_full else 0
      fm = fm_full // scale if fm_full else 0
      tensors = {
          'q': np.ones((nq * h, d), dtype=ml_dtypes.bfloat16),
          'k': np.ones((nkv * h, d), dtype=ml_dtypes.bfloat16),
          'v': np.ones((nkv * h, d), dtype=ml_dtypes.bfloat16),
          'o': np.ones((d, nq * h), dtype=ml_dtypes.bfloat16),
      }
      rules: list[dict[str, object]] = [
          {
              'source_keys': ['q'],
              'target_keys': ['attn.q'],
              'ops': [
                  {'transpose': {'axes': [1, 0]}},
                  {
                      'reshape': {
                          'src_shape': [d, nq * h],
                          'dst_shape': [d, nq, h],
                      }
                  },
              ],
          },
          {
              'source_keys': ['k'],
              'target_keys': ['attn.k'],
              'ops': [
                  {'transpose': {'axes': [1, 0]}},
                  {
                      'reshape': {
                          'src_shape': [d, nkv * h],
                          'dst_shape': [d, nkv, h],
                      }
                  },
              ],
          },
          {
              'source_keys': ['v'],
              'target_keys': ['attn.v'],
              'ops': [
                  {'transpose': {'axes': [1, 0]}},
                  {
                      'reshape': {
                          'src_shape': [d, nkv * h],
                          'dst_shape': [d, nkv, h],
                      }
                  },
              ],
          },
          {
              'source_keys': ['o'],
              'target_keys': ['attn.o'],
              'ops': [
                  {'transpose': {'axes': [1, 0]}},
                  {
                      'reshape': {
                          'src_shape': [nq * h, d],
                          'dst_shape': [nq, h, d],
                      }
                  },
              ],
          },
      ]
      keys = ['attn.q', 'attn.k', 'attn.v', 'attn.o']
      params = [
          jax.ShapeDtypeStruct((d, nq, h), jnp.bfloat16),
          jax.ShapeDtypeStruct((d, nkv, h), jnp.bfloat16),
          jax.ShapeDtypeStruct((d, nkv, h), jnp.bfloat16),
          jax.ShapeDtypeStruct((nq, h, d), jnp.bfloat16),
      ]
      specs_list = [
          ('fsdp', 'tp', None),
          ('fsdp', None, None),
          ('fsdp', None, None),
          ('tp', None, 'fsdp'),
      ]
      if fd > 0:
        for proj, sh, sp in (
            ('gate', (fd, d), ('fsdp', 'tp')),
            ('up', (fd, d), ('fsdp', 'tp')),
            ('down', (d, fd), ('tp', 'fsdp')),
        ):
          tensors[proj] = np.ones(sh, dtype=ml_dtypes.bfloat16)
          rules.append({
              'source_keys': [proj],
              'target_keys': [f'mlp.{proj}'],
              'ops': [{'transpose': {'axes': [1, 0]}}],
          })
          keys.append(f'mlp.{proj}')
          params.append(jax.ShapeDtypeStruct((sh[1], sh[0]), jnp.bfloat16))
          specs_list.append(sp)
      if e_cnt > 0:
        for e_i in range(e_cnt):
          tensors[f'e.{e_i:03d}.gate'] = np.ones(
              (fm, d), dtype=ml_dtypes.bfloat16
          )
          tensors[f'e.{e_i:03d}.up'] = np.ones(
              (fm, d), dtype=ml_dtypes.bfloat16
          )
          tensors[f'e.{e_i:03d}.down'] = np.ones(
              (d, fm), dtype=ml_dtypes.bfloat16
          )
        for kw in ('gate', 'up', 'down'):
          rules.append({
              'source_keys': [f'e.{e_i:03d}.{kw}' for e_i in range(e_cnt)],
              'target_keys': [f'layers.0.mlp.{kw}_proj'],
              'ops': [
                  {'transpose': {'axes': [1, 0]}},
                  {'stack': {'axis': 0}},
              ],
          })
        keys.extend([
            'layers.0.mlp.gate_proj',
            'layers.0.mlp.up_proj',
            'layers.0.mlp.down_proj',
        ])
        params.extend([
            jax.ShapeDtypeStruct((e_cnt, d, fm), jnp.bfloat16),
            jax.ShapeDtypeStruct((e_cnt, d, fm), jnp.bfloat16),
            jax.ShapeDtypeStruct((e_cnt, fm, d), jnp.bfloat16),
        ])
        specs_list.extend([
            ('fsdp', None, 'tp'),
            ('fsdp', None, 'tp'),
            ('fsdp', 'tp', None),
        ])

      manifest = {
          'schema_version': hf_mapping.SCHEMA_VERSION,
          'rules': rules,
      }
      fake_devs = [
          mock.create_autospec(jax.Device, instance=True) for _ in range(tp)
      ]

      def make_sharding(sp, n_hosts=hosts, n_tp=tp, devs=fake_devs):
        s = mock.create_autospec(jax.sharding.NamedSharding, instance=True)

        def addr_map(shape):
          res = {}
          for t_i, dev in enumerate(devs):
            sl = []
            for dim, ax in zip(shape, sp):
              if ax == 'fsdp':
                c = dim // n_hosts
                sl.append(slice(c, 2 * c))
              elif ax == 'tp':
                c = dim // n_tp
                sl.append(slice(t_i * c, (t_i + 1) * c))
              else:
                sl.append(slice(0, dim))
            res[dev] = tuple(sl)
          return res

        s.addressable_devices_indices_map.side_effect = addr_map
        return s

      shardings = [make_sharding(sp) for sp in specs_list]
      t_specs = safetensors_loader._build_target_specs(keys, params, shardings)
      p_file = epath.Path(os.path.join(self.st_dir, f'{name}.safetensors'))
      stnp.save_file(tensors, os.fspath(p_file))
      base_plan = safetensors_loader._plan_file_reads(
          p_file, manifest, [], None
      )
      base_out = safetensors_loader._load_file_tensors(
          p_file, manifest, None, [], None, preplanned=base_plan
      )
      shard_plan = safetensors_loader._plan_file_reads(
          p_file, manifest, [], t_specs
      )
      shard_out = safetensors_loader._load_file_tensors(
          p_file, manifest, None, [], t_specs, preplanned=shard_plan
      )

      base_bytes = sum(a.nbytes for a in base_out.values())
      shard_bytes = sum(a.nbytes for a in shard_out.values())
      self.assertEqual(base_bytes, shard_bytes * hosts)
      check_keys = ['attn.q', 'attn.o'] + (
          ['layers.0.mlp.gate_proj', 'layers.0.mlp.down_proj']
          if e_cnt > 0
          else []
      )
      for k in check_keys:
        np.testing.assert_array_equal(
            shard_out[k], base_out[k][t_specs[k].host_slices]
        )

  def test_shm_range_arena_coalesces_2mib_gaps_splits_16mib_and_unpacks(self):
    self.assertEqual(safetensors_loader._READ_CHUNK_BYTES, 16 * 1024 * 1024)
    self.assertEqual(safetensors_loader._MAX_COALESCE_GAP, 2 * 1024 * 1024)
    with safetensors_loader.ShmRangeReader(
        size_bytes=64 * 1024 * 1024
    ) as arena:
      self.assertTrue(os.path.exists(arena.shm_path))
      shm_path = arena.shm_path
      big_buf = arena.alloc((1, 20 * 1024 * 1024))
      buf_a = np.zeros((1, 1024), dtype=np.uint8)
      buf_b = np.zeros((2, 64), dtype=np.uint8)
      buf_c = np.zeros((1, 512), dtype=np.uint8)
      gap_ok = int(1.5 * 1024 * 1024)
      gap_far = int(2.5 * 1024 * 1024)
      off_a = 100
      off_b = off_a + 1024 + gap_ok
      span_b = 1 * 256 + 64
      off_c = off_b + span_b + gap_far
      chunks = [
          (off_c, 0, buf_c),
          (off_a, 0, buf_a),
          (off_b, 256, buf_b),
          (10 * 1024 * 1024, 0, big_buf),
          (0, 0, arena.alloc((1, 0))),
      ]
      tasks, unpacks = arena.plan_coalesced_tasks([([], chunks)])
      self.assertLen(tasks, 4)
      self.assertEqual(tasks[0][2], 16 * 1024 * 1024)
      self.assertEqual(tasks[1][2], 4 * 1024 * 1024)
      self.assertEqual(tasks[2][2], (off_b + span_b) - off_a)
      self.assertEqual(tasks[3][2], 512)
      self.assertLen(unpacks, 3)

      assert arena.shm_uint8 is not None
      rel_a, _, _, _, _ = unpacks[0]
      rel_b, _, _, _, _ = unpacks[1]
      rel_c, _, _, _, _ = unpacks[2]
      expected_a = np.arange(1024, dtype=np.uint8)
      expected_b0 = np.full(64, 7, dtype=np.uint8)
      expected_b1 = np.full(64, 9, dtype=np.uint8)
      expected_c = np.full(512, 13, dtype=np.uint8)
      arena.shm_uint8[rel_a : rel_a + 1024] = expected_a
      arena.shm_uint8[rel_b : rel_b + 64] = expected_b0
      arena.shm_uint8[rel_b + 256 : rel_b + 256 + 64] = expected_b1
      arena.shm_uint8[rel_c : rel_c + 512] = expected_c
      for item in unpacks:
        arena.unpack_one(item)
      np.testing.assert_array_equal(buf_a[0], expected_a)
      np.testing.assert_array_equal(buf_b[0], expected_b0)
      np.testing.assert_array_equal(buf_b[1], expected_b1)
      np.testing.assert_array_equal(buf_c[0], expected_c)
    self.assertFalse(os.path.exists(shm_path))

  def test_gcs_http_range_reader_header_cache_vip_striping_and_tail_hedging(
      self,
  ):
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)
    with open(filename, 'rb') as fh:
      file_bytes = fh.read()

    range_calls: list[tuple[str, str]] = []
    call_lock = threading.Lock()
    fail_once = [True]

    class _FakeHttpResponse:

      def __init__(
          self, status: int, body: bytes, start: int = 0, total_size: int = 0
      ):
        self.status = status
        self._body = body
        self._pos = 0
        self._headers = {'Content-Length': str(len(body))}
        if status == 206 and body:
          total = total_size or len(body)
          self._headers['Content-Range'] = (
              f'bytes {start}-{start + len(body) - 1}/{total}'
          )

      def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name, default)

      def read(self) -> bytes:
        return self._body

      def readinto(self, mv: memoryview[int]) -> int:
        n = min(len(mv), len(self._body) - self._pos)
        mv[:n] = self._body[self._pos : self._pos + n]
        self._pos += n
        return n

    class _FakeHttpsConn:

      def __init__(self, host: str, timeout: int = 15, context=None):
        del host, timeout, context
        self.sock = None
        self._range = ''

      def connect(self) -> None:
        pass

      def close(self) -> None:
        pass

      def request(
          self, method: str, path: str, headers: dict[str, str]
      ) -> None:
        del method
        self._range = headers['Range']
        with call_lock:
          range_calls.append((path, self._range))

      def getresponse(self) -> _FakeHttpResponse:
        with call_lock:
          if fail_once[0]:
            fail_once[0] = False
            return _FakeHttpResponse(503, b'retry')
        lo_str, hi_str = self._range.removeprefix('bytes=').split('-')
        lo, hi = int(lo_str), int(hi_str)
        return _FakeHttpResponse(
            206, file_bytes[lo : hi + 1], start=lo, total_size=len(file_bytes)
        )

    class _FakeGsDir:

      def glob(self, pattern: str):
        del pattern
        return ['gs://bucket/dir/model.safetensors']

    safetensors_io.THREAD_CONNS.clear()
    self.enter_context(
        mock.patch.dict(os.environ, {'TUNIX_GCS_TOKEN': 'test_bearer_token'})
    )
    self.enter_context(
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 2048)
    )
    self.enter_context(
        mock.patch.object(
            epath,
            'Path',
            side_effect=lambda p: (
                _FakeGsDir() if str(p) == 'gs://bucket/dir' else epath.Path(p)
            ),
        )
    )
    self.enter_context(
        mock.patch.object(
            safetensors_io.socket,
            'getaddrinfo',
            return_value=[
                (0, 0, 0, '', ('142.250.1.1', 443)),
                (0, 0, 0, '', ('142.250.1.2', 443)),
            ],
        )
    )
    self.enter_context(
        mock.patch.object(
            safetensors_io.socket,
            'create_connection',
            return_value=mock.create_autospec(
                safetensors_io.socket.socket, instance=True
            ),
        )
    )
    self.enter_context(
        mock.patch.object(
            safetensors_io.ssl,
            'create_default_context',
            return_value=mock.create_autospec(
                safetensors_io.ssl.SSLContext, instance=True
            ),
        )
    )
    self.enter_context(
        mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            side_effect=_FakeHttpsConn,
        )
    )

    loaded_model = safetensors_loader.load_and_create_model(
        'gs://bucket/dir',
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.float32,
        mode='original',
    )
    self.assertTrue(
        any(r == 'bytes=0-262143' for _, r in range_calls),
        msg=f'Expected 256 KiB 1-RTT header prefetch in {range_calls}',
    )
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(loaded_model)
    )
    safetensors_io.THREAD_CONNS.clear()

  def test_GOOGLE_INTERNAL_STOAGE_PATH_reader_flags_header_cache_and_multiprocess_shm_worker(self):
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)
    real_file = epath.Path(filename)

    flag_argv = safetensors_io.CnsPath.configure_flags()
    self.assertIsInstance(flag_argv, tuple)

    class _FakeCnsPath:

      def __init__(self, GOOGLE_INTERNAL_STOAGE_PATH_uri: str):
        self._uri = GOOGLE_INTERNAL_STOAGE_PATH_uri
        self.name = os.path.basename(GOOGLE_INTERNAL_STOAGE_PATH_uri)

      def __str__(self) -> str:
        return self._uri

      def __fspath__(self) -> str:
        return filename

      def open(self, mode: str = 'rb'):
        return real_file.open(mode)

    class _FakeCnsDir:

      def expanduser(self):
        return self

      def glob(self, pattern: str):
        del pattern
        return [_FakeCnsPath('/GOOGLE_INTERNAL_STOAGE_PATH/cell/dir/model.safetensors')]

    class _FakeProc:

      def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO('READY\n')
        self.killed = False

      def kill(self) -> None:
        self.killed = True

      def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0

    safetensors_io.CNS_PROCS.clear()
    self.addCleanup(safetensors_io.CNS_PROCS.clear)
    fake_proc = _FakeProc()
    self.enter_context(mock.patch.dict(os.environ, {'TUNIX_CNS_PROCS': '1'}))
    self.enter_context(
        mock.patch.object(
            safetensors_io.subprocess,
            'Popen',
            return_value=fake_proc,
        )
    )
    real_path = epath.Path
    self.enter_context(
        mock.patch.object(
            epath,
            'Path',
            side_effect=lambda p: (
                _FakeCnsDir() if str(p) == '/GOOGLE_INTERNAL_STOAGE_PATH/cell/dir' else real_path(p)
            ),
        )
    )
    resolved = safetensors_io.resolve_range_sources('/GOOGLE_INTERNAL_STOAGE_PATH/cell/dir')
    self.assertLen(resolved, 1)
    self.assertIsInstance(resolved[0], safetensors_io.CnsPath)
    self.assertEqual(os.fspath(resolved[0]), filename)
    self.assertEqual(str(resolved[0]), '/GOOGLE_INTERNAL_STOAGE_PATH/cell/dir/model.safetensors')
    self.assertLen(safetensors_io.CNS_PROCS, 1)

    safetensors_io.CNS_PROCS.clear()
    self.enter_context(mock.patch.dict(os.environ, {'TUNIX_CNS_PROCS': '0'}))
    loaded_model = safetensors_loader.load_and_create_model(
        '/GOOGLE_INTERNAL_STOAGE_PATH/cell/dir',
        test_common.ToyTransformer,
        self.model.config,
        self.manifest,
        dtype=jnp.float32,
        mode='original',
    )
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(loaded_model)
    )

    with safetensors_loader.ShmRangeReader(size_bytes=65536) as arena:
      assert arena.shm_uint8 is not None
      cmd_line = (
          f'{arena.shm_path}\t{arena.capacity}\t{filename}\t0\t64\t128\n\n'
      )
      worker_out = io.StringIO()
      with (
          mock.patch.object(safetensors_io.sys, 'stdin', io.StringIO(cmd_line)),
          mock.patch.object(safetensors_io.sys, 'stdout', worker_out),
      ):
        safetensors_io.CnsPath.worker_loop()
      lines = worker_out.getvalue().splitlines()
      self.assertEqual(lines[0], 'READY')
      self.assertEqual(lines[1:], ['64'])
      with open(filename, 'rb') as fh:
        expected_64 = np.frombuffer(fh.read(64), dtype=np.uint8)
      np.testing.assert_array_equal(arena.shm_uint8[128:192], expected_64)

      proc_io = _FakeProc()
      proc_io.stdout = io.StringIO('32\n')
      GOOGLE_INTERNAL_STOAGE_PATH_fh = safetensors_io.CnsFile(
          resolved[0], None, (proc_io, threading.Lock())
      )
      n = GOOGLE_INTERNAL_STOAGE_PATH_fh.read_shm_range(
          16, 32, arena.shm_uint8, 256, arena.shm_path, arena.capacity
      )
      self.assertEqual(n, 32)
      GOOGLE_INTERNAL_STOAGE_PATH_fh.done_flags = bytearray([1])
      GOOGLE_INTERNAL_STOAGE_PATH_fh.task_idx = 0
      self.assertEqual(
          GOOGLE_INTERNAL_STOAGE_PATH_fh.read_shm_range(
              16, 32, arena.shm_uint8, 256, arena.shm_path, arena.capacity
          ),
          0,
      )
      GOOGLE_INTERNAL_STOAGE_PATH_fh.close()

      with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop('TUNIX_CNS_PROCS', None)
        os.environ.pop('TUNIX_GCS_TOKEN', None)
        os.environ.pop('TOKEN', None)
        self.assertEqual(resolved[0].ensure_procs(1024), 0)
        with self.assertRaises(OSError):
          resolved[0].ensure_procs(64 * 1024 * 1024)
        fake_proc.stdout = io.StringIO('READY\nREADY\n')
        self.assertEqual(resolved[0].ensure_procs(128 * 1024 * 1024), 2)
        safetensors_io.CNS_PROCS.clear()

        class _MetaResp:
          status = 200

          def read(self) -> bytes:
            return b'{"access_token": "meta_token_123"}'

        class _MetaConn:

          def __init__(self, host: str, timeout: float = 0.5):
            del host, timeout

          def request(self, method: str, path: str, headers: dict[str, str]):
            del method, path, headers

          def getresponse(self) -> _MetaResp:
            return _MetaResp()

          def close(self) -> None:
            pass

        safetensors_io.GCS_TOKEN_CACHE.clear()
        with mock.patch.object(
            safetensors_io.http.client,
            'HTTPConnection',
            side_effect=_MetaConn,
        ):
          self.assertEqual(
              safetensors_io.GsHttpPath.resolve_token(), 'meta_token_123'
          )
          self.assertEqual(
              safetensors_io.GsHttpPath.resolve_token(), 'meta_token_123'
          )
        safetensors_io.GCS_TOKEN_CACHE.clear()
        with mock.patch.object(
            safetensors_io.http.client,
            'HTTPConnection',
            side_effect=OSError('no metadata server'),
        ):
          self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), '')

  def test_streaming_pipeline_and_sliding_window_shm_slot_recycling(self):
    items = sorted(
        (
            k,
            np.asarray(
                v.value if isinstance(v, nnx.Param) else v, dtype=np.float32
            ),
        )
        for k, v in self.tensors.items()
    )
    q1, q2, q3 = len(items) // 4, len(items) // 2, (3 * len(items)) // 4
    for idx, sub in enumerate(
        (items[:q1], items[q1:q2], items[q2:q3], items[q3:])
    ):
      stnp.save_file(
          dict(sub),
          os.path.join(self.st_dir, f'model-{idx:05d}-of-00004.safetensors'),
      )

    ready_order: list[int] = []
    wrap_observed = [False]
    unit_releases = [0]
    orig_release = safetensors_io.ShmArena.release_file
    orig_release_unit = safetensors_io.ShmArena.release_unit

    def tracked_release(arena_self, f_idx: int, total_files: int = 0):
      ready_order.append(f_idx)
      if sum(arena_self.file_shm_bytes.values()) > arena_self.ring_capacity:
        wrap_observed[0] = True
      orig_release(arena_self, f_idx, total_files)

    def tracked_release_unit(arena_self, u_idx: int):
      unit_releases[0] += 1
      orig_release_unit(arena_self, u_idx)

    with (
        mock.patch.dict(os.environ, {'TUNIX_SHM_BUDGET_BYTES': '16384'}),
        mock.patch.object(
            safetensors_io.ShmArena, 'release_file', tracked_release
        ),
        mock.patch.object(
            safetensors_io.ShmArena,
            'release_unit',
            tracked_release_unit,
        ),
    ):
      loaded_model = safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          dtype=jnp.float32,
          mode='original',
      )

    self.assertEqual(ready_order, [0, 1, 2, 3])
    self.assertTrue(wrap_observed[0])
    self.assertGreater(unit_releases[0], 4)
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(loaded_model)
    )

    with safetensors_loader.ShmRangeReader(shm_budget_bytes=32 * 1024**2) as ar:
      b4m_a = np.zeros((1, 4 * 1024 * 1024), dtype=np.uint8)
      b4m_b = np.zeros((1, 4 * 1024 * 1024), dtype=np.uint8)
      tasks_4m, _ = ar.plan_coalesced_tasks(
          [([], [(0, 0, b4m_a), (4 * 1024 * 1024 + 4096, 0, b4m_b)])]
      )
      self.assertLen(tasks_4m, 2)
      self.assertEqual(tasks_4m[0][2], 4 * 1024 * 1024)
      self.assertEqual(tasks_4m[1][2], 4 * 1024 * 1024)

    dev_a, dev_b = object(), object()
    put_batches: list[int] = []
    with mock.patch.object(
        jax,
        'device_put',
        side_effect=lambda s, d: (put_batches.append(len(s)) or s),
    ):
      placed = safetensors_loader.SpmdHostStager.parallel_device_put(
          [
              np.ones((2, 3), dtype=np.float32),
              np.zeros((3, 2), dtype=np.float32),
          ],
          cast(Any, [dev_a, dev_b]),
          [(1, 0), (1, 0)],
      )
    self.assertEqual(sorted(put_batches), [1, 1])
    self.assertEqual(placed[0].shape, (3, 2))
    self.assertEqual(placed[1].shape, (2, 3))

  def test_three_layer_modularity_safetensors_io_and_shm_moe_allocation(self):
    # 1. `safetensors_io` wire format & dtype codec round-trip:
    for wire, expected_dt in (
        ('BF16', ml_dtypes.bfloat16),
        ('F16', np.float16),
        ('F32', np.float32),
        ('F64', np.float64),
        ('I32', np.dtype(np.int32)),
    ):
      self.assertEqual(safetensors_io.to_np_dtype(wire), expected_dt)
      self.assertEqual(safetensors_io.from_np_dtype(expected_dt), wire)
    self.assertEqual(
        safetensors_io.to_np_dtype(jnp.bfloat16), ml_dtypes.bfloat16
    )
    self.assertEqual(safetensors_io.to_np_dtype(jnp.float16), np.float16)
    self.assertEqual(safetensors_io.to_np_dtype(jnp.float32), np.float32)
    self.assertEqual(safetensors_io.to_np_dtype(jnp.float64), np.float64)
    self.assertEqual(safetensors_io.to_np_dtype(jnp.int32), np.int32)

    hdr_bytes = safetensors_io.format_safetensors_header(
        {'w': {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]}},
        metadata={'format': 'pt'},
    )
    self.assertEqual(len(hdr_bytes) % 8, 0)
    hdr_file = os.path.join(self.st_dir, 'hdr_only.safetensors')
    with open(hdr_file, 'wb') as fh:
      fh.write(hdr_bytes + b'\x00' * 16)
    hdr_size, parsed_hdr = safetensors_io.read_safetensors_header(
        epath.Path(hdr_file)
    )
    self.assertEqual(hdr_size, len(hdr_bytes) - 8)
    self.assertEqual(parsed_hdr['__metadata__']['format'], 'pt')

    # 2. Layer 3 orthogonality & Layer 1 `ShmArena` inheritance:
    self.assertTrue(
        issubclass(safetensors_loader.ShmRangeReader, safetensors_io.ShmArena)
    )
    self.assertTrue(
        issubclass(safetensors_io.CnsFile, safetensors_io.CachedRangeFile)
    )
    self.assertTrue(
        issubclass(safetensors_io.GsHttpFile, safetensors_io.CachedRangeFile)
    )
    self.assertFalse(
        issubclass(safetensors_io.CnsFile, safetensors_io.GsHttpFile)
    )
    self.assertFalse(safetensors_io.RangeSource.supports_hedge)
    self.assertFalse(safetensors_io.CnsPath.supports_hedge)
    self.assertTrue(safetensors_io.GsHttpPath.supports_hedge)

    # 3. Layer 1 hardware/framework independence:
    payload = np.arange(512, dtype=np.uint8)

    class _MemPath:
      name = 'in_mem.bin'

      def open(self, mode: str = 'rb'):
        del mode
        return io.BytesIO(payload.tobytes())

    mem_src = safetensors_io.RangeSource(_MemPath())
    dst_a = np.zeros((1, 128), dtype=np.uint8)
    dst_b = np.zeros((1, 128), dtype=np.uint8)
    ready_files: list[int] = []
    with (
        mock.patch.object(jax, 'device_put') as mock_put,
        safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena,
    ):
      tasks, _ = arena.plan_coalesced_tasks(
          [([], [(0, 0, dst_a), (256, 0, dst_b)])]
      )
      arena.execute_tasks([mem_src], tasks, on_file_ready=ready_files.append)
      mock_put.assert_not_called()
    self.assertEqual(ready_files, [0])
    np.testing.assert_array_equal(dst_a[0], payload[:128])
    np.testing.assert_array_equal(dst_b[0], payload[256:384])

    # 4. Layer 2 `/dev/shm` 3D MoE expert buffer allocation via `RuleSpec`:
    moe_file = os.path.join(self.st_dir, 'moe_shm.safetensors')
    stnp.save_file(
        {
            'e.0.gate': np.ones((4, 8), dtype=np.float32),
            'e.1.gate': np.full((4, 8), 2.0, dtype=np.float32),
        },
        moe_file,
    )
    moe_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [{
            'source_keys': ['e.0.gate', 'e.1.gate'],
            'target_keys': ['layers.0.mlp.gate_proj'],
            'ops': [
                {'transpose': {'axes': [1, 0]}},
                {'stack': {'axis': 0}},
            ],
        }],
    }
    dev0 = jax.local_devices()[0]
    spec = safetensors_loader._TargetSpec(
        0,
        (2, 8, 4),
        (2, 8, 4),
        dev0,
        (slice(0, 2), slice(0, 8), slice(0, 4)),
        ((dev0, (slice(0, 2), slice(0, 8), slice(0, 4))),),
    )
    moe_bufs: dict[str, np.ndarray] = {}
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena:
      safetensors_loader._plan_file_reads(
          safetensors_io.RangeSource(epath.Path(moe_file)),
          moe_manifest,
          [],
          {'layers.0.mlp.gate_proj': spec},
          moe_bufs,
          arena,
      )
      moe_arr = moe_bufs['layers.0.mlp.gate_proj']
      self.assertGreaterEqual(moe_arr.ctypes.data, arena.base_ptr)
      self.assertLess(moe_arr.ctypes.data, arena.base_ptr + arena.capacity)

  def test_direct_coalescing_staggered_task_order_and_identity_transpose(self):
    # 1. Adjacent direct /dev/shm spans coalesce into fewer <=sub_chunk tasks.
    with (
        mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 64),
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 256),
        safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena,
    ):
      buf = arena.alloc((2, 128))
      raw_tasks = []
      staged = arena.split_chunk_spans(
          0, [(1000, 0, buf[0:1]), (1128, 0, buf[1:2])], raw_tasks
      )
      self.assertEmpty(staged)
      self.assertLen(raw_tasks, 1)
      self.assertEqual(raw_tasks[0][1], 1000)
      self.assertEqual(raw_tasks[0][2], 256)

    # 2. Multi-host staggered task ordering rotates files per host while
    # respecting ring-capacity windows.
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena:
      arena.ring_capacity = 1024
      arena.unit_file = [0, 1, 2, 3]
      arena.unit_shm_bytes = [200, 200, 400, 200]
      arena.task_unit = [0, 0, 1, 1, 2, 3]
      tasks = [
          (0, 0, 64, 0),
          (0, 64, 64, 64),
          (1, 0, 64, 128),
          (1, 64, 64, 192),
          (2, 0, 64, 256),
          (3, 0, 64, 320),
      ]
      with mock.patch.object(jax, 'process_index', return_value=0):
        order_h0 = arena._build_staggered_task_order(tasks)
      with mock.patch.object(jax, 'process_index', return_value=1):
        order_h1 = arena._build_staggered_task_order(tasks)
      self.assertCountEqual(order_h0, range(len(tasks)))
      self.assertCountEqual(order_h1, range(len(tasks)))
      # Window 0 = files 0 & 1 (200+200 <= 512), round-robined and rotated.
      self.assertEqual(order_h0[:4], [0, 2, 1, 3])
      self.assertEqual(order_h1[:4], [2, 0, 3, 1])
      # Window 1 = file 2 (400 <= 512), Window 2 = file 3.
      self.assertEqual(order_h0[4:], [4, 5])
      self.assertEqual(order_h1[4:], [4, 5])

    # 3. Identity transpose short-circuit skips XLA transpose op.
    arr = jnp.ones((2, 3), dtype=jnp.float32)
    same = safetensors_loader.SpmdHostStager._maybe_transpose(arr, (0, 1))
    self.assertIs(same, arr)
    transposed = safetensors_loader.SpmdHostStager._maybe_transpose(arr, (1, 0))
    self.assertEqual(transposed.shape, (3, 2))

  def test_load_and_create_model_opt_and_flatten_state(self):
    tensors = dict(self.tensors)
    tensors['unused_multimodal_key'] = np.zeros((2,), dtype=np.float32)
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    loaded_model = safetensors_loader.load_and_create_model_opt(
        self.st_dir,
        test_common.ToyTransformer,
        {},
        self.model.config,
        self.manifest,
        mesh=None,
        dtype=jnp.float32,
    )
    expected_flat = safetensors_loader.flatten_model_state(self.model)
    loaded_flat = safetensors_loader.flatten_model_state(loaded_model)
    self.assertEqual(expected_flat.keys(), loaded_flat.keys())
    for k, exp_arr in expected_flat.items():
      np.testing.assert_array_equal(
          np.asarray(exp_arr), np.asarray(loaded_flat[k])
      )

  def test_hardening_moe_commit_order_3d_slices_and_hedged_shm_pinning(self):
    # 1. align_moe_commit_files moves MoE commit to the highest file writing it.
    moe_buf = np.zeros((2, 4, 4), dtype=np.uint8)
    ht = safetensors_loader._HostTensor(
        'layers.0.mlp.gate',
        None,
        moe_buf.ravel(),
        np.dtype(np.uint8),
        (2, 4, 4),
    )
    planned_all = [
        ([ht], [(0, 4, moe_buf[0])]),
        ([], []),
        ([], [(16, 4, moe_buf[1])]),
    ]
    safetensors_loader.SpmdHostStager.align_moe_commit_files(
        planned_all, {'layers.0.mlp.gate': moe_buf}
    )
    self.assertEmpty(planned_all[0][0])
    self.assertEqual(planned_all[2][0], [ht])

    # 2. 3D compute_slice_geometry includes trailing dims;
    # _invert_transform_slices falls back cleanly on sliced axes >= 2.
    shard_shape, first_off, outer_count, row_stride, row_bytes = (
        safetensors_loader.SpmdHostStager.compute_slice_geometry(
            100,
            (4, 3, 2),
            np.dtype(np.float32),
            (slice(1, 3), slice(1, 3), slice(0, 2)),
        )
    )
    self.assertEqual(shard_shape, (2, 2, 2))
    self.assertEqual(row_bytes, 2 * 2 * 4)
    self.assertEqual(row_stride, 3 * 2 * 4)
    self.assertEqual(outer_count, 2)
    self.assertEqual(first_off, 100 + 1 * 24 + 1 * 8)

    plan_3d = safetensors_loader._SourcePlan(target_key='w3d', transform=None)
    spec_3d = safetensors_loader._TargetSpec(
        param_index=0,
        global_shape=(4, 3, 2),
        host_shape=(4, 3, 1),
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        host_slices=(slice(0, 4), slice(0, 3), slice(0, 1)),
        device_slices=(),
    )
    should_read, src_sl, _, post_sl = (
        safetensors_loader._invert_transform_slices(
            plan_3d,
            (4, 3, 2),
            {'w3d': spec_3d},
        )
    )
    self.assertTrue(should_read)
    self.assertEqual(src_sl, (slice(0, 4), slice(0, 3), slice(0, 2)))
    self.assertEqual(post_sl, (slice(0, 4), slice(0, 3), slice(0, 1)))

    # 3. ShmRangeReader.reset() + deferred release_unit prevents UAF while a
    # hedged reader is still pinned on unit 0.
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=8192) as arena:
      arena.ring_capacity = 4096
      arena.unit_shm_bytes = [2048, 2048]
      arena.unit_active_readers = [0, 0]
      arena.unit_consumed = [False, False]
      arena.admit_units()
      done = bytearray([0])
      ok, tracked = arena._acquire_unit_reader(0, 0, done)
      self.assertTrue(ok and tracked)
      arena.release_unit(0)
      self.assertEqual(arena.in_flight_bytes, 4096)
      # Out-of-order release of unit 1 while unit 0 is still pinned must NOT
      # reclaim ring bytes until unit 0's hedged reader exits (FIFO safety).
      arena.release_unit(1)
      self.assertEqual(arena.in_flight_bytes, 4096)
      arena._release_unit_reader(0)
      self.assertEqual(arena.in_flight_bytes, 0)
      arena.reset()
      self.assertEqual(arena.cursor, 0)
      self.assertEqual(arena.oldest_unreleased_unit, 0)
      self.assertEmpty(arena.unit_active_readers)

  def test_circular_ring_wrap_direct_moe_strided_and_gcs_pool(self):
    # Create 4 shards exercising ALL buffer classes in 1 continuous wrap pass:
    # - Direct dense tensors (>= _DIRECT_SLICE_BYTES=512)
    # - Cross-file 3D MoE experts (e.0 in shard 0, e.1 in shard 2 -> commit 2)
    # - Strided/small tensors (< _DIRECT_SLICE_BYTES)
    # - Composite raw-source tensor (heap-allocated, never aliased on wrap)
    dense0 = np.arange(256, dtype=np.float32).reshape(16, 16)
    dense1 = np.arange(256, 512, dtype=np.float32).reshape(16, 16)
    dense2 = np.arange(512, 768, dtype=np.float32).reshape(16, 16)
    dense3 = np.arange(768, 1280, dtype=np.float32).reshape(32, 16)
    dense4 = np.arange(1280, 1792, dtype=np.float32).reshape(32, 16)
    dense5 = np.arange(1792, 2304, dtype=np.float32).reshape(32, 16)
    dense6 = np.arange(2304, 2816, dtype=np.float32).reshape(32, 16)
    exp0 = np.full((16, 16), 3.0, dtype=np.float32)
    exp1 = np.full((16, 16), 7.0, dtype=np.float32)
    small0 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    comp_a = np.array([10.0, 20.0], dtype=np.float32)
    comp_b = np.array([30.0, 40.0], dtype=np.float32)

    shard_dicts = [
        {'d0': dense0, 'e.0': exp0, 's0': small0, 'ca': comp_a},
        {'d1': dense1},
        {'d2': dense2, 'e.1': exp1},
        {'d3': dense3, 'cb': comp_b},
        {'d4': dense4},
        {'d5': dense5},
        {'d6': dense6},
    ]
    shard_paths = []
    for i, sd in enumerate(shard_dicts):
      p = os.path.join(self.st_dir, f'ring-{i:05d}-of-00007.safetensors')
      stnp.save_file(sd, p)
      shard_paths.append(p)

    manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {'source_keys': ['d0'], 'target_keys': ['d0'], 'ops': []},
            {'source_keys': ['d1'], 'target_keys': ['d1'], 'ops': []},
            {'source_keys': ['d2'], 'target_keys': ['d2'], 'ops': []},
            {'source_keys': ['d3'], 'target_keys': ['d3'], 'ops': []},
            {'source_keys': ['d4'], 'target_keys': ['d4'], 'ops': []},
            {'source_keys': ['d5'], 'target_keys': ['d5'], 'ops': []},
            {'source_keys': ['d6'], 'target_keys': ['d6'], 'ops': []},
            {'source_keys': ['s0'], 'target_keys': ['s0'], 'ops': []},
            {
                'source_keys': ['e.0', 'e.1'],
                'target_keys': ['moe'],
                'ops': [{'stack': {'axis': 0}}],
            },
            {
                'source_keys': ['ca', 'cb'],
                'target_keys': ['cat_ab'],
                'ops': [{'concat': {'axis': 0, 'sizes': [2, 2]}}],
            },
        ],
    }
    source_map, comp_sources, comp_exec = (
        safetensors_loader.SpmdHostStager.compile_manifest(manifest)
    )
    dev0 = jax.local_devices()[0]
    keys = [
        'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 's0', 'moe', 'cat_ab',
    ]
    shapes = [
        (16, 16), (16, 16), (16, 16),
        (32, 16), (32, 16), (32, 16), (32, 16),
        (4,), (2, 16, 16), (4,),
    ]
    expected = {
        'd0': dense0,
        'd1': dense1,
        'd2': dense2,
        'd3': dense3,
        'd4': dense4,
        'd5': dense5,
        'd6': dense6,
        's0': small0,
        'moe': np.stack([exp0, exp1], axis=0),
        'cat_ab': np.concatenate([comp_a, comp_b], axis=0),
    }

    # 1. Verify BOTH 3D MoE inside /dev/shm ring AND circular ring wrap:
    # With capacity=12288, MoE span_est(0..2)=5248 <= 6144 (ring_eligible=True)
    # while total_virt_bytes > 12288 (ring_wraps=True).
    params_local = [jax.ShapeDtypeStruct(s, jnp.float32) for s in shapes]
    target_specs = safetensors_loader._build_target_specs(
        keys, params_local, [dev0] * len(keys)
    )
    local_sources = [
        safetensors_io.RangeSource(epath.Path(p)) for p in shard_paths
    ]
    with (
        mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 512),
        safetensors_loader.ShmRangeReader(shm_budget_bytes=12288) as arena,
    ):
      items_per_file = [
          safetensors_loader.SpmdHostStager.inspect_file_items(
              src, idx, source_map, [], target_specs
          )
          for idx, src in enumerate(local_sources)
      ]
      ring_plan = arena.plan_circular_ring(items_per_file)
      self.assertIsNotNone(ring_plan)
      assert ring_plan is not None
      self.assertTrue(ring_plan.ring_wraps)
      # MoE buffer allocated inside /dev/shm ring and committed at file 2.
      moe_arr = ring_plan.moe_bufs['moe']
      self.assertGreaterEqual(moe_arr.ctypes.data, arena.base_ptr)
      self.assertLess(moe_arr.ctypes.data, arena.base_ptr + arena.capacity)
      self.assertIn(0, ring_plan.group_deps[2])
      self.assertIn(2, ring_plan.pinned_commits[0])
      safetensors_loader.stream_load_files(
          local_sources,
          source_map,
          comp_sources,
          comp_exec,
          np.dtype(np.float32),
          target_specs,
          keys,
          params_local,
          arena=arena,
      )
    for k, arr in zip(keys, params_local):
      np.testing.assert_array_equal(np.asarray(arr), expected[k])

    # Also verify tight-budget wrap where MoE safely spills to heap.
    params_tight = [jax.ShapeDtypeStruct(s, jnp.float32) for s in shapes]
    _, _, comp_exec_tight = safetensors_loader.SpmdHostStager.compile_manifest(
        manifest
    )
    with (
        mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 512),
        safetensors_loader.ShmRangeReader(shm_budget_bytes=5120) as arena,
    ):
      safetensors_loader.stream_load_files(
          local_sources,
          source_map,
          comp_sources,
          comp_exec_tight,
          np.dtype(np.float32),
          target_specs,
          keys,
          params_tight,
          arena=arena,
      )
    for k, arr in zip(keys, params_tight):
      np.testing.assert_array_equal(np.asarray(arr), expected[k])

    # 2. Verify GCS GcsShmReadPool end-to-end streaming over circular ring +
    # 0.60s urgent hedge trigger helper.
    raw_bytes_by_http = {
        f'/bkt/ring-{i}.st': pathlib.Path(p).read_bytes()
        for i, p in enumerate(shard_paths)
    }

    class _FakeGcsResp:

      def __init__(self, data: bytes, start: int = 0, total_size: int = 0):
        self.status = 206
        self._data = data
        self._pos = 0
        self._headers = {'Content-Length': str(len(data))}
        if data:
          total = total_size or len(data)
          self._headers['Content-Range'] = (
              f'bytes {start}-{start + len(data) - 1}/{total}'
          )

      def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._headers.get(name, default)

      def readinto(self, mv) -> int:
        n = min(len(mv), len(self._data) - self._pos)
        if n > 0:
          mv[:n] = self._data[self._pos : self._pos + n]
          self._pos += n
        return n

    class _FakeGcsConn:

      def __init__(self):
        self.timeout = 4.0
        self.sock = None
        self._req = ('', {})

      def request(self, method: str, path: str, headers: dict[str, str]):
        del method
        self._req = (path, headers)

      def getresponse(self):
        path, hdrs = self._req
        lo, hi = (
            int(x) for x in hdrs['Range'].removeprefix('bytes=').split('-')
        )
        raw = raw_bytes_by_http[path]
        return _FakeGcsResp(raw[lo : hi + 1], start=lo, total_size=len(raw))

      def close(self):
        pass

    pool = safetensors_io.GcsShmReadPool(
        1, ('10.0.0.1', '10.0.0.2'), streams_per_proc=2, spawn=False
    )
    gs_sources = [
        safetensors_io.GsHttpPath(
            f'gs://bkt/ring-{i}.st', 'tok', ('10.0.0.1', '10.0.0.2')
        )
        for i in range(len(shard_paths))
    ]
    params_gcs = [jax.ShapeDtypeStruct(s, jnp.float32) for s in shapes]
    _, _, comp_exec_gcs = safetensors_loader.SpmdHostStager.compile_manifest(
        manifest
    )
    with (
        mock.patch.object(
            safetensors_io,
            'get_thread_https_conn',
            side_effect=lambda vip, timeout=15.0: _FakeGcsConn(),
        ),
        mock.patch.object(
            safetensors_io, 'ensure_gcs_read_pool', return_value=pool
        ),
        mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 512),
    ):
      w_th = threading.Thread(
          target=safetensors_io.run_gcs_shm_worker_cli,
          args=([
              '--gcs-shm-worker',
              pool.ctrl_path,
              '0',
              '2',
              '10.0.0.1,10.0.0.2',
          ],),
          daemon=True,
      )
      w_th.start()
      try:
        while int(pool.ctrl.hdr.ready_workers) < 1:
          threading.Event().wait(0.005)
        with safetensors_loader.ShmRangeReader(shm_budget_bytes=12288) as arena:
          safetensors_loader.stream_load_files(
              gs_sources,
              source_map,
              comp_sources,
              comp_exec_gcs,
              np.dtype(np.float32),
              target_specs,
              keys,
              params_gcs,
              arena=arena,
              read_procs=1,
          )
        # Verify _poll_gcs_groups triggers urgent hedge after 0.60s stall.
        dummy_plan = safetensors_loader._CircularRingPlan(
            [], {}, [[]], [], [1], [10], [0, 10], [0], [64],
            [frozenset({0})], [frozenset()], 64, False,
        )
        pool.ctrl.hdr.task_counter = 10
        pool.ctrl.group_rem[0] = 2
        pool.ctrl.hdr.urgent_group_end = 0
        tail_since = [0.0]
        dl_done = [False]
        safetensors_loader.ShmRangeReader._poll_gcs_groups(
            pool, dummy_plan, dl_done, tail_since
        )
        self.assertGreater(tail_since[0], 0.0)
        tail_since[0] -= 1.0
        safetensors_loader.ShmRangeReader._poll_gcs_groups(
            pool, dummy_plan, dl_done, tail_since
        )
        self.assertEqual(int(pool.ctrl.hdr.urgent_group_end), 10)
      finally:
        pool.shutdown()
        w_th.join(timeout=5.0)
    for k, arr in zip(keys, params_gcs):
      np.testing.assert_array_equal(np.asarray(arr), expected[k])

    # 3. Empty-tasks fast path (run_all_empty) on both threaded & GCS ring.
    empty_file = os.path.join(self.st_dir, 'unmapped_only.safetensors')
    stnp.save_file(
        {'unrelated_key': np.zeros((2,), dtype=np.float32)}, empty_file
    )
    safetensors_loader.stream_load_files(
        [safetensors_io.RangeSource(epath.Path(empty_file))],
        source_map,
        comp_sources,
        None,
        np.dtype(np.float32),
        target_specs,
        keys,
        params_local,
    )
    safetensors_loader.stream_load_files(
        [],
        source_map,
        comp_sources,
        None,
        None,
        target_specs,
        keys,
        params_local,
    )

    # 4. Exercise ShmRangeReader._hedge_remaining_tasks via execute_tasks with
    # a slow primary task so an alternate worker hedges the admitted tail task.
    first_lock = threading.Lock()
    first_seen = [False]
    hedged_called = [False]
    release_slow = threading.Event()
    payload = b'HEDGED_TAIL_DATA' * 4

    class _SlowHedgeHandle:

      def __init__(self):
        self.proc_entry = None

      def read_shm_range(self, start, length, shm_uint8, shm_off, *_):
        is_primary_slow = False
        if start == 0:
          with first_lock:
            if not first_seen[0]:
              first_seen[0] = True
              is_primary_slow = True
            else:
              hedged_called[0] = True
        if is_primary_slow:
          release_slow.wait(timeout=2.0)
        shm_uint8[shm_off : shm_off + length] = np.frombuffer(
            payload[start : start + length], dtype=np.uint8
        )
        if start == 0 and not is_primary_slow:
          release_slow.set()
        return length

      def close(self) -> None:
        pass

    class _SlowHedgeSource:
      supports_hedge = True

      def ensure_procs(self, total_bytes: int) -> int:
        del total_bytes
        return 0

      def open(self, mode: str = 'rb', stream_id: int = 0):
        del mode, stream_id
        return _SlowHedgeHandle()

    with (
        mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 16),
        safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as h_arena,
    ):
      h_arena.ring_capacity = 4096
      h_arena.record_unit(0, 64, 2, [])
      h_arena.admit_units()
      h_arena.file_task_rem = [2]
      h_tasks = [(0, 0, 32, 0), (0, 32, 32, 32)]
      h_arena.execute_tasks([_SlowHedgeSource()], h_tasks, enable_hedge=True)
      assert h_arena.shm_uint8 is not None
      self.assertTrue(hedged_called[0])
      self.assertEqual(bytes(h_arena.shm_uint8[:64]), payload)

  def test_parallel_pack_axis0_split_and_h2d_error_propagation(self):
    fake_tpu = types.SimpleNamespace(platform='tpu')
    fake_cpu = types.SimpleNamespace(platform='cpu')
    base = np.arange(32 * 64, dtype=np.float32).reshape(32, 64)
    strided = base[:, ::2]  # Non-contiguous view (32, 32)
    contig_view = base[:4]  # C-contiguous non-owning view
    empty_view = base[:0]
    scalar_view = base[0, 0:1].reshape(())

    with mock.patch.object(safetensors_loader, '_PARALLEL_PACK_MIN_BYTES', 512):
      packed_tpu = safetensors_loader._pack_device_shards(
          [strided, contig_view, empty_view, scalar_view],
          cast(Any, [fake_tpu] * 4),
      )
      self.assertTrue(packed_tpu[0].flags['C_CONTIGUOUS'])
      self.assertTrue(packed_tpu[0].flags['OWNDATA'])
      np.testing.assert_array_equal(packed_tpu[0], strided)
      # Non-CPU keeps zero-copy C-contiguous views.
      self.assertIs(packed_tpu[1], contig_view)
      self.assertEqual(packed_tpu[2].shape, (0, 64))

      packed_cpu = safetensors_loader._pack_device_shards(
          [strided, contig_view, empty_view, scalar_view],
          cast(Any, [fake_cpu] * 4),
      )
      for out_arr, orig in zip(
          packed_cpu, (strided, contig_view, empty_view, scalar_view)
      ):
        self.assertTrue(out_arr.flags['C_CONTIGUOUS'])
        self.assertTrue(out_arr.flags['OWNDATA'])
        np.testing.assert_array_equal(out_arr, orig)

    # Verify async H2D stage exception propagates cleanly through finish() and
    # reclaim_credits() without deadlocking even with a small bounded queue.
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as err_arena:
      dummy_plan = safetensors_loader._CircularRingPlan(
          [([], [])], {}, [[]], [], [0], [0], [0, 0], [0], [64],
          [frozenset({0})], [frozenset()], 64, False,
      )
      with (
          mock.patch.object(safetensors_loader, '_H2D_QUEUE_MAXSIZE', 1),
          mock.patch.object(
              safetensors_loader,
              '_commit_packed_shards',
              side_effect=RuntimeError('h2d boom'),
          ),
      ):
        driver = safetensors_loader._RingH2DDriver(
            err_arena, dummy_plan, lambda g, sink: sink([], [], [], [], [])
        )
        try:
          driver.submit_ready_groups([True])
          with self.assertRaisesRegex(RuntimeError, 'h2d boom'):
            driver.finish()
          with self.assertRaisesRegex(RuntimeError, 'h2d boom'):
            driver.reclaim_credits()
        finally:
          driver.close()

    # Verify group_active blocks ring credit reclamation while active hedged
    # workers still reference group 0, and has_safetensors_key_prefix accepts
    # gs:// URIs without calling expanduser().
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as pin_arena:
      pin_plan = safetensors_loader._CircularRingPlan(
          [([], []), ([], [])],
          {},
          [[], []],
          [(0, 0, 0, 64, 0), (1, 1, 0, 64, 0)],
          [1, 1],
          [1, 2],
          [0, 1, 2],
          [0, 4096],
          [4096, 8192],
          [frozenset({0}), frozenset({1})],
          [frozenset(), frozenset()],
          8192,
          True,
      )
      pin_driver = safetensors_loader._RingH2DDriver(
          pin_arena, pin_plan, lambda g, sink: None
      )
      try:
        pin_driver.submit_ready_groups([True, False])
        pin_driver._futures[0].result()
        self.assertEqual(pin_driver.reclaim_credits(group_active=[1, 0]), 1)
        self.assertEqual(pin_driver.reclaim_credits(group_active=[0, 0]), 2)
      finally:
        pin_driver.close()

      # Also exercise execute_gcs_ring with ring_wraps=True.
      fake_pool = safetensors_io.GcsShmReadPool(1, ('10.0.0.1',), spawn=False)
      try:
        fake_pool.ctrl.hdr.finished_workers = 1
        fake_files = [
            safetensors_io.GsHttpPath('gs://bkt/a.st', 'tok', ('10.0.0.1',)),
            safetensors_io.GsHttpPath('gs://bkt/b.st', 'tok', ('10.0.0.1',)),
        ]
        orig_start = fake_pool.start_batch

        def _instant_start(*args, **kwargs):
          orig_start(*args, **kwargs)
          fake_pool.ctrl.group_rem[0] = 0
          fake_pool.ctrl.group_rem[1] = 0
          fake_pool.ctrl.hdr.finished_workers = 1

        with mock.patch.object(
            fake_pool, 'start_batch', side_effect=_instant_start
        ):
          pin_arena.execute_gcs_ring(
              fake_pool, fake_files, pin_plan, lambda g, sink: None
          )
      finally:
        fake_pool.shutdown()

  def test_streaming_match_params_flush_and_pack_dedupe(self):
    # R-2: Identical non-contiguous slices share one packed host buffer.
    fake_tpus = [types.SimpleNamespace(platform='tpu') for _ in range(6)]
    base = np.arange(16 * 32, dtype=np.float32).reshape(16, 32)
    slice_a, slice_b = base[:, ::2], base[:, 1::2]  # 1024 B each
    shards_in = [slice_a, slice_a, slice_a, slice_a, slice_b, slice_b]
    copy_calls = 0
    orig_copyto = np.copyto

    def _counting_copyto(dst, src):
      nonlocal copy_calls
      copy_calls += 1
      orig_copyto(dst, src)

    with mock.patch.object(safetensors_loader.np, 'copyto', _counting_copyto):
      packed = safetensors_loader._pack_device_shards(
          shards_in, cast(Any, fake_tpus)
      )
    self.assertEqual(copy_calls, 2)
    self.assertIs(packed[0], packed[3])
    self.assertIs(packed[4], packed[5])
    self.assertIsNot(packed[0], packed[4])
    np.testing.assert_array_equal(packed[0], slice_a)
    np.testing.assert_array_equal(packed[4], slice_b)

    # R-1: Pre-append flush prevents overshoot (1024+1024<=2500, flush before
    # 3rd 1024B and before oversized 3072B tensor -> batch sizes [2, 1, 1]).
    shapes = [(16, 16), (16, 16), (16, 16), (48, 16)]
    keys = [f'layers.{i}.mlp.w' for i in range(len(shapes))]
    expected = {
        k: (np.arange(int(np.prod(sh)), dtype=np.float32).reshape(sh) + i * 99)
        for i, (k, sh) in enumerate(zip(keys, shapes))
    }
    dev = jax.devices()[0]
    specs = {
        k: safetensors_loader._TargetSpec(
            param_index=i, global_shape=sh, host_shape=sh, sharding=dev,
            host_slices=tuple(slice(0, d) for d in sh),
            device_slices=((dev, tuple(slice(0, d) for d in sh)),),
        )
        for i, (k, sh) in enumerate(zip(keys, shapes))
    }
    params: list[Any] = [None] * len(keys)
    batch_sizes: list[int] = []
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as arena:
      dummy_plan = safetensors_loader._CircularRingPlan(
          [([], [])], {}, [[]], [], [0], [0], [0, 0], [0], [64],
          [frozenset({0})], [frozenset()], 64, True,
      )
      n_keys = len(keys)
      with mock.patch.object(safetensors_loader, '_PACK_FLUSH_BYTES', 2500):
        driver = safetensors_loader._RingH2DDriver(
            arena, dummy_plan, lambda _g, sink: self.assertEqual(
                safetensors_loader._match_params(
                    keys, specs, expected, params,
                    h2d_sink=lambda ms, *a: (batch_sizes.append(len(ms)),
                                             sink(ms, *a)),
                ), n_keys
            ),
        )
        try:
          driver.submit_ready_groups([True])
          driver.finish()
        finally:
          driver.close()
    self.assertEqual(batch_sizes, [2, 1, 1])
    for i, k in enumerate(keys):
      np.testing.assert_array_equal(np.asarray(params[i]), expected[k])

  def test_stream_load_files_blocks_h2d_on_external_arena_and_error(self):
    dummy_plan = safetensors_loader._CircularRingPlan(
        [([], [])], {}, [[]], [], [0], [0], [0, 0], [0], [64],
        [frozenset({0})], [frozenset()], 64, False,
    )
    arr = jnp.zeros((4,), dtype=jnp.float32)
    fake_file = mock.MagicMock(spec=safetensors_io.RangeSource)
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as arena:
      def _fail_ring(files, plan, consume_fn, *, ici_config=None):
        del files, plan, consume_fn, ici_config
        params[0] = arr
        raise RuntimeError('mid-ring io error')

      params: list[Any] = [jax.ShapeDtypeStruct((4,), jnp.float32)]
      with (
          mock.patch.object(
              safetensors_loader.SpmdHostStager,
              'inspect_file_items',
              return_value=([], []),
          ),
          mock.patch.object(
              arena, 'plan_circular_ring', return_value=dummy_plan
          ),
          mock.patch.object(
              arena, 'execute_threaded_ring', side_effect=_fail_ring
          ),
          mock.patch.object(
              jax, 'block_until_ready', wraps=jax.block_until_ready
          ) as mock_block,
      ):
        with self.assertRaisesRegex(RuntimeError, 'mid-ring io error'):
          safetensors_loader.stream_load_files(
              [fake_file], {}, set(), None, None, {}, ['w'], params, arena=arena
          )
      mock_block.assert_called_once()
      self.assertEqual(mock_block.call_args.args[0], [arr])

  def test_read_ici_plan_and_64dev_hlo_parity(self):
    ps = jax.sharding.PartitionSpec
    cases = [
        (
            '1h_fsdp_2d_t', (8, 1), ('fsdp', 'tp'), (32, 16), (1, 0),
            (16, 32), ps('fsdp', None), 1, 0, True,
        ),
        (
            '1h_tp_2d_t', (2, 4), ('fsdp', 'tp'), (32, 16), (1, 0),
            (16, 32), ps('fsdp', 'tp'), 1, 0, True,
        ),
        (
            '1h_tp_qkv_3d', (2, 4), ('fsdp', 'tp'), (32, 16), (1, 0),
            (16, 8, 4), ps('fsdp', 'tp', None), 1, 0, True,
        ),
        (
            '1h_ep_moe_3d', (2, 2, 2), ('fsdp', 'ep', 'tp'), (16, 8, 16),
            (0, 2, 1), (16, 16, 8), ps('ep', 'fsdp', 'tp'), 2, 0, True,
        ),
        (
            '1h_attndp_rep', (2, 2, 2), ('dp', 'fsdp', 'tp'), (32, 16),
            (1, 0), (16, 32), ps('fsdp', 'tp'), 1, 1, False,
        ),
        (
            '1h_custom_abc', (2, 2, 2), ('a', 'b', 'c'), (16, 8, 16),
            (0, 2, 1), (16, 16, 8), ps('b', 'a', 'c'), 2, 0, True,
        ),
        (
            '8h_fsdp_2d_t', (64, 1), ('fsdp', 'tp'), (128, 64), (1, 0),
            (64, 128), ps('fsdp', None), 1, 0, True,
        ),
        (
            '8h_tp_2d_t', (32, 2), ('fsdp', 'tp'), (128, 64), (1, 0),
            (64, 128), ps('fsdp', 'tp'), 1, 0, True,
        ),
        (
            '8h_tp_moe_3d', (32, 2), ('fsdp', 'tp'), (128, 8, 64),
            (0, 2, 1), (128, 64, 8), ps(None, 'fsdp', 'tp'), 2, 0, True,
        ),
        (
            '8h_ep_moe_3d', (8, 4, 2), ('ep', 'fsdp', 'tp'), (128, 8, 64),
            (0, 2, 1), (128, 64, 8), ps('ep', 'fsdp', 'tp'), 2, 0, True,
        ),
        (
            '8h_attndp_attn', (4, 8, 2), ('dp', 'fsdp', 'tp'), (128, 64),
            (1, 0), (64, 128), ps('fsdp', 'tp'), 1, 1, False,
        ),
        (
            '8h_attndp_tmoe', (4, 8, 2), ('dp', 'fsdp', 'tp'), (128, 8, 64),
            (0, 2, 1), (128, 64, 8), ps(('dp', 'fsdp'), None, 'tp'),
            1, 0, True,
        ),
        (
            '8h_attndp_itup', (4, 8, 2), ('dp', 'fsdp', 'tp'), (128, 64),
            (1, 0), (64, 128), ps(None, ('dp', 'fsdp', 'tp')), 0, 0, True,
        ),
        (
            '8h_fused_s1', (32, 2), ('fsdp', 'tp'), (2, 128, 64),
            (0, 2, 1), (2, 64, 128), ps(None, 'fsdp', 'tp'), 1, 0, True,
        ),
        (
            '8h_full_rep', (4, 8, 2), ('dp', 'fsdp', 'tp'), (128, 16),
            None, (128, 16), ps(None, None), 0, 3, False,
        ),
        (
            '8h_1d_norm', (32, 2), ('fsdp', 'tp'), (128,), None, (128,),
            ps('fsdp'), 0, 1, False,
        ),
    ]
    rng = np.random.default_rng(17)
    for (
        name, msh, axn, ssh, perm, tsh, tsp, exp_a2a, exp_ag, exp_alias
    ) in cases:
      with self.subTest(name=name):
        n_dev = math.prod(msh)
        mesh = jax.sharding.Mesh(
            np.array(jax.devices()[:n_dev]).reshape(msh), axn
        )
        tshd = jax.sharding.NamedSharding(mesh, tsp)
        plan = safetensors_loader.plan_read_ici_reshard(ssh, tsh, tshd, perm)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.stage_axis, 1 if name == '8h_fused_s1' else 0)
        src_np = np.asarray(
            jnp.asarray(
                rng.standard_normal(ssh).astype(np.float32), jnp.bfloat16
            )
        )
        idx_map = plan.host_stage_sharding.addressable_devices_indices_map(ssh)
        shards = [
            jax.device_put(np.ascontiguousarray(src_np[idx_map[d]]), d)
            for d in mesh.devices.flat
        ]
        staged = jax.make_array_from_single_device_arrays(
            ssh, plan.host_stage_sharding, shards
        )
        compiled = (
            safetensors_loader.make_read_ici_kernel(
                plan, out_dtype=jnp.bfloat16
            )
            .lower(staged)
            .compile()
        )
        hlo = compiled.as_text()
        self.assertLen(re.findall(r'=\s*.*all-to-all\(', hlo), exp_a2a)
        self.assertLen(re.findall(r'=\s*.*all-gather\(', hlo), exp_ag)
        if exp_alias:
          self.assertIn('input_output_alias', hlo)
        if name == '8h_attndp_attn':
          # Attn-DP must all-gather ONLY along `dp` (group size 4), never 64.
          m = re.search(r'all-gather\([^\n]*replica_groups=([^\n]+)', hlo)
          self.assertIsNotNone(m)
          assert m is not None
          self.assertNotIn('size=64', m.group(1))
          grp_str = m.group(1)
          self.assertTrue(
              "{'dp'}" in grp_str
              or len(re.findall(r'\d+', grp_str.split('}')[0])) == 4
          )
        out = compiled(staged)
        ref = src_np if perm is None else np.transpose(src_np, perm)
        ref = ref.reshape(tsh).astype(np.dtype(jnp.bfloat16))
        np.testing.assert_array_equal(np.asarray(out), ref)
        self.assertTrue(out.sharding.is_equivalent_to(tshd, ndim=len(tsh)))

    # Verify in-kernel bf16 -> f32 cast preserves donor and bit-exact values.
    mesh8 = jax.sharding.Mesh(
        np.array(jax.devices()[:8]).reshape(2, 4), ('fsdp', 'tp')
    )
    tshd8 = jax.sharding.NamedSharding(mesh8, ps('fsdp', 'tp'))
    p1 = safetensors_loader.plan_read_ici_reshard(
        (32, 16), (16, 32), tshd8, (1, 0)
    )
    assert p1 is not None
    src_bf16 = np.asarray(
        jnp.asarray(rng.standard_normal((32, 16)), jnp.bfloat16)
    )
    staged_bf16 = jax.device_put(src_bf16, p1.host_stage_sharding)
    compiled_cast = (
        safetensors_loader.make_read_ici_kernel(p1, out_dtype=jnp.float32)
        .lower(staged_bf16)
        .compile()
    )
    self.assertIn('buffer_donor', compiled_cast.as_text())
    out_f32 = compiled_cast(staged_bf16)
    self.assertEqual(out_f32.dtype, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(out_f32), src_bf16.T.astype(np.float32)
    )

    # Uneven / non-divisible / invalid cases cleanly fall back (return None).
    mesh64 = jax.sharding.Mesh(
        np.array(jax.devices()[:64]).reshape(32, 2), ('fsdp', 'tp')
    )
    self.assertIsNone(
        safetensors_loader.plan_read_ici_reshard(
            (130, 256),
            (256, 130),
            jax.sharding.NamedSharding(mesh64, ps('tp', None)),
            (1, 0),
        )
    )
    self.assertIsNone(
        safetensors_loader.plan_read_ici_reshard(
            (64, 64), (64, 64), jax.devices()[0]
        )
    )
    self.assertIsNone(
        safetensors_loader.plan_read_ici_reshard(
            (64, 64),
            (64, 64),
            jax.sharding.NamedSharding(mesh64, ps('fsdp', 'tp')),
            (0,),
        )
    )
    self.assertIsNone(
        safetensors_loader.plan_read_ici_reshard(
            (64, 64),
            (32, 32),
            jax.sharding.NamedSharding(mesh64, ps('fsdp', 'tp')),
        )
    )
    # `merge_reshape_spec` edge cases: dim=1, trailing 1s, invalid split.
    self.assertEqual(
        safetensors_loader.merge_reshape_spec(
            (1, 8, 16, 1), ps(None, 'fsdp', 'tp', None), mesh8, (1, 8, 16)
        ),
        (None, 'fsdp', 'tp'),
    )
    self.assertIsNone(
        safetensors_loader.merge_reshape_spec(
            (8, 16), ps('fsdp', 'tp'), mesh8, (128,)
        )
    )
    self.assertIsNone(
        safetensors_loader.merge_reshape_spec(
            (8, 16), ps('fsdp', 'tp'), mesh8, (4, 32)
        )
    )

    # Verify narrowing fp32 -> bf16 cast happens BEFORE shard_map collectives.
    src_f32 = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    staged_f32 = jax.device_put(src_f32, p1.host_stage_sharding)
    lowered_narrow = (
        safetensors_loader.make_read_ici_kernel(p1, out_dtype=jnp.bfloat16)
        .lower(staged_f32)
    )
    narrow_mlir = lowered_narrow.as_text()
    self.assertRegex(narrow_mlir, r'"stablehlo\.all_to_all"\([^)]*\).*bf16')
    self.assertLess(
        narrow_mlir.index('stablehlo.convert'),
        narrow_mlir.index('stablehlo.all_to_all'),
    )
    out_bf16 = lowered_narrow.compile()(staged_f32)
    self.assertEqual(out_bf16.dtype, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(out_bf16), np.asarray(jnp.asarray(src_f32.T, jnp.bfloat16))
    )

    # Verify LRU cache hit + miss in `_compile_read_ici_kernel`.
    p2 = safetensors_loader.plan_read_ici_reshard(
        (64, 16), (16, 64), tshd8, (1, 0)
    )
    assert p2 is not None
    safetensors_loader._compile_read_ici_kernel.cache_clear()
    x1 = jax.device_put(
        jnp.zeros((32, 16), jnp.float32), p1.host_stage_sharding
    )
    _ = safetensors_loader._run_read_ici_kernel(p1, x1, jnp.float32)
    x1b = jax.device_put(
        jnp.ones((32, 16), jnp.float32), p1.host_stage_sharding
    )
    out1b = safetensors_loader._run_read_ici_kernel(p1, x1b, jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(out1b), np.ones((16, 32), np.float32)
    )
    x2 = jax.device_put(
        jnp.ones((64, 16), jnp.float32), p2.host_stage_sharding
    )
    _ = safetensors_loader._run_read_ici_kernel(p2, x2, jnp.float32)
    info = safetensors_loader._compile_read_ici_kernel.cache_info()
    self.assertEqual((info.hits, info.misses), (1, 2))

  def test_read_ici_multihost_contiguity_and_sync_order(self):
    ps = jax.sharding.PartitionSpec
    devs64 = list(jax.devices()[:64])
    # Simulate 8 hosts x 8 devices/host via thin proxies for pure planning.
    fake_devs = np.array(
        [types.SimpleNamespace(id=i, process_index=i // 8) for i in range(64)]
    )
    mesh_adp = jax.sharding.Mesh(
        np.array(devs64).reshape(4, 8, 2), ('dp', 'fsdp', 'tp')
    )
    orig_spans = safetensors_loader._host_dim_spans

    def _sim_8host_spans(shd, shape, axis):
      by_proc: dict[int, list[tuple[int, int]]] = {}
      dim = shape[axis]
      dev_to_proc = {d: i // 8 for i, d in enumerate(devs64)}
      for dev, idx in shd.devices_indices_map(shape).items():
        lo, hi = (idx[axis] or slice(None)).indices(dim)[:2]
        by_proc.setdefault(dev_to_proc[dev], []).append((lo, hi))
      return {
          p: (
              min(a for a, _ in sp),
              max(b for _, b in sp),
              sum(b - a for a, b in sp),
          )
          for p, sp in by_proc.items()
      }

    # Attn-DP Attn P('fsdp','tp') + T(1,0): stage_spec=('tp','dp','fsdp') is
    # non-contiguous across 8 hosts -> auto-select host_stage_spec=('dp',...).
    tshd_attn = jax.sharding.NamedSharding(mesh_adp, ps('fsdp', 'tp'))
    with mock.patch.object(
        safetensors_loader, '_host_dim_spans', _sim_8host_spans
    ):
      plan_attn = safetensors_loader.plan_read_ici_reshard(
          (128, 64), (64, 128), tshd_attn, (1, 0)
      )
    self.assertIsNotNone(plan_attn)
    assert plan_attn is not None
    self.assertEqual(plan_attn.stage_spec, ps(('tp', 'dp', 'fsdp'), None))
    self.assertEqual(plan_attn.host_stage_spec, ps(('dp', 'fsdp', 'tp'), None))

    # Execute `host_stage_spec != stage_spec` kernel and verify bit-exactness.
    src_np = np.arange(128 * 64, dtype=np.float32).reshape(128, 64)
    staged = jax.device_put(src_np, plan_attn.host_stage_sharding)
    out = safetensors_loader.make_read_ici_kernel(plan_attn)(staged)
    np.testing.assert_array_equal(np.asarray(out), src_np.T)

    # Verify `_rep_spans_hosts`: cross-host dp -> True; intra-host tp -> False.
    fake_mesh_adp = types.SimpleNamespace(
        axis_names=('dp', 'fsdp', 'tp'), devices=fake_devs.reshape(4, 8, 2)
    )
    self.assertTrue(
        safetensors_loader._rep_spans_hosts(
            cast(Any, fake_mesh_adp), ('dp', 'fsdp', 'tp'), ('dp',)
        )
    )
    fake_mesh_tp = types.SimpleNamespace(
        axis_names=('fsdp', 'tp'), devices=fake_devs.reshape(32, 2)
    )
    self.assertFalse(
        safetensors_loader._rep_spans_hosts(
            cast(Any, fake_mesh_tp), ('fsdp', 'tp'), ('tp',)
        )
    )

    # MoE host-subset rejection when target dim-0 doesn't cover host_stage.
    tshd_moe_bad = jax.sharding.NamedSharding(mesh_adp, ps('fsdp', 'dp', 'tp'))
    with mock.patch.object(
        safetensors_loader, '_host_dim_spans', _sim_8host_spans
    ):
      self.assertIsNone(
          safetensors_loader.plan_read_ici_reshard(
              (128, 16, 64), (128, 64, 16), tshd_moe_bad, (0, 2, 1), is_moe=True
          )
      )
    self.assertIsNotNone(orig_spans(tshd_attn, (64, 128), 0))

    # Verify `_compute_ordered_ici_indices` multi-host process_allgather branch.
    mesh8 = jax.sharding.Mesh(
        np.array(devs64[:8]).reshape(2, 4), ('fsdp', 'tp')
    )
    p0 = safetensors_loader.plan_read_ici_reshard(
        (32, 16),
        (16, 32),
        jax.sharding.NamedSharding(mesh8, ps('fsdp', 'tp')),
        (1, 0),
    )
    assert p0 is not None
    st0 = safetensors_loader._build_spec_geometry(
        0, (32, 16), p0.host_stage_sharding
    )
    st1 = safetensors_loader._build_spec_geometry(
        1, (32, 16), p0.host_stage_sharding
    )
    ht0 = safetensors_loader._HostTensor(
        'k0', None, np.zeros(1, np.uint8), np.dtype(np.float32), (32, 16)
    )
    ht1 = safetensors_loader._HostTensor(
        'k1', None, np.zeros(1, np.uint8), np.dtype(np.float32), (32, 16)
    )
    planned = [([ht1], []), ([ht0], [])]
    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(
            multihost_utils,
            'process_allgather',
            return_value=np.array([[1, 0], [0, 3]], dtype=np.int32),
        ),
    ):
      order = safetensors_loader._compute_ordered_ici_indices(
          {0: p0, 1: p0}, planned, {'k0': st0, 'k1': st1}
      )
    self.assertEqual(order, [0, 1])

  def test_read_ici_e2e_stream_moe_ring_wrap_and_error_cleanup(self):
    ps = jax.sharding.PartitionSpec
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:8]).reshape(2, 4), ('fsdp', 'tp')
    )
    rng = np.random.default_rng(123)
    dense_w = rng.standard_normal((128, 32)).astype(np.float32)
    qkv_w = rng.standard_normal((64, 32)).astype(np.float32)
    fused_w = rng.standard_normal((2, 128, 16)).astype(np.float32)
    uneven_w = rng.standard_normal((10, 8)).astype(np.float32)
    experts = [
        rng.standard_normal((16, 32)).astype(np.float32) for _ in range(16)
    ]

    # Split across 3 shards: s0 fills & frees before s1 starts cross-file MoE
    # pinning (`e.8..15` in s1, `e.0..7` in s2) so a 64 KiB ring genuinely wraps
    # AND exercises non-monotonic expert arrival.
    s0 = {'dense': dense_w, 'fused': fused_w}
    s1 = {'qkv': qkv_w, **{f'e.{i}': experts[i] for i in range(8, 16)}}
    s2 = {'uneven': uneven_w, **{f'e.{i}': experts[i] for i in range(0, 8)}}

    manifest: hf_mapping.RuleSpec = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'model_type': 'ici_test',
        'rules': [
            {
                'source_keys': ['dense'],
                'target_keys': ['dense'],
                'ops': [{'transpose': {'axes': [1, 0]}}],
            },
            {
                'source_keys': ['qkv'],
                'target_keys': ['qkv'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {
                        'reshape': {
                            'src_shape': [32, 64],
                            'dst_shape': [32, 8, 8],
                        }
                    },
                ],
            },
            {
                'source_keys': ['fused'],
                'target_keys': ['fused'],
                'ops': [{'transpose': {'axes': [0, 2, 1]}}],
            },
            {'source_keys': ['uneven'], 'target_keys': ['uneven'], 'ops': []},
            {
                'source_keys': [f'e.{i}' for i in range(16)],
                'target_keys': ['moe'],
                'ops': [
                    {'transpose': {'axes': [1, 0]}},
                    {'stack': {'axis': 0}},
                ],
            },
        ],
    }
    source_map, comp_sources, comp_exec = (
        safetensors_loader.SpmdHostStager.compile_manifest(manifest)
    )
    keys = ['dense', 'qkv', 'fused', 'uneven', 'moe']
    shapes = [(32, 128), (32, 8, 8), (2, 16, 128), (10, 8), (16, 32, 16)]
    shardings = [
        jax.sharding.NamedSharding(mesh, ps('fsdp', 'tp')),
        jax.sharding.NamedSharding(mesh, ps('fsdp', 'tp', None)),
        jax.sharding.NamedSharding(mesh, ps(None, 'fsdp', 'tp')),
        jax.sharding.NamedSharding(mesh, ps(None, None)),
        jax.sharding.NamedSharding(mesh, ps('fsdp', None, 'tp')),
    ]
    expected = {
        'dense': dense_w.T,
        'qkv': qkv_w.T.reshape(32, 8, 8),
        'fused': np.transpose(fused_w, (0, 2, 1)),
        'uneven': uneven_w,
        'moe': np.stack([e.T for e in experts], axis=0),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
      paths = []
      for i, shard in enumerate((s0, s1, s2)):
        p = os.path.join(tmpdir, f'shard-{i}.safetensors')
        stnp.save_file(shard, p)
        paths.append(p)
      sources = [safetensors_io.RangeSource(epath.Path(p)) for p in paths]

      # Run both non-wrapping and wrapping rings with zero-copy views.
      for budget, expect_wrap in ((256 * 1024, False), (65536, True)):
        with self.subTest(budget=budget, expect_wrap=expect_wrap):
          params: list[Any] = [
              jax.ShapeDtypeStruct(sh, jnp.float32) for sh in shapes
          ]
          target_specs = safetensors_loader._build_target_specs(
              keys, params, shardings
          )
          with (
              mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024),
              mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
              mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 4096),
              safetensors_loader.ShmRangeReader(
                  shm_budget_bytes=budget
              ) as arena,
          ):
            ici_plans: dict[int, safetensors_loader._ReadIciPlan] = {}
            stage_specs: dict[str, safetensors_loader._TargetSpec] = {}
            items = [
                safetensors_loader.SpmdHostStager.inspect_file_items(
                    src, i, source_map, [], target_specs, ici_plans, stage_specs
                )
                for i, src in enumerate(sources)
            ]
            ring_plan = arena.plan_circular_ring(items)
            self.assertIsNotNone(ring_plan)
            assert ring_plan is not None
            self.assertEqual(ring_plan.ring_wraps, expect_wrap)
            self.assertEqual(set(ici_plans), {0, 1, 2, 4})
            safetensors_loader.stream_load_files(
                sources,
                source_map,
                comp_sources,
                comp_exec,
                np.dtype(np.float32),
                target_specs,
                keys,
                params,
                arena=arena,
            )
          for k, arr, shd in zip(keys, params, shardings):
            np.testing.assert_array_equal(np.asarray(arr), expected[k])
            self.assertTrue(
                arr.sharding.is_equivalent_to(shd, ndim=expected[k].ndim)
            )

      # Verify 30B small-expert MoE ring gate: 3 MiB >= 256 KiB passes;
      # 128 KiB < 256 KiB falls back to heap.
      for row_b, expect_ring in ((3 * 1024 * 1024, True), (128 * 1024, False)):
        dummy_plan = safetensors_loader._SourcePlan(
            'moe_30b', ((1, 0), None), (0, 128)
        )
        dummy_item = safetensors_loader._PlannedTensor(
            0, 0, 0, 1, row_b, row_b, (768, 2048),
            np.dtype(ml_dtypes.bfloat16), dummy_plan, None, None,
            'moe_30b', 0, 128,
        )
        moe_info, _ = safetensors_loader._analyze_moe_ring_layout(
            [[dummy_item]], capacity=4 * 1024**3, min_direct=4 * 1024**2
        )
        self.assertEqual(moe_info['moe_30b'].ring_eligible, expect_ring)

      # Error-path cleanup: ICI kernel failure & H2D failure must propagate
      # cleanly and never hang `_h2d_thread` or `_ici_thread`.
      params_err: list[Any] = [
          jax.ShapeDtypeStruct(sh, jnp.float32) for sh in shapes
      ]
      target_specs_err = safetensors_loader._build_target_specs(
          keys, params_err, shardings
      )
      with (
          mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024),
          mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
          mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 4096),
          mock.patch.object(
              safetensors_loader,
              '_run_read_ici_kernel',
              side_effect=RuntimeError('injected ici failure'),
          ),
          safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena,
      ):
        with self.assertRaisesRegex(RuntimeError, 'injected ici failure'):
          safetensors_loader.stream_load_files(
              sources,
              source_map,
              comp_sources,
              comp_exec,
              np.dtype(np.float32),
              target_specs_err,
              keys,
              params_err,
              arena=arena,
          )

      params_h2d_err: list[Any] = [
          jax.ShapeDtypeStruct(sh, jnp.float32) for sh in shapes
      ]
      target_specs_h2d = safetensors_loader._build_target_specs(
          keys, params_h2d_err, shardings
      )
      call_n = 0
      orig_commit = safetensors_loader._commit_packed_shards

      def _fail_second_commit(*args, **kwargs):
        nonlocal call_n
        call_n += 1
        if call_n >= 2:
          raise RuntimeError('injected h2d failure')
        return orig_commit(*args, **kwargs)

      with (
          mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024),
          mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
          mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 4096),
          mock.patch.object(
              safetensors_loader,
              '_commit_packed_shards',
              side_effect=_fail_second_commit,
          ),
          safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena,
      ):
        with self.assertRaisesRegex(RuntimeError, 'injected h2d failure'):
          safetensors_loader.stream_load_files(
              sources,
              source_map,
              comp_sources,
              comp_exec,
              np.dtype(np.float32),
              target_specs_h2d,
              keys,
              params_h2d_err,
              arena=arena,
          )

      # P0 deadlock guard: exception inside `_flush_pending_group` during
      # `_finish_h2d_loop` MUST still wake `_ici_thread` and propagate.
      p_dense = safetensors_loader.plan_read_ici_reshard(
          (128, 32), (32, 128), shardings[0], (1, 0)
      )
      assert p_dense is not None
      with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as d_arena:
        d_plan = safetensors_loader._CircularRingPlan(
            [([], [])], {}, [[]], [], [0], [0], [0, 0], [0], [64],
            [frozenset({0})], [frozenset()], 64, True,
        )
        d_params: list[Any] = [jax.ShapeDtypeStruct((32, 128), jnp.float32)]
        d_cfg = safetensors_loader._ReadIciConfig(
            {0: p_dense}, (0,), d_params, np.dtype(np.float32)
        )
        d_driver = safetensors_loader._RingH2DDriver(
            d_arena, d_plan, lambda _g, _s: None, ici_config=d_cfg
        )
        try:
          with mock.patch.object(
              d_driver,
              '_flush_pending_group',
              side_effect=RuntimeError('flush boom'),
          ):
            with self.assertRaisesRegex(RuntimeError, 'flush boom'):
              d_driver.finish()
          assert d_driver._ici_thread is not None
          self.assertFalse(d_driver._ici_thread.is_alive())
        finally:
          d_driver.close()

      # P1 `/dev/shm` UAF guard: when `ring_wraps=False` and ICI fails after
      # popping `staged_arr`, `_ici_loop` fences `staged_arr`.
      params_uaf: list[Any] = [
          jax.ShapeDtypeStruct(sh, jnp.float32) for sh in shapes
      ]
      target_specs_uaf = safetensors_loader._build_target_specs(
          keys, params_uaf, shardings
      )
      fenced_staged: list[Any] = []
      seen_staged: list[jax.Array] = []
      orig_block = jax.block_until_ready

      def _spy_block(x):
        fenced_staged.extend(jax.tree_util.tree_leaves(x))
        return orig_block(x)

      def _boom_ici(plan, staged_x, out_dtype):
        del plan, out_dtype
        seen_staged.append(staged_x)
        raise RuntimeError('uaf test ici boom')

      with (
          mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024),
          mock.patch.object(safetensors_loader, '_DIRECT_SLICE_BYTES', 512),
          mock.patch.object(safetensors_loader, '_READ_CHUNK_BYTES', 4096),
          mock.patch.object(
              safetensors_loader, '_run_read_ici_kernel', side_effect=_boom_ici
          ),
          mock.patch.object(jax, 'block_until_ready', side_effect=_spy_block),
          safetensors_loader.ShmRangeReader(
              shm_budget_bytes=256 * 1024
          ) as arena,
      ):
        with self.assertRaisesRegex(RuntimeError, 'uaf test ici boom'):
          safetensors_loader.stream_load_files(
              sources,
              source_map,
              comp_sources,
              comp_exec,
              np.dtype(np.float32),
              target_specs_uaf,
              keys,
              params_uaf,
              arena=arena,
          )
      self.assertLen(seen_staged, 1)
      self.assertTrue(
          any(x is seen_staged[0] for x in fenced_staged),
          'Expected popped staged_arr to be fenced on ring_wraps=False error.',
      )

      # P0/P1 multi-host asymmetric fallback sync: if another host has
      # `plan=None`, local host must take `_stream_subfile_fallback` too.
      params_asym: list[Any] = [
          jax.ShapeDtypeStruct(sh, jnp.float32) for sh in shapes
      ]
      target_specs_asym = safetensors_loader._build_target_specs(
          keys, params_asym, shardings
      )
      with (
          mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024),
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              return_value=np.array([True, False], dtype=np.bool_),
          ),
          mock.patch.object(
              safetensors_loader,
              '_stream_subfile_fallback',
              wraps=safetensors_loader._stream_subfile_fallback,
          ) as spy_fallback,
          safetensors_loader.ShmRangeReader(
              shm_budget_bytes=256 * 1024
          ) as arena,
      ):
        safetensors_loader.stream_load_files(
            sources,
            source_map,
            comp_sources,
            comp_exec,
            np.dtype(np.float32),
            target_specs_asym,
            keys,
            params_asym,
            arena=arena,
        )
      spy_fallback.assert_called_once()
      for k, arr in zip(keys, params_asym):
        np.testing.assert_array_equal(np.asarray(arr), expected[k])

      # P1 silent shape-mismatch guard: `_maybe_plan_ici_spec` rejects
      # same-element-count shape mismatches and bad perm lengths.
      spec_dense = target_specs_asym['dense']
      bad_plan_shape = safetensors_loader._SourcePlan('dense', None)
      bad_plan_perm = safetensors_loader._SourcePlan('dense', ((0,), None))
      with mock.patch.object(safetensors_loader, '_ICI_MIN_BYTES', 1024):
        self.assertIsNone(
            safetensors_loader._maybe_plan_ici_spec(
                bad_plan_shape,
                (128, 32),  # Same prod as (32, 128), but no transpose/reshape!
                np.dtype(np.float32),
                spec_dense,
                {},
                {},
            )
        )
        self.assertIsNone(
            safetensors_loader._maybe_plan_ici_spec(
                bad_plan_perm,
                (128, 32),
                np.dtype(np.float32),
                spec_dense,
                {},
                {},
            )
        )

      # 1. merge_reshape_spec returns None on PartitionSpec.UNCONSTRAINED.
      mesh1 = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
      self.assertIsNone(
          safetensors_loader.merge_reshape_spec(
              (4, 4),
              jax.sharding.PartitionSpec(
                  jax.sharding.PartitionSpec.UNCONSTRAINED, None
              ),
              mesh1,
              (16,),
          )
      )

      # 2. stream_load_files([]) participates in _all_hosts_agree before return.
      with mock.patch.object(
          safetensors_loader, '_all_hosts_agree', return_value=True
      ) as spy_agree:
        safetensors_loader.stream_load_files(
            [], {}, frozenset(), None, None, {}, [], []
        )
      spy_agree.assert_called_once_with(True)

      # 3. _commit_packed_shards fences DMA when array assembly raises.
      bad_spec = safetensors_loader._TargetSpec(
          param_index=0,
          global_shape=(4,),
          host_shape=(4,),
          sharding=jax.sharding.NamedSharding(
              mesh1, jax.sharding.PartitionSpec()
          ),
          host_slices=(slice(0, 4),),
          device_slices=((jax.devices()[0], (slice(0, 4),)),),
      )
      with (
          mock.patch.object(
              jax,
              'make_array_from_single_device_arrays',
              side_effect=RuntimeError('assembly boom'),
          ),
          mock.patch.object(
              jax, 'block_until_ready', wraps=jax.block_until_ready
          ) as spy_block,
      ):
        with self.assertRaisesRegex(RuntimeError, 'assembly boom'):
          safetensors_loader._commit_packed_shards(
              [bad_spec],
              [np.zeros(4, dtype=np.float32)],
              [jax.devices()[0]],
              [(0,)],
              [jax.ShapeDtypeStruct((4,), np.float32)],
              block_h2d=False,
          )
        spy_block.assert_called_once()

      # 4. _stream_subfile_fallback passes block_h2d=True and fences in finally.
      with (
          mock.patch.object(
              safetensors_loader.SpmdHostStager,
              'consume_one_file',
              wraps=safetensors_loader.SpmdHostStager.consume_one_file,
          ) as spy_consume,
          mock.patch.object(
              jax, 'block_until_ready', wraps=jax.block_until_ready
          ) as spy_sub_block,
          safetensors_loader.ShmRangeReader(shm_budget_bytes=256 * 1024) as a2,
      ):
        params_sub: list[jax.ShapeDtypeStruct | jax.Array] = [
            jax.ShapeDtypeStruct(expected[k].shape, expected[k].dtype)
            for k in keys
        ]
        safetensors_loader._stream_subfile_fallback(
            a2,
            sources,
            source_map,
            comp_sources,
            comp_exec,
            np.dtype(np.float32),
            target_specs_asym,
            keys,
            params_sub,
        )
      self.assertTrue(
          all(
              call.kwargs.get('block_h2d')
              for call in spy_consume.call_args_list
          )
      )
      spy_sub_block.assert_called()

      # 5. _RingH2DDriver._ici_loop fails fast if a planned param is missing.
      with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as a3:
        empty_plan = safetensors_loader._CircularRingPlan(
            planned_all=[],
            moe_bufs={},
            group_unpacks=[],
            tasks=[],
            group_counts=[],
            group_end_idx=[],
            suffix_min_start=[],
            file_virt_start=[],
            file_virt_end=[],
            pinned_commits=[],
            group_deps=[],
            total_virt_bytes=0,
            ring_wraps=False,
        )
        dummy_ici = safetensors_loader._ReadIciConfig(
            ici_plans={0: mock.MagicMock()},
            ordered_ici_indices=(0,),
            params=[jax.ShapeDtypeStruct((4,), np.float32)],
            target_dtype=None,
        )
        drv = safetensors_loader._RingH2DDriver(
            a3, empty_plan, lambda *_: None, ici_config=dummy_ici
        )
        try:
          with self.assertRaisesRegex(
              RuntimeError, 'Missing ICI staged parameter'
          ):
            drv.finish()
        finally:
          drv.close()

      # 6. execute_gcs_ring / execute_threaded_ring close driver on setup error.
      with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as a4:
        one_task_plan = safetensors_loader._CircularRingPlan(
            planned_all=[],
            moe_bufs={},
            group_unpacks=[],
            tasks=[(0, 0, 0, 8, 0)],
            group_counts=[1],
            group_end_idx=[0],
            suffix_min_start=[0, 1],
            file_virt_start=[0],
            file_virt_end=[8],
            pinned_commits=[frozenset()],
            group_deps=[frozenset()],
            total_virt_bytes=8,
            ring_wraps=False,
        )
        bad_pool = mock.MagicMock()
        bad_pool.start_batch.side_effect = RuntimeError('start_batch boom')
        with mock.patch.object(
            safetensors_loader._RingH2DDriver,
            'close',
            autospec=True,
            wraps=safetensors_loader._RingH2DDriver.close,
        ) as spy_close:
          with self.assertRaisesRegex(RuntimeError, 'start_batch boom'):
            a4.execute_gcs_ring(
                bad_pool,
                [mock.MagicMock(http_path='/b/f', token='t')],
                one_task_plan,
                lambda *_: None,
            )
          spy_close.assert_called_once()
        bad_src = mock.MagicMock()
        bad_src.ensure_procs.side_effect = RuntimeError('ensure_procs boom')
        with mock.patch.object(
            safetensors_loader._RingH2DDriver,
            'close',
            autospec=True,
            wraps=safetensors_loader._RingH2DDriver.close,
        ) as spy_close2:
          with self.assertRaisesRegex(RuntimeError, 'ensure_procs boom'):
            a4.execute_threaded_ring(
                [bad_src], one_task_plan, lambda *_: None
            )
          spy_close2.assert_called_once()

  def test_loader_raises_on_missing_params_and_incomplete_pending_rules(self):
    missing_dir = os.path.join(self.st_dir, 'missing_param')
    os.makedirs(missing_dir, exist_ok=True)
    partial_tensors = {
        k: np.asarray(v.value if isinstance(v, nnx.Param) else v)
        for k, v in self.tensors.items()
        if k != 'lm_head.bias'
    }
    stnp.save_file(
        partial_tensors, os.path.join(missing_dir, 'model.safetensors')
    )
    with self.assertRaisesRegex(
        ValueError,
        r"parameters for declared HF layout 'toy'; missing.*lm_head\.bias",
    ):
      safetensors_loader.load_and_create_model(
          missing_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          dtype=jnp.float32,
          mode='original',
      )

    pending_dir = os.path.join(self.st_dir, 'incomplete_pending')
    os.makedirs(pending_dir, exist_ok=True)
    fused_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            *self.manifest['rules'][:-1],
            {
                'source_keys': ['lm_head.bias_part0', 'lm_head.bias_part1'],
                'target_keys': ['lm_head.bias'],
                'ops': [{'concat': {'axis': 0, 'sizes': [128, 128]}}],
            },
        ],
    }
    pending_tensors = {
        **partial_tensors,
        'lm_head.bias_part0': np.zeros((128,), dtype=np.float32),
    }
    stnp.save_file(
        pending_tensors, os.path.join(pending_dir, 'model.safetensors')
    )
    with self.assertRaisesRegex(
        ValueError,
        r"rules for declared HF layout ''; unconsumed.*lm_head\.bias_part0",
    ):
      safetensors_loader.load_and_create_model(
          pending_dir,
          test_common.ToyTransformer,
          self.model.config,
          fused_manifest,
          dtype=jnp.float32,
          mode='original',
      )

    with (
        mock.patch.object(env_utils, 'is_internal_env', return_value=False),
        mock.patch.object(
            env_utils, 'is_pathways_initialized', return_value=False
        ),
    ):
      with self.assertRaisesRegex(
          ValueError,
          r"parameters for declared HF layout 'toy'; missing.*lm_head\.bias",
      ):
        safetensors_loader.load_and_create_model(
            missing_dir,
            test_common.ToyTransformer,
            self.model.config,
            self.manifest,
            dtype=jnp.float32,
            mode='optimized',
        )
      with self.assertRaisesRegex(
          ValueError,
          r"rules for declared HF layout ''; unconsumed.*lm_head\.bias_part0",
      ):
        safetensors_loader.load_and_create_model(
            pending_dir,
            test_common.ToyTransformer,
            self.model.config,
            fused_manifest,
            dtype=jnp.float32,
            mode='optimized',
        )

  def test_gcs_uri_resolved_once_for_headers_read_and_model_load(self):
    stnp.save_file(
        self.tensors, os.path.join(self.st_dir, 'model.safetensors')
    )
    mock_load = self.enter_context(
        mock.patch.object(
            safetensors_loader,
            'load_file_from_gcs',
            autospec=True,
            spec_set=True,
            return_value=self.st_dir,
        )
    )
    standalone_headers = safetensors_loader.read_safetensors_headers(
        'gs://bucket/standalone'
    )
    self.assertEqual(set(standalone_headers), set(self.tensors))
    mock_load.assert_called_once_with('gs://bucket/standalone')

    mock_load.reset_mock()
    with (
        mock.patch.object(env_utils, 'is_internal_env', return_value=False),
        mock.patch.object(
            env_utils, 'is_pathways_initialized', return_value=False
        ),
    ):
      loaded = safetensors_loader.load_and_create_model(
          'gs://bucket/model',
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          dtype=jnp.float32,
          mode='optimized',
      )
    mock_load.assert_called_once_with('gs://bucket/model')
    jax.tree.map(np.testing.assert_array_equal, self.state, nnx.state(loaded))

  def test_read_checkpoint_keys_prefers_index_then_falls_back_to_headers(self):
    stnp.save_file(
        self.tensors, os.path.join(self.st_dir, 'model.safetensors')
    )
    index_path = epath.Path(self.st_dir) / 'model.safetensors.index.json'
    mock_gcs = self.enter_context(
        mock.patch.object(
            safetensors_loader,
            'load_file_from_gcs',
            autospec=True,
            spec_set=True,
            return_value=self.st_dir,
        )
    )
    # 1. No index: keys come from the shard headers (gs:// resolved once).
    self.assertEqual(
        safetensors_loader.read_checkpoint_keys('gs://bucket/ckpt'),
        frozenset(self.tensors),
    )
    mock_gcs.assert_called_once_with('gs://bucket/ckpt')
    # 2. Empty or malformed `weight_map` falls back to the shard headers.
    for payload in ({'weight_map': {}}, {'weight_map': ['a']}, ['a']):
      index_path.write_text(json.dumps(payload))
      self.assertEqual(
          safetensors_loader.read_checkpoint_keys(self.st_dir),
          frozenset(self.tensors),
      )
    # 3. A populated index is authoritative: shard headers are never opened.
    index_path.write_text(
        json.dumps({'weight_map': {'only.key': 'model.safetensors'}})
    )
    with mock.patch.object(
        safetensors_loader, 'read_safetensors_headers', autospec=True
    ) as mock_headers:
      self.assertEqual(
          safetensors_loader.read_checkpoint_keys(self.st_dir),
          frozenset({'only.key'}),
      )
    mock_headers.assert_not_called()

  def test_cross_shard_duplicates_optimized_dtypes_and_corrupt_headers(self):
    self.assertEqual(safetensors_loader.to_np_dtype('I8'), np.dtype(np.int8))
    self.assertEqual(
        safetensors_loader.to_np_dtype('F8_E4M3'),
        np.dtype(ml_dtypes.float8_e4m3fn),
    )
    self.assertEqual(
        safetensors_loader.to_np_dtype(np.float32), np.dtype(np.float32)
    )
    with self.assertRaisesRegex(ValueError, 'Unsupported safetensors dtype'):
      safetensors_loader.to_np_dtype('UNSUPPORTED')

    # Cross-shard duplicate source keys must fail fast in both modes.
    dup_dir = os.path.join(self.st_dir, 'dup_shards')
    os.makedirs(dup_dir, exist_ok=True)
    raw = {
        k: np.asarray(v.value if isinstance(v, nnx.Param) else v)
        for k, v in self.tensors.items()
    }
    stnp.save_file(raw, os.path.join(dup_dir, 'shard-00001.safetensors'))
    stnp.save_file(
        {'emb.embedding': raw['emb.embedding']},
        os.path.join(dup_dir, 'shard-00002.safetensors'),
    )
    with self.assertRaisesRegex(ValueError, 'Duplicate tensor keys'):
      safetensors_loader.load_and_create_model(
          dup_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          mode='original',
      )

    # Optimized mode with U8 / BF16 / F16 / skipped keys / shape mismatch.
    opt_dir = os.path.join(self.st_dir, 'opt_edge')
    os.makedirs(opt_dir, exist_ok=True)
    edge_header = {
        'a_codes': {'dtype': 'U8', 'shape': [3], 'data_offsets': [0, 3]},
        'b_unaligned': {'dtype': 'F32', 'shape': [4], 'data_offsets': [3, 19]},
        'c_empty': {'dtype': 'F32', 'shape': [0, 4], 'data_offsets': [19, 19]},
        'd_scalar': {'dtype': 'BF16', 'shape': [], 'data_offsets': [19, 21]},
        'e_proj': {'dtype': 'F16', 'shape': [2, 3], 'data_offsets': [21, 33]},
        'z_skip': {'dtype': 'I8', 'shape': [2], 'data_offsets': [33, 35]},
    }
    edge_payload = (
        np.array([1, 2, 3], dtype=np.uint8).tobytes()
        + np.ones((4,), dtype=np.float32).tobytes()
        + np.array(2.0, dtype=ml_dtypes.bfloat16).tobytes()
        + np.arange(6, dtype=np.float16).tobytes()
        + np.array([-1, 1], dtype=np.int8).tobytes()
    )
    hb = json.dumps(edge_header).encode('utf-8')
    with open(os.path.join(opt_dir, 'edge.safetensors'), 'wb') as fh:
      fh.write(struct.pack('<Q', len(hb)) + hb + edge_payload)
    edge_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {'source_keys': ['a_codes'], 'target_keys': ['codes'], 'ops': []},
            {
                'source_keys': ['b_unaligned'],
                'target_keys': ['unaligned_f32'],
                'ops': [],
            },
            {'source_keys': ['c_empty'], 'target_keys': ['empty'], 'ops': []},
            {'source_keys': ['d_scalar'], 'target_keys': ['scalar'], 'ops': []},
            {
                'source_keys': ['e_proj'],
                'target_keys': ['f16_proj'],
                'ops': [{'transpose': {'axes': [1, 0]}}],
            },
        ],
    }
    with (
        mock.patch.object(env_utils, 'is_internal_env', return_value=False),
        mock.patch.object(
            env_utils, 'is_pathways_initialized', return_value=False
        ),
    ):
      opt_loaded = safetensors_loader.load_and_create_model(
          opt_dir,
          _EdgeCaseModel,
          config=None,
          manifest=edge_manifest,
          dtype=None,
          mode='optimized',
      )
      np.testing.assert_array_equal(opt_loaded.codes.value, [1, 2, 3])
      bf16_opt_dir = os.path.join(self.st_dir, 'opt_bf16_cast')
      os.makedirs(bf16_opt_dir, exist_ok=True)
      stnp.save_file(
          {k: v.astype(ml_dtypes.bfloat16) for k, v in raw.items()},
          os.path.join(bf16_opt_dir, 'model.safetensors'),
      )
      opt_f32 = safetensors_loader.load_and_create_model(
          bf16_opt_dir,
          test_common.ToyTransformer,
          self.model.config,
          self.manifest,
          dtype=jnp.float32,
          mode='optimized',
      )
      self.assertEqual(
          nnx.state(opt_f32)['emb']['embedding'].value.dtype, jnp.float32
      )
      with (
          mock.patch.object(
              safetensors_loader.mmap,
              'mmap',
              side_effect=OSError('mmap failed'),
          ),
          self.assertRaises(OSError),
      ):
        safetensors_loader.load_safetensors_with_offsets(
            os.path.join(bf16_opt_dir, 'model.safetensors')
        )
      with self.assertRaisesRegex(ValueError, 'Duplicate tensor keys'):
        safetensors_loader.load_and_create_model(
            dup_dir,
            test_common.ToyTransformer,
            self.model.config,
            self.manifest,
            mode='optimized',
        )
      bad_opt_dir = os.path.join(self.st_dir, 'bad_opt_shape')
      os.makedirs(bad_opt_dir, exist_ok=True)
      stnp.save_file(
          {
              **raw,
              'emb.embedding': np.zeros((4, 4), dtype=np.float32),
          },
          os.path.join(bad_opt_dir, 'model.safetensors'),
      )
      with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
        safetensors_loader.load_and_create_model(
            bad_opt_dir,
            test_common.ToyTransformer,
            self.model.config,
            self.manifest,
            mode='optimized',
        )

    # Corrupt headers: oversized size, non-dict JSON, bad metadata, OOB offsets.
    for idx, (raw_bytes, exc_cls) in enumerate((
        (struct.pack('<Q', 0), ValueError),
        (struct.pack('<Q', 200_000_000), ValueError),
        (struct.pack('<Q', 2) + b'[]', ValueError),
        (struct.pack('<Q', 20) + b'{"x": 1}', OSError),
    )):
      del idx
      bad_path = os.path.join(self.st_dir, 'corrupt.safetensors')
      with open(bad_path, 'wb') as fh:
        fh.write(raw_bytes)
      with self.assertRaises(exc_cls):
        safetensors_loader.load_safetensors_with_offsets(bad_path)

    bad_meta_dir = os.path.join(self.st_dir, 'bad_meta')
    os.makedirs(bad_meta_dir, exist_ok=True)
    for meta, expected_exc in (
        ({'dtype': 'F32', 'shape': [-1], 'data_offsets': [0, 4]}, ValueError),
        ({'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 4]}, ValueError),
        ({'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}, OSError),
    ):
      b = json.dumps({'emb.embedding': meta}).encode('utf-8')
      p = os.path.join(bad_meta_dir, 'bad.safetensors')
      with open(p, 'wb') as fh:
        fh.write(struct.pack('<Q', len(b)) + b)
      with self.assertRaises(expected_exc):
        safetensors_loader.load_safetensors_with_offsets(p)

  def test_audit_fixes_moe_validation_teardown_meta_and_empty_addr_map(self):
    # 1. [P1-1] Missing local MoE experts raise ValueError in both ring and
    # sub-file paths (even when expert_slice.stop - 1 arrived).
    moe_manifest: hf_mapping.RuleSpec = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [{
            'source_keys': ['e.0', 'e.1'],
            'target_keys': ['moe'],
            'ops': [{'stack': {'axis': 0}}],
        }],
    }
    source_map, comp_sources, comp_exec = (
        safetensors_loader.SpmdHostStager.compile_manifest(moe_manifest)
    )
    missing_moe_path = os.path.join(self.st_dir, 'missing_moe.safetensors')
    stnp.save_file({'e.1': np.ones((4, 4), dtype=np.float32)}, missing_moe_path)
    missing_src = safetensors_io.RangeSource(epath.Path(missing_moe_path))
    dev0 = jax.local_devices()[0]
    moe_spec = safetensors_loader._TargetSpec(
        0,
        (2, 4, 4),
        (2, 4, 4),
        dev0,
        (slice(0, 2), slice(0, 4), slice(0, 4)),
        ((dev0, (slice(0, 2), slice(0, 4), slice(0, 4))),),
    )
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena:
      with self.assertRaisesRegex(ValueError, 'Missing local MoE experts'):
        safetensors_loader.stream_load_files(
            [missing_src],
            source_map,
            comp_sources,
            comp_exec,
            np.dtype(np.float32),
            {'moe': moe_spec},
            ['moe'],
            [jax.ShapeDtypeStruct((2, 4, 4), jnp.float32)],
            arena=arena,
        )
      with self.assertRaisesRegex(ValueError, 'Missing local MoE experts'):
        safetensors_loader._stream_subfile_fallback(
            arena,
            [missing_src],
            source_map,
            comp_sources,
            comp_exec,
            np.dtype(np.float32),
            {'moe': moe_spec},
            ['moe'],
            [jax.ShapeDtypeStruct((2, 4, 4), jnp.float32)],
        )
      with self.assertRaisesRegex(ValueError, 'Missing local MoE experts'):
        safetensors_loader._plan_file_reads(
            missing_src, moe_manifest, [], {'moe': moe_spec}, {}, arena
        )

    # 2. [P1-2] MoE shape/count mismatch with non-zero exp_slice.start fails
    # fast with ValueError instead of negative-index overwrite or IndexError.
    host1_spec = safetensors_loader._TargetSpec(
        0,
        (4, 4, 4),
        (2, 4, 4),
        dev0,
        (slice(2, 4), slice(0, 4), slice(0, 4)),
        ((dev0, (slice(0, 2), slice(0, 4), slice(0, 4))),),
    )
    bad_shape_path = os.path.join(self.st_dir, 'bad_moe_shape.safetensors')
    stnp.save_file(
        {
            'e.0': np.ones((4, 2), dtype=np.float32),
            'e.1': np.ones((4, 2), dtype=np.float32),
        },
        bad_shape_path,
    )
    bad_shape_src = safetensors_io.RangeSource(epath.Path(bad_shape_path))
    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      safetensors_loader.SpmdHostStager.inspect_file_items(
          bad_shape_src, 0, source_map, [], {'moe': host1_spec}
      )
    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      safetensors_loader._plan_file_reads(
          bad_shape_src, moe_manifest, [], {'moe': host1_spec}
      )
    # Zero-dim MoE shape mismatch also fails fast instead of bypassing checks.
    zero_dim_path = os.path.join(self.st_dir, 'zero_dim_moe.safetensors')
    stnp.save_file(
        {
            'e.0': np.zeros((4, 0), dtype=np.float32),
            'e.1': np.zeros((4, 0), dtype=np.float32),
        },
        zero_dim_path,
    )
    zero_dim_src = safetensors_io.RangeSource(epath.Path(zero_dim_path))
    with self.assertRaisesRegex(ValueError, 'Shape mismatch'):
      safetensors_loader.SpmdHostStager.inspect_file_items(
          zero_dim_src, 0, source_map, [], {'moe': moe_spec}
      )
    # Inconsistent expert dtype across the same MoE key fails fast in both
    # circular ring and sub-file fallback (including same-byte-width dtypes).
    mixed_dt_path = os.path.join(self.st_dir, 'mixed_moe_dt.safetensors')
    stnp.save_file(
        {
            'e.0': np.ones((4, 4), dtype=np.float32),
            'e.1': np.ones((4, 4), dtype=np.int32),
        },
        mixed_dt_path,
    )
    mixed_src = safetensors_io.RangeSource(epath.Path(mixed_dt_path))
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=65536) as arena:
      mixed_items = safetensors_loader.SpmdHostStager.inspect_file_items(
          mixed_src, 0, source_map, [], {'moe': moe_spec}
      )
      with self.assertRaisesRegex(ValueError, 'Inconsistent MoE expert'):
        arena.plan_circular_ring([mixed_items])
      with self.assertRaisesRegex(ValueError, 'Inconsistent MoE expert'):
        safetensors_loader._plan_file_reads(
            mixed_src, moe_manifest, [], {'moe': moe_spec}, {}, arena
        )
      with self.assertRaisesRegex(ValueError, 'Inconsistent MoE expert'):
        safetensors_loader._stream_subfile_fallback(
            arena,
            [mixed_src],
            source_map,
            comp_sources,
            comp_exec,
            np.dtype(np.float32),
            {'moe': moe_spec},
            ['moe'],
            [jax.ShapeDtypeStruct((2, 4, 4), jnp.float32)],
        )

    # 3. [P1-4] execute_gcs_ring drains pool BEFORE driver.close() even when
    # either raises, and _RingH2DDriver.close() skips deleted _ici_ready arrays.
    one_task_plan = safetensors_loader._CircularRingPlan(
        planned_all=[([], [])],
        moe_bufs={},
        group_unpacks=[[]],
        tasks=[(0, 0, 0, 8, 0)],
        group_counts=[1],
        group_end_idx=[1],
        suffix_min_start=[0, 1],
        file_virt_start=[0],
        file_virt_end=[64],
        pinned_commits=[frozenset({0})],
        group_deps=[frozenset()],
        total_virt_bytes=64,
        ring_wraps=False,
    )
    events: list[str] = []
    fake_pool = mock.MagicMock()
    fake_pool.check_health.side_effect = RuntimeError('health boom')
    fake_pool.wait_batch_drained.side_effect = (
        lambda **_: events.append('drain')
    )
    orig_close = safetensors_loader._RingH2DDriver.close

    def _boom_close(drv_self):
      events.append('close')
      orig_close(drv_self)
      raise RuntimeError('close boom')

    with (
        safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as arena,
        mock.patch.object(
            safetensors_loader._RingH2DDriver, 'close', _boom_close
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, 'close boom'):
        arena.execute_gcs_ring(
            fake_pool,
            [mock.MagicMock(http_path='/b/f', token='t')],
            one_task_plan,
            lambda *_: None,
        )
    self.assertEqual(events, ['drain', 'close'])

    events.clear()
    fake_pool2 = mock.MagicMock()
    fake_pool2.check_health.side_effect = RuntimeError('health boom 2')

    def _boom_drain(**_):
      events.append('drain')
      raise OSError('drain boom')

    fake_pool2.wait_batch_drained.side_effect = _boom_drain

    def _rec_close(drv_self):
      events.append('close')
      orig_close(drv_self)

    with (
        safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as arena,
        mock.patch.object(
            safetensors_loader._RingH2DDriver, 'close', _rec_close
        ),
    ):
      with self.assertRaisesRegex(OSError, 'drain boom'):
        arena.execute_gcs_ring(
            fake_pool2,
            [mock.MagicMock(http_path='/b/f', token='t')],
            one_task_plan,
            lambda *_: None,
        )
    self.assertEqual(events, ['drain', 'close'])

    deleted_arr = jax.device_put(np.zeros((2,), dtype=np.float32), dev0)
    live_arr = jax.device_put(np.ones((2,), dtype=np.float32), dev0)
    deleted_arr.delete()
    with safetensors_loader.ShmRangeReader(shm_budget_bytes=4096) as arena:
      drv = safetensors_loader._RingH2DDriver(
          arena, one_task_plan, lambda *_: None
      )
      drv._ici_ready = {0: deleted_arr, 1: live_arr}
      with mock.patch.object(
          jax, 'block_until_ready', wraps=jax.block_until_ready
      ) as spy_block:
        drv.close()
      spy_block.assert_called_once_with([live_arr])
      self.assertEmpty(drv._ici_ready)

    # 4. [P2-2] _validate_tensor_meta rejects bools and avoids int64 overflow.
    for bad_meta in (
        {'dtype': 'F32', 'shape': [True], 'data_offsets': [0, 4]},
        {'dtype': 'F32', 'shape': [1], 'data_offsets': [False, 4]},
        {'dtype': 'F32', 'shape': [2**32, 2**32], 'data_offsets': [0, 0]},
    ):
      with self.assertRaises(ValueError):
        safetensors_loader._validate_tensor_meta('f.st', 'w', bad_meta)

    # 5. Empty addressable_devices_indices_map returns empty device_slices.
    empty_shd = mock.create_autospec(jax.sharding.NamedSharding, instance=True)
    empty_shd.addressable_devices_indices_map.return_value = {}
    empty_spec = safetensors_loader._build_spec_geometry(0, (4, 8), empty_shd)
    self.assertEqual(empty_spec.device_slices, ())
    self.assertEqual(empty_spec.host_shape, (0, 0))
    self.assertEqual(empty_spec.host_slices, (slice(0, 0), slice(0, 0)))


if __name__ == '__main__':
  absltest.main()
