"""Tests for safetensors_loader."""

from __future__ import annotations

import io
import itertools
import json
import os
import struct
import tempfile  # pylint: disable=unused-import
import threading
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors import numpy as stnp
from tunix.models import safetensors_loader
from tunix.tests import test_common
from tunix.utils import env_utils


def key_mapping(config):
  del config
  return {
      r'^emb\.embedding$': ('emb.embedding', None),
      r'^layers\.(\d+)\.attn\.query\.kernel$': (
          r'layers.\1.attn.query.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.key\.kernel$': (
          r'layers.\1.attn.key.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.value\.kernel$': (
          r'layers.\1.attn.value.kernel',
          None,
      ),
      r'^layers\.(\d+)\.attn\.out\.kernel$': (
          r'layers.\1.attn.out.kernel',
          None,
      ),
      r'^layers\.(\d+)\.w1\.kernel$': (r'layers.\1.w1.kernel', None),
      r'^layers\.(\d+)\.w1\.bias$': (r'layers.\1.w1.bias', None),
      r'^layers\.(\d+)\.w2\.kernel$': (r'layers.\1.w2.kernel', None),
      r'^layers\.(\d+)\.w2\.bias$': (r'layers.\1.w2.bias', None),
      r'^lm_head\.kernel$': ('lm_head.kernel', None),
      r'^lm_head\.bias$': ('lm_head.bias', None),
  }


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
      # Test that nnx.Param are correctly handled.
      cls.tensors[f'layers.{i}.w2.bias'] = nnx.Param(
          np.array(layer_state['w2']['bias'].value),
      )

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.st_dir = self._temp_dir.name
    os.makedirs(self.st_dir, exist_ok=True)

  @parameterized.named_parameters(
      *(
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
      )
  )
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
        key_mapping,
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
    if env_utils.is_internal_env():
      self.skipTest('GCS is not supported in GOOGLE_INTERNAL_PACKAGE_PATH')

    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(self.tensors, filename)

    mock_load = self.enter_context(
        mock.patch.object(
            safetensors_loader,
            'load_file_from_gcs',
            autospec=True,
            spec_set=True,
        )
    )
    mock_load.return_value = self.st_dir
    loaded_model = safetensors_loader.load_and_create_model(
        'gs://bucket/model',
        test_common.ToyTransformer,
        self.model.config,
        key_mapping,
        dtype=jnp.float32,
    )
    mock_load.assert_called_once_with('gs://bucket/model')

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

    def duplicate_key_mapping(config):
      del config
      return {
          r'^lm_head\.kernel$': ('lm_head.kernel', None),
          r'^lm_head\.weight$': ('lm_head.kernel', None),
      }

    with self.assertRaisesRegex(RuntimeError, 'Duplicate key'):
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          duplicate_key_mapping,
          dtype=jnp.float32,
          mode='original',
      )

  def test_load_bfloat16_custom_dtype_avoids_ml_dtypes(self):
    bf16_data = np.random.rand(256, 16).astype(ml_dtypes.bfloat16)
    tensors = {'emb.embedding': bf16_data}
    filename = os.path.join(self.st_dir, 'model.safetensors')
    stnp.save_file(tensors, filename)

    def simple_mapping(config):
      del config
      return {r'^emb\.embedding$': ('emb.embedding', None)}

    result = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        simple_mapping,
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
    # Purely unmapped vision shard should trigger 0 `jax.device_put` calls.
    stnp.save_file(
        {
            'vision.patch_embed.weight': np.ones(
                (8, 8), dtype=ml_dtypes.bfloat16
            )
        },
        os.path.join(self.st_dir, 'model-00004-of-00004.safetensors'),
    )

    host_checks: list[tuple[bool, bool, bool, bool]] = []

    def check_host_and_emit_jax_array(tensors):
      out = dict(tensors)
      for key, arr in out.items():
        host_checks.append((
            isinstance(arr, np.ndarray),
            arr.dtype == np.float32,
            arr.flags.c_contiguous and arr.flags.aligned,
            arr.flags.writeable,
        ))
        if key == 'emb.embedding':
          # Simulate MoE/Gemma `preprocess_fn` emitting a `jax.Array`.
          out[key] = jnp.asarray(arr)
      return out

    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ('fsdp', 'tp')
    )
    mock_device_put = self.enter_context(
        mock.patch.object(jax, 'device_put', wraps=jax.device_put)
    )

    loaded_model = safetensors_loader.load_and_create_model(
        self.st_dir,
        test_common.ToyTransformer,
        self.model.config,
        key_mapping,
        mesh=mesh,
        preprocess_fn=check_host_and_emit_jax_array,
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
    raw_tensors['m_unmapped.vision.tensor'] = np.ones(
        (256,), dtype=np.float32
    )
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
        key_mapping,
        dtype=jnp.float32,
        mode='original',
    )

    self.assertTrue(all('/readahead/' not in p for p in opened_paths))
    self.assertTrue(all(h.closed_flag for h in tracked_handles))
    all_ranges = list(
        itertools.chain.from_iterable(h.read_ranges for h in tracked_handles)
    )
    # Large tensors (e.g. 16 KiB `emb.embedding`) split into 128-byte stripes.
    self.assertGreater(len(all_ranges), len(self.tensors) + 50)
    # Zero bytes of the unmapped tensor's data section are ever read.
    for lo, hi in all_ranges:
      self.assertTrue(hi <= unmapped_abs_lo or lo >= unmapped_abs_hi)

    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(loaded_model)
    )

  @parameterized.named_parameters(
      dict(testcase_name='gfile_minus_one_error', fault_mode='minus_one'),
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
          key_mapping,
          dtype=jnp.float32,
          mode='original',
      )

    self.assertNotEmpty(tracked_handles)
    self.assertTrue(all(h.closed_flag for h in tracked_handles))

  def test_original_loader_supports_scalar_empty_unaligned_and_transformed_tensors(
      self,
  ):
    # Construct raw safetensors manually in deterministic order so `b_unaligned`
    # starts at odd byte offset 3 right after 3-byte `a_codes`, coalescing both
    # into a single span buffer.
    scalar_val = np.array(3.5, dtype=ml_dtypes.bfloat16)
    empty_val = np.zeros((0, 4), dtype=np.float32)
    codes_val = np.array([7, 8, 9], dtype=np.uint8)
    unaligned_val = np.array([1.25, -2.5, 3.75, 4.0], dtype=np.float32)
    # Stored as (2, 6), transposed + reshaped to (3, 2) via `transform`.
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

    def edge_mapping(config):
      del config
      return {
          r'^a_codes$': ('codes', None),
          r'^b_unaligned$': ('unaligned_f32', None),
          r'^c_empty$': ('empty', None),
          r'^d_scalar$': ('scalar', None),
          r'^e_proj$': ('f16_proj', ((1, 0), (3, 2))),
      }

    loaded = safetensors_loader.load_and_create_model(
        self.st_dir,
        _EdgeCaseModel,
        config=None,
        key_mapping=edge_mapping,
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
          key_mapping,
          mode='original',
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
          key_mapping,
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
      safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          key_mapping,
          mode='original',
      )

  def test_original_loader_supports_multihost_sharding_and_local_device_default(
      self,
  ):
    stnp.save_file(
        self.tensors, os.path.join(self.st_dir, 'model.safetensors')
    )
    local_dev = jax.local_devices()[0]
    remote_dev = mock.create_autospec(jax.Device, instance=True)

    with (
        mock.patch.object(
            jax, 'devices', return_value=[remote_dev, local_dev]
        ),
        mock.patch.object(jax, 'local_devices', return_value=[local_dev]),
    ):
      local_loaded = safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          key_mapping,
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

      def to_non_addressable(_):
        s = mock.create_autospec(jax.sharding.NamedSharding, instance=True)
        s.is_fully_addressable = False
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
            'make_array_from_callback',
            side_effect=lambda shape, s, cb: jnp.asarray(
                cb(tuple(slice(0, d) for d in shape))
            ),
        ) as mock_from_callback,
    ):
      mh_loaded = safetensors_loader.load_and_create_model(
          self.st_dir,
          test_common.ToyTransformer,
          self.model.config,
          key_mapping,
          mesh=mesh,
          dtype=jnp.float32,
          mode='original',
      )

    mock_device_put.assert_not_called()
    self.assertLen(mock_from_callback.call_args_list, len(self.tensors))
    jax.tree.map(
        np.testing.assert_array_equal, self.state, nnx.state(mh_loaded)
    )


if __name__ == '__main__':
  absltest.main()
