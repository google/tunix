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

"""Common test utilities."""

from collections.abc import Iterable
import dataclasses
import gc
import os
import shutil
import sys
from typing import Any, List, Tuple

from flax import nnx
import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import qwix
import tenacity
from tunix.models.gemma3 import merge_embeddings as merge_embeddings_lib
from tunix.models.gemma3 import utils as gemma_utils
from tunix.rl import reshard
from tunix.utils import env_utils

import sentencepiece as spm

env_utils.setup_sharding_environment()


def _convert_leaf_to_nparray(leaf):
  if isinstance(leaf, jax.Array):
    jax.block_until_ready(leaf)
    return np.asarray(leaf)
  return leaf


def _convert_to_nparray(tree):
  return jax.tree.map(_convert_leaf_to_nparray, tree)


def assert_equal(path, x, y):
  np.testing.assert_array_equal(
      _convert_to_nparray(x),
      _convert_to_nparray(y),
      err_msg=f'Mismatch at path: {path}',
  )


def assert_not_equal(path, x, y):
  np.testing.assert_(
      np.any(np.not_equal(_convert_to_nparray(x), _convert_to_nparray(y))),
      msg=f'Unexpected match at path: {path}',
  )


def assert_close(path, x, y, atol=1e-5, rtol=1e-5):
  # NOTE: `atol`/`rtol` must be passed by keyword. `np.testing.assert_allclose`
  # has signature (actual, desired, rtol, atol), so passing them positionally
  # silently swaps the two.
  np.testing.assert_allclose(
      _convert_to_nparray(x),
      _convert_to_nparray(y),
      rtol=rtol,
      atol=atol,
      err_msg=f'Mismatch at path: {path}',
  )


# Number of ULPs two independently-compiled computations are allowed to differ
# by. A single rounding of the same real value can land on adjacent
# representable floats, so 1 ULP is the floor; 2 leaves room for one extra
# rounding in the chain.
_DEFAULT_MAX_ULP = 2


def ulp_dist(x, y):
  """Elementwise ULP distance between two float arrays of the same dtype.

  Floats are reinterpreted as sign-magnitude integers and remapped so that the
  integer ordering matches the float ordering. The result is therefore exact,
  scale-free, and well-defined next to zero (where relative error is
  meaningless because it diverges) and across the sign boundary (`+0.0` and
  `-0.0` are 0 ULP apart).

  Args:
    x: Array-like of floats. bfloat16, float16, float32 and float64 supported.
    y: Array-like of the same dtype and broadcastable shape.

  Returns:
    An integer array of ULP distances.
  """
  x = np.asarray(_convert_leaf_to_nparray(x))
  y = np.asarray(_convert_leaf_to_nparray(y))
  if x.dtype != y.dtype:
    raise ValueError(
        f'ulp_dist requires matching dtypes, got {x.dtype} and {y.dtype}. '
        'Compare in the storage dtype rather than upcasting, otherwise the '
        'ULP scale is not the one the computation actually used.'
    )
  int_dtype = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(
      x.dtype.itemsize
  )
  if int_dtype is None:
    raise ValueError(f'Unsupported float width for ulp_dist: {x.dtype}')
  wide = np.int64

  def _to_ordered_int(a):
    i = a.view(int_dtype).astype(wide)
    # Sign-magnitude -> monotonic: negatives are mirrored below zero.
    return np.where(i < 0, wide(-(2 ** (8 * a.dtype.itemsize - 1))) - i, i)

  return np.abs(_to_ordered_int(x) - _to_ordered_int(y))


def assert_close_ulp(path, x, y, max_ulp=_DEFAULT_MAX_ULP):
  """Asserts two float arrays agree to within `max_ulp` ULPs.

  Preferred over `assert_close` for comparing outputs of two independently
  compiled programs (e.g. two `jax.jit` boundaries) in low precision. A fixed
  `atol` cannot work there: gradient tensors span many magnitudes, and entries
  produced by cancellation are near zero with an absolute error set by the
  magnitude of the cancelling terms, so any `atol` tight enough to be
  meaningful for the large entries rejects the small ones.

  Args:
    path: Pytree key path, used in the failure message.
    x: Array-like of floats.
    y: Array-like of the same dtype.
    max_ulp: Maximum tolerated ULP distance, inclusive.
  """
  d = ulp_dist(x, y)
  violations = int(np.count_nonzero(d > max_ulp))
  if not violations:
    return
  hist = np.bincount(np.minimum(d, 8).ravel(), minlength=9)
  np.testing.assert_(
      False,
      msg=(
          f'Mismatch at path: {path}\n'
          f'  dtype        : {np.asarray(_convert_leaf_to_nparray(x)).dtype}\n'
          f'  max ULP      : {int(d.max())} (allowed {max_ulp})\n'
          f'  violating    : {violations}/{d.size}'
          f' ({100.0 * violations / d.size:.4f}%)\n'
          f'  ULP histogram: '
          + ', '.join(
              f'{k if k < 8 else ">=8"}:{int(v)}'
              for k, v in enumerate(hist)
              if v
          )
      ),
  )


def assert_bitwise_equal(path, x, y):
  """Asserts two float arrays are bit-for-bit identical.

  This is the right gate for float32 equivalence between two implementations
  that are supposed to perform the same arithmetic: unlike a norm comparison it
  cannot be masked by a differing reduction order downstream.
  """
  assert_close_ulp(path, x, y, max_ulp=0)


# Dtype kinds whose bytes are not a fixed-width numeric payload and so cannot be
# XOR-folded: object, unicode, bytes, datetime, timedelta.
_UNCHECKSUMMABLE_KINDS = frozenset("OUSMm")

_BIT_VIEW = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def tree_bit_checksum(tree):
  """Order-independent XOR checksum over the raw bits of a pytree's leaves.

  Computed on the host, so it adds no operations to any traced graph. Embedding
  a checksum inside a `jax.jit` region makes every leaf an extra graph output,
  which changes XLA's fusion decisions and can itself perturb the numerics
  being measured.

  Raises on a leaf it cannot fold rather than skipping it. A checksum that
  silently ignores part of the tree reports "identical" for trees that are not,
  which is the worst possible failure mode for this function -- and not a
  hypothetical one: an earlier version guarded with
  `np.issubdtype(dtype, np.number)`, which is False for `ml_dtypes.bfloat16`
  (it is a numpy extension dtype with `kind == 'V'`), so every bfloat16 tree
  checksummed to 0 and every bfloat16 comparison passed vacuously.

  Raises:
    TypeError: If a leaf's dtype has no fixed-width bit representation.
  """
  acc = 0
  folded = 0
  for leaf in jax.tree_util.tree_leaves(tree):
    a = np.asarray(_convert_leaf_to_nparray(leaf))
    view = None if a.dtype.kind in _UNCHECKSUMMABLE_KINDS else _BIT_VIEW.get(
        a.dtype.itemsize
    )
    if view is None:
      raise TypeError(
          f"tree_bit_checksum cannot fold a leaf of dtype {a.dtype!r} "
          f"(kind={a.dtype.kind!r}, itemsize={a.dtype.itemsize}). Skipping it "
          "would make unequal trees compare equal; add an explicit conversion "
          "instead."
      )
    acc ^= int(np.bitwise_xor.reduce(a.view(view).ravel()))
    folded += 1
  if jax.tree_util.tree_leaves(tree) and not folded:
    raise TypeError("tree_bit_checksum folded no leaves from a non-empty tree.")
  return acc


def live_array_census():
  """Groups every live device array by (shape, dtype). Host-only.

  Returns:
    A dict mapping (shape, dtype_name) to (count, total_global_bytes), plus the
    key `None` mapping to (total_count, total_global_bytes) for the whole set.

  Note the byte figures are *global* -- `jax.Array.nbytes` reports the logical
  size of the whole array, not the per-device shard. Divide by the number of
  devices an array is sharded over to compare against a profiler's per-device
  numbers. Counts are unaffected and are usually the more useful signal.
  """
  census = {}
  total_n = 0
  total_b = 0
  for a in jax.live_arrays():
    key = (tuple(a.shape), a.dtype.name)
    n, b = census.get(key, (0, 0))
    census[key] = (n + 1, b + a.nbytes)
    total_n += 1
    total_b += a.nbytes
  census[None] = (total_n, total_b)
  return census


def live_report(tag, top=12, census=None):
  """Prints a census of live device arrays, biggest aggregate first.

  This is the counterpart to reading a memory profile: the profile tells you how
  much the *compiled program* declares, this tells you what the *runtime* is
  still holding. When two programs have identical HLO (same parameters, same
  outputs, same input_output_alias) but different measured peaks, the difference
  is by elimination on this side -- a Python reference keeping a buffer alive.

  Deliberately host-only: it touches `jax.live_arrays()` and nothing else, so
  unlike a hook inside a traced step it cannot change the fusion, scheduling or
  buffer assignment of the thing being measured.

  Args:
    tag: Label for the output line.
    top: How many (shape, dtype) groups to list.
    census: A census from `live_array_census()`; computed fresh if omitted.

  Returns:
    The census, so it can be diffed against a later one with `live_diff`.
  """
  census = live_array_census() if census is None else census
  total_n, total_b = census[None]
  print(f"[live] {tag}: {total_b / 2**30:.2f} GiB global, {total_n} arrays")
  groups = sorted(
      ((k, v) for k, v in census.items() if k is not None),
      key=lambda kv: -kv[1][1],
  )
  for (shape, dt), (n, b) in groups[:top]:
    print(f"        {n:5d} x {str(shape):26s} {dt:9s} = {b / 2**30:7.2f} GiB")
  if len(groups) > top:
    rest = sum(b for _, (_, b) in groups[top:])
    print(f"        ... {len(groups) - top} more groups = {rest / 2**30:.2f} GiB")
  return census


def live_diff(census_a, census_b, label_a="A", label_b="B", top=12):
  """Reports which (shape, dtype) groups differ in COUNT between two censuses.

  Counts are what matters when hunting a duplicated buffer: one side holding a
  parameter tensor twice and the other once shows up here as `+1`, naming the
  exact tensor, without any dependence on sharding or on when each census was
  taken.
  """
  keys = {k for k in census_a if k is not None} | {
      k for k in census_b if k is not None
  }
  rows = []
  for k in keys:
    na = census_a.get(k, (0, 0))[0]
    nb = census_b.get(k, (0, 0))[0]
    if na != nb:
      shape, dt = k
      nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize if shape else 0
      rows.append((abs(na - nb) * nbytes, na, nb, shape, dt, nbytes))
  if not rows:
    print(f"[live-diff] {label_a} vs {label_b}: identical group counts")
    return
  rows.sort(reverse=True)
  total = sum(r[0] for r in rows)
  print(
      f"[live-diff] {label_a} vs {label_b}: {len(rows)} groups differ,"
      f" {total / 2**30:.2f} GiB global"
  )
  for _, na, nb, shape, dt, nbytes in rows[:top]:
    print(
        f"        {na:5d} -> {nb:<5d} ({nb - na:+d})  {str(shape):26s} {dt:9s}"
        f"  {(nb - na) * nbytes / 2**30:+7.2f} GiB"
    )


class Decoder(nnx.Module):
  """Toy decoder for testing."""

  def __init__(self, rngs: nnx.Rngs):
    self.attn = nnx.MultiHeadAttention(
        num_heads=4,
        in_features=16,
        qkv_features=16,
        use_bias=False,
        decode=False,
        rngs=rngs,
    )
    kernel_init_fn = nnx.initializers.lecun_normal()
    self.w1 = nnx.Linear(
        in_features=16,
        out_features=32,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('fsdp', 'tp')),
        bias_init=nnx.with_partitioning(nnx.initializers.zeros_init(), ('tp',)),
    )
    self.w2 = nnx.Linear(
        in_features=32,
        out_features=16,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(kernel_init_fn, ('tp', 'fsdp')),
        bias_init=nnx.with_partitioning(
            nnx.initializers.zeros_init(), ('fsdp',)
        ),
    )

  def __call__(self, x):
    x = self.attn(x) + x
    h = nnx.relu(self.w1(x))
    h = self.w2(h) + x
    return h


@dataclasses.dataclass(kw_only=True, frozen=True)
class VisionConfig:
  """Vision config for testing."""

  num_mm_tokens_per_image: int = 4
  soft_token_placeholder: int = 22
  start_of_image_token: int = 23
  end_of_image_token: int = 24
  double_new_line_token: int = 25


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
  """Model config for testing."""

  num_layers: int = 4
  num_kv_heads: int = 4
  head_dim: int = 16
  vocab_size: int = 256
  vision_config: VisionConfig | None = None
  remat_config: int | None = None


class ToyTransformer(nnx.Module):
  """Toy transformer for testing."""

  def __init__(
      self,
      config: ModelConfig,
      *,
      rngs: nnx.Rngs,
  ):
    self.config = config
    self.emb = nnx.Embed(config.vocab_size, 16, rngs=rngs)
    self.layers = nnx.List(
        [Decoder(rngs=rngs) for _ in range(config.num_layers)]
    )
    self.lm_head = nnx.Linear(
        in_features=16, out_features=config.vocab_size, rngs=rngs
    )

    self.head_dim = 16

  def __call__(
      self,
      x,
      positions,
      cache=None,
      attention_mask=None,
      output_hidden_states=False,
      images=None,
      segment_ids: jax.Array | None = None,
      skip_lm_head: bool = False,
  ):
    tokens = x
    x = self.emb(tokens)
    if images is not None:
      num_images = images.shape[1]
      vision_embs = jax.random.normal(
          key=jax.random.key(2),
          shape=(
              x.shape[0],
              num_images,
              self.config.vision_config.num_mm_tokens_per_image,  # pytype: disable=attribute-error
              x.shape[-1],
          ),
          dtype=x.dtype,
      )
      # Merge the soft tokens back with the text embeddings.
      x = merge_embeddings_lib.merge_embeddings(
          text_embeddings=x,
          vision_embeddings=vision_embs,
          mask=tokens == self.config.vision_config.soft_token_placeholder,  # pytype: disable=attribute-error
      )

    for layer in self.layers:
      x = layer(x)
    if output_hidden_states:
      self.sow(
          nnx.Intermediate,
          'all_hidden_states',
          x,
      )
    if skip_lm_head:
      return x, cache
    logits = self.compute_final_logits(x)
    return logits, cache

  def compute_final_logits(
      self,
      x,
  ):
    """Computes the final logits from the model output."""
    return self.lm_head(x)

  @property
  def num_embed(self) -> int:
    return self.emb.num_embeddings

  def get_attention_mask(self, tokens, inputs_mask=None):
    token_placeholder_id = (
        None
        if self.config.vision_config is None
        else self.config.vision_config.soft_token_placeholder
    )
    return gemma_utils.get_attention_mask(
        tokens,
        inputs_mask=inputs_mask,
        token_placeholder_id=token_placeholder_id,
    )

  def get_model_input(self):
    return get_dummy_inputs_for_lora_toy_transformer_tests()


def get_dummy_inputs_for_lora_toy_transformer_tests():
  return {
      'x': jnp.ones((1, 1), dtype=jnp.int32),
      'positions': jnp.ones((1, 1), dtype=jnp.int32),
      'cache': None,
      'attention_mask': jnp.ones((1, 1, 1), dtype=jnp.bool),
  }


def get_lora_model(
    model: nnx.Module,
    module_path: str = '.*w1|.*w2',
    mesh: jax.sharding.Mesh | None = None,
    rank: int = 4,
    alpha: float = 2.0,
) -> nnx.Module:
  """Apply LoRA to ToyTransformer."""
  lora_provider = qwix.LoraProvider(
      module_path=module_path,
      rank=rank,
      alpha=alpha,
  )
  dummy_model_input = get_dummy_inputs_for_lora_toy_transformer_tests()
  lora_model = qwix.apply_lora_to_model(
      model, lora_provider, **dummy_model_input
  )
  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)  # pyrefly: ignore[bad-argument-type]
  return lora_model  # pyrefly: ignore[bad-return]


class MockVocab(spm.SentencePieceProcessor):
  """Mock vocabulary for testing."""

  DEFAULT_MAPPING = {
      '<pad>': 0,
      '<s>': 1,
      '</s>': 2,
      'input': 3,
      'string': 4,
      'hello': 5,
      'world': 6,
      'Hello': 7,
      'there': 8,
      '!': 9,
      'My': 10,
      'name': 11,
      'is': 12,
      'Morgane': 13,
      'Tunix': 14,
      'Parallax': 15,
      'PT': 16,
      'library': 17,
      'distributed': 18,
      'training': 19,
      'optimizer': 20,
      'quantization': 21,
  }

  def __init__(
      self,
      mapping_text_to_id: dict[str, int] | None = None,
      is_multimodal=False,
  ):
    super().__init__()
    self._start_id = 3
    if is_multimodal:
      self.DEFAULT_MAPPING.update({
          '<img>': 22,
          '<soi>': 23,
          '<eoi>': 24,
          '<doubleline>': 25,
      })
    self._mapping_text_to_id = mapping_text_to_id or self.DEFAULT_MAPPING
    self._vocab_size = len(self._mapping_text_to_id)

  def pad_id(self) -> int:
    return 0

  def bos_id(self) -> int:
    return 1

  def eos_id(self) -> int:
    return 2

  def GetPieceSize(self) -> int:  # pylint: disable=invalid-name
    return self._vocab_size

  def DecodeIds(self, ids: Iterable[int]) -> str:  # pylint: disable=invalid-name
    reverse_mapping = {v: k for k, v in self._mapping_text_to_id.items()}
    return ' '.join(reverse_mapping[e] for e in ids)

  def EncodeAsIds(self, text: str, **kwargs) -> list[int]:  # pylint: disable=invalid-name
    words = text.split(' ')
    res = [
        self._mapping_text_to_id[word]
        for word in words
        if word in self._mapping_text_to_id
    ]
    return res


class ToyTransformerWithScoreHead(nnx.Module):
  """Toy transformer with a score head."""

  def __init__(self, transformer: nnx.Module, rngs: nnx.Rngs):
    """Initializes the transformer with a score head.

    Args:
      transformer: The transformer backbone.
      rngs: The random number generator.
    """

    self.transformer = transformer
    self.score = nnx.Linear(
        in_features=transformer.head_dim,  # pyrefly: ignore[missing-attribute]
        out_features=1,
        use_bias=False,
        rngs=rngs,
    )

  def __call__(self, *args, **kwargs):
    self.transformer(*args, **kwargs, output_hidden_states=True)
    hidden_states = nnx.pop(self.transformer, nnx.Intermediate)[
        'all_hidden_states'
    ].value[-1]
    score = self.score(hidden_states)
    return score


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def safe_list_files(repo_id):
  return huggingface_hub.list_repo_files(repo_id)


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    reraise=True,
)
def safe_download(repo_id, filename, local_dir):
  return huggingface_hub.hf_hub_download(
      repo_id=repo_id, filename=filename, local_dir=local_dir
  )


def download_from_huggingface(repo_id: str, model_path: str):
  """Download checkpoint files from huggingface."""
  print('Make sure you logged in to the huggingface cli.')

  all_files = safe_list_files(repo_id)
  filtered_files = [f for f in all_files if not f.startswith('original/')]

  for filename in filtered_files:

    safe_download(repo_id=repo_id, filename=filename, local_dir=model_path)
  print(f'Downloaded {filtered_files} to: {model_path}')


def batch_templatize(prompts: List[str], tokenizer: Any):
  """Use tokenizer to batch templatize the prompts."""
  assert hasattr(tokenizer, 'apply_chat_template')
  out = []
  for p in prompts:
    out.append(
        tokenizer.apply_chat_template(
            [
                {'role': 'user', 'content': p},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    )
  return out


def validate_llm_outputs(
    expected_output_pattern: List[Tuple[str, List[str]]],
    serving_outputs: List[str],
):
  for (prompt, expectations), generated in zip(
      expected_output_pattern, serving_outputs
  ):
    normalized = generated.strip().lower()
    for keyword in expectations:
      assert keyword.lower() in normalized, (
          f"Response '{generated}' for prompt '{prompt}' does not contain "
          f"expected keyword '{keyword}'."
      )


def delete_directory(path: str):
  """Safely delete directory from filesystem."""
  if os.path.exists(path):
    if os.path.isdir(path):
      shutil.rmtree(path)
      print(f'Deleted directory: {path}')
    else:
      print(f'Path exists but is not a directory: {path}')
  else:
    print(f'Directory does not exist: {path}')


def clear_jax_arrays():
  """Clear all the Jax arrays from hbm."""
  for name, obj in list(globals().items()):
    if isinstance(obj, jnp.ndarray):
      del globals()[name]
  gc.collect()


def is_running_in_colab() -> bool:
  """Checks if the code is running within a Colab IPython kernel."""
  try:
    # get_ipython() is defined in IPython. Check for 'kernel' attribute
    # which is characteristic of a Colab/Jupyter kernel.
    return hasattr(sys.modules['IPython'].get_ipython(), 'kernel')
  except (NameError, KeyError, AttributeError):
    return False
