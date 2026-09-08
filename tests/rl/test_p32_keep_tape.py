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

"""CANON_P32_KEEP_TAPE: the grouped forward keeps its tape for the reverse.

Reuses the sixteen-device toy harness of test_p71_fwd_scan (loaded by path;
tests/ is not a package).  Run inside the pinned CPU image with
XLA_FLAGS=--xla_force_host_platform_device_count=16.
"""

import importlib.util
import os
from pathlib import Path
from unittest import mock

import jax.numpy as jnp
import pytest

from tunix.rl import canonical_qwen3_adapter

_HARNESS_PATH = Path(__file__).with_name("test_p71_fwd_scan.py")
_spec = importlib.util.spec_from_file_location("p71_harness", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)

FME = canonical_qwen3_adapter.FunctionalMappingError


def _env(rank_parallel):
  env = dict(harness._SEGMENTED_ENV)  # pylint: disable=protected-access
  if rank_parallel:
    env["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
  return env


def test_selector_parses_fail_closed():
  parse = canonical_qwen3_adapter._p32_keep_tape  # pylint: disable=protected-access
  for value in ("", "0"):
    with mock.patch.dict(
        os.environ, {"CANON_P32_KEEP_TAPE": value}, clear=False
    ):
      assert parse() is False
  with mock.patch.dict(os.environ, {}, clear=False):
    os.environ.pop("CANON_P32_KEEP_TAPE", None)
    assert parse() is False
  with mock.patch.dict(os.environ, {"CANON_P32_KEEP_TAPE": "1"}, clear=False):
    assert parse() is True
  for junk in ("off", "yes", "2", "true"):
    with mock.patch.dict(
        os.environ, {"CANON_P32_KEEP_TAPE": junk}, clear=False
    ):
      with pytest.raises(FME, match="CANON_P32_KEEP_TAPE"):
        parse()


def _forwards(rank_parallel):
  adapter, runner = harness._group_adapter(rank_parallel)  # pylint: disable=protected-access
  spec = harness._two_chunk_spec(adapter)  # pylint: disable=protected-access
  with mock.patch.dict(os.environ, _env(rank_parallel), clear=False):
    os.environ.pop("CANON_P71_SCAN", None)
    os.environ.pop("CANON_P28_LAYER_SCAN", None)
    engine = canonical_qwen3_adapter.build_p28_segmented_engine_forward(
        runner
    )
    leaves = tuple(runner.state_leaves)
    plain = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=False
    )
    kept = adapter._p32_forward_group(  # pylint: disable=protected-access
        engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
    )
  return adapter, engine, leaves, spec, plain, kept


def test_keep_tape_retains_the_forward_it_already_did():
  _, _, _, spec, plain, kept = _forwards(rank_parallel=True)
  tree_bytes = harness._tree_bytes  # pylint: disable=protected-access
  # The forward's own outputs are untouched by keeping.
  assert tree_bytes(kept["logps"]) == tree_bytes(plain["logps"])
  assert tree_bytes(kept["entropy"]) == tree_bytes(plain["entropy"])
  assert kept["counts"] == plain["counts"]
  # One tape entry per chunk; one hidden per layer, aligned with the caches.
  assert len(kept["hidden_inputs"]) == spec["num_chunks"]
  assert len(kept["final_hiddens"]) == spec["num_chunks"]
  for caches, hidden_ins in zip(kept["cache_inputs"], kept["hidden_inputs"]):
    assert len(hidden_ins) == len(caches)
  # Nothing is kept when not asked.
  assert plain["hidden_inputs"] == ()
  assert plain["final_hiddens"] == ()


def test_keep_tape_requires_cache_inputs_and_per_layer_forward():
  adapter, engine, leaves, spec, _, _ = _forwards(rank_parallel=True)
  with mock.patch.dict(os.environ, _env(True), clear=False):
    with pytest.raises(FME, match="keep_tape requires keep_cache_inputs"):
      adapter._p32_forward_group(  # pylint: disable=protected-access
          engine, leaves, spec, keep_cache_inputs=False, keep_tape=True
      )
    with mock.patch.dict(
        os.environ, {"CANON_P28_LAYER_SCAN": "1"}, clear=False
    ):
      with pytest.raises(FME, match="requires the per-layer forward"):
        adapter._p32_forward_group(  # pylint: disable=protected-access
            engine, leaves, spec, keep_cache_inputs=True, keep_tape=True
        )


@pytest.mark.parametrize("scan", ["", "fwd"], ids=["legacy-tape", "scan-tape"])
def test_reverse_from_kept_tape_is_bitwise_and_skips_the_rebuild(scan):
  adapter, engine, leaves, spec, plain, kept = _forwards(rank_parallel=True)
  tree_bytes = harness._tree_bytes  # pylint: disable=protected-access
  dlogps = jnp.ones_like(plain["logps"])
  dentropy = jnp.zeros_like(plain["entropy"])
  env = _env(True)
  if scan:
    env["CANON_P71_SCAN"] = scan
  with mock.patch.dict(os.environ, env, clear=False):
    if not scan:
      os.environ.pop("CANON_P71_SCAN", None)
    # Reference: the reverse regenerates its own replay and tape.
    rebuilt = adapter._p32_reverse_group(  # pylint: disable=protected-access
        engine, leaves, spec, dlogps, dentropy
    )
    # Treatment: the reverse consumes the tape the forward kept.
    consumed = adapter._p32_reverse_group(  # pylint: disable=protected-access
        engine, leaves, spec, dlogps, dentropy, replay=kept
    )
  assert tree_bytes(consumed["engine_gradients"]) == tree_bytes(
      rebuilt["engine_gradients"]
  )
  assert tree_bytes(consumed["initial_cache_cotangents"]) == tree_bytes(
      rebuilt["initial_cache_cotangents"]
  )
  assert tree_bytes(consumed["replay_logps"]) == tree_bytes(
      rebuilt["replay_logps"]
  )
  # The rebuild is gone: no embed and no layer forward ran in the reverse
  # beyond what the kept forward already counted.
  if "counts" in consumed:
    assert consumed["counts"]["layer_forward"] == kept["counts"]["layer_forward"]
    assert consumed["counts"]["embed_forward"] == kept["counts"]["embed_forward"]
    assert rebuilt["counts"]["layer_forward"] > kept["counts"]["layer_forward"]
