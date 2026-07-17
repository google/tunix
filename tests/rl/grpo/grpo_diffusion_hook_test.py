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

"""CPU unit tests for the pluggable per-token-logprob hook in `grpo_loss_fn`.

Verifies the GRPO loss's new-policy logprob is pluggable so a non-AR policy (e.g.
a block-diffusion policy whose rollout generates by denoising) can inject the SAME
logprob function the rollout uses for its old-policy logps — the P0 invariant that
keeps the importance ratio unbiased:

  * gating OFF (default, ``per_token_logps_fn=None``): the loss routes the
    new-policy logps through ``common.compute_per_token_logps`` (the autoregressive
    next-token logprob) and NOT the override -> AR RL is unchanged;
  * gating ON (``per_token_logps_fn=<fn>``): the loss routes through the injected
    function and NOT ``common.compute_per_token_logps``;
  * only the logprob SOURCE changes: with both paths returning identical logps the
    loss is byte-for-byte identical (clipped surrogate / advantage / KL untouched);
  * ``GRPOLearner.__init__`` accepts and forwards ``per_token_logps_fn`` and only
    forwards it when set (source/AST checks).

``grpo_learner.py`` is loaded standalone with lightweight stubs for its tunix.*
imports (and a spyable ``common``) so the loss's routing is exercised on a bare
jax[cpu] venv without the full tunix dependency tree. The end-to-end GRPO step with
real models/rollout needs TPU + weights and is covered elsewhere.
"""

import ast
import dataclasses
import importlib.util
import pathlib
import sys
import types

import flax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_GRPO_LEARNER = _REPO_ROOT / "tunix" / "rl" / "grpo" / "grpo_learner.py"


# ---------------------------------------------------------------------------
# Standalone loader: stub grpo_learner.py's tunix.* imports (with a spyable
# `common`) so the loss routing runs without the full tunix install.
# ---------------------------------------------------------------------------
def _build_common_stub():
  """A `tunix.rl.common` stand-in: spyable logps + real KL/aggregation math."""

  @flax.struct.dataclass(frozen=True)
  class TrainExample:
    prompt_ids: jax.Array
    prompt_mask: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    advantages: jax.Array
    ref_per_token_logps: jax.Array | None
    old_per_token_logps: jax.Array | None
    segment_ids: jax.Array | None = None
    segment_positions: jax.Array | None = None

  calls = {"count": 0}

  def compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      completion_mask=None,
      stop_gradient=True,
      return_logits=False,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
  ):
    del (
        graphdef,
        state,
        prompt_tokens,
        pad_id,
        eos_id,
        images,
        completion_mask,
        stop_gradient,
        return_logits,
        segment_ids,
        segment_positions,
        temperature,
    )
    calls["count"] += 1
    # Deterministic, non-trivial logps so the surrogate is non-degenerate.
    idx = jnp.arange(completion_tokens.shape[1], dtype=jnp.float32)
    return -0.1 * (
        idx[None, :] + completion_tokens.astype(jnp.float32) * 0.0 + 1.0
    )

  def compute_kl_divergence(
      per_token_logps, ref_per_token_logps, method="low_var_kl"
  ):
    per_token_logps = per_token_logps.astype(jnp.float32)
    ref_per_token_logps = ref_per_token_logps.astype(jnp.float32)
    kl = ref_per_token_logps - per_token_logps
    return jnp.exp(kl) - kl - 1

  def aggregate_loss(per_token_loss, completion_mask, loss_agg_mode, **kwargs):
    del kwargs
    per_token_loss = per_token_loss.astype(jnp.float32)
    if loss_agg_mode == "token-mean":
      return (per_token_loss * completion_mask).sum() / jnp.clip(
          completion_mask.sum(), min=1
      )
    seq_mask = completion_mask.sum(axis=-1)
    non_zero_rows = jnp.clip((seq_mask > 0).sum(), min=1)
    seq_loss = ((per_token_loss * completion_mask).sum(axis=-1)) / jnp.clip(
        seq_mask, min=1
    )
    return seq_loss.sum() / non_zero_rows

  mod = types.ModuleType("tunix.rl.common")
  mod.TrainExample = TrainExample
  mod.compute_per_token_logps = compute_per_token_logps
  mod.compute_kl_divergence = compute_kl_divergence
  mod.aggregate_loss = aggregate_loss
  return mod, calls


def _install_stub(name, **attrs):
  m = types.ModuleType(name)
  m.__path__ = []
  for k, v in attrs.items():
    setattr(m, k, v)
  sys.modules[name] = m
  return m


def _load_grpo_learner():
  """Loads grpo_learner.py against stubbed tunix.* deps; returns (module, common)."""
  common_mod, calls = _build_common_stub()

  # Parent packages.
  for pkg in (
      "tunix",
      "tunix.rl",
      "tunix.generate",
      "tunix.perf",
      "tunix.perf.experimental",
  ):
    if pkg not in sys.modules:
      _install_stub(pkg)

  # Leaf stubs used at grpo_learner import time.
  _install_stub("tunix.generate.utils", pad_to_length=lambda *a, **k: None)
  _install_stub(
      "tunix.perf.experimental.constants",
      STEP="step",
      REFERENCE_INFERENCE="ref",
      OLD_ACTOR_INFERENCE="old",
      ADVANTAGE_COMPUTATION="adv",
  )

  @dataclasses.dataclass(kw_only=True)
  class AlgorithmConfig:  # minimal base for GRPOConfig
    pass

  _install_stub("tunix.rl.algorithm_config", AlgorithmConfig=AlgorithmConfig)
  sys.modules["tunix.rl.common"] = common_mod

  class _Registry:

    def __init__(self):
      self._fns = {}

    def register(self, category):
      def deco(fn):
        self._fns[(category, fn.__name__)] = fn
        return fn

      return deco

    def get(self, name):
      return None

  _reg = _Registry()
  _install_stub(
      "tunix.rl.function_registry",
      register_policy_loss_fn=lambda name: _reg.register("policy"),
      register_advantage_estimator=lambda name: _reg.register("adv"),
      register_reward_manager=lambda name: _reg.register("reward"),
      get_policy_loss_fn=lambda name: _reg.get(name),
      get_advantage_estimator=lambda name: _reg.get(name),
  )

  import enum

  class Mode(enum.Enum):
    TRAIN = "train"
    EVAL = "eval"

  class Role(enum.Enum):
    ACTOR = "actor"
    REFERENCE = "reference"

  _install_stub(
      "tunix.rl.rl_cluster",
      RLCluster=type("RLCluster", (), {}),
      Mode=Mode,
      Role=Role,
  )

  import typing

  T = typing.TypeVar("T")

  class RLLearner(typing.Generic[T]):

    def __init__(self, *a, **k):
      pass

  _install_stub(
      "tunix.rl.rl_learner",
      RLLearner=RLLearner,
      TrainingInputT=dict,
      RewardFn=typing.Callable,
      MetricFn=typing.Callable,
  )

  spec = importlib.util.spec_from_file_location(
      "grpo_learner_under_test", _GRPO_LEARNER
  )
  mod = importlib.util.module_from_spec(spec)
  # Register before exec so `@dataclasses.dataclass` can resolve `cls.__module__`.
  sys.modules[spec.name] = mod
  spec.loader.exec_module(mod)
  return mod, common_mod, calls


_GL, _COMMON, _COMMON_CALLS = _load_grpo_learner()
grpo_loss_fn = _GL.grpo_loss_fn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
class _TinyModel(nnx.Module):

  def __init__(self):
    self.w = nnx.Param(jnp.zeros((), dtype=jnp.float32))


def _algo_config(beta=0.0):
  return types.SimpleNamespace(
      beta=beta,
      epsilon=0.2,
      epsilon_high=0.2,
      loss_algo="grpo",
      loss_agg_mode="token-mean",
      temperature=1.0,
  )


def _train_example(batch=2, comp_len=4):
  return _COMMON.TrainExample(
      prompt_ids=jnp.ones((batch, 3), dtype=jnp.int32),
      prompt_mask=jnp.ones((batch, 3), dtype=jnp.int32),
      completion_ids=jnp.arange(
          1, batch * comp_len + 1, dtype=jnp.int32
      ).reshape(batch, comp_len),
      completion_mask=jnp.ones((batch, comp_len), dtype=jnp.int32),
      advantages=jnp.array([1.0, -1.0])[:batch],
      ref_per_token_logps=None,
      old_per_token_logps=None,
  )


class _Spy:

  def __init__(self, like=_COMMON.compute_per_token_logps):
    self.count = 0

  def __call__(
      self,
      graphdef,
      state,
      prompt_tokens,
      completion_tokens,
      pad_id,
      eos_id,
      images=None,
      completion_mask=None,
      stop_gradient=True,
      return_logits=False,
      segment_ids=None,
      segment_positions=None,
      temperature=1.0,
  ):
    del (
        graphdef,
        state,
        prompt_tokens,
        pad_id,
        eos_id,
        images,
        completion_mask,
        stop_gradient,
        return_logits,
        segment_ids,
        segment_positions,
        temperature,
    )
    self.count += 1
    idx = jnp.arange(completion_tokens.shape[1], dtype=jnp.float32)
    return -0.1 * (idx[None, :] + 1.0)


# ---------------------------------------------------------------------------
# Gating tests
# ---------------------------------------------------------------------------
def test_default_uses_common_compute_per_token_logps():
  """Flag OFF: the AR default `common.compute_per_token_logps` is used, not an override."""
  model = _TinyModel()
  before = _COMMON_CALLS["count"]
  loss, aux = grpo_loss_fn(
      model, _train_example(), _algo_config(), pad_id=0, eos_id=1
  )
  assert _COMMON_CALLS["count"] == before + 1  # AR default invoked exactly once
  assert np.isfinite(float(loss))
  assert "pg_clipfrac" in aux and "kl" in aux


def test_override_uses_injected_fn_not_common():
  """Flag ON: the injected fn is used and `common.compute_per_token_logps` is NOT."""
  model = _TinyModel()
  spy = _Spy()
  before = _COMMON_CALLS["count"]
  loss, _ = grpo_loss_fn(
      model,
      _train_example(),
      _algo_config(),
      pad_id=0,
      eos_id=1,
      per_token_logps_fn=spy,
  )
  assert spy.count == 1  # override invoked
  assert _COMMON_CALLS["count"] == before  # AR default NOT invoked
  assert np.isfinite(float(loss))


def test_only_logps_source_changes_surrogate_identical():
  """If the override returns the same logps as the default, the loss is identical."""
  model = _TinyModel()
  ex = _train_example()
  cfg = _algo_config()

  # An override that reproduces the stub default's logps exactly.
  def same_as_default(
      graphdef, state, prompt_tokens, completion_tokens, pad_id, eos_id, **kw
  ):
    del graphdef, state, prompt_tokens, pad_id, eos_id, kw
    idx = jnp.arange(completion_tokens.shape[1], dtype=jnp.float32)
    return -0.1 * (
        idx[None, :] + completion_tokens.astype(jnp.float32) * 0.0 + 1.0
    )

  loss_default, _ = grpo_loss_fn(model, ex, cfg, pad_id=0, eos_id=1)
  loss_override, _ = grpo_loss_fn(
      model, ex, cfg, pad_id=0, eos_id=1, per_token_logps_fn=same_as_default
  )
  np.testing.assert_allclose(
      float(loss_default), float(loss_override), rtol=0, atol=0
  )


def test_override_with_kl_penalty_runs():
  """Override path also works with the KL penalty (beta != 0) enabled."""
  model = _TinyModel()
  spy = _Spy()
  ex = dataclasses.replace(
      _train_example(),
      ref_per_token_logps=jnp.full((2, 4), -0.2, dtype=jnp.float32),
  )
  loss, aux = grpo_loss_fn(
      model,
      ex,
      _algo_config(beta=0.04),
      pad_id=0,
      eos_id=1,
      per_token_logps_fn=spy,
  )
  assert spy.count == 1
  assert np.isfinite(float(loss))


# ---------------------------------------------------------------------------
# Source/structure checks (learner wiring; instantiation needs a full RLCluster)
# ---------------------------------------------------------------------------
def _tree(node_name, method_name=None):
  src = _GRPO_LEARNER.read_text()
  tree = ast.parse(src)
  for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == node_name:
      if method_name is None:
        return node, src
      for sub in ast.walk(node):
        if isinstance(sub, ast.FunctionDef) and sub.name == method_name:
          return sub, src
    if isinstance(node, ast.FunctionDef) and node.name == node_name:
      return node, src
  return None, src


def test_grpo_loss_fn_defaults_to_common():
  fn, src = _tree("grpo_loss_fn")
  assert fn is not None
  args = {a.arg for a in fn.args.args}
  assert "per_token_logps_fn" in args
  body = ast.get_source_segment(src, fn)
  assert "common.compute_per_token_logps" in body  # the AR default
  assert "logps_fn(" in body  # routed through the selected fn


def test_learner_accepts_and_forwards_hook():
  init, src = _tree("GRPOLearner", "__init__")
  assert init is not None
  args = {a.arg for a in init.args.args}
  assert "per_token_logps_fn" in args
  body = ast.get_source_segment(src, init)
  assert "self._per_token_logps_fn = per_token_logps_fn" in body
  # Only forwarded when set, so other loss fns / the AR default are untouched.
  assert "if self._per_token_logps_fn is not None" in body
  assert (
      'extra_loss_kwargs["per_token_logps_fn"] = self._per_token_logps_fn'
      in body
  )


if __name__ == "__main__":
  raise SystemExit(pytest.main([__file__, "-q"]))
