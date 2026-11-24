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

"""Unit tests for `AbstractTrainer` protocol and dummy MoE payload resolution."""

from collections.abc import Callable
import dataclasses
import importlib
import importlib.machinery
import sys
import types
from typing import Any, Self, cast
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.common import test_utils as common_test_utils
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.train import abstract_trainer

_EXPECTED_ABSTRACT_METHODS = frozenset({
    "close",
    "compile",
    "eval_step",
    "fwd_bwd",
    "fwd_only",
    "get_metrics",
    "prepare_weight_sync",
    "restore_checkpoint",
    "save_checkpoint",
    "update",
    "with_fwd_only_fn",
    "with_gen_model_input_fn",
    "with_loss_fn",
})


@dataclasses.dataclass
class _MoeCfg:
  num_layers: int | None = 3
  num_decoder_layers: int = 5
  num_experts_per_tok: int = 2
  num_experts: int = 8


@dataclasses.dataclass
class _FakeModel:
  config: _MoeCfg | None = None
  base: Any = None


class _NominalTrainer(abstract_trainer.AbstractTrainer):
  """Complete nominal subclass exercising inherited optional hooks."""

  def __init__(self, config: Any = None):
    self.config = config
    self._config: Any = None
    self.model = _FakeModel(config=_MoeCfg())
    self._fwd_only_fn: Callable[..., Any] | None = None

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> Self:
    del loss_fn, has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> Self:
    del gen_model_input_fn
    return self

  def compile(
      self, dummy_data: datatypes.TrainerPayload | None = None
  ) -> None:
    del dummy_data

  def fwd_bwd(self, payload: datatypes.TrainerPayload) -> None:
    del payload

  def update(self) -> int:
    return 1

  def eval_step(self, payload: datatypes.TrainerPayload) -> None:
    del payload

  def with_fwd_only_fn(self, fn: Callable[..., Any]) -> Self:
    self._fwd_only_fn = fn
    return self

  def fwd_only(self, payload: datatypes.TrainerPayload) -> Any:
    if self._fwd_only_fn is None:
      raise ValueError("with_fwd_only_fn must be called first.")
    return self._fwd_only_fn(self.model, payload)

  def save_checkpoint(
      self,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> None:
    del metadata, step, force

  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    return {"step": step or 0}

  def prepare_weight_sync(self, *, sync_request: Any = None) -> Any:
    return sync_request

  def get_metrics(self) -> datatypes.MetricsBuffer:
    return datatypes.MetricsBuffer(id="nominal")

  def close(self) -> None:
    pass


@dataclasses.dataclass(frozen=True)
class _HostWeightedMetric:
  """Host wire `WeightedMetricLike` double."""

  unreduced_sum: float
  denominator: float
  eps: float | None = None
  min_denom: float | None = None


@dataclasses.dataclass(frozen=True)
class _StructuralWeightedMetric:
  """Non-NumPy `TrainerWeightedMetricLike` double for on-device metrics."""

  unreduced_sum: common_test_utils.CustomTensor
  denominator: common_test_utils.CustomTensor
  eps: float | None = None
  min_denom: float | None = None


@dataclasses.dataclass
class _StructuralMetricsBuffer:
  """Standalone buffer mirroring MaxText's on-device `MetricsBuffer`."""

  id: Any
  weighted_metrics: dict[str, _StructuralWeightedMetric] = dataclasses.field(
      default_factory=dict
  )
  scalar_metrics: dict[str, common_test_utils.CustomTensor] = dataclasses.field(
      default_factory=dict
  )
  aggregation_fns: dict[
      str, Callable[[common_test_utils.CustomTensor], Any]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


class _StructuralEngine:
  """Standalone engine with read-only `model` mirroring external engines."""

  def __init__(self, model: _FakeModel):
    self._model = model
    self._config: Any = None

  @property
  def model(self) -> _FakeModel:
    return self._model

  def with_loss_fn(
      self, customized_fn: Callable[..., Any], has_aux: bool = False
  ) -> "_StructuralEngine":
    del customized_fn, has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "_StructuralEngine":
    del gen_model_input_fn
    return self

  def compile(
      self,
      dummy_data: datatypes.TrainerPayload | None = None,
      compiler_options: dict[str, Any] | None = None,
  ) -> None:
    del dummy_data, compiler_options

  def fwd_bwd(self, payload: datatypes.TrainerPayload) -> None:
    del payload

  def update(self) -> int:
    return 1

  def eval_step(self, payload: datatypes.TrainerPayload) -> None:
    del payload

  def with_fwd_only_fn(self, fn: Callable[..., Any]) -> "_StructuralEngine":
    del fn
    return self

  def fwd_only(self, payload: datatypes.TrainerPayload) -> Any:
    return payload

  def save_checkpoint(
      self,
      metadata: Any = None,
      *,
      step: int | None = None,
      force: bool = False,
  ) -> None:
    del metadata, step, force

  def restore_checkpoint(self, *, step: int | None = None) -> Any:
    return {"step": step or 0}

  def set_target_state(self, target_state: Any) -> None:
    del target_state

  def prepare_weight_sync(
      self, *, sync_request: Any = None, staging_transport: str = "raiden"
  ) -> Any:
    del staging_transport
    return sync_request

  def release_weight_sync(self, *, sync_request: Any = None) -> Any:
    return sync_request

  def get_metrics(self, clear_cache: bool = True) -> _StructuralMetricsBuffer:
    return _StructuralMetricsBuffer(
        id=0,
        weighted_metrics={
            "kl": _StructuralWeightedMetric(
                unreduced_sum=common_test_utils.CustomTensor(np.array(3.0)),
                denominator=common_test_utils.CustomTensor(np.array(2.0)),
            )
        },
        scalar_metrics={
            "cleared": common_test_utils.CustomTensor(
                np.array(float(clear_cache))
            )
        },
        aggregation_fns={"cleared": lambda arr: float(np.asarray(arr))},
    )

  def close(self) -> None:
    pass


def _accept_trainer(
    trainer: abstract_trainer.AbstractTrainer,
) -> abstract_trainer.AbstractTrainer:
  return trainer


def _make_rl_payload(
    routed_experts: np.ndarray | None,
    metadata: dict[str, Any] | None = None,
) -> datatypes.RLTrainerPayload:
  return datatypes.RLTrainerPayload(
      prompt_ids=np.zeros((2, 4), dtype=np.int32),
      prompt_mask=np.ones((2, 4), dtype=np.int32),
      completion_ids=np.zeros((2, 3), dtype=np.int32),
      completion_mask=np.ones((2, 3), dtype=np.int32),
      advantages=np.ones((2, 3), dtype=np.float32),
      routed_experts=routed_experts,
      metadata=metadata or {},
  )


class AbstractTrainerTest(absltest.TestCase):

  def test_abstract_methods_preserved_and_incomplete_subclass_rejected(self):
    self.assertEqual(
        abstract_trainer.AbstractTrainer.__abstractmethods__,
        _EXPECTED_ABSTRACT_METHODS,
    )

    class _IncompleteTrainer(abstract_trainer.AbstractTrainer):
      pass

    incomplete_cls: Any = _IncompleteTrainer
    with self.assertRaises(TypeError):
      incomplete_cls()

    trainer = _NominalTrainer()
    payload = datatypes.TrainerPayload()
    stub_calls: list[tuple[str, Callable[[], Any]]] = [
        (
            "with_loss_fn",
            lambda: abstract_trainer.AbstractTrainer.with_loss_fn(
                trainer, lambda m: 0.0
            ),
        ),
        (
            "with_gen_model_input_fn",
            lambda: abstract_trainer.AbstractTrainer.with_gen_model_input_fn(
                trainer, lambda p: {}
            ),
        ),
        ("compile", lambda: abstract_trainer.AbstractTrainer.compile(trainer)),
        (
            "fwd_bwd",
            lambda: abstract_trainer.AbstractTrainer.fwd_bwd(trainer, payload),
        ),
        ("update", lambda: abstract_trainer.AbstractTrainer.update(trainer)),
        (
            "eval_step",
            lambda: abstract_trainer.AbstractTrainer.eval_step(
                trainer, payload
            ),
        ),
        (
            "with_fwd_only_fn",
            lambda: abstract_trainer.AbstractTrainer.with_fwd_only_fn(
                trainer, lambda m, p: p
            ),
        ),
        (
            "fwd_only",
            lambda: abstract_trainer.AbstractTrainer.fwd_only(trainer, payload),
        ),
        (
            "save_checkpoint",
            lambda: abstract_trainer.AbstractTrainer.save_checkpoint(trainer),
        ),
        (
            "restore_checkpoint",
            lambda: abstract_trainer.AbstractTrainer.restore_checkpoint(
                trainer
            ),
        ),
        (
            "prepare_weight_sync",
            lambda: abstract_trainer.AbstractTrainer.prepare_weight_sync(
                trainer
            ),
        ),
        (
            "get_metrics",
            lambda: abstract_trainer.AbstractTrainer.get_metrics(trainer),
        ),
        ("close", lambda: abstract_trainer.AbstractTrainer.close(trainer)),
    ]
    self.assertEqual(
        {name for name, _ in stub_calls}, _EXPECTED_ABSTRACT_METHODS
    )
    for name, call in stub_calls:
      with self.assertRaisesRegex(
          NotImplementedError,
          rf"_NominalTrainer does not implement {name}\.",
      ):
        call()

  def test_nominal_subclass_inherits_default_hooks(self):
    trainer = _NominalTrainer()
    self.assertIs(_accept_trainer(trainer), trainer)
    self.assertIsInstance(trainer, abstract_trainer.AbstractTrainer)
    self.assertIsNone(trainer.set_target_state({"w": 1}))
    self.assertIsNone(trainer.release_weight_sync(sync_request="req-1"))
    self.assertIsNone(trainer.close())

  def test_structural_non_inheriting_engine_conforms(self):
    engine = _StructuralEngine(_FakeModel(config=_MoeCfg()))
    self.assertIsNone(_StructuralEngine.model.fset)
    self.assertNotIn(
        abstract_trainer.AbstractTrainer, type(engine).__mro__
    )
    self.assertIs(_accept_trainer(engine), engine)
    self.assertIsInstance(engine, abstract_trainer.AbstractTrainer)
    metrics_buf: abstract_trainer.MetricsBufferLike = engine.get_metrics()
    self.assertIsInstance(metrics_buf, abstract_trainer.MetricsBufferLike)
    self.assertAlmostEqual(
        float(np.asarray(metrics_buf.weighted_metrics["kl"].unreduced_sum)),
        3.0,
    )
    self.assertAlmostEqual(
        metrics_buf.aggregation_fns["cleared"](
            metrics_buf.scalar_metrics["cleared"]
        ),
        1.0,
    )
    self.assertNotIsInstance(object(), abstract_trainer.AbstractTrainer)

    class _StructuralWithoutSetTargetState:
      model = _FakeModel(config=_MoeCfg())
      with_loss_fn = _StructuralEngine.with_loss_fn
      with_gen_model_input_fn = _StructuralEngine.with_gen_model_input_fn
      compile = _StructuralEngine.compile
      fwd_bwd = _StructuralEngine.fwd_bwd
      update = _StructuralEngine.update
      eval_step = _StructuralEngine.eval_step
      with_fwd_only_fn = _StructuralEngine.with_fwd_only_fn
      fwd_only = _StructuralEngine.fwd_only
      save_checkpoint = _StructuralEngine.save_checkpoint
      restore_checkpoint = _StructuralEngine.restore_checkpoint
      prepare_weight_sync = _StructuralEngine.prepare_weight_sync
      release_weight_sync = _StructuralEngine.release_weight_sync
      get_metrics = _StructuralEngine.get_metrics
      close = _StructuralEngine.close

    class _StructuralWithoutReleaseWeightSync:
      model = _FakeModel(config=_MoeCfg())
      with_loss_fn = _StructuralEngine.with_loss_fn
      with_gen_model_input_fn = _StructuralEngine.with_gen_model_input_fn
      compile = _StructuralEngine.compile
      fwd_bwd = _StructuralEngine.fwd_bwd
      update = _StructuralEngine.update
      eval_step = _StructuralEngine.eval_step
      with_fwd_only_fn = _StructuralEngine.with_fwd_only_fn
      fwd_only = _StructuralEngine.fwd_only
      save_checkpoint = _StructuralEngine.save_checkpoint
      restore_checkpoint = _StructuralEngine.restore_checkpoint
      set_target_state = _StructuralEngine.set_target_state
      prepare_weight_sync = _StructuralEngine.prepare_weight_sync
      get_metrics = _StructuralEngine.get_metrics
      close = _StructuralEngine.close

    self.assertNotIsInstance(
        _StructuralWithoutSetTargetState(), abstract_trainer.AbstractTrainer
    )
    self.assertNotIsInstance(
        _StructuralWithoutReleaseWeightSync(), abstract_trainer.AbstractTrainer
    )

  def test_metrics_buffer_like_conformance_and_rejection(self):
    host_metric = _HostWeightedMetric(
        unreduced_sum=3.0, denominator=2.0, eps=1e-8
    )
    wire_metric: datatypes.WeightedMetricLike = host_metric
    trainer_from_wire: abstract_trainer.TrainerWeightedMetricLike = wire_metric
    self.assertEqual(trainer_from_wire.unreduced_sum, 3.0)
    self.assertEqual(trainer_from_wire.denominator, 2.0)
    self.assertEqual(trainer_from_wire.eps, 1e-8)

    device_metric = _StructuralWeightedMetric(
        unreduced_sum=common_test_utils.CustomTensor(np.array(4.0)),
        denominator=common_test_utils.CustomTensor(np.array(2.0)),
        eps=1e-6,
    )
    trainer_from_device: abstract_trainer.TrainerWeightedMetricLike = (
        device_metric
    )
    self.assertAlmostEqual(
        float(np.asarray(trainer_from_device.unreduced_sum)), 4.0
    )
    self.assertEqual(trainer_from_device.eps, 1e-6)

    host_buf: abstract_trainer.MetricsBufferLike = datatypes.MetricsBuffer(
        id=1,
        weighted_metrics={"kl": host_metric},
        scalar_metrics={"loss": np.float32(0.5)},
        aggregation_fns={"loss": np.mean},
    )
    self.assertIsInstance(host_buf, abstract_trainer.MetricsBufferLike)
    self.assertAlmostEqual(
        float(
            host_buf.aggregation_fns["loss"](host_buf.scalar_metrics["loss"])
        ),
        0.5,
    )

    device_buf: abstract_trainer.MetricsBufferLike = _StructuralMetricsBuffer(
        id="step-1",
        weighted_metrics={"kl": device_metric},
        scalar_metrics={
            "loss": common_test_utils.CustomTensor(np.array(0.25))
        },
        aggregation_fns={"loss": lambda arr: float(np.asarray(arr))},
    )
    self.assertIsInstance(device_buf, abstract_trainer.MetricsBufferLike)
    self.assertAlmostEqual(
        device_buf.aggregation_fns["loss"](device_buf.scalar_metrics["loss"]),
        0.25,
    )

    self.assertNotIsInstance({}, abstract_trainer.MetricsBufferLike)
    self.assertNotIsInstance({"id": 1}, abstract_trainer.MetricsBufferLike)
    self.assertNotIsInstance(object(), abstract_trainer.MetricsBufferLike)

  def test_resolve_dummy_routed_experts_passthrough_cases(self):
    trainer = _NominalTrainer()
    plain = datatypes.TrainerPayload()
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(trainer, plain), plain
    )
    none_experts = _make_rl_payload(routed_experts=None)
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(trainer, none_experts),
        none_experts,
    )
    populated = _make_rl_payload(
        routed_experts=np.ones((2, 7, 3, 2), dtype=np.int32)
    )
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(trainer, populated),
        populated,
    )

  def test_resolve_dummy_routed_experts_expands_and_encodes_on_structural_engine(
      self,
  ):
    wrapped = _FakeModel(
        config=None,
        base=_FakeModel(config=_MoeCfg(num_layers=None, num_decoder_layers=4)),
    )
    engine = _StructuralEngine(wrapped)
    placeholder = np.zeros((2, 7, 0, 0), dtype=np.int32)
    payload = _make_rl_payload(
        routed_experts=placeholder,
        metadata={
            "keep": "yes",
            "routed_experts_encode_fn": (
                lambda arr, *, num_experts: arr + num_experts
            ),
        },
    )
    resolved = abstract_trainer.resolve_dummy_routed_experts(engine, payload)
    self.assertEqual(resolved.metadata, {"keep": "yes"})
    assert resolved.routed_experts is not None
    self.assertEqual(resolved.routed_experts.shape, (2, 7, 4, 2))
    np.testing.assert_array_equal(resolved.routed_experts, 8)
    # Idempotent once non-empty.
    self.assertIs(
        abstract_trainer.resolve_dummy_routed_experts(engine, resolved),
        resolved,
    )

  def test_resolve_dummy_routed_experts_validates_rank_and_config(self):
    engine = _StructuralEngine(_FakeModel(config=None))
    bad_rank = _make_rl_payload(routed_experts=np.zeros((0,), dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "rank >= 2"):
      abstract_trainer.resolve_dummy_routed_experts(engine, bad_rank)

    missing_cfg = _make_rl_payload(
        routed_experts=np.zeros((2, 7, 0, 0), dtype=np.int32)
    )
    with self.assertRaisesRegex(ValueError, "does not expose a non-None"):
      abstract_trainer.resolve_dummy_routed_experts(engine, missing_cfg)

    engine._config = _MoeCfg(num_layers=2, num_experts_per_tok=3)
    resolved = abstract_trainer.resolve_dummy_routed_experts(
        engine, missing_cfg
    )
    assert resolved.routed_experts is not None
    self.assertEqual(resolved.routed_experts.shape, (2, 7, 2, 3))

  def test_nominal_subclass_with_none_model_falls_back_to_config(self):
    trainer = _NominalTrainer()
    trainer.model = None
    self.assertIsNone(trainer.model)
    self.assertIsInstance(trainer, abstract_trainer.AbstractTrainer)
    trainer._config = _MoeCfg(num_layers=2, num_experts_per_tok=4)  # pylint: disable=protected-access
    payload = _make_rl_payload(
        routed_experts=np.zeros((2, 5, 0, 0), dtype=np.int32)
    )
    resolved = abstract_trainer.resolve_dummy_routed_experts(trainer, payload)
    assert resolved.routed_experts is not None
    self.assertEqual(resolved.routed_experts.shape, (2, 5, 2, 4))

  def test_missing_model_fails_fast(self):
    class _NominalMissingModel(abstract_trainer.AbstractTrainer):

      def __init__(self, config: Any = None):
        self.config = config
        self._config: Any = _MoeCfg()
        self._fwd_only_fn: Callable[..., Any] | None = None

      with_loss_fn = _NominalTrainer.with_loss_fn
      with_gen_model_input_fn = _NominalTrainer.with_gen_model_input_fn
      compile = _NominalTrainer.compile
      fwd_bwd = _NominalTrainer.fwd_bwd
      update = _NominalTrainer.update
      eval_step = _NominalTrainer.eval_step
      with_fwd_only_fn = _NominalTrainer.with_fwd_only_fn
      fwd_only = _NominalTrainer.fwd_only
      save_checkpoint = _NominalTrainer.save_checkpoint
      restore_checkpoint = _NominalTrainer.restore_checkpoint
      prepare_weight_sync = _NominalTrainer.prepare_weight_sync
      get_metrics = _NominalTrainer.get_metrics
      close = _NominalTrainer.close

    class _StructuralWithoutModel:

      def __init__(self, config: Any = None):
        self.config = config

      with_loss_fn = _StructuralEngine.with_loss_fn
      with_gen_model_input_fn = _StructuralEngine.with_gen_model_input_fn
      compile = _StructuralEngine.compile
      fwd_bwd = _StructuralEngine.fwd_bwd
      update = _StructuralEngine.update
      eval_step = _StructuralEngine.eval_step
      with_fwd_only_fn = _StructuralEngine.with_fwd_only_fn
      fwd_only = _StructuralEngine.fwd_only
      save_checkpoint = _StructuralEngine.save_checkpoint
      restore_checkpoint = _StructuralEngine.restore_checkpoint
      set_target_state = _StructuralEngine.set_target_state
      prepare_weight_sync = _StructuralEngine.prepare_weight_sync
      release_weight_sync = _StructuralEngine.release_weight_sync
      get_metrics = _StructuralEngine.get_metrics
      close = _StructuralEngine.close

    nominal_missing = _NominalMissingModel()
    self.assertIsInstance(nominal_missing, abstract_trainer.AbstractTrainer)
    self.assertFalse(hasattr(nominal_missing, "model"))
    with self.assertRaisesRegex(
        AttributeError,
        "'_NominalMissingModel' object has no attribute 'model'",
    ):
      _ = nominal_missing.model
    payload = _make_rl_payload(
        routed_experts=np.zeros((2, 5, 0, 0), dtype=np.int32)
    )
    with self.assertRaisesRegex(
        AttributeError,
        "'_NominalMissingModel' object has no attribute 'model'",
    ):
      abstract_trainer.resolve_dummy_routed_experts(nominal_missing, payload)

    structural_missing: Any = _StructuralWithoutModel()
    self.assertNotIsInstance(
        structural_missing, abstract_trainer.AbstractTrainer
    )
    with self.assertRaises(AttributeError):
      _ = structural_missing.model

  def test_importing_abstract_trainer_keeps_ml_frameworks_unloaded(self):
    forbidden = ("jax", "flax", "optax", "torch", "tpu_raiden")

    def _matches_pkg(mod_name: str, pkg: str) -> bool:
      return (
          mod_name == pkg
          or mod_name.startswith(f"{pkg}.")
          or mod_name.endswith(f".{pkg}")
          or f".{pkg}." in mod_name
      )

    expected_ancestors = ("tunix", "tunix.dist_rl", "tunix.dist_rl.train")
    ancestor_paths = {
        pkg: list(sys.modules[pkg].__path__)
        for pkg in expected_ancestors
        if pkg in sys.modules and hasattr(sys.modules[pkg], "__path__")
    }
    self.assertCountEqual(ancestor_paths.keys(), expected_ancestors)
    with mock.patch.dict(sys.modules):
      modules = cast(dict[str, Any], sys.modules)
      for name in list(modules):
        if any(_matches_pkg(name, pkg) for pkg in forbidden):
          modules[name] = None
        elif _matches_pkg(name, "tunix"):
          modules.pop(name, None)
      for pkg in forbidden:
        modules[pkg] = None
      for pkg, pkg_path in ancestor_paths.items():
        spec = importlib.machinery.ModuleSpec(pkg, loader=None, is_package=True)
        spec.submodule_search_locations = pkg_path
        stub = types.ModuleType(pkg)
        stub.__path__ = pkg_path
        stub.__package__ = pkg
        stub.__spec__ = spec
        modules[pkg] = stub
        parent_name, _, child_name = pkg.rpartition(".")
        if parent_name in modules:
          setattr(modules[parent_name], child_name, stub)
      mod = importlib.import_module("tunix.dist_rl.train.abstract_trainer")
      self.assertTrue(hasattr(mod, "AbstractTrainer"))


if __name__ == "__main__":
  absltest.main()
