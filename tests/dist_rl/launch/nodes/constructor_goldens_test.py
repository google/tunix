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

"""Golden tests of what each node's `main` passes to its constructors.

For each node and named config (`GOLDEN_CASES_BY_ROLE`: overrides, plus the
environment its deployment sets), the test runs the node's `main` with its
external constructors (models, samplers, workers, the cluster, the RL program),
the model and tokenizer downloads, the device mesh, and the gRPC server
replaced by recorders. Every recorded call, in order, the environment variables
`main` sets, and the JAX config it changes are compared with
`test_data/constructor_goldens/<role>/<case>.json`, so any change to what a node
builds shows up as a golden diff in review. The nodes' code runs unchanged,
including the tokenizer loading and the mesh checks; each node sees the devices
its mesh needs.

The config dataclasses the nodes build (`_RECORDED_INITS`) are recorded where
they are constructed, with the arguments the caller passed, so wiring that
happens to match a default still shows. Values are written as JSON:

* an instance of a recorded config dataclass as the name of its construction,
  `<module.Class()>` (`#2`, `#3`, ... for later ones);
* other dataclasses as their type and the init fields that differ from the
  defaults;
* enums as `Type.NAME`, functions by their qualified name, sets as sorted
  lists;
* mocks a recorder returned, and their attributes, as `<label()...>`;
* the resolved config as `<hp>`, the tokenizer as `<tokenizer>`, and large
  library values and the config fingerprint (`_known_values`) by name.

Any other type fails the test.

After an intended change, regenerate the goldens with the command in
`_REGENERATE`, which every failure prints, and review the diff.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import difflib
import enum
import importlib
import inspect
import json
import math
import os
import pathlib
import re
import sys
import types
from typing import TypeAlias, cast
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import grain
import jax
from jax.experimental import mesh_utils
from orbax import checkpoint as ocp
import transformers
from tunix.cli import config as cli_config
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_utils
from tunix.dist_rl.common import datatypes as dist_datatypes
from tunix.dist_rl.common import gcs_cache
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import inference
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import orchestrator
from tunix.dist_rl.launch.nodes import rollout
from tunix.dist_rl.launch.nodes import trainer
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import orchestrator as cluster_orchestrator
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.rollout import inprocess_vllm_sampler
from tunix.dist_rl.rollout import vanilla_sampler
from tunix.dist_rl.rollout import vllm_sampler
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import inference_worker as exp_inference_worker
from tunix.dist_rl.worker import rollout_worker
from tunix.dist_rl.worker import trainer_worker
from tunix.generate import mappings
from tunix.generate import tokenizer_adapter
from tunix.models import automodel
from tunix.models.gemma4 import mapping_vllm_jax as gemma4_mapping
from tunix.models.qwen3 import mapping_vllm_jax as qwen3_mapping
from tunix.rl import algorithm_config
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.rl.inference import inference_worker as rl_inference_worker
from tunix.sft import checkpoint_options
from tunix.sft import metrics_logger
from tunix.utils import maxtext_utils

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)

_UPDATE_GOLDENS_DIR = flags.DEFINE_string(
    "update_goldens_dir",
    None,
    "Writes the goldens to this directory instead of comparing with them.",
)

# `test_data` (not `testdata`): the OSS export moves `**/test_data/**` with the
# tests.
_GOLDEN_DIR = (
    pathlib.Path(__file__).parent / "test_data" / "constructor_goldens"
)

_REGENERATE = (
    "If the change is intended, regenerate the goldens with\n"
    "  python tests/dist_rl/launch/nodes/constructor_goldens_test.py"
    " --update_goldens_dir=$PWD/tests/dist_rl/launch/nodes/test_data/"
    "constructor_goldens\n"
    "and review the diff."
)

# Overrides every golden case starts from: values that would otherwise come
# from the clock (`run_id`), the user (the k8s ids), the checkout (a relative
# `artifact_root` or an unset `checkpoint_root_directory`), or the Hugging Face
# hub (a hub tokenizer needs `HF_TOKEN`). The orchestrator's `data_module` is
# recorded, not imported.
FIXED_OVERRIDES = (
    "cluster_config.run_id=golden",
    "cluster_config.artifact_root=/artifacts",
    "rl_training_config.checkpoint_root_directory=/artifacts/checkpoints",
    "cluster_config.k8s.orchestrator_id=golden-orchestrator",
    "cluster_config.k8s.trainer_id=golden-trainer",
    "cluster_config.k8s.rollout_id=golden-rollout",
    "cluster_config.k8s.inference_id=golden-inference",
    "tokenizer_config.tokenizer_path=/models/tokenizer",
    "data_module=golden_data_module",
)


@dataclasses.dataclass(frozen=True)
class GoldenCase:
  """A named config the goldens cover.

  Attributes:
    overrides: Overrides on `base_dist_config.yaml`, after `FIXED_OVERRIDES`.
    env: Environment variables the deployment sets for the nodes, e.g. in a k8s
      manifest. The node sees no others but `_KEPT_ENV_VARS`.
  """

  overrides: tuple[str, ...] = ()
  env: Mapping[str, str] = dataclasses.field(default_factory=dict)


# Two rollout replicas on k8s, without a KL penalty, with Raiden weight sync, a
# trajectory store, sequence packing, and off-policy steps.
_K8S_OVERRIDES = (
    "cluster_config.target=k8s",
    "cluster_config.k8s.rollout.replicas=2",
    "cluster_config.init_timeout_s=900",
    "cluster_config.rpc_timeout_s=3600",
    "rl_training_config.checkpoint_root_directory=",
    "agentic_grpo_config.beta=0.0",
    "agentic_grpo_config.off_policy_steps=1",
    "agentic_grpo_config.max_version_span=2",
    "weight_sync_config.mode=raiden",
    "trajectory_store_config.enable=true",
    "rl_training_config.max_seq_token_per_tpu=4096",
    "actor_model_config.mesh.shape=(2,2)",
)
# `k8s.sh` gives each rollout replica its index; the goldens run the second.
# The other nodes ignore it.
_K8S_ENV = {node_utils.ROLLOUT_REPLICA_ENV: "1"}

GOLDEN_CASES: Mapping[str, GoldenCase] = {
    # `base_dist_config.yaml`: Qwen3-1.7B, `inprocess_vllm`, a KL penalty.
    "base": GoldenCase(),
    "k8s": GoldenCase(_K8S_OVERRIDES, env=_K8S_ENV),
    # The same, with the rollout as the default k8s rollout manifest
    # (`leaderworkerset.mcjax.ray.yaml`) runs it: a multi-host Ray job, so the
    # in-process vLLM engine builds no JAX mesh and runs on Ray.
    "k8s_multihost": GoldenCase(
        _K8S_OVERRIDES, env={**_K8S_ENV, "TPU_MULTIHOST_BACKEND": "ray"}
    ),
    # LoRA on both nodes (`model_config.lora_config`): a QLoRA Tunix trainer
    # with flash attention and checkpoint saving disabled, and a LoRA vLLM
    # rollout with 4-way TP.
    "lora": GoldenCase((
        "actor_model_config.dtype=bfloat16",
        "actor_model_config.remat_config=DECODER",
        "actor_model_config.use_flash_attention=true",
        "actor_model_config.flash_attention_block_size=256",
        (
            "model_config.lora_config={'rank': 16, 'alpha': 32.0,"
            " 'weight_qtype': 'nf4', 'tile_size': 128}"
        ),
        "rl_training_config.checkpointing_options.save_interval_steps=0",
        "rollout_engine=vllm",
        "rollout_model_config.mesh.shape=(1,4)",
    )),
    # MaxText trainer and rollout with profiling and custom MaxText knobs.
    "maxtext": GoldenCase((
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "agentic_grpo_config.beta=0.0",
        "actor_model_config.kwargs.warmup_steps_fraction=0.05",
        "actor_model_config.kwargs.attention=flash",
        "rl_training_config.profiler_options.profiler_steps=3",
        "rl_training_config.profiler_options.skip_first_n_steps=5",
        "dist_training_config.profiler_period=10",
    )),
    # Native JAX `vanilla` rollout with the auto chat template, debug logging,
    # and custom environment/agent kwargs.
    "vanilla": GoldenCase((
        "rollout_engine=vanilla",
        "rollout_model_config.dtype=bfloat16",
        "chat_parser_config.type=auto",
        "chat_parser_config.enable_thinking=true",
        "cluster_config.debug=true",
        "env_class_path=frozenlake_env",
        "agent_class_path=frozenlake_agent",
        "env_kwargs={'map_size': 4, 'slippery': False}",
        "agent_kwargs.use_multistep_prompt=true",
    )),
    # Gemma4 `inprocess_vllm` rollout with LoRA, partial rollout (forcing
    # prefix caching), Gemma4 hf_overrides/sampling_kwargs, and auto parser.
    "gemma4": GoldenCase((
        "rollout_model_config.model_id=google/gemma-4-E2B-it",
        "rollout_model_config.model_name=gemma4-e2b",
        "rollout_model_config.lora_config={'rank': 8, 'alpha': 16.0}",
        "weight_sync_config.in_flight_weight_updates=true",
        "vllm_config.hbm_utilization=0.75",
        "vllm_config.max_num_seqs=32",
        "vllm_config.max_num_batched_tokens=2048",
        "chat_parser_config.type=auto",
    )),
    # MaxText `vllm` (`AsyncEngineArgs`) rollout with LoRA, routed experts,
    # detokenize, and deep-merged `vllm_config.engine_kwargs`.
    "vllm_engine_kwargs": GoldenCase((
        "rollout_engine=vllm",
        "model_config.model_source=maxtext",
        "model_config.model_path=/ckpt/qwen3-1.7b/0/items",
        "model_config.kwargs.maxtext_model_name=qwen3-1.7b",
        "rollout_model_config.kwargs.attention=vllm_rpa",
        "rollout_model_config.kwargs.float32_gate_logits=true",
        "rollout_model_config.lora_config={'rank': 8, 'alpha': 16.0}",
        "agentic_grpo_config.beta=0.0",
        "vllm_config.hbm_utilization=0.85",
        "vllm_config.return_routed_experts=true",
        "vllm_config.routed_experts_encode_mode=uint10",
        "vllm_config.routed_experts_num_experts=128",
        "vllm_config.detokenize=true",
        "vllm_config.max_model_len=8192",
        (
            "vllm_config.engine_kwargs={'enable_expert_parallel': True,"
            " 'block_size': 256, 'additional_config': {'sharding':"
            " {'sharding_strategy': {'expert_parallelism': 4}},"
            " 'maxtext_config': {'use_multimodal': False,"
            " 'allow_split_physical_axes': False}}}"
        ),
    )),
}

# The cases each role's goldens cover.
GOLDEN_CASES_BY_ROLE: Mapping[node_utils.ProcessRole, tuple[str, ...]] = {
    "orchestrator": ("base", "k8s"),
    "inference": ("base",),
    "trainer": ("base", "k8s", "lora", "maxtext"),
    "rollout": (
        "base",
        "k8s",
        "k8s_multihost",
        "lora",
        "maxtext",
        "vanilla",
        "gemma4",
        "vllm_engine_kwargs",
    ),
}

# Environment variables kept while a node's `main` runs. It sees no others, so
# the goldens do not depend on the caller's environment, and every variable it
# sets is recorded.
_KEPT_ENV_VARS = (
    "HF_TOKEN",
    "HOME",
    "PATH",
    "PYTHONPATH",
    "RUNFILES_DIR",
    "RUNFILES_MANIFEST_FILE",
    "TEST_SRCDIR",
    "TEST_TMPDIR",
    "TMPDIR",
)

# The task record the orchestrator's `data_module` returns.
_RECORD = {"prompt": "golden prompt", "metadata": {"answer": "42"}}

# Config dataclasses the nodes build, recorded where they are constructed.
_RECORDED_INITS: tuple[type[object], ...] = (
    algorithm_config.GRPOConfig,
    batch_assembly.BatchConfig,
    checkpoint_options.TunixCheckpointingOptions,
    dist_datatypes.GenerationArgs,
    gcs_cache.RolloutCacheKeyParams,
    maxtext_utils.AdamWOptions,
    maxtext_utils.ProfilerOptions,
    metrics_logger.MetricsLoggerOptions,
    ocp.CheckpointManagerOptions,
    peft_trainer_v2.TrainingConfig,
    rollout_worker.RolloutConfig,
    weight_sync_coordinator.PhaseTimeouts,
)

# The class the nodes load their tokenizer through. `transformers` imports it
# lazily, and that import sets environment variables, so it is looked up here
# rather than inside the first golden's run.
_AUTO_TOKENIZER: type[object] = transformers.AutoTokenizer

# The model section whose mesh sets each worker node's device count: the
# launchers give a node the devices its mesh needs.
_MESH_KEYS: Mapping[node_utils.ProcessRole, str] = {
    "inference": "reference_model_config",
    "trainer": "actor_model_config",
    "rollout": "rollout_model_config",
}

# `returns` default of `_Recorder.record`: a mock named after the call.
_NAMED_MOCK = object()

_MOCK_REPR = re.compile(r"<\w+ name='(?P<name>[^']+)'")


class _FakeTokenizer:
  """The tokenizer the nodes load: Qwen3's pad and eos ids, nothing else."""

  pad_token_id = 151643
  eos_token_id = 151645


def _named_mock(name: str) -> mock.MagicMock:
  return mock.MagicMock(name=name)


def _label(target: types.ModuleType | type[object], attribute: str) -> str:
  """Returns the qualified name of `target.attribute`, the call's label."""
  if isinstance(target, types.ModuleType):
    return f"{target.__name__}.{attribute}"
  return f"{target.__module__}.{target.__qualname__}.{attribute}"


class _Recorder:
  """Records the calls a node's `main` makes to the patched callables.

  Attributes:
    stack: Where the patches are entered; they are undone when it closes.
    calls: `(label, args, kwargs)` of every recorded call, in order.
    instances: Each instance a recorded constructor built, and its golden name.
  """

  def __init__(self, stack: contextlib.ExitStack) -> None:
    self.stack = stack
    self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
    self.instances: list[tuple[object, str]] = []

  def record(
      self,
      label: str,
      *,
      returns: object = _NAMED_MOCK,
      forward_to: Callable[..., object] | None = None,
  ) -> Callable[..., object]:
    """Returns a function that records each of its calls as `label`.

    Args:
      label: Name of the call in the golden.
      returns: What the function returns; by default a new mock named
        `<label>()` for each call.
      forward_to: Called with the arguments instead, returning its result.

    Returns:
      The recording function.
    """

    def recorded(*args: object, **kwargs: object) -> object:
      self.calls.append((label, args, kwargs))
      if forward_to is not None:
        return forward_to(*args, **kwargs)
      if returns is _NAMED_MOCK:
        return _named_mock(f"{label}()")
      return returns

    return recorded

  def record_init(self, cls: type[object]) -> None:
    """Records each construction of `cls` with the arguments its caller passed.

    The real `__init__` still runs, so the caller gets a real instance (and
    `isinstance` checks still hold). Calls that receive the instance later
    write it by name (`instances`).

    Args:
      cls: The class to record.
    """
    label = f"{cls.__module__}.{cls.__qualname__}"
    init = cls.__init__

    def recorded_init(
        instance: object, *args: object, **kwargs: object
    ) -> None:
      self.calls.append((label, args, kwargs))
      built = sum(name.startswith(f"<{label}()") for _, name in self.instances)
      suffix = f"#{built + 1}" if built else ""
      self.instances.append((instance, f"<{label}(){suffix}>"))
      init(instance, *args, **kwargs)

    self.stack.enter_context(mock.patch.object(cls, "__init__", recorded_init))

  def patch(
      self,
      target: types.ModuleType | type[object],
      attribute: str,
      *,
      returns: object = _NAMED_MOCK,
      forward_to: Callable[..., object] | None = None,
      is_async: bool = False,
  ) -> None:
    """Replaces `target.attribute` with a `record` function."""
    recorded = self.record(
        _label(target, attribute), returns=returns, forward_to=forward_to
    )
    if is_async:

      async def recorded_async(*args: object, **kwargs: object) -> object:
        return recorded(*args, **kwargs)

      self.stack.enter_context(
          mock.patch.object(target, attribute, recorded_async)
      )
    else:
      self.stack.enter_context(mock.patch.object(target, attribute, recorded))


def _local_model_dir(model_dir: str, model_id: str, *, model_key: str) -> str:
  """`ensure_local_safetensors` with the weights already in `model_dir`."""
  del model_id, model_key
  return model_dir


def _patch_model_loading(recorder: _Recorder) -> None:
  """Records the worker nodes' model download and model loading."""
  recorder.patch(
      node_utils, "ensure_local_safetensors", forward_to=_local_model_dir
  )
  from_pretrained = _label(automodel.AutoModel, "from_pretrained")
  recorder.patch(
      automodel.AutoModel,
      "from_pretrained",
      returns=(
          _named_mock(f"{from_pretrained}()[0]"),
          _named_mock(f"{from_pretrained}()[1]"),
      ),
  )


def _run_orchestrator(
    recorder: _Recorder, argv: list[str], context: mock.MagicMock
) -> None:
  """Runs the orchestrator's `main` up to the end of the RL program."""
  cluster_label = _label(cluster_orchestrator, "ClusterOrchestrator")
  cluster = _named_mock(f"{cluster_label}()")
  cluster.registry.wait_for_workers = recorder.record(
      f"{cluster_label}().registry.wait_for_workers"
  )

  def fake_cluster(**kwargs: object) -> mock.MagicMock:
    # Like `ClusterOrchestrator`, no trajectory store without its config.
    if kwargs["trajectory_store_config"] is None:
      cluster.trajectory_store = None
    return cluster

  recorder.patch(
      cluster_orchestrator, "ClusterOrchestrator", forward_to=fake_cluster
  )
  program = _named_mock(f"{_label(rl_program, 'StandardRLProgram')}()")
  program.last_step_result = None
  recorder.patch(rl_program, "StandardRLProgram", returns=program)
  recorder.patch(
      data_lib,
      "get_dataset_from_module",
      returns=grain.MapDataset.source([_RECORD]),
  )
  recorder.patch(
      orchestrator, "run_program", forward_to=orchestrator.run_program
  )
  orchestrator.main(argv, context)


def _run_inference(
    recorder: _Recorder, argv: list[str], context: mock.MagicMock
) -> None:
  """Runs the reference inference node's `main` up to serving."""
  _patch_model_loading(recorder)
  recorder.patch(rl_inference_worker, "InferenceWorker")
  recorder.patch(exp_inference_worker, "InferenceWorker")
  inference.main(argv, context)


def _run_trainer(
    recorder: _Recorder, argv: list[str], context: mock.MagicMock
) -> None:
  """Runs the trainer's `main` up to serving, then its trainer factory."""
  _patch_model_loading(recorder)
  recorder.patch(cli_config, "create_optimizer")
  recorder.patch(model_utils, "apply_lora_to_model")
  recorder.patch(peft_trainer_v2, "PeftTrainer")
  recorder.patch(maxtext_utils, "build_maxtext_config")
  recorder.patch(maxtext_utils, "create_maxtext_mesh")
  recorder.patch(maxtext_utils, "create_maxtext_engine")
  recorder.patch(trainer_worker, "TrainerWorker")
  trainer.main(argv, context)
  # `TrainerWorker` builds its trainer when the orchestrator brings it up.
  (factory,) = [
      kwargs["trainer_factory"]
      for label, _, kwargs in recorder.calls
      if label == _label(trainer_worker, "TrainerWorker")
  ]
  recorder.record(
      "trainer_factory", forward_to=cast(Callable[[], object], factory)
  )()


def _run_rollout(
    recorder: _Recorder, argv: list[str], context: mock.MagicMock
) -> None:
  """Runs the rollout node's `main` up to serving."""
  _patch_model_loading(recorder)
  recorder.patch(importlib, "import_module")
  recorder.patch(gcs_cache, "restore_jax_cache")
  recorder.patch(
      rollout,
      "_import_vllm_sampler",
      returns=types.SimpleNamespace(
          VllmConfig=recorder.record("tunix.generate.vllm_sampler.VllmConfig")
      ),
  )
  arg_utils = types.ModuleType("vllm.engine.arg_utils")
  setattr(
      arg_utils,
      "AsyncEngineArgs",
      recorder.record("vllm.engine.arg_utils.AsyncEngineArgs"),
  )
  recorder.stack.enter_context(
      mock.patch.dict(sys.modules, {"vllm.engine.arg_utils": arg_utils})
  )
  recorder.patch(vanilla_sampler, "VanillaSampler")
  recorder.patch(inprocess_vllm_sampler, "InprocessVllmSampler")
  recorder.patch(vllm_sampler, "VllmSampler")
  recorder.patch(tokenizer_adapter, "TokenizerAdapter")
  recorder.stack.enter_context(
      mock.patch.dict(
          rollout.CHAT_PARSERS,
          {
              family: recorder.record(f"{cls.__module__}.{cls.__qualname__}")
              for family, cls in rollout.CHAT_PARSERS.items()
          },
      )
  )
  recorder.patch(chat_parser_lib, "RawTextParser")
  recorder.patch(chat_parser_lib, "DefaultChatTemplateParser")
  worker_label = _label(rollout_worker, "RolloutWorker")

  def fake_rollout_worker(**kwargs: object) -> mock.MagicMock:
    worker = _named_mock(f"{worker_label}()")
    # Like `RolloutWorker`, no trajectory store without its config.
    config = kwargs["config"]
    if not isinstance(config, rollout_worker.RolloutConfig):
      raise TypeError(f"Expected a RolloutConfig, got {config!r}.")
    if config.trajectory_store_config is None:
      worker.trajectory_store = None
    return worker

  recorder.patch(
      rollout_worker, "RolloutWorker", forward_to=fake_rollout_worker
  )
  rollout.main(argv, context)


_RUNS: Mapping[
    node_utils.ProcessRole,
    Callable[[_Recorder, list[str], mock.MagicMock], None],
] = {
    "orchestrator": _run_orchestrator,
    "inference": _run_inference,
    "trainer": _run_trainer,
    "rollout": _run_rollout,
}


def _known_values(
    hp: resolve_config.DistHyperParameters,
) -> list[tuple[object, str]]:
  """Returns values written by name: the value and its golden text."""
  # Large values, and the config fingerprint, which any change to a config
  # default would change.
  known: list[tuple[object, str]] = [
      (resolve_config.plain_config(hp), "<resolve_config.plain_config(hp)>"),
      (
          resolve_config.config_fingerprint(hp),
          "<resolve_config.config_fingerprint(hp)>",
      ),
  ]
  for mapping_module in (qwen3_mapping, gemma4_mapping):
    known.append((
        mappings.MappingConfig(**mapping_module.VLLM_JAX_MAPPING),
        f"<MappingConfig(**{mapping_module.__name__}.VLLM_JAX_MAPPING)>",
    ))
  return known


def _mock_name(value: mock.NonCallableMock) -> str:
  match = _MOCK_REPR.match(repr(value))
  if match is None:
    raise TypeError(f"An unnamed mock reached a recorded call: {value!r}.")
  return match.group("name")


def _render_key(key: object) -> str:
  if isinstance(key, enum.Enum):
    return f"{type(key).__qualname__}.{key.name}"
  if isinstance(key, str):
    return key
  raise TypeError(f"Cannot write mapping key {key!r} to a golden.")


@dataclasses.dataclass(frozen=True)
class _Names:
  """Values a golden writes by name instead of by content.

  Attributes:
    instances: Instances a recorded constructor built (`_Recorder.instances`),
      matched by identity.
    known: Large values (`_known_values`), matched by type and equality.
  """

  instances: Sequence[tuple[object, str]]
  known: Sequence[tuple[object, str]]


def _render_dataclass(
    value: object,
    fields: tuple[dataclasses.Field[object], ...],
    known: _Names,
) -> dict[str, JsonValue]:
  """Writes a dataclass as its type and its init fields that are not default."""
  rendered: dict[str, JsonValue] = {"__type__": type(value).__qualname__}
  for field in fields:
    if not field.init:
      continue
    current = getattr(value, field.name)
    if field.default is not dataclasses.MISSING:
      default = field.default
    elif field.default_factory is not dataclasses.MISSING:
      default = field.default_factory()
    else:
      rendered[field.name] = _render(current, known)
      continue
    if type(current) is not type(default) or current != default:
      rendered[field.name] = _render(current, known)
  return rendered


def _render(value: object, known: _Names) -> JsonValue:
  """Writes `value` as JSON for a golden; fails on an unsupported type."""
  for instance, name in known.instances:
    if value is instance:
      return name
  for known_value, name in known.known:
    if type(value) is type(known_value) and value == known_value:
      return name
  if isinstance(value, enum.Enum):
    return f"{type(value).__qualname__}.{value.name}"
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, resolve_config.DistHyperParameters):
    return "<hp>"
  if isinstance(value, _FakeTokenizer):
    return "<tokenizer>"
  if isinstance(value, mock.NonCallableMock):
    return f"<{_mock_name(value)}>"
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return _render_dataclass(value, dataclasses.fields(value), known)
  if isinstance(value, grain.MapDataset):
    return {"__type__": "grain.MapDataset", "[0]": _render(value[0], known)}
  if isinstance(value, Mapping):
    return {
        _render_key(key): _render(item, known) for key, item in value.items()
    }
  if isinstance(value, (set, frozenset)):
    return sorted(
        (_render(item, known) for item in value),
        key=lambda item: json.dumps(item, sort_keys=True),
    )
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    return [_render(item, known) for item in value]
  if inspect.isfunction(value) or inspect.ismethod(value):
    return f"<function {value.__module__}.{value.__qualname__}>"
  raise TypeError(
      f"Cannot write a {type(value).__qualname__} to a golden: {value!r}. Add"
      " a rule to `_render`."
  )


def _capture(role: node_utils.ProcessRole, case: str) -> str:
  """Runs `role`'s `main` on `case` and returns its golden file contents."""
  golden_case = GOLDEN_CASES[case]
  argv = [*FIXED_OVERRIDES, *golden_case.overrides]
  with contextlib.ExitStack() as stack:
    start_env = {
        name: os.environ[name] for name in _KEPT_ENV_VARS if name in os.environ
    }
    start_env.update(golden_case.env)
    stack.enter_context(mock.patch.dict(os.environ, start_env, clear=True))
    # `main` may extend `sys.path` and pin `jax_platforms`; undo both, and start
    # from an unset platform, as a new process does.
    stack.enter_context(mock.patch.object(sys, "path", list(sys.path)))
    stack.callback(jax.config.update, "jax_platforms", jax.config.jax_platforms)
    jax.config.update("jax_platforms", None)
    hp = resolve_config.initialize(argv)
    known = _known_values(hp)
    recorder = _Recorder(stack)
    for cls in _RECORDED_INITS:
      recorder.record_init(cls)
    context = _named_mock("context")
    context.jax.initialize = recorder.record("context.jax.initialize")
    recorder.patch(_AUTO_TOKENIZER, "from_pretrained", returns=_FakeTokenizer())
    if role != "orchestrator":
      axes = hp.mesh_axes(_MESH_KEYS[role])
      devices = [
          _named_mock(f"jax.devices()[{index}]")
          for index in range(math.prod(axes.values()))
      ]
      stack.enter_context(
          mock.patch.object(jax, "device_count", return_value=len(devices))
      )
      stack.enter_context(
          mock.patch.object(jax, "devices", return_value=devices)
      )
      recorder.patch(mesh_utils, "create_device_mesh")
      recorder.patch(node_utils, "Mesh")
    recorder.patch(node_utils, "serve_worker", is_async=True)
    _RUNS[role](recorder, argv, context)
    env_set = {
        name: value
        for name, value in os.environ.items()
        if name not in start_env or start_env[name] != value
    }
    jax_config: dict[str, JsonValue] = {}
    if jax.config.jax_platforms is not None:
      jax_config["jax_platforms"] = jax.config.jax_platforms
    names = _Names(instances=recorder.instances, known=known)

  calls: list[JsonValue] = []
  for label, args, kwargs in recorder.calls:
    call: dict[str, JsonValue] = {}
    if args:
      call["args"] = _render(args, names)
    if kwargs:
      call["kwargs"] = _render(kwargs, names)
    calls.append({label: call})
  golden: dict[str, JsonValue] = {
      "calls": calls,
      "env": _render(env_set, names),
      "jax_config": jax_config,
  }
  text = json.dumps(golden, indent=2, sort_keys=True, allow_nan=False) + "\n"
  local_paths = [".runfiles"]
  if "TEST_TMPDIR" in os.environ:
    local_paths.append(os.environ["TEST_TMPDIR"])
  for local_path in local_paths:
    if local_path in text:
      raise AssertionError(
          f"The {role} golden for {case!r} holds the local path {local_path!r};"
          " fix it in FIXED_OVERRIDES."
      )
  return text


def _golden_params() -> list[tuple[str, node_utils.ProcessRole, str]]:
  """Returns `(test name suffix, role, case)` for every golden."""
  params: list[tuple[str, node_utils.ProcessRole, str]] = []
  for role, cases in GOLDEN_CASES_BY_ROLE.items():
    for case in cases:
      params.append((f"{role}_{case}", role, case))
  return params


class ConstructorGoldensTest(parameterized.TestCase):

  @parameterized.named_parameters(*_golden_params())
  def test_matches_golden(self, role: node_utils.ProcessRole, case: str):
    actual = _capture(role, case)
    relative = pathlib.Path(role, f"{case}.json")
    if _UPDATE_GOLDENS_DIR.value:
      out = pathlib.Path(_UPDATE_GOLDENS_DIR.value) / relative
      out.parent.mkdir(parents=True, exist_ok=True)
      out.write_text(actual)
      return
    golden = _GOLDEN_DIR / relative
    if not golden.is_file():
      self.fail(f"Missing golden {relative}. {_REGENERATE}")
    expected = golden.read_text()
    if actual == expected:
      return
    if "TEST_UNDECLARED_OUTPUTS_DIR" in os.environ:
      out = (
          pathlib.Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
          / "constructor_goldens"
          / relative
      )
      out.parent.mkdir(parents=True, exist_ok=True)
      out.write_text(actual)
    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            actual.splitlines(keepends=True),
            fromfile=f"golden {relative}",
            tofile="actual",
        )
    )
    self.fail(
        f"What the {role} builds for golden case {case!r} changed:\n{diff}\n"
        f"{_REGENERATE}"
    )


if __name__ == "__main__":
  absltest.main()
