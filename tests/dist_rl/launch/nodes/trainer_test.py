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

"""Unit tests for the distributed RL trainer worker node."""

import contextlib
import importlib
from unittest import mock

from absl.testing import absltest
from jax.sharding import Mesh
from orbax.checkpoint import v1 as ocp_v1
from tunix.cli import config as cli_config
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import trainer
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.sft import checkpoint_options
from tunix.utils import maxtext_utils

_OPT = "rl_training_config.actor_optimizer_config"
_MAXTEXT_ARGV = ("actor_model_config.kwargs.maxtext_model_name=qwen3-0.6b",)


def _hp(*argv: str) -> resolve_config.DistHyperParameters:
  return resolve_config.initialize(list(argv))


class TrainerMainTest(absltest.TestCase):
  """Covers `trainer.main` wiring; `node_utils.serve_worker` has its own tests."""

  def setUp(self):
    super().setUp()
    self.mock_context = mock.MagicMock()
    self.mock_worker_service = mock.MagicMock(spec=trainer_worker.TrainerWorker)
    self.mock_mesh = mock.MagicMock(spec=Mesh)
    self.mock_create_trainer_factory = self.enter_context(
        mock.patch.object(
            trainer,
            "_create_trainer_factory",
            return_value=(mock.MagicMock(), self.mock_mesh),
        )
    )
    self.mock_trainer_worker_cls = self.enter_context(
        mock.patch.object(
            trainer_worker,
            "TrainerWorker",
            return_value=self.mock_worker_service,
        )
    )
    self.mock_serve_worker = self.enter_context(
        mock.patch.object(
            node_utils, "serve_worker", new_callable=mock.AsyncMock
        )
    )

  def test_main_builds_worker_and_serves_with_drain(self):
    trainer.main(
        ["cluster_config.ports.trainer=20050"], context=self.mock_context
    )

    self.mock_context.jax.initialize.assert_called_once()
    self.mock_create_trainer_factory.assert_called_once()
    worker_kwargs = self.mock_trainer_worker_cls.call_args.kwargs
    self.assertEqual(worker_kwargs["worker_id"], trainer.LOCAL_WORKER_ID)
    self.assertIs(worker_kwargs["execution_context"], self.mock_mesh)

    self.mock_serve_worker.assert_awaited_once()
    serve_kwargs = self.mock_serve_worker.call_args.kwargs
    self.assertIs(
        self.mock_serve_worker.call_args.args[0], self.mock_worker_service
    )
    self.assertIs(serve_kwargs["process_context"], self.mock_context)
    self.assertEqual(serve_kwargs["role"], "trainer")
    self.assertEqual(serve_kwargs["worker_id_str"], trainer.LOCAL_WORKER_ID)
    self.assertEqual(serve_kwargs["port"], 20050)
    # The drain callback stops the worker and discards its `Response`.
    self.assertIsNone(serve_kwargs["drain_fn"]())
    self.mock_worker_service.stop.assert_called_once_with()

  def test_k8s_target_registers_with_k8s_trainer_id(self):
    trainer.main(
        [
            "cluster_config.target=k8s",
            "cluster_config.k8s.trainer_id=trainer-1",
        ],
        context=self.mock_context,
    )
    self.assertEqual(
        self.mock_trainer_worker_cls.call_args.kwargs["worker_id"], "trainer-1"
    )
    self.assertEqual(
        self.mock_serve_worker.call_args.kwargs["worker_id_str"], "trainer-1"
    )

  def test_main_raises_without_discovery_context(self):
    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      trainer.main([], context=None)

    with self.assertRaisesRegex(RuntimeError, "Require discovery API"):
      trainer.main([], context=mock.MagicMock(ipc=None))

  def test_main_raises_on_non_positive_micro_batch_size(self):
    with self.assertRaisesRegex(
        ValueError, "train_micro_batch_size must be > 0"
    ):
      trainer.main(
          ["rl_training_config.train_micro_batch_size=0"],
          context=self.mock_context,
      )

  def test_maxtext_raiden_synchronizer_shim_importable_in_runfiles(self):
    # Guards the `raiden_synchronizer` dep of the trainer target.
    self.assertIs(
        importlib.import_module(
            "tunix.experimental.weight_sync.raiden_synchronizer"
        ),
        importlib.import_module(
            "tunix.dist_rl.weight_sync.raiden_synchronizer"
        ),
    )


class TrainerFactoryTest(absltest.TestCase):

  def test_model_loading_separates_compute_and_parameter_dtypes(self):
    for load_dtype, expected_load_dtype in (
        (None, "float32"),
        ("float32", "float32"),
        ("bfloat16", "bfloat16"),
    ):
      with self.subTest(load_dtype=load_dtype):
        argv = ["actor_model_config.dtype=bfloat16"]
        if load_dtype:
          argv.append(f"actor_model_config.load_dtype={load_dtype}")
        with mock.patch.object(
            trainer.automodel.AutoModel,
            "from_pretrained",
            return_value=(mock.sentinel.model, "/tmp/model"),
        ) as load:
          model = trainer._load_actor_model(
              _hp(*argv), "/tmp/model", mock.sentinel.mesh
          )
        self.assertIs(model, mock.sentinel.model)
        self.assertEqual(load.call_args.kwargs["dtype"], "bfloat16")
        self.assertEqual(
            load.call_args.kwargs["load_dtype"], expected_load_dtype
        )

  def test_model_loading_reads_actor_model_config(self):
    hp = _hp(
        "actor_model_config.model_name=Qwen3-32B",
        "actor_model_config.remat_config=DECODER",
        "actor_model_config.use_flash_attention=true",
        "actor_model_config.flash_attention_block_size=256",
        "actor_model_config.lora_config={'rank': 32, 'alpha': 64.0}",
    )
    with (
        mock.patch.object(
            trainer.automodel.AutoModel,
            "from_pretrained",
            return_value=(mock.sentinel.model, "/tmp/model"),
        ) as load,
        mock.patch.object(
            trainer.model_utils,
            "apply_lora_to_model",
            return_value=mock.sentinel.lora_model,
        ) as apply_lora,
    ):
      model = trainer._load_actor_model(hp, "/tmp/model", mock.sentinel.mesh)

    self.assertIs(model, mock.sentinel.lora_model)
    load.assert_called_once_with(
        model_id="Qwen3-32B",
        mesh=mock.sentinel.mesh,
        model_path="/tmp/model",
        dtype="float32",
        load_dtype="float32",
        remat_config="decoder",
        use_flash_attention=True,
        flash_attention_block_size=256,
    )
    lora_config = apply_lora.call_args.kwargs["lora_config"]
    self.assertEqual(lora_config["rank"], 32)
    self.assertEqual(lora_config["alpha"], 64.0)
    self.assertIn(".*q_proj", lora_config["module_path"])

  def test_lora_config_supports_module_path_weight_qtype_and_tile_size(self):
    lora_config = trainer._lora_config(
        _hp(
            "actor_model_config.lora_config={"
            "'rank': 16, 'alpha': 32.0, "
            "'module_path': '.*attn.*', 'weight_qtype': 'nf4', 'tile_size': 128"
            "}"
        ).config_mapping("actor_model_config")
    )
    self.assertEqual(
        lora_config,
        {
            "module_path": ".*attn.*",
            "rank": 16,
            "alpha": 32.0,
            "weight_qtype": "nf4",
            "tile_size": 128,
        },
    )

  def test_lora_config_requires_rank_and_alpha(self):
    with self.assertRaisesRegex(ValueError, "missing \\['alpha'\\]"):
      trainer._lora_config(
          _hp("actor_model_config.lora_config={'rank': 32}").config_mapping(
              "actor_model_config"
          )
      )
    self.assertIsNone(
        trainer._lora_config(_hp().config_mapping("actor_model_config"))
    )

  @mock.patch.object(trainer, "_create_tunix_trainer_factory")
  @mock.patch.object(trainer, "_create_maxtext_trainer_factory")
  def test_create_trainer_factory_delegates_by_model_source(
      self, mock_create_maxtext, mock_create_tunix
  ):
    hp_tunix = _hp()
    trainer._create_trainer_factory(hp_tunix, "trainer-0")
    mock_create_tunix.assert_called_once_with(hp_tunix)
    mock_create_maxtext.assert_not_called()

    mock_create_tunix.reset_mock()
    hp_maxtext = _hp(
        "actor_model_config.model_source=maxtext",
        "actor_model_config.model_path=/ckpt/items",
    )
    trainer._create_trainer_factory(hp_maxtext, "trainer-0")
    mock_create_maxtext.assert_called_once_with(hp_maxtext, "trainer-0")
    mock_create_tunix.assert_not_called()

  def _tunix_factory_patches(self) -> contextlib.ExitStack:
    stack = contextlib.ExitStack()
    stack.enter_context(
        mock.patch.object(
            node_utils, "ensure_local_safetensors", return_value="/models/test"
        )
    )
    self.mock_create_mesh = stack.enter_context(
        mock.patch.object(
            node_utils, "create_mesh", return_value=mock.MagicMock(spec=Mesh)
        )
    )
    self.mock_load_model = stack.enter_context(
        mock.patch.object(trainer, "_load_actor_model")
    )
    return stack

  def test_tunix_trainer_factory_builds_trainer_from_config(self):
    hp = _hp(
        "rollout_model_config.mesh.shape=(1,4)",
        "rollout_engine=vllm",
        "rl_training_config.eval_every_n_steps=7",
        "rl_training_config.mini_batch_size=2",
        "agentic_grpo_config.num_generations=8",
        "rl_training_config.train_micro_batch_size=4",
        "rl_training_config.checkpointing_options.save_interval_steps=5",
        "rl_training_config.checkpointing_options.max_to_keep=3",
        "rl_training_config.checkpoint_root_directory=/checkpoints/test",
    )
    with (
        self._tunix_factory_patches(),
        mock.patch.object(peft_trainer_v2, "PeftTrainer") as mock_peft_trainer,
    ):
      factory, mesh = trainer._create_tunix_trainer_factory(hp)
      self.assertIs(mesh, self.mock_create_mesh.return_value)
      self.mock_create_mesh.assert_called_once_with(
          hp, "actor_model_config", "Trainer"
      )
      self.mock_load_model.assert_not_called()

      factory()

    self.mock_load_model.assert_called_once_with(hp, "/models/test", mesh)
    _, optimizer, training_config = mock_peft_trainer.call_args.args
    self.assertIsNotNone(optimizer)
    # sampler_type and rollout_tp_size describe the weight-sync destination,
    # so they ride on the trainer's weight-sync surface, not TrainingConfig.
    self.assertEqual(mock_peft_trainer.call_args.kwargs["sampler_type"], "vllm")
    self.assertEqual(mock_peft_trainer.call_args.kwargs["rollout_tp_size"], 4)
    self.assertIsInstance(training_config, peft_trainer_v2.TrainingConfig)
    self.assertEqual(training_config.eval_every_n_steps, 7)
    self.assertEqual(training_config.gradient_accumulation_steps, 4)
    self.assertEqual(training_config.metrics_prefix, "actor")
    self.assertEqual(training_config.data_sharding_axis, ("fsdp",))
    self.assertEqual(
        training_config.checkpoint_root_directory, "/checkpoints/test"
    )
    self.assertFalse(training_config.resume_from_checkpoint_on_init)
    self.assertIsNone(training_config.max_seq_token_per_tpu)
    self.assertIsNone(training_config.metrics_logging_options)
    self.assertIsNone(training_config.profiler_options)
    # Checkpointing options come from the shared tunix.cli builder.
    ckpt = training_config.checkpointing_options
    self.assertIsInstance(ckpt, checkpoint_options.TunixCheckpointingOptions)
    save_policy = ckpt.save_decision_policy
    if not isinstance(
        save_policy, ocp_v1.training.save_decision_policies.FixedIntervalPolicy
    ):
      self.fail(f"Unexpected save_decision_policy {save_policy!r}.")
    self.assertEqual(save_policy.interval, 5)
    preservation_policy = ckpt.preservation_policy
    if not isinstance(
        preservation_policy, ocp_v1.training.preservation_policies.LatestN
    ):
      self.fail(f"Unexpected preservation_policy {preservation_policy!r}.")
    self.assertEqual(preservation_policy.n, 3)

  def test_tunix_trainer_factory_ignores_max_seq_token_per_tpu(self):
    # The field is for MaxText's max_target_length. Under the orchestrator the
    # PeftTrainer never packs -- the SequencePackedBatchAssembler does, and the
    # packing weights arrive on the payload -- so it stays off TrainingConfig.
    hp = _hp("rl_training_config.max_seq_token_per_tpu=4096")
    with (
        self._tunix_factory_patches(),
        mock.patch.object(
            peft_trainer_v2, "TrainingConfig"
        ) as mock_training_config,
    ):
      trainer._create_tunix_trainer_factory(hp)
    mock_training_config.assert_called_once()
    self.assertNotIn(
        "max_seq_token_per_tpu", mock_training_config.call_args.kwargs
    )
    self.assertEqual(
        mock_training_config.call_args.kwargs["gradient_accumulation_steps"], 1
    )

  def test_tunix_trainer_factory_disables_checkpoint_root_when_interval_zero(
      self,
  ):
    hp = _hp(
        "rl_training_config.checkpointing_options.save_interval_steps=0",
        "rl_training_config.checkpoint_root_directory=/checkpoints/test",
    )
    with (
        self._tunix_factory_patches(),
        mock.patch.object(peft_trainer_v2, "PeftTrainer") as mock_peft_trainer,
    ):
      factory, _ = trainer._create_tunix_trainer_factory(hp)
      factory()
    training_config = mock_peft_trainer.call_args.args[2]
    self.assertIsNone(training_config.checkpoint_root_directory)

  def test_checkpoint_root_directory_kept_when_saving_enabled(self):
    hp = _hp(
        "rl_training_config.checkpointing_options.save_interval_steps=5",
        "rl_training_config.checkpoint_root_directory=/checkpoints/test",
    )
    self.assertEqual(
        trainer._checkpoint_root_directory(hp), "/checkpoints/test"
    )

  def test_checkpoint_root_directory_dropped_when_saving_disabled(self):
    """Interval 0 with a root hits `step % 0` and force-saves on close."""
    hp = _hp(
        "rl_training_config.checkpointing_options.save_interval_steps=0",
        "rl_training_config.checkpoint_root_directory=/checkpoints/test",
    )
    self.assertIsNone(trainer._checkpoint_root_directory(hp))

  def test_save_interval_negative_raises(self):
    hp = _hp("rl_training_config.checkpointing_options.save_interval_steps=-1")
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      trainer._save_interval_steps(hp)

  def test_gradient_accumulation_uses_prompt_level_mini_batch(self):
    hp = _hp(
        "rl_training_config.mini_batch_size=2",
        "agentic_grpo_config.num_generations=8",
        "rl_training_config.train_micro_batch_size=4",
    )
    self.assertEqual(trainer._gradient_accumulation_steps(hp), 4)

  def test_gradient_accumulation_with_trajectory_micro_batch_above_mini(self):
    # e.g. trellis_gsm8k_qwen3p5_35b: 4 prompts * 2 generations / 8 = 1.
    hp = _hp(
        "rl_training_config.mini_batch_size=4",
        "agentic_grpo_config.num_generations=2",
        "rl_training_config.train_micro_batch_size=8",
    )
    self.assertEqual(trainer._gradient_accumulation_steps(hp), 1)

  def test_gradient_accumulation_with_sequence_packing(self):
    hp = _hp(
        "batch_size=5",
        "rl_training_config.mini_batch_size=5",
        "agentic_grpo_config.num_generations=3",
        "rl_training_config.train_micro_batch_size=4",
        "actor_model_config.mesh.shape=(4,1)",
        "rl_training_config.max_seq_token_per_tpu=4096",
    )
    self.assertEqual(trainer._gradient_accumulation_steps(hp), 1)

  def test_gradient_accumulation_requires_exact_divisibility(self):
    with self.assertRaisesRegex(ValueError, "must be divisible"):
      _hp(
          "rl_training_config.mini_batch_size=2",
          "agentic_grpo_config.num_generations=3",
          "rl_training_config.train_micro_batch_size=4",
      )


class TrainerOptimizerTest(absltest.TestCase):

  def _forwarded_optimizer_config(self, hp: resolve_config.DistHyperParameters):
    create_optimizer = cli_config.create_optimizer
    with mock.patch.object(
        cli_config, "create_optimizer", side_effect=create_optimizer
    ) as mock_create:
      trainer._create_optimizer(hp)
    return mock_create.call_args_list[0].args[0]

  def test_gsm8k_recipe_reaches_create_optimizer_untranslated(self):
    """The gsm8k defaults reproduce the qwen3_grpo_demo.py recipe."""
    forwarded = self._forwarded_optimizer_config(
        _hp(
            f"{_OPT}.learning_rate=2.0e-7",
            f"{_OPT}.warmup_steps=50",
            f"{_OPT}.decay_steps=500",
        )
    )
    for key, value in {
        "opt_type": "adamw",
        "learning_rate": 2.0e-7,
        "weight_decay": 0.01,
        "b1": 0.9,
        "b2": 0.999,
        "eps": 1.0e-8,
        "opt_chain_type": "clip_by_global_norm",
        "chain_kwargs": {"max_norm": 1.0},
        "schedule_type": "warmup_cosine_decay_schedule",
        "init_value": 0.0,
        # Resolved from learning_rate by DistHyperParameters.
        "peak_value": 2.0e-7,
        "end_value": 0.0,
        "warmup_steps": 50,
        "decay_steps": 500,
    }.items():
      self.assertEqual(forwarded[key], value, key)

  def test_frozenlake_recipe_reaches_create_optimizer_untranslated(self):
    forwarded = self._forwarded_optimizer_config(
        _hp(
            f"{_OPT}.learning_rate=1e-6",
            f"{_OPT}.b2=0.95",
            f"{_OPT}.weight_decay=0.0",
            f"{_OPT}.schedule_type=",
            f"{_OPT}.chain_kwargs.max_norm=100.0",
        )
    )
    self.assertEqual(forwarded["b2"], 0.95)
    self.assertEqual(forwarded["weight_decay"], 0.0)
    self.assertEqual(forwarded["schedule_type"], "")
    self.assertEqual(forwarded["opt_chain_type"], "clip_by_global_norm")
    self.assertEqual(forwarded["chain_kwargs"], {"max_norm": 100.0})


@mock.patch.object(
    node_utils,
    "load_hf_tokenizer",
    return_value=mock.Mock(pad_token_id=7, eos_token_id=9),
)
@mock.patch.object(maxtext_utils, "create_maxtext_mesh")
@mock.patch.object(maxtext_utils, "build_maxtext_config", autospec=True)
class MaxTextTrainerFactoryTest(absltest.TestCase):

  def _build_kwargs(self, mock_build_cfg, *argv: str):
    trainer._create_maxtext_trainer_factory(
        _hp(*_MAXTEXT_ARGV, *argv), "trainer-0"
    )
    mock_build_cfg.assert_called_once()
    return mock_build_cfg.call_args.kwargs

  def test_requires_maxtext_model_name(self, mock_build_cfg, *_):
    for argv in (
        (),
        ("actor_model_config.kwargs.maxtext_model_name=null",),
        ("actor_model_config.kwargs.maxtext_model_name=",),
    ):
      with self.subTest(argv=argv):
        with self.assertRaisesRegex(ValueError, "kwargs.maxtext_model_name"):
          trainer._create_maxtext_trainer_factory(_hp(*argv), "trainer-0")
    mock_build_cfg.assert_not_called()

  def test_plumbs_core_config(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg,
        "actor_model_config.mesh.shape=(1,2)",
        "rollout_model_config.mesh.shape=(1,4)",
        "rollout_config.max_prompt_length=128",
        "rollout_config.total_generation_steps=64",
        f"{_OPT}.learning_rate=3e-6",
        "rl_training_config.train_micro_batch_size=2",
    )
    self.assertEqual(kwargs["model_name"], "qwen3-0.6b")
    self.assertEqual(kwargs["worker_id"], "trainer-0")
    self.assertEqual(kwargs["mesh_fsdp"], 1)
    self.assertEqual(kwargs["mesh_tp"], 2)
    self.assertEqual(kwargs["rollout_mesh_tp"], 4)
    self.assertEqual(kwargs["max_prompt_length"], 128)
    self.assertEqual(kwargs["max_response_length"], 64)
    self.assertEqual(kwargs["learning_rate"], 3e-6)
    self.assertEqual(kwargs["train_micro_batch_size"], 2)
    # Default config: 4 prompts * 8 generations / 2 trajectories.
    self.assertEqual(kwargs["gradient_accumulation_steps"], 16)
    self.assertFalse(kwargs["prefuse_moe_weights"])
    self.assertTrue(kwargs["use_weight_converter"])
    self.assertEqual(kwargs["max_seq_token_per_tpu"], 0)
    self.assertIsNone(kwargs["profiling_options"])

  def test_plumbs_max_seq_token_per_tpu(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg, "rl_training_config.max_seq_token_per_tpu=4096"
    )
    self.assertEqual(kwargs["max_seq_token_per_tpu"], 4096)
    self.assertEqual(kwargs["gradient_accumulation_steps"], 1)

  def test_plumbs_profiler_and_tpu_options(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg,
        "rl_training_config.profiler_options.profiler_steps=3",
        "rl_training_config.profiler_options.skip_first_n_steps=8",
        "dist_training_config.profiler_period=10",
        "actor_model_config.kwargs.enable_tpu_profiling_options=false",
        "actor_model_config.kwargs.tpu_num_chips_to_profile_per_task=2",
        "actor_model_config.kwargs.tpu_num_sparse_cores_to_trace=3",
        "actor_model_config.kwargs.tpu_num_sparse_core_tiles_to_trace=4",
        "actor_model_config.kwargs.upload_all_profiler_results=true",
    )
    self.assertEqual(
        kwargs["profiling_options"],
        maxtext_utils.ProfilerOptions(
            skip_first_n_steps=8,
            profiler_steps=3,
            profiler_period=10,
            enable_tpu_profiling_options=False,
            tpu_num_chips_to_profile_per_task=2,
            tpu_num_sparse_cores_to_trace=3,
            tpu_num_sparse_core_tiles_to_trace=4,
            upload_all_profiler_results=True,
        ),
    )

  def test_plumbs_adamw_options(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg,
        f"{_OPT}.b2=0.98",
        f"{_OPT}.weight_decay=0.0",
        f"{_OPT}.opt_chain_type=",
    )
    self.assertEqual(
        kwargs["adamw_options"],
        maxtext_utils.AdamWOptions(
            b1=0.9,
            b2=0.98,
            eps=1e-8,
            weight_decay=0.0,
            gradient_clipping_threshold=0.0,
        ),
    )

  def test_plumbs_weight_dtype(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg, "actor_model_config.load_dtype=bfloat16"
    )
    self.assertEqual(kwargs["weight_dtype"], "bfloat16")

  def test_output_dir_defaults_under_artifact_root(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg, "cluster_config.artifact_root=/tmp/artifacts/run"
    )
    self.assertEqual(
        kwargs["base_output_directory"], "/tmp/artifacts/run/maxtext"
    )

  def test_plumbs_optional_maxtext_knobs(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg,
        "actor_model_config.kwargs.mesh_context=2",
        "actor_model_config.kwargs.base_num_kv_heads=8",
        "actor_model_config.kwargs.attention=flash",
        "actor_model_config.kwargs.padded_moe_mlp_dim=1024",
        "actor_model_config.kwargs.warmup_steps_fraction=0.1",
        "actor_model_config.kwargs.maxtext_output_dir=/tmp/maxtext_out",
        "actor_model_config.kwargs.prefuse_moe_weights=true",
        "weight_sync_config.use_weight_converter=false",
    )
    self.assertEqual(kwargs["mesh_context"], 2)
    self.assertEqual(kwargs["base_num_kv_heads"], 8)
    self.assertEqual(kwargs["attention"], "flash")
    self.assertEqual(kwargs["padded_moe_mlp_dim"], 1024)
    self.assertEqual(kwargs["warmup_steps_fraction"], 0.1)
    self.assertEqual(kwargs["base_output_directory"], "/tmp/maxtext_out")
    self.assertTrue(kwargs["prefuse_moe_weights"])
    self.assertFalse(kwargs["use_weight_converter"])

  def test_null_maxtext_knobs_keep_defaults(self, mock_build_cfg, *_):
    defaults = self._build_kwargs(mock_build_cfg)
    mock_build_cfg.reset_mock()
    kwargs = self._build_kwargs(
        mock_build_cfg,
        "actor_model_config.kwargs.mesh_context=null",
        "actor_model_config.kwargs.attention=null",
        "actor_model_config.kwargs.warmup_steps_fraction=null",
        "actor_model_config.kwargs.prefuse_moe_weights=null",
    )
    for key in (
        "mesh_context",
        "attention",
        "warmup_steps_fraction",
        "prefuse_moe_weights",
    ):
      self.assertEqual(kwargs[key], defaults[key], key)

  def test_maxtext_pad_id_comes_from_hf_tokenizer(
      self, mock_build_cfg, mock_create_mesh, mock_load_tokenizer
  ):
    factory, _ = trainer._create_maxtext_trainer_factory(
        _hp(*_MAXTEXT_ARGV), "trainer-0"
    )
    mock_load_tokenizer.assert_called_once()
    with mock.patch.object(maxtext_utils, "create_maxtext_engine") as engine:
      factory()
    engine.assert_called_once_with(
        mock_build_cfg.return_value,
        mesh=mock_create_mesh.return_value,
        tokenizer_pad_id=7,
        wrap_with_tunix_adapter=True,
    )

  def test_plumbs_checkpointing_options(self, mock_build_cfg, *_):
    kwargs = self._build_kwargs(
        mock_build_cfg,
        "rl_training_config.checkpointing_options.save_interval_steps=5",
        "rl_training_config.checkpointing_options.max_to_keep=3",
    )
    self.assertEqual(kwargs["checkpointing_options"].save_interval_steps, 5)
    self.assertEqual(kwargs["checkpointing_options"].max_to_keep, 3)


class MaxTextOptionsTest(absltest.TestCase):

  def test_checkpointing_options_zero_is_read_only(self):
    opts = trainer._maxtext_checkpointing_options(
        _hp("rl_training_config.checkpointing_options.save_interval_steps=0")
    )
    self.assertTrue(opts.read_only)
    self.assertEqual(opts.save_interval_steps, 0)

  def test_adamw_options_follow_optimizer_config(self):
    # Names are case-insensitive, as cli.config.create_optimizer reads them.
    self.assertEqual(
        trainer._maxtext_adamw_options(
            _hp(
                f"{_OPT}.opt_type=AdamW",
                f"{_OPT}.b1=0.8",
                f"{_OPT}.b2=0.98",
                f"{_OPT}.eps=1e-6",
                f"{_OPT}.weight_decay=0.05",
                f"{_OPT}.opt_chain_type=CLIP_BY_GLOBAL_NORM",
                f"{_OPT}.chain_kwargs.max_norm=0.125",
            )
        ),
        maxtext_utils.AdamWOptions(
            b1=0.8,
            b2=0.98,
            eps=1e-6,
            weight_decay=0.05,
            gradient_clipping_threshold=0.125,
        ),
    )

  def test_adamw_options_without_chain_disable_clipping(self):
    # Like the tunix backend, an unchained optimizer does not clip; MaxText's
    # own defaults (b2=0.95, weight_decay=0.1, clipping at 1.0) never apply.
    options = trainer._maxtext_adamw_options(
        _hp(f"{_OPT}.weight_decay=0.0", f"{_OPT}.opt_chain_type=")
    )
    self.assertEqual(
        (options.b2, options.weight_decay, options.gradient_clipping_threshold),
        (0.999, 0.0, 0.0),
    )

  def test_adamw_options_reject_unmapped_settings(self):
    for argv, error in (
        ([f"{_OPT}.opt_type=sgd"], "maps only adamw.*MAXTEXT_EXTRA_FLAGS"),
        (
            [f"{_OPT}.opt_chain_type=clip"],
            "maps only clip_by_global_norm.*MAXTEXT_EXTRA_FLAGS",
        ),
        ([f"{_OPT}.chain_kwargs.max_norm=null"], "needs a max_norm"),
        ([f"{_OPT}.chain_kwargs.max_norm=0"], "must be positive"),
    ):
      with self.subTest(argv=argv):
        with self.assertRaisesRegex(ValueError, error):
          trainer._maxtext_adamw_options(_hp(*argv))


if __name__ == "__main__":
  absltest.main()
