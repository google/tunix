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

"""Generation-evaluation lifecycle and pinned model control-plane contracts."""

import copy
import unittest
from unittest import mock

from tunix.dist_rl.distillation import identity
from tunix.dist_rl.distillation import on_policy
from tunix.dist_rl.examples.math_gsm8k_dist import run_gsm8k_dist_grpo as entry
from tunix.dist_rl.examples.on_policy_distillation import evaluation
from tunix.dist_rl.examples.on_policy_distillation import models
from tunix.dist_rl.orchestrator import rl_program


class OPDWorkflowTest(unittest.IsolatedAsyncioTestCase):

  def _program(self, *, eval_only=False, require_resume=False):
    args = entry._parse_args(["--num_generations=2", "--mini_batch_size=1"])
    self.scorer = mock.Mock(score=mock.AsyncMock())
    with mock.patch.object(rl_program, "MetricsLogger"):
      program = entry._EvaluatedOPDProgram(
          algo=entry._build_algo(args),
          generation_args=entry.datatypes.GenerationArgs(
              temperature=1, top_p=1, top_k=0, seed=42
          ),
          opd_config=on_policy.OnPolicyDistillationConfig(),
          teacher_scorer=self.scorer,
          opd_teacher_identity=identity.TeacherIdentity(
              declared_revision="fixed-teacher",
              tokenizer_fingerprint="fixed-tokenizer",
          ),
          evaluation_config=evaluation.EvaluationConfig(),
          evaluation_examples=[],
          evaluation_output_dir="/unused/evaluation",
          model_identity={"repo_id": "fixed-student"},
          eval_only=eval_only,
          require_resume=require_resume,
      )
    self.addCleanup(program.close)
    return program

  def _engine(self, program, *, step=0):
    return mock.Mock(
        resume_from_checkpoint=mock.AsyncMock(return_value=step),
        restored_checkpoint_metadata=(
            {
                **identity.checkpoint_metadata(program._opd_manifest),
                "policy_version": step,
            }
            if step
            else None
        ),
        prepare_rollout_policy=mock.AsyncMock(),
    )

  async def test_fresh_training_evaluates_baseline_then_current_final_policy(
      self,
  ):
    program = self._program()
    engine = self._engine(program)
    events = []

    async def train(instance, actual_engine, **kwargs):
      del kwargs
      instance.engine = actual_engine
      await instance._resume_from_checkpoint()
      events.append("train")
      instance._step = 3
      instance.policy_version = 3

    async def evaluate(actual_engine, examples, **kwargs):
      self.assertIs(actual_engine, engine)
      del examples
      events.append(kwargs["label"])
      self.assertEqual(
          kwargs["policy_version"], 0 if kwargs["label"] == "baseline" else 3
      )
      self.assertEqual(
          kwargs["model_identity"]["evaluation_source"], "live_synced_rollout"
      )

    with mock.patch.object(rl_program.StandardRLProgram, "run_async", train):
      with mock.patch.object(
          evaluation, "evaluate_engine", side_effect=evaluate
      ):
        await program.run_async(engine)
    self.assertEqual(events, ["baseline", "train", "final"])
    engine.prepare_rollout_policy.assert_awaited_once()
    self.scorer.score.assert_not_awaited()

  async def test_resume_never_labels_restored_weights_as_baseline(self):
    program = self._program()
    program.engine = self._engine(program, step=2)
    with mock.patch.object(evaluation, "evaluate_engine") as evaluate:
      await program._resume_from_checkpoint()
    self.assertEqual(program.step, 2)
    evaluate.assert_not_called()
    program.engine.prepare_rollout_policy.assert_not_awaited()

  async def test_eval_only_requires_checkpoint_and_never_enters_train_program(
      self,
  ):
    for step in (0, 2):
      with self.subTest(step=step):
        program = self._program(eval_only=True)
        engine = self._engine(program, step=step)
        with mock.patch.object(
            rl_program.StandardRLProgram, "run_async"
        ) as train:
          with mock.patch.object(
              evaluation, "evaluate_engine", new_callable=mock.AsyncMock
          ) as evaluate:
            if step:
              await program.run_async(engine)
              self.assertEqual(evaluate.call_args.kwargs["label"], "checkpoint")
              self.assertEqual(evaluate.call_args.kwargs["policy_version"], 2)
              self.assertEqual(
                  evaluate.call_args.kwargs["model_identity"][
                      "restored_checkpoint_metadata"
                  ],
                  engine.restored_checkpoint_metadata,
              )
            else:
              with self.assertRaisesRegex(
                  ValueError, "existing nonzero checkpoint"
              ):
                await program.run_async(engine)
              evaluate.assert_not_awaited()
          train.assert_not_called()
        self.scorer.score.assert_not_awaited()

  async def test_eval_only_cannot_bypass_saved_opd_identity(self):
    program = self._program(eval_only=True)
    engine = self._engine(program, step=2)
    engine.restored_checkpoint_metadata[identity.METADATA_KEY][
        "coefficient"
    ] = 0.1
    with mock.patch.object(evaluation, "evaluate_engine") as evaluate:
      with self.assertRaisesRegex(ValueError, "objective differs"):
        await program.run_async(engine)
    evaluate.assert_not_called()

  async def test_explicit_resume_cannot_silently_start_fresh(self):
    program = self._program(require_resume=True)
    program.engine = self._engine(program)
    with mock.patch.object(evaluation, "evaluate_engine") as evaluate:
      with self.assertRaisesRegex(ValueError, "resume requires an existing"):
        await program._resume_from_checkpoint()
    evaluate.assert_not_called()


class OPDPinnedConfigTest(unittest.TestCase):

  def _args(self):
    return entry._parse_args([
        "--opd_models_manifest=/prepared/models.json",
        "--model_id=Qwen/Qwen3-8B",
        "--model_dir=/prepared/student",
        "--tokenizer_path=/prepared/student",
        "--opd_teacher_url=http://teacher:30000",
        "--opd_teacher_tokenizer_path=/prepared/teacher",
        "--opd_teacher_revision=fixed-teacher",
    ])

  def test_prepared_model_settings_must_match_manifest(self):
    manifest = {
        "student": {
            "repo_id": "Qwen/Qwen3-8B",
            "snapshot_path": "/prepared/student",
            "tokenizer_path": "/prepared/student",
        },
        "teacher": {
            "tokenizer_path": "/prepared/teacher",
            "declared_revision": "fixed-teacher",
        },
    }
    with mock.patch.object(models, "load_manifest", return_value=manifest):
      self.assertEqual(entry._load_opd_models_manifest(self._args()), manifest)
      for field, value in (
          ("model_id", "other-model"),
          ("model_dir", "/other/weights"),
          ("tokenizer_path", "/other/student"),
          ("opd_teacher_tokenizer_path", "/other/teacher"),
          ("opd_teacher_revision", "mutable-teacher"),
      ):
        args = copy.copy(self._args())
        setattr(args, field, value)
        with self.subTest(field=field):
          with self.assertRaisesRegex(ValueError, "manifest"):
            entry._load_opd_models_manifest(args)

  def test_eval_config_requires_prepared_identity_and_eval_only_requires_config(
      self,
  ):
    args = entry._parse_args(["--opd_eval_config=evaluation.json"])
    with self.assertRaisesRegex(ValueError, "models_manifest"):
      entry._load_opd_models_manifest(args)
    args = self._args()
    args.opd_eval_only = True
    with self.assertRaisesRegex(ValueError, "eval_config"):
      entry._load_opd_evaluation(args)

  def test_dataset_version_and_seed_reach_loader_without_repeating_budget(self):
    args = entry._parse_args([
        "--tfds_dataset=gsm8k:1.0.0",
        "--max_steps=2",
        "--batch_size=2",
        "--seed=7",
    ])
    records = [
        dict(prompts=f"p{i}", question=f"q{i}", answer=str(i)) for i in range(6)
    ]
    with mock.patch.object(
        entry.gsm8k, "load_gsm8k_dataset", return_value=records
    ) as load:
      prompts = list(entry._iter_prompt_items(args))
    self.assertEqual([p["prompt"] for p in prompts], ["p0", "p1", "p2", "p3"])
    self.assertEqual(load.call_args.kwargs["dataset_name"], "gsm8k:1.0.0")
    self.assertEqual(load.call_args.kwargs["seed"], 7)

  def _configured_main_program(self, *extra_args):
    tokenizer = mock.MagicMock(pad_token_id=0, eos_token_id=1)
    manifest = {
        "student": {"repo_id": "fixed-student"},
        "tokenizer_fingerprint": "fp",
    }
    config = evaluation.EvaluationConfig()
    cluster = mock.Mock(trajectory_store=None)
    cluster.registry.infos.return_value = []
    teacher_identity = identity.TeacherIdentity(
        declared_revision="fixed-teacher", tokenizer_fingerprint="fp"
    )
    with mock.patch.object(
        entry, "_load_opd_models_manifest", return_value=manifest
    ):
      with mock.patch.object(
          entry, "_load_opd_evaluation", return_value=config
      ):
        with mock.patch.object(
            entry,
            "_opd_program_kwargs",
            return_value={
                "opd_config": on_policy.OnPolicyDistillationConfig(),
                "teacher_scorer": mock.Mock(score=mock.AsyncMock()),
                "opd_teacher_identity": teacher_identity,
            },
        ):
          with mock.patch.object(
              entry.AutoTokenizer, "from_pretrained", return_value=tokenizer
          ):
            with mock.patch.object(
                entry.orchestrator, "ClusterOrchestrator", return_value=cluster
            ):
              with mock.patch.object(
                  evaluation, "load_examples", return_value=[]
              ):
                with mock.patch.object(
                    rl_program, "MetricsLogger"
                ), mock.patch.object(entry.logging, "basicConfig"):
                  entry.main(
                      [
                          "--seed=7",
                          "--opd_eval_output_dir=/run/reports",
                          "--weight_sync_mode=raiden",
                          "--opd_require_resume",
                          "--opd_eval_only",
                          *extra_args,
                      ],
                      context=mock.Mock(),
                  )
    return cluster.run.call_args.kwargs["program"]

  def test_main_selects_evaluated_program_and_passes_seed_and_resume_policy(
      self,
  ):
    program = self._configured_main_program("--generation_seed=7")
    self.assertIsInstance(program, entry._EvaluatedOPDProgram)
    self.assertEqual(program.generation_args.seed, 7)
    self.assertEqual(program._evaluation_config, evaluation.EvaluationConfig())
    self.assertTrue(program._require_resume)
    self.assertTrue(program._eval_only)
    self.assertEqual(program._evaluation_output_dir, "/run/reports")

  def test_main_keeps_legacy_data_seed_out_of_generation_and_opd_manifest(self):
    program = self._configured_main_program()
    self.assertIsNone(program.generation_args.seed)
    self.assertNotIn("seed", program.generation_args.as_kwargs())
    self.assertNotIn("seed", program._opd_manifest["sampling"])


if __name__ == "__main__":
  unittest.main()
