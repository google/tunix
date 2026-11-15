# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native CISPO training of a shared repair swarm on bounded public tasks."""

import argparse
import asyncio
import collections
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import uuid

from absl import flags
from flax import nnx
import jax
import numpy as np
import optax
from examples.swarm_math import train as shared_train
from examples.swarm_repair import cpu_fixture
from examples.swarm_repair import environment
from examples.swarm_repair import workflow
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.swarm import checkpoint
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import rlcr
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algorithm_config
from tunix.rl import cispo  # pylint: disable=unused-import; registers the native loss

ROOT = Path(__file__).parent


def team_credit(episodes):
  """Drop constant-reward task groups before token normalization, as in RLCR."""
  return rlcr.build_rlcr_credit(episodes)


def validate_config(cfg):
  integer_keys = (
      "num_steps",
      "tasks_per_step",
      "train_tasks",
      "dev_tasks",
      "test_tasks",
      "eval_attempts",
      "max_agents",
      "max_calls",
      "max_turns_per_agent",
      "max_messages",
      "max_workspace_chars",
      "max_total_tokens",
      "max_prompt_length",
      "max_response_length",
      "micro_batch_size",
      "rollout_batch_size",
      "checkpoint_every_steps",
      "val_every_steps",
  )
  if any(type(cfg[k]) is not int or cfg[k] < 1 for k in integer_keys):
    raise ValueError("Resource and training budgets must be positive integers.")
  if (
      type(cfg["group_size"]) is not int
      or cfg["group_size"] < 2
      or cfg["max_agents"] < 2
  ):
    raise ValueError(
        "Require group_size>=2 and max_agents>=2 for all baselines."
    )
  for key in ("deadline_seconds", "learning_rate"):
    if not np.isfinite(cfg[key]) or cfg[key] <= 0:
      raise ValueError(f"{key} must be positive and finite.")
  if not np.isfinite(cfg["max_logprob_error"]) or cfg["max_logprob_error"] < 0:
    raise ValueError("max_logprob_error must be finite and nonnegative.")
  if (
      cfg["train_temperature"] != 1
      or cfg["eval_temperature"] != 1
      or cfg["train_top_p"] != 1
      or cfg["eval_top_p"] != 1
  ):
    raise ValueError(
        "This profile uses temperature=1 and top_p=1 for both stages."
    )
  if cfg["rollout_engine"] != "vanilla" or cfg["rollout_split_fraction"] != 0:
    raise ValueError(
        "The initial public recipe uses native sampling on one mesh."
    )
  if cfg["dtype"] != "float32" or cfg["use_flash_attention"]:
    raise ValueError("Retain the validated float32 reference-attention path.")
  if cfg["tasks_per_step"] > cfg["train_tasks"]:
    raise ValueError("tasks_per_step exceeds the generated training pool.")
  if any(mode not in workflow.MODES for mode in cfg["eval_modes"]):
    raise ValueError("Unknown evaluation mode.")
  if cfg.get("training_mode", "swarm") not in workflow.MODES:
    raise ValueError("Unknown training topology.")


class Collector:

  def __init__(
      self, sampler, encode, tasks, config, *, stage="train", trace_path=None
  ):
    self.sampler, self.encode, self.tasks, self.config = (
        sampler,
        encode,
        tasks,
        config,
    )
    self.stage, self.trace_path = stage, trace_path
    self.workflow = workflow.RepairWorkflow(config)
    self.last_results = []
    self.retries = 0

  async def collect(self, selected, version, step=0, mode="swarm"):
    cfg = self.config
    attempts = (
        cfg["group_size"] if self.stage == "train" else cfg["eval_attempts"]
    )
    if isinstance(self.sampler, shared_train.NativeMathSampler):
      sampler_seed = int(
          np.random.SeedSequence([
              cfg["seed"],
              step,
              int(self.stage != "train"),
              workflow.MODES.index(mode),
          ]).generate_state(1)[0]
      )
      self.sampler.sampler._rng = jax.random.PRNGKey(sampler_seed)  # pylint: disable=protected-access

    async def group(task, index):
      for retry in range(3):

        async def attempt(sample):
          seed = int(
              np.random.SeedSequence([
                  cfg["seed"],
                  step,
                  index,
                  sample,
                  int(self.stage != "train"),
                  retry,
              ]).generate_state(1)[0]
          )
          return await self.workflow.run(
              task,
              self.sampler,
              self.encode,
              policy_version=version,
              seed=seed,
              episode_id=(
                  f"{self.stage}/{step}/{task.task_id}/{mode}/{retry}/{sample}"
              ),
              mode=mode,
          )

        outcomes = await asyncio.gather(
            *(attempt(i) for i in range(attempts)), return_exceptions=True
        )
        no_tokens = any(
            isinstance(r, workflow.NoCompletedActions) for r in outcomes
        )
        for result in outcomes:
          if isinstance(result, BaseException) and not isinstance(
              result, workflow.NoCompletedActions
          ):
            raise result
        if not no_tokens:
          return outcomes
        self.retries += 1
      raise RuntimeError(
          "A complete comparison group could not be collected after three"
          " attempts."
      )

    groups = await asyncio.gather(
        *(group(task, i) for i, task in enumerate(selected))
    )
    self.last_results = [
        result for group_results in groups for result in group_results
    ]
    if self.trace_path:
      with Path(self.trace_path).open("a") as stream:
        for result in self.last_results:
          stream.write(json.dumps(dataclasses.asdict(result)) + "\n")
    return tuple(r.episode for r in self.last_results)

  async def __call__(self, version, step):
    order = np.random.default_rng(self.config["seed"] + step).permutation(
        len(self.tasks)
    )
    return await self.collect(
        [self.tasks[i] for i in order[: self.config["tasks_per_step"]]],
        version,
        step,
        mode=self.config.get("training_mode", "swarm"),
    )


async def evaluate(collector, version):
  output = {}
  for mode in collector.config["eval_modes"]:
    episodes = await collector.collect(collector.tasks, version, mode=mode)
    grouped = collections.defaultdict(list)
    for episode in episodes:
      grouped[episode.task_id].append(episode.reward)
    results = collector.last_results
    output[mode] = {
        "questions": len(grouped),
        "attempts": len(episodes),
        "avg_at_k": float(np.mean([e.reward for e in episodes])),
        "pass_at_k": float(np.mean([max(v) for v in grouped.values()])),
        "mean_generated_tokens": float(
            np.mean([r.generated_tokens for r in results])
        ),
        "mean_total_tokens": float(
            np.mean([
                sum(len(c.prompt_ids) + len(c.completion_ids) for c in e.calls)
                for e in episodes
            ])
        ),
        "mean_calls": float(np.mean([len(e.calls) for e in episodes])),
        "mean_latency_seconds": float(
            np.mean([r.elapsed_seconds for r in results])
        ),
        "p95_latency_seconds": float(
            np.quantile([r.elapsed_seconds for r in results], 0.95)
        ),
        "mean_peak_concurrent_calls": float(
            np.mean([r.max_concurrent_calls for r in results])
        ),
        "deadline_fraction": float(
            np.mean([r.status == "deadline" for r in results])
        ),
        "failed_fraction": float(
            np.mean([r.status == "failed" for r in results])
        ),
        "max_calls_budget": collector.config["max_calls"],
        "total_tokens_budget": collector.config["max_total_tokens"],
        "deadline_seconds": collector.config["deadline_seconds"],
    }
  return output


def backends(backend, cfg, mesh, model_path):
  if backend == "cpu":
    if cfg["model_id"] != "tiny-categorical-fixture":
      raise ValueError("Use the CPU fixture profile.")
    actor = cpu_fixture.make_policy()
    serving = nnx.clone(actor)

    def sync():
      nnx.update(serving, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))

    sync()
    ds, es = cpu_fixture.TinySampler(serving), cpu_fixture.TinySampler(serving)
    return actor, ds, es, ds.encode, es.encode, 0, 0, sync
  actor, ds, es, encode, pad, eos, sync = shared_train.create_backends(
      "qwen", cfg, mesh, model_path
  )
  return actor, ds, es, encode, encode, pad, eos, sync


def run(
    backend,
    cfg,
    output,
    *,
    model_path=None,
    checkpoint_dir=None,
    resume=False,
    mode="train",
    eval_split="dev",
):
  validate_config(cfg)
  if mode == "train" and eval_split != "dev":
    raise ValueError(
        "Training monitors development data only; test is evaluation-only."
    )
  if mode == "eval" and not resume:
    raise ValueError("Evaluation requires --resume.")
  training = environment.tasks(
      "train", count=cfg["train_tasks"], seed=cfg["seed"]
  )
  evaluation = environment.tasks(
      eval_split, count=cfg[f"{eval_split}_tasks"], seed=cfg["seed"]
  )
  train_manifest = environment.manifest(training, cfg["seed"])
  eval_manifest = environment.manifest(evaluation, cfg["seed"])
  invariant = {k: v for k, v in cfg.items() if k != "num_steps"}
  config_hash = hashlib.sha256(
      json.dumps(invariant, sort_keys=True).encode()
  ).hexdigest()
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()) and not resume:
    raise ValueError("Use an empty run directory or --resume.")
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["swarm_repair"])
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top

    pathwaysutils.initialize()
  mesh, _ = shared_train.build_meshes(cfg, backend=backend)
  with mesh:
    actor, ds, es, encode, eval_encode, pad, eos, sync = backends(
        backend, cfg, mesh, model_path
    )
    optimizer = (
        optax.sgd(cfg["learning_rate"])
        if backend == "cpu"
        else optax.adamw(cfg["learning_rate"])
    )
    trainer = peft_trainer_v2.PeftTrainer(
        actor,
        optimizer,
        peft_trainer_v2.TrainingConfig(
            max_steps=cfg["num_steps"],
            eval_every_n_steps=cfg["num_steps"] + 1,
            gradient_accumulation_steps=2,
            resume_from_checkpoint_on_init=False,
            checkpoint_root_directory=checkpoint_dir
            or str(output / "checkpoints"),
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: trainer,
        logps_chunk_size=cfg["logps_chunk_size"],
        execution_context=mesh,
    )
    worker.initialize()
    engine = local_engine.LocalSwarmEngine(worker, sync)
    collector = Collector(
        ds, encode, training, cfg, trace_path=output / "train_traces.jsonl"
    )
    evaluator = Collector(es, eval_encode, evaluation, cfg, stage="eval")
    validations = []
    initial = None
    initial_manifest = eval_manifest
    last_saved = -1
    run_id = uuid.uuid4().hex
    cursor_path = output / "round_state.json"

    def metadata():
      return {
          "repair_round": program.step,
          "policy_version": program.policy_version,
          "history": program.history,
          "optimizer_step": trainer.train_steps,
          "config_hash": config_hash,
          "train_manifest": train_manifest,
          "eval_manifest": eval_manifest,
          "initial_eval": initial,
          "initial_eval_manifest": initial_manifest,
          "validations": validations,
          "run_id": run_id,
      }

    def commit():
      nonlocal last_saved
      state = metadata()
      if trainer.train_steps > last_saved:
        worker.save_checkpoint(state, step=trainer.train_steps, force=True)
        trainer.checkpoint_manager.wait()
        last_saved = trainer.train_steps
      checkpoint.write_manifest(cursor_path, state)

    async def collect(version, step):
      nonlocal last_saved
      if (
          step
          and step % cfg["val_every_steps"] == 0
          and not any(v["step"] == step for v in validations)
      ):
        validations.append(
            {"step": step, "metrics": await evaluate(evaluator, version)}
        )
      if step and step % cfg["checkpoint_every_steps"] == 0:
        commit()
      return await collector(version, step)

    algo = algorithm_config.GRPOConfig(
        policy_loss_fn="cispo",
        num_generations=cfg["group_size"],
        beta=0.0,
        epsilon_high=5.0,
        loss_agg_mode="token-mean",
        use_rollout_logps=True,
    )
    algo.temperature = 1.0
    program = swarm_program.SwarmRLProgram(
        collect,
        swarm_program.SwarmConfig(
            group_size=cfg["group_size"],
            micro_batch_size=cfg["micro_batch_size"],
            max_prompt_length=cfg["max_prompt_length"],
            max_response_length=cfg["max_response_length"],
            pad_id=pad,
            eos_id=eos,
            normalization="episode",
            invalid_action_penalty=0.0,
            max_logprob_error=cfg["max_logprob_error"],
            sampler_is=None,
        ),
        credit_builder=team_credit,
        algo_config=algo,
    )
    try:
      if resume:
        restored = worker.restore_checkpoint()
        if cursor_path.exists():
          cursor = checkpoint.read_manifest(cursor_path)
          if restored.get("run_id") != cursor.get("run_id"):
            raise ValueError(
                "Round manifest refers to a different training run."
            )
          if cursor["optimizer_step"] > restored["step"]:
            raise ValueError(
                "Round manifest references an unavailable checkpoint."
            )
          if cursor["repair_round"] >= restored["repair_round"]:
            if cursor["optimizer_step"] != restored["step"]:
              raise ValueError("Checkpoint and round cursor disagree.")
            restored = {**cursor, "step": restored["step"]}
        if (
            restored.get("config_hash") != config_hash
            or restored.get("train_manifest") != train_manifest
            or (
                mode == "train"
                and restored.get("eval_manifest") != eval_manifest
            )
        ):
          raise ValueError(
              "Checkpoint configuration or dataset identity changed."
          )
        program.restore_state(
            step=restored["repair_round"], history=restored["history"]
        )
        engine.version = restored["policy_version"]
        program.policy_version = engine.version
        validations[:] = restored["validations"]
        initial = restored["initial_eval"]
        initial_manifest = restored.get(
            "initial_eval_manifest", restored["eval_manifest"]
        )
        run_id = restored["run_id"]
        last_saved = trainer.train_steps
        sync()
      else:
        program.policy_version = 0
      before = (
          np.asarray(actor.logits[...]).copy() if backend == "cpu" else None
      )
      if initial is None and mode == "train":
        initial = asyncio.run(evaluate(evaluator, engine.version))
      if mode == "train" and not resume:
        commit()
      if mode == "train" and cfg["num_steps"] > program.step:
        program.run(engine, num_steps=cfg["num_steps"] - program.step)
      report = {
          "backend": backend,
          "mode": mode,
          "implementation": "public_dsl_rlcr_adaptation",
          "train_manifest": train_manifest,
          "eval_manifest": eval_manifest,
          "initial_eval_manifest": initial_manifest,
          "initial_untrained_eval": initial,
          "final_eval": asyncio.run(
              evaluate(evaluator, program.policy_version)
          ),
          "validations": validations,
          "history": program.history,
          "optimizer_step": trainer.train_steps,
          "complete_group_retries": collector.retries,
      }
      if backend == "cpu":
        report["parameter_delta_l2"] = float(
            np.linalg.norm(np.asarray(actor.logits[...]) - before)
        )
        np.savez(
            output / "parameters.npz",
            initial=before,
            final=np.asarray(actor.logits[...]),
        )
      if mode == "train":
        commit()
      (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
      (
          output / ("report.json" if mode == "train" else "eval_report.json")
      ).write_text(json.dumps(report, indent=2) + "\n")
      return report
    finally:
      for sampler in (ds, es):
        if hasattr(sampler, "close"):
          asyncio.run(sampler.close())
      trainer.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--config")
  parser.add_argument("--output", required=True)
  parser.add_argument("--model-path")
  parser.add_argument("--checkpoint-dir")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--mode", choices=("train", "eval"), default="train")
  parser.add_argument("--eval-split", choices=("dev", "test"), default="dev")
  parser.add_argument("--num-steps", type=int)
  parser.add_argument("--training-mode", choices=workflow.MODES)
  args = parser.parse_args()
  path = args.config or ROOT / "configs" / (
      "cpu_smoke.json" if args.backend == "cpu" else "qwen3_4b.json"
  )
  cfg = json.loads(Path(path).read_text())
  if args.num_steps is not None:
    cfg["num_steps"] = args.num_steps
  if args.training_mode:
    cfg["training_mode"] = args.training_mode
  print(
      json.dumps(
          run(
              args.backend,
              cfg,
              args.output,
              model_path=args.model_path,
              checkpoint_dir=args.checkpoint_dir,
              resume=args.resume,
              mode=args.mode,
              eval_split=args.eval_split,
          ),
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
