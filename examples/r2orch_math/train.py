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

"""Train a Director with frozen mathematical Workers through native dist_rl."""

import argparse
import asyncio
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
from examples.r2orch_math import cpu_fixture
from examples.r2orch_math import optimizers
from examples.r2orch_math import prepare_data
from examples.r2orch_math import workflow
from examples.swarm_math import train as shared_train
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import director_credit
from tunix.dist_rl.swarm import director_loss
from tunix.dist_rl.swarm import checkpoint
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import runtime
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algorithm_config

ROOT = Path(__file__).parent


def load_tasks(path):
  rows = [
      json.loads(line)
      for line in Path(path).read_text().splitlines()
      if line.strip()
  ]
  tasks = [
      workflow.MathTask(
          **{k: row[k] for k in ("task_id", "problem", "answer")},
          benchmark=row.get("benchmark", "aime"),
      )
      for row in rows
  ]
  if not tasks or len({t.task_id for t in tasks}) != len(tasks):
    raise ValueError("Task IDs must be unique and the split must be nonempty.")
  if any(t.benchmark != "aime" for t in tasks):
    raise ValueError(
        "This launcher scores AIME only; OOD requires a math-equivalence"
        " scorer."
    )
  return tasks


def validate_config(config):
  if config.get("executor", "native") not in ("native", "source"):
    raise ValueError("executor must be native or source.")
  if (
      config.get("executor") == "source"
      and config.get("require_source_tools", True)
      and not config.get("tool_image")
  ):
    raise ValueError(
        "The original AIME recipe requires --tool-image pinned by digest."
    )
  if (
      config["rollout_engine"] != "vanilla"
      or config["rollout_split_fraction"] != 0
  ):
    raise ValueError(
        "The first Director recipe uses native sampling on one mesh."
    )
  if config["dtype"] != "float32" or config["use_flash_attention"]:
    raise ValueError("Keep validated float32 reference-attention gradients.")
  if config["train_temperature"] != 1 or config["train_top_p"] != 1:
    raise ValueError("Director training requires temperature=1 and top_p=1.")
  if config["eval_top_p"] != 1:
    raise ValueError("The native Director evaluation profile requires top_p=1.")
  for key in ("worker_temperature", "learning_rate", "eval_temperature"):
    if not np.isfinite(config[key]) or config[key] <= 0:
      raise ValueError(f"{key} must be finite and positive.")
  for key in ("kl_beta", "cost_lambda", "max_logprob_error"):
    if not np.isfinite(config[key]) or config[key] < 0:
      raise ValueError(f"{key} must be finite and nonnegative.")
  positive = (
      "num_steps",
      "tasks_per_step",
      "max_agents",
      "max_rounds",
      "parallel_workers",
      "max_total_tokens",
      "max_prompt_length",
      "max_response_length",
      "worker_max_tokens",
      "micro_batch_size",
      "rollout_batch_size",
      "val_every_steps",
      "checkpoint_every_steps",
      "eval_attempts",
  )
  if (
      any(type(config[k]) is not int or config[k] < 1 for k in positive)
      or config["group_size"] < 2
  ):
    raise ValueError("Invalid positive integer budget or rollout group size.")


def learning_plan():
  return policy.LearningPlan(
      bindings=(
          policy.PolicyBinding(policy_id="director", trainable=True),
          policy.PolicyBinding(policy_id="worker", trainable=False, version=0),
      ),
      learner_policy_id="director",
  )


def make_workflow(
    config,
    *,
    grader,
    reference_source=None,
    model_path=None,
    actor=None,
    pad_id=0,
    eos_id=0,
):
  """Select the original pinned executor or the separate compact CPU fixture."""
  budgets = {
      key: config[key]
      for key in (
          "max_agents",
          "max_rounds",
          "parallel_workers",
          "max_total_tokens",
      )
  }
  if config.get("executor", "native") == "native":
    return workflow.DirectorWorkflow(grader=grader, **budgets)
  if reference_source is None or model_path is None or actor is None:
    raise ValueError(
        "The source executor requires pinned source/model paths and an actor."
    )
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from examples.r2orch_math import source_tools  # pylint: disable=g-import-not-at-top
  from examples.r2orch_math import source_workflow  # pylint: disable=g-import-not-at-top

  image = config.get("tool_image")
  if config.get("require_source_tools", True) and not image:
    raise ValueError(
        "The original AIME recipe requires --tool-image pinned by digest."
    )
  tools = (
      source_tools.original_aime_tools(reference_source, image=image)
      if image
      else {}
  )
  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

  async def binary_sampler(prompt_ids, support_ids, seed):
    del seed  # The original transport samples from this complete distribution.
    return np.asarray(
        director_loss.binary_logprobs(
            actor,
            prompt_ids,
            support_ids,
            pad_id=pad_id,
            eos_id=eos_id,
            chunk_size=config["logps_chunk_size"],
        )
    )

  return source_workflow.SourceDirectorWorkflow(
      reference_source,
      tokenizer=tokenizer,
      binary_sampler=binary_sampler,
      grader=grader,
      tools=tools,
      **budgets,
      timeout_seconds=config.get("episode_timeout_seconds", 300.0),
      enable_thinking=config.get("enable_thinking", False),
  )


class Collector:

  def __init__(
      self,
      director_sampler,
      worker_sampler,
      encode,
      tasks,
      config,
      *,
      trace_path=None,
      mode="train",
      grader=workflow.grade_aime,
      workflow_instance=None,
  ):
    self.director_sampler, self.worker_sampler = (
        director_sampler,
        worker_sampler,
    )
    self.encode, self.tasks, self.config = encode, tasks, config
    self.trace_path, self.mode = trace_path, mode
    self.workflow = workflow_instance or workflow.DirectorWorkflow(
        grader=grader,
        **{
            k: config[k]
            for k in (
                "max_agents",
                "max_rounds",
                "parallel_workers",
                "max_total_tokens",
            )
        },
    )
    self.last_results = []

  async def collect(self, tasks, version, step=0):
    cfg = self.config
    attempts = (
        cfg["group_size"] if self.mode == "train" else cfg["eval_attempts"]
    )
    # Native decoding owns its PRNG instead of using individual request seeds.
    # Reinitialize at a round boundary so checkpoint resume replays exactly.
    for index, sampler in enumerate(
        (self.director_sampler, self.worker_sampler)
    ):
      if isinstance(sampler, shared_train.NativeMathSampler):
        seed = int(
            np.random.SeedSequence(
                [cfg["seed"], step, index, int(self.mode != "train")]
            ).generate_state(1)[0]
        )
        sampler.sampler._rng = jax.random.PRNGKey(seed)  # pylint: disable=protected-access

    async def attempt(task, index, sample):
      seed = int(
          np.random.SeedSequence(
              [cfg["seed"], step, index, sample, int(self.mode != "train")]
          ).generate_state(1)[0]
      )
      episode_id = f"{self.mode}/{step}/{task.task_id}/{sample}"
      director = runtime.RecordingRuntime(
          self.director_sampler,
          self.encode,
          sampler_lib.SamplingParams(
              max_tokens=cfg["max_response_length"],
              temperature=cfg[
                  f"{'train' if self.mode == 'train' else 'eval'}_temperature"
              ],
              top_p=1.0,
              seed=seed,
          ),
          policy_id="director",
          policy_version=version,
          episode_id=episode_id + "/director",
          max_prompt_length=cfg["max_prompt_length"],
      )
      worker = runtime.RecordingRuntime(
          self.worker_sampler,
          self.encode,
          sampler_lib.SamplingParams(
              max_tokens=cfg["worker_max_tokens"],
              temperature=cfg["worker_temperature"],
              top_p=1.0,
              seed=seed,
          ),
          policy_id="worker",
          policy_version=0,
          episode_id=episode_id + "/worker",
          max_prompt_length=cfg.get(
              "worker_max_prompt_length", cfg["max_prompt_length"]
          ),
      )
      return await self.workflow.run(
          task, director, worker, episode_id=episode_id
      )

    self.last_results = await asyncio.gather(*(
        attempt(task, index, sample)
        for index, task in enumerate(tasks)
        for sample in range(attempts)
    ))
    if self.trace_path:
      with Path(self.trace_path).open("a") as stream:
        for result in self.last_results:
          stream.write(json.dumps(dataclasses.asdict(result)) + "\n")
    return tuple(result.episode for result in self.last_results)

  async def __call__(self, version, step):
    # A deterministic full-pool permutation per round; no benchmark leakage.
    indices = np.random.default_rng(self.config["seed"] + step).permutation(
        len(self.tasks)
    )
    return await self.collect(
        [self.tasks[i] for i in indices[: self.config["tasks_per_step"]]],
        version,
        step,
    )


async def evaluate(collector, version):
  episodes = await collector.collect(collector.tasks, version)
  groups = {}
  for episode in episodes:
    groups.setdefault(episode.task_id, []).append(episode.reward)
  return {
      "questions": len(groups),
      "attempts": len(episodes),
      "avg_at_k": float(np.mean([e.reward for e in episodes])),
      "pass_at_k": float(np.mean([max(values) for values in groups.values()])),
      "mean_calls": float(np.mean([len(e.calls) for e in episodes])),
      "mean_generated_tokens": float(
          np.mean(
              [sum(len(c.completion_ids) for c in e.calls) for e in episodes]
          )
      ),
      "mean_worker_cost": float(np.mean([e.worker_cost for e in episodes])),
      "finished_fraction": float(
          np.mean([r.termination == "finished" for r in collector.last_results])
      ),
  }


def _snapshot_digest(model):
  digest = hashlib.sha256()
  for leaf in jax.tree.leaves(nnx.state(model)):
    array = np.asarray(leaf)
    digest.update(str((array.shape, array.dtype)).encode())
    digest.update(array.tobytes())
  return digest.hexdigest()


def _backends(backend, cfg, mesh, model_path):
  if backend == "cpu":
    if cfg["model_id"] != "tiny-categorical-fixture":
      raise ValueError("Use cpu_smoke.json for the CPU categorical fixture.")
    actor, frozen = cpu_fixture.make_policy(), cpu_fixture.make_policy()
    serving = nnx.clone(actor)

    def sync():
      nnx.update(serving, jax.tree.map(lambda a: a.copy(), nnx.state(actor)))

    sync()
    ds = cpu_fixture.TinySampler(serving)
    return (
        actor,
        frozen,
        ds,
        ds,
        cpu_fixture.TinySampler(frozen),
        cpu_fixture.encode,
        0,
        0,
        sync,
    )
  from examples.r2orch_math import backends  # pylint: disable=g-import-not-at-top

  return backends.create_qwen_backends(cfg, mesh, model_path)


def run(
    backend,
    config,
    train_tasks,
    eval_tasks,
    output,
    *,
    model_path=None,
    checkpoint_dir=None,
    resume=False,
    mode="train",
    reference_source=None,
):
  validate_config(config)
  if config["tasks_per_step"] > len(train_tasks):
    raise ValueError("tasks_per_step exceeds the training pool.")
  if {t.task_id for t in train_tasks} & {t.task_id for t in eval_tasks} or {
      " ".join(t.problem.split()) for t in train_tasks
  } & {" ".join(t.problem.split()) for t in eval_tasks}:
    raise ValueError("Training and evaluation splits must be disjoint.")
  if mode == "eval" and not resume:
    raise ValueError("Evaluation mode requires --resume and a checkpoint.")
  grader = workflow.grade_aime
  if backend == "qwen":
    if reference_source is None:
      raise ValueError(
          "Qwen requires --reference-source for the pinned original AIME"
          " scorer."
      )
    from examples.r2orch_math.grading import load_original_aime_grader  # pylint: disable=g-import-not-at-top

    grader = load_original_aime_grader(reference_source)
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()) and not resume:
    raise ValueError("Use an empty output directory, or --resume.")
  checkpoint_root = Path(checkpoint_dir or output / "checkpoints")
  if not resume and checkpoint_root.exists() and any(checkpoint_root.iterdir()):
    raise ValueError("Use an empty checkpoint directory, or --resume.")
  manifest_path = checkpoint_root / "round_state.json"
  run_id = uuid.uuid4().hex
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["r2orch_math"])
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top

    pathwaysutils.initialize()
  mesh, _ = shared_train.build_meshes(config, backend=backend)
  data_hash = hashlib.sha256(
      json.dumps(
          [dataclasses.asdict(t) for t in train_tasks], sort_keys=True
      ).encode()
  ).hexdigest()
  eval_hash = hashlib.sha256(
      json.dumps(
          [dataclasses.asdict(t) for t in eval_tasks], sort_keys=True
      ).encode()
  ).hexdigest()
  invariant_config = {k: v for k, v in config.items() if k != "num_steps"}
  invariant_hash = hashlib.sha256(
      json.dumps(invariant_config, sort_keys=True).encode()
  ).hexdigest()
  with mesh:
    actor, frozen, ds, es, ws, encode, pad, eos, sync = _backends(
        backend, config, mesh, model_path
    )
    frozen_before = _snapshot_digest(frozen) if backend == "cpu" else None
    optimizer = optimizers.make_optimizer(
        config, backend, num_optimizer_steps=2 * config["num_steps"]
    )
    trainer = peft_trainer_v2.PeftTrainer(
        actor,
        optimizer,
        peft_trainer_v2.TrainingConfig(
            max_steps=2 * config["num_steps"],
            eval_every_n_steps=2 * config["num_steps"] + 1,
            gradient_accumulation_steps=2,
            checkpoint_root_directory=str(checkpoint_root),
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: trainer,
        logps_chunk_size=config["logps_chunk_size"],
        execution_context=mesh,
    )
    worker.initialize()

    def actor_score(payload):
      return np.asarray(
          director_loss.policy_logprobs(
              actor,
              payload,
              pad_id=pad,
              eos_id=eos,
              chunk_size=config["logps_chunk_size"],
          )[0]
      )

    engine = local_engine.LocalSwarmEngine(
        worker, sync, actor_scorer=actor_score
    )

    async def reference_score(payload):
      values, _, support = director_loss.policy_logprobs(
          frozen,
          payload,
          pad_id=pad,
          eos_id=eos,
          chunk_size=config["logps_chunk_size"],
      )
      return swarm_program.ReferenceLogprobs(
          np.asarray(values), None if support is None else np.asarray(support)
      )

    collector = Collector(
        ds,
        ws,
        encode,
        train_tasks,
        config,
        trace_path=output / "train_traces.jsonl",
        grader=grader,
        workflow_instance=make_workflow(
            config,
            grader=grader,
            reference_source=reference_source,
            model_path=model_path,
            actor=actor,
            pad_id=pad,
            eos_id=eos,
        ),
    )
    evaluator = Collector(
        es,
        ws,
        encode,
        eval_tasks,
        config,
        mode="eval",
        grader=grader,
        workflow_instance=collector.workflow,
    )
    plan = learning_plan()
    algo = algorithm_config.GRPOConfig(
        policy_loss_fn="r2orch_director",
        num_generations=config["group_size"],
        beta=config["kl_beta"],
        epsilon=0.2,
        epsilon_high=0.28,
        epsilon_c=None,
        kl_loss_mode="low_var_kl",
        loss_agg_mode="token-mean",
        use_rollout_logps=True,
    )
    algo.temperature = 1.0
    algo.kl_term_cap = config.get("kl_term_cap", 0.0)
    validations = []
    last_saved_optimizer_step = -1

    def metadata():
      return {
          "r2orch_step": program.step,
          "policy_version": program.policy_version,
          "history": program.history,
          "config_hash": invariant_hash,
          "train_data_sha256": data_hash,
          "eval_data_sha256": eval_hash,
          "source_commit": prepare_data.SOURCE_COMMIT,
          "validations": validations,
          "optimizer_step": trainer.train_steps,
          "run_id": run_id,
      }

    def save_boundary():
      nonlocal last_saved_optimizer_step
      state = metadata()
      if trainer.train_steps > last_saved_optimizer_step:
        worker.save_checkpoint(state, step=trainer.train_steps, force=True)
        trainer.checkpoint_manager.wait()
        last_saved_optimizer_step = trainer.train_steps
      # Collection can advance without an optimizer update. Persist its cursor
      # independently instead of attempting to overwrite an immutable snapshot.
      checkpoint.write_manifest(manifest_path, state)

    async def collect(version, step):
      if (
          step
          and step % config["val_every_steps"] == 0
          and not any(v["step"] == step for v in validations)
      ):
        validations.append({"step": step, **await evaluate(evaluator, version)})
      if step and step % config["checkpoint_every_steps"] == 0:
        save_boundary()
      return await collector(version, step)

    program = swarm_program.SwarmRLProgram(
        collect,
        swarm_program.SwarmConfig(
            group_size=config["group_size"],
            micro_batch_size=config["micro_batch_size"],
            max_prompt_length=config["max_prompt_length"],
            max_response_length=config["max_response_length"],
            pad_id=pad,
            eos_id=eos,
            normalization="episode",
            invalid_action_penalty=0.0,
            max_logprob_error=config["max_logprob_error"],
            sampler_is="token",
        ),
        learning_plan=plan,
        credit_builder=lambda episodes: director_credit.build_director_credit(
            episodes,
            plan=plan,
            config=director_credit.DirectorCreditConfig(
                cost_lambda=config["cost_lambda"],
                reference_stride=config.get("kl_reference_stride", 1),
            ),
            reference_step=program.step + 1,
        ),
        algo_config=algo,
        reference_scorer=reference_score,
        num_policy_minibatches=2,
        precompute_sampler_is=True,
    )
    try:
      if resume:
        restored = worker.restore_checkpoint()
        if manifest_path.exists():
          committed = checkpoint.read_manifest(manifest_path)
          if restored.get("run_id") != committed.get("run_id"):
            raise ValueError(
                "Checkpoint run identity does not match the manifest."
            )
          # A durable snapshot includes all state of this one-learner recipe.
          # If publication failed after that save, it is safe to roll forward
          # to its boundary. A newer zero-update cursor lives in the manifest.
          if committed["r2orch_step"] >= restored["r2orch_step"]:
            if committed["optimizer_step"] != trainer.train_steps:
              raise ValueError("Checkpoint optimizer identity does not match.")
            restored = committed
        if (
            restored.get("config_hash") != invariant_hash
            or restored.get("train_data_sha256") != data_hash
            or restored.get("source_commit") != prepare_data.SOURCE_COMMIT
            or (
                mode == "train"
                and restored.get("eval_data_sha256") != eval_hash
            )
        ):
          raise ValueError(
              "Checkpoint configuration or training-data identity does not"
              " match."
          )
        program.restore_state(
            step=restored["r2orch_step"], history=restored["history"]
        )
        engine.version = restored["policy_version"]
        program.policy_version = engine.version
        last_saved_optimizer_step = trainer.train_steps
        validations[:] = restored.get("validations", [])
        run_id = restored.get("run_id", run_id)
        engine.sync()
      else:
        program.policy_version = 0
      director_before = (
          np.asarray(actor.logits[...]).copy() if backend == "cpu" else None
      )
      initial = asyncio.run(evaluate(evaluator, engine.version))
      if mode == "train" and not resume:
        save_boundary()
      if mode == "train" and config["num_steps"] > program.step:
        program.run(engine, num_steps=config["num_steps"] - program.step)
      report = {
          "backend": backend,
          "mode": mode,
          "source_commit": prepare_data.SOURCE_COMMIT,
          "implementation": (
              "original_director_math"
              if config.get("executor") == "source"
              else "director_only_fixture_adaptation"
          ),
          "train_data_sha256": data_hash,
          "eval_data_sha256": eval_hash,
          "history": program.history,
          "validations": validations,
          "initial_eval": initial,
          "final_eval": asyncio.run(
              evaluate(evaluator, program.policy_version)
          ),
          "frozen_policy_version": 0,
          "optimizer_step": trainer.train_steps,
      }
      if backend == "cpu":
        report["frozen_parameters_unchanged"] = (
            frozen_before == _snapshot_digest(frozen)
        )
        report["director_parameter_delta_l2"] = float(
            np.linalg.norm(np.asarray(actor.logits[...]) - director_before)
        )
        np.savez(
            output / "parameters.npz",
            director=np.asarray(actor.logits[...]),
            frozen=np.asarray(frozen.logits[...]),
        )
      if mode == "train":
        save_boundary()
      (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
      (
          output / ("eval_report.json" if mode == "eval" else "report.json")
      ).write_text(json.dumps(report, indent=2) + "\n")
      return report
    finally:
      for sampler in {id(s): s for s in (ds, es, ws)}.values():
        if hasattr(sampler, "close"):
          asyncio.run(sampler.close())
      trainer.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--config")
  parser.add_argument("--train-data")
  parser.add_argument("--eval-data")
  parser.add_argument("--output", required=True)
  parser.add_argument("--model-path")
  parser.add_argument("--reference-source")
  parser.add_argument(
      "--tool-image",
      help="Preinstalled original-tool container, pinned by sha256 digest.",
  )
  parser.add_argument("--checkpoint-dir")
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--mode", choices=("train", "eval"), default="train")
  parser.add_argument("--num-steps", type=int)
  args = parser.parse_args()
  cfg = json.loads(
      Path(
          args.config
          or ROOT
          / "configs"
          / ("cpu_smoke.json" if args.backend == "cpu" else "qwen3_4b.json")
      ).read_text()
  )
  if args.num_steps is not None:
    cfg["num_steps"] = args.num_steps
  if args.tool_image:
    cfg["tool_image"] = args.tool_image
  if args.backend == "qwen" and (not args.train_data or not args.eval_data):
    parser.error(
        "Qwen requires explicit original --train-data and --eval-data paths."
    )
  report = run(
      args.backend,
      cfg,
      load_tasks(args.train_data or ROOT / "data/cpu_train.jsonl"),
      load_tasks(args.eval_data or ROOT / "data/cpu_eval.jsonl"),
      args.output,
      model_path=args.model_path,
      checkpoint_dir=args.checkpoint_dir,
      resume=args.resume,
      mode=args.mode,
      reference_source=args.reference_source,
  )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
