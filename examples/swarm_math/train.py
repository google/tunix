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

"""Local transport entry point for shared Qwen3-4B or a tiny CPU fixture.

Run from the repository root with `python -m examples.swarm_math.train`.
The CPU backend exercises real native training without downloading weights.
The Qwen backend requires a local HF snapshot and accelerator capacity.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
from pathlib import Path

from absl import flags
from flax import nnx
import jax
import numpy as np
import optax
from examples.swarm_math import cpu_fixture
from examples.swarm_math import workflow
from tunix.common import configs
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import runtime
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker

ROOT = Path(__file__).parent


def load_tasks(path):
  tasks = [
      workflow.MathTask(**json.loads(line))
      for line in Path(path).read_text().splitlines()
      if line.strip()
  ]
  if not tasks or len({t.task_id for t in tasks}) != len(tasks):
    raise ValueError("Task data must be nonempty with unique task_id values.")
  if any(
      not all(
          isinstance(x, str) and x.strip()
          for x in (t.task_id, t.problem, t.answer)
      )
      for t in tasks
  ):
    raise ValueError(
        "Each task needs nonempty string task_id, problem and answer."
    )
  return tasks


def validate_config(config):
  if not config["model_sharing"] or config["enable_thinking"]:
    raise ValueError(
        "This recipe requires shared weights and enable_thinking=false."
    )
  if config["train_temperature"] != 1.0 or config["train_top_p"] != 1.0:
    raise ValueError(
        "V0 training requires temperature=1 and full-distribution sampling."
    )
  if (
      min(
          config[k]
          for k in (
              "tasks_per_step",
              "num_steps",
              "max_solver_turns",
              "eval_attempts",
          )
      )
      < 1
  ):
    raise ValueError("Task, step, turn and evaluation counts must be positive.")
  if config["learning_rate"] <= 0 or not np.isfinite(config["learning_rate"]):
    raise ValueError("learning_rate must be finite and positive.")
  if config["eval_temperature"] <= 0 or not 0 < config["eval_top_p"] <= 1:
    raise ValueError("Invalid evaluation sampling configuration.")


class MathCollector:
  """Serial local transport; independent episodes may later run on workers."""

  def __init__(self, sampler, encode, tasks, config, *, trace_path=None):
    self.sampler, self.encode, self.tasks, self.config = (
        sampler,
        encode,
        tasks,
        config,
    )
    self.trace_path = trace_path
    self.workflow = workflow.SolverVerifierWorkflow(
        max_solver_turns=config["max_solver_turns"]
    )

  async def episodes(
      self, tasks, version, step, *, evaluate=False, max_solver_turns=None
  ):
    config = self.config
    attempts = config["eval_attempts"] if evaluate else config["group_size"]
    stage = "eval" if evaluate else "train"
    flow = (
        self.workflow
        if max_solver_turns is None
        else workflow.SolverVerifierWorkflow(max_solver_turns=max_solver_turns)
    )
    episodes = []
    for index, task in enumerate(tasks):
      for attempt in range(attempts):
        seed = int(
            np.random.SeedSequence(
                [config["seed"], step, index, attempt, int(evaluate)]
            ).generate_state(1)[0]
        )
        rt = runtime.RecordingRuntime(
            self.sampler,
            self.encode,
            sampler_lib.SamplingParams(
                max_tokens=config["max_response_length"],
                seed=seed,
                temperature=config[f"{stage}_temperature"],
                top_p=config[f"{stage}_top_p"],
            ),
            policy_id=config["policy_id"],
            policy_version=version,
            episode_id=f"{stage}/{step}/{task.task_id}/{attempt}",
            max_prompt_length=config["max_prompt_length"],
        )
        episodes.append(await flow.run(task, rt))
    return episodes

  async def __call__(self, version, step):
    size = self.config["tasks_per_step"]
    # Deterministic epoch permutation; do not duplicate a task within a group.
    order = np.random.default_rng(
        self.config["seed"] + step // max(1, len(self.tasks) // size)
    ).permutation(len(self.tasks))
    start = (step * size) % len(self.tasks)
    indices = [
        order[(start + offset) % len(self.tasks)] for offset in range(size)
    ]
    episodes = await self.episodes(
        [self.tasks[i] for i in indices], version, step
    )
    if self.trace_path is not None:
      with Path(self.trace_path).open("a") as stream:
        for episode in episodes:
          stream.write(json.dumps(dataclasses.asdict(episode)) + "\n")
    return episodes


class NativeMathSampler:
  """Keep native static decode settings explicit and evaluation seeds stable.

  VanillaSampler uses its configured random stream, not SamplingParams.seed.
  Reconstructing only the eval sampler resets that stream for paired evaluation.
  No per-call parameter mutation is hidden behind a compiled sampling function.
  """

  def __init__(self, actor, tokenizer, rollout_config, stage):
    self.actor, self.tokenizer, self.config, self.stage = (
        actor,
        tokenizer,
        rollout_config,
        stage,
    )
    self.reset()

  def reset(self):
    from tunix.dist_rl.rollout import vanilla_sampler  # pylint: disable=g-import-not-at-top

    rollout = nnx.clone(self.actor)
    # nnx.clone clones graph objects but shares JAX buffers. Native optimizer
    # JITs donate actor buffers; a serving snapshot needs independent storage.
    nnx.update(rollout, jax.tree.map(lambda x: x.copy(), nnx.state(self.actor)))
    self.sampler = vanilla_sampler.VanillaSampler(
        server_id=f"swarm-{self.stage}",
        transformer=rollout,
        tokenizer=self.tokenizer,
        config=self.config,
    )

  async def sample(self, request):
    params = request.sampling_params
    if (
        params.temperature != self.config.temperature
        or params.top_p != self.config.top_p
    ):
      raise ValueError(
          "Requested sampling distribution differs from the native static"
          " config."
      )
    return await self.sampler.sample(
        request,
        max_prompt_length=self.config.max_prompt_length,
        include_eos=True,
    )

  def sync(self):
    # Some CPU leaves start SingleDeviceSharding and become NamedSharding in
    # the optimizer JIT. Preserve each serving leaf's placement explicitly.
    copied = jax.tree.map(
        lambda source, target: jax.device_put(
            source, target.sharding, may_alias=False
        ),
        nnx.state(self.actor, nnx.Param),
        self.sampler.transformer_state,
    )
    self.sampler.transformer_state = copied


def create_backends(backend, config, mesh, model_path=None):
  if backend == "cpu":
    if config["model_id"] != "tiny-categorical-fixture":
      raise ValueError("Use cpu_smoke.json with the CPU fixture.")
    actor = cpu_fixture.TinyPolicy()
    rollout = nnx.clone(actor)

    def sync_cpu():
      nnx.update(rollout, jax.tree.map(lambda x: x.copy(), nnx.state(actor)))

    sync_cpu()
    return (
        actor,
        cpu_fixture.TinySampler(rollout),
        cpu_fixture.TinySampler(rollout),
        cpu_fixture.encode,
        0,
        0,
        sync_cpu,
    )
  if (
      config["model_id"] != "Qwen/Qwen3-4B"
      or not model_path
      or not Path(model_path).is_dir()
  ):
    raise ValueError(
        "Qwen requires model_id=Qwen/Qwen3-4B and a local --model-path HF"
        " snapshot."
    )
  if Path(model_path).resolve().name != config["model_revision"]:
    raise ValueError(
        "Use the HF cache snapshot directory for the pinned model_revision (see"
        " README)."
    )
  # Imports are lazy: the CPU fixture does not need model downloads or KV cache.
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
  actor, _ = automodel.AutoModel.from_pretrained(
      config["model_id"], mesh, model_path=model_path
  )
  samplers = []
  for stage in ("train", "eval"):
    samplers.append(
        NativeMathSampler(
            actor=actor,
            tokenizer=tokenizer,
            stage=stage,
            rollout_config=configs.RolloutConfig(
                max_prompt_length=config["max_prompt_length"],
                max_tokens_to_generate=config["max_response_length"],
                kv_cache_size=config["max_prompt_length"]
                + config["max_response_length"],
                temperature=config[f"{stage}_temperature"],
                top_p=config[f"{stage}_top_p"],
                top_k=None,
                seed=config["seed"],
                return_logprobs=True,
            ),
        )
    )

  def sync():
    for sampler in samplers:
      sampler.sync()

  return (
      actor,
      *samplers,
      workflow.chat_encoder(tokenizer, config["max_prompt_length"]),
      tokenizer.pad_token_id,
      tokenizer.eos_token_id,
      sync,
  )


async def evaluate(collector, version, *, single_solver=False):
  if isinstance(collector.sampler, NativeMathSampler):
    collector.sampler.reset()
  episodes = await collector.episodes(
      collector.tasks,
      version,
      0,
      evaluate=True,
      max_solver_turns=1 if single_solver else None,
  )
  return {
      "accuracy": float(np.mean([e.reward for e in episodes])),
      "attempts": len(episodes),
      "mean_calls": float(np.mean([len(e.calls) for e in episodes])),
      "mean_generated_tokens": float(
          np.mean(
              [sum(len(c.completion_ids) for c in e.calls) for e in episodes]
          )
      ),
  }


def run(backend, config, train_tasks, eval_tasks, output, *, model_path=None):
  validate_config(config)
  if config["tasks_per_step"] > len(train_tasks):
    raise ValueError("tasks_per_step exceeds the number of training tasks.")
  if {t.task_id for t in train_tasks} & {t.task_id for t in eval_tasks} or {
      t.problem.strip() for t in train_tasks
  } & {t.problem.strip() for t in eval_tasks}:
    raise ValueError("Training and evaluation tasks must be disjoint.")
  output = Path(output).resolve()
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()):
    raise ValueError(
        "Use an empty output directory; existing run artifacts are preserved."
    )
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["swarm_math"])
  mesh = jax.sharding.Mesh(
      np.asarray(jax.devices()).reshape(-1, 1), ("fsdp", "tp")
  )
  with mesh:
    actor, sampler, eval_sampler, encode, pad_id, eos_id, sync = (
        create_backends(backend, config, mesh, model_path)
    )
    before = np.asarray(actor.logits[...]).copy() if backend == "cpu" else None
    optimizer = (
        optax.sgd(config["learning_rate"])
        if backend == "cpu"
        else optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                config["learning_rate"], b1=0.9, b2=0.999, weight_decay=0.01
            ),
        )
    )
    trainer = peft_trainer_v2.PeftTrainer(
        actor,
        optimizer,
        peft_trainer_v2.TrainingConfig(
            eval_every_n_steps=config["num_steps"] + 1,
            max_steps=config["num_steps"],
            gradient_accumulation_steps=2,
            checkpoint_root_directory=str(output / "checkpoints"),
            resume_from_checkpoint_on_init=False,
        ),
        sampler_type="vanilla",
    )
    worker = trainer_worker.TrainerWorker(
        lambda: trainer, execution_context=mesh
    )
    worker.initialize()
    engine = local_engine.LocalSwarmEngine(worker, sync)
    collector = MathCollector(
        sampler,
        encode,
        train_tasks,
        config,
        trace_path=output / "train_traces.jsonl",
    )
    eval_collector = MathCollector(eval_sampler, encode, eval_tasks, config)
    program = swarm_program.SwarmRLProgram(
        collector,
        swarm_program.SwarmConfig(
            **{
                key: config[key]
                for key in (
                    "group_size",
                    "micro_batch_size",
                    "max_prompt_length",
                    "max_response_length",
                    "normalization",
                    "invalid_action_penalty",
                )
            },
            pad_id=pad_id,
            eos_id=eos_id,
        ),
    )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {
        "backend": backend,
        "model_path": str(model_path) if model_path else None,
        "reference_commit": "c2f724e7358e7dc5bf2ea792ab07a3b5d90290c9",
        "train_data_sha256": (
            hashlib.sha256(
                json.dumps(
                    [dataclasses.asdict(t) for t in train_tasks], sort_keys=True
                ).encode()
            ).hexdigest()
        ),
        "eval_data_sha256": (
            hashlib.sha256(
                json.dumps(
                    [dataclasses.asdict(t) for t in eval_tasks], sort_keys=True
                ).encode()
            ).hexdigest()
        ),
    }
    try:
      initial_eval = asyncio.run(evaluate(eval_collector, 0))
      program.run(engine, num_steps=config["num_steps"])
      report = {
          **manifest,
          "history": program.history,
          "initial_team_eval": initial_eval,
          "final_team_eval": asyncio.run(
              evaluate(eval_collector, program.policy_version)
          ),
          "final_single_solver_eval": asyncio.run(
              evaluate(
                  eval_collector, program.policy_version, single_solver=True
              )
          ),
      }
      if before is not None:
        after = np.asarray(actor.logits[...])
        report["parameter_delta_l2"] = float(np.linalg.norm(after - before))
        np.savez(output / "parameters.npz", initial=before, final=after)
      worker.save_checkpoint(
          {"swarm_policy_version": program.policy_version, "recipe": config},
          step=program.step,
          force=True,
      )
      (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
      return report
    finally:
      trainer.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--config", type=Path)
  parser.add_argument("--train-data", type=Path)
  parser.add_argument("--eval-data", type=Path)
  parser.add_argument("--model-path", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--normalization", choices=("role", "episode"))
  args = parser.parse_args()
  if args.backend == "qwen" and (
      args.train_data is None or args.eval_data is None
  ):
    parser.error(
        "Qwen requires explicit disjoint --train-data and --eval-data JSONL"
        " files."
    )
  config_path = args.config or ROOT / "configs" / (
      "cpu_smoke.json" if args.backend == "cpu" else "qwen3_4b.json"
  )
  config = json.loads(config_path.read_text())
  if args.normalization:
    config["normalization"] = args.normalization
  report = run(
      args.backend,
      config,
      load_tasks(args.train_data or ROOT / "data/cpu_train.jsonl"),
      load_tasks(args.eval_data or ROOT / "data/cpu_eval.jsonl"),
      args.output,
      model_path=args.model_path,
  )
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
