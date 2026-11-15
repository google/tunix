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

"""Alternating native Director/Proposer training with a skill-withdrawal loop.

The CPU backend is a small categorical optimizer fixture. The Qwen backend
uses the original AIME task pool, source ADS/TSDS candidate construction and
E5 skill retrieval. It is a small-model math adaptation, not a benchmark claim.
"""

import argparse
import asyncio
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

from absl import flags
from flax import nnx
import jax
import numpy as np
from examples.r2orch_math import cpu_fixture
from examples.r2orch_math import backends
from examples.r2orch_math import grading
from examples.r2orch_math import optimizers
from examples.r2orch_math import prepare_data
from examples.r2orch_math import rsi
from examples.r2orch_math import train
from examples.r2orch_math import workflow
from examples.swarm_math.cpu_fixture import TinyPolicy
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import director_credit
from tunix.dist_rl.swarm import director_loss
from tunix.dist_rl.swarm import local_engine
from tunix.dist_rl.swarm import proposer_loss
from tunix.dist_rl.swarm import runtime
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algorithm_config
from tunix.sft import checkpoint_manager
from tunix.sft import utils as sft_utils

ROOT = Path(__file__).parent


def _digest(value):
  return hashlib.sha256(
      json.dumps(value, sort_keys=True, allow_nan=False).encode()
  ).hexdigest()


def _write(path, value):
  path = Path(path)
  temporary = path.with_name(path.name + ".tmp")
  with temporary.open("w") as file:
    json.dump(value, file, indent=2, sort_keys=True, allow_nan=False)
    file.write("\n")
    file.flush()
    os.fsync(file.fileno())
  os.replace(temporary, path)
  descriptor = os.open(path.parent, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def fixture_tasks():
  """Distinct easy arithmetic questions with reference answers for CPU tests."""
  return [
      workflow.MathTask(
          f"fixture/train/{i}",
          f"Case {i}: What is 1 + {1 + i % 2}?",
          str(2 + i % 2),
      )
      for i in range(12)
  ]


class CandidateCurriculum:
  """Source ADS/TSDS in production; deterministic small-pool CPU fixture."""

  def __init__(
      self,
      tasks,
      config,
      *,
      source=None,
      rows=None,
      tokenizer=None,
      output=None,
  ):
    self.tasks = {t.task_id: t for t in tasks}
    self.config = config
    self.observations = {}
    self.source = source
    if source is None:
      return
    from examples.r2orch_math.source_adapter import load_source_module

    curriculum = load_source_module(source, "curriculum")
    proposer = load_source_module(source, "proposer")
    self.pats = load_source_module(source, "pats")
    source_rows = [row["source"] for row in rows]
    original = Path(output) / "source_train_pool.jsonl"
    original.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    self.pool = curriculum.FixedTaskPool.from_jsonl(
        [original], require_ads_metadata=True
    )
    self.scheduler = curriculum.ADSBoundaryScheduler(
        self.pool,
        active_clusters=2,
        mini_cluster_size=8,
        boundary_eps=0.17,
        alpha=0.3,
        seed=config["seed"],
        cooldown=64,
        cluster_weighting="learnability",
        cluster_floor=0.2,
        cluster_ema=0.3,
        anchor_mode="coverage_random",
        mini_init="spread",
        saturated_min_obs=2,
        hopeless_min_obs=3,
        history_window=48,
        learnable_first=True,
    )
    self.builder = proposer.PairProposer(
        None,
        self.pool,
        self.scheduler,
        curriculum.TSDSRetriever(self.pool),
        tokenizer,
        candidates_per_list=8,
        second_pick="distinct",
    )

  def lists(self, cycle, skill_count, *, skill_tokens=None):
    if self.source is None:
      ids = sorted(self.tasks)
      for index in range(self.config["lists_per_cycle"]):
        seed = np.random.SeedSequence([self.config["seed"], cycle, index])
        chosen = np.random.default_rng(seed).permutation(ids)[:8].tolist()
        prompt = "Choose a useful next training task. Return its letter.\n"
        prompt += "\n".join(
            f"{chr(65 + i)}: {self.tasks[k].problem}"
            for i, k in enumerate(chosen)
        )
        yield chosen, prompt
      return
    anchors = self.scheduler.anchor_ids(
        per_dataset=self.config["lists_per_cycle"],
        cycle=cycle,
        seed=self.config["seed"],
    )
    seeds = self.pool.seeds_for_ids(anchors)
    for seed in seeds:
      item = self.builder.build_candidate_list(
          seed,
          cycle=cycle,
          scope_view={
              self.pats.resolve_scope(
                  entry.task.task_type, entry.task.metadata
              ): {"card_count": skill_count, "card_tokens": skill_tokens}
              for entry in self.pool.tasks.values()
          },
      )
      yield [c["pool_id"] for c in item.candidates], list(item.messages)

  def reserve(self, task_ids):
    if self.source is not None:
      for task_id in task_ids:
        self.scheduler.reserve(task_id)

  def record(self, observations):
    if self.source is not None:
      self.scheduler.record_batch(observations)
    for key, outcomes in observations:
      self.observations[key] = list(outcomes)

  def state(self):
    return {
        "observations": self.observations,
        "scheduler": self.scheduler.state_dict() if self.source else None,
    }

  def restore(self, state):
    self.observations = state["observations"]
    if self.source is not None:
      self.scheduler.load_state_dict(state["scheduler"])


class CycleCheckpoint:
  """Commit both native optimizer states and all outer-loop state together.

  Native Orbax writes finish before CURRENT.json is atomically replaced. Failed
  attempts remain unreferenced directories; resume never selects their latest
  individual model checkpoint. A local filesystem is required for the manifest.
  """

  def __init__(self, directory, identity):
    self.directory = Path(directory)
    self.directory.mkdir(parents=True, exist_ok=True)
    self.identity = identity

  def save(self, trainers, state, ledger, *, fault=None):
    target = self.directory / f"cycle-{state['cycle']:06d}-{uuid.uuid4().hex}"
    target.mkdir()
    steps = {}
    for name, trainer in trainers.items():
      manager = checkpoint_manager.CheckpointManager(str(target / name))
      try:
        manager.save(
            trainer.train_steps,
            trainer.model,
            trainer.optimizer,
            save_only_lora_params=sft_utils.is_lora_enabled(trainer.model),
            force=True,
            custom_metadata={"identity": self.identity},
        )
        manager.wait()
      finally:
        manager.close()
      steps[name] = trainer.train_steps
      if fault:
        fault(f"after_{name}_checkpoint", state["cycle"])
    ledger.save(target / "ledger.json")
    descriptor = os.open(target, os.O_RDONLY)
    try:
      os.fsync(descriptor)
    finally:
      os.close(descriptor)
    complete = {**state, "identity": self.identity, "optimizer_steps": steps}
    _write(target / "state.json", complete)
    pointer = {
        "directory": target.name,
        "state_sha256": _digest(complete),
        "ledger_sha256": (
            hashlib.sha256((target / "ledger.json").read_bytes()).hexdigest()
        ),
    }
    if fault:
      fault("before_commit", state["cycle"])
    _write(self.directory / "CURRENT.json", pointer)

  def restore(self, trainers):
    pointer = json.loads((self.directory / "CURRENT.json").read_text())
    relative = Path(pointer["directory"])
    if relative.is_absolute() or len(relative.parts) != 1:
      raise ValueError("Invalid committed checkpoint path.")
    target = self.directory / relative
    state = json.loads((target / "state.json").read_text())
    if (
        state["identity"] != self.identity
        or _digest(state) != pointer["state_sha256"]
    ):
      raise ValueError("Cycle checkpoint identity or state checksum mismatch.")
    if (
        hashlib.sha256((target / "ledger.json").read_bytes()).hexdigest()
        != pointer["ledger_sha256"]
    ):
      raise ValueError("Cycle ledger checksum mismatch.")
    for name, trainer in trainers.items():
      original = trainer.checkpoint_manager
      manager = checkpoint_manager.CheckpointManager(str(target / name))
      trainer.checkpoint_manager = manager
      try:
        restored = trainer.restore_checkpoint(
            step=state["optimizer_steps"][name]
        )
        if restored.get("identity") != self.identity:
          raise ValueError("Native learner checkpoint identity mismatch.")
      finally:
        manager.close()
        trainer.checkpoint_manager = original
    return state, rsi.SkillLedger.load(target / "ledger.json")


def _learner(model, config, cycles, mesh, backend, *, updates_per_cycle):
  optimizer = optimizers.make_optimizer(
      config, backend, num_optimizer_steps=cycles * updates_per_cycle
  )
  trainer = peft_trainer_v2.PeftTrainer(
      model,
      optimizer,
      peft_trainer_v2.TrainingConfig(
          max_steps=cycles * 2 + 1,
          eval_every_n_steps=cycles * 2 + 2,
          gradient_accumulation_steps=2,
          resume_from_checkpoint_on_init=False,
      ),
      sampler_type="vanilla",
  )
  worker = trainer_worker.TrainerWorker(lambda: trainer, execution_context=mesh)
  worker.initialize()
  return trainer, worker


def _proposer_model(backend, frozen, config, mesh, model_path):
  if backend == "cpu":
    logits = np.zeros((16, 16), dtype=np.float32)
    return TinyPolicy(logits), TinyPolicy(logits.copy()), None
  from transformers import AutoTokenizer

  tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
  model = backends.load_trainable_policy(config, mesh, model_path)
  return model, frozen, tokenizer


def _reviewer_sampler(backend, frozen, worker_sampler, config, tokenizer):
  """Keep the deterministic reviewer separate from the 0.3 Worker sampler."""
  if backend == "cpu":
    return worker_sampler
  from tunix.common import configs

  max_prompt_length = config.get(
      "reviewer_max_prompt_length", config["max_prompt_length"]
  )
  return train.shared_train.NativeMathSampler(
      frozen,
      tokenizer,
      configs.RolloutConfig(
          max_prompt_length=max_prompt_length,
          max_tokens_to_generate=config["reviewer_max_tokens"],
          kv_cache_size=max_prompt_length + config["reviewer_max_tokens"],
          temperature=0.0,
          top_p=None,
          top_k=None,
          eos_tokens=worker_sampler.config.eos_tokens,
          seed=config["seed"] + 100000,
          return_logprobs=True,
      ),
      "frozen-reviewer",
      rollout_model=frozen,
      batch_size=config["rollout_batch_size"],
  )


def _pair_payload(prompt_ids, labels, reference=None, advantages=None):
  shape = (1, len(labels))
  return datatypes.RLTrainerPayload(
      prompt_ids=np.asarray(prompt_ids, np.int32)[None, :],
      prompt_mask=np.ones((1, len(prompt_ids)), np.float32),
      completion_ids=np.asarray(labels, np.int32)[None, :],
      completion_mask=np.ones(shape, np.float32),
      completion_attention_mask=np.ones(shape, np.float32),
      advantages=np.zeros(shape, np.float32)
      if advantages is None
      else advantages,
      ref_per_token_logps=reference,
  )


def _skills_for(ledger, problem, count, embedder=None):
  keys = sorted(
      k
      for k, skill in ledger.skills.items()
      if skill["status"] in ("trial", "active")
  )
  if not keys:
    return []
  if embedder is None:
    return keys[:count]
  query = np.asarray(embedder.encode([problem], query=True)[0])
  vectors = np.asarray(
      embedder.encode([ledger.skills[k]["content"] for k in keys], query=False)
  )
  order = np.argsort(-(vectors @ query), kind="stable")
  return [keys[i] for i in order[:count]]


async def _review(
    ledger, failed_groups, frozen_sampler, encode, cfg, cycle, backend, source
):
  """Distill two-task process evidence using the frozen reviewer, never labels."""
  failed_groups = list(dict(failed_groups).items())
  if len(failed_groups) < 2 or len(ledger.skills) >= cfg["max_skills"]:
    return {"status": "insufficient_evidence", "calls": 0}
  evidence = []
  for index, (task_id, episodes) in enumerate(
      failed_groups[: cfg["max_review_groups"]]
  ):
    evidence.append({
        "id": f"E{index + 1}",
        "task_id": task_id,
        "traces": [{"role": c.role, "text": c.text} for c in episodes[0].calls],
        "correctness": 0,
    })
  if backend == "qwen":
    from examples.r2orch_math.source_adapter import load_source_module

    refiner = load_source_module(source, "pats_refiner")
    system = refiner.review_system_prompt(
        SimpleNamespace(
            min_evidence_cards=2,
            director_prompt_variant="r2orch",
            field_char_limit=300,
        ),
        "DISTILL",
    )
  else:
    system = (
        "Distill a reusable orchestration skill from these unsuccessful process"
        " traces."
    )
  prompt = system + "\nEvidence:\n" + json.dumps(evidence)
  if isinstance(frozen_sampler, train.shared_train.NativeMathSampler):
    frozen_sampler.sampler._rng = jax.random.PRNGKey(
        cfg["seed"] + 100000 + cycle
    )
  reviewer = runtime.RecordingRuntime(
      frozen_sampler,
      (lambda _: [0, 2]) if backend == "cpu" else encode,
      sampler_lib.SamplingParams(
          max_tokens=cfg["reviewer_max_tokens"],
          temperature=1.0 if backend == "cpu" else 0.0,
          top_p=1.0 if backend == "cpu" else None,
          seed=cfg["seed"] + 100000 + cycle,
      ),
      policy_id="worker",
      policy_version=0,
      episode_id=f"review/{cycle}",
      max_prompt_length=cfg["max_prompt_length"],
  )
  if len(reviewer.encode(prompt)) > reviewer.max_prompt_length:
    return {"status": "context_budget", "calls": 0}
  call = await reviewer.generate("skill_reviewer", prompt)
  try:
    payload = (
        json.loads(call.text)
        if backend == "cpu"
        else refiner.parse_review_response(call.text)
    )
    if backend == "cpu":
      operations = [{
          "op": "ADD",
          "card": {
              "name": "Check arithmetic",
              "description": payload["objective"],
              "trigger": "Uncertain solution",
              "plan": payload["objective"],
              "pitfall": "Guessing without verification",
              "constraint": "Use the graph actions",
              "kind": "orchestration",
          },
          "evidence_ids": ["E1", "E2"],
      }]
    else:
      operations = payload["operations"]
    if not isinstance(operations, list) or len(operations) > 2:
      raise ValueError("At most two ADD operations are allowed.")
    added, proposals = [], []
    allowed = {e["id"] for e in evidence}
    for index, operation in enumerate(operations):
      if (
          operation.get("op") != "ADD"
          or len(set(operation.get("evidence_ids", [])) & allowed) < 2
      ):
        raise ValueError("A new skill needs evidence from two distinct tasks.")
      card = operation["card"]
      names = (
          "name",
          "description",
          "trigger",
          "plan",
          "pitfall",
          "constraint",
          "kind",
      )
      if any(
          not isinstance(card.get(n), str)
          or not card[n].strip()
          or len(card[n]) > 300
          for n in names
      ):
        raise ValueError("Invalid bounded skill card.")
      if card["kind"] not in {
          "orchestration",
          "decomposition",
          "communication",
          "verification",
          "tool-use",
          "bug-repair",
      }:
        raise ValueError("Unknown source skill kind.")
      content = json.dumps({k: card[k] for k in names}, sort_keys=True)
      proposals.append((index, content))
    for index, content in proposals:
      if any(item["content"] == content for item in ledger.skills.values()):
        continue
      if len(ledger.skills) >= cfg["max_skills"]:
        break
      added.append(
          ledger.register(
              f"distilled-{cycle}-{index}", version=1, content=content
          )
      )
    return {
        "status": "reviewed",
        "calls": 1,
        "added": added,
        "trace": dataclasses.asdict(call),
    }
  except (ValueError, TypeError, KeyError) as exc:
    return {
        "status": "invalid_review",
        "calls": 1,
        "error": str(exc),
        "trace": dataclasses.asdict(call),
    }


def run(
    backend,
    config,
    tasks,
    evaluation,
    output,
    *,
    model_path=None,
    reference_source=None,
    train_rows=None,
    embedding_model=None,
    resume=False,
    mode="train",
    fault=None,
):
  """Execute real alternating updates; commit or replay whole outer cycles."""
  train.validate_config(config)
  if mode not in ("train", "eval") or (mode == "eval" and not resume):
    raise ValueError("Evaluation requires --resume and a committed cycle.")
  for name in (
      "lists_per_cycle",
      "hidden_siblings",
      "skill_min_tasks",
      "bootstrap_draws",
      "max_skills",
      "max_review_groups",
      "reviewer_max_tokens",
      "retrieve_top_k",
  ):
    if type(config[name]) is not int or config[name] < 1:
      raise ValueError(f"{name} must be a positive integer.")
  for name in (
      "min_proposer_pair_samples",
      "reviewer_max_prompt_length",
      "skill_max_tokens",
      "skill_prompt_token_budget",
      "kl_reference_stride",
  ):
    if name in config and (type(config[name]) is not int or config[name] < 1):
      raise ValueError(f"{name} must be a positive integer.")
  if (
      not math.isfinite(config.get("kl_term_cap", 0.0))
      or config.get("kl_term_cap", 0.0) < 0
  ):
    raise ValueError("kl_term_cap must be finite and nonnegative.")
  for name in (
      "proposer_learning_rate",
      "proposer_kl_beta",
      "proposer_gap_clip",
      "time_exponent",
  ):
    if not math.isfinite(config[name]) or config[name] < 0:
      raise ValueError(f"{name} must be finite and nonnegative.")
  if config["proposer_learning_rate"] == 0 or config["proposer_gap_clip"] == 0:
    raise ValueError("Proposer learning rate and gap clip must be positive.")
  if config["group_size"] != 5 or config["hidden_siblings"] != 3:
    raise ValueError("RSI requires the source three-hidden/two-visible split.")
  if len(tasks) < 8 or config["lists_per_cycle"] < 1:
    raise ValueError(
        "Need at least eight training tasks and one candidate list."
    )
  if {t.task_id for t in tasks} & {t.task_id for t in evaluation} or {
      " ".join(t.problem.split()) for t in tasks
  } & {" ".join(t.problem.split()) for t in evaluation}:
    raise ValueError("Training and evaluation tasks must be disjoint.")
  if backend == "qwen" and (
      not reference_source or not train_rows or not embedding_model
  ):
    raise ValueError(
        "Qwen RSI requires original source/data and a local E5 embedding model."
    )
  if backend == "qwen" and not Path(embedding_model).is_dir():
    raise ValueError("E5 embedding_model must be a local model directory.")
  if backend == "qwen":
    if len(train_rows) != 947 or [row["task_id"] for row in train_rows] != [
        t.task_id for t in tasks
    ]:
      raise ValueError(
          "Qwen RSI requires the original 947-task AIME training pool."
      )
    if any(
        row["problem"] != task.problem or str(row["answer"]) != task.answer
        for row, task in zip(train_rows, tasks)
    ):
      raise ValueError(
          "Training task data must match its source curriculum rows."
      )
    import subprocess

    original_rows = [
        json.loads(line)
        for line in subprocess.check_output([
            "git",
            "-C",
            str(reference_source),
            "show",
            f"{prepare_data.SOURCE_COMMIT}:data/r2orch/train/train_pool.jsonl",
        ]).splitlines()
        if line.strip()
    ]
    expected_rows = [row for row in original_rows if row["dataset"] == "aime"]
    if [row["source"] for row in train_rows] != expected_rows:
      raise ValueError(
          "Curriculum rows differ from the pinned source AIME pool."
      )
  output = Path(output)
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()) and not resume:
    raise ValueError("Use an empty run directory or --resume.")
  if not flags.FLAGS.is_parsed():
    flags.FLAGS(["r2orch_selfplay"])
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils

    pathwaysutils.initialize()
  identity = _digest({
      "config": {k: v for k, v in config.items() if k != "num_steps"},
      "tasks": [dataclasses.asdict(t) for t in tasks],
      "source": prepare_data.SOURCE_COMMIT,
      "embedding_model": str(embedding_model),
      "curriculum_rows": (
          _digest(train_rows) if train_rows is not None else None
      ),
  })
  mesh, _ = train.shared_train.build_meshes(config, backend=backend)
  if 8 % int(mesh.shape["fsdp"]) != 0:
    raise ValueError(
        "The eight independent candidate completions require FSDP size dividing"
        " eight."
    )
  with mesh:
    actor, frozen, ds, es, ws, encode, pad, eos, sync = train._backends(
        backend, config, mesh, model_path
    )
    proposer, proposer_ref, tokenizer = _proposer_model(
        backend, frozen, config, mesh, model_path
    )
    reviewer_sampler = _reviewer_sampler(backend, frozen, ws, config, tokenizer)
    if backend == "cpu":
      # Visible-skill state has a useful prior; gradients still update the real
      # categorical actor. This fixture does not measure language capability.
      values = np.asarray(actor.logits[...]).copy()
      values[5] = values[2]
      values[5, 8], values[5, 9] = np.log(0.8), np.log(0.2)
      actor.logits[...] = values
      sync()
    director_reference = (
        TinyPolicy(np.asarray(actor.logits[...]).copy())
        if backend == "cpu"
        else frozen
    )
    frozen_before = train._snapshot_digest(frozen) if backend == "cpu" else None
    director_trainer, director_worker = _learner(
        actor, config, config["num_steps"], mesh, backend, updates_per_cycle=2
    )
    proposer_trainer, proposer_worker = _learner(
        proposer,
        {
            **config,
            "learning_rate": config["proposer_learning_rate"],
            "optimizer": config.get("proposer_optimizer", {}),
        },
        config["num_steps"],
        mesh,
        backend,
        updates_per_cycle=1,
    )
    trainers = {"director": director_trainer, "proposer": proposer_trainer}
    proposer_config = SimpleNamespace(
        beta=config["proposer_kl_beta"],
        gap_clip=config["proposer_gap_clip"],
        logps_chunk_size=config["logps_chunk_size"],
    )

    def pair_loss(model, train_example):
      return proposer_loss.proposer_pair_loss(
          model, train_example, proposer_config, pad, eos
      )

    proposer_worker.with_loss_fn(pair_loss, has_aux=True)
    proposer_worker.with_gen_model_input_fn(
        lambda payload: {"train_example": payload}
    )
    engine = local_engine.LocalSwarmEngine(
        director_worker,
        sync,
        actor_scorer=lambda payload: director_loss.policy_logprobs(
            actor,
            payload,
            pad_id=pad,
            eos_id=eos,
            chunk_size=config["logps_chunk_size"],
        )[0],
    )
    grader = (
        workflow.grade_aime
        if backend == "cpu"
        else grading.load_original_aime_grader(reference_source)
    )
    graph = train.make_workflow(
        config,
        grader=grader,
        reference_source=reference_source,
        model_path=model_path,
        actor=actor,
        pad_id=pad,
        eos_id=eos,
    )
    curriculum = CandidateCurriculum(
        tasks,
        config,
        source=reference_source if backend == "qwen" else None,
        rows=train_rows,
        tokenizer=tokenizer,
        output=output,
    )
    ledger = rsi.SkillLedger(
        rsi.LedgerConfig(
            min_tasks=config["skill_min_tasks"],
            bootstrap_draws=config["bootstrap_draws"],
        ),
        seed=config["seed"],
    )
    embedder = None
    if backend == "qwen":
      from examples.r2orch_math.source_adapter import load_source_module

      skills_module = load_source_module(reference_source, "skills")
      embedder = skills_module.E5SkillEmbedder(embedding_model)
      import subprocess

      seeds = json.loads(
          subprocess.check_output(
              [
                  "git",
                  "-C",
                  str(reference_source),
                  "show",
                  f"{prepare_data.SOURCE_COMMIT}:configs/skills/seed_cards.json",
              ],
              text=True,
          )
      )
      for item in seeds:
        card = item["card"]
        if "math" in card.get("task_types", []):
          ledger.register(
              card["skill_id"],
              version=1,
              content=json.dumps(card, sort_keys=True),
          )
    else:
      ledger.register(
          "fixture-check",
          version=1,
          content="Check arithmetic before committing a final answer.",
      )
    state = {
        "cycle": 0,
        "history": [],
        "prior": 0.5,
        "prior_cycles": 0,
        "duration_mean": None,
        "director_version": 0,
        "program_step": 0,
        "program_history": [],
        "global_role_step": 0,
        "review_evidence": [],
        "development_tasks": _digest(
            [dataclasses.asdict(t) for t in evaluation]
        ),
    }
    checkpoint = CycleCheckpoint(output / "checkpoints", identity)
    if resume:
      state, ledger = checkpoint.restore(trainers)
      if state["cycle"] > config["num_steps"]:
        raise ValueError("num_steps cannot precede the committed outer cycle.")
      if mode == "train" and state["development_tasks"] != _digest(
          [dataclasses.asdict(t) for t in evaluation]
      ):
        raise ValueError("Training resume cannot change the development split.")
      curriculum.restore(state["curriculum"])
      engine.version = state["director_version"]
      sync()
    else:
      state["curriculum"] = curriculum.state()
      checkpoint.save(trainers, state, ledger)

    async def reference_score(payload):
      values, _, support = director_loss.policy_logprobs(
          director_reference,
          payload,
          pad_id=pad,
          eos_id=eos,
          chunk_size=config["logps_chunk_size"],
      )
      return swarm_program.ReferenceLogprobs(values, support)

    pending_episodes = []

    async def collected(version, step):
      del version, step
      return pending_episodes

    algo = algorithm_config.GRPOConfig(
        num_generations=5,
        beta=config["kl_beta"],
        epsilon=0.2,
        epsilon_high=0.28,
        epsilon_c=None,
        kl_loss_mode="low_var_kl",
        loss_agg_mode="token-mean",
        use_rollout_logps=True,
        policy_loss_fn="r2orch_director",
    )
    algo.temperature = 1.0
    algo.kl_term_cap = config.get("kl_term_cap", 0.0)
    program = swarm_program.SwarmRLProgram(
        collected,
        swarm_program.SwarmConfig(
            group_size=5,
            micro_batch_size=config["micro_batch_size"],
            max_prompt_length=config["max_prompt_length"],
            max_response_length=config["max_response_length"],
            pad_id=pad,
            eos_id=eos,
            max_logprob_error=config["max_logprob_error"],
            sampler_is="token",
        ),
        learning_plan=train.learning_plan(),
        credit_builder=lambda episodes: director_credit.build_director_credit(
            episodes,
            plan=train.learning_plan(),
            config=director_credit.DirectorCreditConfig(
                cost_lambda=config["cost_lambda"],
                reference_stride=config.get("kl_reference_stride", 1),
            ),
            reference_step=state["global_role_step"] + 1,
        ),
        algo_config=algo,
        reference_scorer=reference_score,
        num_policy_minibatches=2,
        precompute_sampler_is=True,
    )
    program.restore_state(
        step=state["program_step"], history=state["program_history"]
    )
    evaluator = train.Collector(
        es,
        ws,
        encode,
        evaluation,
        config,
        mode="eval",
        grader=grader,
        workflow_instance=graph,
    )
    initial_eval = asyncio.run(train.evaluate(evaluator, engine.version))
    if mode == "eval":
      report = {
          "backend": backend,
          "mode": mode,
          "cycles": state["cycle"],
          "zero_skill_eval": initial_eval,
          "director_optimizer_steps": director_trainer.train_steps,
          "proposer_optimizer_steps": proposer_trainer.train_steps,
      }
      evaluation_id = _digest([dataclasses.asdict(t) for t in evaluation])[:12]
      _write(output / f"evaluation-{evaluation_id}.json", report)
      for sampler in {
          id(s): s for s in (ds, es, ws, reviewer_sampler)
      }.values():
        if hasattr(sampler, "close"):
          asyncio.run(sampler.close())
      director_trainer.close()
      proposer_trainer.close()
      return report
    initial_eval = state.get("initial_zero_skill_eval", initial_eval)

    async def cycle_rollouts(cycle, selected):
      # Source-native sampler randomness is reset deterministically at the
      # cycle boundary, independent from development evaluation calls.
      for i, sampler in enumerate((ds, ws)):
        if isinstance(sampler, train.shared_train.NativeMathSampler):
          sampler.sampler._rng = jax.random.PRNGKey(
              config["seed"] + cycle * 17 + i
          )
      groups, timings = {}, {}
      for unique_id, task in selected:
        if backend == "qwen":
          from examples.r2orch_math import skill_review

          retrieved = skill_review.retrieve(
              ledger,
              task.problem,
              embedder=embedder,
              tokenizer=tokenizer,
              source=reference_source,
              top_k=config["retrieve_top_k"],
              prompt_token_budget=config.get("skill_prompt_token_budget", 3072),
          )
        else:
          retrieved = _skills_for(
              ledger, task.problem, config["retrieve_top_k"]
          )
        active_ids = {
            s["skill_id"]
            for s in ledger.skills.values()
            if s["status"] in ("trial", "active")
        }
        audit_pool = sorted(
            k
            for k, skill in ledger.skills.items()
            if skill["status"] == "retired"
            and skill["skill_id"] not in active_ids
        )
        candidate, audit = (
            ledger.choose(
                [*retrieved, *audit_pool], cycle=cycle, task_id=task.task_id
            )
            if retrieved or audit_pool
            else (None, False)
        )
        active = [
            key
            for key in retrieved
            if ledger.skills[key]["status"] in ("trial", "active")
        ]
        if audit and candidate not in active:
          active.append(candidate)
        assignment = rsi.arm_assignment(cycle, task.task_id)
        episodes = []
        start = time.monotonic()
        for sibling, arm in enumerate(assignment):
          shown = [
              k for k in active if not (arm == "hidden" and k == candidate)
          ]
          skills = tuple(ledger.skills[k]["content"] for k in shown)
          if backend == "qwen":
            context = skill_review.render(
                ledger, shown, source=reference_source
            )
            skills = (context,) if context else ()
          episode_id = f"cycle/{cycle}/{unique_id}/{sibling}"
          seed = int(
              np.random.SeedSequence(
                  [config["seed"], cycle, len(groups), sibling]
              ).generate_state(1)[0]
          )

          def director_encode(prompt, visible=bool(skills)):
            ids = list(encode(prompt))
            if backend == "cpu" and visible and ids[-1] == 2:
              ids[-1] = 5
            return ids

          director = runtime.RecordingRuntime(
              ds,
              director_encode,
              sampler_lib.SamplingParams(
                  max_tokens=config["max_response_length"],
                  temperature=1.0,
                  top_p=1.0,
                  seed=seed,
              ),
              policy_id="director",
              policy_version=engine.version,
              episode_id=episode_id + "/director",
              max_prompt_length=config["max_prompt_length"],
          )
          worker_runtime = runtime.RecordingRuntime(
              ws,
              encode,
              sampler_lib.SamplingParams(
                  max_tokens=config["worker_max_tokens"],
                  temperature=config["worker_temperature"],
                  top_p=1.0,
                  seed=seed,
              ),
              policy_id="worker",
              policy_version=0,
              episode_id=episode_id + "/worker",
              max_prompt_length=config.get(
                  "worker_max_prompt_length", config["max_prompt_length"]
              ),
          )
          result = await graph.run(
              dataclasses.replace(task, task_id=unique_id),
              director,
              worker_runtime,
              episode_id=episode_id,
              skills=skills,
              scaffold_condition=arm if candidate else "none",
          )
          episodes.append(result.episode)
        duration = (
            sum(sum(len(c.completion_ids) for c in e.calls) for e in episodes)
            if backend == "cpu"
            else time.monotonic() - start
        )
        groups[unique_id] = (task, candidate, episodes, assignment)
        timings[unique_id] = max(float(duration), 1e-6)
      return groups, timings

    try:
      for cycle in range(state["cycle"], config["num_steps"]):
        skill_tokens = None
        if backend == "qwen":
          from examples.r2orch_math import skill_review

          skill_tokens = len(
              tokenizer.encode(
                  curriculum.pats.render_cards(
                      skill_review._active_records(ledger)
                  ),
                  add_special_tokens=False,
              )
          )
        lists = curriculum.lists(
            cycle,
            sum(
                s["status"] in ("trial", "active")
                for s in ledger.skills.values()
            ),
            skill_tokens=skill_tokens,
        )
        selections, selected, payloads = [], [], []
        for index, (ids, prompt) in enumerate(lists):
          if backend == "cpu":
            prompt_ids, labels = [0, 1], list(range(2, 2 + len(ids)))
          else:
            messages = (
                prompt
                if isinstance(prompt, list)
                else [{"role": "user", "content": prompt}]
            )
            prompt_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            labels = [
                curriculum.builder.label_token_ids[chr(65 + i)]
                for i in range(len(ids))
            ]
          if len(prompt_ids) > config["max_prompt_length"]:
            raise ValueError(
                "Proposer candidate prompt exceeds its token budget."
            )
          payload = _pair_payload(prompt_ids, labels)
          current = np.asarray(
              proposer_loss.candidate_logps(
                  proposer,
                  payload,
                  pad_id=pad,
                  eos_id=eos,
                  chunk_size=config["logps_chunk_size"],
              )
          )[0]
          reference = np.asarray(
              proposer_loss.candidate_logps(
                  proposer_ref,
                  payload,
                  pad_id=pad,
                  eos_id=eos,
                  chunk_size=config["logps_chunk_size"],
              )
          )
          pair = rsi.sample_candidate_pair(
              current, seed=config["seed"] + cycle * 101 + index
          )
          selected_ids = [ids[i] for i in pair.indices]
          curriculum.reserve(selected_ids)
          group_ids = [f"list-{index}/{task_id}" for task_id in selected_ids]
          selected.extend(
              (key, curriculum.tasks[task_id])
              for key, task_id in zip(group_ids, selected_ids)
          )
          selections.append({
              "candidate_ids": ids,
              "selected_ids": selected_ids,
              "group_ids": group_ids,
              "indices": list(pair.indices),
              "behavior_logprobs": list(pair.logprobs),
              "label_logprobs": current.tolist(),
          })
          payloads.append(
              dataclasses.replace(payload, ref_per_token_logps=reference)
          )
        groups, timings = asyncio.run(cycle_rollouts(cycle, selected))
        pending_episodes[:] = [
            episode
            for _, _, episodes, _ in groups.values()
            for episode in episodes
        ]
        rewards, hidden_all, observations, failures = {}, [], [], []
        evidence_by_task = {}
        for key, (task, candidate, episodes, assignment) in groups.items():
          hidden = [
              e.reward
              for e, arm in zip(episodes, assignment)
              if arm == "hidden"
          ]
          visible = [
              e.reward
              for e, arm in zip(episodes, assignment)
              if arm == "visible"
          ]
          factor = (
              (state["duration_mean"] or timings[key]) / timings[key]
          ) ** config["time_exponent"]
          rewards[key] = rsi.recursive_learnability(
              hidden if candidate else [],
              visible if candidate else [],
              siblings=[e.reward for e in episodes],
              prior=state["prior"],
              time_factor=factor,
          )
          observations.append((task.task_id, [e.reward for e in episodes]))
          if candidate:
            hidden_all.extend(hidden)
            pair_evidence = evidence_by_task.setdefault(
                (task.task_id, candidate), [[], []]
            )
            pair_evidence[0].extend(hidden)
            pair_evidence[1].extend(visible)
          if not any(e.reward for e in episodes):
            failures.append((task.task_id, episodes))
        for (task_id, candidate), (hidden, visible) in evidence_by_task.items():
          ledger.record(
              cycle=cycle,
              task_id=task_id,
              skill_key=candidate,
              hidden=hidden,
              visible=visible,
          )
        pairs = [
            [rewards[key] for key in item["group_ids"]] for item in selections
        ]
        advantages = rsi.proposer_pair_advantages(pairs)
        proposer_updated = len(pairs) >= config.get(
            "min_proposer_pair_samples", 1 if backend == "cpu" else 4
        )
        if proposer_updated:
          for payload, item, advantage in zip(payloads, selections, advantages):
            signed = np.zeros_like(payload.advantages)
            signed[0, item["indices"]] = advantage
            payload = dataclasses.replace(payload, advantages=signed)
            proposer_loss.validate_payload(payload)
            proposer_worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
          proposer_worker.update()
          state["global_role_step"] += 1
        if fault:
          fault("after_proposer_update", cycle + 1)
        director_steps_before = director_trainer.train_steps
        program.run(engine, num_steps=1)
        if director_trainer.train_steps > director_steps_before:
          state["global_role_step"] += 1
        if fault:
          fault("after_director_update", cycle + 1)
        curriculum.record(observations)
        decisions = ledger.decide(cycle=cycle)
        if backend == "qwen":
          from examples.r2orch_math import skill_review

          review, review_evidence = asyncio.run(
              skill_review.review(
                  ledger,
                  [
                      (task.task_id, episodes)
                      for task, _, episodes, _ in groups.values()
                  ],
                  state["review_evidence"],
                  frozen_sampler=reviewer_sampler,
                  tokenizer=tokenizer,
                  config=config,
                  cycle=cycle,
                  policy_version=next(
                      c.policy_version
                      for e in pending_episodes
                      for c in e.calls
                      if c.policy_id == "director"
                  ),
                  source=reference_source,
              )
          )
        else:
          review_evidence = []
          review = asyncio.run(
              _review(
                  ledger,
                  failures,
                  reviewer_sampler,
                  encode,
                  config,
                  cycle,
                  backend,
                  reference_source,
              )
          )
        metrics = {
            "cycle": cycle + 1,
            "selections": selections,
            "proposer_rewards": pairs,
            "proposer_advantages": advantages.tolist(),
            "accuracy": float(np.mean([e.reward for e in pending_episodes])),
            "director_optimizer_steps": director_trainer.train_steps,
            "proposer_optimizer_steps": proposer_trainer.train_steps,
            "director_version": engine.version,
            "proposer_updated": proposer_updated,
            "global_role_step": state["global_role_step"],
            "skill_decisions": decisions,
            "review": review,
            "skill_count": len(ledger.skills),
        }
        if (cycle + 1) % config["val_every_steps"] == 0:
          metrics["zero_skill_eval"] = asyncio.run(
              train.evaluate(evaluator, engine.version)
          )
        state = {
            "cycle": cycle + 1,
            "initial_zero_skill_eval": initial_eval,
            "history": [*state["history"], metrics],
            "prior": (
                (
                    float(np.mean(hidden_all))
                    if not state["prior_cycles"]
                    else 0.9 * state["prior"] + 0.1 * float(np.mean(hidden_all))
                )
                if hidden_all
                else state["prior"]
            ),
            "prior_cycles": state["prior_cycles"] + int(bool(hidden_all)),
            "duration_mean": (
                float(np.mean(list(timings.values())))
                if state["duration_mean"] is None
                else 0.9 * state["duration_mean"]
                + 0.1 * float(np.mean(list(timings.values())))
            ),
            "director_version": engine.version,
            "program_step": program.step,
            "global_role_step": state["global_role_step"],
            "program_history": program.history,
            "review_evidence": review_evidence,
            "development_tasks": state["development_tasks"],
            "curriculum": curriculum.state(),
        }
        _write(
            output / f"cycle-{cycle:06d}-rollouts.json",
            [dataclasses.asdict(e) for e in pending_episodes],
        )
        checkpoint.save(trainers, state, ledger, fault=fault)
        _write(output / "progress.json", state)
      report = {
          "backend": backend,
          "implementation": "r2orch_math_selfplay_adaptation",
          "source_commit": prepare_data.SOURCE_COMMIT,
          "cycles": state["cycle"],
          "history": state["history"],
          "initial_zero_skill_eval": initial_eval,
          "final_zero_skill_eval": asyncio.run(
              train.evaluate(evaluator, engine.version)
          ),
          "director_optimizer_steps": director_trainer.train_steps,
          "proposer_optimizer_steps": proposer_trainer.train_steps,
      }
      if backend == "cpu":
        report["frozen_parameters_unchanged"] = (
            frozen_before == train._snapshot_digest(frozen)
        )
        np.savez(
            output / "parameters.npz",
            director=np.asarray(actor.logits[...]),
            proposer=np.asarray(proposer.logits[...]),
            frozen=np.asarray(frozen.logits[...]),
        )
      _write(output / "selfplay_report.json", report)
      return report
    finally:
      for sampler in {
          id(s): s for s in (ds, es, ws, reviewer_sampler)
      }.values():
        if hasattr(sampler, "close"):
          asyncio.run(sampler.close())
      director_trainer.close()
      proposer_trainer.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--backend", choices=("cpu", "qwen"), default="cpu")
  parser.add_argument("--config")
  parser.add_argument("--output", required=True)
  parser.add_argument("--model-path")
  parser.add_argument("--reference-source")
  parser.add_argument(
      "--tool-image",
      help="Digest-pinned image for the original isolated AIME tools.",
  )
  parser.add_argument("--embedding-model")
  parser.add_argument("--train-data")
  parser.add_argument("--eval-data")
  parser.add_argument("--num-steps", type=int)
  parser.add_argument("--resume", action="store_true")
  parser.add_argument("--mode", choices=("train", "eval"), default="train")
  args = parser.parse_args()
  path = (
      args.config
      or ROOT
      / "configs"
      / f"selfplay_{'cpu' if args.backend == 'cpu' else 'qwen3_4b'}.json"
  )
  config = json.loads(Path(path).read_text())
  if args.tool_image:
    config["tool_image"] = args.tool_image
  if args.num_steps is not None:
    config["num_steps"] = args.num_steps
  if args.backend == "qwen" and (not args.train_data or not args.eval_data):
    parser.error(
        "Qwen requires original --train-data and held-out --eval-data."
    )
  tasks = (
      train.load_tasks(args.train_data) if args.train_data else fixture_tasks()
  )
  evaluation = train.load_tasks(args.eval_data or ROOT / "data/cpu_eval.jsonl")
  rows = (
      [
          json.loads(line)
          for line in Path(args.train_data).read_text().splitlines()
          if line.strip()
      ]
      if args.train_data
      else None
  )
  report = run(
      args.backend,
      config,
      tasks,
      evaluation,
      args.output,
      model_path=args.model_path,
      reference_source=args.reference_source,
      train_rows=rows,
      embedding_model=args.embedding_model,
      resume=args.resume,
      mode=args.mode,
  )
  print(
      json.dumps(
          {
              k: report[k]
              for k in (
                  "cycles",
                  "director_optimizer_steps",
                  "proposer_optimizer_steps",
              )
          },
          indent=2,
      )
  )


if __name__ == "__main__":
  main()
