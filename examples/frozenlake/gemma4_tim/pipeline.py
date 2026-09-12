"""One-host default-workload carrier using real Tunix CLI/GRPO components.

Import only after offline input/source/runtime preflight. Native and TIS use
the same stock model and loss code. No existing example or Qwen bundle is
reconfigured by importing this module.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from examples.frozenlake.gemma4_tim import admission, artifacts, observer, recipe
from tunix.cli.grpo_main import GrpoPipeline
from tunix.cli.utils import data as data_lib
from tunix.models.gemma4 import model as gemma_model
from tunix.models.gemma4 import params_safetensors
from tunix.rl import rl_cluster as cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner


class GemmaPipeline(GrpoPipeline):

  def __init__(self, contract: dict, *, snapshot: Path, data: Path, output: Path):
    recipe.validate_resolved(contract)
    self.contract = copy.deepcopy(contract)
    self.snapshot, self.output = snapshot, output
    # The signed resolver replaces unrestricted CLI/env/.env merging. The
    # actual model, optimizer, data, parser and GRPO builders remain shared.
    self.config = recipe.bind_paths(contract["config"], snapshot=snapshot,
                                    data=data, output=output)
    self.data_module = None
    self.batch_observer = observer.BatchObserver(output, contract)

  def create_rl_cluster(self, tokenizer):
    roles = self.create_role_to_mesh()
    actor_mesh = roles[cluster_lib.Role.ACTOR]
    if any(mesh is not actor_mesh for mesh in roles.values()):
      raise recipe.RecipeError("all Gemma roles must share the same mesh object")
    model_cfg = gemma_model.ModelConfig.gemma4_e2b()
    model_cfg.remat_config = gemma_model.RematConfig.DECODER
    model_cfg.use_flash_attention = True
    model_cfg.flash_attention_block_size = 256
    model_cfg.use_sliding_window_kv_cache = False
    model_cfg.dtype = jnp.bfloat16
    reference = params_safetensors.create_model_from_safe_tensors(
        str(self.snapshot), model_cfg, actor_mesh, dtype=jnp.bfloat16,
        text_only=True,
    )
    # Exactly the stock YAML pipeline's actor storage rule: clone the loaded
    # BF16 reference values into FP32 master weights, not a second download.
    graph, state = nnx.split(reference)
    actor = nnx.merge(graph, jax.tree.map(lambda x: x.astype(jnp.float32), state))
    rollout = self.create_rollout_config(role_to_mesh=roles)
    config = self.create_cluster_config(role_to_mesh=roles, rollout_config=rollout)
    cluster = cluster_lib.RLCluster(
        actor=actor, reference=reference, tokenizer=tokenizer,
        cluster_config=config, perf_config=self.create_perf_config(config),
    )
    return cluster

  def _run(self, mode="agentic_grpo"):
    started = time.perf_counter()
    if mode != "agentic_grpo":
      raise recipe.RecipeError("Gemma carrier requires agentic GRPO")
    if self.contract["arm"] == "zero":
      raise recipe.RecipeError("Gemma canonical serving/trainer target path not admitted")
    tokenizer = self._get_tokenizer()
    raw, batch_fn = self._load_raw_dataset(tokenizer)
    self.compute_params(raw)
    artifacts.write_json(self.output / "resolved-runtime-config.json", self.config)
    dataset, _ = data_lib.post_init_dataset(
        raw, tokenizer, batch_size=self.config["batch_size"],
        num_batches=self.config["num_batches"], max_prompt_length=2048,
        fraction=self.config["train_fraction"], num_epochs=self.config["num_train_epochs"],
        prompt_key="prompts", custom_batch_fn=batch_fn,
    )
    cluster = self.create_rl_cluster(tokenizer)
    learner = None
    try:
      learner = GRPOLearner(
          rl_cluster=cluster, algo_config=self._create_agentic_grpo_config(),
          reward_fns=None, chat_parser=self._create_chat_parser(tokenizer),
          agent_class=self._load_class_from_path(self.config["agent_class_path"]),
          agent_kwargs=self.config["agent_kwargs"],
          env_class=self._load_class_from_path(self.config["env_class_path"]),
          env_kwargs=self.config["env_kwargs"],
          processed_batch_observer=self.batch_observer,
      )
      if self.contract["stage"] == "stock-admission":
        return admission.stock_admission(learner, dataset, self.output)
      before = int(cluster.actor_trainer.train_steps)
      train_started = time.perf_counter()
      learner.train(dataset)
      train_seconds = time.perf_counter() - train_started
      after = int(cluster.actor_trainer.train_steps)
      if before != 0 or after != 5 or int(cluster.global_steps) != 5:
        raise recipe.RecipeError("expected five fresh completed optimizer/sync updates")
      if self.batch_observer.rows_by_step != {i: 512 for i in range(5)}:
        raise recipe.RecipeError("batch observer must cover 512 trajectories in each of five updates")
      return {"status": "GREEN", "arm": self.contract["arm"],
              "optimizer_updates": after, "sync_steps": int(cluster.global_steps),
              "batch_records": self.batch_observer.records,
              "startup_seconds": train_started - started,
              "train_wall_seconds": train_seconds,
              "observer_cumulative_seconds": self.batch_observer.wall_seconds,
              "timing_scope": "cold-start five updates including rollout, compilation and observer; not steady-state",
              "claim": "stock-five-update-carrier-not-zero-tim"}
    finally:
      if learner is not None and learner._trajectory_logger is not None:
        learner._trajectory_logger.stop()
      cluster.close()
