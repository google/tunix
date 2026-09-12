"""Full default-batch stock rollout admission, without a training conversion."""

import asyncio
import time

import numpy as np

from examples.frozenlake.gemma4_tim import artifacts, recipe


def stock_admission(learner, dataset, output):
  first_batch = next(iter(dataset))
  prompts = list(learner._create_micro_batch_iterator(iter([first_batch]), 1))
  if len(prompts) != 64 or learner._num_generations() != 8:
    raise recipe.RecipeError("stock admission requires the full default B64/G8")
  cluster = learner.rl_cluster
  before = (int(cluster.actor_trainer.train_steps), int(cluster.global_steps))
  if before != (0, 0):
    raise recipe.RecipeError("stock admission requires a fresh policy")
  learner._full_batch_size = 64
  orchestrator = learner._build_orchestrator()
  started = time.perf_counter()

  async def collect():
    seen = set()
    records = []
    async for group in learner._orchestrator_producer(orchestrator, prompts, num_generations=8):
      if (len(group) != 8 or len({int(item.group_id) for item in group}) != 1
          or {int(item.pair_index) for item in group} != set(range(8))):
        raise recipe.RecipeError("stock admission lost a generation group")
      payload = {}
      for item in group:
        key = (int(item.group_id), int(item.pair_index))
        if key[0] not in range(64):
          raise recipe.RecipeError("admission group cannot join its prompt")
        if key in seen:
          raise recipe.RecipeError("duplicate admission trajectory")
        seen.add(key)
        traj = item.traj
        tokens = np.asarray(traj["conversation_tokens"])
        masks = np.asarray(traj["conversation_masks"])
        logps = np.asarray(traj["old_logprobs"])
        if (tokens.ndim != 1 or not np.issubdtype(tokens.dtype, np.integer)
            or masks.shape != tokens.shape or logps.shape != tokens.shape
            or not np.isin(masks, [0, 1]).all()
            or not masks.astype(bool).any() or not np.isfinite(logps[masks.astype(bool)]).all()
            or int(traj["policy_version"]) != 0 or not np.isfinite(traj["trajectory_reward"])):
          raise recipe.RecipeError("invalid sampled admission trajectory")
        for name, value in (("tokens", tokens), ("actions", masks.astype(bool)), ("A", logps)):
          payload[f"generation_{key[1]}_{name}"] = value
        prompt = np.asarray(traj["prompt_tokens"])
        prompt_length = int(traj["prompt_length"])
        if (prompt.ndim != 1 or not np.issubdtype(prompt.dtype, np.integer)
            or not 0 < prompt_length <= prompt.size):
          raise recipe.RecipeError("missing semantic prompt identity")
        payload[f"generation_{key[1]}_prompt"] = prompt
        payload[f"generation_{key[1]}_prompt_length"] = np.int32(prompt_length)
        records.append({"group": key[0], "generation": key[1],
                        "tokens": int(tokens.size), "actions": int(np.count_nonzero(masks)),
                        "reward": float(traj["trajectory_reward"]), "policy_version": 0})
      path = output / f"admission-group-{int(group[0].group_id):03d}.npz"
      with path.open("xb") as stream:
        path.chmod(0o600)
        np.savez_compressed(stream, **payload)
    if len(records) != 512 or len({r["group"] for r in records}) != 64:
      raise recipe.RecipeError("stock admission requires 512/512 trajectories")
    return records

  records = asyncio.run_coroutine_threadsafe(collect(), learner.loop).result()
  after = (int(cluster.actor_trainer.train_steps), int(cluster.global_steps))
  if after != before:
    raise recipe.RecipeError("rollout-only admission advanced optimizer or policy")
  result = {"status": "GREEN", "stage": "stock-admission", "prompts": 64,
            "generations": 8, "trajectories": len(records), "optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - started,
            "records": records, "claim": "stock-rollout-only-not-zero-tim"}
  artifacts.write_json(output / "stock-admission.json", result)
  return result
