"""Execute pinned upstream source excerpts without importing Ray/verl."""

import collections
import json
from pathlib import Path
import re
import textwrap
import types

import numpy as np
import torch
from examples.swarm_math import drmas_math

FIXTURE = json.loads(
    (Path(__file__).parent / "reference/drmas.json").read_text()
)


def namespace():
  env = {
      "torch": torch,
      "np": np,
      "re": re,
      "defaultdict": collections.defaultdict,
      "extract_answer": drmas_math.extract_answer,
  }
  for source in FIXTURE["files"].values():
    for symbol, code in source["symbols"].items():
      exec("from __future__ import annotations\n" + textwrap.dedent(code), env)  # pylint: disable=exec-used
  env["verl_F"] = types.SimpleNamespace(
      masked_mean=env["masked_mean"], masked_sum=env["masked_sum"]
  )
  return env


def advantages(episodes, *, role=True, penalty=0.1):
  rows = [(e, c) for e in episodes for c in e.calls]
  width = max(len(c.completion_ids) for _, c in rows)
  rewards = torch.zeros((len(rows), width), dtype=torch.float32)
  mask = torch.zeros_like(rewards)
  indices, trajectories = [], []
  for i, (episode, call) in enumerate(rows):
    mask[i, : len(call.loss_mask)] = torch.tensor(call.loss_mask)
    last = max(j for j, value in enumerate(call.loss_mask) if value)
    rewards[i, last] = episode.reward
    rewards[i, last] -= penalty * (not call.valid_action)
    indices.append(
        f"{episode.task_id}/{call.role}" if role else episode.task_id
    )
    trajectories.append(episode.episode_id)
  return namespace()["compute_grpo_outcome_advantage"](
      rewards,
      mask,
      np.asarray(indices, dtype=object),
      np.asarray(trajectories, dtype=object),
      group_by_agent_id=role,
  )[0].numpy()


def loss_and_gradient(logits, payload):
  """Source PPO loss through independent PyTorch autoregressive scoring."""
  weights = torch.tensor(np.asarray(logits), requires_grad=True)
  prompts = np.asarray(payload.prompt_ids)
  completions = np.asarray(payload.completion_ids)
  previous = np.concatenate([prompts[:, -1:], completions[:, :-1]], axis=1)
  logps = torch.log_softmax(
      weights[torch.tensor(previous.astype(np.int64))], dim=-1
  )
  logps = logps.gather(
      -1, torch.tensor(completions.astype(np.int64))[..., None]
  ).squeeze(-1)
  loss = namespace()["compute_policy_loss"](
      torch.tensor(np.asarray(payload.old_per_token_logps)),
      logps,
      torch.tensor(np.asarray(payload.advantages)),
      torch.tensor(np.asarray(payload.completion_mask)),
      cliprange=0.2,
      clip_ratio_c=3.0,
  )[0]
  loss.backward()
  return float(loss.detach()), weights.grad.numpy()
