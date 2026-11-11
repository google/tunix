"""Reference comparisons for role credit, episode credit and trace invariants."""

import dataclasses

import numpy as np
import pytest
from tests.dist_rl.swarm import reference
from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import trace


def episode(
    eid, reward, roles=("Solver Agent", "Verifier Agent"), invalid=(), task="a"
):
  return trace.SwarmEpisode(
      task_id=task,
      episode_id=eid,
      reward=reward,
      final_text="answer",
      calls=tuple(
          trace.CallTrace(
              call_id=f"{eid}/{i}",
              role=role,
              policy_id="shared",
              policy_version=0,
              prompt_ids=(1, 0, 2),
              completion_ids=(4, 0, 5),
              rollout_logprobs=(-1.0, -2.0, -1.5),
              loss_mask=(1, 0, 1),
              text="answer",
              valid_action=i not in invalid,
          )
          for i, role in enumerate(roles)
      ),
  )


@pytest.mark.parametrize("mode", ["role", "episode"])
def test_matches_original_with_unequal_calls_and_format_penalties(mode):
  episodes = [
      episode("e0", 1),
      episode(
          "e1",
          0,
          ("Solver Agent", "Verifier Agent", "Solver Agent"),
          invalid=(1,),
      ),
      episode(
          "e2",
          1,
          ("Solver Agent", "Verifier Agent", "Solver Agent"),
          invalid=(0, 2),
      ),
      episode("e3", 0, ("Solver Agent",)),
      # Independent task and singleton role are intentionally present.
      episode("e4", 1, ("Solver Agent", "Verifier Agent"), task="b"),
      episode("e5", 0, ("Solver Agent",), task="b"),
  ]
  actual = credit.build_credit(episodes, normalization=mode)
  expected = reference.advantages(episodes, role=mode == "role")
  np.testing.assert_allclose(
      np.concatenate([p.advantages for p in actual.payloads]),
      expected,
      atol=2e-6,
  )
  for payload in actual.payloads:
    np.testing.assert_array_equal(
        payload.completion_attention_mask, [[1, 1, 1]]
    )
    np.testing.assert_array_equal(payload.completion_mask, [[1, 0, 1]])
    assert payload.advantages[0, 1] == 0


def test_equal_rewards_zero_signal_and_singleton_source_convention():
  zero = credit.build_credit([episode("a", 1), episode("b", 1)])
  assert all(x == 0 for x in zero.advantages)
  singleton = credit.build_credit([episode("a", 1)])
  np.testing.assert_allclose(singleton.advantages, [1 / (1 + 1e-6)] * 2)


@pytest.mark.parametrize("mode", ["role", "episode"])
@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_constant_decimal_penalties_do_not_create_roundoff_advantage(
    mode, reward
):
  episodes = [
      episode(
          f"e{i}",
          reward,
          ("Solver Agent",) * (i + 1),
          invalid=tuple(range(i + 1)),
      )
      for i in range(8)
  ]
  actual = credit.build_credit(episodes, normalization=mode)
  np.testing.assert_array_equal(actual.advantages, 0.0)


@pytest.mark.parametrize(
    "failure", ["duplicate_episode", "duplicate_call", "version", "policy"]
)
def test_rejects_mixed_or_duplicated_collection(failure):
  a, b = episode("a", 1), episode("b", 0)
  if failure == "duplicate_episode":
    b = a
  elif failure == "duplicate_call":
    b = dataclasses.replace(b, calls=a.calls)
  else:
    field = (
        {"policy_version": 1}
        if failure == "version"
        else {"policy_id": "other"}
    )
    b = dataclasses.replace(
        b, calls=tuple(dataclasses.replace(c, **field) for c in b.calls)
    )
  with pytest.raises(ValueError):
    credit.build_credit([a, b])


@pytest.mark.parametrize(
    "field,value",
    [
        ("loss_mask", (0, 0, 0)),
        ("loss_mask", (1,)),
        ("rollout_logprobs", (float("nan"), 0, 0)),
        ("prompt_ids", ()),
    ],
)
def test_invalid_trace_rejected(field, value):
  with pytest.raises(ValueError):
    dataclasses.replace(episode("a", 1).calls[0], **{field: value})
