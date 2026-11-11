"""Compare actual upstream orchestration and projection with the async port."""

import asyncio
import dataclasses
import types

import numpy as np
import pytest
from examples.swarm_math import prompts
from examples.swarm_math import workflow
from tests.dist_rl.swarm import reference
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import runtime


def source_run(texts, turns):
  env = reference.namespace()
  pending = iter(texts)
  observed = []

  class Agent:

    def __init__(self, role):
      self.role = role

    def call(self, *, team_context, **kwargs):
      del kwargs
      prompt = (
          prompts.SOLVER_PROMPT
          if self.role == workflow.SOLVER
          else prompts.VERIFIER_PROMPT
      ).format(env_prompt="Compute 1 + 1.", team_context=team_context[0])
      observed.append((self.role, prompt))
      raw = next(pending)
      if self.role == workflow.VERIFIER:
        projected, valid = env["general_projection"](
            [raw], "<verify>", "</verify>", return_whole_response=True
        )
      else:
        projected, valid = env["math_projection"]([raw], check_think_tag=False)
      return (raw, bool(valid[0])), projected

    def update_approved_vector(self, *args):
      return env["update_approved_vector"](self, *args)

  buffer = []
  obj = types.SimpleNamespace(
      max_loop_num=turns,
      SOLVER_AGENT=workflow.SOLVER,
      VERIFIER_AGENT=workflow.VERIFIER,
      agents={r: Agent(r) for r in (workflow.SOLVER, workflow.VERIFIER)},
      agents_to_wg_mapping={
          r: "shared" for r in (workflow.SOLVER, workflow.VERIFIER)
      },
      reset_buffer=buffer.clear,
      initialize_context=lambda obs: ([""], [""], obs),
      save_to_buffer=lambda role, batch: buffer.append((role, batch)),
      multiagent_batch_buffer=buffer,
  )
  final, calls = env["run"](
      obj,
      [None],
      {"text": ["Compute 1 + 1."]},
      {"shared": object()},
      np.array([True]),
      1,
  )
  return final[0], calls, observed


class ScriptedSampler:

  def __init__(self, texts):
    self.texts = iter(texts)
    self.requests = []

  async def sample(self, request):
    self.requests.append(request)
    return sampler.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt),
        token_ids=np.array([4, 0]),
        logprobs=np.array([-1.0, -2.0]),
        text=next(self.texts),
    )


@pytest.mark.parametrize(
    "verdict",
    [
        "<verify>approve</verify>",
        "<verify>reject</verify>",
        "<verify> reject </verify>",
        "missing verdict",
        "<verify>maybe</verify>",
        "<verify>approve</verify><verify>reject</verify>",
        "</verify>reject<verify>",
        "\u4e2d<verify>reject</verify>",
    ],
)
def test_solver_verifier_matches_source(verdict):
  texts = [r"First \boxed{3}", verdict, r"Revised \boxed{2}"]
  expected_final, expected_calls, expected_prompts = source_run(texts, 2)
  observed = []
  backend = ScriptedSampler(texts)

  def encode(prompt):
    role = (
        workflow.VERIFIER
        if 'You are a "Verifier Agent"' in prompt.rsplit("# Your Role", 1)[-1]
        else workflow.SOLVER
    )
    observed.append((role, prompt))
    return [1, 0, len(observed)]

  rt = runtime.RecordingRuntime(
      backend,
      encode,
      sampler.SamplingParams(max_tokens=3, seed=7),
      policy_id="shared",
      policy_version=0,
      episode_id="a",
      max_prompt_length=32,
  )
  result = asyncio.run(
      workflow.SolverVerifierWorkflow().run(
          workflow.MathTask("task", "Compute 1 + 1.", "2"), rt
      )
  )
  assert result.final_text == expected_final
  assert observed == expected_prompts
  assert [
      (c.role, (c.text, c.valid_action)) for c in result.calls
  ] == expected_calls
  assert result.reward == float(expected_final == texts[-1])
  assert all(c.policy_id == "shared" for c in result.calls)
  assert [r.sampling_params.seed for r in backend.requests] == list(
      range(7, 7 + len(result.calls))
  )
  assert all(
      c.loss_mask == (1, 1) for c in result.calls
  )  # EOS=pad stays active.


def test_single_solver_and_source_reward_think_handling():
  final, calls, _ = source_run([r"\boxed{2}"], 1)
  assert final == r"\boxed{2}" and len(calls) == 1
  assert workflow.grade(r"<think>\boxed{3}</think>\boxed{2}", "2")
  assert not workflow.grade(r"<think>\boxed{2}</think>no answer", "2")
  assert workflow.grade(r"\boxed{\frac{1}{2}}", "0.5")


@pytest.mark.parametrize(
    "failure", ["prompt", "request", "error", "logprobs", "overflow"]
)
def test_runtime_rejects_unusable_response(failure):
  class BrokenSampler(ScriptedSampler):

    async def sample(self, request):
      response = await super().sample(request)
      if failure == "prompt":
        response.prompt_token_ids = np.array([7, 7])
      elif failure == "request":
        response.request_id = "other"
      elif failure == "error":
        response.finish_reason = "cancelled"
      elif failure == "logprobs":
        response.logprobs = None
      return response

  rt = runtime.RecordingRuntime(
      BrokenSampler(["text"]),
      lambda _: [1, 2],
      sampler.SamplingParams(max_tokens=1 if failure == "overflow" else 3),
      policy_id="shared",
      policy_version=0,
      episode_id="a",
      max_prompt_length=3,
  )
  with pytest.raises((ValueError, RuntimeError)):
    asyncio.run(rt.generate("role", "prompt"))
