"""User-defined workflows share recording, policy ownership and grading APIs."""

import asyncio

import numpy as np
import pytest
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import api
from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import runtime


class ObservedSampler:

  def __init__(self):
    self.requests = []
    self.active = 0
    self.peak = 0

  async def sample(self, request):
    self.requests.append(request)
    self.active += 1
    self.peak = max(self.peak, self.active)
    # Reversed completion timing exercises deterministic request provenance.
    await asyncio.sleep(0.005 if len(self.requests) == 1 else 0)
    self.active -= 1
    return sampler.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=np.asarray(request.prompt),
        token_ids=np.array([3, 4]),
        logprobs=np.array([-0.5, -1.0]),
        text="approved" if request.metadata["role"] == "verifier" else "42",
    )


def setup(plan, workflow, **kwargs):
  backend = ObservedSampler()
  prompts, factory_calls = [], []

  def factory(policy_id, version, namespace):
    factory_calls.append((policy_id, version, namespace))

    def encode(prompt):
      prompts.append(prompt)
      return [1, 2]

    return runtime.RecordingRuntime(
        backend,
        encode,
        sampler.SamplingParams(max_tokens=4, seed=41),
        policy_id=policy_id,
        policy_version=version,
        episode_id=namespace,
        max_prompt_length=32,
    )

  spec = api.SwarmSpec(workflow=workflow, learning_plan=plan, **kwargs)
  return spec, factory, backend, prompts, factory_calls


def shared_plan():
  return policy.LearningPlan(
      learner_policy_id="shared",
      bindings=(policy.PolicyBinding(policy_id="shared", trainable=True),),
  )


def test_solver_verifier_workflow_separates_private_grading():
  completed = []
  private_answer = "PRIVATE_TARGET_NEVER_EXPOSED"

  async def solver_verifier(team, question):
    draft = await team.generate("solver", question)
    review = await team.generate("verifier", f"Check {question}: {draft.text}")
    assert review.text == "approved"
    assert not hasattr(team, "answer") and not hasattr(team, "grade")
    completed.append(True)
    return draft.text

  spec, factory, backend, prompts, factory_calls = setup(
      shared_plan(),
      solver_verifier,
      roles={"solver": "shared", "verifier": "shared"},
  )

  def grade(final):
    assert completed and private_answer
    return float(final == "42")

  episode = asyncio.run(
      spec.run(
          task_id="math",
          episode_id="attempt",
          prompt="Compute 6 * 7.",
          learner_version=3,
          runtime_factory=factory,
          grade=grade,
      )
  )
  assert episode.reward == 1
  assert [c.role for c in episode.calls] == ["solver", "verifier"]
  assert {c.policy_version for c in episode.calls} == {3}
  assert factory_calls == [("shared", 3, "attempt/p0")]
  assert all(private_answer not in p for p in prompts)
  assert len(backend.requests) == 2
  with pytest.raises(TypeError):
    spec.roles["new"] = "shared"


def test_director_and_frozen_worker_use_same_workflow_api():
  plan = policy.LearningPlan(
      learner_policy_id="director",
      bindings=(
          policy.PolicyBinding(policy_id="director", trainable=True),
          policy.PolicyBinding(policy_id="worker", trainable=False, version=0),
      ),
  )

  async def workflow(team, question):
    action = await team.generate("director", f"Plan: {question}")
    result = await team.generate("solver", f"{question}\nPlan: {action.text}")
    return api.SwarmOutput(final_text=result.text, worker_cost=0.5)

  spec, factory, _, _, _ = setup(
      plan, workflow, roles={"director": "director", "solver": "worker"}
  )
  episode = asyncio.run(
      spec.run(
          task_id="t",
          episode_id="e",
          prompt="Compute.",
          learner_version=7,
          runtime_factory=factory,
          grade=lambda _: 1,
      )
  )
  assert [(c.policy_id, c.policy_version) for c in episode.calls] == [
      ("director", 7),
      ("worker", 0),
  ]
  assert episode.worker_cost == 0.5
  assert [c.role for c in plan.select_calls(episode)] == ["director"]


def test_parallel_calls_have_unique_ids_seeds_and_bounded_concurrency():
  async def workflow(team, question):
    results = await asyncio.gather(
        *(team.generate("solver", f"{question} {i}") for i in range(5))
    )
    return results[0].text

  spec, factory, backend, _, _ = setup(
      shared_plan(),
      workflow,
      roles={"solver": "shared"},
      max_concurrent_calls=2,
  )
  episode = asyncio.run(
      spec.run(
          task_id="t",
          episode_id="e",
          prompt="Solve",
          learner_version=0,
          runtime_factory=factory,
          grade=lambda _: 1,
      )
  )
  assert [c.call_id for c in episode.calls] == [f"e/p0/{i}" for i in range(5)]
  assert [r.sampling_params.seed for r in backend.requests] == list(
      range(41, 46)
  )
  assert backend.peak == 2


def test_pending_workflow_calls_are_cancelled_before_grading():
  async def workflow(team, question):
    await team.generate("solver", question)
    pending = asyncio.create_task(team.generate("solver", question))
    await asyncio.sleep(0)
    assert not pending.done()
    return "42"

  spec, factory, _, _, _ = setup(
      shared_plan(), workflow, roles={"solver": "shared"}
  )
  with pytest.raises(RuntimeError, match="pending generation"):
    asyncio.run(
        spec.run(
            task_id="t",
            episode_id="e",
            prompt="Solve",
            learner_version=0,
            runtime_factory=factory,
            grade=lambda _: pytest.fail("Must not grade dangling work"),
        )
    )


@pytest.mark.parametrize("failure", ["role", "budget", "timeout"])
def test_invalid_workflow_attempts_fail_explicitly(failure):
  async def workflow(team, question):
    if failure == "role":
      await team.generate("unknown", question)
    elif failure == "budget":
      await team.generate("solver", question)
      await team.generate("solver", question)
    else:
      await asyncio.Event().wait()
    return "42"

  spec, factory, _, _, _ = setup(
      shared_plan(),
      workflow,
      roles={"solver": "shared"},
      max_calls=1,
      timeout_seconds=0.01 if failure == "timeout" else 1,
  )
  expected = {
      "role": ValueError,
      "budget": RuntimeError,
      "timeout": TimeoutError,
  }[failure]
  with pytest.raises(expected):
    asyncio.run(
        spec.run(
            task_id="t",
            episode_id="e",
            prompt="Solve",
            learner_version=0,
            runtime_factory=factory,
            grade=lambda _: pytest.fail(
                "Failed attempts must not reach grader"
            ),
        )
    )


def test_recording_runtime_reserves_concurrent_request_identity_before_await():
  backend = ObservedSampler()
  rt = runtime.RecordingRuntime(
      backend,
      lambda _: [1, 2],
      sampler.SamplingParams(max_tokens=4, seed=7),
      policy_id="shared",
      policy_version=0,
      episode_id="e",
      max_prompt_length=8,
  )

  async def parallel():
    return await asyncio.gather(
        rt.generate("solver", "one"), rt.generate("solver", "two")
    )

  calls = asyncio.run(parallel())
  assert [c.call_id for c in calls] == ["e/0", "e/1"]
  assert [r.sampling_params.seed for r in backend.requests] == [7, 8]


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_noninteger_or_nonpositive_call_budgets_are_rejected(value):
  async def workflow(team, question):
    return "42"

  with pytest.raises(ValueError, match="budgets"):
    api.SwarmSpec(
        roles={"solver": "shared"},
        workflow=workflow,
        learning_plan=shared_plan(),
        max_calls=value,
    )


def test_just_queued_generation_cannot_escape_workflow_lifetime():
  pending = []

  async def workflow(team, question):
    await team.generate("solver", question)
    pending.append(asyncio.create_task(team.generate("solver", question)))
    return "42"  # Intentionally does not yield to the newly queued task.

  async def scenario():
    spec, factory, _, _, _ = setup(
        shared_plan(), workflow, roles={"solver": "shared"}
    )
    with pytest.raises(RuntimeError, match="pending generation"):
      await spec.run(
          task_id="t",
          episode_id="e",
          prompt="Solve",
          learner_version=0,
          runtime_factory=factory,
          grade=lambda _: pytest.fail("Pending work reached grader"),
      )
    assert pending[0].done() and pending[0].cancelled()

  asyncio.run(scenario())


def test_workflow_exception_cleans_just_queued_calls():
  pending = []

  async def workflow(team, question):
    pending.append(asyncio.create_task(team.generate("solver", question)))
    raise ValueError("workflow failure")

  async def scenario():
    spec, factory, _, _, _ = setup(
        shared_plan(), workflow, roles={"solver": "shared"}
    )
    with pytest.raises(ValueError, match="workflow failure"):
      await spec.run(
          task_id="t",
          episode_id="e",
          prompt="Solve",
          learner_version=0,
          runtime_factory=factory,
          grade=lambda _: pytest.fail("Failed work reached grader"),
      )
    assert pending[0].done() and pending[0].cancelled()

  asyncio.run(scenario())
