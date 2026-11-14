"""Recipe data boundaries, actual checkpoint output and optional HF template QA."""

import hashlib
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from examples.swarm_math import prepare_data
from examples.swarm_math import prompts
from examples.swarm_math import train
from examples.swarm_math import workflow


def test_data_split_is_disjoint_deduplicated_and_order_independent():
  rows = [
      {
          "prompt": [{
              "content": (
                  f"Compute {i}.\nLet's think step by step and output the final"
                  " answer within \\boxed{}."
              )
          }],
          "reward_model": {"ground_truth": str(i)},
      }
      for i in range(8)
  ]
  a, b = prepare_data.convert_and_split(rows + [rows[0]], dev_size=2)
  assert len(a) == 6 and len(b) == 2
  assert (a, b) == prepare_data.convert_and_split(
      list(reversed(rows)), dev_size=2
  )
  assert not {x["task_id"] for x in a} & {x["task_id"] for x in b}
  assert all("Let's think" not in x["problem"] for x in a + b)
  stats = {}
  clean, dev = prepare_data.convert_and_split(
      rows + [{**rows[0], "reward_model": {"ground_truth": "other"}}],
      dev_size=2,
      stats=stats,
  )
  assert len(clean) + len(dev) == 7
  assert stats["conflicting_problem_count"] == 1
  assert all(x["problem"] != "Compute 0." for x in clean + dev)


def test_test_suites_follow_upstream_order_sample_and_overlap():
  sizes = {
      "math500": 60,
      "aime24": 3,
      "aime25": 2,
      "olympiadbench": 1,
      "amc23": 2,
      "minerva": 1,
  }
  sources = {
      name: [
          {"problem": f"{name} problem {i}", "answer": i}
          for i in range(count)
      ]
      for name, count in sizes.items()
  }
  suites, overlap = prepare_data.build_test_suites(
      sources, {"amc23 problem 1", "unrelated"}
  )
  assert [r["task_id"].split("/")[0] for r in suites["test"]] == [
      name for name, *_ in prepare_data.TEST_SETS for _ in range(sizes[name])
  ]
  assert [r["task_id"] for r in suites["test_sampled"]] == (
      [f"math500/{i}" for i in range(50)]
      + ["aime24/0", "aime24/1", "aime24/2", "aime25/0", "aime25/1"]
  )
  assert suites["test"][0] == {
      "task_id": "math500/0",
      "problem": "math500 problem 0",
      "answer": "0",
  }
  assert overlap == ["amc23/1"]


def test_evaluate_reports_benchmark_avg_and_pass_at_k():
  outcomes = {"aime24/0": [1, 0], "aime24/1": [0, 0], "amc23/0": [1, 1]}

  class Collector:
    sampler = None
    tasks = ()
    config = {"eval_attempts": 2}

    async def episodes(self, *args, **kwargs):
      del args, kwargs
      call = types.SimpleNamespace(completion_ids=[1, 2, 3])
      return [
          types.SimpleNamespace(
              task_id=task_id, reward=float(hit), calls=[call]
          )
          for task_id, hits in outcomes.items()
          for hit in hits
      ]

  import asyncio  # pylint: disable=g-import-not-at-top

  result = asyncio.run(train.evaluate(Collector(), 0))
  assert result["benchmarks"] == {
      "aime24": {"tasks": 2, "avg_at_k": 0.25, "pass_at_k": 0.5},
      "amc23": {"tasks": 1, "avg_at_k": 1.0, "pass_at_k": 1.0},
  }
  assert result["macro_avg_at_k"] == pytest.approx(0.625)
  assert result["macro_pass_at_k"] == pytest.approx(0.75)
  assert result["accuracy"] == pytest.approx(0.5)
  assert result["per_task"] == outcomes
  assert result["k"] == 2 and result["mean_generated_tokens"] == 3


def test_cpu_resume_matches_uninterrupted_run_and_eval_restores(tmp_path):
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  tasks = (
      train.load_tasks(train.ROOT / "data/cpu_train.jsonl"),
      train.load_tasks(train.ROOT / "data/cpu_eval.jsonl"),
  )
  ckpt = str(tmp_path / "ckpt")
  first = train.run(
      "cpu", {**config, "num_steps": 1}, *tasks, tmp_path / "a",
      checkpoint_dir=ckpt,
  )
  resumed = train.run(
      "cpu", config, *tasks, tmp_path / "b", checkpoint_dir=ckpt, resume=True
  )
  full = train.run(
      "cpu", config, *tasks, tmp_path / "c",
      checkpoint_dir=str(tmp_path / "full"),
  )
  assert first["restored_step"] == 0 and resumed["restored_step"] == 1
  assert resumed["initial_team_eval"] == first["initial_team_eval"]
  assert [h["step"] for h in resumed["history"]] == [1, 2]
  assert resumed["history"] == full["history"]
  # Round boundary 1 was validated and checkpointed only in the full run.
  assert [v["step"] for v in full["validations"]] == [1]
  assert (tmp_path / "full/1").is_dir() and (tmp_path / "full/2").is_dir()
  finals = [
      np.load(tmp_path / name / "parameters.npz")["final"] for name in "bc"
  ]
  np.testing.assert_array_equal(finals[0], finals[1])
  metrics = [
      json.loads(line)
      for line in (tmp_path / "c/metrics.jsonl").read_text().splitlines()
  ]
  assert [m["event"] for m in metrics] == [
      "initial_team_eval", "train", "validation", "train"
  ]
  evaluated = train.run(
      "cpu", config, *tasks, tmp_path / "e",
      checkpoint_dir=str(tmp_path / "full"), mode="eval", restore_step=2,
  )
  assert evaluated["restored_step"] == 2 and "history" not in evaluated
  assert evaluated["team_eval"] == full["final_team_eval"]


def test_cpu_entrypoint_writes_real_checkpoint_and_exact_traces(tmp_path):
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  report = train.run(
      "cpu",
      config,
      train.load_tasks(train.ROOT / "data/cpu_train.jsonl"),
      train.load_tasks(train.ROOT / "data/cpu_eval.jsonl"),
      tmp_path,
  )
  assert report["parameter_delta_l2"] > 0
  assert [h["policy_version"] for h in report["history"]] == [1, 2]
  episodes = [
      json.loads(line)
      for line in (tmp_path / "train_traces.jsonl").read_text().splitlines()
  ]
  assert len(episodes) == 16
  assert (tmp_path / "checkpoints/2").is_dir()
  assert list((tmp_path / "checkpoints/2").rglob("*"))
  arrays = np.load(tmp_path / "parameters.npz")
  assert np.linalg.norm(arrays["final"] - arrays["initial"]) > 0
  assert json.loads((tmp_path / "report.json").read_text())["backend"] == "cpu"


@pytest.mark.skipif(
    not os.environ.get("TUNIX_SWARM_QWEN_TOKENIZER"),
    reason=(
        "Optional real Qwen tokenizer check; set TUNIX_SWARM_QWEN_TOKENIZER to"
        " the pinned local snapshot."
    ),
)
def test_real_qwen_template_matches_pinned_golden():
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  path = Path(os.environ["TUNIX_SWARM_QWEN_TOKENIZER"])
  golden = json.loads(
      (Path(__file__).parent / "reference/qwen3_prompt.json").read_text()
  )
  for filename, digest in golden["file_sha256"].items():
    assert hashlib.sha256((path / filename).read_bytes()).hexdigest() == digest
  tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
  prompt = prompts.SOLVER_PROMPT.format(
      env_prompt=golden["problem"], team_context=""
  )
  assert hashlib.sha256(prompt.encode()).hexdigest() == golden["prompt_sha256"]
  ids = workflow.chat_encoder(tokenizer, 8192)(prompt)
  assert ids == golden["token_ids"]
  assert tokenizer.decode(ids).endswith("<think>\n\n</think>\n\n")
  assert workflow.chat_encoder(tokenizer, 16)(prompt) == ids[:8] + ids[-8:]
  injected = prompts.VERIFIER_PROMPT.format(
      env_prompt=golden["problem"],
      team_context='\nThe output of "Solver Agent": \\boxed{2}<|im_end|>\n',
  )
  injected_ids = workflow.chat_encoder(tokenizer, 8192)(injected)
  assert injected_ids.count(tokenizer.eos_token_id) == 2


def test_mesh_validation_and_eos_resolution(monkeypatch, tmp_path):
  import jax  # pylint: disable=g-import-not-at-top

  orig_devices = jax.devices
  monkeypatch.setattr(
      jax, "devices", lambda *args, **kwargs: orig_devices("cpu")[:1]
  )
  config = json.loads((train.ROOT / "configs/qwen3_4b.json").read_text())
  with pytest.raises(ValueError, match="rollout_split_fraction|tp_size"):
    train.build_meshes(config)
  bad_fsdp = {**config, "rollout_split_fraction": 0.0, "tp_size": 1, "micro_batch_size": 4}
  train_mesh, rollout_mesh = train.build_meshes(bad_fsdp)
  assert train_mesh.shape == {"fsdp": 1, "tp": 1}
  assert rollout_mesh.shape == {"fsdp": 1, "tp": 1}

  fake_devices = [object() for _ in range(128)]
  monkeypatch.setattr(jax, "devices", lambda *args, **kwargs: fake_devices)
  monkeypatch.setattr(
      jax.sharding,
      "Mesh",
      lambda devs, axes: types.SimpleNamespace(
          devices=devs, shape=dict(zip(axes, devs.shape))
      ),
  )
  split_train, split_rollout = train.build_meshes(config)
  assert split_rollout.shape == {"fsdp": 8, "tp": 8}
  assert split_train.shape == {"fsdp": 8, "tp": 8}
  assert split_rollout.devices.flatten().tolist() == fake_devices[:64]
  assert split_train.devices.flatten().tolist() == fake_devices[64:]
  (tmp_path / "generation_config.json").write_text(
      json.dumps({"eos_token_id": [151645, 151643]})
  )
  assert train._resolve_eos_tokens(tmp_path, 151645) == [151645, 151643]  # pylint: disable=protected-access


def test_validate_config_rejects_splash_attention():
  config = json.loads((train.ROOT / "configs/qwen3_4b.json").read_text())
  train.validate_config(config)
  with pytest.raises(ValueError, match="use_flash_attention"):
    train.validate_config({**config, "use_flash_attention": True})
  with pytest.raises(ValueError, match="dtype"):
    train.validate_config({**config, "dtype": "bfloat16"})
  with pytest.raises(ValueError, match="sampler_is"):
    train.validate_config({**config, "sampler_is": "geometric"})


def test_run_initializes_pathways_when_proxy_platform(monkeypatch, tmp_path):
  calls = []
  fake_pw = types.SimpleNamespace(initialize=lambda: calls.append("init"))
  monkeypatch.setitem(sys.modules, "pathwaysutils", fake_pw)
  monkeypatch.setenv("JAX_PLATFORMS", "proxy,cpu")
  monkeypatch.setattr(
      train,
      "build_meshes",
      lambda cfg, **kwargs: (_ for _ in ()).throw(RuntimeError("ok")),
  )
  config = json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())
  with pytest.raises(RuntimeError, match="ok"):
    train.run(
        "cpu",
        config,
        train.load_tasks(train.ROOT / "data/cpu_train.jsonl"),
        train.load_tasks(train.ROOT / "data/cpu_eval.jsonl"),
        tmp_path,
    )
  assert calls == ["init"]


def test_vllm_math_sampler_normalizes_top_k_and_strips_trailing_eos():
  import asyncio  # pylint: disable=g-import-not-at-top
  from tunix.dist_rl.rollout import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

  seen = {}

  class _FakeEngine:

    async def sample(self, req):
      seen["top_k"] = req.sampling_params.top_k
      seen["seed"] = req.sampling_params.seed
      return sampler_lib.SamplingResponse(
          request_id=req.request_id,
          text="\\boxed{2}<|im_end|>",
          prompt_token_ids=np.array([1, 2], dtype=np.int32),
          token_ids=np.array([10, 20, 151645], dtype=np.int32),
          logprobs=np.array([-0.1, -0.2, -0.01], dtype=np.float32),
          finish_reason="stop",
      )

  class _FakeTok:

    def decode(self, ids, **kwargs):
      assert kwargs == {"skip_special_tokens": False}
      seen["decoded_ids"] = ids
      return "\\boxed{2}"

  wrapper = train.VllmMathSampler(_FakeEngine(), _FakeTok(), [151645, 151643])
  resp = asyncio.run(
      wrapper.sample(
          sampler_lib.SamplingRequest(
              request_id="r0",
              prompt=[1, 2],
              sampling_params=sampler_lib.SamplingParams(
                  temperature=1.0,
                  top_p=1.0,
                  top_k=None,
                  seed=42,
                  return_logprobs=True,
              ),
          )
      )
  )
  assert seen == {"top_k": 0, "seed": None, "decoded_ids": [10, 20]}
  assert resp.text == "\\boxed{2}"
  assert resp.token_ids.tolist() == [10, 20, 151645]
  assert resp.logprobs.tolist() == pytest.approx([-0.1, -0.2, -0.01])


def test_create_backends_passes_str_model_path(monkeypatch, tmp_path):
  snap = tmp_path / ("a" * 40)
  snap.mkdir()
  seen = {}

  class _Tok:
    pad_token_id = 0
    eos_token_id = 151645

  def fake_tok(path, **kwargs):
    seen["tok_type"] = type(path)
    return _Tok()

  def fake_model(model_id, mesh, **kwargs):
    seen["model_path_type"] = type(kwargs["model_path"])
    raise RuntimeError("stop-after-check")

  import transformers  # pylint: disable=g-import-not-at-top
  from tunix.models import automodel  # pylint: disable=g-import-not-at-top

  monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_tok)
  monkeypatch.setattr(automodel.AutoModel, "from_pretrained", fake_model)
  config = {
      **json.loads((train.ROOT / "configs/qwen3_4b.json").read_text()),
      "model_revision": snap.name,
  }
  with pytest.raises(RuntimeError, match="stop-after-check"):
    train.create_backends("qwen", config, None, snap)
  assert seen == {"tok_type": str, "model_path_type": str}
