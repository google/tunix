"""Recipe data boundaries, actual checkpoint output and optional HF template QA."""

import hashlib
import json
import os
from pathlib import Path

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
