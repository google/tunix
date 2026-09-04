# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validation script comparing Tunix Vanilla Sampler and Jax-Inference Sampler on Math-500."""

import json
import os
import sys
import jax

# Ensure paths
sys.path.insert(0, "/mnt/disks/persist/atwigg/tunix")
sys.path.insert(0, "/mnt/disks/persist/atwigg/trellis/experimental/jax-inference")

from examples.deepscaler.math_eval_nb import (
    MODEL_MAPPING,
    Qwen25MathEvaluator,
    _resolve_model_path,
)


def run_comparison(
    model_version: str = "Qwen/Qwen3-1.7B-base",
    num_samples: int = 4,
    batch_size: int = 2,
):
  print(f"\n=======================================================")
  print(f"Comparing Samplers on {model_version} (samples={num_samples})")
  print(f"=======================================================\n")

  model_config, model_path = MODEL_MAPPING[model_version]
  max_tp = getattr(model_config, "num_kv_heads", len(jax.devices()))
  tp_size = min(len(jax.devices()), max_tp)
  mesh_config = [[1, tp_size], ["fsdp", "tp"]]
  dataset_name = "HuggingFaceH4/MATH-500"

  num_batches = (num_samples + batch_size - 1) // batch_size

  # 1. Run Vanilla Sampler
  print("\n>>> [1/2] Running Evaluation with Vanilla Sampler...")
  evaluator_vanilla = Qwen25MathEvaluator(
      model_config=model_config,
      model_version=model_version,
      model_path=model_path,
      dataset=dataset_name,
      mesh_config=mesh_config,
      max_prompt_length=1024,
      max_generation_steps=128,
      sampler_type="vanilla",
  )
  evaluator_vanilla.load_model()
  results_vanilla = evaluator_vanilla.evaluate(
      batch_size=batch_size,
      num_batches=num_batches,
      temperature=0.0,
      top_k=1,
      top_p=1.0,
      num_passes=1,
      debug_first_n=num_samples,
  )

  # Free TPU memory
  del evaluator_vanilla
  import gc
  gc.collect()

  # 2. Run Jax-Inference Sampler
  print("\n>>> [2/2] Running Evaluation with Jax-Inference Sampler...")
  evaluator_jax_inf = Qwen25MathEvaluator(
      model_config=model_config,
      model_version=model_version,
      model_path=model_path,
      dataset=dataset_name,
      mesh_config=mesh_config,
      max_prompt_length=1024,
      max_generation_steps=128,
      sampler_type="jax_inference",
  )
  evaluator_jax_inf.load_model()
  results_jax_inf = evaluator_jax_inf.evaluate(
      batch_size=batch_size,
      num_batches=num_batches,
      temperature=0.0,
      top_k=1,
      top_p=1.0,
      num_passes=1,
      debug_first_n=num_samples,
  )

  # 3. Compare Results
  print("\n=======================================================")
  print("Validation & Comparison Summary")
  print("=======================================================")
  vanilla_detailed = results_vanilla["detailed_results"][:num_samples]
  jax_inf_detailed = results_jax_inf["detailed_results"][:num_samples]

  matches = 0
  reward_matches = 0
  for i, (v, j) in enumerate(zip(vanilla_detailed, jax_inf_detailed)):
    v_resp = v["responses"][0].strip() if v["responses"] else ""
    j_resp = j["responses"][0].strip() if j["responses"] else ""
    v_ans = v["extracted_answers"][0] if v["extracted_answers"] else None
    j_ans = j["extracted_answers"][0] if j["extracted_answers"] else None
    v_corr = v["correct"]
    j_corr = j["correct"]

    text_match = v_resp == j_resp
    ans_match = v_ans == j_ans
    reward_match = v_corr == j_corr

    if text_match:
      matches += 1
    if reward_match:
      reward_matches += 1

    print(f"\n--- Item {i+1} ---")
    print(f"Question (prefix): {v['question'][:80]}...")
    print(f"Ground Truth Answer: {v['answer']}")
    print(f"Vanilla Extracted: {v_ans} | Correct: {v_corr}")
    print(f"Jax-Inf Extracted: {j_ans} | Correct: {j_corr}")
    print(f"Text Match: {text_match} | Reward Match: {reward_match}")

  print(f"\nTotal compared: {len(vanilla_detailed)}")
  print(f"Exact text matches: {matches}/{len(vanilla_detailed)}")
  print(f"Reward (correctness) matches: {reward_matches}/{len(vanilla_detailed)}")
  print(f"Vanilla Accuracy: {results_vanilla['accuracy']:.2f}%")
  print(f"Jax-Inference Accuracy: {results_jax_inf['accuracy']:.2f}%")
  print("=======================================================\n")


if __name__ == "__main__":
  model = os.environ.get("MODEL_VERSION", "Qwen/Qwen3-1.7B-base")
  num = int(os.environ.get("NUM_SAMPLES", "4"))
  run_comparison(model_version=model, num_samples=num)
