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

"""Validation script comparing any two Tunix samplers on Math benchmarks."""

import argparse
import gc
import json
import os
import sys
from typing import Any, Dict
import jax

# Ensure paths
sys.path.insert(0, "/mnt/disks/persist/atwigg/tunix")
sys.path.insert(0, "/mnt/disks/persist/atwigg/trellis/experimental/jax-inference")

from examples.deepscaler.math_eval_nb import (
    MODEL_MAPPING,
    Qwen25MathEvaluator,
    _resolve_model_path,
)

SUPPORTED_SAMPLERS = ("vanilla", "jax_inference", "vllm", "sglang_jax")


def run_sampler_eval(
    sampler_type: str,
    model_version: str,
    dataset: str,
    batch_size: int,
    num_batches: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    max_generation_steps: int,
    max_prompt_length: int,
    num_samples: int,
) -> Dict[str, Any]:
  """Runs evaluation using the specified sampler backend and frees resources."""
  model_config, model_path = MODEL_MAPPING[model_version]
  max_tp = getattr(model_config, "num_kv_heads", len(jax.devices()))
  tp_size = min(len(jax.devices()), max_tp)
  mesh_config = [[1, tp_size], ["fsdp", "tp"]]

  print(f"\n>>> Running Evaluation with [{sampler_type}] Sampler...")
  evaluator = Qwen25MathEvaluator(
      model_config=model_config,
      model_version=model_version,
      model_path=model_path,
      dataset=dataset,
      mesh_config=mesh_config,
      max_prompt_length=max_prompt_length,
      max_generation_steps=max_generation_steps,
      sampler_type=sampler_type,
  )
  evaluator.load_model()
  results = evaluator.evaluate(
      batch_size=batch_size,
      num_batches=num_batches,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      num_passes=1,
      debug_first_n=num_samples,
  )

  # Free TPU memory between evaluations
  del evaluator
  gc.collect()

  return results


def run_comparison(
    sampler1: str = "vanilla",
    sampler2: str = "jax_inference",
    model_version: str = "Qwen/Qwen3-1.7B-base",
    dataset: str = "HuggingFaceH4/MATH-500",
    num_samples: int = 4,
    batch_size: int = 2,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
    seed: int = 42,
    max_generation_steps: int = 128,
    max_prompt_length: int = 1024,
    output_json: str | None = None,
):
  """Compares two samplers side-by-side on MATH benchmark."""
  if sampler1 not in SUPPORTED_SAMPLERS:
    raise ValueError(f"sampler1 '{sampler1}' not supported. Must be one of {SUPPORTED_SAMPLERS}")
  if sampler2 not in SUPPORTED_SAMPLERS:
    raise ValueError(f"sampler2 '{sampler2}' not supported. Must be one of {SUPPORTED_SAMPLERS}")

  print(f"\n=======================================================")
  print(f"Comparing Samplers: [{sampler1}] vs [{sampler2}]")
  print(f"Model: {model_version} | Samples: {num_samples} | Batch Size: {batch_size}")
  print(f"Sampling Params: temperature={temperature}, top_k={top_k}, top_p={top_p}, seed={seed}")
  print(f"=======================================================\n")

  num_batches = (num_samples + batch_size - 1) // batch_size

  # 1. Run Sampler 1
  results1 = run_sampler_eval(
      sampler_type=sampler1,
      model_version=model_version,
      dataset=dataset,
      batch_size=batch_size,
      num_batches=num_batches,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      seed=seed,
      max_generation_steps=max_generation_steps,
      max_prompt_length=max_prompt_length,
      num_samples=num_samples,
  )

  # 2. Run Sampler 2
  results2 = run_sampler_eval(
      sampler_type=sampler2,
      model_version=model_version,
      dataset=dataset,
      batch_size=batch_size,
      num_batches=num_batches,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      seed=seed,
      max_generation_steps=max_generation_steps,
      max_prompt_length=max_prompt_length,
      num_samples=num_samples,
  )

  # 3. Compare Results
  print("\n=======================================================")
  print(f"Validation & Comparison Summary: [{sampler1}] vs [{sampler2}]")
  print("=======================================================")
  detailed1 = results1.get("detailed_results", [])[:num_samples]
  detailed2 = results2.get("detailed_results", [])[:num_samples]

  text_matches = 0
  answer_matches = 0
  reward_matches = 0

  for i, (d1, d2) in enumerate(zip(detailed1, detailed2)):
    resp1 = d1["responses"][0].strip() if d1["responses"] else ""
    resp2 = d2["responses"][0].strip() if d2["responses"] else ""
    ans1 = d1["extracted_answers"][0] if d1["extracted_answers"] else None
    ans2 = d2["extracted_answers"][0] if d2["extracted_answers"] else None
    corr1 = d1["correct"]
    corr2 = d2["correct"]

    text_match = resp1 == resp2
    ans_match = ans1 == ans2
    reward_match = corr1 == corr2

    if text_match:
      text_matches += 1
    if ans_match:
      answer_matches += 1
    if reward_match:
      reward_matches += 1

    print(f"\n--- Item {i+1} ---")
    print(f"Question (prefix): {d1['question'][:80]}...")
    print(f"Ground Truth Answer: {d1['answer']}")
    print(f"[{sampler1}] Extracted: {ans1} | Correct: {corr1}")
    print(f"[{sampler2}] Extracted: {ans2} | Correct: {corr2}")
    print(f"Exact Text Match: {text_match} | Extracted Answer Match: {ans_match} | Reward Match: {reward_match}")
    if not text_match:
      print(f"  [{sampler1} Output]: {repr(resp1[-100:])}")
      print(f"  [{sampler2} Output]: {repr(resp2[-100:])}")

  total_compared = len(detailed1)
  print(f"\nTotal Compared: {total_compared}")
  print(f"Exact Text Matches: {text_matches}/{total_compared} ({text_matches/total_compared*100:.1f}%)" if total_compared else "N/A")
  print(f"Extracted Answer Matches: {answer_matches}/{total_compared} ({answer_matches/total_compared*100:.1f}%)" if total_compared else "N/A")
  print(f"Reward (Correctness) Matches: {reward_matches}/{total_compared} ({reward_matches/total_compared*100:.1f}%)" if total_compared else "N/A")
  print(f"[{sampler1}] Accuracy: {results1.get('accuracy', 0.0):.2f}%")
  print(f"[{sampler2}] Accuracy: {results2.get('accuracy', 0.0):.2f}%")
  print("=======================================================\n")

  if output_json:
    summary = {
        "sampler1": sampler1,
        "sampler2": sampler2,
        "model_version": model_version,
        "num_samples": num_samples,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "text_matches": text_matches,
        "answer_matches": answer_matches,
        "reward_matches": reward_matches,
        "total_compared": total_compared,
        f"{sampler1}_accuracy": results1.get("accuracy", 0.0),
        f"{sampler2}_accuracy": results2.get("accuracy", 0.0),
    }
    with open(output_json, "w") as f:
      json.dump(summary, f, indent=2)
    print(f"Saved comparison summary to {output_json}")


def main():
  parser = argparse.ArgumentParser(
      description="Compare any two Tunix samplers side-by-side on MATH benchmarks."
  )
  parser.add_argument(
      "--sampler1",
      type=str,
      default=os.environ.get("SAMPLER1", "vanilla"),
      choices=SUPPORTED_SAMPLERS,
      help="First sampler backend to evaluate (default: vanilla).",
  )
  parser.add_argument(
      "--sampler2",
      type=str,
      default=os.environ.get("SAMPLER2", "jax_inference"),
      choices=SUPPORTED_SAMPLERS,
      help="Second sampler backend to evaluate (default: jax_inference).",
  )
  parser.add_argument(
      "--model-version",
      type=str,
      default=os.environ.get("MODEL_VERSION", "Qwen/Qwen3-1.7B-base"),
      help="Model version key in MODEL_MAPPING (default: Qwen/Qwen3-1.7B-base).",
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default=os.environ.get("DATASET", "HuggingFaceH4/MATH-500"),
      help="Dataset name or path (default: HuggingFaceH4/MATH-500).",
  )
  parser.add_argument(
      "--num-samples",
      type=int,
      default=int(os.environ.get("NUM_SAMPLES", "4")),
      help="Total number of samples to evaluate and compare (default: 4).",
  )
  parser.add_argument(
      "--batch-size",
      type=int,
      default=int(os.environ.get("BATCH_SIZE", "2")),
      help="Batch size for evaluation (default: 2).",
  )
  parser.add_argument(
      "--temperature",
      type=float,
      default=float(os.environ.get("TEMPERATURE", "0.0")),
      help="Sampling temperature (default: 0.0 for greedy).",
  )
  parser.add_argument(
      "--top-k",
      type=int,
      default=int(os.environ.get("TOP_K", "1")),
      help="Top-k sampling parameter (default: 1).",
  )
  parser.add_argument(
      "--top-p",
      type=float,
      default=float(os.environ.get("TOP_P", "1.0")),
      help="Top-p nucleus parameter (default: 1.0).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=int(os.environ.get("SEED", "42")),
      help="Random seed for sampling (default: 42).",
  )
  parser.add_argument(
      "--max-generation-steps",
      type=int,
      default=int(os.environ.get("MAX_GENERATION_STEPS", "128")),
      help="Maximum generation steps / new tokens (default: 128).",
  )
  parser.add_argument(
      "--max-prompt-length",
      type=int,
      default=int(os.environ.get("MAX_PROMPT_LENGTH", "1024")),
      help="Maximum prompt length (default: 1024).",
  )
  parser.add_argument(
      "--output-json",
      type=str,
      default=os.environ.get("OUTPUT_JSON", None),
      help="Optional path to output comparison summary JSON.",
  )

  args = parser.parse_args()

  run_comparison(
      sampler1=args.sampler1,
      sampler2=args.sampler2,
      model_version=args.model_version,
      dataset=args.dataset,
      num_samples=args.num_samples,
      batch_size=args.batch_size,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      seed=args.seed,
      max_generation_steps=args.max_generation_steps,
      max_prompt_length=args.max_prompt_length,
      output_json=args.output_json,
  )


if __name__ == "__main__":
  main()
