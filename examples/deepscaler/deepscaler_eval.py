# %%
from pprint import pprint
import datasets as datasets_lib
import grain
import pandas as pd
import os
import fsspec

import transformers
from tunix.generate import mappings

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

file_open = fsspec.open

from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib
from tunix.generate import sampler as sampler_lib
from tunix.utils import math_utils

from typing import Any, Dict, Optional
import jax
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from tqdm.auto import tqdm
import re

THOUGHT_DELIMITER_END = "</think>"

class Qwen25MathEvaluator:

  def __init__(
      self,
      model_config,
      model_version: str,
      model_path: str,
      dataset: str,
      mesh_config=None,
      max_prompt_length: int = 1024,  # Increased from 512
      max_generation_steps: int = 1024,  # Increased from 512
      sampler_type: str = "vllm",  # vanilla, vllm, or sglang-jax
  ):
    self.model_config = model_config
    self.model_version = model_version
    self.model_path = model_path
    self.dataset = dataset
    self.max_prompt_length = max_prompt_length
    self.max_generation_steps = max_generation_steps
    self.sampler_type = sampler_type

    if mesh_config is None:
      # Default: 4-way tensor parallelism
      mesh_config = [[1, 4], ["fsdp", "tp"]]
    self.mesh = jax.make_mesh(*mesh_config, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_config[0]))
    self.tokenizer = None
    self.model = None
    self.sampler = None

    print(f"Initializing {self.model_version} evaluator")
    print(f"Model path: {model_path}")
    print(f"Mesh config: {mesh_config}")
    print(f"Available devices: {jax.devices()}")

  def model_from_safe_tensors(self):
    print("Loading model from safe tensors...")
    with self.mesh:
      self.model = qwen2_params_lib.create_model_from_safe_tensors(
          file_dir=self.model_path, config=self.model_config, mesh=self.mesh
      )

  def load_model(self):
    print("Loading model components...")

    print("Loading tokenizer...")

    # Huggingface API doesn't work with gcs, OSS loads from model directly
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_version)

    print("Setting up model config...")

    self.model_from_safe_tensors()
    print("Model loaded successfully!")
    print("Creating sampler...")
    cache_config = sampler_lib.CacheConfig(
        cache_size=self.max_prompt_length + self.max_generation_steps + 100,
        num_layers=self.model_config.num_layers,
        num_kv_heads=self.model_config.num_kv_heads,
        head_dim=self.model_config.head_dim,
    )

    from tunix.generate import vllm_sampler   # pylint: disable=g-import-not-at-top

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=None,
        model=self.model,
        backend="vllm_jax",
    )
    self.sampler_vllm = vllm_sampler.VllmSampler(
        tokenizer=self.tokenizer,
        config=vllm_sampler.VllmConfig(
            mesh=self.mesh,
            hbm_utilization=0.8,
            init_with_random_weights=False,
            mapping_config=mapping_config,
            engine_kwargs={
                "model": self.model_version,
                "max_model_len": (
                    self.max_prompt_length + self.max_generation_steps + 100
                ),
                "max_num_seqs": 30,
                "max_num_batched_tokens": 30 * 10 * 1024 // 8,
            },
        ),
    )
    # sync weights from self.model to the sampler's internal model
    print("Syncing model weights to VLLM sampler...")
    self.sampler_vllm.update_params(nnx.state(self.model))

    print("Sampler created successfully!")

    return {
        "model": self.model,
        "tokenizer": self.tokenizer,
        "sampler": self.sampler,
        "config": self.model_config,
    }

  def load_dataset(self, split: str = "test") -> grain.MapDataset:
    print(f"Loading {self.dataset} dataset (split: {split})...")
    def preprocess_fn(example, index):
      return {
          "question": example["problem"],
          "ground_truth": example["answer"],
          "data_source": "math",
      }

    with file_open(self.dataset) as train_f:
      train_df = pd.read_json(train_f)

    train_ds = Dataset.from_pandas(train_df).map(preprocess_fn, with_indices=True)

    def process_item(item):
      question = item["question"]
      answer = item["answer"]

      instruction = (
          "Let's think step by step, and put your final answer within \\boxed{}."
      )
      question = f"{question} {instruction}"

      return {
          "prompts": question,
          "question": question,
          "answer": answer,
      }

    train_ds = grain.MapDataset.source(train_ds).map(process_item)
    print("\n" + "=" * 60)
    print("DEBUG: First formatted prompt:")
    first_item = train_ds[0]
    print(first_item)
    print("=" * 60 + "\n")
    return train_ds

  def generate(
      self,
      prompts: list[str],
      temperature: float = 0.6,
      top_k: int = 50,
      top_p: float = 0.95,
      seed: int | None = None,
  ) -> list[str]:
    if self.tokenizer is None:
      raise RuntimeError(
          "Model components not loaded. Call load_model() first."
      )
    max_length = max(len(self.tokenizer.encode(p)) for p in prompts)
    cache_size = self.max_prompt_length + self.max_generation_steps + 100
    safe_gen_length = min(
        self.max_generation_steps,
        cache_size - max_length - 100,  # 100 token buffer
    )
    if safe_gen_length < 256:
      print(
          f"WARNING: Short generation length ({safe_gen_length} tokens) due to"
          f" long prompt ({max_length} tokens)"
      )


    # Generate
    out_data = self.sampler_vllm(
        input_strings=prompts,
        max_generation_steps=safe_gen_length,
        max_prompt_length=self.max_prompt_length,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=None,
        echo=False,
        pad_output=True,
    )
    return out_data.text

  def evaluate(
      self,
      batch_size: int = 8,
      num_batches: int | None = None,
      temperature: float = 0.6,
      top_k: Optional[int] = None,
      top_p: Optional[float] = 0.95,
      num_passes: int = 1,
      debug_first_n: int = 3,  # NEW: Debug first N examples
  ) -> Dict[str, Any]:
    print("=" * 60)
    print("Starting Evaluation")
    print("=" * 60)
    print("Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num batches: {num_batches or 'all'}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")
    print(f"  Passes per question: {num_passes}")
    print(f"  Debug first N examples: {debug_first_n}")
    print("=" * 60)

    # Load dataset
    dataset = self.load_dataset()

    # Create batched dataset
    if num_batches is not None:
      dataset = dataset.batch(batch_size)[:num_batches]
    else:
      dataset = dataset.batch(batch_size)

    correct = 0
    total = 0
    results = []
    debug_count = 0

    # Evaluate batch by batch
    for batch_idx, batch in enumerate(tqdm(dataset, desc="Evaluating")):
      prompts = batch["prompts"]

      questions = batch["question"]
      answers = batch["answer"]

      responses_collection = [[] for _ in range(len(prompts))]
      for pass_idx in range(num_passes):
        batch_response = self.generate(
            prompts=prompts,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=pass_idx
            if self.sampler_type != "vllm"
            else None,  # vllm handles seeding differently
        )
        for i, r in enumerate(batch_response):
          responses_collection[i].append(r)

      for prompt, question, answer, responses in zip(
          prompts, questions, answers, responses_collection
      ):
        is_correct = False
        extracted_answers = []
        answer_correct = []
        for response in responses:
          # Extract solution.
          if THOUGHT_DELIMITER_END in response:
            model_solution = response.split(THOUGHT_DELIMITER_END)[1]
          else:
            model_solution = response

          model_answer = math_utils.extract_answer(model_solution)
          if model_answer:
            # Process the ground truth(s)
            ground_truths = answer
            if ground_truths is not None:
              # Convert single answer to list for uniform processing
              if isinstance(ground_truths, str | float | int):
                ground_truths = [ground_truths]

              # Process each ground truth
              processed_ground_truths = []
              for truth in ground_truths:
                truth = str(truth)
                if "\\boxed" in truth:
                  processed_truth = math_utils.extract_answer(truth)
                  if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
                else:
                  processed_ground_truths.append(truth)

              # Check against all possible correct answers
              if processed_ground_truths:
                for ground_truth in processed_ground_truths:
                  print(f"{model_answer = }")
                  print(f"{ground_truth = }")
                  is_correct = (
                      math_utils.grade_answer_mathd(model_answer, ground_truth)
                      or math_utils.grade_answer_sympy(model_answer, ground_truth)
                      or math_utils.grade_answer_special_handling(
                          model_answer, ground_truth
                      )
                  )
          if is_correct:
            break

        if is_correct:
          correct += 1

        should_debug = debug_count < debug_first_n

        if should_debug:
          print(f"\n{'='*60}")
          print(f"DEBUG Example {debug_count + 1}/{debug_first_n}")
          print(f"Question: {question[:]}")
          print("=" * 60 + "\n")
          print(f"Ground truth: {answer}")
          print("=" * 60 + "\n")
          print(f"Prompt (first 300 chars): {prompt[:]}")
          if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            print(f"Prompt length: {len(self.tokenizer.encode(prompt))} tokens")
          print("=" * 60 + "\n")
          for i, (response, ans, cor) in enumerate(
              zip(responses, extracted_answers, answer_correct)
          ):
            print(f"Response {i}: {response}")
            print("=" * 120 + "\n")
            print(f"\nExtracted answer{i}: {ans}")
            print(f"Is correct: {cor}")
          print(f"Final result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
          print(
              f"Running accuracy: {correct}/{total+1} ="
              f" {(correct/(total+1)*100):.2f}%"
          )
          debug_count += 1

        total += 1

        # Store result
        results.append({
            "question": question,
            "answer": answer,
            "responses": responses,
            "extracted_answers": extracted_answers,
            "correct": is_correct,
        })

        # Print progress
        if total % 10 == 0:
          current_acc = (correct / total * 100) if total > 0 else 0
          print(f"\nProgress: {correct}/{total} = {current_acc:.2f}%")

    # Calculate final metrics
    accuracy = (correct / total * 100) if total > 0 else 0

    eval_results = {
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "num_passes": num_passes,
        "detailed_results": results,
    }

    return eval_results
# %%

DATA_PATH_PREFIX = "gs://tunix/data"

DEEPSCALER_DATA_PATH = os.path.join(
    DATA_PATH_PREFIX, "DeepScaleR-Preview-Dataset/deepscaler.json"
)

model_version = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id=model_version, max_workers=16)


mesh_config = [[1, 2], ["fsdp", "tp"]]  # 2-way tensor parallelism

num_batches_env = os.environ.get("NUM_BATCHES")
num_batches = int(num_batches_env) if num_batches_env and int(num_batches_env) > 0 else None

# model_version = "Qwen/Qwen2.5-1.5B-Instruct"
model_version = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
dataset = DEEPSCALER_DATA_PATH
model_config = qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()

evaluator = Qwen25MathEvaluator(
    model_config=model_config,
    model_version=model_version,
    model_path=model_path,
    dataset=dataset,
    mesh_config=mesh_config,
    max_prompt_length=1024,  # Increased
    max_generation_steps=2048,  # Increased
)

evaluator.load_model()

print("\nStarting evaluation...")
results = evaluator.evaluate(
    batch_size=8,
    num_batches=num_batches,
    temperature=0.6,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    debug_first_n=5,
)

# Print results
print("\n" + "=" * 60)
print("Evaluation Results")
print("=" * 60)
print(f"Model: {model_path}")
print(f"Dataset: {dataset}")
print(f"Correct: {results['correct']}/{results['total']}")
print(f"Accuracy: {results['accuracy']:.2f}%")
print("=" * 60)
