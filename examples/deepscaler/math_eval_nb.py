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

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )

  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"

  import contextlib
  cm = contextlib.nullcontext()

  file_open = fsspec.open

with cm:
  from tunix.models.qwen2 import model as qwen2_lib
  from tunix.models.qwen2 import params as qwen2_params_lib
  from tunix.generate import sampler as sampler_lib
  from tunix.utils import math_utils
# %%
from typing import Any, Dict, Optional
import jax
from tqdm.auto import tqdm
import re

# Only used for Math500
def extract_answer_robust(passage: str) -> str:
  if not passage:
    return ""

  # Pattern 1: Look for \boxed{...} with proper matching braces
  # This handles nested braces like \boxed{\frac{1}{2}}
  stack = []
  i = passage.find("\\boxed")
  if i != -1:
    i += 6  # Skip '\boxed'
    # Skip whitespace
    while i < len(passage) and passage[i].isspace():
      i += 1
    if i < len(passage) and passage[i] == "{":
      i += 1
      start = i
      brace_count = 1
      while i < len(passage) and brace_count > 0:
        if passage[i] == "{":
          brace_count += 1
        elif passage[i] == "}":
          brace_count -= 1
        i += 1
      if brace_count == 0:
        answer = passage[start : i - 1]
        return answer.strip()

  # Pattern 2: Lenient matching - extract up to common terminators
  patterns = [
      r"\\boxed\{([^}]+)\}",  # Standard
      r"boxed\{([^}]+)\}",  # Missing backslash
      r"\\boxed\s*\{(.+?)(?:\.\s|\)\.|\.$)",  # Ends with period
      r"final answer is.*?\\boxed\{([^}]+)",  # "final answer is"
      r"answer is.*?\\boxed\{([^}]+)",
  ]

  for pattern in patterns:
    matches = re.findall(pattern, passage, re.IGNORECASE | re.DOTALL)
    if matches:
      answer = matches[-1].strip()
      # Clean up
      answer = answer.rstrip(".,;:)")
      # Try to fix common LaTeX issues
      if "\\frac" in answer:
        # Count braces - each \frac needs 2 pairs
        open_braces = answer.count("{")
        close_braces = answer.count("}")
        if open_braces > close_braces:
          answer += "}" * (open_braces - close_braces)
      return answer

  # Pattern 3: Super lenient - just find anything after boxed{
  super_lenient = r"boxed\s*\{([^\n]{1,200})"
  matches = re.findall(super_lenient, passage, re.IGNORECASE)
  if matches:
    answer = matches[-1]
    # Find the first reasonable endpoint
    for char in [".", ")", "\n", "The ", "Thus", "Therefore"]:
      if char in answer:
        answer = answer[: answer.index(char)]
        break
    return answer.strip().rstrip(".,;:)")

  return ""
# %%

# only used for AIME-2024
THOUGHT_DELIMITER_END = "</think>"
def evaluate_correctness(response: Any, ground_truths: Any) -> bool:
  """Evaluate the correctness of a response."""
  if response is None or response == "":
    print(f"{response=} {ground_truths=} IS NOT CORRECT")
    return False
  if THOUGHT_DELIMITER_END in response:
    response = response.split(THOUGHT_DELIMITER_END)
    model_solution = response[1]
    print(f"{model_solution=} after THOUGHT_DELIMITER_END in evaluate_correctness")
  else:
    print(f"{response=} in evaluate_correctness")
    model_solution = response
  model_answer = math_utils.extract_answer(model_solution)
  if model_answer is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  if ground_truths is None:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
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
  print(f"{processed_ground_truths=} in evaluate_correctness")
  if not processed_ground_truths:
    print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
    return False
  # Check against all possible correct answers
  for ground_truth in processed_ground_truths:
    is_correct = math_utils.grade_answer_mathd(
        model_answer, ground_truth
    ) or math_utils.grade_answer_sympy(model_answer, ground_truth)
    if is_correct:
      print(f" {model_answer=} {ground_truth=} IS CORRECT")
      return True
  print(f" {model_answer=} {ground_truths=} IS NOT CORRECT")
  return False
# %%

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
      sampler_type: str = "vanilla",  # vanilla, vllm, or sglang-jax
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
    self.mesh = jax.make_mesh(*mesh_config)
    self.tokenizer = None
    self.model = None
    self.sampler = None

    print(f"Initializing {self.model_version} evaluator")
    print(f"Model path: {model_path}")
    print(f"Mesh config: {mesh_config}")
    print(f"Available devices: {jax.devices()}")

  def load_model(self):
    print("Loading model components...")

    print("Loading tokenizer...")

    # Huggingface API doesn't work with gcs, OSS loads from model directly
    tokenizer_source = self.model_version if NOTEBOOK_ENV != "g3" else self.model_path
    self.tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source, trust_remote_code=True
    )

    print("Setting up model config...")


    print("Loading model from safe tensors...")
    with self.mesh:
      self.model = qwen2_params_lib.create_model_from_safe_tensors(
          file_dir=self.model_path, config=self.model_config, mesh=self.mesh
      )

    print("Model loaded successfully!")
    print("Creating sampler...")
    cache_config = sampler_lib.CacheConfig(
        cache_size=self.max_prompt_length + self.max_generation_steps + 100,
        num_layers=self.model_config.num_layers,
        num_kv_heads=self.model_config.num_kv_heads,
        head_dim=self.model_config.head_dim,
    )

    if self.sampler_type == "vanilla":
      self.sampler_vanilla = sampler_lib.Sampler(
          transformer=self.model,
          tokenizer=self.tokenizer,
          cache_config=cache_config,
      )
    elif self.sampler_type == "sglang-jax":
      from tunix.google.stubs import sglang_jax_sampler_stub as sglang_jax_sampler  # pylint: disable=g-import-not-at-top

      mapping_config = mappings.MappingConfig.build(
          mapping_obj=None,
          model=self.model,
          backend="sglang_jax",
      )
      self.sampler_sglang = sglang_jax_sampler.SglangJaxSampler(
          tokenizer=self.tokenizer,
          config=sglang_jax_sampler.SglangJaxConfig(
              mesh=self.mesh,
              context_length=self.max_prompt_length
              + self.max_generation_steps
              + 100,
              model_version=self.model_version,
              mem_fraction_static=0.4,
              init_with_random_weights=False,
              disable_radix_cache=True,
              enable_deterministic_sampling=False,
              mapping_config=mapping_config,
          ),
      )
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")

    print("Sampler created successfully!")

    return {
        "model": self.model,
        "tokenizer": self.tokenizer,
        "sampler": self.sampler,
        "config": self.model_config,
    }

  def load_dataset(self, split: str = "test") -> grain.MapDataset:
    print(f"Loading {self.dataset} dataset (split: {split})...")

    def preprocess_fn(example, idx):
      return {
          "question": example["problem"],
          "answer": example["answer"],
          "data_source": "math",
          }

    with file_open(self.dataset, "rb") as test_f:
      if self.dataset.endswith("jsonl"):
        test_df = pd.read_json(test_f, lines=True)
      elif self.dataset.endswith("json"):
        test_df = pd.read_json(test_f)
      else:
        test_df = pd.read_parquet(test_f)

    test_ds = Dataset.from_pandas(test_df).map(preprocess_fn, with_indices=True)


    print(f"Loaded {len(test_ds)} examples")
    print("Example data:")
    pprint(test_ds[0])

    def process_item(item):
      question = item["question"]
      answer = item["answer"]

      if "aime_2024" in self.dataset:
        instruction = "Let's think step by step, and put your final answer within \\boxed{}."
        prompt = f"{question} {instruction}"
      else:
        instruction = "Please reason step by step. Your final answer must appear inside \\boxed{...} and nothing else."
        prompt = f"{instruction} {question}"
      prompt = self.tokenizer.apply_chat_template(
          [{"role": "user", "content": prompt}],
          tokenize=False, add_generation_prompt=True)

      return {
          "prompt": prompt,
          "question": question,
          "answer": answer,
      }

    dataset = grain.MapDataset.source(test_ds).map(process_item)
    print("\n" + "=" * 60)
    print("DEBUG: First formatted prompt:")
    first_item = dataset[0]
    print(first_item["prompt"])
    print("=" * 60 + "\n")

    return dataset

  def generate(
      self,
      prompts: list[str],
      temperature: float = 0.6,
      top_k: int = 50,
      top_p: float = 0.95,
      seed: int | None = None,
  ) -> str:
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

    stop_token_id = self.tokenizer.encode("<|im_end|>")[0]

    # Generate
    if self.sampler_type == "vanilla":
      out_data = self.sampler_vanilla(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          temperature=temperature,
          top_k=top_k,
          top_p=top_p,
          echo=False,
          eos_tokens=[stop_token_id],
          seed=jax.random.PRNGKey(seed) if seed is not None else None,
      )
    elif self.sampler_type == "sglang-jax":
      out_data = self.sampler_sglang(
          input_strings=prompts,
          max_generation_steps=safe_gen_length,
          max_prompt_length=self.max_prompt_length,
          temperature=temperature,
          top_p=top_p,
          top_k=top_k,
          seed=seed,
          echo=False,
          pad_output=True,
      )
    else:
      raise ValueError(f"Unsupported sampler type: {self.sampler_type}")
    return out_data.text[0]

  def evaluate(
      self,
      batch_size: int = 8,
      num_batches: int | None = None,
      temperature: float = 0.6,
      top_k: Optional[int] = 50,
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
      prompts = batch["prompt"]

      questions = batch["question"]
      answers = batch["answer"]

      responses_collection = [[] for _ in range(len(prompts))]
      for pass_idx in range(num_passes):
        batch_response = self.generate(
            prompts=prompts,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=pass_idx,
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
          # Grade answer using both methods from utils.py
          if "aime_2024" in self.dataset:
            is_correct = evaluate_correctness(response, answer)
          else:
            model_answer = extract_answer_robust(response)
            extracted_answers.append(model_answer)

            if model_answer is None:
              continue
            # Grade answer using both methods from utils.py
            is_correct = math_utils.grade_answer_mathd(
                model_answer, answer
            ) or math_utils.grade_answer_sympy(model_answer, answer)

          answer_correct.append(is_correct)

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

if NOTEBOOK_ENV == "g3":
    DATA_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
    MODEL_PATH_PREFIX = "/GOOGLE_INTERNAL_STOAGE_PATH/gg-d/home/qwix-dev"
else:
    DATA_PATH_PREFIX = "gs://tunix/data"
    MODEL_PATH_PREFIX = "gs://tunix/models"

MATH_500_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "rl/data/MATH-500/test.jsonl")
AIME_2024_DATA_PATH = os.path.join(DATA_PATH_PREFIX, "rl/data/HuggingFaceH4/aime_2024/train-00000-of-00001.parquet")
MODEL_MAPPING = {
    "qwen2.5-1.5b-it": (qwen2_lib.ModelConfig.qwen2_5_1_5b() ,os.path.join(MODEL_PATH_PREFIX, "qwen2_5/torch/1.5b-it")),
    "DeepSeek-R1-Distill-Qwen-1.5B": (qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1_5b(),  os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B")),
    "DeepScaleR-1.5B-Preview": (qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1_5b(), os.path.join(MODEL_PATH_PREFIX, "DeepScaleR-1.5B-Preview")),
}

mesh_config = [[1, 4], ["fsdp", "tp"]]  # 4-way tensor parallelism
# %%
#MATH-500
model_version = "Qwen/Qwen2.5-1.5B-Instruct"
dataset = MATH_500_DATA_PATH
model_config, model_path = MODEL_MAPPING[model_version]

evaluator = Qwen25MathEvaluator(
    model_config=model_config,
    model_version=model_version,
    model_path=model_path,
    dataset=dataset,
    mesh_config=mesh_config,
    max_prompt_length=1024,  # Increased
    max_generation_steps=1024,  # Increased
)

evaluator.load_model()

print("\nStarting evaluation...")
results = evaluator.evaluate(
    batch_size=8,
    num_batches=None,
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
# %%
# AIME-2024
model_version = "agentica-org/DeepScaleR-1.5B-Preview"
dataset = AIME_2024_DATA_PATH
model_config, model_path = MODEL_MAPPING[model_version]

evaluator = Qwen25MathEvaluator(
    model_config=model_config,
    model_version=model_version,
    model_path=model_path,
    dataset=dataset,
    mesh_config=mesh_config,
    max_prompt_length=2048,  # Increased
    max_generation_steps=32768,  # Increased
)

evaluator.load_model()

print("\nStarting evaluation...")

results = evaluator.evaluate(
    batch_size=1,
    num_batches=None,
    temperature=0.6,
    top_k=None,
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
# %%
