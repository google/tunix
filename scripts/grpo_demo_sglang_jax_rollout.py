r"""Demo script for GRPO with Llama3.2 3B model.

This script demonstrates how to run GRPO with a Llama3.2 3B or DeepSeek-R1-Distill-Qwen-1.5B model and sglang-jax rollout. It includes
training, evaluation, and inference. In addition, It is based on examples/grpo_demo.ipynb

"""

import argparse
import csv
import functools
import gc
import os
from pathlib import Path
from pprint import pprint
import re
import shutil

from flax import nnx
import grain
import huggingface_hub
import humanize
import jax
import jax.numpy as jnp
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import tensorflow_datasets as tfds
from tqdm.auto import tqdm
import transformers
# from tunix.generate import sampler as sampler_lib
from tunix.generate import mappings
from tunix.generate import sglang_jax_sampler as sampler_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama3_params_lib
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig
from tunix.rl.grpo.grpo_learner import GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import sglang_jax_rollout
from tunix.sft import metrics_logger

# Parse command line options
parser = argparse.ArgumentParser(description="Arguments for GRPO demo")
parser.add_argument(
    "--model-version",
    type=str,
    default="meta-llama/Llama-3.2-3B-Instruct",
    required=False,
    help="The model version to use.",
)
args = parser.parse_args()

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 1.0

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH = [(1, 4), ("fsdp", "tp")]

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 1024
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 2

# === other GRPO configs ===
# The number of iterations per batch (𝜇 in GRPO algo 1).
NUM_ITERATIONS = 1
# The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# Important to keep a high enough value for this, otherwise, the KL divergence
# can increase unchecked.
BETA = 0.08
# Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# stable updates.
EPSILON = 0.2

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
# Increase `NUM_BATCHES` and `MAX_STEPS` for better results.
NUM_BATCHES = 3738
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 2

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
# == Cosine decay with warmup scheduler ==
# Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# steps, and then gradually decrease the learning rate to 0 using cosine
# scheduler.
WARMUP_STEPS = 0.1 * MAX_STEPS
# == Grad clipping ==
# Grad clipping to prevent large gradients. Found this
# important to keep KL divergence in check.
MAX_GRAD_NORM = 0.1

# Checkpoint saving
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}


def show_hbm_usage():
  """Displays memory usage per device."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


repo_id = args.model_version
model_tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"


SYSTEM_PROMPT = f"""You are given a problem. Think about the problem and \
provide your reasoning. Place it between {reasoning_start} and \
{reasoning_end}. Then, provide the final answer (i.e., just one numerical \
value) between {solution_start} and {solution_end}."""

TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def extract_hash_answer(text: str) -> str | None:
  if "####" not in text:
    return None
  return text.split("####")[1].strip()


def _load_from_tfds(data_dir: str, split: str):
  import tensorflow_datasets.text.gsm8k

  return tfds.data_source(
      "gsm8k",
      split=split,
      data_dir=data_dir,
      builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
      download=True,
  )


def download_kaggle_dataset(target_dir="./data/gsm8k"):
  os.makedirs(target_dir, exist_ok=True)
  src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
  src = Path(src)
  dst = Path(target_dir)

  for csv_file in src.glob("*.csv"):  # match all CSV files
    shutil.copy2(csv_file, dst / csv_file.name)
    print(f"Copied {csv_file.name} → {dst/csv_file.name}")
  return target_dir


def get_dataset(data_dir, split="train", source="tfds") -> grain.MapDataset:
  # Download data
  if not os.path.exists(data_dir):
    os.makedirs(data_dir)

  if source == "tfds":
    import tensorflow_datasets.text.gsm8k

    data = tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )

  elif source == "kaggle":
    kaggle_dir = download_kaggle_dataset(data_dir)
    file_name = "main_" + split + ".csv"
    csv_path = os.path.join(kaggle_dir, file_name)  # adjust filename if needed

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
      reader = csv.DictReader(csvfile)
      for row in reader:
        data.append({
            "question": row["question"],
            "answer": row["answer"],
        })

  else:
    raise ValueError(f"Unknown source: {source}")

  def _as_text(v):
    return v if isinstance(v, str) else v.decode("utf-8")

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=42)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": model_tokenizer.apply_chat_template(
                  [
                      {"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": x["question"]},
                  ],
                  tokenize=False,
                  add_generation_prompt=True,
              ),
              # passed to reward functions
              "question": _as_text(x["question"]),
              # passed to reward functions
              "answer": extract_hash_answer(_as_text(x["answer"])),
          }
      )
  )
  return dataset


# source = input("Choose data source [tfds/kaggle]: ").strip().lower()
source = "kaggle"
if source not in ("tfds", "kaggle"):
  print("Invalid choice. Defaulting to 'tfds'.")
  source = "tfds"

print(f"Using data source: {source}")

dataset = get_dataset(TRAIN_DATA_DIR, "train", source).batch(
    TRAIN_MICRO_BATCH_SIZE
)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_DIR, "test", source).batch(
    TRAIN_MICRO_BATCH_SIZE
)[:NUM_TEST_BATCHES]

dataset_lengths = (
    len(train_dataset),
    len(val_dataset) if val_dataset is not None else 0,
    len(test_dataset),
)
print(f"dataset contains {dataset_lengths} of batches")

for ele in train_dataset[:1]:
  pprint(ele)

# Log in
# if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
#  kagglehub.login()

# model_path = {
#     "gemma2": "google/gemma-2/flax/",
# }
# model_family = "gemma2"
# model_version = "gemma2-2b-it"
# print(f"{model_path[model_family]}{model_version}")
#
# kaggle_ckpt_path = kagglehub.model_download(
#     f"{model_path[model_family]}{model_version}"
# )


def download_from_huggingface(repo_id: str, model_path: str):
  """Download checkpoint files from huggingface."""
  print("Make sure you logged in to the huggingface cli.")
  all_files = huggingface_hub.list_repo_files(repo_id)
  filtered_files = [f for f in all_files if not f.startswith("original/")]

  for filename in filtered_files:
    huggingface_hub.hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=model_path
    )
  print(f"Downloaded {filtered_files} to: {model_path}")


import tempfile

temp_dir = tempfile.gettempdir()
model_path = os.path.join(temp_dir, "models", repo_id)
download_from_huggingface(repo_id=repo_id, model_path=model_path)

# if model_family == "gemma2":
#   params = params_lib.load_and_format_params(
#       os.path.join(kaggle_ckpt_path, "gemma2-2b-it")
#   )
#   gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")
#   checkpointer = ocp.StandardCheckpointer()
#   _, state = nnx.split(gemma)
#   checkpointer.save(os.path.join(INTERMEDIATE_CKPT_DIR, "state"), state)
#   checkpointer.wait_until_finished()
#   # Delete the intermediate model to save memory.
#   del params
#   del gemma
#   del state
#   gc.collect()
#
# def get_gemma_ref_model(ckpt_path):
#   mesh = jax.make_mesh(*MESH)
#   model_config = gemma_lib.ModelConfig.gemma2_2b()
#   abs_gemma: nnx.Module = nnx.eval_shape(
#       lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
#   )
#   abs_state = nnx.state(abs_gemma)
#   abs_state = jax.tree.map(
#       lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
#       abs_state,
#       nnx.get_named_sharding(abs_state, mesh),
#   )
#   checkpointer = ocp.StandardCheckpointer()
#   restored_params = checkpointer.restore(ckpt_path, target=abs_state)
#
#   graph_def, _ = nnx.split(abs_gemma)
#   gemma = nnx.merge(graph_def, restored_params)
#   return gemma, mesh, model_config
#


def get_lora_model(base_model, mesh):
  # lora_provider = qwix.LoraProvider(
  #     module_path=(
  #         ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|"
  #         ".*attn_vec_einsum"
  #     ),
  #     rank=RANK,
  #     alpha=ALPHA,
  # )
  #
  # model_input = base_model.get_model_input()
  # lora_model = qwix.apply_lora_to_model(
  #     base_model, lora_provider, **model_input
  # )
  lora_model = base_model
  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model


# Reference model
# if model_family == "gemma2":
#   ref_model, mesh, model_config = get_gemma_ref_model(
#       ckpt_path=os.path.join(INTERMEDIATE_CKPT_DIR, "state")
#   )
def load_model(model_version: str, enable_lora: bool = False):
  model_config = {
      "meta-llama/Llama-3.2-3B-Instruct": llama_lib.ModelConfig.llama3_2_3b,
      "meta-llama/Llama-3.1-8B-Instruct": llama_lib.ModelConfig.llama3_1_8b,
      "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (
          qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1_5b
      ),
  }
  assert (
      model_version in model_config
  ), f"Invalid model version: {model_version}"
  model_config = model_config[model_version]()

  mesh_shape = (1, len(jax.devices()))  # e.g., (1, 8) for v2-8
  if "Qwen" in model_version:
    mesh_shape = (1, 2)  # because the num_key_value_heads is 2
  axis_names = ("fsdp", "tp")
  mesh = jax.make_mesh(mesh_shape, axis_names, devices=jax.devices())
  if "Llama-3" in model_version:
    model = llama3_params_lib.create_model_from_safe_tensors(
        model_path, model_config, mesh
    )
  elif "Qwen" in model_version:
    model = qwen2_params_lib.create_model_from_safe_tensors(
        model_path, model_config, mesh
    )
  return model, mesh, model_config


print("before reference")
ref_model, mesh, model_config = load_model(repo_id)
show_hbm_usage()
# Policy model

lora_policy = ref_model
print("after lora_policy")
show_hbm_usage()
# nnx.display(lora_policy)

# if model_family == "gemma2":
#   tokenizer = tokenizer_lib.Tokenizer(
#       tokenizer_path=os.path.join(kaggle_ckpt_path, "tokenizer.model")
#   )

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_format.search(
    f"{reasoning_start}Let me"
    f" think!{reasoning_end}{solution_start}2{solution_end}",
)


def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]


def match_format_approximately(prompts, completions, **kwargs):
  scores = []

  for completion in completions:
    score = 0
    response = completion
    # Count how many keywords are seen - we penalize if too many!
    # If we see 1, then plus some points!
    score += 0.5 if response.count(reasoning_start) == 1 else -0.5
    score += 0.5 if response.count(reasoning_end) == 1 else -0.5
    score += 0.5 if response.count(solution_start) == 1 else -0.5
    score += 0.5 if response.count(solution_end) == 1 else -0.5
    scores.append(score)
  return scores


def check_answer(prompts, completions, answer, **kwargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  assert len(extracted_responses) == len(
      answer
  ), f"{extracted_responses} and {answer} have mismatching length"
  for guess, true_answer in zip(extracted_responses, answer):
    score = 0
    if guess is None:
      scores.append(0)
      continue
    # Correct answer gets 3 points!
    if guess == true_answer:
      score += 3.0
    # Match if spaces are seen
    elif guess.strip() == true_answer.strip():
      score += 1.5
    else:
      # We also reward it if the answer is close via ratios!
      # Ie if the answer is within some range, reward it!
      try:
        ratio = float(guess) / float(true_answer)
        if ratio >= 0.9 and ratio <= 1.1:
          score += 0.5
        elif ratio >= 0.8 and ratio <= 1.2:
          score += 0.25
        else:
          score -= 1.0  # Penalize wrong answers
      except:
        score -= 0.5  # Penalize
    scores.append(score)
  return scores


match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)
match_numbers.findall(f"{solution_start}  0.34  {solution_end}")


def check_numbers(prompts, completions, answer, **kwargs):
  question = kwargs["question"]
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_numbers.search(r)) is not None else None
      for r in responses
  ]

  scores = []
  print("START ============================")
  print(f"Question: {question[0]}")
  print(f"Answer: {answer[0]}")
  print(f"Response: {responses[0]}")
  print(f"Extracted: {extracted_responses[0]}")
  print("END ==============================")
  for guess, true_answer in zip(extracted_responses, answer):
    if guess is None:
      scores.append(0)
      continue
    # Convert to numbers
    try:
      true_answer = float(true_answer.strip())
      guess = float(guess.strip())
      scores.append(1.5 if guess == true_answer else 0.0)
    except:
      scores.append(0)
      continue
  return scores


def generate(
    question, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None
):
  """Given prompt, generates text."""

  if isinstance(question, str):
    input_batch = [
        model_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    ]
  else:
    input_batch = [
        model_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in question
    ]

  out_data = sampler(
      input_strings=input_batch,
      max_generation_steps=128,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed if seed is not None else None,
  )

  output = out_data.text
  if isinstance(question, str):
    return output[0]
  return output


def evaluate(
    dataset,
    sampler,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_passes=1,
    corr_lst=False,
    make_lst=False,
):
  """Computes accuracy and percentage of outputs matching the format."""

  response_lst = []
  corr = 0
  partially_corr = 0
  corr_format = 0
  total = 0

  for batch in tqdm(dataset):
    print("in eval")
    show_hbm_usage()
    answers = batch["answer"]
    questions = batch["question"]

    multiple_call_responses = [[] for _ in range(len(questions))]
    for p in range(num_passes):
      responses = generate(
          questions, sampler, temperature, top_k, top_p, seed=p
      )
      for idx, response in enumerate(responses):
        multiple_call_responses[idx].append(response)

    for question, multiple_call_response, answer in zip(
        questions, multiple_call_responses, answers
    ):
      # check answer
      corr_ctr_per_question = 0
      partially_corr_per_question = 0
      corr_format_per_question = 0
      for response in multiple_call_response:
        extracted_response = (
            guess.group(1)
            if (guess := match_numbers.search(response)) is not None
            else "-1000000"
        )
        try:
          if float(extracted_response.strip()) == float(answer.strip()):
            corr_ctr_per_question += 1

          ratio = float(extracted_response.strip()) / float(answer.strip())
          if ratio >= 0.9 and ratio <= 1.1:
            partially_corr_per_question += 1
        except:
          print("SKIPPED")

        # check format
        if match_format.search(response) is not None:
          corr_format_per_question += 1

        if (
            corr_ctr_per_question > 0
            and partially_corr_per_question > 0
            and corr_format_per_question > 0
        ):
          break

      if corr_ctr_per_question > 0:
        corr += 1
        if corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      else:
        if not corr_lst and make_lst:
          response_lst.append((question, answer, multiple_call_response))
      if partially_corr_per_question > 0:
        partially_corr += 1
      if corr_format_per_question > 0:
        corr_format += 1

      total += 1
      if total % 10 == 0:
        print(
            f"===> {corr=}, {total=}, {corr / total * 100=}, "
            f"{partially_corr / total * 100=}, {corr_format / total * 100=}"
        )

  to_return = (
      corr,
      total,
      corr / total * 100,
      partially_corr / total * 100,
      corr_format / total * 100,
  )
  if make_lst:
    return to_return, response_lst
  return to_return


mapping_config = mappings.MappingConfig.build(
    model=ref_model, backend="sglang_jax"
)

sglang_jax_config = sampler_lib.SglangJaxConfig(
    model_version=model_path,
    context_length=2048,
    mesh=mesh,
    mem_fraction_static=0.3,
    init_with_random_weights=True,
    disable_radix_cache=True,
    enable_deterministic_sampling=False,
    mapping_config=mapping_config,
)

# sampler = sampler_lib.SglangJaxSampler(
#    tokenizer=tokenizer,
#    config=sglang_jax_config,
# )
# The evaluation might take up to couple of minutes to finish. Please be patient.
# print("before eval")
show_hbm_usage()

# (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
#    test_dataset,
#    sampler,
#    **GENERATION_CONFIGS["greedy"],
# )
# print(
#    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
#    f" {format_accuracy=}%"
# )

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/content/tmp/tensorboard/grpo", flush_every_n_steps=20
)

optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    ),
    b1=B1,
    b2=B2,
    weight_decay=WEIGHT_DECAY,
)

if MAX_GRAD_NORM is not None:
  optimizer = optax.chain(
      optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM),
      optimizer,
  )

# Training config
cluster_config = rl_cluster_lib.ClusterConfig(
    role_to_mesh={
        rl_cluster_lib.Role.ACTOR: mesh,
        rl_cluster_lib.Role.REFERENCE: mesh,
        rl_cluster_lib.Role.ROLLOUT: mesh,
    },
    rollout_engine="sglang_jax",
    offload_to_cpu=False,
    training_config=rl_cluster_lib.RLTrainingConfig(
        actor_optimizer=optimizer,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
        train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
        # metrics logging
        metrics_logging_options=metrics_logging_options,
        # checkpoint saving
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        rollout_sglang_jax_model_version=model_path,
        rollout_sglang_jax_context_length=2048,
        rollout_sglang_jax_mem_fraction_static=0.4,
        rollout_sglang_jax_init_with_random_weights=True,
        rollout_sglang_jax_disable_radix_cache=True,
        rollout_sglang_jax_enable_deterministic_sampling=False,
    ),
)

grpo_config = GRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
)
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_policy,
    reference=ref_model,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# GRPO Trainer
grpo_trainer = GRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    grpo_config=grpo_config,
)

with mesh:
  grpo_trainer.train(dataset)

# Load checkpoint first.

trained_ckpt_path = os.path.join(
    CKPT_DIR, "actor", str(MAX_STEPS), "model_params"
)

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_policy, nnx.Param),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_policy,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_policy, nnx.Param),
        trained_lora_params,
    ),
)

sampler = sampler_lib.SglangJaxSampler(
    tokenizer=tokenizer,
    config=sglang_jax_config,
    # cache_config=sampler_lib.CacheConfig(
    #     cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
    #     num_layers=model_config.num_layers,
    #     num_kv_heads=model_config.num_kv_heads,
    #     head_dim=model_config.head_dim,
    # ),
)

# The evaluation might take up to couple of minutes to finish. Please be patient.
(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%, {format_accuracy}"
)
