import contextlib
import functools
import json
import logging
import os
from pprint import pprint
import re

from flax import nnx
import grain
import humanize
import jax
from jax import numpy as jnp
import optax
from orbax import checkpoint as ocp
from qwix import lora
from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.models.gemma import data as data_lib
from tunix.models.gemma import gemma as gemma_lib
from tunix.models.llama3 import model as llama_lib #YY
from tunix.models.llama3.model import ModelConfig as llama_model_config
from tunix.models.llama3.params import create_model_from_safe_tensors, create_model_from_safe_tensors_original #YY

from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner
from tunix.rl.rollout import vanilla_rollout
from tunix.rl.rollout import vllm_rollout #YY
from tunix.generate import vllm_sampler #YY
from tunix.sft import metrics_logger

from transformers import AutoTokenizer #YY

os.environ["TPU_BACKEND_TYPE"] = "jax"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
# os.environ["JAX_RANDOM_WEIGHTS"] = "True"
os.environ["SKIP_JAX_PRECOMPILE"] = "1"  # Disable precompilation

#YY
# from google3.perftools.accelerators.xprof.api.python import xprof_session
# from google3.pyglib import gfile

# ====== Data ======
TRAIN_DATA_PATH = "/workspace/tunix/rl/grpo/data/gsm8k_train.json" #YY
TEST_DATA_PATH = "/workspace/tunix/rl/grpo/data/gsm8k_test.json" #YY

USE_VLLM = True  # YY
VLLM_MODEL_VERSION="/workspace/tunix/rl/grpo/models/meta-llama/Meta-Llama-3-8B-Instruct/"
# VLLM_MODEL_VERSION="meta-llama/Llama-3.1-8B" #YY

if USE_VLLM: #YY
  os.environ["TPU_BACKEND_TYPE"] = "jax"
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


TRAIN_FRACTION = 1.0

# ====== Base Model ======
NNX_CKPT_DIR = "/workspace/tunix/rl/grpo/nnx/"
MODEL_VERSION = "2b-it"

if USE_VLLM: # YY
  NNX_CKPT_DIR="/workspace/tunix/rl/grpo/models/meta-llama/Meta-Llama-3-8B-Instruct/" # Need llama3 8B safetensor checkpoint
  MODEL_VERSION = "8b"

# ====== Reproducibility ======
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH = [(1, 8), ("fsdp", "tp")] #YY

# ====== GRPO ======
# === Generation during GRPO training ===
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 1024 #YY 768
# Important to keep a high-ish temperature for varied, diverse responses during
# training.
TEMPERATURE = 0.9
TOP_P = 1.0  # implies we don't do nucleus sampling
TOP_K = 50
# The number of times the policy generates multiple responses for a given prompt
# within a single training step. This corresponds to `G` in Algorithm 1 in the
# paper. The "group" in GRPO comes from here.
NUM_GENERATIONS = 4

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
BATCH_SIZE = 4
NUM_BATCHES = 2 #YY 1869 -> 2
# Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# increased to a max. of 330 (if batch size is 4).
NUM_TEST_BATCHES = 50 # YY 50 -> 1

EVAL_EVERY_N_STEPS = 10  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
NUM_EPOCHS = 1  # can potentially train for more epochs

# Number of training steps.
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# === AdamW, warmup, cosine scheduler ===
LEARNING_RATE = 3e-6
B1 = 0.9  # Adam beta1
B2 = 0.99  # Adam beta2
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

# ====== Checkpoint saving ======
CKPT_DIR = (
    "/workspace/tunix/rl/grpo/demo/experiments/gemma2/training_runs/2"
)
if USE_VLLM:
  CKPT_DIR = (
      "/workspace/tunix/rl/grpo/demo/experiments/llama3/training_runs/2"
  )

SAVE_INTERVAL_STEPS = 500 #YY 500 -> 5
MAX_TO_KEEP = 4
DO_MEM_PROFILING = False

# ====== Inference ======
GENERATION_CONFIGS = {
    # greedy search
    "greedy": {"temperature": 1e-2, "top_k": 1, "top_p": 1.0},
    # some randomness
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    # liberal
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

import gc
for name, obj in list(globals().items()):
    if isinstance(obj, jnp.ndarray):
        del globals()[name]
gc.collect()


def load_json_from_cns(path):
  # with gfile.Open(path, "rb") as f:
  with open(path, "rb") as f: #YY
    return json.loads(f.read())

@contextlib.contextmanager
def profile_and_capture_log(tag):
  return
  xprof = xprof_session.XprofSession()
  xprof.start_session(
      device_name='viperfish', enable_python_tracer=True, host_trace_level=2
  )
  log_handler = logging.StreamHandler()
  logging.root.addHandler(log_handler)
  try:
    yield
  finally:
    logging.root.removeHandler(log_handler)
    xprof_url = xprof.end_session_and_get_url(
        tag=tag,
        ttl_seconds=60 * 60 * 24 * 365,
    )
    print(xprof_url)

def show_hbm_usage(title=""):
  fmt_size = functools.partial(humanize.naturalsize, binary=True)

  for d in jax.local_devices():
    stats = d.memory_stats()
    used = stats['bytes_in_use']
    limit = stats['bytes_limit']
    print(f'{title} -- Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}')

show_hbm_usage()

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


def get_dataset(path: str) -> grain.MapDataset:

  data = load_json_from_cns(path)

  dataset = (
      grain.MapDataset.source(data)
      .shuffle(seed=SEED)
      .map(
          lambda x: {
              # passed to model forward pass
              "prompts": TEMPLATE.format(
                  system_prompt=SYSTEM_PROMPT, question=x["question"]
              ),
              # passed to reward functions
              "question": x["question"],
              # passed to reward functions
              "answer": extract_hash_answer(x["answer"]),
          }
      )
  )
  return dataset

dataset = get_dataset(TRAIN_DATA_PATH).batch(BATCH_SIZE)[:NUM_BATCHES]

if TRAIN_FRACTION == 1.0:
  train_dataset = dataset.repeat(NUM_EPOCHS)
  val_dataset = None
else:
  train_dataset = dataset[: int(len(dataset) * TRAIN_FRACTION)]
  train_dataset = train_dataset.repeat(NUM_EPOCHS)

  val_dataset = dataset[int(len(dataset) * TRAIN_FRACTION) :].repeat(NUM_EPOCHS)

test_dataset = get_dataset(TEST_DATA_PATH).batch(BATCH_SIZE)[:NUM_TEST_BATCHES]

len(train_dataset), len(val_dataset) if val_dataset is not None else 0, len(
    test_dataset
)

for ele in train_dataset[:1]:
  pprint(ele)

MODEL_CONFIG = {
    "2b": gemma_lib.TransformerConfig.gemma2_2b,
    "2b-it": gemma_lib.TransformerConfig.gemma2_2b,
    "8b": llama_lib.ModelConfig.llama3_8b,
}

#YY
def get_gemma_model(ckpt_path, mesh):
  abs_gemma: nnx.Module = nnx.eval_shape(
      lambda: gemma_lib.Transformer(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)

  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma

#YY
def get_llama_model(ckpt_path, mesh):
  return create_model_from_safe_tensors(ckpt_path, llama_model_config.llama3_8b(), mesh)

def get_ref_model():
  ckpt_path = os.path.join(NNX_CKPT_DIR, MODEL_VERSION)
  mesh = jax.make_mesh(*MESH)
  model_config = MODEL_CONFIG[MODEL_VERSION]()
  if MODEL_VERSION in ("2b", "2b-it"):
    model = get_gemma_model(ckpt_path, mesh)
  else:
    model = get_llama_model(NNX_CKPT_DIR, mesh)
  return model, mesh, model_config


# YY
def get_lora_model(base_model, mesh=None):
  if isinstance(base_model, llama_lib.Llama3):
    module_path = ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj"
  else:
    module_path = ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj|.*attn_vec_einsum"

  lora_provider = lora.LoraProvider(
      module_path=(
          module_path
      ),
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = lora.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  with mesh:
    state = nnx.state(lora_model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(lora_model, sharded_state)

  return lora_model

# Reference model
transformer, mesh, model_config = get_ref_model() #YY rename gemma to transformer
# nnx.display(transformer)

# Policy model
lora_transformer = get_lora_model(transformer, mesh=mesh) #YY rename lora_gemma to lora_transformer
# nnx.display(lora_transformer)

show_hbm_usage("After creating the reference lora model")

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

def match_format_exactly(prompts, completions, **kargs):
  scores = []
  for completion in completions:
    score = 0
    response = completion
    # Match if format is seen exactly!
    if match_format.search(response) is not None:
      score += 3.0
    scores.append(score)
  return scores

def match_format_approximately(prompts, completions, **kargs):
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

def check_answer(prompts, completions, answer, **kargs):
  responses = completions

  extracted_responses = [
      guess.group(1) if (guess := match_format.search(r)) is not None else None
      for r in responses
  ]

  scores = []
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

def check_numbers(prompts, completions, answer, **kargs):
  question = kargs["question"]
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
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=question,
        ),
    ]
  else:
    input_batch = [
        TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=q,
        )
        for q in question
    ]

  #YY
  if seed is not None:
    if not USE_VLLM:
      seed = jax.random.PRNGKey(seed)
    else:
      seed = None #YY vLLM doesn't support per reqeust seed yet. Set temperature to 0 for fixed outputs.

  out_data = sampler(
      input_strings=input_batch,
      total_generation_steps=768,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      echo=False,
      seed=seed,
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


#YY
if USE_VLLM:
  model_tokenizer = AutoTokenizer.from_pretrained(VLLM_MODEL_VERSION)
  # sampler = vllm_sampler.vLLMSampler(
  #   tokenizer=model_tokenizer,
  #   model=lora_transformer,
  #   model_version=VLLM_MODEL_VERSION,
  #   lora_config={
  #         "rank": 64,
  #         "alpha": 64.0,
  #         "module_path":
  #             ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
  #       }
  # )
else:
  model_tokenizer = data_lib.GemmaTokenizer() #YY rename gemma_tokenizer to model_tokenizer
  # sampler = sampler_lib.Sampler(
  #     transformer=lora_transformer,
  #     tokenizer=model_tokenizer,
  #     cache_config=sampler_lib.CacheConfig(
  #         cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256, # YY 1280 / page size:32=40, needs to be divisable to 16, that's is to say, the total cache size should be 512X
  #         num_layers=model_config.num_layers,
  #         num_kv_heads=model_config.num_kv_heads,
  #         head_dim=model_config.head_dim,
  #     ),
  # )
# show_hbm_usage("After creating a raw sampler")
# YY disable sampling
# (corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
#     test_dataset,
#     sampler,
#     **GENERATION_CONFIGS["greedy"],
# )
# print(
#     f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
#     f" {format_accuracy=}%"
# )

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")

# %load_ext google3.learning.brain.tensorboard.notebook.extension

# Ckpt saving
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
)

# Logs
# %tensorboard --logdir /tmp/tensorboard/grpo --port=0

# Training config
training_config = GrpoConfig(
    max_prompt_length=MAX_PROMPT_LENGTH,
    total_generation_steps=TOTAL_GENERATION_STEPS,
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    max_steps=MAX_STEPS,
    # metrics logging
    metrics_logging_options=metrics_logging_options,
    # checkpoint saving
    checkpoint_root_directory=CKPT_DIR,
    checkpointing_options=checkpointing_options,
)

# Rollout worker
# model_tokenizer = data_lib.GemmaTokenizer() #YY
if USE_VLLM:
  rollout_worker = vllm_rollout.vLLMRollout(
    model=lora_transformer,
    tokenizer=model_tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
    lora_config={
          "rank": 64,
          "alpha": 64.0,
          "module_path":
              ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*down_proj|.*up_proj",
        },
    mesh=mesh,
    model_version=VLLM_MODEL_VERSION, #YY
  )
else:
  rollout_worker = vanilla_rollout.VanillaRollout(
      model=lora_transformer,
      tokenizer=model_tokenizer,
      cache_config=sampler_lib.CacheConfig(
          cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
          num_layers=model_config.num_layers,
          num_kv_heads=model_config.num_kv_heads,
          head_dim=model_config.head_dim,
      ),
  )
show_hbm_usage("After creating a new rollout worker")
# Optimizer, learning rate scheduler, gradient clipping
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

# GRPO Trainer
grpo_trainer = GrpoLearner(
    model=lora_transformer,
    ref_model=transformer,  # use the base model as reference
    reward_fns=[
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ],
    rollout_worker=rollout_worker,
    optimizer=optimizer,
    training_config=training_config,
    trainer_mesh=mesh,
    rollout_worker_mesh=mesh,
)

show_hbm_usage("Right before training")
with mesh:
  if DO_MEM_PROFILING:
    with profile_and_capture_log("gemma_benchmark"):
      grpo_trainer.train(train_dataset)
  else:
    grpo_trainer.train(train_dataset, eval_ds=val_dataset)

# Load checkpoint first.

show_hbm_usage("After training the reference lora model")

trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")

abs_params = jax.tree.map(
    lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
    nnx.state(lora_transformer, nnx.LoRAParam),
)
checkpointer = ocp.StandardCheckpointer()
trained_lora_params = checkpointer.restore(trained_ckpt_path, target=abs_params)

nnx.update(
    lora_transformer,
    jax.tree.map(
        lambda a, b: b,
        nnx.state(lora_transformer, nnx.LoRAParam),
        trained_lora_params,
    ),
)

# model_tokenizer = data_lib.GemmaTokenizer() #YY
# sampler = sampler_lib.Sampler(
#     transformer=lora_transformer,
#     tokenizer=model_tokenizer,
#     cache_config=sampler_lib.CacheConfig(
#         cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
#         num_layers=model_config.num_layers,
#         num_kv_heads=model_config.num_kv_heads,
#         head_dim=model_config.head_dim,
#     ),
# )
sampler = rollout_worker._sampler

(corr, total, accuracy, partial_accuracy, format_accuracy) = evaluate(
    test_dataset,
    sampler,
    **GENERATION_CONFIGS["greedy"],
)
print(
    f"{corr=}, {total=}, {accuracy=}%, {partial_accuracy=}%,"
    f" {format_accuracy=}%"
)

# for eval_example in QUALITATIVE_EVAL_EXAMPLES:
#   question = eval_example["question"]
#   answer = eval_example["answer"]
#   response = generate(
#       question,
#       sampler,
#       temperature=INFERENCE_TEMPERATURE,
#       top_k=INFERENCE_TOP_K,
#       top_p=INFERENCE_TOP_P,
#   )

#   print(f"Question:\n{question}")
#   print(f"Answer:\n{answer}")
#   print(f"Response:\n{response}")
#   print("===============")


