import argparse
import logging
import os
import sys
import time

from absl import logging as absl_logging
import jax
from jax import numpy as jnp
import numpy as np
import optax
from transformers import AutoTokenizer
from tunix.models.qwen2 import model as model_lib
from tunix.models.qwen2 import params as params_lib
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig
from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.rollout import base_rollout
from tunix.sft import utils as sft_utils

absl_logging.use_python_logging()

# 2. Configure the root logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)

# 4. Set absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

print("jax devices: ", jax.devices())

MESH = [(4, 1), ("fsdp", "tp")]
mesh = jax.make_mesh(
    *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
)


def main():
  arg_parser = argparse.ArgumentParser(description="Benchmark sequence packing")
  arg_parser.add_argument(
      "--model",
      type=str,
      default="qwen3_30b",
      choices=["qwen3_30b", "qwen3_32b", "deepseek_distill"],
      help="Model to use for benchmarking",
  )
  arg_parser.add_argument(
      "--use_flash_attention",
      action="store_true",
      default=False,
      help="Use flash attention during benchmarking",
  )
  arg_parser.add_argument(
      "--enable_profiling",
      action="store_true",
      default=False,
      help="Enable JAX profiler trace",
  )
  arg_parser.add_argument(
      "--batch_size",
      type=int,
      default=16,
      help="Batch size for training",
  )
  arg_parser.add_argument(
      "--mini_batch_size",
      type=int,
      default=16,
      help="Mini batch size for training",
  )
  arg_parser.add_argument(
      "--micro_batch_size",
      type=int,
      default=8,
      help="Micro batch size for training",
  )
  arg_parser.add_argument(
      "--iterations",
      type=int,
      default=20,
      help="Number of iterations for benchmark",
  )
  arg_parser.add_argument(
      "--mock_generation_mode",
      type=str,
      default="high_variance",
      choices=["random", "high_variance"],
      help="Mode of mock generation for sequence lengths",
  )
  args, _ = arg_parser.parse_known_args()

  logging.info(f"Running benchmark with args: {args}")

  # TODO: update this to your local file path
  MODEL_PATH_PREFIX = "/tmp/models/"
  DEESCALER_MODEL_PATH = os.path.join(
      MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B"
  )
  DEEPSWE_MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "Qwen3-4B-Instruct-2507")

  QWEN3_32B_MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "Qwen3-32B")
  QWEN3_30B_MODEL_PATH = os.path.join(MODEL_PATH_PREFIX, "Qwen3-30B-A3B")

  use_flash_attention = args.use_flash_attention

  global tokenizer, model

  class DummyTokenizer:

    def encode(self, x):
      return [1]

    def decode(self, x):
      return "dummy"

    def bos_id(self):
      return 1

    def eos_id(self):
      return 2

    def pad_id(self):
      return 0

  if args.model == "qwen3_30b":
    try:
      tokenizer = AutoTokenizer.from_pretrained(QWEN3_30B_MODEL_PATH)
    except:
      tokenizer = DummyTokenizer()
    config = qwen3_model_lib.ModelConfig.qwen3_30b_a3b()
    config.remat_config = qwen3_model_lib.RematConfig.BLOCK
    config.dtype = jnp.bfloat16
    config.use_flash_attention = use_flash_attention
    try:
      model = qwen3_params_lib.create_model_from_safe_tensors(
          QWEN3_30B_MODEL_PATH, config, mesh, dtype=jnp.bfloat16
      )
    except Exception as e:
      print(
          "Failed to load model from path, initializing dummy weights for"
          f" benchmark: {e}"
      )
      from flax import nnx

      rngs = nnx.Rngs(0)
      with mesh:
        model = qwen3_model_lib.Qwen3(config=config, rngs=rngs)
  elif args.model == "qwen3_32b":
    try:
      tokenizer = AutoTokenizer.from_pretrained(QWEN3_32B_MODEL_PATH)
    except:
      tokenizer = DummyTokenizer()
    config = qwen3_model_lib.ModelConfig.qwen3_32b()
    config.remat_config = qwen3_model_lib.RematConfig.BLOCK
    config.dtype = jnp.bfloat16
    config.use_flash_attention = use_flash_attention
    try:
      model = qwen3_params_lib.create_model_from_safe_tensors(
          QWEN3_32B_MODEL_PATH, config, mesh, dtype=jnp.bfloat16
      )
    except Exception as e:
      print(
          "Failed to load model from path, initializing dummy weights for"
          f" benchmark: {e}"
      )
      from flax import nnx

      rngs = nnx.Rngs(0)
      with mesh:
        model = qwen3_model_lib.Qwen3(config=config, rngs=rngs)
  elif args.model == "deepseek_distill":
    try:
      tokenizer = AutoTokenizer.from_pretrained(DEESCALER_MODEL_PATH)
    except:
      tokenizer = DummyTokenizer()
    config = model_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b()
    config.remat_config = model_lib.RematConfig.BLOCK
    config.dtype = jnp.bfloat16
    config.use_flash_attention = use_flash_attention
    try:
      model = params_lib.create_model_from_safe_tensors(
          DEESCALER_MODEL_PATH, config, mesh, dtype=jnp.float32
      )
    except Exception as e:
      print(
          "Failed to load model from path, initializing dummy weights for"
          f" benchmark: {e}"
      )
      from flax import nnx

      rngs = nnx.Rngs(0)
      with mesh:
        model = model_lib.Qwen2(config=config, rngs=rngs)
  else:
    raise ValueError("No model selected for benchmark")
  sft_utils.show_hbm_usage()

  # ====== Data ======
  global MAX_PROMPT_LENGTH, MAX_RESPONSE_LENGTH, TEMPERATURE, TOP_P, TOP_K
  global NUM_GENERATIONS, NUM_ITERATIONS, BETA, EPSILON, ENABLE_REMAT
  global BATCH_SIZE, MINI_BATCH_SIZE, MICRO_BATCH_SIZE, NUM_BATCHES, NUM_TEST_BATCHES
  global EVAL_EVERY_N_STEPS, NUM_EPOCHS, MAX_STEPS, LEARNING_RATE, B1, B2, WEIGHT_DECAY
  global WARMUP_STEPS, MAX_GRAD_NORM, TRAIN_FRACTION, SEED

  TRAIN_FRACTION = 1.0
  SEED = 42
  MAX_PROMPT_LENGTH = 1024
  MAX_RESPONSE_LENGTH = 1024
  TEMPERATURE = 0.6
  TOP_P = 0.95
  TOP_K = 50
  NUM_GENERATIONS = 8
  NUM_ITERATIONS = 1
  BETA = 0.001
  EPSILON = 0.2
  ENABLE_REMAT = True
  BATCH_SIZE = args.batch_size
  MINI_BATCH_SIZE = args.mini_batch_size
  MICRO_BATCH_SIZE = args.micro_batch_size
  NUM_BATCHES = 100
  NUM_TEST_BATCHES = 50
  EVAL_EVERY_N_STEPS = 1000
  NUM_EPOCHS = 100
  MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
  LEARNING_RATE = 1e-6
  B1 = 0.9
  B2 = 0.99
  WEIGHT_DECAY = 0.1
  WARMUP_STEPS = int(0.1 * MAX_STEPS)
  MAX_GRAD_NORM = 0.1

  model_id = args.model

  duration_without_packing = None
  try:
    duration_without_packing = run_benchmark(
        use_sequence_packing=False,
        model_id=model_id,
        enable_profiling=args.enable_profiling,
        max_steps=args.iterations,
        mock_generation_mode=args.mock_generation_mode,
    )
  except Exception as e:
    print(f"Benchmark without packing failed: {e}")

  duration_with_packing = None
  try:
    duration_with_packing = run_benchmark(
        use_sequence_packing=True,
        model_id=model_id,
        enable_profiling=args.enable_profiling,
        max_steps=args.iterations,
        mock_generation_mode=args.mock_generation_mode,
    )
  except Exception as e:
    print(f"Benchmark with packing failed: {e}")

  print(f"\n{'='*50}")
  print("Benchmark Results:")

  if duration_without_packing is not None:
    print(f"Without packing: {duration_without_packing:.2f}s")
  else:
    print("Without packing: Failed")

  if duration_with_packing is not None:
    print(f"With packing: {duration_with_packing:.2f}s")
  else:
    print("With packing: Failed")

  if (
      duration_without_packing is not None
      and duration_with_packing is not None
      and duration_with_packing > 0
  ):
    speedup = duration_without_packing / duration_with_packing
    print(f"Speedup: {speedup:.2f}x")
  else:
    print("Speedup: N/A")
  print(f"{'='*50}")


def run_benchmark(
    use_sequence_packing: bool,
    model_id: str,
    enable_profiling: bool = False,
    max_steps: int = 20,
    max_token_len_per_tpu: int = 4096,
    mock_generation_mode: str = "high_variance",
):
  print(f"\n{'='*50}")
  print(
      f"Starting Benchmark: use_sequence_packing = {use_sequence_packing},"
      f" mock_generation_mode = {mock_generation_mode}"
  )
  print(f"{'='*50}")

  base_rollout_dict = {
      "max_prompt_length": MAX_PROMPT_LENGTH,
      "kv_cache_size": MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 256,
      "temperature": TEMPERATURE,
      "top_p": TOP_P,
      "top_k": TOP_K,
      "eos_tokens": (
          [tokenizer.encode("<|im_end|>")[0]]
          if hasattr(tokenizer, "encode")
          else [2]
      ),
      "max_tokens_to_generate": MAX_RESPONSE_LENGTH,
      "return_logprobs": True,
  }
  rollout_engine_config = base_rollout.RolloutConfig(**base_rollout_dict)

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

  cluster_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: mesh,
          rl_cluster_lib.Role.REFERENCE: mesh,
          rl_cluster_lib.Role.ROLLOUT: mesh,
      },
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_cluster_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          eval_every_n_steps=EVAL_EVERY_N_STEPS,
          max_steps=max_steps,
          mini_batch_size=MINI_BATCH_SIZE,
          train_micro_batch_size=MICRO_BATCH_SIZE,
      ),
      rollout_config=rollout_engine_config,
  )
  grpo_config = GRPOConfig(
      num_generations=NUM_GENERATIONS,
      num_iterations=NUM_ITERATIONS,
      max_response_length=MAX_RESPONSE_LENGTH,
      beta=BETA,
      epsilon=EPSILON,
      system_prompt="",
      max_concurrency=32,
      use_sequence_packing=use_sequence_packing,
      max_token_len_per_tpu=max_token_len_per_tpu,
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=model,
      reference=model,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  def mock_generate(prompts, **kwargs):
    num_prompts = len(prompts)
    if mock_generation_mode == "high_variance":
      # High variance sequences to stress test sequence packing
      # 80% short sequences, 20% long sequences
      is_long = np.random.rand(num_prompts) < 0.2
      lengths = np.where(
          is_long,
          np.random.randint(MAX_RESPONSE_LENGTH // 2, int(MAX_RESPONSE_LENGTH * 0.9), size=num_prompts),
          np.random.randint(16, 128, size=num_prompts)
      )
    else:
      # Variable length sequences to show the benefit of sequence packing
      # lengths = np.random.randint(128, MAX_RESPONSE_LENGTH // 2, size=num_prompts)
      lengths = np.random.randint(
          MAX_RESPONSE_LENGTH // 2, MAX_RESPONSE_LENGTH, size=num_prompts
      )

    texts = ["hello world " * (l // 2) for l in lengths]
    tokens = [np.ones((l,), dtype=int) for l in lengths]
    logits = [np.ones((l, 2), dtype=float) for l in lengths]
    logprobs = [np.zeros((l,), dtype=float) for l in lengths]

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits,
        tokens=tokens,
        left_padded_prompt_tokens=np.array([[0, 1]] * num_prompts),
        logprobs=logprobs,  # Ensure logprobs are returned
    )

  rl_cluster.generate = mock_generate

  try:
    chat_parser = parser.QwenChatTemplateParser(tokenizer)
  except:
    chat_parser = parser.DefaultChatTemplateParser(tokenizer)

  grpo_trainer = GRPOLearner(
      rl_cluster=rl_cluster,
      reward_fns=[
          lambda **kargs: [1.0] * len(kargs.get("prompts", [1])),
      ],
      algo_config=grpo_config,
      chat_parser=chat_parser,
  )

  def create_dummy_dataset(itr=max_steps):
    dummy_inputs = {"prompts": ["my initial prompt"] * MINI_BATCH_SIZE}
    for _ in range(itr):
      yield dummy_inputs

  sft_utils.show_hbm_usage()
  input_data = create_dummy_dataset(itr=max_steps)

  start = time.time()

  # TODO: update this to your gs bucket if needed
  packing_status = "with_packing" if use_sequence_packing else "without_packing"
  log_dir = f"gs://noghabi-dev/sequence_packing/{model_id}/{packing_status}"

  if enable_profiling:
    try:
      with jax.profiler.trace(log_dir=log_dir):
        grpo_trainer.train(input_data)
    except Exception as e:
      print(f"Profiler trace failed, running without profiler. Error: {e}")
      grpo_trainer.train(input_data)
  else:
    grpo_trainer.train(input_data)

  end = time.time()
  duration = end - start
  print(f"Training took {duration:.2f} seconds")
  sft_utils.show_hbm_usage()

  return duration


if __name__ == "__main__":
  main()
