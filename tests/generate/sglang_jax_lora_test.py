"""
Note: This test is based on scripts/grpo_demo_sglang_jax_rollout.py.
For the meanings of constants, please refer to the above file.
"""

import csv
import os
from pathlib import Path
import re
import shutil
from absl.testing import absltest

import grain
import huggingface_hub
import jax
import kagglehub
import optax
from orbax import checkpoint as ocp
import qwix
import transformers
from tunix.generate import mappings
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama3_params_lib
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig
from tunix.rl.grpo.grpo_learner import GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.rl.utils import VERIFY_UPDATE_PARAMS_KEY
from tunix.sft import metrics_logger


############################################# CONSTANTS ###########################################

TRAIN_DATA_DIR = "./data/train"
TRAIN_FRACTION = 1.0
RANK = 64
ALPHA = 64.0
TOTAL_TPU_TO_USE = jax.device_count()
MESH = [
    (
        1,
        TOTAL_TPU_TO_USE,
    ),
    ("fsdp", "tp"),
]
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 1024
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2
NUM_ITERATIONS = 1
BETA = 0.08
EPSILON = 0.2
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 2
NUM_TEST_BATCHES = 2
EVAL_EVERY_N_STEPS = 5
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1
INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
CKPT_DIR = "/tmp/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

############################################# PROMPTS ###########################################

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


def load_model():
  model_config = llama_lib.ModelConfig.llama3p2_3b()

  mesh = jax.make_mesh(
      *MESH,
      devices=jax.devices()[:TOTAL_TPU_TO_USE],
      axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]),
  )
  model = llama3_params_lib.create_model_from_safe_tensors(
      model_path, model_config, mesh
  )
  return model, mesh, model_config


def get_rollout_mesh():
  mesh = jax.make_mesh(
      *MESH,
      devices=jax.devices()[-TOTAL_TPU_TO_USE:],
      axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]),
  )
  return mesh


def get_lora_model(base_model):
  lora_provider = qwix.LoraProvider(
      module_path=".*gate_proj",
      rank=RANK,
      alpha=ALPHA,
  )

  model_input = base_model.get_model_input()
  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, **model_input
  )

  return lora_model


model_version = "meta-llama/Llama-3.2-3B-Instruct"

repo_id = model_version
model_tokenizer = transformers.AutoTokenizer.from_pretrained(repo_id)

import tempfile

temp_dir = tempfile.gettempdir()
model_path = os.path.join(temp_dir, "models", repo_id)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(prompts, completions, **kwargs):
  return [
      0 if match_format.search(response) is None else 3.0
      for response in completions
  ]


class SglangJaxLoRATest(absltest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    VERIFY_UPDATE_PARAMS_VAL = "layers.0.mlp.gate_proj.kernel_lora_a,model.layers.0.mlp.gate_proj.A_buffer"
    os.environ[VERIFY_UPDATE_PARAMS_KEY] = VERIFY_UPDATE_PARAMS_VAL

    super().setUpClass()

    ## ====================================== Get Dataset ===================================
    source = "kaggle"
    cls.dataset = get_dataset(TRAIN_DATA_DIR, "train", source).batch(
        TRAIN_MICRO_BATCH_SIZE
    )[:NUM_BATCHES]

    ## ====================================== Get LoRA model =================================
    download_from_huggingface(repo_id=repo_id, model_path=model_path)

    ref_model, cls.mesh, _ = load_model()
    rollout_mesh = get_rollout_mesh()

    lora_policy = get_lora_model(ref_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    mapping_config = mappings.MappingConfig.build(
        model=ref_model, backend="sglang_jax"
    )

    ## ========================== Iniitialize RL cluster and trainer ==========================
    cluster_config, grpo_config = cls.prepare_configs(
        cls.mesh, rollout_mesh, mapping_config
    )

    # RL cluster
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora_policy,
        reference=ref_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    # GRPO Trainer
    cls.grpo_trainer = GRPOLearner(
        rl_cluster=rl_cluster,
        reward_fns=[
            match_format_exactly,
        ],
        algo_config=grpo_config,
    )

  @classmethod
  def prepare_configs(cls, mesh, rollout_mesh, mapping_config):
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
            rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
        },
        rollout_engine="sglang_jax",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=MAX_STEPS,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=metrics_logging_options,
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
            rollout_mapping_config=mapping_config,
            rollout_sglang_jax_model_version=model_path,
            rollout_sglang_jax_context_length=2048,
            rollout_sglang_jax_mem_fraction_static=0.3,
            rollout_sglang_jax_init_with_random_weights=True,
            rollout_sglang_jax_disable_radix_cache=True,
            rollout_sglang_jax_enable_deterministic_sampling=False,
            rollout_sglang_jax_use_sort_for_toppk_minp=True,
            rollout_sglang_jax_enable_static_lora=True,
            rollout_sglang_jax_enable_single_process=True,
            rollout_sglang_jax_lora_target_modules=["all"],
            rollout_sglang_jax_max_lora_rank=RANK,
            rollout_sglang_jax_lora_scaling=ALPHA / RANK,
            rollout_sglang_jax_precompile_bs_paddings=[8],
            rollout_sglang_jax_precompile_token_paddings=[2048],
        ),
    )

    grpo_config = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
    )

    return (cluster_config, grpo_config)

  def test_lora(self):
    with self.mesh:
      self.grpo_trainer.train(self.dataset)


if __name__ == "__main__":
  absltest.main()
