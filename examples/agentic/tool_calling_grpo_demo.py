# %%
"""Tool Calling GRPO Training Demo.

This tutorial demonstrates training an LLM to use tools during mathematical
reasoning using the Agentic GRPO algorithm. The model learns to call a
calculator tool instead of computing arithmetic mentally, which can improve
accuracy on math problems.

Key concepts demonstrated:
1. Defining tools (ExpressionCalculatorTool)
2. Creating reward functions for tool usage
3. Configuring AgenticGRPOLearner with tool support
4. Multi-turn training where model calls tools and receives results

We use the GSM8K dataset and train the model to:
- Put reasoning inside <reasoning>...</reasoning> tags
- Call calculator via <tool_call>...</tool_call> for arithmetic
- Put final answer in <answer>...</answer> tags
"""

# %%
# Imports
import contextlib
import os
import re
import json
import time
from pprint import pprint

# %%
import jax
from jax import numpy as jnp
import optax
from orbax import checkpoint as ocp

# %%
# Tunix imports
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.sft import metrics_logger
from tunix.sft import utils
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.cli.utils import model as model_utils
from tunix.rl.agentic.parser.chat_template_parser import parser
from flax import nnx

# Tool calling imports
from tunix.rl.agentic import AgenticGRPOConfig, AgenticGRPOLearner
from tunix.rl.agentic.tools import ExpressionCalculatorTool

# %%
show_hbm_usage = utils.show_hbm_usage
show_hbm_usage()

# %%
# ------------------------------------------------------------------------------
# Section 1: Configuration
# ------------------------------------------------------------------------------

# ====== Data ======
TRAIN_DATA_PATH = "./data/train"
TEST_DATA_PATH = "./data/test"

# ====== Base Model ======
MODEL_DOWNLOAD_PATH = "/tmp/content/model_download/"
NNX_CKPT_DIR = "/tmp/content/intermediate_ckpt/"

# ====== Checkpoint saving ======
run_name = f"tool_calling_grpo_{int(time.time())}"
CKPT_DIR = f"/tmp/content/ckpts/{run_name}"

# %%
# --- Training Configs ---
TRAIN_FRACTION = 1.0
MODEL_VERSION = "2b-it"
SEED = 42

# ====== LoRA ======
RANK = 64
ALPHA = 64.0

# ====== Sharding ======
MESH = [(1, 1), ("fsdp", "tp")]  # Adjust based on available devices

# %%
# ====== GRPO ======
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 4

# GRPO-specific
NUM_ITERATIONS = 1
BETA = 0.04
EPSILON = 0.2

# ====== Tool Calling ======
MAX_TOOL_STEPS = 5  # Max tool interaction turns
TOOL_PARSER_NAME = "qwen"  # Parser for <tool_call> tags

# %%
# ====== Training ======
BATCH_SIZE = 4
NUM_BATCHES = 100
NUM_TEST_BATCHES = 5
EVAL_EVERY_N_STEPS = 1000
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# %%
# === Optimizer ===
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1

# %%
# === Checkpoint ===
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

# %%
# ------------------------------------------------------------------------------
# Section 2: Tool Definition
# ------------------------------------------------------------------------------

# Define the tools available to the model
TOOL_MAP = {"calculator": ExpressionCalculatorTool}

# %%
# ------------------------------------------------------------------------------
# Section 3: System Prompt & Templates
# ------------------------------------------------------------------------------

reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

# System prompt that teaches the model to use tools
SYSTEM_PROMPT = """You MUST use the calculator tool for ALL arithmetic. Never calculate in your head!

REQUIRED FORMAT:
1. Start with <reasoning>
2. Think through the problem step by step
3. When you need to calculate, write:
   <tool_call>{"name": "calculator", "arguments": {"expression": "YOUR_MATH"}}</tool_call>
   <end_of_turn>
4. After seeing the tool response, continue your reasoning
5. Close with </reasoning>
6. Put ONLY the final numeric answer in <answer>NUMBER</answer>

EXAMPLE (follow this EXACTLY):
<reasoning>
I need to find 77 + 33.
<tool_call>{"name": "calculator", "arguments": {"expression": "77 + 33"}}</tool_call>
<end_of_turn>
<tool_response>calculator: 110</tool_response>
The calculator says the answer is 110.
</reasoning>
<answer>110</answer>

CRITICAL RULES:
- ALWAYS use the calculator - NEVER write things like "5 + 3 = 8" yourself
- ALWAYS generate <end_of_turn> after </tool_call> to receive the tool response
- ALL tool calls must be INSIDE <reasoning> tags
- The <answer> tag must come AFTER </reasoning> and contain ONLY a number

NOW SOLVE:"""

# %%
# ------------------------------------------------------------------------------
# Section 4: Reward Functions
# ------------------------------------------------------------------------------

# Regex for format matching
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(prompts, completions, **kargs):
    """Reward for correct format: reasoning tags followed by answer tags."""
    scores = []
    for completion in completions:
        score = 0.0
        if match_format.search(completion) is not None:
            score += 3.0
        scores.append(score)
    return scores


# Pattern to detect manual arithmetic
MANUAL_CALC_PATTERN = re.compile(
    r"\b\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+", re.IGNORECASE
)


def check_tool_usage(prompts, completions, **kwargs):
    """Reward for proper tool usage.

    Rewards:
    - Tool call with valid JSON format (+2.0)
    - Proper <end_of_turn> after </tool_call> (+1.0)
    - <tool_response> present (+1.0)
    - Tool call inside <reasoning> tags (+0.5)

    Penalties:
    - Manual arithmetic without tool use (-3.0)
    - Tool calls inside <answer> tags (-2.0)
    - Malformed JSON (-1.0)
    """
    scores = []
    for completion in completions:
        score = 0.0
        comp_lower = completion.lower()

        # PENALTY: Tool calls inside <answer> tags
        answer_section = re.search(
            r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE
        )
        if answer_section and "<tool_call>" in answer_section.group(1).lower():
            scores.append(-2.0)
            continue

        # Find ALL tool calls
        tool_calls = re.findall(
            r"<tool_call>(.*?)</tool_call>", completion, re.DOTALL
        )

        has_manual_calc = bool(MANUAL_CALC_PATTERN.search(completion))

        if not tool_calls:
            if has_manual_calc:
                scores.append(-3.0)  # Strong penalty for manual arithmetic
            else:
                scores.append(0.0)
            continue

        # Evaluate tool call quality
        valid_calls = 0
        malformed_calls = 0
        for tc in tool_calls:
            try:
                tool_json = json.loads(tc.strip())
                args = tool_json.get("arguments", {})
                if "expression" in args and isinstance(args.get("expression"), str):
                    expr = args["expression"].strip()
                    if expr:
                        valid_calls += 1
            except (json.JSONDecodeError, TypeError):
                malformed_calls += 1

        if malformed_calls > 0:
            score -= 1.0

        if valid_calls > 0:
            score += 2.0

        # Proper turn-taking
        if re.search(r"</tool_call>\s*\n?\s*<end_of_turn>", completion, re.IGNORECASE):
            score += 1.0

        # Tool response present
        if "<tool_response>" in comp_lower:
            score += 1.0

        # Tool call inside reasoning
        reasoning_start_pos = comp_lower.find("<reasoning>")
        reasoning_end_pos = comp_lower.rfind("</reasoning>")
        first_tool = comp_lower.find("<tool_call>")
        if (
            reasoning_start_pos != -1
            and reasoning_end_pos != -1
            and first_tool > reasoning_start_pos
            and first_tool < reasoning_end_pos
        ):
            score += 0.5

        if has_manual_calc:
            score -= 1.0  # Penalty for redundant manual calc

        scores.append(max(0.0, min(5.0, score)))

    return scores


# Number extraction pattern
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})", flags=re.MULTILINE | re.DOTALL
)


def check_answer(prompts, completions, answer, **kargs):
    """Reward for correct answers."""
    responses = completions

    extracted_responses = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in responses
    ]

    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(0)
            continue
        if guess == true_answer:
            score += 3.0
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            try:
                ratio = float(guess) / float(true_answer)
                if 0.9 <= ratio <= 1.1:
                    score += 0.5
                elif 0.8 <= ratio <= 1.2:
                    score += 0.25
                else:
                    score -= 1.0
            except (ValueError, ZeroDivisionError):
                score -= 0.5
        scores.append(score)
    return scores


# %%
# ------------------------------------------------------------------------------
# Section 5: Data Loading
# ------------------------------------------------------------------------------

from tunix.utils import script_utils


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# Load datasets
train_dataset, val_dataset = script_utils.get_train_and_eval_datasets(
    data_path=TRAIN_DATA_PATH,
    split="train",
    seed=SEED,
    system_prompt=SYSTEM_PROMPT,
    batch_size=BATCH_SIZE,
    num_batches=NUM_BATCHES,
    train_fraction=TRAIN_FRACTION,
    num_epochs=NUM_EPOCHS,
    answer_extractor=extract_hash_answer,
)

test_dataset = script_utils.get_dataset(
    TEST_DATA_PATH,
    split="test",
    seed=SEED,
    system_prompt=SYSTEM_PROMPT,
    answer_extractor=extract_hash_answer,
).batch(BATCH_SIZE)[:NUM_TEST_BATCHES]

# %%
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset)}")

# %%
# Sample batch
for ele in train_dataset[:1]:
    pprint(ele)

# %%
# ------------------------------------------------------------------------------
# Section 6: Model Loading
# ------------------------------------------------------------------------------

MODEL_CONFIG = {
    "2b": gemma_lib.ModelConfig.gemma2_2b,
    "2b-it": gemma_lib.ModelConfig.gemma2_2b,
}


def get_ref_model():
    """Load the reference model."""
    mesh = jax.make_mesh(
        *MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0])
    )

    model_name = f"gemma2-{MODEL_VERSION}"
    model_config_dict = {
        "model_name": model_name,
        "model_source": "kaggle",
        "model_id": f"google/gemma-2/flax/{model_name}",
        "model_download_path": MODEL_DOWNLOAD_PATH,
        "intermediate_ckpt_dir": NNX_CKPT_DIR,
        "model_display": False,
    }
    tokenizer_config = {"tokenizer_path": None}
    gemma, tokenizer_path = model_utils.create_model(
        model_config_dict, tokenizer_config, mesh
    )
    return gemma, mesh, tokenizer_path


# %%
# Load reference model
gemma, mesh, tokenizer_path = get_ref_model()
nnx.display(gemma)

# %%
# Apply LoRA
lora_config = {
    "module_path": ".*attention",
    "rank": RANK,
    "alpha": ALPHA,
}
lora_gemma = model_utils.apply_lora_to_model(gemma, mesh=mesh, lora_config=lora_config)
nnx.display(lora_gemma)

# %%
show_hbm_usage()

# %%
# ------------------------------------------------------------------------------
# Section 7: Training Setup
# ------------------------------------------------------------------------------

# Checkpoint options
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
)

# Metrics logger
metrics_logging_options = metrics_logger.MetricsLoggerOptions(
    log_dir="/tmp/tensorboard/tool_calling_grpo", flush_every_n_steps=20
)

# %%
# Optimizer
optimizer = optax.adamw(
    learning_rate=optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=int(WARMUP_STEPS),
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

# %%
# Cluster config
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
        max_steps=MAX_STEPS,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
        train_micro_batch_size=1,
        mini_batch_size=4,
    ),
    rollout_config=base_rollout.RolloutConfig(
        max_tokens_to_generate=TOTAL_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
    ),
)

# %%
# Agentic GRPO config with tool calling
grpo_config = AgenticGRPOConfig(
    num_generations=NUM_GENERATIONS,
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    system_prompt=SYSTEM_PROMPT,
    max_concurrency=8,
    max_tool_steps=MAX_TOOL_STEPS,
    tool_parser_name=TOOL_PARSER_NAME,
)

# %%
# Tokenizer and chat parser
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=tokenizer_path)
chat_parser = parser.GemmaChatTemplateParser(tokenizer)

# %%
# RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    actor=lora_gemma,
    reference=gemma,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# %%
# Agentic GRPO Trainer with tool support
grpo_trainer = AgenticGRPOLearner(
    rl_cluster=rl_cluster,
    reward_fns=[
        match_format_exactly,
        check_tool_usage,  # New: rewards proper tool usage
        check_answer,
    ],
    algo_config=grpo_config,
    chat_parser=chat_parser,
    tool_map=TOOL_MAP,  # Enable tool calling
)

# %%
# ------------------------------------------------------------------------------
# Section 8: Execute Training
# ------------------------------------------------------------------------------
print("Starting tool-calling GRPO training...")
print(f"Tools available: {list(TOOL_MAP.keys())}")
print(f"Max tool steps: {MAX_TOOL_STEPS}")

grpo_trainer.train(train_dataset, eval_dataset=val_dataset)

print("Training complete!")

