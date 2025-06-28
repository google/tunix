# ppo_demo.py
# Standalone script for training Qwen3-0.6B on an instruction tuning
# dataset with PPO. Uses a reward model to compare the policy output
# against the reference answer from the dataset.

# ---------- Install dependencies ----------
# (run these in your shell before running the script)
# pip install tensorflow tensorboardX grain git+https://github.com/google/tunix git+https://github.com/google/qwix git+https://github.com/google/flax.git datasets transformers

# ---------- Imports ----------
import functools
import gc
import os
from pprint import pprint
import re
import time

from flax import nnx
import grain
import humanize
import jax
import jax.numpy as jnp
import optax
from orbax import checkpoint as ocp
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from tunix.generate import sampler as sampler_lib
from tunix.models.qwen3 import params as qwen3_params

from tunix.rl.ppo.ppo_trainer import PpoTrainer, PpoTrainingConfig, TrainExample
from tunix.sft import metrics_logger
import torch
from transformers import AutoModelForCausalLM

REPO_ID = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
hf_cfg = AutoConfig.from_pretrained(REPO_ID)

# Torch reward model
RM_NAME = "NiuTrans/GRAM-Qwen3-1.7B-RewardModel"
rm_tokenizer = AutoTokenizer.from_pretrained(RM_NAME, padding_side="left")
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token
rm_model = AutoModelForCausalLM.from_pretrained(
    RM_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# ---------- Hyperparameters ----------
TRAIN_FRACTION = 1.0


MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9
TOP_P = 1.0
TOP_K = 50
NUM_PPO_EPOCHS = 1
NUM_MINI_BATCHES = 1
CLIP_RANGE = 0.2
CLIP_RANGE_VALUE = 0.2

BATCH_SIZE = 1
NUM_BATCHES = 3738
NUM_TEST_BATCHES = 100
EVAL_EVERY_N_STEPS = 10
NUM_EPOCHS = 1

MAX_STEPS = int(NUM_BATCHES * TRAIN_FRACTION * NUM_EPOCHS)

LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = int(0.1 * MAX_STEPS)
MAX_GRAD_NORM = 0.1
CKPT_DIR = "/content/ckpts/"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4

GENERATION_CONFIGS = {
    "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
    "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ---------- Utility Functions ----------
def show_hbm_usage():
    """Displays memory usage per device."""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"]
        limit = stats["bytes_limit"]
        print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")



def templatize(prompts, tokenizer):
    out = []
    for p in prompts:
        out.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )
    return out

def get_dataset(split="train") -> grain.MapDataset:
    ds = load_dataset("yahma/alpaca-cleaned", split=split)
    ds = ds.shuffle(seed=42)

    def _map_fn(x):
        prompt = x["instruction"]
        if x["input"]:
            prompt += "\n" + x["input"]
        return {
            "prompts": templatize([prompt], tokenizer)[0],
            "reference": x["output"],
        }

    dataset = grain.MapDataset.source(ds).map(_map_fn)
    return dataset

train_dataset = get_dataset("train[:80%]").batch(BATCH_SIZE)[:NUM_BATCHES].repeat(NUM_EPOCHS)
val_dataset = None
test_dataset = get_dataset("train[80%:]").batch(BATCH_SIZE)[:NUM_TEST_BATCHES]

print(
    "Train/Val/Test sizes:",
    len(train_dataset),
    len(val_dataset) if val_dataset is not None else 0,
    len(test_dataset),
)

for ele in train_dataset[:1]:
    pprint(ele)

def get_model() -> nnx.Module:
    return qwen3_params.from_pretrained(REPO_ID)


class ValueHeadModel(nnx.Module):
    """Wraps a Transformer with a scalar value head."""

    def __init__(self, base_model, *, rngs: nnx.Rngs) -> None:
        self.base_model = base_model
        self.value_head = nnx.Linear(base_model.embedder.input_embedding.shape[0], 1, rngs=rngs)

    def __call__(
        self,
        last_tokens: jax.Array,
        positions: jax.Array,
        cache: dict | None,
        attention_mask: jax.Array,
    ) -> tuple[jax.Array, None]:
        logits, new_cache = self.base_model(
            last_tokens,
            positions=positions,
            cache=cache,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = self.base_model.all_hidden_states.value
        values = self.value_head(hidden)
        return values, new_cache


def get_value_model() -> nnx.Module:
    base_model = get_model()
    return ValueHeadModel(base_model, rngs=nnx.Rngs(params=0))


class DatasetPpoTrainer(PpoTrainer):
    """PPO trainer that forwards dataset outputs to the reward function."""

    def _build_train_example(self, training_input: dict[str, any]) -> TrainExample:
        pad_value = self.sampler.tokenizer.pad_id()
        eos_value = self.sampler.tokenizer.eos_id()

        completion_output = self.sampler(
            input_strings=training_input["prompts"],
            total_generation_steps=self.ppo_config.total_generation_steps,
            max_prompt_length=self.ppo_config.max_prompt_length,
            echo=False,
            temperature=self.ppo_config.temperature,
            top_p=self.ppo_config.top_p,
            top_k=self.ppo_config.top_k,
        )

        completion_ids = common.pad_inputs(
            completion_output.tokens,
            target_length=self.ppo_config.total_generation_steps,
            pad_value=pad_value,
            left=False,
        )
        prompt_ids = completion_output.padded_prompt_tokens

        (
            positions,
            prompt_completion_ids,
            completion_mask,
            _,
            prompt_completion_causal_mask,
        ) = common.process_ids(prompt_ids, completion_ids, pad_value, eos_value)

        logits_to_keep = completion_ids.shape[1]
        old_per_token_logps = common.get_per_token_logps(
            self.model,
            input_tokens=prompt_completion_ids,
            positions=positions,
            attn_mask=prompt_completion_causal_mask,
            logits_to_keep=logits_to_keep,
        )
        ref_per_token_logps = common.get_per_token_logps(
            self.ref_model,
            input_tokens=prompt_completion_ids,
            positions=positions,
            attn_mask=prompt_completion_causal_mask,
            logits_to_keep=logits_to_keep,
        )

        value_logits, _ = self.value_model(
            prompt_completion_ids,
            positions=positions,
            cache=None,
            attention_mask=prompt_completion_causal_mask,
        )
        old_values = value_logits[:, -logits_to_keep - 1 : -1, 0]

        common.clear_memory()

        scores = jnp.array(
            self.reward_fn(training_input["prompts"], completion_output.text, training_input["reference"])
        )
        logr = ref_per_token_logps - old_per_token_logps
        if self.ppo_config.kl_estimator == "k3":
            kl = jnp.exp(logr) - 1 - logr
        else:
            kl = -logr
        non_score_reward = -self.ppo_config.kl_coef * kl
        rewards = non_score_reward
        seq_lens = completion_mask.sum(axis=1) - 1
        rewards = rewards.at[jnp.arange(rewards.shape[0]), seq_lens].add(scores)
        if self.ppo_config.whiten_rewards:
            rewards = common.masked_whiten(rewards, completion_mask, shift_mean=False)

        advantages, returns = common.generalized_advantage_estimation(
            rewards,
            old_values,
            completion_mask,
            self.ppo_config.gamma,
            self.ppo_config.lam,
        )
        advantages = common.masked_whiten(advantages, completion_mask)

        ex = TrainExample(
            input_ids=prompt_completion_ids,
            positions=positions,
            attention_mask=prompt_completion_causal_mask,
            old_logps=old_per_token_logps,
            old_values=old_values,
            advantages=advantages,
            returns=returns,
            completion_mask=completion_mask,
            logits_to_keep=logits_to_keep,
        )
        common.clear_memory()
        return ex

# Policy, reference, and value models
policy_model = get_model()
ref_model = get_model()
value_model = get_value_model()

policy_sampler = sampler_lib.Sampler(
    transformer=policy_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
        num_layers=hf_cfg.num_hidden_layers,
        num_kv_heads=getattr(hf_cfg, "num_key_value_heads", hf_cfg.num_attention_heads),
        head_dim=getattr(hf_cfg, "head_dim", hf_cfg.hidden_size // hf_cfg.num_attention_heads),
    ),
    use_jit=False
)

JUDGE_PROMPT = (
    "You are a fair judge. Given an instruction and two answers, A and B, "
    "decide which answer follows the instruction better. Respond with 'A' or 'B'.\n"
    "Instruction: {input}\nAnswer A: {response_a}\nAnswer B: {response_b}\nJudgement:"
)


def torch_reward_fn(prompts, completions, reference, **kargs):
    """Score policy outputs against reference completions using the Torch reward model."""
    pairs = []
    for inp, pol, ref in zip(prompts, completions, reference):
        pairs.append(JUDGE_PROMPT.format(input=inp, response_a=pol, response_b=ref))
        pairs.append(JUDGE_PROMPT.format(input=inp, response_a=ref, response_b=pol))

    inputs = rm_tokenizer(pairs, return_tensors="pt", padding=True).to(rm_model.device)
    with torch.no_grad():
        out = rm_model(**inputs)
    label_ids = torch.tensor(
        [rm_tokenizer("A", add_special_tokens=False).input_ids[0], rm_tokenizer("B", add_special_tokens=False).input_ids[0]],
        device=rm_model.device,
    )
    last_logits = out.logits[:, -1, :]
    probs = torch.softmax(last_logits[:, label_ids], dim=-1)
    probs = probs.view(len(completions), 2, 2)
    policy_scores = 0.5 * (probs[:, 0, 0] + probs[:, 1, 1])
    return policy_scores.tolist()



# ---------- Evaluation Utility ----------
def evaluate_reward(dataset, sampler):
    """Compute average reward model score against reference answers."""
    scores = []
    for batch in tqdm(dataset):
        prompts = batch["prompts"]
        refs = batch["reference"]
        out = sampler(
            input_strings=prompts,
            total_generation_steps=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            echo=False,
            temperature=0.01,
            top_p=1.0,
            top_k=None,
        )
        batch_scores = torch_reward_fn(prompts, out.text, refs)
        scores.extend(batch_scores)
    return sum(scores) / len(scores)

# ---------- Main Train/Eval Logic ----------
def main():
    # Initial evaluation using the reward model
    init_reward = evaluate_reward(test_dataset, policy_sampler)
    print(f"Initial avg reward: {init_reward}")

    # ---------- Training ----------
    checkpointing_options = ocp.CheckpointManagerOptions(
        save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
    )
    metrics_logging_options = metrics_logger.MetricsLoggerOptions(
        log_dir="/content/tmp/tensorboard/ppo", flush_every_n_steps=20
    )
    training_config = PpoTrainingConfig(
        max_prompt_length=MAX_PROMPT_LENGTH,
        total_generation_steps=TOTAL_GENERATION_STEPS,
        num_ppo_epochs=NUM_PPO_EPOCHS,
        num_mini_batches=NUM_MINI_BATCHES,
        cliprange=CLIP_RANGE,
        cliprange_value=CLIP_RANGE_VALUE,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        eval_every_n_steps=EVAL_EVERY_N_STEPS,
        max_steps=MAX_STEPS,
        metrics_logging_options=metrics_logging_options,
        checkpoint_root_directory=CKPT_DIR,
        checkpointing_options=checkpointing_options,
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
    ppo_trainer = DatasetPpoTrainer(
        model=policy_model,
        ref_model=ref_model,
        value_model=value_model,
        reward_fn=torch_reward_fn,
        sampler=policy_sampler,
        optimizer=optimizer,
        training_config=training_config,
    )
    ppo_trainer.train(train_dataset)

    # ---------- Load Trained Model & Evaluate ----------
    trained_ckpt_path = os.path.join(CKPT_DIR, str(MAX_STEPS), "model_params")
    abs_params = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
        nnx.state(policy_model),
    )
    checkpointer = ocp.StandardCheckpointer()
    trained_params = checkpointer.restore(trained_ckpt_path, target=abs_params)
    nnx.update(policy_model, trained_params)
    final_reward = evaluate_reward(test_dataset, policy_sampler)
    print(f"Final avg reward: {final_reward}")

if __name__ == "__main__":
    main()
