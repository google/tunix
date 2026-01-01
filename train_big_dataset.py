from typing import Any, Iterator

from absl import logging
import datasets
import jax
from flax import nnx
from huggingface_hub import snapshot_download, HfApi, create_repo
import numpy as np
import optax
from tqdm import tqdm
import os
import shutil
import safetensors.numpy as safe_np
from datetime import datetime

from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils
import orbax.checkpoint as ocp
import jax.numpy as jnp

# --- Configuration ---
# Dataset and Model
DATASET_ID = "G-reen/big_set" # Replaced medium_set with big_set
MODEL_ID = "google/gemma-2-2b"

# Training Hyperparameters
LEARNING_RATE = 2e-5
GLOBAL_BATCH_SIZE = 64
MAX_TARGET_LENGTH = 4096
# 200k rows / 64 batch size ~= 3125 micro-steps
# 3125 / 5 grad_accum ~= 625 update steps.
TOTAL_DATASET_ROWS = 359000
ROWS_PER_SESSION = 200000 
GRADIENT_ACCUMULATION_STEPS = 5
WARMUP_STEPS = 20

# Calculated batch config
MICRO_STEPS_PER_SESSION = ROWS_PER_SESSION // GLOBAL_BATCH_SIZE
UPDATE_STEPS_PER_SESSION = MICRO_STEPS_PER_SESSION // GRADIENT_ACCUMULATION_STEPS

# Paths
CHECKPOINT_DIR = "/tmp/checkpoints"
FINAL_MODEL_DIR = "/tmp/final_model"
HF_REPO_ID_SAFETENSORS = "G-reen/gemma-2-2b-big-set-safetensors" # Update repo name
HF_REPO_ID_ORBAX = "G-reen/gemma-2-2b-big-set-orbax" # Update repo name

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

print(f"Session Configuration:")
print(f"  Rows per session: {ROWS_PER_SESSION}")
print(f"  Micro steps: {MICRO_STEPS_PER_SESSION}")
print(f"  Update steps: {UPDATE_STEPS_PER_SESSION}")

def create_train_dataset(
    batch_size: int,
    max_length: int,
    skip_examples: int = 0,
) -> Iterator[peft_trainer.TrainingInput]:
    """Creates a streaming iterator over the dataset."""
    
    logging.info(f"Loading {DATASET_ID} (streaming)...")
    ds = datasets.load_dataset(
        DATASET_ID,
        split="train",
        streaming=True,
    )
    
    if skip_examples > 0:
        print(f"Skipping first {skip_examples} examples in dataset...")
        ds = ds.skip(skip_examples)
    
    # We iterate indefinitely (or until dataset exhaustion). 
    # Control of when to stop is handled by the Trainer's max_steps.
    batch_iterator = ds.iter(batch_size=batch_size)
    
    total_tokens = 0
    
    # tqdm is for visual feedback, using a large number as proxy
    for i, batch in tqdm(enumerate(batch_iterator), desc="Streaming Batches"):
        tokens = np.array(batch['text_tokenized'], dtype=np.int32)

        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
        elif tokens.shape[1] < max_length:
             tokens = np.pad(tokens, ((0,0), (0, max_length - tokens.shape[1])), constant_values=0)
        
        mask = (tokens != 0).astype(np.int32)
        total_tokens += np.sum(mask)
        
        yield peft_trainer.TrainingInput(
            input_tokens=tokens,
            input_mask=mask,
        )

# --- 1. Checkpoint Preparation ---
print(f"Checking for existing checkpoints in {HF_REPO_ID_ORBAX}...")
try:
    # Attempt to download existing Orbax checkpoints to resume training
    snapshot_download(
        repo_id=HF_REPO_ID_ORBAX,
        local_dir=CHECKPOINT_DIR,
        ignore_patterns=["*.git*"],
    )
    print("Found and downloaded existing checkpoints.")
except Exception as e:
    print(f"No existing checkpoints found or download failed (starting fresh?): {e}")

# --- 2. Model Initialization ---
devices = jax.devices()
num_devices = len(devices)
tp_size = 1
fsdp_size = num_devices // tp_size

mesh = jax.sharding.Mesh(
    np.array(devices).reshape(fsdp_size, tp_size),
    ('fsdp', 'tp'),
)

print(f"Devices: {num_devices}, Mesh: {mesh.shape} (fsdp={fsdp_size}, tp={tp_size})")

print(f"Downloading {MODEL_ID} base model...")
local_model_path = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.pth", "original/*"],
)

model_config = gemma_lib.ModelConfig.gemma2_2b()
model_config.remat_config = gemma_lib.RematConfig.LAYER

print("Loading model from safetensors...")
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=local_model_path,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )

nnx.display(model)

# --- 3. Trainer Setup ---
# Calculate total steps for the entire dataset
TOTAL_TRAINING_STEPS = (TOTAL_DATASET_ROWS // GLOBAL_BATCH_SIZE) // GRADIENT_ACCUMULATION_STEPS
print(f"Total training steps for scheduler: {TOTAL_TRAINING_STEPS}")

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=TOTAL_TRAINING_STEPS,
)
optimizer = optax.adamw(learning_rate=schedule)

checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=UPDATE_STEPS_PER_SESSION, # Try to save at end of session
    max_to_keep=2,
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=10000, # Disable freq eval
    max_steps=None, # Will set dynamically
    checkpoint_root_directory=CHECKPOINT_DIR,
    use_weighted_gradient_accumulation=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    checkpointing_options=checkpointing_options,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir="/kaggle/working/tensorboard",
        flush_every_n_steps=10,
    ),
)

trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)
trainer = trainer.with_cce_loss()

def gen_model_input_fn(x: peft_trainer.TrainingInput):
    input_tokens = x.input_tokens
    # Ensure working with JAX arrays
    
    batch_size, seq_len = input_tokens.shape

    bos_id = 2 
    pad_id = 0

    token_ids = jnp.arange(seq_len)[None, :]
    is_bos = (input_tokens == bos_id)

    bos_indices = jnp.where(is_bos, token_ids, 0)
    
    last_bos_idx = jax.lax.cummax(bos_indices, axis=1)
    
    positions = (token_ids - last_bos_idx).astype(jnp.int32)

    is_bos_int = is_bos.astype(jnp.int32)
    segment_ids = jnp.cumsum(is_bos_int, axis=1) + 1

    valid_mask = (input_tokens != pad_id).astype(jnp.int32)
    attention_mask = segment_ids * valid_mask
    
    input_mask = valid_mask * (1 - is_bos_int)

    return {
        'input_tokens': input_tokens,
        'input_mask': input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

# --- 4. Resume & Train ---
current_step = trainer.train_steps
print(f"Resuming from step: {current_step}")

target_step = current_step + UPDATE_STEPS_PER_SESSION
print(f"Targeting step: {target_step}")

# Update config to stop at target step
trainer.config.max_steps = target_step

# Calculate examples to skip
# iter_steps is the total number of micro-batches (accumulated batches) processed
# trainer.iter_steps matches the number of times `next(iterator)` was called during prior training
prior_micro_batches = trainer.iter_steps
skipped_examples = prior_micro_batches * GLOBAL_BATCH_SIZE

print(f"Skipping {prior_micro_batches} prior micro-batches ({skipped_examples} examples)...")

# IMPORTANT: We manage the iterator cursor externally by skipping.
# We must tell PeftTrainer NOT to skip internally using is_managed_externally.
trainer.is_managed_externally = True

train_ds = create_train_dataset(
    GLOBAL_BATCH_SIZE, 
    MAX_TARGET_LENGTH, 
    skip_examples=skipped_examples
)

print("Starting training session...")
with mesh:
    trainer.train(train_ds, None)

print(f"Session complete. Reached step {trainer.train_steps}.")

# --- 5. Save & Upload ---

# Force save last checkpoint
trainer.close()

def save_fft_model_as_safetensors(model, local_model_path, output_dir):
    """Saves the FFT model in standard Hugging Face safetensors format."""
    print(f"Saving FFT model to {output_dir}...")
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    tensors = {}

    print("Extracting embeddings...")
    input_embed = np.array(model.embedder.input_embedding.value)
    tensors["model.embed_tokens.weight"] = input_embed
    tensors["lm_head.weight"] = input_embed
    
    print("Extracting final norm...")
    tensors["model.norm.weight"] = np.array(model.final_norm.scale.value)
    
    print("Extracting layers...")
    for i, layer in enumerate(tqdm(model.layers, desc="Saving layers")):
        prefix = f"model.layers.{i}"
        
        tensors[f"{prefix}.input_layernorm.weight"] = np.array(layer.pre_attention_norm.scale.value)
        if layer.use_post_attn_norm:
             tensors[f"{prefix}.post_attention_layernorm.weight"] = np.array(layer.post_attn_norm.scale.value)
        tensors[f"{prefix}.pre_feedforward_layernorm.weight"] = np.array(layer.pre_ffw_norm.scale.value)
        if layer.use_post_ffw_norm:
             tensors[f"{prefix}.post_feedforward_layernorm.weight"] = np.array(layer.post_ffw_norm.scale.value)
             
        tensors[f"{prefix}.mlp.gate_proj.weight"] = np.array(layer.mlp.gate_proj.kernel.value).T
        tensors[f"{prefix}.mlp.up_proj.weight"] = np.array(layer.mlp.up_proj.kernel.value).T
        tensors[f"{prefix}.mlp.down_proj.weight"] = np.array(layer.mlp.down_proj.kernel.value).T
        
        q_w = np.array(layer.attn.q_einsum.w.value)
        tensors[f"{prefix}.self_attn.q_proj.weight"] = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])
        
        kv_w = np.array(layer.attn.kv_einsum.w.value)
        k_w = kv_w[0]
        v_w = kv_w[1]
        
        tensors[f"{prefix}.self_attn.k_proj.weight"] = k_w.transpose(0, 2, 1).reshape(-1, k_w.shape[1])
        tensors[f"{prefix}.self_attn.v_proj.weight"] = v_w.transpose(0, 2, 1).reshape(-1, v_w.shape[1])
        
        o_w = np.array(layer.attn.attn_vec_einsum.w.value)
        tensors[f"{prefix}.self_attn.o_proj.weight"] = o_w.transpose(2, 0, 1).reshape(o_w.shape[2], -1)

    print(f"Saving to {os.path.join(output_dir, 'model.safetensors')}...")
    safe_np.save_file(tensors, os.path.join(output_dir, "model.safetensors"))
    
    print("Copying config and tokenizer files...")
    for file in os.listdir(local_model_path):
        if file.startswith("config") or file.startswith("tokenizer") or file.endswith(".json") or file.endswith(".model"):
              if not file.endswith(".safetensors") and file != "model.safetensors.index.json":
                src = os.path.join(local_model_path, file)
                if os.path.isfile(src):
                     shutil.copy(src, os.path.join(output_dir, file))
    print("Save complete.")

save_fft_model_as_safetensors(trainer.model, local_model_path, FINAL_MODEL_DIR)

print(f"Pushing Safetensors model to {HF_REPO_ID_SAFETENSORS}...")
api = HfApi()
create_repo(HF_REPO_ID_SAFETENSORS, private=False, exist_ok=True)
api.upload_folder(
    folder_path=FINAL_MODEL_DIR,
    repo_id=HF_REPO_ID_SAFETENSORS,
    commit_message=f"Session update: Step {trainer.train_steps}",
)
print(f"Safetensors Model pushed.")

print(f"Pushing Orbax checkpoint to {HF_REPO_ID_ORBAX}...")
create_repo(HF_REPO_ID_ORBAX, private=False, exist_ok=True)

# Note: CheckpointManager saves into subfolders like checkpoint_0, checkpoint_100, etc.
# We want to upload the whole CHECKPOINT_DIR
api.upload_folder(
    folder_path=CHECKPOINT_DIR,
    repo_id=HF_REPO_ID_ORBAX,
    commit_message=f"Session update: Step {trainer.train_steps}",
)
print(f"Orbax Checkpoint pushed.")
