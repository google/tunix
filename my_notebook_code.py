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


from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils
import orbax.checkpoint as ocp
import jax.numpy as jnp

# Model
MODEL_ID = "google/gemma-2-2b"

# Params
LEARNING_RATE = 2e-5
GLOBAL_BATCH_SIZE = 64
MAX_TARGET_LENGTH = 4096
NUM_ROWS = 100000
GRADIENT_ACCUMULATION_STEPS = 5
WARMUP_STEPS = 20

def create_train_dataset(
    batch_size: int,
    max_length: int,
) -> Iterator[peft_trainer.TrainingInput]:
    """Creates a streaming iterator over G-reen/instruct-set-longer-fixed efficiently."""
    
    logging.info("Loading G-reen/instruct-set-longer-fixed (streaming)...")
    ds = datasets.load_dataset(
        "G-reen/instruct-set-longer-fixed",
        split="train",
        streaming=True,
    )
    
    total_steps = NUM_ROWS // batch_size
    
    batch_iterator = ds.iter(batch_size=batch_size)
    
    total_tokens = 0
    
    for i, batch in tqdm(enumerate(batch_iterator), total=total_steps, desc="Training Steps"):
        if i >= total_steps:
            break
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
    
    print(f"Total tokens trained on: {total_tokens:,}")

devices = jax.devices()
num_devices = len(devices)

tp_size = 1
fsdp_size = num_devices // tp_size

mesh = jax.sharding.Mesh(
    np.array(devices).reshape(fsdp_size, tp_size),
    ('fsdp', 'tp'),
)

print(f"Devices: {num_devices}, Mesh: {mesh.shape} (fsdp={fsdp_size}, tp={tp_size})")

print(f"Downloading {MODEL_ID} from Hugging Face...")
local_model_path = snapshot_download(
    repo_id=MODEL_ID,
    ignore_patterns=["*.pth", "original/*"],
)
print(f"Model downloaded to: {local_model_path}")


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

TOTAL_UPDATES = (NUM_ROWS // GLOBAL_BATCH_SIZE) // GRADIENT_ACCUMULATION_STEPS
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=TOTAL_UPDATES,
)
optimizer = optax.adamw(learning_rate=schedule)

checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=NUM_ROWS // GLOBAL_BATCH_SIZE,
    max_to_keep=1,
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=100,
    max_steps=None,
    checkpoint_root_directory="/tmp/checkpoints",
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
    
    # Mask out BOS tokens so we don't predict the start of the next document
    # (isolating the documents completely)
    input_mask = valid_mask * (1 - is_bos_int)

    return {
        'input_tokens': input_tokens,
        'input_mask': input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

train_ds = create_train_dataset(GLOBAL_BATCH_SIZE, MAX_TARGET_LENGTH)

print("Starting training...")
with mesh:
    trainer.train(train_ds, None)

print("Training complete!")

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

CHECKPOINT_DIR = "/tmp/checkpoints"
FINAL_MODEL_DIR = "/tmp/final_model"
HF_REPO_ID_SAFETENSORS = "G-reen/gemma-2-2b-ultrafine"
HF_REPO_ID_ORBAX = "G-reen/gemma-2-2b-ultrafine-orbax"

save_fft_model_as_safetensors(trainer.model, local_model_path, FINAL_MODEL_DIR)

print(f"Pushing Safetensors model to {HF_REPO_ID_SAFETENSORS}...")
api = HfApi()
create_repo(HF_REPO_ID_SAFETENSORS, private=False, exist_ok=True)
api.upload_folder(
    folder_path=FINAL_MODEL_DIR,
    repo_id=HF_REPO_ID_SAFETENSORS,
    commit_message="Upload Gemma-2-2B fine-tuned with tunix (Safetensors FFT)",
)

print(f"Safetensors Model pushed to: https://huggingface.co/{HF_REPO_ID_SAFETENSORS}")

print(f"Pushing Orbax checkpoint to {HF_REPO_ID_ORBAX}...")
create_repo(HF_REPO_ID_ORBAX, private=False, exist_ok=True)
api.upload_folder(
    folder_path=CHECKPOINT_DIR,
    repo_id=HF_REPO_ID_ORBAX,
    commit_message="Upload Orbax checkpoints",
)
print(f"Orbax Checkpoint pushed to: https://huggingface.co/{HF_REPO_ID_ORBAX}")

# Note: Tags like "safetensors" and "orbax" are usually added to the README.md metadata 
# manually or via card update, but separating repos clearly distinguishes them.

# To reload the model later:
# from tunix.sft.checkpoint_manager import CheckpointManager
# ckpt_manager = CheckpointManager(root_directory="path/to/downloaded/checkpoint")
# ckpt_manager.maybe_restore(model)