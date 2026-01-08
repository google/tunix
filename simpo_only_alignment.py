
import gc
from typing import Any, Iterator
import time

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
import transformers

from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import utils
from tunix.sft.dpo import dpo_trainer
import orbax.checkpoint as ocp
import jax.numpy as jnp

# --- Common Params ---
# CHANGED: Use the already FFT'd model
MODEL_ID = "G-reen/gemma-2-2b-it-fft"
MAX_TARGET_LENGTH = 4096

# FFT Params removed as we are skipping FFT

# SimPO Params (Identical to original)
SIMPO_LEARNING_RATE = 8e-7
SIMPO_NUM_ROWS = 24000
SIMPO_BATCH_SIZE = 32
SIMPO_GRADIENT_ACCUMULATION_STEPS = 2

# --- Prompt Wrapper ---
PROMPT_WRAPPER = """You are a helpful reasoning assistant that always begins your response with <reasoning>, followed by a chain of thought that plans out your response, then </reasoning> and your actual response in <answer> </answer>. 
For example:
<reasoning>I need to measure exactly 2 liters of milk using a 5-liter container and a 9-liter container. I start with both containers empty, and I have a milk tank to fill from. The goal is to get 2 liters in one of the containers.

This is a classic water jug problem... (continued)</reasoning>
<answer>Mr. Fat can measure exactly 2 liters of milk using the 5-liter and 9-liter containers with the following steps... (continued) ...are poured, leaving 2 liters in the 5-liter container).  

After step 10, the 5-liter container holds exactly 2 liters of milk.

\boxed{2}</answer>

Here is the prompt you should respond to:
{question}
Begin your response with <reasoning>."""

def create_simpo_dataset(
    batch_size: int,
    tokenizer: Any,
) -> Iterator[dpo_trainer.DataInput]:
    """Creates a streaming iterator over G-reen/sumthink_fixed_cleaned."""
    
    logging.info("Loading G-reen/sumthink_fixed_cleaned (streaming)...")
    ds = datasets.load_dataset(
        "G-reen/sumthink_fixed_cleaned",
        split="train",
        streaming=True,
    )
    
    # Calculate steps (approximate since we are streaming)
    total_steps = SIMPO_NUM_ROWS // batch_size
    
    batch_iterator = ds.iter(batch_size=batch_size)
    
    for i, batch in tqdm(enumerate(batch_iterator), total=total_steps, desc="SimPO Training Steps"):
        # Apply wrapper
        wrapped_prompts = [PROMPT_WRAPPER.replace("{question}", p) for p in batch['prompt']]
        
        # Apply chat template
        formatted_prompts = []
        for p in wrapped_prompts:
            messages = [{"role": "user", "content": p}]
            # Ensure we get a string back, not tokens, since DataInput expects strings
            formatted_p = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(formatted_p)
            
        chosen = batch['chosen']
        rejected = batch['rejected']
        
        yield dpo_trainer.DataInput(
            prompts=formatted_prompts,
            chosen_responses=chosen,
            rejected_responses=rejected,
        )

def save_model_as_safetensors(model, local_model_path, output_dir):
    """Saves the model in standard Hugging Face safetensors format."""
    print(f"Saving model to {output_dir}...")
    
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

# ==========================================
# PHASE 2: SimPO Alignment (Directly starting here)
# ==========================================
print("\n=== Starting SimPO Alignment (Skipping FFT) ===")

model_config = gemma_lib.ModelConfig.gemma2_2b()
model_config.remat_config = gemma_lib.RematConfig.LAYER

# Load Tokenizer
print("Loading Tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)

# Load Model directly from downloaded path
print(f"Loading model from {local_model_path}...")
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=local_model_path,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )

nnx.display(model)

# SimPO Scheduler & Optimizer
SIMPO_TOTAL_UPDATES = (SIMPO_NUM_ROWS // SIMPO_BATCH_SIZE) // SIMPO_GRADIENT_ACCUMULATION_STEPS
simpo_warmup_steps = int(0.1 * SIMPO_TOTAL_UPDATES)

simpo_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=SIMPO_LEARNING_RATE,
    warmup_steps=simpo_warmup_steps,
    decay_steps=SIMPO_TOTAL_UPDATES,
)
optimizer = optax.adamw(learning_rate=simpo_schedule)

# SimPO Config
simpo_config = dpo_trainer.SimPOTrainingConfig(
    algorithm="simpo",
    beta=2.0, # Typical for SimPO
    gamma=1.0, # Target reward margin
    max_prompt_length=1536,
    max_response_length=2560,
    eval_every_n_steps=100,
    max_steps=None,
    checkpoint_root_directory="/tmp/checkpoints_simpo",
    use_weighted_gradient_accumulation=True,
    gradient_accumulation_steps=SIMPO_GRADIENT_ACCUMULATION_STEPS,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir="/kaggle/working/tensorboard_simpo",
        flush_every_n_steps=10,
    ),
)

# Initialize SimPO Trainer
# Note: SimPO is reference-free, so ref_model is None
trainer = dpo_trainer.SimPOTrainer(
    model=model,
    ref_model=None, 
    optimizer=optimizer,
    training_config=simpo_config,
    tokenizer=tokenizer,
)

simpo_ds = create_simpo_dataset(SIMPO_BATCH_SIZE, tokenizer)

print("Starting SimPO training...")
with mesh:
    trainer.train(simpo_ds, None)
    
print("SimPO Training complete!")

FINAL_SIMPO_DIR = "/tmp/final_simpo_model"
save_model_as_safetensors(trainer.model, local_model_path, FINAL_SIMPO_DIR)

# ==========================================
# UPLOAD PHASE
# ==========================================
HF_REPO_ID_SIMPO = "G-reen/gemma-2-2b-it-fft-simpo"

print(f"Pushing Final SimPO model to {HF_REPO_ID_SIMPO}...")
api = HfApi()
create_repo(HF_REPO_ID_SIMPO, private=False, exist_ok=True)
api.upload_folder(
    folder_path=FINAL_SIMPO_DIR,
    repo_id=HF_REPO_ID_SIMPO,
    commit_message="Upload Gemma-2-2B FFT+SimPO Aligned (SimPO Only Run)",
)

print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID_SIMPO}")
