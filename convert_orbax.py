
import os
import shutil
from typing import Any, Iterator

from absl import logging
import jax
from flax import nnx
from huggingface_hub import snapshot_download, HfApi, create_repo
import numpy as np
import optax
from tqdm import tqdm
import safetensors.numpy as safe_np
import jax.numpy as jnp
import orbax.checkpoint as ocp

from tunix.models.gemma import model as gemma_lib
from tunix.sft import checkpoint_manager
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer

# Constants
MODEL_ID = "google/gemma-2-2b"
ORBAX_REPO_ID = "G-reen/gemma-2-2b-ultrafine-orbax"
FINAL_REPO_ID = "G-reen/gemma-2-2b-ultrafine"
LOCAL_ORBAX_DIR = "/tmp/orbax_restore"
LOCAL_BASE_MODEL_DIR = "/tmp/base_model"
OUTPUT_DIR = "/tmp/final_converted_model"

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
             # Exclude .safetensors (handled separately) and the index file (we are creating a monolith file)
             if not file.endswith(".safetensors") and file != "model.safetensors.index.json":
                src = os.path.join(local_model_path, file)
                if os.path.isfile(src):
                     shutil.copy(src, os.path.join(output_dir, file))
    print("Save complete.")

def main():
    # Setup JAX Mesh
    devices = jax.devices()
    num_devices = len(devices)
    tp_size = 1
    fsdp_size = num_devices // tp_size
    mesh = jax.sharding.Mesh(
        np.array(devices).reshape(fsdp_size, tp_size),
        ('fsdp', 'tp'),
    )
    print(f"Devices: {num_devices}, Mesh: {mesh.shape}")

    # 1. Download Orbax Checkpoint
    print(f"Downloading Orbax checkpoint from {ORBAX_REPO_ID}...")
    snapshot_download(
        repo_id=ORBAX_REPO_ID,
        local_dir=LOCAL_ORBAX_DIR,
    )
    
    # 2. Download Base Model Configs
    print(f"Downloading base model configs from {MODEL_ID}...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_BASE_MODEL_DIR,
        ignore_patterns=["*.safetensors", "*.pth", "*.bin", "original/*"],
    )

    # 3. Initialize Model
    print("Initializing model structure...")
    model_config = gemma_lib.ModelConfig.gemma2_2b()
    model_config.remat_config = gemma_lib.RematConfig.LAYER
    
    with mesh:
        model = gemma_lib.Gemma(model_config, rngs=nnx.Rngs(0))

    # 4. Restore Checkpoint
    print("Restoring checkpoint...")
    ckpt_mgr = checkpoint_manager.CheckpointManager(root_directory=LOCAL_ORBAX_DIR)
    step = ckpt_mgr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {LOCAL_ORBAX_DIR}")
    
    print(f"Found latest step: {step}")
    with mesh:
        ckpt_mgr.maybe_restore(model, step=step)
    
    # 5. Convert and Save
    print("Converting to Safetensors...")
    save_fft_model_as_safetensors(model, LOCAL_BASE_MODEL_DIR, OUTPUT_DIR)
    
    # 6. Upload
    print(f"Pushing to {FINAL_REPO_ID}...")
    api = HfApi()
    create_repo(FINAL_REPO_ID, private=False, exist_ok=True)
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=FINAL_REPO_ID,
        commit_message="Re-upload converted model from Orbax checkpoint",
    )
    print(f"Done! Model available at https://huggingface.co/{FINAL_REPO_ID}")

if __name__ == "__main__":
    main()
