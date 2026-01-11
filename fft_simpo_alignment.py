# Generic Parameters
MODEL_ID = "google/gemma-2-2b-it"
MAX_TARGET_LENGTH = 4096

# FFT Parameters
FFT_LEARNING_RATE = 2e-5
FFT_NUM_EPOCHS = 1
FFT_BATCH_SIZE = 64
FFT_GRADIENT_ACCUMULATION_STEPS = 5
WARMUP_STEPS = 20

# SimPO Parameters
SIMPO_LEARNING_RATE = 8e-7
SIMPO_NUM_ROWS = 24000
SIMPO_BATCH_SIZE = 32
SIMPO_GRADIENT_ACCUMULATION_STEPS = 2

# Prompt format for Simpo
PROMPT_FORMAT = """You are a helpful reasoning assistant that always begins your response with <reasoning>, followed by a chain of thought that plans out your response, then </reasoning> and your actual response in <answer> </answer>. 
For example:
<reasoning>I need to measure exactly 2 liters of milk using a 5-liter container and a 9-liter container. I start with both containers empty, and I have a milk tank to fill from. The goal is to get 2 liters in one of the containers.

This is a classic water jug problem... (continued)</reasoning>
<answer>Mr. Fat can measure exactly 2 liters of milk using the 5-liter and 9-liter containers with the following steps... (continued) ...are poured, leaving 2 liters in the 5-liter container).  

After step 10, the 5-liter container holds exactly 2 liters of milk.

\boxed{2}</answer>

Here is the prompt you should respond to:
{question}
Begin your response with <reasoning>."""

# Debugging Parameters (Set to None to disable)
debugging_max_steps_fft = None
debugging_max_steps_simpo = None

# DO NOT CHANGE BELOW

# Use these standard output tags so that your model's output follow this format in plain text (no JSON/XML):
# <reasoning>model_reasoning_trace</reasoning>
# <answer>model_final_answer</answer>

REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# This prompt template includes the gemma2 2b it prompt format and BOS. Please do not double apply the prompt format and BOS! 
# Also note that the single \ is intentional, due to a formatting issue with the dataset that baked it into the model. 
PROMPT_TEMPLATE = """<bos><start_of_turn>user
You are a helpful reasoning assistant that always begins your response with <reasoning>, followed by a chain of thought that plans out your response, then </reasoning> and your actual response in <answer> </answer>. 
For example:
<reasoning>I need to measure exactly 2 liters of milk using a 5-liter container and a 9-liter container. I start with both containers empty, and I have a milk tank to fill from. The goal is to get 2 liters in one of the containers.

This is a classic water jug problem... (continued)</reasoning>
<answer>Mr. Fat can measure exactly 2 liters of milk using the 5-liter and 9-liter containers with the following steps... (continued) ...are poured, leaving 2 liters in the 5-liter container).  

After step 10, the 5-liter container holds exactly 2 liters of milk.

\boxed{2}</answer>

Here is the prompt you should respond to:
{question}
Begin your response with <reasoning>.<end_of_turn>
<start_of_turn>model
"""

# Use these parameters for greedy decoding; used in competition evaluation
INF_TEMPERATURE=None
INF_TOP_K=1
INF_TOP_P=None
SEED=42
MAX_GENERATION_STEPS=4096 # Changed to fit my model which reasons at 4096 ctx

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1" # This is required to fix a tqdm error

from typing import Any, Iterator
from absl import logging
import datasets
import jax
from flax import nnx
from huggingface_hub import snapshot_download, HfApi, create_repo
import numpy as np
import optax
from tqdm import tqdm
import shutil
import wandb
import safetensors.numpy as safe_np
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
import orbax.checkpoint as ocp
import jax.numpy as jnp
import gc
import transformers
from tunix.sft.dpo import dpo_trainer

def create_fft_dataset(
    batch_size: int,
    max_length: int,
    num_epochs: int,
) -> Iterator[peft_trainer.TrainingInput]:
    """Creates a streaming iterator over the dataset for Full Fine-Tuning.

    This function loads the 'G-reen/instruct-set-longer-fixed' dataset in streaming mode.
    This dataset has been pre-packed, meaning each row in 'text_tokenized'
    already contains multiple concatenated examples up to the context length.

    It iterates through the dataset for the specified number of epochs and yields batches
    of training inputs, handling minor padding/truncation if the pre-packed length
    doesn't perfectly match `max_length`.

    Args:
        batch_size: The number of samples per batch.
        max_length: The maximum sequence length for tokens. Sequences longer than this
            will be truncated, and shorter ones will be padded.
        num_epochs: The number of times to iterate over the entire dataset.

    Yields:
        peft_trainer.TrainingInput: A dataclass containing:
            - input_tokens: Batch of tokenized input sequences (padded/truncated).
            - input_mask: Binary mask indicating valid tokens (1) vs padding (0).
    
    Note:
        The dataset is expected to have a 'text_tokenized' column containing pre-packed sequences.
    """
    
    logging.info("Loading G-reen/instruct-set-longer-fixed (streaming)...")
    
    # This size is approximate for tqdm display
    estimated_size = 151000 
    steps_per_epoch = estimated_size // batch_size
    
    total_tokens = 0
    
    for epoch in range(num_epochs):
        ds = datasets.load_dataset(
            "G-reen/instruct-set-longer-fixed",
            split="train",
            streaming=True,
        )
        
        batch_iterator = ds.iter(batch_size=batch_size, drop_last_batch=True)
        
        for i, batch in tqdm(enumerate(batch_iterator), total=steps_per_epoch, desc=f"FFT Training Epoch {epoch+1}/{num_epochs}"):
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

def save_model_as_safetensors(model: Any, local_model_path: str, output_dir: str):
    """Saves the JAX/Flax model in Hugging Face Safetensors format.

    This function extracts weights from the Flax model, converts them to numpy arrays,
    maps them to the standard Hugging Face naming convention, and saves them as a
    .safetensors file. It also copies necessary config and tokenizer files from the
    original model directory.

    Args:
        model: The Flax/NNX model instance to save.
        local_model_path: The local path to the original model directory (used to
            copy config and tokenizer files).
        output_dir: The directory where the converted model and config files will be saved.
    """
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

def gen_model_input_fn(x: peft_trainer.TrainingInput) -> dict[str, Any]:
    """Prepares model inputs for training with Cross-Entropy Loss (CCE) and packed sequences.

    This function processes a batch of input tokens to generate the necessary components
    for the model's forward pass and loss calculation, specifically handling packed
    sequences where multiple examples are concatenated in a single row.

    It calculates:
    - Position IDs: Resets position indices at the start of each new example (indicated by BOS tokens).
    - Attention Mask: Ensures tokens only attend to other tokens within the same example (segment).
    - Input Mask: Masks out padding tokens for loss calculation.

    Args:
        x: A TrainingInput object containing the raw 'input_tokens' and 'input_mask'.

    Returns:
        dict[str, Any]: A dictionary containing:
            - 'input_tokens': The input token IDs.
            - 'input_mask': Mask for loss calculation (1 for valid tokens, 0 for padding/BOS).
            - 'positions': Position IDs for the tokens, resetting at BOS.
            - 'attention_mask': Segment IDs used to mask attention between packed examples.
    """
    input_tokens = x.input_tokens
    batch_size, seq_len = input_tokens.shape
    bos_id = 2 
    pad_id = 0
    token_ids = jnp.arange(seq_len)[None, :]
    
    # Identify BOS tokens to reset position IDs
    is_bos = (input_tokens == bos_id)
    bos_indices = jnp.where(is_bos, token_ids, 0)
    last_bos_idx = jax.lax.cummax(bos_indices, axis=1)
    
    # Calculate positions relative to the last BOS token
    positions = (token_ids - last_bos_idx).astype(jnp.int32)
    
    # Create segment IDs for attention masking
    is_bos_int = is_bos.astype(jnp.int32)
    segment_ids = jnp.cumsum(is_bos_int, axis=1) + 1
    
    # Create attention mask (allow attention only within the same segment)
    valid_mask = (input_tokens != pad_id).astype(jnp.int32)
    attention_mask = segment_ids * valid_mask
    
    # Input mask for loss calculation (ignore BOS tokens)
    input_mask = valid_mask * (1 - is_bos_int)
    
    return {
        'input_tokens': input_tokens,
        'input_mask': input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

# ==========================================
# Model/TPU Setup
# ==========================================
# Initialize JAX devices and setup the mesh for distributed training (FSDP).
# We use a 1D mesh here where all devices are used for FSDP (Fully Sharded Data Parallel).
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
# Full Fine-Tuning (FFT)
# ==========================================
# To maximize training performance, we perform full-finetuning on the dataset instead of LoRA.
print("\n=== Starting Full Fine-Tuning (FFT) ===")

# Layer-level rematerialization (gradient checkpointing) 
# This increases compute usage, but allows a bigger bsz (64) which maximizes TPU utilization.
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

# Calculate (estimated) total training steps for the learning rate scheduler.
ESTIMATED_DATASET_SIZE = 151000
TOTAL_UPDATES = (ESTIMATED_DATASET_SIZE * FFT_NUM_EPOCHS // FFT_BATCH_SIZE) // FFT_GRADIENT_ACCUMULATION_STEPS

# Setup the optimizer and scheduler.
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=FFT_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=TOTAL_UPDATES,
)
optimizer = optax.adamw(learning_rate=schedule)

# We don't really need extra checkpoints so we disable them and prevent possible storage issues.
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=1000000,
    max_to_keep=1,
)

# Configure the training loop.
training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=100,
    max_steps=debugging_max_steps_fft,
    checkpoint_root_directory=None, # Disable intermediate checkpoints to prevent freezing
    use_weighted_gradient_accumulation=True,
    gradient_accumulation_steps=FFT_GRADIENT_ACCUMULATION_STEPS,
    checkpointing_options=checkpointing_options,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir="tensorboard_fft",
        flush_every_n_steps=10,
    ),
)

# Note: Although named 'PeftTrainer', it supports full fine-tuning when not using LoRA.
trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)
# CCE saves >10gb memory per device. FA (not shown, comes bundled with model.py) saves another chunk, just enough for training to fit.
trainer = trainer.with_cce_loss()

trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
train_ds = create_fft_dataset(FFT_BATCH_SIZE, MAX_TARGET_LENGTH, FFT_NUM_EPOCHS)

print("Starting FFT training...")
with mesh:
    trainer.train(train_ds, None)

print("FFT Training complete!")



# ==========================================
# Transition to SimPO
# ==========================================
# 1. Save the FFT model to a temporary location so we can reload it fresh.
FFT_MODEL_DIR = "/tmp/fft_model"
save_model_as_safetensors(trainer.model, local_model_path, FFT_MODEL_DIR)

# 2. Cleanup Memory & Storage
print("Cleaning up FFT resources...")
# Delete the trainer and model to free up JAX memory
del trainer
del model
del optimizer
del train_ds
gc.collect()

# Clear JAX caches if possible
jax.clear_caches()

# Explicitly clear any monitoring listeners (e.g. wandb) from the previous phase
# This prevents "wandb: You must call wandb.init()" errors if JAX ops trigger callbacks
try:
    jax.monitoring.clear_event_listeners()
except Exception:
    pass

# Delete the FFT checkpoints to save disk space (we only need the safetensors now)
if os.path.exists("/tmp/checkpoints_fft"):
    shutil.rmtree("/tmp/checkpoints_fft")
print("FFT resources cleaned up.")



# ==========================================
# SimPO Alignment
# ==========================================
def create_simpo_dataset(
    batch_size: int,
    tokenizer: Any,
) -> Iterator[dpo_trainer.DataInput]:
    """Creates a streaming iterator for SimPO training using the 'sumthink' dataset.

    This function loads the 'G-reen/sumthink_fixed_cleaned' dataset in streaming mode,
    which contains preference pairs (chosen/rejected) for alignment training.

    It performs the following processing steps:
    1. Shuffles the dataset with a buffer.
    2. Wraps each prompt using the global `PROMPT_TEMPLATE` to enforce a specific reasoning format.
    3. Applies the tokenizer's chat template to format the prompt for the model.
    4. Yields batches of `dpo_trainer.DataInput` containing the processed prompts and responses.

    Args:
        batch_size: The number of samples per batch.
        tokenizer: The tokenizer used to apply the chat template.

    Yields:
        dpo_trainer.DataInput: A dataclass containing:
            - prompts: List of formatted prompt strings.
            - chosen_responses: List of preferred response strings.
            - rejected_responses: List of rejected response strings.
    """
    
    logging.info("Loading G-reen/sumthink_fixed_cleaned (streaming)...")
    ds = datasets.load_dataset(
        "G-reen/sumthink_fixed_cleaned",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=42, buffer_size=10000)
    # Calculate steps (approximate since we are streaming)
    total_steps = SIMPO_NUM_ROWS // batch_size
    
    batch_iterator = ds.iter(batch_size=batch_size, drop_last_batch=True)
    
    for i, batch in tqdm(enumerate(batch_iterator), total=total_steps, desc="SimPO Training Steps"):
        # Apply wrapper
        wrapped_prompts = [PROMPT_TEMPLATE.replace("{question}", p) for p in batch['prompt']]
        
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

print("\n=== Starting SimPO Alignment ===")

# Ensure wandb is initialized for SimPO phase to catch any early logs
if wandb.run is None:
    wandb.init(project="tunix", name="simpo_alignment", resume="allow")

# Load Tokenizer
print("Loading Tokenizer...")
tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)

# Load Model from the FFT result
print(f"Loading model from {FFT_MODEL_DIR}...")
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=FFT_MODEL_DIR,
        config=model_config, # Re-use the config from earlier
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
    max_steps=debugging_max_steps_simpo,
    checkpoint_root_directory=None, # Disable intermediate checkpoints to prevent freezing
    checkpointing_options=ocp.CheckpointManagerOptions(
        max_to_keep=1,
        save_interval_steps=10000,
    ),
    use_weighted_gradient_accumulation=False,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir="tensorboard_simpo",
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

# NOTE TO JUDGES: 
# Since this notebook requires a tunix fork, it may be more stable to upload the model to HF, then eval in a seperate notebook with a standard tunix env that downloads the model from HF.
"""
api = HfApi()
HF_REPO_ID_SIMPO_FINAL = "G-reen/gemma-2-2b-it-fft-simpo-tpu"
print(f"Pushing Final SimPO model to {HF_REPO_ID_SIMPO_FINAL}...")
create_repo(HF_REPO_ID_SIMPO_FINAL, private=False, exist_ok=True)
api.upload_folder(
    folder_path=FINAL_SIMPO_DIR,
    repo_id=HF_REPO_ID_SIMPO_FINAL,
    commit_message="Upload Gemma-2-2B FFT+SimPO Aligned",
)

print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID_SIMPO_FINAL}")
"""

# ==========================================
# INFERENCE VERIFICATION
# ==========================================
# Cleanup SimPO Memory
print("Cleaning up SimPO resources...")
del trainer
del model
del optimizer
del simpo_ds
gc.collect()
jax.clear_caches()
try:
    jax.monitoring.clear_event_listeners()
except Exception:
    pass
print("SimPO resources cleaned up.")

CKPT_DIR = '/tmp/final_simpo_model'

from tunix.generate import sampler as sampler_lib

print("\n=== Starting Inference Verification ===")

print(f"Loading model from {CKPT_DIR}...")
with mesh:
    # Load model for inference
    inference_model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=CKPT_DIR,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )
    
    # Create Sampler
    MAX_PROMPT_LENGTH = 1024 

    sampler = sampler_lib.Sampler(
        transformer=inference_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    # Prepare Prompt
    inference_question = "Write a heartbreaking story about a dog."
    # Create the prompt using the template, replacing {question}
    prompt = PROMPT_TEMPLATE.replace("{question}", inference_question)

    print(f"Prompting model with: '{inference_question}'")

    # Run Inference
    output = sampler(
        input_strings=[prompt],
        max_generation_steps=MAX_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        temperature=INF_TEMPERATURE if INF_TEMPERATURE is not None else 0.0,
        top_k=INF_TOP_K,
        top_p=INF_TOP_P,
        seed=SEED
    )

    print("\n=== Model Output ===")
    print(output.text[0])
    print("====================")

# ==========================================
# UNRESTRICTED MODEL INFERENCE (KAGGLE)
# ==========================================
import kagglehub
print("\n=== Starting Unrestricted Model Inference ===")

# 1. Cleanup Memory from previous step
print("Cleaning up previous inference resources...")
try:
    del inference_model
    del sampler
except NameError:
    pass
gc.collect()
jax.clear_caches()
print("Resources cleaned up.")

KAGGLE_MODEL_HANDLE = "green000/gemma-2-2b-it-fft-3epoch-simpo-adj"
print(f"Downloading model {KAGGLE_MODEL_HANDLE} from Kaggle...")
unrestricted_model_path = kagglehub.model_download(KAGGLE_MODEL_HANDLE)
print(f"Model downloaded to: {unrestricted_model_path}")

print("Loading unrestricted model...")
with mesh:
    # Load model
    unrestricted_model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=unrestricted_model_path,
        config=model_config, # Re-use global config
        mesh=mesh,
        dtype=jnp.bfloat16,
    )
    
    # Create Sampler
    unrestricted_sampler = sampler_lib.Sampler(
        transformer=unrestricted_model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=MAX_PROMPT_LENGTH + MAX_GENERATION_STEPS + 256,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )

    # Use same prompt as before
    inference_question = "Write a heartbreaking story about a dog."
    prompt = PROMPT_TEMPLATE.replace("{question}", inference_question)
    print(f"Prompting unrestricted model with: '{inference_question}'")

    # Run Inference
    output = unrestricted_sampler(
        input_strings=[prompt],
        max_generation_steps=MAX_GENERATION_STEPS,
        max_prompt_length=MAX_PROMPT_LENGTH,
        temperature=INF_TEMPERATURE if INF_TEMPERATURE is not None else 0.0,
        top_k=INF_TOP_K,
        top_p=INF_TOP_P,
        seed=SEED
    )

    print("\n=== Unrestricted Model Output ===")
    print(output.text[0])
    print("====================")