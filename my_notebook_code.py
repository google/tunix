from typing import Any, Iterator

from absl import logging
import datasets
import jax
from flax import nnx
from huggingface_hub import snapshot_download, HfApi, create_repo
import numpy as np
import optax
from tqdm import tqdm

from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils
import orbax.checkpoint as ocp
import jax.numpy as jnp

# Model
MODEL_ID = "google/gemma-2-2b"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma2.model"

LEARNING_RATE = 5e-6
GLOBAL_BATCH_SIZE = 64
MAX_TARGET_LENGTH = 3072
NUM_ROWS = 100000
GRADIENT_ACCUMULATION_STEPS = 12
WARMUP_STEPS = 50

def create_train_dataset(
    tokenizer: Any,
    batch_size: int,
    max_length: int,
) -> Iterator[peft_trainer.TrainingInput]:
    """Creates a streaming iterator over Ultra-FineWeb."""
    
    logging.info("Loading openbmb/Ultra-FineWeb (streaming)...")
    ds = datasets.load_dataset(
        "openbmb/Ultra-FineWeb",
        data_files="data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet",
        split="train",
        streaming=True,
    ).take(NUM_ROWS)
    
    pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else 0
    batch_tokens = []
    batch_masks = []
    total_tokens = 0
    
    for example in tqdm(ds, total=NUM_ROWS, desc="Processing dataset"):
        tokens = np.array(tokenizer.encode(example['content']), dtype=np.int32)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = np.pad(tokens, (0, max_length - len(tokens)), constant_values=pad_id)
        
        mask = (tokens != pad_id).astype(np.int32)
        total_tokens += np.sum(mask)
        batch_tokens.append(tokens)
        batch_masks.append(mask)
        
        if len(batch_tokens) == batch_size:
            yield peft_trainer.TrainingInput(
                input_tokens=np.array(batch_tokens),
                input_mask=np.array(batch_masks),
            )
            batch_tokens = []
            batch_masks = []
    
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
model_config.remat_config = gemma_lib.RematConfig.MLP_ONLY

print("Loading model from safetensors...")
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(
        file_dir=local_model_path,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
    )

nnx.display(model)

print("Loading tokenizer...")
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)
print(f"Tokenizer loaded from: {GEMMA_TOKENIZER_PATH}")

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    decay_steps=NUM_ROWS // GLOBAL_BATCH_SIZE,  # Total training steps
)
optimizer = optax.adamw(learning_rate=schedule)

# Only save checkpoint at end and keep max 1 (Kaggle has limited disk space ~20GB)
checkpointing_options = ocp.CheckpointManagerOptions(
    save_interval_steps=NUM_ROWS // GLOBAL_BATCH_SIZE,  # Only save at the end
    max_to_keep=1,
)

training_config = peft_trainer.TrainingConfig(
    eval_every_n_steps=100,
    max_steps=None,
    checkpoint_root_directory="/kaggle/working/checkpoints",  # Kaggle writable dir
    use_weighted_gradient_accumulation=True,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    checkpointing_options=checkpointing_options,
    metrics_logging_options=metrics_logger.MetricsLoggerOptions(
        log_dir="/kaggle/working/tensorboard",  # Kaggle writable dir
        flush_every_n_steps=10,
    ),
)

trainer = peft_trainer.PeftTrainer(model, optimizer, training_config)
trainer = trainer.with_cce_loss()

def gen_model_input_fn(x: peft_trainer.TrainingInput):
    pad_mask = x.input_tokens != 0
    positions = utils.build_positions_from_mask(pad_mask)
    attention_mask = utils.make_causal_attn_mask(pad_mask)
    return {
        'input_tokens': x.input_tokens,
        'input_mask': x.input_mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)

train_ds = create_train_dataset(tokenizer, GLOBAL_BATCH_SIZE, MAX_TARGET_LENGTH)

print("Starting training...")
with mesh:
    trainer.train(train_ds, None)

print("Training complete!")

CHECKPOINT_DIR = "/kaggle/working/checkpoints"
HF_REPO_ID = "G-reen/gemma-2-2b-ultrafine"

# Push checkpoint to Hugging Face Hub
api = HfApi()
create_repo(HF_REPO_ID, private=False, exist_ok=True)
api.upload_folder(
    folder_path=CHECKPOINT_DIR,
    repo_id=HF_REPO_ID,
    commit_message="Upload Gemma-2-2B fine-tuned with tunix (Orbax checkpoint)",
)
print(f"Model pushed to: https://huggingface.co/{HF_REPO_ID}")

# To reload the model later:
# from tunix.sft.checkpoint_manager import CheckpointManager
# ckpt_manager = CheckpointManager(root_directory="path/to/downloaded/checkpoint")
# ckpt_manager.maybe_restore(model)