# Tuning

Fine-tuning examples using Google Tunix.

## Notebooks

The following notebooks provide comprehensive examples of different fine-tuning techniques:

- **`qlora_gemma.ipynb`** - LoRA and QLoRA fine-tuning with Gemma models. Demonstrates parameter-efficient fine-tuning techniques using low-rank adaptation.
- **`grpo_gemma.ipynb`** - GRPO (Group Relative Policy Optimization) reinforcement learning. Shows how to fine-tune models using policy optimization for improved response generation.
- **`dpo_gemma.ipynb`** - DPO (Direct Preference Optimization). Demonstrates preference-based fine-tuning to align model outputs with desired behaviors.
- **`logit_distillation.ipynb`** - Knowledge distillation from larger models. Shows how to transfer knowledge from a teacher model to a student model.

## Subdirectories

### `deepscaler/`
Contains scripts for training and evaluating models with DeepScaler:
- `train_deepscaler_nb.py` - Training script for DeepScaler models
- `math_eval_nb.py` - Mathematical reasoning evaluation utilities

### `model_load/`
Examples for loading models from different formats:
- `from_safetensor_load/` - Contains notebooks for loading Gemma2 and Gemma3 models from safetensors format
  - `gemma2_model_load.ipynb`
  - `gemma3_model_load.ipynb`

### `rl/`
Reinforcement learning examples and hardware resource requirements:
- `grpo/gsm8k/` - GRPO implementation scripts for GSM8K mathematical reasoning tasks
  - Launch scripts for various models (Gemma 7b, Gemma2 2b, Llama3.1 8b, Llama3.2 1b)
- `README.md` - Detailed hardware resource requirements and configuration recommendations for RL training

### `sft/`
Supervised fine-tuning examples:
- `mtnt/` - MTNT translation task examples with launch scripts for multiple models
  - Launch scripts for Gemma 2b, Gemma2 2b, Gemma3 4b, Llama3.2 3b, Qwen2.5 0.5b
  - `README.md` - Hardware resource requirements for SFT training

## GCE VM Setup for Fine-Tuning

### 1. Create TPU VM

Create a v5litepod-8 TPU VM in GCE:
- SW version: `v2-alpha-tpuv5-lite`
- Name: `v5-8`

Reference: [TPU Runtime Versions](https://docs.cloud.google.com/tpu/docs/runtimes?hl=en&_gl=1*1tpeg3j*_ga*MTk1NzE5MjMyNy4xNzYwOTEwNjk3*_ga_WH2QY8WWF5*czE3NjIxNTU1OTEkbzE3JGcwJHQxNzYyMTU1NTkxJGo2MCRsMCRoMA..#training-v5p-v5e)

### 2. Configure VM

SSH into the VM using the supplied gcloud command, then run:

```bash
# Create .env file with required credentials
vim .env

# Download and install Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash ~/Anaconda3-2025.06-0-Linux-x86_64.sh  # always input "yes"/enter
source ~/.bashrc

# Create conda environment (Python 3.12 - MUST BE 12, NOT 11!)
conda create -n colab python=3.12 -y
conda activate colab

# Install dependencies
pip install 'ipykernel<7' jupyterlab
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade clu
```

Reference: [Run JAX on TPU](https://docs.cloud.google.com/tpu/docs/run-calculation-jax)

Exit the SSH session after setup is complete.

### 3. Connect from Local Machine

From your local machine, run the following to connect to Jupyter Lab:

```bash
gcloud compute tpus tpu-vm ssh v5-8 --zone=us-west1-c \
  -- -L 8080:localhost:8080 -L 6006:localhost:6006 \
  "source \$HOME/anaconda3/etc/profile.d/conda.sh && \
   conda activate colab && \
   jupyter lab \
     --ServerApp.allow_origin='https://colab.research.google.com' \
     --port=8080 \
     --no-browser \
     --ServerApp.port_retries=0 \
     --ServerApp.allow_credentials=True"
```

Reference: [Local Runtimes in Colab](https://research.google.com/colaboratory/local-runtimes.html)

### 4. Environment Variables

Example `.env` file:

```bash
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
```

## Loading Saved Safetensors Models

To load a saved safetensors model back into JAX (with a given local_path):

```python
import os
import jax
import jax.numpy as jnp
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib


local_path = '[PLACEHOLDER]'
MESH = [(1, 1), ("fsdp", "tp")]

mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))
with mesh:
  model = params_safetensors_lib.create_model_from_safe_tensors(
      os.path.abspath(local_path), (model_config), mesh, dtype=jnp.bfloat16
  )
```

## Notes

- **IMPORTANT**: Use `%pip` not `!pip` in notebooks!
- Python 3.12 is the recommended version
