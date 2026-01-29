# Examples and Guides

``` {toctree}
:maxdepth: 1
:hidden:

_collections/examples/grpo_gemma
_collections/examples/logit_distillation
_collections/examples/qlora_gemma
_collections/examples/dpo_gemma
```

This section provides a high-level overview of the Colab notebooks, scripts, and
example directories.

All examples are located in this
[directory](https://github.com/google/tunix/tree/main/examples).

<table>
  <thead>
    <tr>
      <th align="center">Category</th>
      <th align="center">Name/Path</th>
      <th align="center">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" align="center" valign="middle" style="text-align: center; vertical-align: middle;"><b>Colab Notebook</b></td>
      <td><a href="https://github.com/google/tunix/tree/main/examples/qlora_gemma.ipynb"><code>qlora_gemma.ipynb</code></a></td>
      <td>End-to-end tutorial on fine-tuning (SFT) Gemma 270M model for English-French translation using parameter-efficient LoRA and QLoRA techniques.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/grpo_gemma.ipynb"><code>grpo_gemma.ipynb</code></a></td>
      <td>Reinforcement learning tutorial using Group Relative Policy Optimization (GRPO) to train the Gemma 3 1B IT model for math reasoning on the GSM8K benchmark.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/dpo_gemma.ipynb"><code>dpo_gemma.ipynb</code></a></td>
      <td>Preference tuning using Direct Preference Optimization (DPO) to tune the Gemma 3 1B-IT model on the GSM8K dataset.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/logit_distillation.ipynb"><code>logit_distillation.ipynb</code></a></td>
      <td>Demonstrates knowledge distillation from a Gemma 7B-IT teacher to a Gemma 2B-IT student for translation task.</td>
    </tr>
    <tr>
      <td rowspan="6" align="center" valign="middle" style="text-align: center; vertical-align: middle;"><b>Script</b></td>
      <td><a href="https://github.com/google/tunix/tree/main/examples/rl/grpo/gsm8k/"><code>rl/grpo/gsm8k/</code></a></td>
      <td>Bash scripts for fine-tuning different models and presets (Gemma, Llama, etc.) on the GSM8K mathematical reasoning task using GRPO.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/rl/grpo/gsm8k/verl_compatible/"><code>rl/grpo/gsm8k/verl_compatible/</code></a></td>
      <td>Bash scripts for GRPO-training on the GSM8K dataset to train with a verl-compatible setup.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/deepscaler/"><code>deepscaler/</code></a></td>
      <td>Scripts and notebooks for reproducing the <a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Deepscaler experiment</a> (<code>train_deepscaler_nb.py</code>) and math evaluation.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/sft/mtnt/"><code>sft/mtnt/</code></a></td>
      <td>Bash scripts for SFT examples on the MTNT translation task for Gemma, Llama, and Qwen models.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/model_load/"><code>model_load/</code></a></td>
      <td>Examples for loading Gemma2 and Gemma3 models from safetensors format.</td>
    </tr>
    <tr>
      <td><a href="https://github.com/google/tunix/tree/main/examples/agentic/"><code>agentic/</code></a></td>
      <td>Examples and scripts for agentic workflows, with async rollout.</td>
    </tr>
  </tbody>
</table>

## GCE VM Setup for Fine-Tuning

### 1. Create TPU VM

Create a v5litepod-8 TPU VM in GCE:

*   SW version: `v2-alpha-tpuv5-lite`
*   Name: `v5-8`

Reference:
[TPU Runtime Versions](https://cloud.google.com/tpu/docs/runtimes?hl=en#training-v5p-v5e)

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

Reference:
[Run JAX on TPU](https://cloud.google.com/tpu/docs/run-calculation-jax)

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

Reference:
[Local Runtimes in Colab](https://research.google.com/colaboratory/local-runtimes.html)

### 4. Environment Variables

Example `.env` file:

```bash
HF_TOKEN=
KAGGLE_USERNAME=
KAGGLE_KEY=
WANDB_API_KEY=
```

## Loading Saved Safetensors Models

To load a saved safetensors model back into JAX (with a given `local_path`):

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

*   **IMPORTANT**: Use `%pip` not `!pip` in notebooks!
*   Python 3.12 is the recommended version
