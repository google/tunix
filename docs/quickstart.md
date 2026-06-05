<!-- DO NOT REMOVE! Placeholder for TOC. -->

# Quick Start

This page contains several quickstart guides and is a great place to understand
how to get started with Tunix. It covers installation and provides several
hands-on examples across the board for SFT, RL, and Agentic RL training.
Additionally, it shows how to enable to multi-node training.

## Installation

Tunix is written in Python and **requires Python 3.11** or later. We recommend
installing Tunix in a Python virtual environment.

1.  Create a project specific environment.

    ```sh
    python3 -m venv .venv
    # Or simply `python -m venv .venv` depending on your system configuration.
    ```

2.  Activate the Environment

    ```sh
    source .venv/bin/activate
    ```

3.  Install Tunix dependency

    Make sure you have an updated pip version installed:

    ```sh
    pip install --upgrade pip
    ```

There are several ways to install Tunix. Please select one from below.

### Option A: From PyPI (**Recommended**)

You can install the latest stable release of Tunix from PyPI. Tunix relies on
JAX for computation, which must be installed with support for your specific
hardware (TPU, GPU, or CPU).

**TPU**

Tunix is optimized for execution on TPUs. If you have TPU hardware, you can
install Tunix and JAX with TPU support by specifying the `[prod]` extra:

```sh
pip install "google-tunix[prod]"
```

**GPU**

If you are using GPUs, first install Tunix, then install JAX with GPU (CUDA)
support. You may need to adjust the CUDA version based on your system setup.
Refer to the
[JAX installation guide](https://github.com/google/jax#installation) for more
details.

```sh
pip install google-tunix
# Install JAX with CUDA 13 support
pip install -U "jax[cuda13]"
```

**CPU**

To run Tunix in a CPU-only environment:

```sh
pip install google-tunix "jax[cpu]"
```

### Option B: From GitHub

You can install the latest development version directly from GitHub:

```sh
# For TPU
pip install "git+https://github.com/google/tunix#egg=google-tunix[prod]"

# For GPU/CPU
pip install git+https://github.com/google/tunix
# Then install JAX for GPU or CPU as described above.

```

### Option 3: From Source

If you plan to modify Tunix, you can perform an editable installation from a
local clone of the repository:

```sh
git clone https://github.com/google/tunix.git
cd tunix
pip install -e ".[dev]"
# Then install JAX for your hardware as described above.
```

For TPU development, you can use:

```sh
pip install -e ".[prod]"
```

### Optional Dependencies

For accelerated inference, Tunix supports integration with vLLM and SGLang-Jax.
These need to be installed manually.

**vLLM on TPU**

The TPU-inference supported version of `vllm` is not always available as a
single PyPI release, and installing the TPU build sometimes requires extra pip flags
so that `libtpu` wheels (hosted by the JAX project) can be resolved. You can
install the pinned vLLM + TPU requirements from this repository using one of
the raw requirement-file URLs below.

Install from remote:

```sh
pip install -r https://github.com/google/tunix/raw/main/requirements/requirements.txt
pip install -r https://github.com/google/tunix/raw/main/requirements/special_requirements.txt
```

Or (direct raw.githubusercontent URL):

```sh
pip install -r https://raw.githubusercontent.com/google/tunix/main/requirements/requirements.txt
pip install -r https://raw.githubusercontent.com/google/tunix/main/requirements/special_requirements.txt
```

If you prefer a single-line install that directly overrides `tpu-inference`, you can also run:

```sh
pip install vllm @git+https://github.com/vllm-project/vllm.git@<commit>
pip install --extra-index-url https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \\
            --find-links https://storage.googleapis.com/jax-releases/libtpu_releases.html \\
            --pre \\
            tpu-inference@git+https://github.com/vllm-project/tpu-inference.git@<commit>
```

Or install from source:
```sh
bash scripts/install_tunix_vllm_requirements.sh
```

**SGLang-Jax**

After installing Tunix, you can install SGLang-Jax from source:

```sh
git clone git@github.com:sgl-project/sglang-jax.git
cd sglang-jax/python
pip install -e .
```

**GCS File System**

If you need to access models or data stored in Google Cloud Storage (GCS), e.g.,
this is commonly used as the default option for Gemma3 models when using Tunix
CLI, you may need to install `gcsfs`:

```sh
pip install gcsfs
```

## Quick start: GRPO

To get started with the library, let's walk through an example of training (full
, LoRA and QLoRA fine-tuning) the Gemma 3 270M model on the English-to-French
translation dataset. We will use Tunix's `PeftTrainer` for this task.

Note: This example is meant to be a quick-start. For the complete example, refer
to
[this](https://github.com/google/tunix/blob/main/examples/qlora_gemma.ipynb)
notebook.

### Load the model

First up, let's load the model:

```python
from huggingface_hub import snapshot_download
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

# Define sharding mesh for the model (assuming 1 TPU).
MESH = [(1, 1), ("fsdp", "tp")]
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

# Load the model.
model_path = snapshot_download(
    repo_id=model_id, ignore_patterns=["*.pth"]
)
config = gemma_lib.ModelConfig.gemma3_270m()
with mesh:
    model = params_safetensors_lib.create_model_from_safe_tensors(
      model_path, config, mesh
    )
```

Note: we could have simply used Tunix's `AutoModel` class, but don't use it here
since Gemma 3 isn't supported for now. `AutoModel` is the preferred way of
loading models.

### Load and preprocess the dataset

Next, we load the English-French translation dataset. Note you can use your own
datasets too (PyGrain, Hugging Face dataset, TFDS, etc.).

```sh
gcloud storage cp gs://gemma-data/tokenizers/tokenizer_gemma3.model .
```

```python
from tunix.generate import tokenizer_adapter
from tunix.examples.data import translation_dataset as data_lib

tokenizer = tokenizer_adapter.Tokenizer("./tokenizer_gemma.model")
train_ds, val_ds = data_lib.create_datasets(
    'mtnt/en-fr',
    global_batch_size=64,
    max_target_length=256,
    num_train_epochs=3,
    tokenizer=tokenizer,
)
```

We need to process the inputs to make sure we are feeding the data to the model
in the right format.

```python
def input_fn(x):
    mask = x.input_tokens != tokenizer.pad_id()
    return {
        'input_tokens': x.input_tokens, 'input_mask': x.input_mask,
        'positions': utils.build_positions_from_mask(mask),
        'attention_mask': utils.make_causal_attn_mask(mask),
    }
```

### Train the model

#### Full fine-tuning

We can now train our model. We need to pass the `input_fn` defined above here:

```python
from tunix.sft import peft_trainer

trainer = peft_trainer.PeftTrainer(
    model=model,
    optimizer=optax.adamw(learning_rate=1e-4),
    mesh=mesh,
    model_input_fn=input_fn,
)

trainer.train(train_ds=train_ds, num_steps=100, eval_ds=val_ds, eval_steps=20)
```

#### LoRA/QLoRA fine-tuning

The above case handles the full SFT case where all model parameters are updated.
We can choose to use LoRA. In this case, we just need to use Qwix, like so:

```python
import qwix

lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=RANK,
    alpha=ALPHA,
    # for QLoRA, uncomment the lines below.
    # weight_qtype="nf4",
    # tile_size=128,
)

model_input = model.get_model_input()
lora_model = qwix.apply_lora_to_model(
    model, lora_provider, **model_input
)

with mesh:
  state = nnx.state(lora_model)
  pspecs = nnx.get_partition_spec(state)
  sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
  nnx.update(lora_model, sharded_state)
```

The rest of the flow remains the same.

### Evaluate the model

To evaluate the model, we can use the `Sampler` API to generate outputs.

```python
sampler = sampler_lib.Sampler(
    transformer=lora_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
    "Translate this into French:\nThis dish is delicious!\n",
]

out_data = sampler(
    input_strings=input_batch,
    max_generation_steps=10,  # number of generated tokens
)
```

### Trajectory Logging

During reinforcement learning (RL) training, it is often useful to analyze the
generated trajectories (prompts, responses, rewards, etc.). Tunix provides an
`AsyncTrajectoryLogger` to log this data asynchronously to CSV files without
blocking the training loop. It's enabled in agentic_grpo_learner by default, if
you provide a log directory in your cluster configuration training config.

```python
# In your cluster configuration setup
cluster_config.training_config.metrics_logging_options.log_dir = "./logs"
# GCS paths are also supported
```

When enabled, the learner will automatically log trajectories during the
training process. Users can then consume the logged data by loading the CSV
files into a pandas DataFrame or other query engine.

## Quick Start: Multi-Node Training
Tunix supports running on a multi-node setup using Pathways in GKE ([more details](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster)). This is a
transparent change that simply requires you to submit your job through Pathways
instead of running directly on a VM. To run Tunix in a multi-node Pathways
cluster basically requires 3 steps: 1. create a Pathways cluster, 2. Build a
docker image, 3. launch a Tunix job. The following sections cover each step in
further detail.

### 1. Create a Pathways cluster in GKE

#### Install xpx

We will use XPK to create a Pathways cluster in GKE.

```sh
pip install xpk
```

#### Install gcloud cli

For Debian or Ubuntu, install gcloud via apt. Make sure prerequisites are met:

```sh
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates gnupg curl
```

Import the Google Cloud public key:

```sh
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
```

Add the Google Cloud CLI distribution URI as a package source:

```sh
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
```

Update and install:

```sh
sudo apt-get update && sudo apt-get install google-cloud-cli
```

#### Create a Pathways cluster
Then we will create the Pathways cluster.

```sh
# install gcloud beta commands
gcloud components install beta

# create pathways cluster
export CLUSTER_NAME='your-cluster-name'
export ZONE='your-tpu-zones'
export TPU_TYPE='your-tpu-type' # e.g. v5p-16
export CLUSTER_CPU_MACHINE_TYPE=n2d-standard-32 # you can adjust this to use beefier CPU node
export PROJECT='your-gke-projec'

NETWORK_NAME=${CLUSTER_NAME}-mtu9k-wx
NETWORK_FW_NAME=${NETWORK_NAME}-fw-wx

export CLUSTER_ARGUMENTS="--network=${NETWORK_NAME} --subnetwork=${NETWORK_NAME}"

# run `gcloud auth application-default login` and
# `gcloud auth login --update-adc` if you encounter permission issue when creating the network.

# Check if this is the service account you want to use.
gcloud auth list

gcloud compute networks create ${NETWORK_NAME} \
    --mtu=8896 \
    --project=${PROJECT} \
    --subnet-mode=auto \
    --bgp-routing-mode=regional

gcloud compute firewall-rules create ${NETWORK_FW_NAME} \
    --network ${NETWORK_NAME} \
    --allow tcp,icmp,udp \
    --project=${PROJECT}

xpk cluster create-pathways \
    --cluster $CLUSTER_NAME \
    --cluster-cpu-machine-type=$CLUSTER_CPU_MACHINE_TYPE \
    --num-slices=1 \
    --tpu-type=$TPU_TYPE \
    --zone $ZONE \
    --project $PROJECT \
    --custom-cluster-arguments="${CLUSTER_ARGUMENTS}"
```

### 2. Build a Tunix Docker Image

Build local docker image. We will be using the `build_docker.sh` [script](https://github.com/google/tunix/blob/main/build_docker.sh).
 in the `tunix` directory.


```sh
# cleanup unused docker images and caches if disk is not enough
sudo docker system prune

bash ./build_docker.sh
# It will default to generate a local docker image
export LOCAL_IMAGE_NAME=tunix_base_image

# You can also optionally push to GKE's artifact registry for faster download in the future
```

### 3. Launch the job

Now you are ready to submit your Tunix workload. You will use `xpk` to do this, 
similar to the cmd below.

```sh
xpk workload create-pathways \
    --cluster=$CLUSTER_NAME \
    --workload=$WORKLOAD_NAME \
    --command="TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 JAX_PLATFORMS=proxy JAX_BACKEND_TARGET=grpc://127.0.0.1:29000 ENABLE_PATHWAYS_PERSISTENCE='1' source your-script-to-launch-job.sh" \
    --num-slices=1 \
    --tpu-type=$TPU_TYPE \
    --base-docker-image docker.io/library/tunix_base_image \
    --priority=medium
```


## Next Steps

Now that you've completed the quick start, you can explore other training
techniques and models. In particular, the following would be worth exploring:

-   [SFT and PEFT](https://github.com/google/tunix/blob/main/examples/qlora_gemma.ipynb)
-   [Agentic RL](https://github.com/google/tunix/blob/main/examples/agentic/gemma_grpo_demo_nb.py)

A complete list is given [here](examples.md).
