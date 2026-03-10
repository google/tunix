import sys

# sys.path.insert(0, '/home/lancewang_google_com/github/rllm') # change these based on your local repo setup
# sys.path.insert(0, '/home/lancewang_google_com/github/pathways-utils')

sys.path.insert(
    0, "/usr/github/rllm"
)  # change these based on your local repo setup
sys.path.insert(0, "/usr/github/pathways-utils")

import asyncio
import logging
import sys

# Remove existing handlers to prevent duplicate logs or conflicts
for handler in logging.root.handlers[:]:
  logging.root.removeHandler(handler)

logging.basicConfig(
    stream=sys.stdout,  # Direct logs to standard output (notebook cell)
    level=logging.INFO,  # Set the minimum level to INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # Optional: customize the format
    datefmt="%Y-%m-%d %H:%M:%S",  # Optional: customize the date format
)


# In[3]:


import os

from datasets import load_dataset

DATASET_CACHE = os.getenv("DATASET_CACHE", "/scratch/dataset_cache")
TASKS_TO_PROCESS = 100

if os.getenv("JAX_PLATFORMS", None) == "proxy":
  import pathwaysutils

  pathwaysutils.initialize()


# In[4]:


# dataset = load_dataset(
#     "R2E-Gym/R2E-Gym-V1",
#     split="train",
#     cache_dir=DATASET_CACHE,
#     num_proc=32,
#     trust_remote_code=True,
# )

# entries = []
# unique_images = set()
# for i, entry in enumerate(dataset):
#   if "docker_image" in entry:
#     unique_images.add(entry["docker_image"])
#     entries.append(entry)
#   if i >= TASKS_TO_PROCESS - 1:
#     break
# unique_images = list(unique_images)
# print(f"Found {len(unique_images)} unique Docker images to download")
# IDS = [f"task-{i}" for i in range(len(entries))]


# # In[5]:


# import os

# # os.getenv("KUBECONFIG", "~/.kube/config")
# # os.getenv("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
# # os.getenv("NODE_SELECTOR_VAL", "lance-cpu-pool") # NB: change based on your node pool name
# os.environ["KUBECONFIG"] = "~/.kube/config"
# os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
# # os.environ["NODE_SELECTOR_VAL"] = "lance-cpu-pool"

# from kubernetes import client
# from kubernetes import config

# config.load_kube_config()
# k8s_client = client.CoreV1Api()
# print(f"YY k8s_client status:")
# k8s_client.list_namespace(timeout_seconds=5)


# In[6]:


# MODEL_PATH = "/scratch/models/DeepSeek-R1-Distill-Qwen-1.5B/"
MODEL_VERSION = os.getenv("MODEL_VERSION", "Qwen/Qwen3-4B-Instruct-2507")
MODEL_PATH = os.path.join("/scratch/models/", MODEL_VERSION)


from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from tunix.rl.agentic.parser.chat_template_parser import parser

if not os.path.isdir(MODEL_PATH) or not os.listdir(MODEL_PATH):
  os.makedirs(MODEL_PATH, exist_ok=True)
  snapshot_download(
      repo_id=MODEL_VERSION,
      local_dir=MODEL_PATH,
      local_dir_use_symlinks=False,
  )

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

chat_parser = parser.QwenChatTemplateParser(tokenizer)


# In[7]:


import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import numpy as np
from tunix.models.qwen3 import model as model_lib
from tunix.models.qwen3 import params as params_lib
from tunix.sft import utils as sft_utils

devices = jax.devices()
# split = int(len(devices) / 2)
# rollout_devices = np.array(devices[:split-2]).reshape(split-2, 1)
# train_devices = np.array(devices[split:]).reshape(split, 1)
# rollout_mesh = Mesh(rollout_devices, axis_names=('fsdp', 'tp'))
train_devices = np.array(devices).reshape(len(devices), 1)
train_mesh = Mesh(train_devices, axis_names=("fsdp", "tp"))

if MODEL_VERSION == "Qwen/Qwen3-4B-Instruct-2507":
  config = model_lib.ModelConfig.qwen3_4b_instruct_2507()
elif MODEL_VERSION == "Qwen/Qwen3-32B":
  config = model_lib.ModelConfig.qwen3_32b()

print(f"YY checkpoint loading begins", flush=True)
qwen_actor = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, train_mesh, dtype=jnp.float32
)
print(f"YY checkpoint loading is done", flush=True)
# qwen_ref = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, train_mesh, dtype=jnp.float32)
sft_utils.show_hbm_usage()


# In[8]:


from tunix.generate import sampler

print(f"YY sampler init begins!", flush=True)
sampler = sampler.Sampler(
    qwen_actor,
    tokenizer,
    sampler.CacheConfig(
        cache_size=16384,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    ),
)
print(f"YY sampler init done!", flush=True)
res = sampler(["which is bigger 9 or 11?"], max_generation_steps=8096)
res.text[0]
