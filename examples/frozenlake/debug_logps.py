from pprint import pprint
import datasets as datasets_lib
import grain
import pandas as pd
import os
import fsspec
import numpy as np

import transformers
from tunix.generate import mappings

Dataset = datasets_lib.Dataset
AutoTokenizer = transformers.AutoTokenizer

from absl import logging as absl_logging

import logging
import sys

# ====== Logging Configuration ======
# 1. Force absl to use python logging
absl_logging.use_python_logging()

# 2. Configure the root logger
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)

# 3. Explicitly set levels for relevant loggers
logging.getLogger().setLevel(logging.INFO)
logging.getLogger("absl").setLevel(logging.INFO)

# 4. Set absl verbosity
absl_logging.set_verbosity(absl_logging.INFO)
absl_logging.set_stderrthreshold("info")

print("Logging configured at INFO level.")

try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  from etils import ecolab

  cm = ecolab.adhoc(
      source=ecolab.FROM_NOTEBOOK_OR_HEAD,
      reload="tunix",
      behavior="preferred",
      cell_autoreload=True,
  )

  file_open = gfile.Open

  NOTEBOOK_ENV = "g3"
except Exception:
  NOTEBOOK_ENV = "git"

  import contextlib
  cm = contextlib.nullcontext()

  file_open = fsspec.open
  
with cm:
  from tunix.models.qwen2 import model as qwen2_lib
  from tunix.models.qwen2 import params as qwen2_params_lib
  from tunix.models.gemma4 import model as gemma4_lib
  from tunix.models.gemma4 import params_safetensors as gemma4_params_lib
  from tunix.generate import sampler as sampler_lib
  from tunix.utils import math_utils
# %%
from typing import Any, Dict, Optional
import jax
from jax import numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from tqdm.auto import tqdm
import re

MODEL_PATH_PREFIX = "gs://tunix/models"
MODEL_MAPPING = {
    "Qwen/Qwen2.5-1.5B-Instruct": (
        qwen2_lib.ModelConfig.qwen2p5_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "qwen2_5/torch/1.5b-it"),
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepSeek-R1-Distill-Qwen-1.5B"),
    ),
    "agentica-org/DeepScaleR-1.5B-Preview": (
        qwen2_lib.ModelConfig.deepseek_r1_distill_qwen_1p5b(),
        os.path.join(MODEL_PATH_PREFIX, "DeepScaleR-1.5B-Preview"),
    ),
    "google/gemma-4-31B-it": (
      gemma4_lib.ModelConfig.gemma4_31b(),
      "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-31B-it/snapshots/439edf5652646a0d1bd8b46bfdc1d3645761a445",
    ),
    "google/gemma-4-E2B-it": (
      gemma4_lib.ModelConfig.gemma4_e2b(),
      "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf",
    ),
    "google/gemma-4-E4B-it": (
      gemma4_lib.ModelConfig.gemma4_e4b(),
      "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/83df0a889143b1dbfc61b591bbc639540fd9ce4c",
    ),
    "google/gemma-4-26B-A4B-it": (
      gemma4_lib.ModelConfig.gemma4_26b_a4b(),
      "/mnt/disks/linchai-data/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/7d4c97e54145f8ffd1a4dd1b4986a5015a517842",
    ),
    
}

import os
os.environ["TPU_LOG_DIR"] = "~/my_tpu_logs"
os.environ["SKIP_JAX_PRECOMPILE"] = "True"


# model_version = "google/gemma-4-26B-A4B-it"
model_version = "google/gemma-4-31B-it"
model_config, model_path = MODEL_MAPPING[model_version]

tokenizer = AutoTokenizer.from_pretrained(model_version)

