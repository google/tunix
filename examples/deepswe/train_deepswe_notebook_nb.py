#!/usr/bin/env python
# coding: utf-8

# ## Pre steps to setup the CPU node pool and get k8s credential
# 1. Create a CPU node pool in GKE (update the env var based on your setup)
#
# ```
# export PROJECT_ID=cloud-tpu-multipod-dev
# export CLUSTER_NAME=mlperf-v5p
# export ZONE=europe-west4
# export CPU_POOL_NAME="tsbao-cpu-pool"
# export MACHINE_TYPE="n2-standard-8"
# export NUM_NODES=1
#
# gcloud container node-pools create ${CPU_POOL_NAME}   --cluster=${CLUSTER_NAME}   --zone=${ZONE}   --project=${PROJECT_ID}    --machine-type=${MACHINE_TYPE}   --num-nodes=${NUM_NODES}   --enable-autoscaling --min-nodes=1 --max-nodes=5  --node-labels="cloud.google.com/gke-nodepool=${CPU_POOL_NAME}"
# ```
#
# 2. Create k8s credential (this will add credential to your local ~/.kube/config)
#
# ```
#  gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE} --project ${PROJECT_ID}
# ```
#
# 3. Checkout R2E-Gym and patch this change (I ended up creating a fork due to no writer permission on the original repo): https://github.com/R2E-Gym/R2E-Gym/commit/046275291d34773657dbe170c96266b9736c938f

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:


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


dataset = load_dataset(
    "R2E-Gym/R2E-Gym-V1",
    split="train",
    cache_dir=DATASET_CACHE,
    num_proc=32,
    trust_remote_code=True,
)

entries = []
unique_images = set()
for i, entry in enumerate(dataset):
  if "docker_image" in entry:
    unique_images.add(entry["docker_image"])
    entries.append(entry)
  if i >= TASKS_TO_PROCESS - 1:
    break
unique_images = list(unique_images)
print(f"Found {len(unique_images)} unique Docker images to download")
IDS = [f"task-{i}" for i in range(len(entries))]


# In[5]:


import os

# os.getenv("KUBECONFIG", "~/.kube/config")
# os.getenv("NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool")
# os.getenv("NODE_SELECTOR_VAL", "lance-cpu-pool") # NB: change based on your node pool name
os.environ["KUBECONFIG"] = "~/.kube/config"
os.environ["NODE_SELECTOR_KEY"] = "cloud.google.com/gke-nodepool"
os.environ["NODE_SELECTOR_VAL"] = "lance-cpu-pool"

from kubernetes import client
from kubernetes import config

config.load_kube_config()
k8s_client = client.CoreV1Api()
print(f"YY k8s_client status:")
k8s_client.list_namespace(timeout_seconds=5)


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

qwen_actor = params_lib.create_model_from_safe_tensors(
    MODEL_PATH, config, train_mesh, dtype=jnp.float32
)
# qwen_ref = params_lib.create_model_from_safe_tensors(MODEL_PATH, config, train_mesh, dtype=jnp.float32)
sft_utils.show_hbm_usage()


# In[8]:


from tunix.generate import sampler

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


# In[9]:


# from tunix.generate.vllm_sampler import VllmSampler, VllmConfig
# from tunix.generate import mappings

# mapping_config = mappings.MappingConfig.build(
#     mapping_obj=None,
#     model=qwen_actor,
#     backend="vllm_jax",
# )

# vllm_config = VllmConfig(
#     model_path=MODEL_PATH,
#     max_model_len=8192,
#     mesh=train_mesh,
#     hbm_utilization_target=0.5,
#     init_with_random_weights=True,
#     tpu_backend_type="jax",
#     mapping_config=mapping_config
# )
# vllm_sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)


# In[10]:


# from tunix.generate import sglang_jax_sampler
# from tunix.generate import mappings

# MAX_PROMPT_LENGTH = 8192
# MAX_GENERATION_STEPS = 1024

# mapping_config = mappings.MappingConfig.build(
#     mapping_obj=None,
#     model=qwen_actor,
#     backend="sglang_jax",
# )
# sampler_sglang = sglang_jax_sampler.SglangJaxSampler(
#     tokenizer=tokenizer,
#     config=sglang_jax_sampler.SglangJaxConfig(
#         mesh=train_mesh,
#         context_length=MAX_PROMPT_LENGTH
#         + MAX_GENERATION_STEPS
#         + 100,
#         model_version=MODEL_VERSION,
#         mem_fraction_static=0.4,
#         init_with_random_weights=False,
#         disable_radix_cache=True,
#         enable_deterministic_sampling=False,
#         mapping_config=mapping_config,
#         precompile_token_paddings=[8192],
#         precompile_bs_paddings=[1],
#         max_running_requests=1,
#     ),
# )

# res = sampler_sglang(["which is bigger 9 or 11?"], max_generation_steps=MAX_GENERATION_STEPS)
# res.text[0]


# In[11]:


# from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser

# msgs = [
#   {'role': 'system',
#   'content': 'You are a programming agent who is provided a github issue and repository bash environment and is tasked to solve certain tasks (e.g., file localization, testcase generation, code repair and editing etc) to resolve the issue.\n\nWe have access to the following functions:\n\n–– BEGIN FUNCTION #1: file_editor ––\nDescription:\nCustom editing tool for viewing, creating and editing files\n  •\tState is persistent across command calls and discussions with the user\n  •\tIf path is a file, view displays the result of applying cat -n. If path is a directory, view lists non-hidden files and directories up to 2 levels deep\n  •\tThe create command cannot be used if the specified path already exists as a file\n  •\tIf a command generates a long output, it will be truncated and marked with <response clipped>\n  •\tThe undo_edit command will revert the last edit made to the file at path\n\nNotes for using the str_replace command:\n  •\tThe old_str parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!\n  •\tIf the old_str parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in old_str to make it unique\n  •\tThe new_str parameter should contain the edited lines that should replace the old_str\n\nParameters:\n  1.\tcommand (string, required)\nAllowed values: [view, create, str_replace, insert, undo_edit]\nThe command to run.\n  2.\tpath (string, required)\nAbsolute path to file or directory, e.g. /testbed/file.py or /testbed.\n  3.\tfile_text (string, optional)\nRequired for the create command. Contains the content of the file to be created.\n  4.\told_str (string, optional)\nRequired for the str_replace command. The exact string in path to replace.\n  5.\tnew_str (string, optional)\n  •\tOptional for the str_replace command to specify the replacement string.\n  •\tRequired for the insert command to specify the string to insert.\n  6.\tinsert_line (integer, optional)\nRequired for the insert command. The new_str will be inserted after the line number specified here.\n  7.\tview_range (array, optional)\n  •\tOptional for the view command (when path is a file).\n  •\tIf provided, specifies the line range to view, e.g. [11, 12] shows lines 11 and 12.\n  •\t[start_line, -1] will show all lines from start_line to the end of file.\n  8.\tconcise (boolean, optional)\n  •\tOptional for the view command.\n  •\tDefaults to True; displays a concise skeletal view of the file. If set to False, displays the full content in the specified view_range.\n\n–– END FUNCTION #1 ––\n\n–– BEGIN FUNCTION #2: execute_bash ––\nDescription:\nExecute a bash command in the terminal.\n\nBehavior notes:\n  •\tIf a command may run indefinitely (long-running), consider running it in the background and redirecting output, e.g. python3 app.py > server.log 2>&1 &.\n  •\tIf the bash command returns exit code -1, it means the process is still running. The assistant may:\n  •\tCall this function again with command as an empty string ("") to retrieve additional logs.\n  •\tSend more input to STDIN of the running process by calling this function again with command set to the text input.\n  •\tSend command="ctrl+c" to interrupt the currently running process.\n  •\tIf the command times out, it will be interrupted (SIGINT). The assistant may then retry or do further steps if needed.\n\nParameters:\n  1.\tcmd (string, required)\nThe bash command (and optional arguments) to execute.\n  •\tCan be empty ("") to retrieve more logs if the process is still running.\n  •\tCan be "ctrl+c" to interrupt the running process.\n\n–– END FUNCTION #2 ––\n\n–– BEGIN FUNCTION #3: search ––\nDescription:\nSearch for a term in a directory or a single file.\n  •\tIf path is a directory (or unspecified, default is .), it recursively searches all non-hidden files and directories for the search term.\n  •\tIf path points to a file, it runs a grep -n in that file to show line numbers matching the search term.\n  •\tIf more than 100 files match in a directory search, results are truncated and the tool will inform you to narrow your search.\n  •\tIf no matches are found, it will inform you as well.\n\nParameters:\n  1.\tsearch_term (string, required)\nThe term or string to search for in files.\n  2.\tpath (string, optional)\nThe file or directory to search in. Defaults to . if not specified.\n\n–– END FUNCTION #3 ––\n\n–– BEGIN FUNCTION #4: finish ––\nDescription:\nFinish the interaction once the task is complete or if no further progress can be made.\n\nBehavior notes:\n  •\tThe submit command finalizes your output.\n\nParameters:\n  1.\tcommand (string, required)\nCurrently allowed value: [submit]\n  2.\tresult (string, optional)\nThe result text or final message to submit. Defaults to an empty string if not provided.\n\n–– END FUNCTION #4 ––\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<function=example_function_name>\n<parameter=example_parameter_1>value_1</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format, start with <function= and end with </function>\n- Required parameters MUST be specified\n- Only call one function at a time\n- VERY IMPORTANT: Each response must include both reasoning (as natural text) and function call (in above format) to solve the task.\n'},
#  {'role': 'user',
#   'content': "Consider the following github issue:\n<github_issue>\n\n**Title:** Context migration fails to remove incompatible contexts, causing initialization errors\n\n**Description:**\nWhen initializing the `ContextHandler` with a mix of compatible and incompatible contexts, the migration process does not remove the incompatible contexts as expected. Instead, it raises an `IncompatibleContext` error, preventing successful initialization.\n\n**Example Code:**\n```python\nhandler = ContextHandler()\nhandler.bind(SimpleWidget)\n\nwidget = SimpleWidget()\ncontexts = [Context(foo=i) for i in (13, 13, 0, 1, 13, 2, 13)]\n\ndef migrate_context(context, _):\n    if context.foo == 13:\n        raise IncompatibleContext()\n\nhandler.initialize(widget, dict(context_settings=contexts))\n# Expected: Incompatible contexts with foo=13 should be removed\n# Actual: IncompatibleContext error is raised, and contexts are not removed\n```\n\n**Expected Behavior:**\nDuring initialization, contexts that are incompatible (e.g., those that cause `IncompatibleContext` to be raised) should be automatically removed, allowing the `ContextHandler` to proceed with only the compatible contexts.\n\n**Actual Behavior:**\nThe `ContextHandler` does not remove incompatible contexts, resulting in an `IncompatibleContext` error being raised and preventing successful initialization.\n\n\n</github_issue>\n\nCan you help me implement the necessary changes to the repository to fix the <github_issue>?\nI've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!\nYour task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.\n\nIMPORTANT TIP:\nFollow these steps to resolve the issue:\n1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.\n2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error\n3. Edit the sourcecode of the repo to resolve the issue\n4. Rerun your reproduce script and confirm that the error is fixed!\n5. Think about edgecases and make sure your fix handles them as well\n6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required\n7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files. \n"},
# ]
# chat_parser = QwenChatTemplateParser(tokenizer)
# s = chat_parser.parse(msgs)

# print(s)

# res = sampler_sglang(s, max_generation_steps=MAX_GENERATION_STEPS)
# print(f"sglang res: {res.text[0]}")

# vanilla_res = sampler(s, max_generation_steps=MAX_GENERATION_STEPS)
# print(f"vanilla res: {vanilla_res.text[0]}")


# In[12]:


from swe_agent import SWEAgent
from swe_env import SWEEnv
from tunix.rl.agentic.parser.chat_template_parser.parser import QwenChatTemplateParser
from tunix.rl.agentic.rewards.reward_types import RewardOutput
from tunix.rl.agentic.trajectory import trajectory_collect_engine

chat_parser = QwenChatTemplateParser(tokenizer)

# def model_call(chat_lists, rl_cluster):
#     result = rl_cluster.generate(
#         prompts=chat_lists,
#         apply_chat_template=True,
#         mode=rl_cluster_lib.Mode.TRAIN,
#     )
#     return result.text[0]


def model_call(chat_completions, _):
  p = chat_parser.parse(chat_completions)
  out = sampler(p, max_generation_steps=512, echo=False)
  return out.text[0]


MAX_STEPS = 10
agent = SWEAgent()
env = SWEEnv(entry=entries[0], max_steps=MAX_STEPS)

print(chat_parser.parse(agent.chat_completions))

engine = trajectory_collect_engine.TrajectoryCollectEngine(
    agent=agent,
    env=env,
    model_call=model_call,
    final_reward_fn=lambda x, y: RewardOutput(reward=0, metadata={}),
    max_steps=MAX_STEPS,
    gamma=0.9,
    timeout=120,
)


# res = await engine.collect(mode="Trajectory")


# In[13]:


async def _collect_trajectory():
  return await engine.collect(mode="Trajectory")


res = asyncio.run(_collect_trajectory())


# In[14]:


# print(env.total_steps)

STEP = 9
print(f"Step {STEP} ###################")
print(f"Observation ###################")
print(res.steps[STEP].observation)
print(f"Model Response ###################")
print(res.steps[STEP].model_response)

agent._messages


#

# In[ ]:


# # ====== Data ======
# TRAIN_FRACTION = 1.0

# # ====== Reproducibility ======
# SEED = 42

# # ====== LoRA ======
# RANK = 64
# ALPHA = 64.0
# TRAIN_WITH_LORA = False

# # ====== Sharding ======
# MESH = [(2, 4), ("fsdp", "tp")]

# # ====== GRPO ======
# # === Generation during GRPO training ===
# MAX_PROMPT_LENGTH = 2048
# TOTAL_GENERATION_STEPS = 512
# # Important to keep a high-ish temperature for varied, diverse responses during
# # training.
# TEMPERATURE = 0.6
# TOP_P = 0.95
# TOP_K = 50
# # The number of times the policy generates multiple responses for a given prompt
# # within a single training step. This corresponds to `G` in Algorithm 1 in the
# # paper. The "group" in GRPO comes from here.
# NUM_GENERATIONS = 2

# # === other GRPO configs ===
# # The number of iterations per batch (𝜇 in GRPO algo 1).
# NUM_ITERATIONS = 1
# # The coefficient for the KL divergence penalty (𝛽) in the GRPO loss function.
# # Important to keep a high enough value for this, otherwise, the KL divergence
# # can increase unchecked.
# BETA = 0.001
# # Epsilon value for clipping (𝜀 in GRPO loss in paper). Similar to PPO, for
# # stable updates.
# EPSILON = 0.2

# # ====== Training ======
# BATCH_SIZE = 16
# MINI_BATCH_SIZE = 16
# # ROLLOUT_MICRO_BATCH_SIZE = 8
# # LOGPS_MICRO_BATCH_SIZE = 8
# NUM_BATCHES = 100
# # Keep `NUM_TEST_BATCHES` low so that evaluation runs quickly. It can be
# # increased to a max. of 330 (if batch size is 4).
# NUM_TEST_BATCHES = 50

# EVAL_EVERY_N_STEPS = 1000  # this doesn't matter if `TRAIN_FRACTION = 1.0`.
# NUM_EPOCHS = 100 # can potentially train for more epochs

# # Number of training steps.
# MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# # === AdamW, warmup, cosine scheduler ===
# LEARNING_RATE = 1e-6
# B1 = 0.9  # Adam beta1
# B2 = 0.99  # Adam beta2
# WEIGHT_DECAY = 0.1
# # == Cosine decay with warmup scheduler ==
# # Linearly increase learning rate from 0. to 5e-6 in the first 10% training
# # steps, and then gradually decrease the learning rate to 0 using cosine
# # scheduler.
# WARMUP_STEPS = int(0.1 * MAX_STEPS)
# # == Grad clipping ==
# # Grad clipping to prevent large gradients. Found this
# # important to keep KL divergence in check.
# MAX_GRAD_NORM = 0.1

# # ====== Checkpoint saving ======
# SAVE_INTERVAL_STEPS = 500
# MAX_TO_KEEP = 4
# DO_MEM_PROFILING = False

# # ====== Inference ======
# GENERATION_CONFIGS = {
#     # greedy search
#     "greedy": {"temperature": 1e-4, "top_k": 1, "top_p": 1.0},
#     # some randomness
#     "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
#     # liberal
#     "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
# }
# # ====== Rollout ======
# ROLLOUT_ENGINE = "sglang_jax" # one of "vanilla", "vllm" or "sglang_jax"

# CKPT_DIR = os.path.join("/tmp/cp", "deepscaler_ckpt/01")


# In[ ]:


# from tunix.rl import rl_cluster as rl_cluster_lib
# import optax
# from tunix.sft import metrics_logger
# from orbax import checkpoint as ocp
# from tunix.rl.rollout import base_rollout

# checkpointing_options = ocp.CheckpointManagerOptions(
#     save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP
# )
# metrics_logging_options = metrics_logger.MetricsLoggerOptions(
#     log_dir="/tmp/tensorboard/grpo", flush_every_n_steps=20
# )

# optimizer = optax.adamw(
#     learning_rate=optax.schedules.warmup_cosine_decay_schedule(
#         init_value=0.0,
#         peak_value=LEARNING_RATE,
#         warmup_steps=WARMUP_STEPS,
#         decay_steps=MAX_STEPS,
#         end_value=0.0,
#     ),
#     b1=B1,
#     b2=B2,
#     weight_decay=WEIGHT_DECAY,
# )

# cluster_config = rl_cluster_lib.ClusterConfig(
#     role_to_mesh={
#         rl_cluster_lib.Role.ACTOR: train_mesh,
#         rl_cluster_lib.Role.REFERENCE: train_mesh,
#         rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
#     },
#     rollout_engine=ROLLOUT_ENGINE,
#     offload_to_cpu=False,
#     training_config=rl_cluster_lib.RLTrainingConfig(
#         actor_optimizer=optimizer,
#         eval_every_n_steps=EVAL_EVERY_N_STEPS,
#         max_steps=20,
#         mini_batch_size=MINI_BATCH_SIZE,
#         train_micro_batch_size = 1,  # larger than 1 will cause OOM on HBM
#         # metrics logging
#         metrics_logging_options=metrics_logging_options,
#         # checkpoint saving
#         checkpoint_root_directory=CKPT_DIR,
#         checkpointing_options=checkpointing_options,
#     ),
#     rollout_config=base_rollout.RolloutConfig(
#         max_tokens_to_generate=TOTAL_GENERATION_STEPS,
#         max_prompt_length=MAX_PROMPT_LENGTH,
#         kv_cache_size=MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256,
#         temperature=TEMPERATURE,
#         top_p=TOP_P,
#         top_k=TOP_K,
#         eos_tokens=[tokenizer.encode("<|im_end|>")[0]],
#         # sglang-jax specific configs
#         rollout_sglang_jax_model_version="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
#         rollout_sglang_jax_mem_fraction_static=0.2,
#         rollout_sglang_jax_init_with_random_weights=True,
#         rollout_sglang_jax_disable_radix_cache=True,
#         rollout_sglang_jax_enable_deterministic_sampling=False,
#         rollout_sglang_jax_precompile_bs_paddings=[1, 2],
#         rollout_sglang_jax_precompile_token_paddings=[2048, 4096, 8192],
#         rollout_sglang_jax_chunked_prefill_size=2048,
#         rollout_sglang_jax_page_size=64,
#     ),
# )

# rl_cluster = rl_cluster_lib.RLCluster(
#     actor=qwen2_actor,
#     reference=qwen2_ref,
#     tokenizer=tokenizer,
#     cluster_config=cluster_config,
# )


# # Random stuff for debugging

# In[ ]:


# from rllm.environments.swe.swe import R2EGYM_COMMAND_FILES
# import r2egym

# print(r2egym.__file__)
# from r2egym.agenthub.runtime.docker import DockerRuntime
# from r2egym.agenthub.utils.log import get_logger
# from r2egym.agenthub.environment.env import EnvArgs, RepoEnv

# env_args = EnvArgs(ds=entries[0])
# env = RepoEnv(env_args, backend="kubernetes")

# env.add_commands(cmd_files=R2EGYM_COMMAND_FILES)


# In[ ]:


# runtime = DockerRuntime(ds=entries[0], command=["/bin/bash", "-l"], logger=get_logger(), backend="kubernetes", id=IDS[0])
# runtime.get_task_instruction()


# In[ ]:


# runtime.run(code="ls -l")
# runtime.stop_container()


# In[ ]:


# DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# pod_name = "tsbao-test-cpu-pod"
# docker_image = entries[0]["docker_image"]
# command = "/bin/bash"

# env_vars = {"PATH": DOCKER_PATH}
# env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
# pod_body = {
#     "apiVersion": "v1",
#     "kind": "Pod",
#     "metadata": {"name": pod_name},
#     "spec": {
#         "restartPolicy": "Never",
#         "containers": [
#             {
#                 "name": pod_name,
#                 "image": docker_image,
#                 "command": ["/bin/sh", "-c"],
#                 "args": [command] if isinstance(command, str) else command,
#                 "stdin": True,
#                 "tty": True,
#                 "env": env_spec,
#                 "resources": {
#                     "requests": {"cpu": "1", "memory": "1Gi"},
#                 },
#             }
#         ],
#         "imagePullSecrets": [{"name": "dockerhub-pro"}],
#         "nodeSelector": {"cloud.google.com/gke-nodepool": "tsbao-cpu-pool"},
#         "tolerations": [
#             {
#                 "key": "node.kubernetes.io/disk-pressure",
#                 "operator": "Exists",
#                 "effect": "NoExecute",
#                 "tolerationSeconds": 10800
#             }
#         ],
#     },
# }

# pod = k8s_client.create_namespaced_pod(
#     namespace="default", body=pod_body, _request_timeout=60,
# )


# In[ ]:


# k8s_client.list_namespaced_pod(namespace="default")
# pod_name = "tsbao-test-pod"
# pod = k8s_client.read_namespaced_pod(name=pod_name, namespace="default")
# pod.status.phase


# In[ ]:


# from kubernetes.stream import stream

# full_command = ["/bin/sh", "-c", "ls -l"]
# resp = stream(
#     k8s_client.connect_get_namespaced_pod_exec,
#     name=pod_name,
#     namespace="default",
#     command=full_command,
#     stderr=True,
#     stdin=False,
#     stdout=True,
#     tty=False,  # Match docker exec_run settings
#     _preload_content=False,  # Important for streaming
# )
# resp


# In[ ]:


# combined_chunks = []
# stdout_chunks = []
# stderr_chunks = []
# while resp.is_open():
#     resp.update(timeout=1)  # wait for data
#     if resp.peek_stdout():
#         chunk = resp.read_stdout()
#         stdout_chunks.append(chunk)
#         combined_chunks.append(chunk)
#     if resp.peek_stderr():
#         chunk = resp.read_stderr()
#         stderr_chunks.append(chunk)
#         combined_chunks.append(chunk)
# resp.close()
# exit_code = resp.returncode
# combined_output = "".join(combined_chunks)


# In[ ]:


# from r2egym.agenthub.agent.commands import ParseCommandBash

# cmd_parser = ParseCommandBash()
# cmds = cmd_parser.parse_command_file("/scratch/git/R2E-Gym/src/r2egym/agenthub/tools/r2egym/file_editor.py")
# cmds[0]


# In[ ]:
