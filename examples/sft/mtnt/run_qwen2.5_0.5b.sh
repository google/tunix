# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -x # Enable xtrace

batch_size=${batch_size:-8}
max_steps=${max_steps:-100}

# Mesh shape / axis names. mesh_shape MUST equal the GLOBAL device count across
# all hosts (e.g. on two v5e-4 hosts that is 8 devices, so mesh_shape="(4,2)" or
# "(8,1)"). The single-host default below ("(2,2)" = 4 devices) is unchanged.
mesh_shape=${mesh_shape:-"(2,2)"}
mesh_axis_names=${mesh_axis_names:-"('fsdp','tp')"}

echo "Using parameters:"
echo "  Batch Size: $batch_size"
echo "  Max Steps: $max_steps"
echo "  Mesh Shape: $mesh_shape"
echo "  Mesh Axis Names: $mesh_axis_names"

# The python invocation is identical for single-host and multi-host runs; under
# multi-controller JAX every host runs this exact command and
# jax.distributed.initialize() (in peft_main) stitches the hosts into one mesh.
read -r -d '' PYTHON_CMD <<EOF
python3 -m tunix.cli.peft_main \
  base_config.yaml \
  model_config.model_name="qwen2.5-0.5b" \
  model_config.model_id="Qwen/Qwen2.5-0.5B" \
  model_config.model_source="huggingface" \
  model_config.lora_config={} \
  model_config.mesh.shape="${mesh_shape}" \
  model_config.mesh.axis_names="${mesh_axis_names}" \
  model_config.rng_seed=0 \
  model_config.use_flash_attn=true \
  model_config.model_download_path="/tmp/models/qwen2.5-0.5b" \
  tokenizer_config.tokenizer_path="Qwen/Qwen2.5-0.5B" \
  tokenizer_config.tokenizer_type="huggingface" \
  dataset_name="mtnt/en-fr" \
  batch_size=${batch_size} \
  optimizer_config.opt_type="adamw" \
  optimizer_config.learning_rate=1e-5 \
  max_target_length=1024 \
  training_config.eval_every_n_steps=20 \
  training_config.max_steps=${max_steps} \
  training_config.metrics_logging_options.log_dir="/tmp/tensorboard/full" \
  training_config.metrics_logging_options.flush_every_n_steps=20 \
  $@
EOF

if [[ -n "${TPU_NAME}" ]]; then
  # Multi-host fan-out: run the SAME command on every worker of the TPU pod via
  # multi-controller JAX. ZONE is required to address the pod.
  if [[ -z "${ZONE}" ]]; then
    echo "ERROR: TPU_NAME is set but ZONE is not. Set ZONE to the TPU's zone." >&2
    exit 1
  fi
  REPO_DIR=${REPO_DIR:-$(pwd)}
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
    --zone="${ZONE}" \
    --worker=all \
    --command="cd ${REPO_DIR} && mesh_shape='${mesh_shape}' batch_size=${batch_size} max_steps=${max_steps} ${PYTHON_CMD}"
else
  # Single-host: run locally exactly as before.
  eval "${PYTHON_CMD}"
fi

