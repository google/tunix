#!/usr/bin/env bash
# Ground-truth check for the trainer->rollout weight conversion, on ONE v5p
# slice (4 chips) instead of the full 40-chip RL topology.
#
#   ./run_qwen35_validate_converter.sh start|logs|stop
#
# Why this exists
# ---------------
# The distributed run reports reward 0 because the rollout generates digit
# soup, and the Raiden checksums say the transport is faithful: every tensor's
# abs-sum matches between trainer and rollout, 633/633 tensors, same element
# count. Those checksums are abs-sums, so they are invariant under any
# permutation or reshape -- which is exactly the class of bug left standing.
# Comparing source to destination can never catch it: the source *is* the
# converted array, so both sides agree on a wrong arrangement.
#
# maxtext.integration.vllm.validate_converter is the tool for this. It loads
# the MaxText model from the real Orbax checkpoint, runs the same
# WeightConverter production uses (MaxTextVllmSampler.update_params), assigns
# into vLLM, and then *generates greedily*. Coherent text means the conversion
# is right and the blocker is elsewhere in the distributed path; digit soup
# reproduces the bug in a 4-chip, single-process harness with no Raiden, no
# orchestrator and no cross-mesh transfer to hide behind.
#
# rollout_tensor_parallelism=2 mirrors production (ROLLOUT_MESH_TP=2), because
# the MoE fuse is per-shard: _interleave_moe_weights lays wi out as
# [gate_shard0 | up_shard0 | gate_shard1 | up_shard1] so that each TP shard
# sees a plain [local_gate | local_up], which is what gmm_v2 splits at its
# midpoint (tokamax gmm_v2.py: rhs_up_ref = rhs[..., out_size_n:]). Get the
# shard count wrong and shard 0 receives all-gate and shard 1 all-up -- a
# permutation, hence checksum-invariant.

set -euo pipefail

export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/config.cloud-tpu-shared-capacity.europe-west4.bodaborg-v5p-nap}"

LAUNCHER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_GEN="${LAUNCHER_DIR}/../../distributed/deployment/yaml_generator.py"
YAML_DIR="${LAUNCHER_DIR}/../../distributed/deployment/yamls"
PYTHON="${PYTHON:-python3}"

# ${USER} is igorts_google_com on this host and underscores are illegal in k8s
# object names, so derive the short form rather than interpolating ${USER}.
RUN_USER="${RUN_USER:-$(echo "${USER}" | cut -d_ -f1)}"
JOB_NAME="${JOB_NAME:-${RUN_USER}-vconv}"

export K8S_NAMESPACE="trellis"
export KUEUE_QUEUE_NAME="default"
export KUEUE_PRIORITY_CLASS="${PRIORITY_CLASS:-medium}"
TPU_SLICE="${TPU_SLICE:-tpuv5p:2x2x1}"
IMAGE="${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/igorts_google_com-runner:qwen35-repro-v12}"

MAXTEXT_CKPT="${MAXTEXT_CKPT:-gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/scanned/2026-06-11-10-27/0/items}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
# Two separate knobs, deliberately. `prefuse_moe_weights` means two different
# things on the two ends and they are NOT supposed to agree:
#
#   trainer (attention != vllm_rpa): moe.py:3692 reads `wi` as a *global*
#     concat -- w0 = wi[..., :n], w1 = wi[..., n:]. _fuse_moe_weights in
#     model_creation_utils.py builds exactly that at checkpoint load, with
#     n_shards taken from the trainer's own sharding of wi's last axis, which
#     under FSDP is unsharded => n_shards=1.
#   rollout (attention == vllm_rpa): `wi` goes whole into fused_moe_func ->
#     gmm_v2, which splits the *local* shard at its midpoint. At TP=2 that
#     wants [gate_s0|up_s0|gate_s1|up_s1].
#
# The converter only reaches its per-shard _interleave_moe_weights when the
# source tree still holds wi_0/wi_1 (convert_utils.py:149 skips otherwise). So
# the trainer must stay UNfused and let the converter do the fuse for the
# destination's shard count. Set both to true and the trainer's global-concat
# `wi` is copied verbatim: shard 0 receives all-gate, shard 1 all-up. That is a
# pure permutation, so every abs-sum checksum still matches.
TRAINER_PREFUSE="${TRAINER_PREFUSE:-${PREFUSE:-true}}"
ROLLOUT_PREFUSE="${ROLLOUT_PREFUSE:-true}"
PREFUSE="${TRAINER_PREFUSE}"
USE_CONVERTER="${USE_CONVERTER:-true}"
# debug_converter=true stops after the conversion checks (key coverage + weight
# stats) without generating. Leave it false: the generation is the whole point.
DEBUG_CONVERTER="${DEBUG_CONVERTER:-false}"
PROMPT="${PROMPT:-Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?}"

# One line, deliberately: the yaml generator substitutes this into a YAML block
# scalar, so embedded newlines would have to carry the block's indentation.
# `$` is also off-limits here -- the generator runs string.Template over the
# whole document.
CMD="python -m maxtext.integration.vllm.validate_converter /app/maxtext/src/maxtext/configs/post_train/rl.yml"
CMD+=" model_name=qwen3.5-35b-a3b tokenizer_type=huggingface tokenizer_path=Qwen/Qwen3.5-35B-A3B"
CMD+=" load_parameters_path=${MAXTEXT_CKPT} run_name=qwen35_converter_validation"
CMD+=" per_device_batch_size=1 max_prefill_predict_length=64 max_target_length=320 steps=1"
CMD+=" scan_layers=true skip_jax_distributed_system=true weight_dtype=bfloat16 dtype=bfloat16"
CMD+=" rollout_tensor_parallelism=${ROLLOUT_TP} hbm_utilization_vllm=0.85 async_scheduling=false"
CMD+=" use_weight_converter=${USE_CONVERTER} prefuse_moe_weights=${TRAINER_PREFUSE}"
CMD+=" debug_converter=${DEBUG_CONVERTER} use_chat_template=true"
# Single-quoted for the container's shell: unquoted, bash would strip the JSON
# double quotes and brace-expand the comma-separated inner object.
CMD+=" vllm_hf_overrides='{\"architectures\":[\"MaxTextForCausalLM\"]}'"
CMD+=" vllm_additional_config='{\"maxtext_config\":{\"model_name\":\"qwen3.5-35b-a3b\",\"model_call_mode\":\"inference\",\"prefuse_moe_weights\":${ROLLOUT_PREFUSE}}}'"
CMD+=" prompt='${PROMPT}'"

start() {
  "${PYTHON}" "${YAML_GEN}" \
    "${YAML_DIR}/jobset.tpu.yaml" \
    --jobset_name="${JOB_NAME}" \
    --namespace="${K8S_NAMESPACE}" \
    --queue_name="${KUEUE_QUEUE_NAME}" \
    --priority_class="${KUEUE_PRIORITY_CLASS}" \
    --tpu_slice="${TPU_SLICE}" \
    --worker_container_image="${IMAGE}" \
    --worker_container_port=20001 \
    --max_restarts=0 \
    --worker_startup_command="${CMD}" \
    | kubectl apply -f -
  echo "Launched ${JOB_NAME} (${TPU_SLICE}, rollout_tp=${ROLLOUT_TP}, trainer_prefuse=${TRAINER_PREFUSE}, rollout_prefuse=${ROLLOUT_PREFUSE})"
}

case "${1:-start}" in
  start) start ;;
  logs)  kubectl logs -n "${K8S_NAMESPACE}" -l "jobset.sigs.k8s.io/jobset-name=${JOB_NAME}" --tail=-1 --all-containers --prefix ;;
  stop)  kubectl delete jobset -n "${K8S_NAMESPACE}" "${JOB_NAME}" --wait=false ;;
  yaml)  "${PYTHON}" "${YAML_GEN}" "${YAML_DIR}/jobset.tpu.yaml" --jobset_name="${JOB_NAME}" \
           --namespace="${K8S_NAMESPACE}" --queue_name="${KUEUE_QUEUE_NAME}" \
           --priority_class="${KUEUE_PRIORITY_CLASS}" --tpu_slice="${TPU_SLICE}" \
           --worker_container_image="${IMAGE}" --worker_container_port=20001 \
           --max_restarts=0 --worker_startup_command="${CMD}" ;;
  *) echo "usage: $0 start|logs|stop|yaml" >&2; exit 1 ;;
esac
