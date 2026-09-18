#!/bin/bash
# 35B dist stack with Raiden weight sync ON and TIS sequence masking ON.
#
# Topology, images and memory limits are verbatim from the arm-B run that
# completed 5 steps with 6 successful Raiden syncs ("registered 1 work unit(s)
# with 633 variables"). Nothing here is a guess; the only additions are the TIS
# knobs and the routed-expert capture switch.
#
# Usage:
#   ./run_raiden_tis.sh --dry_run      # render manifests, launch nothing
#   ./run_raiden_tis.sh                # launch
set -euo pipefail

export PROJECT=cloud-tpu-shared-capacity
export CLUSTER=bodaborg-v5p-nap
export LOCATION_NAME=europe-west4
export ZONE=${ZONE:-europe-west4}
export CPU_MACHINE=n2d-standard-64
export QUEUE_NAME=default

export TUNIX_IMAGE=${TUNIX_IMAGE:-gcr.io/cloud-tpu-multipod-dev/jiantan-raiden-r3:v10}

# --- Raiden -----------------------------------------------------------------
# The wheel's FFI ABI is pinned to these two images and to jax 0.11.0 /
# libtpu 0.0.44. The launcher refuses to start if these are the stock Pathways
# images, because the trainer enables RAIDEN_USE_FFI off WEIGHT_SYNC_MODE alone
# and a stock proxy yields a run that trains but never syncs.
export WEIGHT_SYNC_MODE=raiden
export PATHWAYS_SERVER_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_server:raiden_20260904
export PATHWAYS_PROXY_IMAGE=us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/datenglin/unsanitized_proxy_server:raiden_20260904
export USE_WEIGHT_CONVERTER=true
# Must match on both sides: under Raiden the two ends exchange raw weight
# buffers, so a mismatch is silent corruption rather than an error. The trainer
# entrypoint defaults this to false and the rollout to true, so both are pinned.
export PREFUSE_MOE_WEIGHTS=true
export ENABLE_PREFIX_CACHING=false
export RETURN_ROUTED_EXPERTS=${RETURN_ROUTED_EXPERTS:-true}
# Logs source/destination tensor checksums on both sides. Cheap, and the only
# direct evidence that the bytes that left the trainer are the bytes that landed.
export VERIFY_WEIGHTS=true

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY=${WANDB_ENTITY:-jiantan-google}
export WANDB_PROJECT=${WANDB_PROJECT:-tunix}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-raiden-r14b-mlperf-replay-on-capturefix-lr1e6}

# --- MLPerf Submission Optimizer & Schedule Hyperparameters -----------------
export LEARNING_RATE=${LEARNING_RATE:-1.0e-6}
export ADAM_B1=${ADAM_B1:-0.9}
export ADAM_B2=${ADAM_B2:-0.999}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
export MAX_GRAD_NORM=${MAX_GRAD_NORM:-0.125}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export EPSILON_HIGH=${EPSILON_HIGH:-0.28}
export OVERLONG_LOSS_MASKING=${OVERLONG_LOSS_MASKING:-false}
export SEQ_LOGPROB_ERROR_THRESHOLD=${SEQ_LOGPROB_ERROR_THRESHOLD:-2.0}
if [[ -z "${WANDB_API_KEY:-}" && -f "/usr/local/google/home/jiantan/workingspace/credentials/wandb.md" ]]; then
  WANDB_API_KEY="$(awk '/API key:/ {print $3}' /usr/local/google/home/jiantan/workingspace/credentials/wandb.md | tr -d '[:space:]')"
elif [[ -z "${WANDB_API_KEY:-}" && -f "/usr/local/google/home/jiantan/.config/wandb_key" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < /usr/local/google/home/jiantan/.config/wandb_key)"
fi
export WANDB_API_KEY

# --- Model and topology -----------------------------------------------------
export TRAINER_BACKEND=maxtext
export MAXTEXT_MODEL_NAME=qwen3.5-35b-a3b
export MODEL_NAME=Qwen3.5-35B-A3B
export MODEL_ID=Qwen/Qwen3.5-35B-A3B
export TOKENIZER_PATH=Qwen/Qwen3.5-35B-A3B
# Both engines now load TWO LAYOUTS OF ONE CHECKPOINT.
#
# Before: the trainer read hengtaoguo's Orbax conversion
# (gs://hengtaoguo-maxtext-logs/.../scanned/2026-06-11-10-27/0/items) while the
# sampler converted the HF safetensors on the fly. Two independent conversions
# into the two engines, and the measured consequence was KL(sampler||trainer)
# ~= 0.07 nats/token at step 0 -- before any optimizer update -- against an
# offline floor of 0.003 when one checkpoint is loaded into both engines. The
# TIS band [0.999, 1.002] permits 0.002, so every sequence was rejected and
# 9 of 10 steps produced exactly zero gradient.
#
# These two paths are the scanned and unscanned layouts of the same golden
# checkpoint, produced by one conversion; see maxtext
# tests/end_to_end/tpu/qwen/moe/qwen3.5-35b-a3b/2_test_qwen3.5_35b_a3b.sh:36-37.
# The unscanned one is what the offline parity harness used to measure 0.003.
#
# The layouts are NOT interchangeable. The trainer runs scan_layers=True and
# needs `scanned`; the sampler runs scan_layers=False and needs `unscanned`.
# Crossing them fails at load with "Checkpoint structure mismatch: 67 of 70
# model parameter paths were not found".
export MAXTEXT_CKPT=gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/scanned/0/items
export ROLLOUT_MAXTEXT_CKPT=gs://maxtext-model-checkpoints/qwen3.5-35b-a3b/unscanned/0/items

export TRAINER_JOBSET_YAML=jobset.pathways.yaml
export TRAINER_TPU_SLICE=tpuv5p:2x2x2
export TRAINER_MESH_FSDP=${TRAINER_MESH_FSDP:-2}
# Trainer slice is 8 devices: fsdp 2 x tp 2 x expert 2 = 8.
#
# NOT tp=4. The MLPerf 6.1 study's arm used ici_tensor_parallelism=4, but it
# paired that with base_num_kv_heads=4. This model has num_kv_heads=2, and
# attention heads are atomic under TP, so tp=4 dies at init with:
#   ValueError: num_kv_heads (2) ... must be divisible by 4, the combined size
#   of the mesh axes ['tensor'] that shard KV heads
# Raising base_num_kv_heads is a model-architecture override and would risk
# Raiden weight-sync parity against the sampler, which runs the same model.
#
# Safe to diverge here: the same study measured trainer sharding as having NO
# impact on oob_ratio ("Using EP for sampler reduces oob_ratio. Trainer
# sharding does not impact."). expert=2 is retained to match the arm.
export TRAINER_MESH_TP=${TRAINER_MESH_TP:-2}
export TRAINER_MESH_EXPERT=${TRAINER_MESH_EXPERT:-2}

export ROLLOUT_JOBSET_YAML=jobset.tpu.yaml
export ROLLOUT_TPU_SLICE=tpuv5p:2x2x1
# SAMPLER-side expert parallelism is the lever that actually moves oob_ratio:
# fp32 lm_head alone gives 68.75% per-seq oob, adding sampler EP gives 53.12%.
#
# Two constraints pin this config, and they nearly conflict:
#
# 1. EP is NOT free over the existing devices. tpu-inference computes
#      total_devices = prod(tensor, expert, sequence, data, attn_dp, ...)
#    and asserts it equals the device list it was handed
#    (sharding.py:199-202), so ep=2 multiplies the requirement.
#
# 2. The rollout slice CANNOT span hosts. The in-process vLLM worker resolves
#    device_indexes against jax.local_devices() (tpu_worker.py:383-390), which
#    only ever contains this host's 4 chips. A tpuv5p:2x2x2 slice is 2 hosts,
#    so global device id 4 is absent locally and init_device dies with a bare
#    `KeyError: 4`. 8 sampler chips would need a Ray multihost backend.
#
# 3. kv_tp_size = rollout tp * ep (tests/utils/maxtext_utils_test.py:143). The
#    vLLM rollout REPLICATES KV heads up to kv_tp_size when the model has fewer
#    (maxtext_utils.py:300-309). This model has 2. With tp=2,ep=2 the sampler
#    replicated to 4 while the trainer built 2, and Raiden weight sync died in
#    preflight:
#      'decoder.layers.11.attention.attention.key.kernel' global shape differs:
#      source (2048, 2, 256), destination (2048, 4, 256)
#    Setting tp=1 keeps kv_tp_size = 1*2 = 2 = native, so NO replication happens
#    and the trainer needs no base_num_kv_heads override. This is also the
#    study's reference shape: EP replaces TP for the MoE
#    (expert_parallelism: N, tensor_parallelism: 1).
#
# CURRENT ARM (2026-09-16): tp=2, ep=1 -- the KNOWN-GOOD BASELINE, rerun with
# the top_k fix as the only moved variable.
#
# Why we are back here. Two sampler-EP arms have now failed:
#   tp=1, ep=2  -> ran, but regressed 11x (logp_diff_mean 0.1235 -> 1.3942,
#                  seq_geomean 0.9390 -> 0.2812, ppo_kl 0.0662 -> 1.1072).
#                  TWO variables moved (tp 2->1 and ep 1->2), so unattributable.
#   tp=1, ep=1  -> OOM at init: 264.17GiB requested vs 191.46GiB cap. With
#                  tp=1 and ep=1 NOTHING shards the model, so dp=4 replicates
#                  35B four times. The isolator cannot be built this way.
# The only clean single-variable isolator against this baseline is tp=2, ep=2,
# and that needs kv_tp_size = tp*ep threaded into the trainer (or attention DP
# enabled) before Raiden weight sync will pass. That is the NEXT change, not
# this one.
#
# What this arm actually tests. The sampler was silently drawing from a top-20
# truncated distribution the entire time: generation_config.json ships
# top_k=20, and vllm_sampler only assigned top_k when the caller passed a
# non-None value -- which the CLI never does, because --top_k=-1 is mapped to
# None at run_gsm8k_dist_grpo.py:446. So every oob number measured so far was
# taken against a behaviour policy the trainer was not scoring. Truncation
# renormalises mass onto the surviving 20 tokens, which lifts the sampler
# logprob, which drives trainer/sampler below 1 -- and seq_geomean has been
# below 1 in every single run. vllm_sampler.py now sets top_p/top_k/min_p
# unconditionally and is injected into the rollout pod.
#
# Read logp_diff_mean against the 0.1235 baseline. Confirm the fix landed by
# grepping the rollout log for "vLLM sampling truncation:"; it must report
# top_k=-1.
#
# Mesh: 4 chips, one host. fsdp 2 x tp 2 = 4 = jax.device_count(). ep=1 so
# run_rollout_node passes dp=mesh_fsdp directly (no -1 inference). kv_tp_size
# = tp*ep = 2 = the model's native num_kv_heads, so no KV replication and no
# trainer-side base_num_kv_heads override is needed.
export ROLLOUT_MESH_FSDP=2
export ROLLOUT_MESH_TP=2
export ROLLOUT_MESH_EP=1
export ROLLOUT_REPLICAS=1
export CHAT_PARSER=${CHAT_PARSER:-raw}

export PATHWAYS_PROXY_MEMORY_LIMIT=120G
export USER_CONTAINER_MEMORY=260G

# Disable Orbax checkpoint saving (0) so Orbax CheckpointManager never
# transfers 64.6 GiB of model+optimizer state to host memory over pathways-proxy.
export CHECKPOINT_SAVE_INTERVAL_STEPS=0

# --- Trainer precision ------------------------------------------------------
# fp32 ACTIVATIONS in the transformer body are a measured no-op for parity and
# cost roughly 2x trainer forward time, so they are off.
#
# Evidence (offline parity harness, arm B vs arm E, 32 sequences, teacher-forced
# over identical tokens so the pairing is genuine):
#   bf16 acts : seq is_oob 56.25%  |dlogp| med 0.0073  p99 0.256
#   fp32 acts : seq is_oob 53.12%  |dlogp| med 0.0068  p99 0.229
#   McNemar exact two-sided p = 0.7539 on 10 discordant pairs -> no effect.
#
# What DOES matter is fp32 at the LOGITS HEAD, on BOTH sides. That is
# logits_dot_in_fp32 + cast_logits_to_fp32 + float32_logits, set unconditionally
# in maxtext_utils.py for the trainer (:348-352) and the sampler (:84-88).
# Turning those off costs 4.5x on the median |dlogp|. Do not touch them.
export TRAINER_ACT_DTYPE=${TRAINER_ACT_DTYPE:-bfloat16}
export TRAINER_MATMUL_PRECISION=${TRAINER_MATMUL_PRECISION:-default}

# --- TIS sequence masking ---------------------------------------------------
# seq-mask-tis zeroes the per-token weights of any sequence whose geometric-mean
# sampler-to-trainer ratio leaves [min, max]. Dropped sequences stay in the loss
# DENOMINATOR, so a high tis/is_oob_ratio is a learning-rate cut, not just a
# smaller batch. Read tis/is_oob_ratio before trusting any convergence curve
# from this run.
export TIS_TYPE=${TIS_TYPE-seq-mask-tis}
export TIS_RATIO_MIN=${TIS_RATIO_MIN-0.999}
export TIS_RATIO_MAX=${TIS_RATIO_MAX-1.002}

# The ratio is only meaningful if old_per_token_logps are the SAMPLER's. With
# --no-use_rollout_logps the ratio is identically 1, every sequence sits at the
# band centre, and TIS silently masks nothing.
export USE_ROLLOUT_LOGPS=true

# --- Run shape --------------------------------------------------------------
export MAX_PROMPT_LENGTH=512
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
export MAX_STEPS=${MAX_STEPS:-50}
export BATCH_SIZE=4
export NUM_GENERATIONS=4
export TRAIN_MICRO_BATCH_SIZE=8
# Trajectories (4 groups x 4 generations = 16), NOT prompt groups, even though
# the trainer flag's help text says prompt groups. The image's trainer omits
# num_generations from its grad-accumulation formula, and this value cancels
# that omission: ceil(16/8) = 2 micro-steps per update. See k8s_launcher.sh:49.
export MINI_BATCH_SIZE=16

# reward_mean was 0.0000 on all 10 steps of every run to date. The cause is a
# wiring bug plus a default: k8s_launcher.sh exported REWARD_MODE but never
# passed --reward_mode, so argparse used its own default "env", and
# run_gsm8k_dist_grpo.py:437-441 builds `reward_fns = []` for anything other
# than "exact". The orchestrator therefore scored nothing and relied entirely on
# an environment reward that never arrived.
#
# "exact" recomputes the GSM8K reward in the orchestrator from the returned
# trajectory text. It is what the dense-GRPO bringup ran, and it got
# reward_mean ~0.5 with a 1.7B model.
export REWARD_MODE=exact

export MAXTEXT_OUTPUT_DIR=gs://jiantan-trellis-europe-west4/raiden-tis/$(date +%Y%m%d-%H%M%S)/maxtext
export DEBUG=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CMD="start"
if [[ $# -gt 0 && ( "$1" == "start" || "$1" == "stop" || "$1" == "orchestrator" || "$1" == "trainer" || "$1" == "rollout" ) ]]; then
  CMD="$1"
  shift
fi
bash "${SCRIPT_DIR}/k8s_launcher.sh" "$CMD" "$@"
