RUN_ID="wuhaotest-r$((RANDOM % 90000 + 10000))"
export USER="$RUN_ID"
export TUNIX_IMAGE=gcr.io/cloud-tpu-multipod-dev/wuhaotest/tunix-maxtext-rlvllm
export PROJECT=cloud-tpu-multipod-dev
export REGION=europe-west4
export CLUSTER=mlperf-v5p
export CPU_MACHINE=n2d-standard-128
export GCS_SCRATCH_LOCATION=gs://mohitkhatwani_multipods/pathways_scratch

RAIDEN_WEIGHT_SYNC_CHUNKS=1 \
MODEL_NAME=Qwen3-0.6B MODEL_ID=Qwen/Qwen3-0.6B MAXTEXT_MODEL_NAME=qwen3-0.6b \
TRAINER_BACKEND=maxtext MAXTEXT_CKPT= \
TRAINER_TPU_SLICE=tpuv5:2x2x1 TRAINER_MESH_FSDP=4 TRAIN_MICRO_BATCH_SIZE=8 \
ROLLOUT_TPU_SLICE=tpuv5:2x2x1 ROLLOUT_TENSOR_PARALLEL_SIZE=4 ROLLOUT_REPLICAS=1 \
SAMPLER=vllm WEIGHT_SYNC_BACKEND=raiden VERIFY_WEIGHTS=true MAX_STEPS=2 \
bash tunix/experimental/examples/math_gsm8k_dist/k8s_launcher.sh --command=start --image=$TUNIX_IMAGE
