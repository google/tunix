# Thin overlay on Yixuan's runner image for the Qwen3.5-35B-A3B GRPO run.
#
# Six changes, applied as patches rather than file copies. A file copy would overwrite
# whatever else the base image has done to these files; a patch fails the build instead.
# Nothing else in the image is modified: same tunix, maxtext, tpu-inference, vLLM, Raiden
# wheel and JAX pins as the base. The patch files themselves do not survive the layer.
#
#   maxtext-0001  MaxText PR 5219   stop a per-micro-batch recompile of the 35B graph
#   maxtext-0002  MaxText PR 5234   bound device->host staging during a checkpoint save
#   tunix-0001    tunix PR 2229     make SAMPLER=vllm work with Raiden weight sync
#   tunix-0002    tunix PR 2228     checkpoint/Raiden host-memory fixes
#   tunix-0003    tunix bf13cd2c    read the trajectory reward key the rollout emits
#   tunix-0004    (local)           let the packing budget raise max_target_length
#
# Most of tunix PR 2228 is already in the base image; tunix-0002 carries only the two
# parts that are not (the CKPT_D2H_CONCURRENT_GB override and the lost-device-set exit).
# The rest of that PR changes the launcher, the YAML generator and the JobSet template,
# which run on the submitting host and need no image change.
#
# tunix-0003 is upstream commit bf13cd2c, minus its one hunk against the trajectory
# logging block that landed after this base image was cut. Without it the orchestrator
# reads `traj["reward"]`, the rollout writes `traj["trajectory_reward"]`, and every GRPO
# advantage in the run is computed from a reward of 0.0.
#
# tunix-0004 has no upstream equivalent. `max_target_length` was pinned to
# max_prompt_length + max_response_length, which is the smallest budget batch_assembly
# accepts, so a packed row held exactly one maximal trajectory and MAX_SEQ_TOKEN_PER_TPU
# did nothing.
ARG BASE_IMAGE=gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8
FROM ${BASE_IMAGE}

COPY maxtext-0001-clear-payload-metadata.patch \
     maxtext-0002-checkpoint-device-host-concurrency.patch \
     tunix-0001-vllm-sampler-weight-sync.patch \
     tunix-0002-checkpoint-raiden-oom.patch \
     tunix-0003-trajectory-reward-key.patch \
     tunix-0004-packing-max-target-length.patch \
     /tmp/patches/

# `--forward` off and no fuzz. An already-applied or inexactly-matching patch means the
# base image differs from the one this overlay was written against, so the resulting run
# would not match the documented configuration.
#
# /app is the tunix repo root and /app/tunix is the package inside it, so `-p1` from /app
# resolves the a/tunix/... paths these patches carry.
RUN cd /app/maxtext \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/maxtext-0001-clear-payload-metadata.patch \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/maxtext-0002-checkpoint-device-host-concurrency.patch \
 && cd /app \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/tunix-0001-vllm-sampler-weight-sync.patch \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/tunix-0002-checkpoint-raiden-oom.patch \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/tunix-0003-trajectory-reward-key.patch \
 && patch -p1 --no-backup-if-mismatch --fuzz=0 < /tmp/patches/tunix-0004-packing-max-target-length.patch \
 && rm -rf /tmp/patches

# Assert each patch landed where it is read from, so a silently reordered or reverted
# hunk fails the build rather than the job.
RUN set -e \
 && grep -q 'dataclasses.replace(payload, metadata={})' /app/maxtext/src/maxtext/training_engine/maxtext_engine.py \
 && grep -q 'checkpoint_storage_device_host_concurrent_gb' /app/maxtext/src/maxtext/configs/base.yml \
 && grep -q 'checkpoint_storage_device_host_concurrent_gb' /app/maxtext/src/maxtext/configs/types.py \
 && grep -q 'save_device_host_concurrent_gb=config.checkpoint_storage_device_host_concurrent_gb' \
      /app/maxtext/src/maxtext/training_engine/checkpointing.py \
 && grep -q '_maybe_register_pathways_persistence' /app/maxtext/src/maxtext/training_engine/checkpointing.py \
 && grep -q '_canonicalize_variable_names' /app/tunix/experimental/rollout/vllm_sampler_adapter.py \
 && grep -q 'data_parallel_size=dp_size' /app/tunix/experimental/examples/common/run_rollout_node.py \
 && grep -q 'CKPT_D2H_CONCURRENT_GB' /app/tunix/utils/maxtext_utils.py \
 && grep -q '_is_unrecoverable_runtime_error' /app/tunix/experimental/worker/remote_execution.py \
 && grep -q 'traj\["trajectory_reward"\]' /app/tunix/experimental/orchestrator/rl_program.py \
 && ! grep -q '"reward", 0.0' /app/tunix/experimental/orchestrator/rl_program.py \
 && grep -q '"trajectory_reward": 0.0' /app/tunix/experimental/orchestrator/distributed_rl_engine.py \
 && grep -q 'metadata\["trajectory_reward"\]' /app/tunix/experimental/rollout/collector.py \
 && grep -q 'max_seq_token_per_tpu' /app/tunix/utils/maxtext_utils.py \
 && grep -q 'max_target_length={max_target_length}' /app/tunix/utils/maxtext_utils.py \
 && grep -q 'max_seq_token_per_tpu' /app/tunix/experimental/examples/common/run_trainer_node.py \
 && python3 -c "import ast,sys; [ast.parse(open(p).read()) for p in sys.argv[1:]]" \
      /app/maxtext/src/maxtext/training_engine/maxtext_engine.py \
      /app/maxtext/src/maxtext/configs/types.py \
      /app/maxtext/src/maxtext/training_engine/checkpointing.py \
      /app/tunix/experimental/rollout/vllm_sampler_adapter.py \
      /app/tunix/experimental/examples/common/run_rollout_node.py \
      /app/tunix/utils/maxtext_utils.py \
      /app/tunix/experimental/worker/remote_execution.py \
      /app/tunix/experimental/orchestrator/rl_program.py \
      /app/tunix/experimental/orchestrator/distributed_rl_engine.py \
      /app/tunix/experimental/rollout/collector.py \
      /app/tunix/experimental/examples/common/run_trainer_node.py
