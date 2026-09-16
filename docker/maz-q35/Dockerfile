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
# Plus one wheel replacement: the Raiden weight-sync engine is moved from the base
# image's 2026-09-08 build to the 2026-09-14 one, to match the Pathways images. The wheel
# is not in this branch -- .gitignore excludes *.whl and it is 109 MB -- so build through
# docker/maz-q35/build_push.sh, which fetches it from
# gs://cloud-tpu-inference-test-datenglin/ into the build context first.
#
# tunix-0002 carries tunix PR 2228 at head 3421417e. The base image already contains an
# independently written version of most of it, so the patch holds only the differences:
# the CKPT_D2H_CONCURRENT_GB override, the lost-device-set exit, and the four
# _MeshBoundTrainer changes the PR makes on top of what the base image has --
#   * _drain_inflight_checkpoint also looks up the public `checkpoint_manager`, so the
#     PeftTrainer backend is drained too, not only MaxText's private attribute;
#   * prepare_weight_sync no longer drains. The PR now requires the Orbax write and the
#     Raiden transfer to overlap, and ships a test asserting the drain is absent;
#   * _suppress_final_checkpoint drains before either branch and neutralises
#     PeftTrainer's _save_last_checkpoint, which the manager-dropping branch never reached;
#   * _current_train_step requires an int, because PeftTrainer's `train_step` is a method
#     and a bound method compares equal to itself, making every close() look like a
#     duplicate save and silently dropping the final checkpoint.
# Only the second of those is live on this run: TRAINER_BACKEND=maxtext, and the MaxText
# engine exposes `_checkpoint_manager` and an int `train_step`.
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

# Raiden weight-sync engine, upgraded from the base image's 2026-09-08 build to the
# 2026-09-14 one that matches the raiden_20260914 Pathways server and proxy.
#
# The distribution was renamed between the two: tpu_raiden_jax -> tpu_sync_jax, same
# top-level `tpu_sync` package. pip has no way to know they are the same thing, so
# installing over the top would leave the old build's files in place wherever the new
# wheel does not happen to overwrite them, and two dist-info directories claiming the
# same paths. Uninstall by the old name first.
#
# --no-deps: the wheel pins jax==0.11.0, jaxlib==0.11.0 and libtpu==0.0.47. The first two
# already match. libtpu here is 0.0.44 and moving it is not a change this overlay should
# be making silently -- it is the runtime for the rollout's vLLM as well. The .so files in
# the wheel are the thing being upgraded; if they turn out to need 0.0.47 it will show up
# as a load failure at weight-sync time, not as a subtle wrong answer.
ARG TPU_SYNC_WHEEL=tpu_sync_jax-0.0.1.dev20260914193202-cp312-cp312-manylinux_2_31_x86_64.whl
RUN --mount=type=bind,source=${TPU_SYNC_WHEEL},target=/tmp/${TPU_SYNC_WHEEL} \
    pip uninstall -y tpu_raiden_jax \
 && pip install --no-deps --force-reinstall /tmp/${TPU_SYNC_WHEEL} \
 && python3 -c "import importlib.metadata as m; print('tpu_sync_jax', m.version('tpu_sync_jax'))"

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
 && grep -q '_current_train_step' /app/tunix/experimental/examples/common/run_trainer_node.py \
 && grep -q '_save_last_checkpoint' /app/tunix/experimental/examples/common/run_trainer_node.py \
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
