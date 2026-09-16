#!/bin/bash
# Builds and pushes the Qwen3.5-35B-A3B GRPO overlay image.
#
#   bash docker/maz-q35/build_push.sh [TAG]
#
# Env overrides: BASE_IMAGE, IMAGE_REPO, PUSH=false, DOCKER_CMD.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-gcr.io/cloud-tpu-multipod-dev/yixuannwang_google_com-runner:yixuann-e2e-0912head-v8}"
IMAGE_REPO="${IMAGE_REPO:-gcr.io/cloud-tpu-multipod-dev/mazumdera-runner}"
TAG="${1:-q35-$(date +%m%d)-v1}"
TARGET_IMAGE="${IMAGE_REPO}:${TAG}"
PUSH="${PUSH:-true}"

DOCKER_CMD="${DOCKER_CMD:-docker}"
if ! ${DOCKER_CMD} info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
  else
    echo "ERROR: docker is not usable from this account and passwordless sudo is" >&2
    echo "       unavailable. Re-run under sudo, or add yourself to the docker group." >&2
    exit 1
  fi
fi

echo "=== Building ${TARGET_IMAGE}"
echo "    base: ${BASE_IMAGE}"
DOCKER_BUILDKIT=1 ${DOCKER_CMD} build \
  --network=host \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${TARGET_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

# The Dockerfile's own greps prove the patches landed in the layer; this proves the
# image that came out of the build is the one that will be pulled, and that the patched
# modules still import -- a syntactically valid file that fails at import time would
# otherwise only surface as a crash-looping pod. Each assertion reads the symbol through
# the import system rather than off disk, so a stale copy earlier on sys.path fails here.
echo "=== Verifying the built image"
${DOCKER_CMD} run --rm --entrypoint bash "${TARGET_IMAGE}" -c '
python3 - <<PY
import inspect

from maxtext.training_engine import checkpointing, maxtext_engine
from tunix.experimental.examples.common import run_rollout_node, run_trainer_node
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.rollout import collector, vllm_sampler_adapter
from tunix.experimental.worker import remote_execution
from tunix.utils import maxtext_utils

# maxtext-0001: PR 5219
assert "metadata={}" in inspect.getsource(
    maxtext_engine.MaxTextTrainingEngine._prepare_batch)

# maxtext-0002: PR 5234. The config field has to survive pyconfig as well as exist in
# types.py, and the handler has to be the one that consumes it.
assert hasattr(checkpointing, "_maybe_register_pathways_persistence")
assert "save_device_host_concurrent_gb" in inspect.getsource(
    checkpointing.CheckpointManager.__init__)
from maxtext.configs import types
assert "checkpoint_storage_device_host_concurrent_gb" in types.OrbaxStorage.model_fields

# tunix-0001: PR 2229
assert hasattr(vllm_sampler_adapter, "_canonicalize_variable_names")
assert "data_parallel_size" in inspect.getsource(
    run_rollout_node._create_vllm_sampler)

# tunix-0002: PR 2228 at head 3421417e, the parts the base image lacks.
assert "CKPT_D2H_CONCURRENT_GB" in inspect.getsource(
    maxtext_utils.build_maxtext_config)
assert hasattr(remote_execution, "_is_unrecoverable_runtime_error")

# The four _MeshBoundTrainer changes. Exercise _current_train_step against both
# backends' attribute shapes rather than reading its source: PeftTrainer's
# train_step is a method, and picking it up instead of the int train_steps is the
# failure this method exists to prevent.
_mbt = run_trainer_node._MeshBoundTrainer


class _PeftLike:
  train_steps = 7

  def train_step(self, payload=None):
    return 1


class _MaxTextLike:
  train_step = 4


assert _mbt(_PeftLike(), None)._current_train_step() == 7
assert _mbt(_MaxTextLike(), None)._current_train_step() == 4
assert _mbt(object(), None)._current_train_step() is None

_drain_src = inspect.getsource(_mbt._drain_inflight_checkpoint)
assert '"_checkpoint_manager"' in _drain_src and '"checkpoint_manager"' in _drain_src
assert "_save_last_checkpoint" in inspect.getsource(_mbt._suppress_final_checkpoint)
# The PR requires the Orbax write and the Raiden transfer to overlap, so the
# weight-sync path must not drain.
assert "_drain_inflight_checkpoint" not in inspect.getsource(_mbt.prepare_weight_sync)

# tunix-0003: upstream bf13cd2c. Exercise _extract_reward rather than reading its
# source: it must return the rollout key and refuse the old one, not default to 0.0.
assert rl_program._extract_reward({"trajectory_reward": 1.5}) == 1.5
try:
  rl_program._extract_reward({"reward": 1.5})
except KeyError:
  pass
else:
  raise AssertionError("_extract_reward still accepts the pre-bf13cd2c key")
assert "trajectory_reward" in inspect.getsource(
    collector.TrajectoryCollectorEngine)

# tunix-0004: the packing budget has to reach max_target_length through both the
# trainer node argument and build_maxtext_config.
assert "max_seq_token_per_tpu" in inspect.signature(
    maxtext_utils.build_maxtext_config).parameters
assert "--max_seq_token_per_tpu" in inspect.getsource(run_trainer_node)

print("ok: all six overlay patches are live in the image, PR 2228 at head 3421417e")
PY'

if [ "${PUSH}" != "true" ]; then
  echo "=== PUSH=${PUSH}, stopping after build: ${TARGET_IMAGE}"
  exit 0
fi

echo "=== Pushing ${TARGET_IMAGE}"
gcloud auth configure-docker gcr.io --quiet
${DOCKER_CMD} push "${TARGET_IMAGE}"

# Pin the manifests to the digest: the tag is mutable, and the orchestrator, trainer and
# rollout pods have to be running identical code for weight sync to mean anything.
DIGEST="$(${DOCKER_CMD} inspect --format='{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "${TARGET_IMAGE}")"
echo "=== Pushed"
echo "    ${TARGET_IMAGE}"
echo "    ${DIGEST}"
