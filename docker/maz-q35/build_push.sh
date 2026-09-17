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

# The Raiden wheel is 109 MB and .gitignore excludes *.whl, so it is not in the branch.
# Fetch it into the build context; the Dockerfile bind-mounts it by name.
TPU_SYNC_WHEEL="tpu_sync_jax-0.0.1.dev20260914193202-cp312-cp312-manylinux_2_31_x86_64.whl"
TPU_SYNC_WHEEL_URI="gs://cloud-tpu-inference-test-datenglin/${TPU_SYNC_WHEEL}"
if [ ! -f "${SCRIPT_DIR}/${TPU_SYNC_WHEEL}" ]; then
  echo "=== Fetching ${TPU_SYNC_WHEEL_URI}"
  gcloud storage cp "${TPU_SYNC_WHEEL_URI}" "${SCRIPT_DIR}/${TPU_SYNC_WHEEL}"
fi

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
#
# The script is fed to the container's stdin under a quoted heredoc. An earlier version
# wrapped it in `bash -c '...'`, where a Python quote closes the shell argument: the
# heredoc terminator was swallowed and the assertions never ran, while the build still
# reported success. With <<'PY' the host shell performs no quote or dollar processing.
${DOCKER_CMD} run --rm -i --entrypoint python3 "${TARGET_IMAGE}" - <<'PY'
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

# tunix-0005: upstream babc1c70 + 0cfdab45. The flag has to reach the constructor and
# the constructor has to accept it, so check both ends rather than either alone.
from tunix.experimental.examples.math_gsm8k_dist import run_gsm8k_dist_grpo

assert "trajectory_log_dir" in inspect.signature(
    rl_program.StandardRLProgram.__init__).parameters
assert hasattr(rl_program, "trajectory_logger")
_row_src = inspect.getsource(rl_program.StandardRLProgram._log_consumed_trajectories)
for _field in ("reward", "completion", "gold_answer", "prompt_id", "global_step"):
  assert '"%s"' % _field in _row_src, _field
# The row must report the reward the run trained on, not recompute one, so that it is
# correct under REWARD_MODE=env where no orchestrator-side reward_fn exists.
assert "trajectory_reward" in _row_src
assert "TRAJECTORY_LOG_DIR" in inspect.getsource(run_gsm8k_dist_grpo._parse_args)
assert "trajectory_log_dir=args.trajectory_log_dir" in inspect.getsource(
    run_gsm8k_dist_grpo.main)

# The Raiden wheel replacement. Check the version through the metadata and the FFI
# extension through an actual import: a wheel whose .so cannot load against this image
# libtpu would otherwise only surface at the first weight sync.
import importlib.metadata as _md

assert _md.version("tpu_sync_jax") == "0.0.1.dev20260914193202", _md.version(
    "tpu_sync_jax")
try:
  _md.version("tpu_raiden_jax")
except _md.PackageNotFoundError:
  pass
else:
  raise AssertionError("the superseded tpu_raiden_jax distribution is still installed")
from tpu_sync.frameworks.jax import weight_synchronizer_ffi  # noqa: F401

print("ok: all seven overlay patches are live in the image, PR 2228 at head 3421417e,")
print("    tpu_sync_jax 0.0.1.dev20260914193202")
PY

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
