#!/bin/bash
# =============================================================================
# raiden_smoke_test/build.sh -- build and push the smoke-test image
# =============================================================================
#   bash raiden_smoke_test/build.sh              # build + push
#   bash raiden_smoke_test/build.sh --no-push    # build only
#   IMAGE_TAG=mytag bash raiden_smoke_test/build.sh
#
# Takes ~8 min warm, ~25 min cold. See README for why this does not simply call
# the repo-root build_docker.sh.
# =============================================================================
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

PUSH=true
[[ "${1:-}" == "--no-push" ]] && PUSH=false

echo "================================================================="
echo " 0. Preflight"
echo "================================================================="
for p in "${TUNIX_DIR}" "${MAXTEXT_DIR}" "${TPU_INFERENCE_DIR}" "${DOCKERFILE}"; do
  [[ -e "$p" ]] || c_die "missing required path: $p
  maxtext/ and tpu-inference/ must be siblings of tunix/ (docker cannot COPY
  outside the build context). Override with MAXTEXT_DIR / TPU_INFERENCE_DIR."
done
c_info "context      ${PROJECTS_DIR}"
c_info "dockerfile   ${DOCKERFILE}"
c_info "image        ${TUNIX_IMAGE}"

# Commit provenance. A mismatch is a warning, not an error -- you may
# legitimately be testing a patched tree.
for pair in "tunix:${TUNIX_DIR}:${EXPECT_TUNIX_COMMIT}" \
            "maxtext:${MAXTEXT_DIR}:${EXPECT_MAXTEXT_COMMIT}" \
            "tpu-inference:${TPU_INFERENCE_DIR}:${EXPECT_TPU_INFERENCE_COMMIT}"; do
  name="${pair%%:*}"; rest="${pair#*:}"; dir="${rest%%:*}"; want="${rest##*:}"
  got="$(git -C "$dir" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
  dirty=""; [[ -n "$(git -C "$dir" status --porcelain 2>/dev/null)" ]] && dirty=" (DIRTY)"
  if [[ "$got" == "$want" ]]; then
    c_info "$(printf '%-14s %s  as expected%s' "$name" "${got:0:9}" "$dirty")"
  else
    c_warn "$(printf '%-14s %s  EXPECTED %s%s' "$name" "${got:0:9}" "${want:0:9}" "$dirty")"
  fi
done

echo "================================================================="
echo " 1. Raiden wheel"
echo "================================================================="
mkdir -p "${RAIDEN_WHEELS_DIR}"
if [[ ! -f "${RAIDEN_WHEELS_DIR}/${WHEEL_NAME}" ]]; then
  c_info "downloading ${WHEEL_NAME}"
  gcloud storage cp "${GCS_WHEEL_PATH}" "${RAIDEN_WHEELS_DIR}/" \
    || gsutil cp "${GCS_WHEEL_PATH}" "${RAIDEN_WHEELS_DIR}/" \
    || c_die "could not fetch wheel from ${GCS_WHEEL_PATH}"
else
  c_info "present: ${WHEEL_NAME}"
fi

# Park every other wheel. The Dockerfile installs raiden_wheels/*.whl as a GLOB,
# so a second wheel here silently double-installs -- and because the package was
# renamed tpu_raiden_jax -> tpu_sync_jax, a name-based prune (as the old
# build_and_push_image.sh used) no longer matches the stale ones.
BACKUP_DIR="${PROJECTS_DIR}/raiden_wheels_backup"
shopt -s nullglob
for w in "${RAIDEN_WHEELS_DIR}"/*.whl; do
  [[ "$(basename "$w")" == "${WHEEL_NAME}" ]] && continue
  mkdir -p "${BACKUP_DIR}"; mv "$w" "${BACKUP_DIR}/"
  c_warn "parked stale wheel $(basename "$w") -> ${BACKUP_DIR}/"
done
n=( "${RAIDEN_WHEELS_DIR}"/*.whl ); shopt -u nullglob
[[ "${#n[@]}" -eq 1 ]] || c_die "expected exactly 1 wheel in ${RAIDEN_WHEELS_DIR}, found ${#n[@]}"
c_info "exactly 1 wheel present"

echo "================================================================="
echo " 2. Docker auth"
echo "================================================================="
DOCKER_CMD="${DOCKER_CMD:-docker}"
gcloud auth configure-docker gcr.io --quiet 2>/dev/null || true
TOKEN="$(gcloud auth print-access-token 2>/dev/null || true)"
[[ -n "$TOKEN" ]] && echo "$TOKEN" | ${DOCKER_CMD} login -u oauth2accesstoken \
  --password-stdin https://gcr.io >/dev/null 2>&1 || true

echo "================================================================="
echo " 3. Build  (vllm=${VLLM_COMMIT:0:9} tpu-inference=${TPU_INFERENCE_COMMIT:0:9})"
echo "================================================================="
# The Dockerfile ends with an import assertion that fails the BUILD if the
# overlaid tpu-inference is incompatible with the installed vLLM. Without it the
# mismatch only surfaces ~25 min into a cluster run as a rollout
# CrashLoopBackOff, after TPUs are allocated. Watch for "BUILD CHECK OK".
( cd "${PROJECTS_DIR}" && ${DOCKER_CMD} build \
    --build-arg VLLM_COMMIT="${VLLM_COMMIT}" \
    --build-arg TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT}" \
    -t "${TUNIX_IMAGE}" -f "${DOCKERFILE}" . ) || c_die "docker build failed"

if [[ "$PUSH" != "true" ]]; then
  echo "built (not pushed): ${TUNIX_IMAGE}"
  exit 0
fi

echo "================================================================="
echo " 4. Push"
echo "================================================================="
${DOCKER_CMD} push "${TUNIX_IMAGE}" || c_die "docker push failed"

DIGEST="$(${DOCKER_CMD} inspect --format='{{index .RepoDigests 0}}' "${TUNIX_IMAGE}" 2>/dev/null || true)"
echo
echo "================================================================="
echo " OK  ${TUNIX_IMAGE}"
[[ -n "$DIGEST" ]] && echo "     ${DIGEST}"
echo "================================================================="
echo "Next:  bash raiden_smoke_test/run.sh dryrun   # render + assert manifests, no TPUs"
