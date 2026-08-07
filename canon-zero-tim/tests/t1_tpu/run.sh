#!/usr/bin/env bash
# T1 -- topology admission gates.  Needs >=2 TPU chips.  No model, no checkpoint, no engine.
#
#   ./run.sh                 # host: wraps itself in the pinned image
#   CANON_IN_CONTAINER=1 ./run.sh    # already inside a TPU-capable container (cluster)
#
# Run this FIRST on any new cluster.  It costs seconds and it answers the questions that
# decide whether the canonical switch set transfers at all:
#
#   probe_waycount    on full-slice (replica,tp) meshes, compare replicated, stock-AR and F4
#                     arms using identical arrays; never infer TP4 from devices[:4]
#   probe_mesh_order  what order did topology-aware placement actually pick, and is this
#                     topology multi-slice (a collective family with zero coverage)?
#   probe_bucket      what MIN_TOKEN_BUCKET does this dp geometry need?  (it is a GLOBAL
#                     token count; copying the dp=1 value silently unpins the bucket)
#   probe_f4_cost     what does the fixed-order tree cost at this width?  (analytic)
#
# The four historical minimal reproducers are run too, so a new host can be compared against
# the documented v5p-8 numbers directly.
#
# Fail-closed: a probe that prints no measurement line did not run.  That is never a pass.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${CANON_IMAGE:-tunix_frozenlake_image:vllm-tpu0.25.0}"
XTRA_XLA="${XTRA_XLA:---xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false}"

if [ "${CANON_IN_CONTAINER:-0}" != "1" ]; then
  # Host mode: re-enter inside the pinned image with TPU access.
  echo "[t1] host mode -- re-entering $IMAGE (set CANON_IN_CONTAINER=1 to run directly)"
  exec ${DOCKER:-sudo docker} run --rm --privileged --net=host \
    --name "canon_t1_$$" \
    -v "$HERE":"$HERE":ro \
    -e CANON_IN_CONTAINER=1 \
    -e XLA_FLAGS="$XTRA_XLA" \
    -e CANON_WAYCOUNT_WIDTHS="${CANON_WAYCOUNT_WIDTHS:-}" \
    -e CANON_WAYCOUNT_DEPTHS="${CANON_WAYCOUNT_DEPTHS:-}" \
    -e CANON_MINREPRO_N="${CANON_MINREPRO_N:-}" \
    -e CANON_EXPECT_MODEL_MESH_IDS="${CANON_EXPECT_MODEL_MESH_IDS:-}" \
    -e CANON_MESH_SHAPE="${CANON_MESH_SHAPE:-}" \
    -e CANON_REQUIRE_PATHWAYS="${CANON_REQUIRE_PATHWAYS:-}" \
    -e CANON_DP_SIZE="${CANON_DP_SIZE:-}" \
    -e CANON_TARGET_M="${CANON_TARGET_M:-}" \
    -w "$HERE" "$IMAGE" bash "$HERE/run.sh"
fi

export XLA_FLAGS="${XLA_FLAGS:-$XTRA_XLA}"
case "$XLA_FLAGS" in
  *--xla_allow_excess_precision=false*) ;;
  *) echo "[t1] REFUSING: XLA_FLAGS lacks --xla_allow_excess_precision=false -- every number "\
"below would belong to a different program family than a canonical run." >&2; exit 2;;
esac

echo "[t1] XLA_FLAGS=$XLA_FLAGS"
python3 -u "$HERE/unified_runner.py"
RC=$?
exit $RC
