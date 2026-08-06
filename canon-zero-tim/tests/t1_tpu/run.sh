#!/usr/bin/env bash
# T1 -- topology admission gates.  Needs >=2 TPU chips.  No model, no checkpoint, no engine.
#
#   ./run.sh                 # host: wraps itself in the pinned image
#   CANON_IN_CONTAINER=1 ./run.sh    # already inside a TPU-capable container (cluster)
#
# Run this FIRST on any new cluster.  It costs seconds and it answers the questions that
# decide whether the canonical switch set transfers at all:
#
#   probe_waycount    does the third-program drift appear at THIS reduction width, and does
#                     the fixed-order tree remove it?  (only widths 2 and 4 were ever measured)
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

RC=0
run_probe() {  # <label> <script> <required-line-regex> [required-min-count] [pathways-marker]
  local label="$1" script="$2" need="$3" minc="${4:-1}" check_pathways="${5:-0}"
  echo
  echo "== $label =="
  local tmp_out
  tmp_out="$(mktemp -t canon_probe.XXXXXX)"
  python3 -u "$HERE/$script" 2>&1 | tee "$tmp_out" | sed 's/^/  /'
  local rc="${PIPESTATUS[0]}"
  local out
  out="$(cat "$tmp_out")"
  rm -f "$tmp_out"
  local n
  n=$(echo "$out" | grep -acE "$need")
  if [ "$n" -lt "$minc" ]; then
    echo "  FAIL: $label produced $n measurement line(s) matching /$need/, need >= $minc" >&2
    RC=1
  fi
  if [ "$rc" -ne 0 ]; then
    echo "  FAIL: $label exited $rc" >&2
    RC=1
  fi
  if [ "$check_pathways" = 1 ]; then
    local pn bad_required
    pn=$(echo "$out" | grep -acE '^\[T1\.PATHWAYS\] required=[01] initialized=[01] status=[A-Za-z0-9_-]+$')
    if [ "$pn" -ne 1 ]; then
      echo "  FAIL: $label produced $pn Pathways status marker(s), need exactly 1" >&2
      RC=1
    fi
    bad_required=$(echo "$out" | grep -acE '^\[T1\.PATHWAYS\] required=1 initialized=0 ')
    if [ "$bad_required" -ne 0 ]; then
      echo "  FAIL: $label required Pathways but did not initialize it" >&2
      RC=1
    fi
  fi
}

echo "[t1] XLA_FLAGS=$XLA_FLAGS"
sleep 10

run_probe "P0  Pathways/JAX registration"   probe_devices.py         '^\[t1\.devices\] '   1 1
run_probe "P1  way-count scan (NEW)"        probe_waycount.py        '^\[waycount\] width=' 2 1
run_probe "P2  mesh order / slice (NEW)"    probe_mesh_order.py      '^\[mesh\] VERDICT:'   1 1
run_probe "P3  bucket contract (NEW)"       probe_bucket_contract.py '^\[bucket\] VERDICT:' 1
run_probe "P4  F4 cost model (NEW)"         probe_f4_cost.py         '^\[f4cost\] +[0-9]'   2
run_probe "H1  minrepro: F4 tree"           p19_minrepro_f4.py       '^\[f4\] '             2 1
run_probe "H2  minrepro: third program"     p19_minrepro_thirdprog.py 'DIFFER|SAME'          1 1
run_probe "H3  minrepro: device topology"   p19_minrepro_topo.py     'DIFFER|SAME'          1 1
run_probe "H4  minrepro: mesh geometry"     p19_minrepro_mesh2d.py   '^\[m2d\] '            2 1

echo
if [ "$RC" = 0 ]; then
  echo "===== T1 COMPLETE -- all probes produced measurements ====="
  echo "NOTE: 'complete' means every probe ran and reported.  Whether the numbers ADMIT this"
  echo "topology is a judgement against CLUSTER_ADMISSION.md, not an exit code."
else
  echo "===== T1 FAIL -- a probe did not run or exited nonzero ====="
fi
exit $RC
