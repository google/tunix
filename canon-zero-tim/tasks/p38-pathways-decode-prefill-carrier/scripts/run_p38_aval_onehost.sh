#!/usr/bin/env bash
# Run the model-free P38 aval discriminator on the authorized four-chip host.
set -euo pipefail

label="${1:?usage: run_p38_aval_onehost.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P38.AVAL.ONEHOST] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
probe="$pkg/tests/t1_tpu/probe_logprob_aval.py"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
evidence=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
state="$evidence/p38_aval_${label}"
raw="$evidence/p38_aval_${label}.raw.log"
report="$evidence/p38_aval_${label}.result.json"
canon_out="$state/canon"
container="p38_aval_${label}"
sp=/usr/local/lib/python3.12/site-packages/tpu_inference

source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
for path in "$state" "$raw" "$report"; do
  if [ -e "$path" ]; then
    echo "[P38.AVAL.ONEHOST] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done
mkdir -p "$state"

{
  echo "[P38.AVAL.ONEHOST] source=$(git -C "$repo" rev-parse HEAD) image=$image"
  echo "[P38.AVAL.ONEHOST] topology=DP1xTP4 model=none mutation=none"
  sha256sum "$0" "$probe" "$repo/tunix/rl/canonical_qwen3_adapter.py"
} >"$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen1p7b \
  >>"$raw" 2>&1

set +e
timeout --foreground --signal=TERM --kill-after=60s 1200s \
sudo docker run --rm --privileged --net=host --name "$container" \
  -v /mnt/disks/tunix-data:/mnt/disks/tunix-data \
  -v "$repo":"$repo":ro \
  -v "$canon_out":"$canon_out":ro \
  -v "$canon_out/attn_iface_patched.py":"$sp/layers/common/attention_interface.py":ro \
  -v "$canon_out/linear_p22xk.py":"$sp/layers/jax/linear.py":ro \
  -v "$canon_out/embed_patched.py":"$sp/layers/jax/embed.py":ro \
  -v "$canon_out/tpu_runner_p21_l30.py":"$sp/runner/tpu_runner.py":ro \
  -v "$canon_out/qwen3_p22xk.py":"$sp/models/jax/qwen3.py":ro \
  -v "$canon_out/qwen2_p22xk.py":"$sp/models/jax/qwen2.py":ro \
  -e PYTHONPATH="$canon_out:$repo:$pkg/tests/t1_tpu" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e CANON_P38_AVAL_REPORT="$report" \
  -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_P32_TRAIN_ADMITTED=0 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 \
  -e CANON_LOGPROB_M=256 -e CANON_TARGET_M=256 \
  -e MIN_TOKEN_BUCKET=256 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -w "$repo" "$image" \
  python3 -u "$probe" >>"$raw" 2>&1
docker_rc=$?
set -e

echo "[P38.AVAL.ONEHOST] docker_exit=$docker_rc" >>"$raw"
if [ "$docker_rc" -ne 0 ]; then
  tail -n 120 "$raw"
  exit "$docker_rc"
fi
test -s "$report"
python3 -c "import json; r=json.load(open('$report')); assert r['measurement_count']==5; assert r['negative_control']['differing_elements']==1; assert r['contract']['device_count']==4"
sha256sum "$raw" "$report" "$0"
