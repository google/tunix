#!/usr/bin/env bash
set -euo pipefail

root=/mnt/disks/tunix-data/claude_work
output=/mnt/disks/tunix-data/logp_probe_1host
image=tunix_frozenlake_image:vllm-tpu0.25.0
probe="$root/p19_minrepro_thirdprog.py"
expected_probe_sha=faf65c53223c8ccf1b7d5545084aefe1eabb0918d88ea43127e61ecc577b602f
label="${1:-p36_onehost_excess}"

case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P36.ONEHOST] REFUSING: invalid label: $label" >&2
    exit 2
    ;;
esac

actual_probe_sha="$(sha256sum "$probe" | awk '{print $1}')"
if [ "$actual_probe_sha" != "$expected_probe_sha" ]; then
  echo "[P36.ONEHOST] REFUSING: probe SHA drifted: $actual_probe_sha" >&2
  exit 2
fi

off_log="$output/${label}_off.raw.log"
on_log="$output/${label}_on.raw.log"
for path in "$off_log" "$on_log"; do
  if [ -e "$path" ]; then
    echo "[P36.ONEHOST] REFUSING: evidence path already exists: $path" >&2
    exit 3
  fi
done

echo "[P36.ONEHOST] probe_sha256=$actual_probe_sha"
echo "[P36.ONEHOST] image=$image"

sudo docker run --rm --privileged --net=host \
  --name "${label}_off" \
  -v "$root:$root:ro" \
  -e XLA_FLAGS=--xla_cpu_max_isa=AVX2 \
  -w "$root" \
  "$image" \
  python3 "$probe" >"$off_log" 2>&1
echo "[P36.ONEHOST] OFF_COMPLETE sha256=$(sha256sum "$off_log" | awk '{print $1}')"

sudo docker run --rm --privileged --net=host \
  --name "${label}_on" \
  -v "$root:$root:ro" \
  -e 'XLA_FLAGS=--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false' \
  -w "$root" \
  "$image" \
  python3 "$probe" >"$on_log" 2>&1
echo "[P36.ONEHOST] ON_COMPLETE sha256=$(sha256sum "$on_log" | awk '{print $1}')"

echo "[P36.ONEHOST] PAIR_COMPLETE off=$off_log on=$on_log"
