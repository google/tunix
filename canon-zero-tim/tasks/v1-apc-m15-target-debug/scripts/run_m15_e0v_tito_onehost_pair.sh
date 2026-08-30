#!/usr/bin/env bash
# Run and classify a fresh APC-off/APC-on E0v exact-TiTO one-host pair.
set -euo pipefail

label="${1:?usage: run_m15_e0v_tito_onehost_pair.sh <fresh-lowercase-label>}"
if [[ ! "$label" =~ ^[a-z0-9]([a-z0-9-]{0,30}[a-z0-9])?$ ]]; then
  echo "[M15.E0V.ONEHOST] REFUSING label must be 1-32 lowercase DNS characters" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
root="/mnt/disks/tunix-data/logp_probe_1host/m15_e0v_tito_${label}_pair"
driver="$root/driver.log"
classification="$root/pair.classification.json"
arm_runner="$script_dir/run_m15_e0v_tito_onehost_arm.sh"

test ! -e "$root"
mkdir "$root"
{
  echo "[M15.E0V.ONEHOST] START label=$label"
  echo "[M15.E0V.ONEHOST] pair=off,on topology=DP1xTP4 rounds=3"
  echo "[M15.E0V.ONEHOST] target_executed=0 gcs=0 kubernetes=0"
  sha256sum "$0" "$arm_runner" \
    "$script_dir/classify_m15_e0v_onehost_arm.py" \
    "$script_dir/classify_m15_e0v_onehost_pair.py" \
    "$script_dir/classify_m15_apc_debug_tito.py"
} > "$driver"

off_rc=0
on_rc=0
set +e
bash "$arm_runner" off "$label" "$root" >> "$driver" 2>&1
off_rc=$?
if [ "$off_rc" -eq 0 ]; then
  bash "$arm_runner" on "$label" "$root" >> "$driver" 2>&1
  on_rc=$?
fi
set -e

if [ "$off_rc" -ne 0 ] || [ "$on_rc" -ne 0 ]; then
  python3 - "$root/PAIR_STATUS.json" "$off_rc" "$on_rc" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(
    json.dumps({
        "schema": "m15-e0v-tito-onehost-pair-status-v1",
        "status": "INCONCLUSIVE",
        "off_exit": int(sys.argv[2]),
        "on_exit": int(sys.argv[3]),
        "target_executed": False,
        "numerical_repair_authorized": False,
    }, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
PY
  terminal="[M15.E0V.ONEHOST] INCONCLUSIVE off_exit=$off_rc on_exit=$on_rc root=$root"
  echo "$terminal" >> "$driver"
  sha256sum "$driver" "$root/PAIR_STATUS.json" > "$root/SHA256SUMS"
  echo "$terminal" >&2
  exit 5
fi

python3 "$script_dir/classify_m15_e0v_onehost_pair.py" \
  --root "$root" --output "$classification" >> "$driver" 2>&1
pair_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$classification")"
terminal="[M15.E0V.ONEHOST] $pair_status label=$label topology=DP1xTP4 off_rounds=3 on_rounds=3 B-C=0/0 tito_exact=1 target_executed=0"
echo "$terminal" >> "$driver"
sha256sum "$driver" "$classification" "$root/off/SHA256SUMS" \
  "$root/on/SHA256SUMS" > "$root/SHA256SUMS"
sha256sum -c "$root/SHA256SUMS" --quiet
echo "$terminal"
echo "[M15.E0V.ONEHOST] EVIDENCE root=$root manifest_sha256=$(sha256sum "$root/SHA256SUMS" | awk '{print $1}')"
