#!/usr/bin/env bash
# Run the immutable P66 full-depth TP4 causal arms serially.
set -euo pipefail

serial_label="${1:?usage: run_tp4_campaign.sh <serial-label> <unsafe-label> <p59-label> <gather-off-label> <campaign-label>}"
unsafe_label="${2:?usage: run_tp4_campaign.sh <serial-label> <unsafe-label> <p59-label> <gather-off-label> <campaign-label>}"
p59_label="${3:?usage: run_tp4_campaign.sh <serial-label> <unsafe-label> <p59-label> <gather-off-label> <campaign-label>}"
gather_off_label="${4:?usage: run_tp4_campaign.sh <serial-label> <unsafe-label> <p59-label> <gather-off-label> <campaign-label>}"
campaign_label="${5:?usage: run_tp4_campaign.sh <serial-label> <unsafe-label> <p59-label> <gather-off-label> <campaign-label>}"
for label in "$serial_label" "$unsafe_label" "$p59_label" "$gather_off_label" "$campaign_label"; do
  case "$label" in
    *[!a-zA-Z0-9_-]*|'') echo "[P66.TP4] invalid campaign label: $label" >&2; exit 2 ;;
  esac
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
runner="$script_dir/run_onehost_tp4_arm.sh"
classifier="$pkg/tests/p66_backward/classify_tp4_campaign.py"
evidence=/mnt/disks/tunix-data/logp_probe_1host
campaign_root="$evidence/p66_tp4_campaign_${campaign_label}"
result="$campaign_root/result.json"
driver="$campaign_root/driver.log"
declare -A labels=(
  [tp4-serial]="$serial_label"
  [tp4-p59-old]="$unsafe_label"
  [tp4-p59]="$p59_label"
  [tp4-gather-off]="$gather_off_label"
)
if [ -e "$campaign_root" ]; then
  echo "[P66.TP4] REFUSING: campaign label already exists" >&2
  exit 2
fi
for arm in tp4-serial tp4-p59-old tp4-p59 tp4-gather-off; do
  root="$evidence/p66_tp4_${arm}_${labels[$arm]}"
  if [ -e "$root" ]; then
    echo "[P66.TP4] REFUSING: arm label already exists: $root" >&2
    exit 2
  fi
done
mkdir -p "$campaign_root"
{
  echo "[P66.TP4] CAMPAIGN_BEGIN label=$campaign_label optimizer_commits=0"
  echo "[P66.TP4] order=tp4-serial,tp4-p59-old,tp4-p59,tp4-gather-off"
  echo "[P66.TP4] unsafe_expected=diagnostic_fatal repaired_expected=17/17_strict"
} >"$driver"

for arm in tp4-serial tp4-p59-old tp4-p59 tp4-gather-off; do
  bash "$runner" "$arm" "${labels[$arm]}" >>"$driver" 2>&1
done

args=()
for arm in tp4-serial tp4-p59-old tp4-p59 tp4-gather-off; do
  root="$evidence/p66_tp4_${arm}_${labels[$arm]}/train"
  args+=("--${arm}-classification" "$root/classification.json")
  args+=("--${arm}-pre" "$root/pre_alignment.jsonl")
  if [ "$arm" != tp4-p59-old ]; then
    args+=("--${arm}-update" "$root/update.json")
  fi
done
set +e
python3 "$classifier" "${args[@]}" --output "$result" >>"$driver" 2>&1
classifier_rc=$?
set -e
verdict=MISSING
if [ -s "$result" ]; then
  verdict="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$result")"
fi
echo "[P66.TP4] CAMPAIGN_END verdict=$verdict classifier=$classifier_rc" >>"$driver"
manifest_inputs=("$driver" "$classifier" "$0")
[ ! -s "$result" ] || manifest_inputs=("$result" "${manifest_inputs[@]}")
for arm in tp4-serial tp4-p59-old tp4-p59 tp4-gather-off; do
  manifest_inputs+=("$evidence/p66_tp4_${arm}_${labels[$arm]}/SHA256SUMS")
done
sha256sum "${manifest_inputs[@]}" >"$campaign_root/SHA256SUMS"
if [ "$classifier_rc" -ne 0 ]; then
  exit "$classifier_rc"
fi
echo "P66_TP4_CAMPAIGN_COMPLETE verdict=$verdict evidence=$campaign_root"
