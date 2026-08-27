#!/usr/bin/env bash
# Fetch only the small Attempt-9 seam verdict artifacts and audit them locally.
set -euo pipefail

receipt="${1:?usage: run_m15_wide_seam_gcs_salvage.sh <attempt9-receipt.json> <output-dir> [scratch-parent]}"
output="${2:?usage: run_m15_wide_seam_gcs_salvage.sh <attempt9-receipt.json> <output-dir> [scratch-parent]}"
scratch_parent="${3:-/tmp}"
test -f "$receipt"
test -d "$scratch_parent"
test ! -e "$output"

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[M15.WIDE.SALVAGE] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-wide-salvage.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT

mapfile -t arm_rows < <(python3 - "$receipt" <<'PY'
import json
import pathlib
import re
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for arm, key in (("off", "control_arm_off"), ("on", "treatment_arm_on")):
  value = record.get(key)
  if not isinstance(value, dict):
    raise SystemExit(f"receipt lacks {key}")
  uri = str(value.get("gcs_source_uri", ""))
  if not re.fullmatch(
      r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/"
      r"[a-z0-9-]+/attempt-0",
      uri,
  ):
    raise SystemExit(f"receipt has invalid {arm} GCS root")
  print(f"{arm}\t{uri}")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.WIDE.SALVAGE] REFUSING: receipt did not resolve two arms" >&2
  exit 2
}

objects=(
  PREFLIGHT.json
  COLLECTED.json
  COMPLETE.json
  SHA256SUMS
  seam-classification.json
  p38_seam.classification.json
  m15_wide_seam_bundle.tar
)
for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm root <<< "$row"
  destination="$scratch/$arm"
  mkdir "$destination"
  : > "$destination/remote-inventory.txt"
  for name in "${objects[@]}"; do
    if gcs_exists "$root/$name"; then
      gcs_cp "$root/$name" "$destination/$name"
      printf '%s present\n' "$name" >> "$destination/remote-inventory.txt"
    else
      printf '%s absent\n' "$name" >> "$destination/remote-inventory.txt"
    fi
  done
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$script_dir/audit_m15_wide_seam_gcs_salvage.py" \
  --receipt "$receipt" \
  --off-root "$scratch/off" \
  --on-root "$scratch/on" \
  --output "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/SALVAGE_SUMMARY.json")"
summary_sha="$(sha256sum "$output/SALVAGE_SUMMARY.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.WIDE.SALVAGE] COMPLETE status=$status summary_sha256=$summary_sha manifest_sha256=$manifest_sha output=$output"
