#!/usr/bin/env bash
# Download and audit only small three-round M15 verdict artifacts.
set -euo pipefail

render_dir="${1:?usage: run_m15_multiround_gcs_return.sh <render-dir> <output-dir> [scratch-parent] [preserve-failures:0|1]}"
output="${2:?usage: run_m15_multiround_gcs_return.sh <render-dir> <output-dir> [scratch-parent] [preserve-failures:0|1]}"
scratch_parent="${3:-/tmp}"
preserve_failures="${4:-0}"
test -d "$render_dir"
test -d "$scratch_parent"
test ! -e "$output"
case "$preserve_failures" in
  0|1) ;;
  *) echo "[M15.MULTIROUND] REFUSING: preserve-failures must be 0 or 1" >&2; exit 2 ;;
esac

if command -v gcloud >/dev/null 2>&1; then
  gcs_cp() { gcloud storage cp "$1" "$2" >/dev/null; }
  gcs_exists() { gcloud storage ls "$1" >/dev/null 2>&1; }
elif command -v gsutil >/dev/null 2>&1; then
  gcs_cp() { gsutil -q cp "$1" "$2"; }
  gcs_exists() { gsutil -q stat "$1" >/dev/null 2>&1; }
else
  echo "[M15.MULTIROUND] REFUSING: gcloud or gsutil is required" >&2
  exit 2
fi

scratch="$(mktemp -d -p "$scratch_parent" m15-multiround.XXXXXX)"
cleanup() {
  local exit_code="$1"
  trap - EXIT
  if [ "$exit_code" -ne 0 ] && [ "$preserve_failures" = "1" ]; then
    echo "[M15.MULTIROUND] FAILURE_PRESERVED scratch=$scratch" >&2
  else
    rm -rf -- "$scratch"
  fi
  exit "$exit_code"
}
trap 'cleanup $?' EXIT

mapfile -t arm_rows < <(python3 - "$render_dir" <<'PY'
import pathlib
import re
import sys
import yaml

root = pathlib.Path(sys.argv[1])
paths = sorted(root.glob("jobset-v1-apc-m15-*-*.yaml"))
if len(paths) != 2:
  raise SystemExit("render directory must contain exactly the off/on layer pair")
seen = set()
for path in paths:
  document = yaml.safe_load(path.read_text(encoding="utf-8"))
  container = document["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]["containers"][0]
  env = {row["name"]: str(row["value"]) for row in container["env"] if "value" in row}
  arm = env.get("CANON_APC_M15_TARGET_DEBUG", "")
  source = env.get("CANON_EXPECT_COMMIT", "")
  rounds = env.get("CANON_P38_DIAGNOSTIC_ROUNDS", "")
  observer = env.get("CANON_P38_SEAM_OBSERVER", "")
  uri = env.get("CANON_P38_GCS_PREFIX", "")
  if arm not in ("off", "on") or arm in seen:
    raise SystemExit("rendered pair has invalid or duplicate arm")
  if not re.fullmatch(r"[0-9a-f]{40}", source) or rounds != "3" or observer not in ("layer", "full"):
    raise SystemExit("rendered source/round contract drifted")
  if not re.fullmatch(r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/[a-z0-9-]+/attempt-0", uri):
    raise SystemExit("rendered GCS root is invalid")
  seen.add(arm)
  print(f"{arm}\t{source}\t{rounds}\t{observer}\t{uri}")
PY
)
[ "${#arm_rows[@]}" -eq 2 ] || {
  echo "[M15.MULTIROUND] REFUSING: rendered pair did not resolve two arms" >&2
  exit 2
}

source_commit=""
for row in "${arm_rows[@]}"; do
  IFS=$'\t' read -r arm source rounds observer root <<< "$row"
  if [ -z "$source_commit" ]; then
    source_commit="$source"
  elif [ "$source_commit" != "$source" ]; then
    echo "[M15.MULTIROUND] REFUSING: paired source commits differ" >&2
    exit 2
  fi
  arm_root="$scratch/$arm"
  mkdir -p "$arm_root/root"
  for round_index in 0 1 2; do
    printf -v round_text '%06d' "$round_index"
    remote="$root/wide/rounds/$round_text"
    local_round="$arm_root/round-$round_text"
    mkdir "$local_round" "$local_round/stages"
    : > "$local_round/remote-inventory.txt"
    for name in ROUND_INPUT_RECEIPT.json p38_seam.classification.json \
        WIDE_SHA256SUMS WIDE_ROUND_COMPLETE.json m15_wide_seam_bundle.tar; do
      if gcs_exists "$remote/$name"; then
        printf '%s present\n' "$name" >> "$local_round/remote-inventory.txt"
        if [ "$name" != m15_wide_seam_bundle.tar ]; then
          gcs_cp "$remote/$name" "$local_round/$name"
        fi
      else
        printf '%s absent\n' "$name" >> "$local_round/remote-inventory.txt"
      fi
    done
    mkdir "$local_round/classifier-input"
    for name in ROUND_INPUT_RECEIPT.json m15-replay-envelope.jsonl \
        pre-alignment.jsonl mismatch-capsule.npz \
        CLASSIFIER_INPUT_SHA256SUMS CLASSIFIER_INPUT_RECEIPT.json; do
      if gcs_exists "$remote/classifier-input/$name"; then
        printf 'classifier-input/%s present\n' "$name" \
          >> "$local_round/remote-inventory.txt"
        case "$name" in
          CLASSIFIER_INPUT_SHA256SUMS|CLASSIFIER_INPUT_RECEIPT.json)
            gcs_cp "$remote/classifier-input/$name" \
              "$local_round/classifier-input/$name"
            ;;
        esac
      else
        printf 'classifier-input/%s absent\n' "$name" \
          >> "$local_round/remote-inventory.txt"
      fi
    done
    for stage_spec in \
        10:assemble 15:checkpoint-input 20:classify 30:package 35:local-export \
        40:manifest 50:upload 60:remote-verify 70:completion; do
      ordinal="${stage_spec%%:*}"
      stage_name="${stage_spec#*:}"
      for stage_status in STARTED PASS FAIL; do
        name="STAGE_${ordinal}_${stage_name}_${stage_status}.json"
        if gcs_exists "$remote/stages/$name"; then
          printf 'stages/%s present\n' "$name" \
            >> "$local_round/remote-inventory.txt"
          gcs_cp "$remote/stages/$name" "$local_round/stages/$name"
        else
          printf 'stages/%s absent\n' "$name" \
            >> "$local_round/remote-inventory.txt"
        fi
      done
    done
  done
  for name in PREFLIGHT.json COLLECTED.json COMPLETE.json; do
    if gcs_exists "$root/$name"; then
      gcs_cp "$root/$name" "$arm_root/root/$name"
    fi
  done
done

script_dir="$(cd "$(dirname "$0")" && pwd)"
python3 "$script_dir/audit_m15_multiround_gcs_return.py" \
  --source-commit "$source_commit" \
  --rounds 3 \
  --off-root "$scratch/off" \
  --on-root "$scratch/on" \
  --output "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/MULTIROUND_SUMMARY.json")"
summary_sha="$(sha256sum "$output/MULTIROUND_SUMMARY.json" | awk '{print $1}')"
manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
echo "[M15.MULTIROUND] COMPLETE status=$status summary_sha256=$summary_sha manifest_sha256=$manifest_sha output=$output"
