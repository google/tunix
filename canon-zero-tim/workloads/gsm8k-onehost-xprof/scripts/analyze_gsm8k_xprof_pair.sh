#!/usr/bin/env bash
# Postflight for two completed GSM8K one-host XProf arms.
set -euo pipefail

native_root="${1:?usage: analyze_gsm8k_xprof_pair.sh <native-root> <zero-root> <fresh-output-dir>}"
zero_root="${2:?usage: analyze_gsm8k_xprof_pair.sh <native-root> <zero-root> <fresh-output-dir>}"
output_dir="${3:?usage: analyze_gsm8k_xprof_pair.sh <native-root> <zero-root> <fresh-output-dir>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
trace_summary="${V1_GSM8K_XPROF_TRACE_SUMMARY:-$script_dir/xprof_trace_summary.py}"

for root in "$native_root" "$zero_root"; do
  if [ "$root" = "${root#/}" ] || [ ! -s "$root/train/classification.json" ]; then
    echo "[V1.GSM8K.XPROF.PAIR] invalid arm root: $root" >&2
    exit 2
  fi
done
if [ "$output_dir" = "${output_dir#/}" ] || [ -e "$output_dir" ]; then
  echo "[V1.GSM8K.XPROF.PAIR] output must be a fresh absolute directory" >&2
  exit 2
fi
if [ ! -f "$trace_summary" ]; then
  echo "[V1.GSM8K.XPROF.PAIR] xprof-trace-analysis helper absent: $trace_summary" >&2
  exit 2
fi

mapfile -d '' native_traces < <(
  find "$native_root/train/xprof" -type f -name '*.trace.json.gz' -size +0 -print0
)
mapfile -d '' zero_traces < <(
  find "$zero_root/train/xprof" -type f -name '*.trace.json.gz' -size +0 -print0
)
if [ "${#native_traces[@]}" -ne 1 ] || [ "${#zero_traces[@]}" -ne 1 ]; then
  echo "[V1.GSM8K.XPROF.PAIR] expected exactly one non-empty trace per arm; native=${#native_traces[@]} zero=${#zero_traces[@]}" >&2
  exit 2
fi

mkdir -p "$output_dir"
set +e
python3 "$script_dir/classify_gsm8k_xprof_pair.py" \
  --native "$native_root/train/classification.json" \
  --zero-hp "$zero_root/train/classification.json" \
  --output "$output_dir/pair_classification.json" \
  >"$output_dir/pair_classifier.txt" 2>&1
pair_rc=$?
set -e
if [ "$pair_rc" -ne 0 ] && [ "$pair_rc" -ne 3 ]; then
  cat "$output_dir/pair_classifier.txt" >&2
  exit "$pair_rc"
fi

python3 "$trace_summary" \
  --control "${native_traces[0]}" \
  --treatment "${zero_traces[0]}" \
  --top 20 \
  --output "$output_dir/xprof_trace_summary.json" \
  >"$output_dir/xprof_trace_summary.txt"

sha256sum \
  "$native_root/train/classification.json" \
  "$native_root/train/xprof_census.txt" \
  "$native_root/train/semantic_census.txt" \
  "$zero_root/train/classification.json" \
  "$zero_root/train/xprof_census.txt" \
  "$zero_root/train/semantic_census.txt" \
  "$output_dir/pair_classification.json" \
  "$output_dir/xprof_trace_summary.json" \
  >"$output_dir/SHA256SUMS"

if [ "$pair_rc" -eq 0 ]; then
  echo "[V1.GSM8K.XPROF.PAIR] PASS matched_work=1 output=$output_dir"
else
  echo "[V1.GSM8K.XPROF.PAIR] INCONCLUSIVE_INPUT_MISMATCH matched_work=0 output=$output_dir"
fi
exit "$pair_rc"
