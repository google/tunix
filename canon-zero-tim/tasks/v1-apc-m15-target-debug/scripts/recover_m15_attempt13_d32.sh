#!/usr/bin/env bash
# Recover and mechanically analyze the existing Attempt-13 d32 GCS rounds.
set -euo pipefail

output="${1:?usage: recover_m15_attempt13_d32.sh <output-dir> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
test -d "$scratch_parent"
test ! -e "$output"

script_dir="$(cd "$(dirname "$0")" && pwd)"
task_dir="$(cd "$script_dir/.." && pwd)"
repo="$(cd "$task_dir/../../.." && pwd)"
receipt="$task_dir/evidence/v1_apc_m15_attempt13_paired_d32_20260828/receipt.json"
expected_source="7d30f3827480e6f9d5ae972f55ca4d16f07de6df"
expected_receipt_sha="d1941c2de85050a5652bc5c6e809987f6bf72b996aa817371b08b43870835f95"

test -f "$receipt"
actual_receipt_sha="$(sha256sum "$receipt" | awk '{print $1}')"
[ "$actual_receipt_sha" = "$expected_receipt_sha" ] || {
  echo "[M15.ATTEMPT13] REFUSING: checked-in receipt SHA drifted" >&2
  exit 2
}
git -C "$repo" cat-file -e "$expected_source^{commit}"

scratch="$(mktemp -d -p "$scratch_parent" m15-attempt13-d32.XXXXXX)"
trap 'rm -rf -- "$scratch"' EXIT
render_dir="$scratch/render"
mkdir "$render_dir"

python3 - "$receipt" "$render_dir" "$expected_source" <<'PY'
import hashlib
import json
import pathlib
import re
import sys
import yaml

receipt_path = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
expected_source = sys.argv[3]
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("attempt") != 13 or receipt.get("source_commit") != expected_source:
  raise SystemExit("Attempt-13 receipt identity drifted")
expected = {
    "off": "canon-v1-apc-m15-off-d32-7d30f382",
    "on": "canon-v1-apc-m15-on-d32-7d30f382",
}
for arm, field in (("off", "control_arm_off"), ("on", "treatment_arm_on")):
  value = receipt.get(field)
  if not isinstance(value, dict) or value.get("jobset_name") != expected[arm]:
    raise SystemExit(f"Attempt-13 {arm} JobSet identity drifted")
  uri = str(value.get("gcs_source_uri", ""))
  pattern = rf"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/{expected[arm]}/attempt-0"
  if re.fullmatch(pattern, uri) is None:
    raise SystemExit(f"Attempt-13 {arm} GCS root drifted")
  document = {
      "apiVersion": "jobset.x-k8s.io/v1alpha2",
      "kind": "JobSet",
      "metadata": {"name": expected[arm]},
      "spec": {"replicatedJobs": [{"template": {"spec": {"template": {"spec": {
          "containers": [{"name": "worker", "env": [
              {"name": "CANON_APC_M15_TARGET_DEBUG", "value": arm},
              {"name": "CANON_EXPECT_COMMIT", "value": expected_source},
              {"name": "CANON_P38_DIAGNOSTIC_ROUNDS", "value": "3"},
              {"name": "CANON_P38_SEAM_OBSERVER", "value": "full"},
              {"name": "CANON_P38_SEAM_LAYER", "value": "0"},
              {"name": "CANON_P38_GCS_PREFIX", "value": uri},
          ]}],
      }}}}}]},
  }
  path = output / f"jobset-v1-apc-m15-{arm}-full.yaml"
  path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
names = sorted(path.name for path in output.glob("*.yaml"))
(output / "SHA256SUMS").write_text("".join(
    f"{hashlib.sha256((output / name).read_bytes()).hexdigest()}  {name}\n"
    for name in names
), encoding="ascii")
PY

(cd "$render_dir" && sha256sum -c SHA256SUMS --quiet)
bash "$script_dir/run_m15_multiround_gcs_return.sh" \
  "$render_dir" "$output" "$scratch_parent"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)
python3 "$script_dir/analyze_m15_attempt13_return.py" --return-dir "$output"
(cd "$output" && sha256sum -c SHA256SUMS --quiet)

decision="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' \
  "$output/ATTEMPT13_ANALYSIS.json")"
echo "[M15.ATTEMPT13] RETURN_READY decision=$decision numerical_repair_authorized=0 output=$output"
