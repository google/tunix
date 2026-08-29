#!/usr/bin/env bash
# Read-only D3e reclassification of Attempt 17 at the canonical first action.
set -euo pipefail

output="${1:?usage: run_m15_attempt17_d3e_canonical_action.sh <output-dir> [scratch-parent]}"
scratch_parent="${2:-/tmp}"
script_dir="$(cd "$(dirname "$0")" && pwd)"

test -d "$scratch_parent"
test ! -e "$output"

bash "$script_dir/run_m15_attempt17_d36_offline_binding.sh" \
  "$output" "$scratch_parent"

python3 - "$output" <<'PY'
import json
from pathlib import Path
import sys


class D3EReviewError(RuntimeError):
  pass


def require(condition, message):
  if not condition:
    raise D3EReviewError(message)


root = Path(sys.argv[1])
summary = json.loads((root / "D36_OFFLINE_REVIEW.json").read_text(
    encoding="utf-8"))
classification = json.loads((root / "D36_RECLASSIFICATION.json").read_text(
    encoding="utf-8"))

require(summary.get("target_executed") is False, "D3e unexpectedly ran a target")
require(summary.get("remote_mutation") is False, "D3e unexpectedly mutated remote state")
require(
    summary.get("numerical_repair_authorized") is False,
    "D3e unexpectedly authorized a numerical repair",
)
require(
    summary.get("decision_scope") == "COMPLETION_POSITION_ZERO"
    and classification.get("decision_scope") == "COMPLETION_POSITION_ZERO",
    "D3e canonical first-action scope is absent",
)
alignment = classification.get("alignment", {})
require(
    alignment.get("a_b_differing_bytes") == 207
    and alignment.get("a_b_differing_elements") == 95
    and alignment.get("b_c_differing_bytes") == 0,
    "D3e Attempt-17 numerical boundary drifted",
)
coverage = classification.get("coverage", {})
require(
    coverage.get("total_red_points") == 95
    and coverage.get("first_action_joinable_red_points") == 1
    and coverage.get("decision_candidate_anchors") == 1,
    "D3e canonical first-action coverage drifted",
)

status = summary.get("status")
require(
    status in ("FIRST_RED_LOCALIZED", "FIRST_RED_CANDIDATE_SET_PRESERVED"),
    f"D3e returned an invalid status: {status}",
)
if status == "FIRST_RED_LOCALIZED":
  require(
      classification.get("gate") == "FIRST_RED_LOCALIZED"
      and classification.get("first_red_boundary", {}).get("layer") == 0
      and classification.get("first_red_boundary", {}).get("checkpoint")
      == "rpa_output"
      and classification.get("last_exact_boundary")
      == {"layer": 0, "checkpoint": "k_post_rope"},
      "D3e localized boundary differs from the reviewed candidate interval",
  )
  anchors = classification.get("anchors", [])
  require(len(anchors) == 1, "D3e did not return one canonical anchor")
  anchor = anchors[0]
  binding = anchor.get("source_request_binding", {})
  require(
      anchor.get("source_row") == 217
      and anchor.get("completion_position") == 0
      and anchor.get("source_position") == 1225
      and binding.get("status") == "UNIQUE_FUTURE_PREFIX_BINDING"
      and binding.get("selected_proof_prefix_tokens", -1)
      >= binding.get("required_disambiguation_prefix_tokens", 10**18),
      "D3e canonical source/request binding drifted",
  )
  for arm in ("a", "b"):
    geometry = anchor.get(arm, {}).get("record_geometry", {})
    require(
        geometry.get("layer_fingerprint_shape") == [2048, 1, 15, 8]
        and geometry.get("final_fingerprint_shape") == [2048, 8],
        f"D3e {arm.upper()} fingerprint geometry drifted",
    )
  source_interval = classification.get("source_interval", {})
  require(
      source_interval.get("last_exact", {}).get("line")
      and source_interval.get("first_red", {}).get("line"),
      "D3e source anchors are absent",
  )
  selected_request = binding.get("selected_request_id")
  receipt = classification.get("replay_ledger_receipts", {}).get(
      f"A:{anchor['a']['call_index']}:{selected_request}", {}
  )
  require(
      receipt.get("physical_pages"),
      "D3e selected A request lacks cache-page coordinates",
  )

print(
    "M15_D3E_CANONICAL_ACTION_REVIEW_PASS "
    f"status={status} decision_scope=completion-position-zero "
    f"all_join_mixed={int(bool(classification.get('all_join_mixed_first_difference_signatures')))} "
    "numerical_repair_authorized=0"
)
PY

manifest_sha="$(sha256sum "$output/SHA256SUMS" | awk '{print $1}')"
status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$output/D36_OFFLINE_REVIEW.json")"
echo "[M15.D3E.OFFLINE] COMPLETE status=$status manifest_sha256=$manifest_sha output=$output"
echo "[M15.D3E.OFFLINE] TARGET_NOT_RUN gcs_read=1 gcs_write=0 kubernetes=0 tpu=0"
