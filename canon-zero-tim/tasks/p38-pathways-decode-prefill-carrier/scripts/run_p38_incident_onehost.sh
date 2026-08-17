#!/usr/bin/env bash
# Rehearse the P38 multi-round incident hook on real Qwen3-8B DP1xTP4.
set -euo pipefail

mode="${1:?usage: run_p38_incident_onehost.sh off|on|seam-layer|seam-tail|terminal-discriminator <unique-label>}"
label="${2:?usage: run_p38_incident_onehost.sh off|on|seam-layer|seam-tail|terminal-discriminator <unique-label>}"
case "$mode" in off|on|seam-layer|seam-tail|terminal-discriminator) ;; *) echo "invalid mode: $mode" >&2; exit 2 ;; esac
case "$label" in *[!a-zA-Z0-9_-]*|'') echo "invalid label: $label" >&2; exit 2 ;; esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
pkg="$repo/canon-zero-tim"
canon_env=/mnt/disks/tunix-data/claude_work/canon_env.sh
hf_cache=/mnt/disks/tunix-data/hf/hub/models--Qwen--Qwen3-8B
hf_snapshot_sha="$(cat "$hf_cache/refs/main")"
model="$hf_cache/snapshots/$hf_snapshot_sha"
data=/mnt/disks/tunix-data/frozenlake/data_convergence_v1
evidence=/mnt/disks/tunix-data/logp_probe_1host
deps="$evidence/p38_incident_deps_nodeps"
image=tunix_frozenlake_image:vllm-tpu0.25.0
sp=/usr/local/lib/python3.12/site-packages/tpu_inference
state="$evidence/p38_incident_${label}_${mode}"
raw="$evidence/p38_incident_${label}_${mode}.raw.log"
pre_report="$state/pre_alignment.jsonl"
capsule="$state/mismatch.npz"
round_file="$state/diagnostic_round"
capture="$state/capture"
canon_out="$state/canon"
container="p38_incident_${label}_${mode}"
timeout_seconds="${P38_INCIDENT_ONEHOST_TIMEOUT_SECONDS:-3600}"

source "$canon_env"
canon_preflight
test "$(hostname)" = t1v-n-4a77ebd0-w-0
test -s "$model/config.json"
test -s "$data/train.parquet"
test -s "$deps/gymnasium/__init__.py"
for path in "$state" "$raw"; do
  if [ -e "$path" ]; then
    echo "[P38.INCIDENT.ONEHOST] REFUSING: evidence path exists: $path" >&2
    exit 3
  fi
done
mkdir -p "$state"
printf '0\n' > "$round_file"

source_sha="$(git -C "$repo" rev-parse HEAD)"
diff_sha="$(git -C "$repo" diff --binary HEAD | sha256sum | awk '{print $1}')"
{
  echo "[P38.INCIDENT.ONEHOST] source=$source_sha diff_sha256=$diff_sha image=$image mode=$mode"
  echo "[P38.INCIDENT.ONEHOST] topology=DP1xTP4 model=Qwen3-8B rounds=3 prefix_cache=disabled"
  echo "[P38.INCIDENT.ONEHOST] backward=0 optimizer_commits=0 wandb=disabled timeout=$timeout_seconds"
  sha256sum "$0" "$pkg/install.sh" "$repo/tunix/rl/alignment.py" \
    "$repo/tunix/rl/agentic/agentic_rl_learner.py" \
    "$repo/examples/frozenlake/train_frozenlake_qwen3.py" \
    "$model/config.json" "$model/model.safetensors.index.json"
} > "$raw"

bash "$pkg/install.sh" "$canon_out" --from-image "$image" --model qwen8b \
  >> "$raw" 2>&1

capture_env=()
if [ "$mode" != off ]; then
  mkdir -p "$capture"
  capture_env+=(
    -e CANON_P38_SERVING_CAPTURE_DIR="$capture"
    -e CANON_P38_REQUEST_JOURNAL="$capture/p38_request_journal.jsonl"
    -e CANON_P38_INCIDENT_LEDGER="$capture/p38_incident_ledger.jsonl"
    -e CANON_P38_INCIDENT_MIN_PREFIX=0
    -e CANON_P38_INCIDENT_MAX_PREFIX=3072
    -e CANON_P38_INCIDENT_MAX_BYTES=134217728
    -e CANON_P38_DIAGNOSTIC_ROUND_FILE="$round_file"
    -e CANON_P38_SERVING_CAPTURE_MAX_CALLS=4
    -e CANON_P38_SERVING_CAPTURE_MIN_PREFIX=0
    -e CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS=0,512,1024,1536,3072
    -e CANON_P38_SERVING_CAPTURE_FREE_SPACE_MULTIPLIER=5
    -e CANON_P38_SERVING_CAPTURE_EXPECTED_PATH=standard
  )
fi
if [ "$mode" = on ]; then
  capture_env+=(
    -e CANON_P38_KV_OBSERVER_DIR="$capture"
    -e CANON_P38_KV_OBSERVER_MAX_CANDIDATES=3
    -e CANON_P38_KV_OBSERVER_MAX_PAGES=16
    -e CANON_P38_KV_OBSERVER_MAX_BYTES=134217728
    -e CANON_P38_KV_OBSERVER_MAX_READ_BYTES=671088640
  )
elif [ "$mode" = seam-layer ] || [ "$mode" = seam-tail ] || \
     [ "$mode" = terminal-discriminator ]; then
  capture_env+=(
    -e CANON_P38_SEAM_OBSERVER=layer
    -e CANON_P38_SEAM_OBSERVER_DIR="$capture"
    -e CANON_P38_SEAM_MIN_POSITION=1400
    -e CANON_P38_SEAM_MAX_POSITION=1600
    -e CANON_P38_SEAM_MAX_BYTES=536870912
  )
  if [ "$mode" = seam-tail ] || [ "$mode" = terminal-discriminator ]; then
    capture_env+=(
      -e CANON_P38_TAIL_OBSERVER=1
      -e CANON_P38_TAIL_MAX_BYTES=268435456
    )
  fi
  if [ "$mode" = terminal-discriminator ]; then
    capture_env+=(
      -e CANON_P38_TERMINAL_DISCRIMINATOR=1
      -e CANON_P38_TERMINAL_MAX_BYTES=4294967296
    )
  fi
fi

set +e
started="$(date +%s)"
timeout --foreground --signal=TERM --kill-after=120s "${timeout_seconds}s" \
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
  -e PYTHONPATH="$deps:$canon_out:$repo" -e PYTHONDONTWRITEBYTECODE=1 \
  -e CANON_SHIM_ROOT="$canon_out" \
  -e HF_HOME=/mnt/disks/tunix-data/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 \
  -e MODEL_DOWNLOAD_DIR="$model" -e FROZENLAKE_DATA_DIR="$data" \
  -e WANDB_MODE=disabled -e ROLLOUT_ENGINE=vllm -e NEW_MODEL_DESIGN=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e CANON_FROZENLAKE_L3=1 \
  -e CANON_ALIGNMENT_GATE=1 -e CANON_ALIGNMENT_GATE_ONLY=1 \
  -e CANON_ALIGNMENT_UPDATE_CANARY=0 -e CANON_ALIGNMENT_TRAIN=0 \
  -e CANON_PRE_ALIGN_GATE=1 -e CANON_PRE_ALIGN_REPORT="$pre_report" \
  -e CANON_P38_PRECHECK_ONLY=1 -e CANON_P38_CONTROLLED_EXIT=1 \
  -e CANON_P38_DIAGNOSTIC_ROUNDS=3 \
  -e CANON_P38_ONEHOST_REHEARSAL=1 \
  -e CANON_P38_DIAGNOSTIC_ROUND_FILE="$round_file" \
  -e CANON_P38_MISMATCH_CAPSULE="$capsule" \
  -e CANON_P38_MISMATCH_CAPSULE_MAX_ROWS=2 \
  -e CANON_DP_SIZE=1 -e CANON_TP_SIZE=4 -e CANON_TARGET_M=256 \
  -e CANON_P32_TRAIN_ADMITTED=0 -e FL_SHARED_MESH=1,4 \
  -e XLA_FLAGS="$XTRA_XLA" \
  -e CANON_RPA_D=128,512,128,512 -e CANON_RPA_P=128,512,128,512 \
  -e CANON_RPA_M=128,512,128,512 -e MIN_TOKEN_BUCKET=256 \
  -e CANON_FIXED_AR=1 -e CANON_FIXED_AR_EMBED=1 \
  -e CANON_RPA_VJP2=1 -e CANON_VJP2_MAX_SEQS=1 \
  -e CANON_ENGINE_MODULE_C=1 -e CANON_PROMPT_PROCESSED_LOGPROBS=1 \
  -e CANON_LOGPROB_M=256 -e CANON_PALLAS_LOGSOFTMAX=1 \
  -e CANON_PALLAS_ALL_PROJ=1 -e CANON_PALLAS_ALL_RMSNORM=1 \
  -e CANON_PALLAS_SWIGLU=1 -e CANON_PALLAS_MPAD=1 \
  -e CANON_PALLAS_SWIGLU_MPAD=1 -e CANON_PALLAS_CANONICAL_VJP=1 \
  -e CANON_QWEN3_HIDDEN_SIZE=4096 -e CANON_QWEN3_INTERMEDIATE_SIZE=12288 \
  -e CANON_QWEN3_NUM_ATTENTION_HEADS=32 -e CANON_QWEN3_NUM_KV_HEADS=8 \
  -e CANON_QWEN3_HEAD_DIM=128 -e CANON_QWEN3_TP_SIZE=4 \
  -e CANON_EXPECT_MODEL_MESH_IDS=0,2,1,3 \
  "${capture_env[@]}" \
  -w "$repo" "$image" bash -lc "
    set +e
    python3 -u examples/frozenlake/train_frozenlake_qwen3.py \\
      --batch_size=2 --mini_batch_size=2 \\
      --train_trajectory_micro_batch_size=4 \\
      --max_steps=1 --num_generations=2 --num_batches=3 \\
      --max_prompt_length=4096 --max_response_length=512 \\
      --max_concurrency=4 --vllm_max_num_seqs=4 \\
      --vllm_max_num_batched_tokens=256 --env_max_steps=5 \\
      --temperature=0.7 --top_k=0 --top_p=1.0
    workload_rc=\$?
    chmod a+r "$pre_report" "$round_file" || exit 97
    if [ -d "$capture" ]; then
      chmod -R a+rX "$capture" || exit 97
    fi
    for capsule_path in "$state"/mismatch*.npz; do
      [ ! -e "\$capsule_path" ] || chmod a+r "\$capsule_path" || exit 97
    done
    exit \$workload_rc
  " >> "$raw" 2>&1
docker_rc=$?
elapsed=$(( $(date +%s) - started ))
set -e

echo "[P38.INCIDENT.ONEHOST] docker_exit=$docker_rc elapsed_seconds=$elapsed" >> "$raw"
test "$docker_rc" -eq 42
round_markers="$(grep -ac '^\[CANON_P38\] PRECHECK_ROUND_COMPLETE ' "$raw" || true)"
backward_markers="$(grep -aEc 'backward(_begin|=1)|optimizer_commits=[1-9]' "$raw" || true)"
test "$round_markers" -eq 3
test "$backward_markers" -eq 0
test "$(tr -d '[:space:]' < "$round_file")" = 2
test -s "$pre_report"
if [ "$mode" = on ]; then
  test -s "$capture/p38_incident_ledger.jsonl"
  grep -q '^\[CANON_P38_INCIDENT_LEDGER\]' "$raw"
  python3 - "$capture/p38_incident_ledger.jsonl" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text().splitlines()]
assert rows
assert {row["diagnostic_round"] for row in rows} <= {0, 1, 2}
assert all(row["requests"] for row in rows)
print(f"[P38.INCIDENT.ONEHOST] LEDGER_PASS records={len(rows)} bytes={path.stat().st_size}")
PY
  python3 - "$capture" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

root = pathlib.Path(sys.argv[1])
records = [json.loads(path.read_text()) for path in sorted(
    root.glob("p38_kv_observer_*.json"))]
assert records, "P38 one-host observer produced no records"
assert {record["arm"] for record in records} == {"A", "B"}
by_a = {record["record_index"]: record for record in records
        if record["arm"] == "A"}
by_b = [record for record in records if record["arm"] == "B"]
assert len(by_a) == len(by_b), (len(by_a), len(by_b))
for record in records:
  npz = root / f"p38_kv_observer_{record['record_index']:04d}_{record['arm'].lower()}.npz"
  assert hashlib.sha256(npz.read_bytes()).hexdigest() == record["npz_sha256"]
  with np.load(npz) as arrays:
    assert set(arrays.files) == set(record["array_keys"])
for record in records:
  if record["arm"] != "B":
    continue
  source = by_a[record["source_a_record_index"]]
  assert record["source_a_request_id"] == source["request_id"]
  assert record["token_history_sha256"] == source["token_history_sha256"]
  assert record["target_seq_len"] == source["target_seq_len"]
  a_path = root / f"p38_kv_observer_{source['record_index']:04d}_a.npz"
  b_path = root / f"p38_kv_observer_{record['record_index']:04d}_b.npz"
  with np.load(a_path) as a, np.load(b_path) as b:
    np.testing.assert_array_equal(a["token_ids"], b["token_ids"])
    np.testing.assert_array_equal(a["valid_tokens"], b["valid_tokens"])
print(f"[P38.INCIDENT.ONEHOST] KV_OBSERVER_PASS records={len(records)} A={len(by_a)} B={len(by_b)}")
PY
  python3 \
    "$pkg/tasks/p38-pathways-decode-prefill-carrier/scripts/classify_p38_kv_observer.py" \
    --directory "$capture" \
    --output "$state/p38_kv_observer.classification.json"
  grep -q '"classification": "observer_pairs_valid_red_join_pending"' \
    "$state/p38_kv_observer.classification.json"
elif [ "$mode" = seam-layer ] || [ "$mode" = seam-tail ] || \
     [ "$mode" = terminal-discriminator ]; then
  grep -q '^\[CANON_P38_SEAM_OBSERVER_INIT\] enabled=1 mode=layer ' "$raw"
  grep -q '^\[CANON_P38_SEAM_OBSERVER_RECORD\] ' "$raw"
  python3 - "$capture" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

root = pathlib.Path(sys.argv[1])
records = [json.loads(path.read_text()) for path in sorted(
    root.glob("p38_seam_*.json"))]
assert records, "P38 one-host seam observer produced no records"
rounds = {record["diagnostic_round"] for record in records}
assert rounds and rounds <= {0, 1, 2}, rounds
for record in records:
  assert record["observer_mode"] == "layer"
  assert record["checkpoint_names"] == ["layer_input", "layer_output"]
  assert record["observed_layer"] is None
  npz = root / f"p38_seam_{record['record_index']:06d}.npz"
  assert hashlib.sha256(npz.read_bytes()).hexdigest() == record["npz_sha256"]
  with np.load(npz) as arrays:
    assert set(arrays.files) == set(record["array_keys"])
    values = arrays["layer_fingerprints"]
    assert values.ndim == 4
    assert values.shape[1:] == (36, 2, 8), values.shape
print(
    "[P38.INCIDENT.ONEHOST] SEAM_LAYER_PASS "
    f"records={len(records)} observed_rounds={sorted(rounds)} endpoint_rounds=3"
)
PY
  if [ "$mode" = seam-tail ] || [ "$mode" = terminal-discriminator ]; then
    grep -q '^\[CANON_P38_TAIL_OBSERVER_INIT\] enabled=1 ' "$raw"
    grep -q '^\[CANON_P38_TAIL_OBSERVER_RECORD\] ' "$raw"
    python3 - "$capture" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

root = pathlib.Path(sys.argv[1])
records = [json.loads(path.read_text()) for path in sorted(
    root.glob("p38_tail_*.json"))]
assert records, "P38 one-host terminal-tail observer produced no records"
assert {record["arm"] for record in records} == {"A", "B"}
rounds = {record["diagnostic_round"] for record in records}
assert rounds and rounds <= {0, 1, 2}, rounds
for record in records:
  assert record["schema"] == "p38-tail-values-v1"
  npz = root / f"p38_tail_{record['record_index']:06d}.npz"
  assert hashlib.sha256(npz.read_bytes()).hexdigest() == record["npz_sha256"]
  with np.load(npz) as arrays:
    assert set(arrays.files) == set(record["array_keys"])
    assert arrays["tail_values"].shape[1] == 6
print(
    "[P38.INCIDENT.ONEHOST] TERMINAL_TAIL_PASS "
    f"records={len(records)} observed_rounds={sorted(rounds)}"
)
PY
  fi
  if [ "$mode" = terminal-discriminator ]; then
    grep -q '^\[CANON_P38_TERMINAL_DISCRIMINATOR_INIT\] enabled=1 ' "$raw"
    grep -q '^\[CANON_P38_TERMINAL_DISCRIMINATOR_RECORD\] ' "$raw"
    classification="$evidence/p38_terminal_${label}.classification.json"
    python3 "$script_dir/classify_p38_terminal_discriminator.py" \
      --input "$capture" --output "$classification"
    python3 - "$capture" "$classification" <<'PY'
import hashlib, json, pathlib, sys
import numpy as np

root = pathlib.Path(sys.argv[1])
classification = json.loads(pathlib.Path(sys.argv[2]).read_text())
assert classification["records"] > 0
assert classification["joined_rows"] > 0
assert classification["classification"] == "terminal_rows_exact", classification
assert classification["missing_b_rows"] == 0, classification
assert classification["red_rows"] == [], classification
records = [json.loads(path.read_text()) for path in sorted(
    root.glob("p38_terminal_*.json"))]
assert {record["arm"] for record in records} == {"A", "B"}
for record in records:
  assert record["schema"] == "p38-terminal-discriminator-v1"
  assert record["reduction_program"] == "shared-fixed-four-row-v1"
  npz = root / f"p38_terminal_{record['record_index']:06d}.npz"
  assert hashlib.sha256(npz.read_bytes()).hexdigest() == record["npz_sha256"]
  with np.load(npz) as arrays:
    assert arrays["final_hidden_rows"].ndim == 2
    assert arrays["raw_logit_signatures"].shape[-1] == 6
    assert arrays["raw_block_exp_sum"].shape == arrays["raw_block_max"].shape
    assert arrays["processed_logit_signatures"].shape[-1] == 6
    assert arrays["processed_block_exp_sum"].shape == arrays["processed_block_max"].shape
print(
    "[P38.INCIDENT.ONEHOST] TERMINAL_DISCRIMINATOR_PASS "
    f"records={len(records)} joins={classification['joined_rows']} "
    f"classification={classification['classification']}"
)
PY
  fi
fi
sha256sum "$raw" "$pre_report" "$round_file" "$0"
echo "[P38.INCIDENT.ONEHOST] PASS mode=$mode rounds=3 backward=0 optimizer_commits=0"
