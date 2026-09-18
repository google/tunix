#!/usr/bin/env python3
"""Reduce immutable M15 Attempt-2 evidence into a replay-readiness receipt."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys
from typing import Any


ALIGN_PREFIX = "[CANON_" "ALIGN_PRE_JSON] "
CACHE_METRIC_RE = re.compile(
    r"Running: (?P<running>\d+) reqs, Waiting: (?P<waiting>\d+) reqs, "
    r"GPU KV cache usage: (?P<usage>[0-9.]+)%, Prefix cache hit rate: "
    r"(?P<hit>[0-9.]+)%"
)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(evidence_dir: pathlib.Path) -> dict[str, Any]:
    manifest = evidence_dir / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"missing manifest: {manifest}")
    entries = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, name = raw.split(maxsplit=1)
        name = name.lstrip("*")
        path = evidence_dir / name
        actual = sha256_file(path) if path.is_file() else None
        entries.append(
            {
                "name": name,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "valid": actual == expected,
            }
        )
    return {
        "path": str(manifest),
        "entries": entries,
        "valid": bool(entries) and all(entry["valid"] for entry in entries),
    }


def counter_dict(values: list[Any]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(collections.Counter(values).items())
    }


def summarize_mismatches(
    boundary: dict[str, Any], num_generations: int | None
) -> dict[str, Any]:
    mismatches = boundary["mismatches"]
    if boundary.get("mismatches_truncated"):
        raise ValueError("alignment mismatch list is truncated")
    if len(mismatches) != boundary["differing_elements"]:
        raise ValueError(
            "mismatch count disagrees with differing_elements: "
            f"{len(mismatches)} != {boundary['differing_elements']}"
        )

    row_summaries = []
    for row, records_iter in sorted(
        collections.defaultdict(list, {
            key: [item for item in mismatches if item["sequence_row"] == key]
            for key in {item["sequence_row"] for item in mismatches}
        }).items()
    ):
        records = list(records_iter)
        row_summary = {
                "sequence_row": row,
                "mismatch_count": len(records),
                "prompt_lengths": sorted({item["prompt_length"] for item in records}),
                "turns": sorted({item["turn_index"] for item in records}),
                "completion_position_min": min(item["completion_position"] for item in records),
                "completion_position_max": max(item["completion_position"] for item in records),
                "logical_kv_prefix_min": min(item["logical_kv_prefix_length"] for item in records),
                "logical_kv_prefix_max": max(item["logical_kv_prefix_length"] for item in records),
                "action_run_start_count": sum(bool(item["action_run_start"]) for item in records),
                "action_run_end_count": sum(bool(item["action_run_end"]) for item in records),
                "max_abs": max(item["abs_delta"] for item in records),
            }
        if num_generations:
            row_summary["prompt_group"] = row // num_generations
            row_summary["generation_index"] = row % num_generations
        row_summaries.append(row_summary)

    abs_values = [float(item["abs_delta"]) for item in mismatches]
    amplitude_bins = {
        "le_1e-6": sum(value <= 1e-6 for value in abs_values),
        "gt_1e-6_le_1e-4": sum(1e-6 < value <= 1e-4 for value in abs_values),
        "gt_1e-4_le_1e-2": sum(1e-4 < value <= 1e-2 for value in abs_values),
        "gt_1e-2_le_1e-1": sum(1e-2 < value <= 1e-1 for value in abs_values),
        "gt_1e-1": sum(value > 1e-1 for value in abs_values),
    }
    logical_prefixes = [item["logical_kv_prefix_length"] for item in mismatches]
    return {
        "reported_elements": boundary["differing_elements"],
        "reported_bytes": boundary["differing_bytes"],
        "reported_max_abs": boundary["max_abs"],
        "records_present": len(mismatches),
        "records_complete": len(mismatches) == boundary["differing_elements"],
        "first_mismatch": boundary["first_mismatch"],
        "max_abs_mismatch": boundary["max_abs_mismatch"],
        "sequence_rows": counter_dict([item["sequence_row"] for item in mismatches]),
        "prompt_groups": counter_dict(
            [item["sequence_row"] // num_generations for item in mismatches]
        )
        if num_generations
        else None,
        "generation_indices": counter_dict(
            [item["sequence_row"] % num_generations for item in mismatches]
        )
        if num_generations
        else None,
        "turns": counter_dict([item["turn_index"] for item in mismatches]),
        "prompt_lengths": counter_dict([item["prompt_length"] for item in mismatches]),
        "completion_chunks": counter_dict([item["completion_chunk_index"] for item in mismatches]),
        "logical_256_blocks": counter_dict([value // 256 for value in logical_prefixes]),
        "logical_256_offsets": counter_dict([value % 256 for value in logical_prefixes]),
        "exact_256_boundary_count": sum(value % 256 == 0 for value in logical_prefixes),
        "action_run_start_count": sum(bool(item["action_run_start"]) for item in mismatches),
        "action_run_end_count": sum(bool(item["action_run_end"]) for item in mismatches),
        "previous_token_environment_count": sum(
            bool(item["previous_token_is_environment"]) for item in mismatches
        ),
        "logical_kv_prefix_min": min(logical_prefixes),
        "logical_kv_prefix_max": max(logical_prefixes),
        "amplitude_bins": amplitude_bins,
        "row_summaries": row_summaries,
    }


def decode_log(log_path: pathlib.Path) -> dict[str, Any]:
    align_records: list[tuple[int, dict[str, Any]]] = []
    cache_metrics: list[dict[str, Any]] = []
    marker_lines: dict[str, list[int]] = collections.defaultdict(list)
    evidence_path = None
    run_command = None
    engine_config_line = None
    engine_model = None
    source_describe = None
    source_head = None
    resolved_model_revision = None
    engine_target_path_sha256 = None
    num_generations = None

    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, raw in enumerate(stream, 1):
            line = raw.rstrip("\n")
            if ALIGN_PREFIX in line:
                payload = line.split(ALIGN_PREFIX, 1)[1]
                align_records.append((line_number, json.loads(payload)))
            if "[CANON_" "ALIGN_PRE_EVIDENCE]" in line:
                marker_lines["align_evidence"].append(line_number)
                match = re.search(r"path=(\S+)\s+sha256=([0-9a-f]{64})", line)
                if match:
                    evidence_path = {
                        "path": match.group(1),
                        "sha256": match.group(2),
                        "line": line_number,
                    }
            if "Successfully reset prefix cache" in line:
                marker_lines["prefix_cache_reset"].append(line_number)
            if "enable_prefix_caching' with value 'True'" in line:
                marker_lines["apc_enabled_writer"].append(line_number)
            if "enable_prefix_caching=True" in line:
                marker_lines["apc_enabled_engine"].append(line_number)
                if engine_config_line is None:
                    engine_config_line = line_number
                    model_match = re.search(r"model='([^']+)'", line)
                    engine_model = model_match.group(1) if model_match else None
            if "[VLLM.LOGPROB_REQUEST]" in line:
                marker_lines["logprob_request"].append(line_number)
            if "[run] cmd:" in line:
                run_command = {"line": line_number, "text": line.split("[run] cmd:", 1)[1].strip()}
                generations_match = re.search(r"--num_generations=(\d+)", line)
                num_generations = int(generations_match.group(1)) if generations_match else None
            if "[sync] describe=" in line:
                source_describe = {"line": line_number, "text": line.split("describe=", 1)[1]}
            if "[sync] HEAD=" in line:
                source_head = {
                    "line": line_number,
                    "sha": line.split("[sync] HEAD=", 1)[1].strip(),
                }
            revision_match = re.search(r"/([0-9a-f]{40})/(?:config|tokenizer_config)\.json", line)
            if revision_match:
                resolved_model_revision = revision_match.group(1)
            target_match = re.search(r"'target_path_sha256': '([0-9a-f]{64})'", line)
            if target_match:
                engine_target_path_sha256 = target_match.group(1)
            metric_match = CACHE_METRIC_RE.search(line)
            if metric_match:
                cache_metrics.append(
                    {
                        "line": line_number,
                        "running": int(metric_match.group("running")),
                        "waiting": int(metric_match.group("waiting")),
                        "usage_percent": float(metric_match.group("usage")),
                        "hit_rate_percent": float(metric_match.group("hit")),
                    }
                )

    if len(align_records) != 1:
        raise ValueError(
            f"expected one alignment-pre JSON record, found {len(align_records)}"
        )
    align_line, alignment = align_records[0]
    ab = alignment["boundaries"]["S_decode_vs_S_prefill"]
    bc = alignment["boundaries"]["S_prefill_vs_T_old"]
    if alignment["verdict"] != "FAIL" or bc["differing_bytes"] != 0:
        raise ValueError("m15i no longer matches the frozen A-B-red/B-C-exact contract")

    cache_summary = {
        "records": len(cache_metrics),
        "first_line": cache_metrics[0]["line"] if cache_metrics else None,
        "last_line": cache_metrics[-1]["line"] if cache_metrics else None,
        "hit_rate_percent_min": min(item["hit_rate_percent"] for item in cache_metrics)
        if cache_metrics
        else None,
        "hit_rate_percent_max": max(item["hit_rate_percent"] for item in cache_metrics)
        if cache_metrics
        else None,
        "running_max": max(item["running"] for item in cache_metrics) if cache_metrics else None,
        "waiting_max": max(item["waiting"] for item in cache_metrics) if cache_metrics else None,
        "usage_percent_max": max(item["usage_percent"] for item in cache_metrics)
        if cache_metrics
        else None,
    }
    return {
        "alignment_line": align_line,
        "alignment": {
            "step": alignment["step"],
            "verdict": alignment["verdict"],
            "n_action": alignment["N_action"],
            "action_geometry": alignment["action_geometry"],
            "context": alignment["context"],
            "hashes": alignment["hashes"],
            "masked_hashes": alignment["masked_hashes"],
            "a_minus_b": summarize_mismatches(ab, num_generations),
            "b_minus_c": {
                "differing_elements": bc["differing_elements"],
                "differing_bytes": bc["differing_bytes"],
                "max_abs": bc["max_abs"],
                "finite": bc["finite"],
            },
        },
        "runtime": {
            "run_command": run_command,
            "source_describe": source_describe,
            "source_head": source_head,
            "engine_model": engine_model,
            "engine_config_line": engine_config_line,
            "resolved_model_revision": resolved_model_revision,
            "engine_target_path_sha256": engine_target_path_sha256,
            "num_generations": num_generations,
            "marker_lines": {key: value for key, value in sorted(marker_lines.items())},
            "cache_metrics": cache_summary,
            "ephemeral_alignment_evidence": evidence_path,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    evidence_dir = args.evidence_dir.resolve()
    receipt_path = evidence_dir / "receipt.json"
    log_path = evidence_dir / "m15_m15i_error.log"
    if not receipt_path.is_file() or not log_path.is_file():
        raise ValueError("evidence directory lacks receipt.json or m15_m15i_error.log")

    manifest = verify_manifest(evidence_dir)
    if not manifest["valid"]:
        raise ValueError("SHA256SUMS verification failed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    decoded = decode_log(log_path)

    receipt_source = receipt.get("source_commit")
    runtime_source = decoded["runtime"].get("source_head")
    runtime_source_sha = runtime_source.get("sha") if runtime_source else None
    source_identity = {
        "receipt_source_commit": receipt_source,
        "runtime_source_commit": runtime_source_sha,
        "runtime_source_line": runtime_source.get("line") if runtime_source else None,
        "equal": bool(
            isinstance(receipt_source, str)
            and isinstance(runtime_source_sha, str)
            and receipt_source == runtime_source_sha
        ),
    }
    if not source_identity["equal"]:
        source_identity["verdict"] = "PROVENANCE_CONTRADICTION"
        source_identity["authority"] = (
            "runtime HEAD is authoritative because 10_sync_repo.sh would abort "
            "when CANON_EXPECT_COMMIT differs from the checked-out HEAD"
        )
    else:
        source_identity["verdict"] = "EXACT"

    durable_files = sorted(path.name for path in evidence_dir.iterdir() if path.is_file())
    raw_array_suffixes = {".npy", ".npz", ".safetensors"}
    durable_raw_arrays = [
        name for name in durable_files if pathlib.Path(name).suffix in raw_array_suffixes
    ]
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    field_presence = {
        "full_tokens_array": bool(durable_raw_arrays),
        "full_action_mask_array": bool(durable_raw_arrays),
        "full_a_b_vectors": bool(durable_raw_arrays),
        "request_ids_or_order": "request_id" in log_text,
        "token_history": "token_history" in log_text,
        "block_table": "block_table" in log_text or "block table" in log_text.lower(),
        "num_computed_tokens": "num_computed_tokens" in log_text,
        "per_request_cached_tokens": "num_cached_tokens" in log_text,
        "page_owner_generation_hash": any(
            token in log_text for token in ("page_owner", "page_generation", "kv_page_hash")
        ),
    }
    missing = [name for name, present in field_presence.items() if not present]

    report = {
        "schema": "m15-first-red-input-contract-v1",
        "evidence_dir": str(evidence_dir),
        "manifest": manifest,
        "receipt": receipt,
        "source_identity": source_identity,
        "decoded": decoded,
        "durable_artifacts": {
            "files": durable_files,
            "raw_array_files": durable_raw_arrays,
        },
        "strict_replay": {
            "field_presence": field_presence,
            "missing_fields": missing,
            "verdict": "READY" if not missing else "INSUFFICIENT_FOR_STRICT_REPLAY",
            "reason": (
                "All required raw replay fields are durable."
                if not missing
                else "Hashes and mismatch coordinates do not reconstruct full token histories, "
                "request chronology, or cache lineage."
            ),
        },
        "claim_ceiling": "PHASE_A_IMMUTABLE_EVIDENCE_AUDIT",
    }

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(
            "M15_FIRST_RED_INPUT_CONTRACT "
            f"manifest=PASS mismatches={decoded['alignment']['a_minus_b']['reported_elements']} "
            f"replay={report['strict_replay']['verdict']} output={args.output}"
        )
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
