#!/usr/bin/env python3
"""Create and durably mirror immutable P57 TiTO evidence snapshots."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from typing import Any


_GCS_PREFIX_RE = re.compile(
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
    r"[a-z0-9](?:[-a-z0-9]{0,62}[a-z0-9])?/attempt-(?:direct|[0-9]+)"
)
_LIVE_EVIDENCE_PATTERNS = (
    "p57_tito_witness/single-writer.json",
    "p57_tito_witness/host/host-request-*.json",
    "p57_tito_witness/runner/runner-input-*.json",
    "p57_tito_witness/update-sidecars/step-*.npz",
    "p57_tito_witness/actor-snapshot-requests/step-*.json",
    "p57_tito_witness/actor-snapshot-receipts/step-*.json",
    "token-continuity-first-diff/*.json",
    "p57_tito_gcs/orbax-probe.json",
    "p57_tito_gcs/journal-deltas/**/*.jsonl",
)
_FINAL_EVIDENCE_PATTERNS = (
    "p57_tito_witness/diagnostic-summary.json",
    "p57_tito_witness/full-record-summary.json",
    "p57_tito_witness/full-row-map.jsonl",
    "p57_tito_collection.classification.json",
    "p57_tito_full_record.classification.json",
    "p33_frozenlake-dp8-tp8_full.classification.json",
    "v1_hp_p45_full.classification.json",
    "v1_hp_m15_full.classification.json",
    "pre_alignment.jsonl",
    "alignment.jsonl",
    "updates.jsonl",
    "p57_tito_witness/journal-reconstruction.json",
)
_APPEND_JOURNALS = (
    "p57_tito_witness/full-row-map.jsonl",
    "pre_alignment.jsonl",
    "alignment.jsonl",
    "updates.jsonl",
)
_JOURNAL_CHUNK_RE = re.compile(
    r"chunk-(?P<start>[0-9]{12})-(?P<end>[0-9]{12})-"
    r"(?P<sha>[0-9a-f]{64})\.jsonl"
)


def _sha256_bytes(payload: bytes) -> str:
  return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _atomic_json(
    path: Path, record: dict[str, Any], *, accept_identical: bool = False
) -> None:
  payload = (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()
  if path.exists():
    if accept_identical and path.read_bytes() == payload:
      return
    raise FileExistsError(f"refusing to overwrite TiTO sync receipt: {path}")
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  partial = path.with_name(f".{path.name}.partial-{os.getpid()}")
  descriptor = os.open(
      partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.link(partial, path)
    partial.unlink()
  except BaseException:
    try:
      partial.unlink()
    except FileNotFoundError:
      pass
    raise


def _safe_relative(state: Path, path: Path) -> str:
  relative = path.relative_to(state).as_posix()
  parsed = PurePosixPath(relative)
  if (
      parsed.is_absolute()
      or not parsed.parts
      or any(part in ("", ".", "..") for part in parsed.parts)
  ):
    raise ValueError(f"unsafe TiTO evidence path: {relative!r}")
  return relative


def _journal_slug(relative: str) -> str:
  return relative.replace("/", "__").removesuffix(".jsonl")


def _complete_jsonl_end(payload: bytes) -> int:
  """Returns the byte boundary after the final complete JSONL record."""
  return payload.rfind(b"\n") + 1


def _existing_journal_chunks(
    state: Path, relative: str, source_payload: bytes
) -> tuple[int, list[dict[str, Any]]]:
  """Validates immutable chunks and returns their contiguous covered prefix."""
  root = state / "p57_tito_gcs" / "journal-deltas" / _journal_slug(relative)
  chunks: list[dict[str, Any]] = []
  cursor = 0
  for path in sorted(root.glob("chunk-*.jsonl")):
    match = _JOURNAL_CHUNK_RE.fullmatch(path.name)
    if match is None or path.is_symlink() or not path.is_file():
      raise ValueError(f"invalid TiTO journal chunk: {path}")
    start = int(match.group("start"))
    end = int(match.group("end"))
    payload = path.read_bytes()
    if (
        start != cursor
        or end <= start
        or end > len(source_payload)
        or len(payload) != end - start
        or not payload.endswith(b"\n")
        or payload != source_payload[start:end]
        or _sha256_bytes(payload) != match.group("sha")
        or path.stat().st_mode & 0o077
    ):
      raise ValueError(f"TiTO journal chunk chain differs: {path}")
    chunks.append({
        "path": _safe_relative(state, path),
        "start": start,
        "end": end,
        "bytes": len(payload),
        "sha256": match.group("sha"),
    })
    cursor = end
  return cursor, chunks


def materialize_journal_deltas(
    state: Path, *, final: bool
) -> dict[str, Any]:
  """Copies complete JSONL records into immutable, live-uploadable chunks."""
  state = state.resolve()
  journals = []
  for relative in _APPEND_JOURNALS:
    source = state / relative
    if not source.exists():
      if final:
        raise ValueError(f"final TiTO journal is absent: {relative}")
      continue
    if source.is_symlink() or not source.is_file():
      raise ValueError(f"TiTO journal must be a regular file: {relative}")
    payload = source.read_bytes()
    complete_end = _complete_jsonl_end(payload)
    cursor, chunks = _existing_journal_chunks(state, relative, payload)
    if cursor > complete_end:
      raise ValueError(f"TiTO journal lost a previously complete line: {relative}")
    if complete_end > cursor:
      delta = payload[cursor:complete_end]
      digest = _sha256_bytes(delta)
      root = (
          state / "p57_tito_gcs" / "journal-deltas" / _journal_slug(relative)
      )
      root.mkdir(parents=True, exist_ok=True, mode=0o700)
      output = root / (
          f"chunk-{cursor:012d}-{complete_end:012d}-{digest}.jsonl"
      )
      partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
      descriptor = os.open(
          partial,
          os.O_WRONLY | os.O_CREAT | os.O_EXCL,
          0o600,
      )
      try:
        with os.fdopen(descriptor, "wb") as target:
          target.write(delta)
          target.flush()
          os.fsync(target.fileno())
        os.link(partial, output)
        partial.unlink()
      except BaseException:
        try:
          os.close(descriptor)
        except OSError:
          pass
        partial.unlink(missing_ok=True)
        raise
      cursor, chunks = _existing_journal_chunks(state, relative, payload)
    if final and (complete_end != len(payload) or cursor != len(payload)):
      raise ValueError(f"final TiTO journal is not fully reconstructable: {relative}")
    journals.append({
        "source": relative,
        "complete_bytes": complete_end,
        "source_bytes": len(payload),
        "source_sha256": _sha256_bytes(payload) if final else None,
        "chunks": chunks,
    })
  report = {
      "schema": "canon.p57-tito-journal-reconstruction.v1",
      "status": "PASS",
      "final": final,
      "journals": journals,
  }
  if final:
    if len(journals) != len(_APPEND_JOURNALS):
      raise ValueError("final TiTO journal set is incomplete")
    _atomic_json(
        state / "p57_tito_witness" / "journal-reconstruction.json",
        report,
        accept_identical=True,
    )
  return report


def evidence_inventory(
    state: Path,
    *,
    final: bool,
    prior_records: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
  """Returns the immutable files admitted to one evidence snapshot.

  Live polling reuses the SHA of a previously uploaded immutable file after
  checking its path, mode, and size. This keeps the 30-second observer cost
  proportional to newly published evidence instead of repeatedly hashing all
  earlier multi-GB update sidecars. Finalization deliberately re-hashes every
  local file and every delta tar before issuing the terminal manifest.
  """
  state = state.resolve()
  prior_records = prior_records or {}
  if not state.is_dir():
    raise ValueError(f"TiTO state directory is absent: {state}")
  paths: set[Path] = set()
  patterns = _LIVE_EVIDENCE_PATTERNS
  if final:
    patterns += _FINAL_EVIDENCE_PATTERNS
  for pattern in patterns:
    paths.update(state.glob(pattern))
  records = []
  for path in sorted(paths, key=lambda item: item.as_posix()):
    if path.is_symlink() or not path.is_file():
      raise ValueError(f"TiTO evidence must be a regular non-symlink: {path}")
    resolved_path = path.resolve(strict=True)
    if resolved_path != path.absolute() or not resolved_path.is_relative_to(state):
      raise ValueError(f"TiTO evidence escapes its state directory: {path}")
    relative = _safe_relative(state, path)
    if (
        relative.startswith("p57_tito_witness/")
        or relative.startswith("token-continuity-first-diff/")
        or relative.startswith("p57_tito_gcs/journal-deltas/")
    ) and path.stat().st_mode & 0o077:
      raise ValueError(f"raw TiTO evidence is not mode 0600: {path}")
    payload_bytes = path.stat().st_size
    if payload_bytes <= 0:
      raise ValueError(f"TiTO evidence is empty: {relative}")
    prior = prior_records.get(relative)
    if (
        not final
        and prior is not None
        and prior.get("path") == relative
        and prior.get("bytes") == payload_bytes
        and isinstance(prior.get("sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", prior["sha256"]) is not None
    ):
      records.append(dict(prior))
    else:
      records.append({
          "path": relative,
          "bytes": payload_bytes,
          "sha256": _sha256_file(path),
      })
  if final:
    observed = {record["path"] for record in records}
    full = "p57_tito_witness/full-record-summary.json" in observed
    required = (
        {
            "p57_tito_witness/single-writer.json",
            "p57_tito_witness/full-record-summary.json",
            "p57_tito_witness/full-row-map.jsonl",
            "p57_tito_full_record.classification.json",
            "p33_frozenlake-dp8-tp8_full.classification.json",
            "pre_alignment.jsonl",
            "alignment.jsonl",
            "updates.jsonl",
            "p57_tito_witness/journal-reconstruction.json",
            "p57_tito_gcs/orbax-probe.json",
        }
        if full
        else {
            "p57_tito_witness/diagnostic-summary.json",
            "p57_tito_collection.classification.json",
        }
    )
    if full:
      summary = json.loads(
          (state / "p57_tito_witness/full-record-summary.json").read_text(
              encoding="utf-8"
          )
      )
      expected_capsules = summary.get("collection", {}).get(
          "token_difference_events"
      )
      observed_capsules = sum(
          path.startswith("token-continuity-first-diff/")
          for path in observed
      )
      if (
          type(expected_capsules) is not int
          or expected_capsules < 0
          or observed_capsules != expected_capsules
      ):
        raise ValueError(
            "full TiTO token-difference inventory differs: "
            f"observed={observed_capsules} expected={expected_capsules!r}"
        )
      v1 = {
          "v1_hp_p45_full.classification.json",
          "v1_hp_m15_full.classification.json",
      } & observed
      if len(v1) != 1:
        raise ValueError("full TiTO evidence requires exactly one V1 classifier")
    missing = required - observed
    if missing:
      raise ValueError(f"final TiTO evidence is incomplete: {sorted(missing)}")
  return records


def _manifest_payload(records: list[dict[str, Any]]) -> bytes:
  return "".join(
      f"{record['sha256']}  {record['bytes']}  {record['path']}\n"
      for record in records
  ).encode()


def _verify_snapshot(
    path: Path,
    records: list[dict[str, Any]],
    manifest: bytes,
) -> None:
  """Re-hashes every regular member in one immutable evidence snapshot."""
  expected = {record["path"]: record for record in records}
  expected_names = {"SHA256SUMS", *expected}
  with tarfile.open(path, mode="r:") as archive:
    members = archive.getmembers()
    names = [member.name for member in members]
    if len(names) != len(set(names)) or set(names) != expected_names:
      raise ValueError(f"TiTO snapshot member set differs: {path}")
    for member in members:
      if not member.isfile() or member.mode != 0o600:
        raise ValueError(f"TiTO snapshot member is not a mode-0600 file: {path}")
      payload = archive.extractfile(member)
      if payload is None:
        raise ValueError(f"TiTO snapshot member is unreadable: {path}")
      digest = hashlib.sha256()
      payload_bytes = 0
      for chunk in iter(lambda: payload.read(1024 * 1024), b""):
        digest.update(chunk)
        payload_bytes += len(chunk)
      if member.name == "SHA256SUMS":
        if payload_bytes != len(manifest) or digest.hexdigest() != _sha256_bytes(
            manifest
        ):
          raise ValueError(f"TiTO snapshot manifest differs: {path}")
        continue
      record = expected[member.name]
      if (
          payload_bytes != record["bytes"]
          or member.size != record["bytes"]
          or digest.hexdigest() != record["sha256"]
      ):
        raise ValueError(f"TiTO snapshot payload differs: {member.name}")


def create_snapshot(
    state: Path,
    records: list[dict[str, Any]],
    output: Path,
) -> tuple[str, str, int]:
  """Writes a deterministic mode-0600 tar with an internal SHA manifest."""
  if not records:
    raise ValueError("cannot snapshot an empty TiTO evidence inventory")
  output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  manifest = _manifest_payload(records)
  inventory_sha = _sha256_bytes(manifest)
  if output.exists():
    _verify_snapshot(output, records, manifest)
    return inventory_sha, _sha256_file(output), output.stat().st_size
  partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
  descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(descriptor, "wb") as raw:
      with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
        manifest_info = tarfile.TarInfo("SHA256SUMS")
        manifest_info.size = len(manifest)
        manifest_info.mode = 0o600
        manifest_info.mtime = 0
        manifest_info.uid = manifest_info.gid = 0
        manifest_info.uname = manifest_info.gname = ""
        archive.addfile(manifest_info, io.BytesIO(manifest))
        for record in records:
          source = state / record["path"]
          info = tarfile.TarInfo(record["path"])
          info.size = record["bytes"]
          info.mode = 0o600
          info.mtime = 0
          info.uid = info.gid = 0
          info.uname = info.gname = ""
          with source.open("rb") as payload:
            archive.addfile(info, payload)
      raw.flush()
      os.fsync(raw.fileno())
    os.link(partial, output)
    partial.unlink()
  except BaseException:
    try:
      partial.unlink()
    except FileNotFoundError:
      pass
    raise
  _verify_snapshot(output, records, manifest)
  return inventory_sha, _sha256_file(output), output.stat().st_size


def _parse_gcs_url(url: str) -> tuple[str, str]:
  if not url.startswith("gs://"):
    raise ValueError(f"invalid GCS URL: {url}")
  parts = url[5:].split("/", 1)
  return parts[0], parts[1] if len(parts) > 1 else ""


_GCS_CLIENT: Any = None


def _get_gcs_client() -> Any:
  global _GCS_CLIENT
  if _GCS_CLIENT is None:
    from google.cloud import storage  # pylint: disable=import-outside-toplevel

    _GCS_CLIENT = storage.Client()
  return _GCS_CLIENT


def _run_gcloud_cp(source: str, destination: str, *, no_clobber: bool) -> int:
  if shutil.which("gcloud"):
    command = ["gcloud", "storage", "cp"]
    if no_clobber:
      command.append("--no-clobber")
    command.extend((source, destination))
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode

  if shutil.which("gsutil"):
    command = ["gsutil", "-q", "cp"]
    if no_clobber:
      command.append("-n")
    command.extend((source, destination))
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode

  try:
    client = _get_gcs_client()
    if destination.startswith("gs://"):
      bucket_name, blob_name = _parse_gcs_url(destination)
      blob = client.bucket(bucket_name).blob(blob_name)
      if no_clobber:
        blob.upload_from_filename(source, if_generation_match=0)
      else:
        blob.upload_from_filename(source)
      return 0
    if source.startswith("gs://"):
      bucket_name, blob_name = _parse_gcs_url(source)
      blob = client.bucket(bucket_name).blob(blob_name)
      blob.download_to_filename(destination)
      return 0
    raise ValueError(
        f"neither source nor destination is gs://: {source} -> {destination}"
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return 1


def _upload_and_verify(
    local: Path,
    remote: str,
    expected_sha: str,
    *,
    attempts: int = 4,
) -> None:
  # A nonzero no-clobber return may mean that an identical retry already won.
  # The independent readback, not the upload exit code, is the final verdict.
  if attempts < 1:
    raise ValueError("TiTO GCS attempts must be positive")
  readback_root = local.parent / ".readback"
  readback_root.mkdir(parents=True, exist_ok=True, mode=0o700)
  last_error = "TiTO GCS transfer failed"
  for attempt in range(1, attempts + 1):
    # A nonzero no-clobber return may mean that an identical retry already won.
    # The independent readback, not the upload exit code, is the verdict.
    _run_gcloud_cp(str(local), remote, no_clobber=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{local.name}.", suffix=".partial", dir=readback_root
    )
    os.close(descriptor)
    readback = Path(temporary)
    readback.chmod(0o600)
    try:
      if _run_gcloud_cp(remote, str(readback), no_clobber=False) != 0:
        last_error = "TiTO GCS readback failed"
      elif _sha256_file(readback) != expected_sha:
        # A readable object under the no-clobber identity with different
        # content is permanent corruption, not a transient worth retrying.
        raise RuntimeError("TiTO GCS readback SHA256 differs")
      else:
        return
    finally:
      readback.unlink(missing_ok=True)
    if attempt < attempts:
      time.sleep(2 ** (attempt - 1))
  raise RuntimeError(f"{last_error} after {attempts} attempts")


def _validate_prefix(prefix: str) -> str:
  prefix = prefix.rstrip("/")
  if _GCS_PREFIX_RE.fullmatch(prefix) is None:
    raise ValueError("TiTO GCS prefix is outside the registered evidence root")
  return prefix


def probe_gcs(state: Path, prefix: str) -> dict[str, Any]:
  """Proves immutable write plus independent read access before collection."""
  state = state.resolve()
  if not state.is_dir():
    raise ValueError(f"TiTO state directory is absent: {state}")
  prefix = _validate_prefix(prefix)
  probe = state / "p57_tito_gcs" / "admission-probe.json"
  record = {
      "schema": "canon.p57-tito-gcs-admission-probe.v1",
      "status": "PASS",
      "content": "non-sensitive-deterministic-probe",
  }
  _atomic_json(probe, record, accept_identical=True)
  probe_sha = _sha256_file(probe)
  _upload_and_verify(probe, f"{prefix}/admission-probe.json", probe_sha)
  return {
      "status": "PASS",
      "probe_sha256": probe_sha,
      "readback_verified": True,
  }


def _load_receipts(
    state: Path, *, verify_snapshots: bool
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
  """Loads the immutable delta ledger and optionally re-hashes each delta."""
  sync_root = state / "p57_tito_gcs"
  uploaded: dict[str, dict[str, Any]] = {}
  summaries = []
  receipt_paths = sorted((sync_root / "receipts").glob("snapshot-*.json"))
  for sequence, path in enumerate(receipt_paths, 1):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    records = receipt.get("records")
    if (
        receipt.get("schema") != "canon.p57-tito-gcs-snapshot.v2"
        or receipt.get("status") != "PASS"
        or receipt.get("sequence") != sequence
        or receipt.get("kind") not in ("live", "final")
        or not isinstance(records, list)
        or not records
        or receipt.get("files") != len(records)
        or receipt.get("delta_inventory_sha256")
        != _sha256_bytes(_manifest_payload(records))
        or receipt.get("readback_verified") is not True
    ):
      raise ValueError(f"TiTO delta receipt differs: {path}")
    for record in records:
      relative = record.get("path")
      if not isinstance(relative, str):
        raise ValueError(f"TiTO delta receipt path differs: {path}")
      previous = uploaded.get(relative)
      if previous is not None and previous != record:
        raise ValueError(f"TiTO evidence identity changed: {relative}")
      if previous is not None:
        raise ValueError(f"TiTO evidence was uploaded twice: {relative}")
      uploaded[relative] = record
    snapshot_name = receipt.get("snapshot")
    snapshot_sha = receipt.get("snapshot_sha256")
    snapshot_bytes = receipt.get("snapshot_bytes")
    if (
        not isinstance(snapshot_name, str)
        or re.fullmatch(
            rf"snapshot-{sequence:06d}-(?:live|final)-[0-9a-f]{{16}}\.tar",
            snapshot_name,
        ) is None
        or not isinstance(snapshot_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", snapshot_sha) is None
        or type(snapshot_bytes) is not int
        or snapshot_bytes <= 0
    ):
      raise ValueError(f"TiTO delta snapshot identity differs: {path}")
    snapshot = sync_root / "snapshots" / snapshot_name
    if (
        not snapshot.is_file()
        or snapshot.stat().st_size != snapshot_bytes
        or (verify_snapshots and _sha256_file(snapshot) != snapshot_sha)
    ):
      raise ValueError(f"TiTO delta snapshot is missing or changed: {snapshot}")
    if verify_snapshots:
      _verify_snapshot(snapshot, records, _manifest_payload(records))
    summaries.append({
        "sequence": sequence,
        "kind": receipt["kind"],
        "receipt": path.name,
        "receipt_sha256": _sha256_file(path),
        "snapshot": snapshot_name,
        "snapshot_sha256": snapshot_sha,
        "records": records,
    })
  return uploaded, summaries


def _reuse_final(
    state: Path,
    prefix: str,
    records: list[dict[str, Any]],
    inventory_sha: str,
) -> dict[str, Any] | None:
  """Completes an interrupted final-manifest upload idempotently."""
  sync_root = state / "p57_tito_gcs"
  final_path = sync_root / "final-manifest.json"
  if not final_path.exists():
    return None
  uploaded, deltas = _load_receipts(state, verify_snapshots=True)
  final_record = json.loads(final_path.read_text(encoding="utf-8"))
  if (
      final_record.get("schema") != "canon.p57-tito-gcs-final-manifest.v2"
      or final_record.get("status") != "PASS"
      or final_record.get("files") != records
      or final_record.get("file_count") != len(records)
      or final_record.get("inventory_sha256") != inventory_sha
      or final_record.get("deltas") != deltas
      or uploaded != {record["path"]: record for record in records}
      or not deltas
      or deltas[-1]["kind"] != "final"
      or final_record.get("readback_verified") is not True
  ):
    raise ValueError("existing TiTO final manifest differs from current evidence")
  final_sha = _sha256_file(final_path)
  _upload_and_verify(final_path, f"{prefix}/final-manifest.json", final_sha)
  last_receipt = json.loads(
      (sync_root / "receipts" / deltas[-1]["receipt"]).read_text(
          encoding="utf-8"
      )
  )
  return {**last_receipt, "final_manifest_sha256": final_sha}


def sync_once(state: Path, prefix: str, *, final: bool) -> dict[str, Any]:
  """Uploads only new immutable records and proves a complete final ledger."""
  state = state.resolve()
  prefix = _validate_prefix(prefix)
  full_record = (
      state / "p57_tito_witness" / "full-record-summary.json"
  ).is_file()
  materialize_journal_deltas(state, final=final and full_record)
  uploaded, _ = _load_receipts(state, verify_snapshots=final)
  records = evidence_inventory(
      state,
      final=final,
      prior_records=uploaded if not final else None,
  )
  if not records and not final:
    return {"status": "EMPTY", "final": False, "files": 0}
  inventory_sha = _sha256_bytes(_manifest_payload(records)) if records else ""
  if final:
    reused = _reuse_final(state, prefix, records, inventory_sha)
    if reused is not None:
      return reused
  sync_root = state / "p57_tito_gcs"
  receipts = sync_root / "receipts"
  receipts.mkdir(parents=True, exist_ok=True, mode=0o700)
  observed = {record["path"]: record for record in records}
  for relative, prior_record in uploaded.items():
    if relative not in observed or observed[relative] != prior_record:
      raise ValueError(f"uploaded TiTO evidence changed or disappeared: {relative}")
  delta = [record for record in records if record["path"] not in uploaded]
  if not delta and not final:
    return {
        "status": "UNCHANGED",
        "final": False,
        "files": 0,
        "complete_files": len(records),
        "inventory_sha256": inventory_sha,
    }
  if not delta:
    raise ValueError("final TiTO evidence has no final-only delta")
  sequence = len(list(receipts.glob("snapshot-*.json"))) + 1
  kind = "final" if final else "live"
  delta_sha = _sha256_bytes(_manifest_payload(delta))
  snapshot_name = f"snapshot-{sequence:06d}-{kind}-{delta_sha[:16]}.tar"
  snapshot = sync_root / "snapshots" / snapshot_name
  observed_inventory_sha, snapshot_sha, snapshot_bytes = create_snapshot(
      state, delta, snapshot
  )
  if observed_inventory_sha != delta_sha:
    raise RuntimeError("TiTO delta changed during snapshot construction")
  _upload_and_verify(
      snapshot,
      f"{prefix}/snapshots/{snapshot_name}",
      snapshot_sha,
  )
  receipt = {
      "schema": "canon.p57-tito-gcs-snapshot.v2",
      "status": "PASS",
      "kind": kind,
      "sequence": sequence,
      "files": len(delta),
      "complete_files": len(records),
      "delta_inventory_sha256": delta_sha,
      "complete_inventory_sha256": inventory_sha,
      "records": delta,
      "snapshot": snapshot_name,
      "snapshot_sha256": snapshot_sha,
      "snapshot_bytes": snapshot_bytes,
      "readback_verified": True,
  }
  receipt_path = receipts / f"snapshot-{sequence:06d}-{delta_sha}.json"
  _atomic_json(receipt_path, receipt)
  if final:
    complete_uploaded, deltas = _load_receipts(
        state, verify_snapshots=True
    )
    if complete_uploaded != {record["path"]: record for record in records}:
      raise ValueError("final TiTO delta ledger does not cover its inventory")
    final_record = {
        "schema": "canon.p57-tito-gcs-final-manifest.v2",
        "status": "PASS",
        "files": records,
        "file_count": len(records),
        "inventory_sha256": inventory_sha,
        "deltas": deltas,
        "readback_verified": True,
    }
    final_path = sync_root / "final-manifest.json"
    _atomic_json(final_path, final_record, accept_identical=True)
    final_sha = _sha256_file(final_path)
    _upload_and_verify(
        final_path,
        f"{prefix}/final-manifest.json",
        final_sha,
    )
    return {**receipt, "final_manifest_sha256": final_sha}
  return receipt


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--gcs-prefix", required=True)
  parser.add_argument("--final", action="store_true")
  parser.add_argument("--probe", action="store_true")
  args = parser.parse_args()
  if args.final and args.probe:
    parser.error("--final and --probe are mutually exclusive")
  try:
    result = (
        probe_gcs(args.state, args.gcs_prefix)
        if args.probe
        else sync_once(args.state, args.gcs_prefix, final=args.final)
    )
  except Exception as error:  # pylint: disable=broad-exception-caught
    print(f"P57_TITO_GCS_FAIL reason={error}")
    return 1
  print(
      "P57_TITO_GCS_PASS "
      f"status={result['status']} probe={int(args.probe)} "
      f"final={int(args.final)} "
      f"files={result.get('files', 0)} "
      f"inventory_sha256={result.get('inventory_sha256', 'none')}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
