#!/usr/bin/env python3
"""Create and verify deterministic flat P38 evidence archives."""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import re
import shutil
import tarfile


_MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  ([^/]+)$")


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def _sha256_path(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _sha256_stream(stream) -> str:
  digest = hashlib.sha256()
  for chunk in iter(lambda: stream.read(1024 * 1024), b""):
    digest.update(chunk)
  return digest.hexdigest()


def _parse_manifest_bytes(payload: bytes) -> list[tuple[str, str]]:
  try:
    lines = payload.decode("utf-8").splitlines()
  except UnicodeDecodeError as exc:
    raise ValueError("SHA256SUMS is not UTF-8") from exc
  _require(lines, "SHA256SUMS is empty")
  records: list[tuple[str, str]] = []
  seen: set[str] = set()
  for line_number, line in enumerate(lines, start=1):
    match = _MANIFEST_RE.fullmatch(line)
    _require(match is not None,
             f"invalid SHA256SUMS line {line_number}: {line!r}")
    digest, name = match.groups()
    _require(name not in (".", "..", "SHA256SUMS"),
             f"invalid evidence filename: {name!r}")
    _require(name not in seen, f"duplicate evidence filename: {name}")
    seen.add(name)
    records.append((name, digest))
  _require([name for name, _ in records] == sorted(seen),
           "SHA256SUMS filenames are not sorted")
  return records


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
  info = tarfile.TarInfo(name=name)
  info.size = size
  info.mtime = 0
  info.mode = 0o600
  info.uid = 0
  info.gid = 0
  info.uname = ""
  info.gname = ""
  return info


def create_archive(root: Path, manifest: Path, output: Path) -> tuple[int, str]:
  _require(root.is_dir(), f"archive root is absent: {root}")
  _require(manifest.is_file(), f"SHA256SUMS is absent: {manifest}")
  _require(not output.exists(), f"archive output already exists: {output}")
  manifest_bytes = manifest.read_bytes()
  records = _parse_manifest_bytes(manifest_bytes)
  member_sizes = [len(manifest_bytes)]
  for name, expected in records:
    source = root / name
    _require(source.is_file() and not source.is_symlink(),
             f"manifest file is absent or unsafe: {name}")
    _require(_sha256_path(source) == expected,
             f"manifest SHA failed before archive creation: {name}")
    member_sizes.append(source.stat().st_size)
  output.parent.mkdir(parents=True, exist_ok=True)
  tar_bytes = sum(
      512 + ((size + 511) // 512) * 512 for size in member_sizes
  ) + 1024
  tar_bytes = ((tar_bytes + 10239) // 10240) * 10240
  free_bytes = shutil.disk_usage(output.parent).free
  _require(
      free_bytes >= tar_bytes + 16 * 1024 * 1024,
      f"insufficient free space for evidence archive: required>="
      f"{tar_bytes + 16 * 1024 * 1024} free={free_bytes}",
  )
  partial = output.with_name(output.name + ".partial")
  _require(not partial.exists(), f"partial archive already exists: {partial}")
  try:
    with tarfile.open(partial, mode="w", format=tarfile.GNU_FORMAT) as archive:
      archive.addfile(
          _tar_info("SHA256SUMS", len(manifest_bytes)),
          io.BytesIO(manifest_bytes),
      )
      for name, _ in records:
        source = root / name
        with source.open("rb") as stream:
          archive.addfile(_tar_info(name, source.stat().st_size), stream)
    partial.replace(output)
  finally:
    if partial.exists():
      partial.unlink()
  return len(records), _sha256_path(output)


def verify_archive(archive_path: Path, expected_sha256: str | None) -> tuple[int, str]:
  _require(archive_path.is_file(), f"archive is absent: {archive_path}")
  archive_sha = _sha256_path(archive_path)
  if expected_sha256 is not None:
    _require(re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is not None,
             "expected archive SHA256 is malformed")
    _require(archive_sha == expected_sha256,
             f"archive SHA failed: {archive_sha} != {expected_sha256}")
  with tarfile.open(archive_path, mode="r:") as archive:
    members = archive.getmembers()
    names = [member.name for member in members]
    _require(len(names) == len(set(names)), "archive has duplicate members")
    _require(all(member.isfile() for member in members),
             "archive contains a non-regular member")
    _require("SHA256SUMS" in names, "archive has no SHA256SUMS")
    manifest_member = archive.getmember("SHA256SUMS")
    manifest_stream = archive.extractfile(manifest_member)
    _require(manifest_stream is not None, "cannot read archived SHA256SUMS")
    records = _parse_manifest_bytes(manifest_stream.read())
    expected_names = ["SHA256SUMS"] + [name for name, _ in records]
    _require(names == expected_names,
             "archive members do not exactly match sorted SHA256SUMS")
    for name, expected in records:
      stream = archive.extractfile(name)
      _require(stream is not None, f"cannot read archived member: {name}")
      _require(_sha256_stream(stream) == expected,
               f"archived member SHA failed: {name}")
  return len(records), archive_sha


def extract_archive(archive_path: Path, output: Path) -> tuple[int, str]:
  count, archive_sha = verify_archive(archive_path, None)
  _require(not output.exists(), f"extract output already exists: {output}")
  output.mkdir(parents=True, mode=0o700)
  try:
    with tarfile.open(archive_path, mode="r:") as archive:
      for member in archive.getmembers():
        stream = archive.extractfile(member)
        _require(stream is not None, f"cannot extract member: {member.name}")
        destination = output / member.name
        with destination.open("xb") as sink:
          shutil.copyfileobj(stream, sink, length=1024 * 1024)
        destination.chmod(0o600)
  except Exception:
    shutil.rmtree(output)
    raise
  manifest = output / "SHA256SUMS"
  records = _parse_manifest_bytes(manifest.read_bytes())
  for name, expected in records:
    _require(_sha256_path(output / name) == expected,
             f"extracted member SHA failed: {name}")
  return count, archive_sha


def main() -> int:
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command", required=True)
  create = subparsers.add_parser("create")
  create.add_argument("--root", required=True, type=Path)
  create.add_argument("--manifest", required=True, type=Path)
  create.add_argument("--output", required=True, type=Path)
  verify = subparsers.add_parser("verify")
  verify.add_argument("--archive", required=True, type=Path)
  verify.add_argument("--expected-sha256")
  extract = subparsers.add_parser("extract")
  extract.add_argument("--archive", required=True, type=Path)
  extract.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.command == "create":
    count, digest = create_archive(args.root, args.manifest, args.output)
    action = "CREATED"
    path = args.output
  elif args.command == "verify":
    count, digest = verify_archive(args.archive, args.expected_sha256)
    action = "VERIFIED"
    path = args.archive
  else:
    count, digest = extract_archive(args.archive, args.output)
    action = "EXTRACTED"
    path = args.archive
  print(
      f"[P38.ARCHIVE] {action} path={path} logical_files={count} "
      f"sha256={digest}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
