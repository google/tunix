#!/usr/bin/env python3
"""Continuously preserve one strict JobSet's logs and Kubernetes evidence.

This is an operator-side observer.  It never mutates a Kubernetes resource and
never runs inside a training Pod.  Open evidence is periodically mirrored to
``<gcs-prefix>/live``; a terminal, checksummed package is mirrored to
``<gcs-prefix>/sealed``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


_DNS_NAME = re.compile(r"^[a-z0-9](?:[-a-z0-9]{0,61}[a-z0-9])?$")
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_WORKER_NAME = re.compile(r"(?:^|-)pathways-worker-\d+-(\d+)$")
_TIMESTAMP = re.compile(
    rb"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s"
)
_FOLLOW_CONTAINERS = frozenset(
    ("jax-tpu", "pathways-worker", "pathways-proxy", "pathways-rm")
)


def _utc_now() -> str:
  return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
  return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_bytes(data)
  os.replace(temporary, path)


def _safe_component(value: str) -> str:
  safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
  if not safe or safe in (".", ".."):
    raise ValueError(f"unsafe path component: {value!r}")
  return safe


def validate_identity(
    jobset: str, source_sha: str, attempt: int, gcs_prefix: str
) -> None:
  if not _DNS_NAME.fullmatch(jobset):
    raise ValueError("jobset must be a Kubernetes DNS label")
  if not _SHA40.fullmatch(source_sha):
    raise ValueError("source SHA must be exactly 40 lowercase hexadecimal characters")
  if attempt != 0:
    raise ValueError("P57 evidence collection admits only JobSet attempt 0")
  if not gcs_prefix.startswith("gs://"):
    raise ValueError("gcs-prefix must be a non-root gs:// prefix")
  gcs_parts = [part for part in gcs_prefix[5:].split("/") if part]
  if len(gcs_parts) < 2:
    raise ValueError("gcs-prefix must include a path below the bucket root")
  if any(char.isspace() for char in gcs_prefix) or ".." in gcs_prefix.split("/"):
    raise ValueError("gcs-prefix contains an unsafe component")


def jobset_terminal(jobset: Mapping[str, Any]) -> str | None:
  for condition in jobset.get("status", {}).get("conditions", ()):
    if condition.get("status") == "True" and condition.get("type") in (
        "Completed",
        "Failed",
    ):
      return str(condition["type"])
  return None


def _replicated_job_name(pod: Mapping[str, Any]) -> str:
  metadata = pod.get("metadata", {})
  for collection in (metadata.get("labels", {}), metadata.get("annotations", {})):
    value = collection.get("jobset.sigs.k8s.io/replicatedjob-name")
    if value:
      return str(value)
  name = str(metadata.get("name", ""))
  if "pathways-worker" in name:
    return "pathways-worker"
  if "pathways-head" in name:
    return "pathways-head"
  return "unknown"


def worker_index(pod: Mapping[str, Any]) -> int | None:
  metadata = pod.get("metadata", {})
  for collection in (metadata.get("labels", {}), metadata.get("annotations", {})):
    raw = collection.get("batch.kubernetes.io/job-completion-index")
    if raw is not None and str(raw).isdigit():
      return int(raw)
  match = _WORKER_NAME.search(str(metadata.get("name", "")))
  return int(match.group(1)) if match else None


@dataclass(frozen=True, order=True)
class StreamKey:
  pod_uid: str
  pod_name: str
  role: str
  index: int | None
  container: str

  @property
  def relative_log(self) -> Path:
    index = "head" if self.index is None else f"worker-{self.index:02d}"
    return Path("logs") / index / (
        f"{_safe_component(self.role)}.{_safe_component(self.pod_name)}."
        f"{_safe_component(self.pod_uid)}.{_safe_component(self.container)}.log"
    )


def discover_streams(pods: Mapping[str, Any]) -> tuple[StreamKey, ...]:
  discovered: set[StreamKey] = set()
  for pod in pods.get("items", ()):
    metadata = pod.get("metadata", {})
    pod_name = str(metadata.get("name", ""))
    pod_uid = str(metadata.get("uid", ""))
    role = _replicated_job_name(pod)
    if not pod_name or not pod_uid or role == "unknown":
      continue
    index = worker_index(pod) if role == "pathways-worker" else None
    spec = pod.get("spec", {})
    containers = list(spec.get("containers", ())) + list(
        spec.get("initContainers", ())
    )
    for container in containers:
      name = str(container.get("name", ""))
      if name not in _FOLLOW_CONTAINERS:
        continue
      if role == "pathways-worker" and name != "pathways-worker":
        continue
      if role == "pathways-head" and name == "pathways-worker":
        continue
      discovered.add(StreamKey(pod_uid, pod_name, role, index, name))
  return tuple(sorted(discovered))


def build_log_command(
    namespace: str, key: StreamKey, since_time: str | None = None
) -> list[str]:
  command = [
      "kubectl",
      "--namespace",
      namespace,
      "logs",
      "--follow",
      "--timestamps",
      key.pod_name,
      "--container",
      key.container,
  ]
  if since_time:
    command.extend(("--since-time", since_time))
  return command


def build_previous_log_command(namespace: str, key: StreamKey) -> list[str]:
  return [
      "kubectl",
      "--namespace",
      namespace,
      "logs",
      "--timestamps",
      key.pod_name,
      "--container",
      key.container,
      "--previous",
  ]


def restarted_streams(
    pods: Mapping[str, Any], streams: Iterable[StreamKey]
) -> tuple[StreamKey, ...]:
  restarts: dict[tuple[str, str], int] = {}
  for pod in pods.get("items", ()):
    uid = str(pod.get("metadata", {}).get("uid", ""))
    status = pod.get("status", {})
    for item in list(status.get("containerStatuses", ())) + list(
        status.get("initContainerStatuses", ())
    ):
      restarts[(uid, str(item.get("name", "")))] = int(item.get("restartCount", 0))
  return tuple(
      key for key in streams if restarts.get((key.pod_uid, key.container), 0) > 0
  )


def _last_log_timestamp(path: Path) -> str | None:
  if not path.exists() or path.stat().st_size == 0:
    return None
  with path.open("rb") as handle:
    handle.seek(max(0, path.stat().st_size - 256 * 1024))
    lines = handle.readlines()
  for line in reversed(lines):
    match = _TIMESTAMP.match(line)
    if match:
      return match.group(1).decode("ascii")
  return None


class CommandError(RuntimeError):
  pass


class Runner:

  def run_json(self, command: Sequence[str]) -> Mapping[str, Any]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
      message = result.stderr.strip().splitlines()[-1:] or ["no stderr"]
      raise CommandError(f"command failed rc={result.returncode}: {message[0]}")
    try:
      value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
      raise CommandError("command returned invalid JSON") from error
    if not isinstance(value, dict):
      raise CommandError("command JSON root is not an object")
    return value

  def run(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=False)

  def follow(self, command: Sequence[str], output: Any) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _content_snapshot(root: Path, category: str, value: Any) -> Path:
  data = _json_bytes(value)
  digest = hashlib.sha256(data).hexdigest()
  destination = root / "snapshots" / category / f"{digest}.json"
  if not destination.exists():
    _atomic_write(destination, data)
  return destination


def _filter_events(
    events: Mapping[str, Any], names: set[str], uids: set[str]
) -> Mapping[str, Any]:
  selected = []
  for event in events.get("items", ()):
    involved = event.get("involvedObject", {})
    if involved.get("name") in names or involved.get("uid") in uids:
      selected.append(event)
  return {"apiVersion": events.get("apiVersion", "v1"), "items": selected}


def _node_names(pods: Mapping[str, Any]) -> list[str]:
  return sorted(
      {
          str(pod.get("spec", {}).get("nodeName"))
          for pod in pods.get("items", ())
          if pod.get("spec", {}).get("nodeName")
      }
  )


def _sha256_manifest(root: Path) -> str:
  manifest = root / "SHA256SUMS"
  if manifest.exists():
    raise FileExistsError(f"refusing to replace existing manifest: {manifest}")
  lines = []
  for path in sorted(item for item in root.rglob("*") if item.is_file()):
    if path.name == "SHA256SUMS":
      continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root).as_posix()}\n")
  _atomic_write(manifest, "".join(lines).encode("utf-8"))
  return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _nonempty(path: Path) -> bool:
  return path.is_file() and path.stat().st_size > 0


def classify_sealed(
    sealed: Path, expected_workers: int, terminal: str | None, upload_failures: int
) -> dict[str, Any]:
  worker_indices: set[int] = set()
  for path in (sealed / "logs").glob("worker-*/*.pathways-worker.log.gz"):
    match = re.fullmatch(r"worker-(\d+)", path.parent.name)
    if match and _nonempty(path):
      worker_indices.add(int(match.group(1)))
  head_logs = [
      path
      for path in (sealed / "logs" / "head").glob("*.jax-tpu.log.gz")
      if _nonempty(path)
  ]
  required_metadata = (
      sealed / "final" / "jobset.json",
      sealed / "final" / "pods.json",
      sealed / "final" / "events.json",
      sealed / "final" / "nodes.json",
  )
  expected = set(range(expected_workers))
  reasons = []
  if terminal not in ("Completed", "Failed"):
    reasons.append("jobset_not_terminal")
  if worker_indices != expected:
    reasons.append(
        "worker_log_coverage="
        f"{len(worker_indices)}/{expected_workers};missing="
        + ",".join(str(item) for item in sorted(expected - worker_indices))
    )
  if not head_logs:
    reasons.append("head_log_missing")
  missing_metadata = [path.name for path in required_metadata if not _nonempty(path)]
  if missing_metadata:
    reasons.append("metadata_missing=" + ",".join(missing_metadata))
  if upload_failures:
    reasons.append(f"live_upload_failures={upload_failures}")
  return {
      "classification": "PASS" if not reasons else "INCONCLUSIVE",
      "scope": "evidence_collection_only",
      "terminal_condition": terminal,
      "expected_workers": expected_workers,
      "worker_indices": sorted(worker_indices),
      "head_logs": len(head_logs),
      "upload_failures": upload_failures,
      "reasons": reasons,
  }


class Collector:

  def __init__(self, args: argparse.Namespace, runner: Runner | None = None):
    self.args = args
    self.runner = runner or Runner()
    self.root = args.output_dir
    self.live = self.root / "live"
    self.sealed = self.root / "sealed"
    self.streams: dict[StreamKey, tuple[subprocess.Popen[bytes], Any]] = {}
    self.captured_previous: set[StreamKey] = set()
    self.upload_failures = 0
    self.poll_errors: list[dict[str, str]] = []
    self.stop_requested = False
    self.terminal: str | None = None
    self.latest: dict[str, Mapping[str, Any]] = {}

  def prepare(self) -> None:
    validate_identity(
        self.args.jobset,
        self.args.source_sha,
        self.args.attempt,
        self.args.gcs_prefix,
    )
    if not _DNS_NAME.fullmatch(self.args.namespace):
      raise ValueError("namespace must be a Kubernetes DNS label")
    if self.args.expected_workers < 1:
      raise ValueError("expected-workers must be positive")
    if self.args.poll_seconds < 1 or self.args.upload_seconds < 1:
      raise ValueError("poll and upload intervals must be positive")
    if self.root.exists():
      raise FileExistsError(f"output directory already exists: {self.root}")
    for binary in ("kubectl", "gcloud"):
      if shutil.which(binary) is None:
        raise FileNotFoundError(f"required command is unavailable: {binary}")
    remote = self.runner.run(
        ["gcloud", "storage", "ls", self.args.gcs_prefix.rstrip("/") + "/"]
    )
    if remote.returncode == 0 and remote.stdout.strip():
      raise FileExistsError("refusing to reuse a nonempty GCS evidence prefix")
    if remote.returncode != 0:
      missing_phrases = ("matched no objects", "no urls matched")
      if not any(phrase in remote.stderr.lower() for phrase in missing_phrases):
        raise CommandError("could not prove that the GCS evidence prefix is empty")
    self.live.mkdir(parents=True)
    identity = {
        "schema": "p57-jobset-log-collector-v1",
        "jobset": self.args.jobset,
        "namespace": self.args.namespace,
        "source_sha": self.args.source_sha,
        "attempt": self.args.attempt,
        "expected_workers": self.args.expected_workers,
        "started_at": _utc_now(),
    }
    _atomic_write(self.live / "IDENTITY.json", _json_bytes(identity))

  def _kubectl_json(self, args: Sequence[str]) -> Mapping[str, Any]:
    return self.runner.run_json(
        ["kubectl", "--namespace", self.args.namespace, *args, "-o", "json"]
    )

  def snapshot(self) -> None:
    jobset = self._kubectl_json(("get", "jobset", self.args.jobset))
    pods = self._kubectl_json(
        (
            "get",
            "pods",
            "--selector",
            f"jobset.sigs.k8s.io/jobset-name={self.args.jobset}",
        )
    )
    all_events = self._kubectl_json(("get", "events"))
    names = {self.args.jobset}
    uids = {str(jobset.get("metadata", {}).get("uid", ""))}
    for pod in pods.get("items", ()):
      names.add(str(pod.get("metadata", {}).get("name", "")))
      uids.add(str(pod.get("metadata", {}).get("uid", "")))
    events = _filter_events(all_events, names, uids)
    nodes = {"apiVersion": "v1", "items": []}
    for node in _node_names(pods):
      nodes["items"].append(
          self.runner.run_json(("kubectl", "get", "node", node, "-o", "json"))
      )
    for category, value in (
        ("jobset", jobset),
        ("pods", pods),
        ("events", events),
        ("nodes", nodes),
    ):
      _content_snapshot(self.live, category, value)
    self.latest = {
        "jobset": jobset,
        "pods": pods,
        "events": events,
        "nodes": nodes,
    }
    self.terminal = jobset_terminal(jobset)
    discovered = discover_streams(pods)
    self._reconcile_streams(discovered)
    self._capture_previous(restarted_streams(pods, discovered))
    self._write_live_receipt()

  def _reconcile_streams(self, keys: Iterable[StreamKey]) -> None:
    for key, (process, handle) in list(self.streams.items()):
      if process.poll() is not None:
        handle.close()
        del self.streams[key]
    for key in keys:
      if key in self.streams:
        continue
      path = self.live / key.relative_log
      path.parent.mkdir(parents=True, exist_ok=True)
      since_time = _last_log_timestamp(path)
      handle = path.open("ab", buffering=0)
      command = build_log_command(self.args.namespace, key, since_time)
      process = self.runner.follow(command, handle)
      self.streams[key] = (process, handle)

  def _capture_previous(self, keys: Iterable[StreamKey]) -> None:
    for key in keys:
      if key in self.captured_previous:
        continue
      result = self.runner.run(build_previous_log_command(self.args.namespace, key))
      if result.returncode:
        self.poll_errors.append(
            {
                "at": _utc_now(),
                "operation": "previous_log",
                "error": f"rc={result.returncode}",
            }
        )
        continue
      destination = self.live / key.relative_log.with_suffix(".previous.log")
      _atomic_write(destination, result.stdout.encode("utf-8"))
      self.captured_previous.add(key)

  def _write_live_receipt(self) -> None:
    workers = sorted(
        {
            key.index
            for key in self.streams
            if key.role == "pathways-worker" and key.index is not None
        }
    )
    receipt = {
        "updated_at": _utc_now(),
        "jobset": self.args.jobset,
        "terminal_condition": self.terminal,
        "expected_workers": self.args.expected_workers,
        "discovered_worker_indices": workers,
        "active_streams": len(self.streams),
        "upload_failures": self.upload_failures,
        "poll_errors": self.poll_errors[-20:],
    }
    _atomic_write(self.live / "LIVE.json", _json_bytes(receipt))

  def upload_live(self) -> None:
    result = self.runner.run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(self.live),
            self.args.gcs_prefix.rstrip("/") + "/live",
        ]
    )
    if result.returncode:
      self.upload_failures += 1
      self.poll_errors.append(
          {"at": _utc_now(), "operation": "live_upload", "error": f"rc={result.returncode}"}
      )
      self._write_live_receipt()

  def _stop_streams(self) -> None:
    for process, _ in self.streams.values():
      if process.poll() is None:
        process.terminate()
    deadline = time.monotonic() + 10
    for process, _ in self.streams.values():
      remaining = max(0, deadline - time.monotonic())
      try:
        process.wait(timeout=remaining)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    for _, handle in self.streams.values():
      handle.close()
    self.streams.clear()

  def seal(self) -> dict[str, Any]:
    self._stop_streams()
    if self.sealed.exists():
      raise FileExistsError(f"sealed directory already exists: {self.sealed}")
    (self.sealed / "logs").mkdir(parents=True)
    for source in sorted((self.live / "logs").rglob("*.log")):
      relative = source.relative_to(self.live)
      destination = self.sealed / relative.with_suffix(relative.suffix + ".gz")
      destination.parent.mkdir(parents=True, exist_ok=True)
      with source.open("rb") as incoming, gzip.open(destination, "wb", compresslevel=9) as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    final = self.sealed / "final"
    final.mkdir()
    for name in ("jobset", "pods", "events", "nodes"):
      if name in self.latest:
        _atomic_write(final / f"{name}.json", _json_bytes(self.latest[name]))
    classification = classify_sealed(
        self.sealed,
        self.args.expected_workers,
        self.terminal,
        self.upload_failures,
    )
    classification.update(
        {
            "schema": "p57-jobset-log-collector-v1",
            "jobset": self.args.jobset,
            "source_sha": self.args.source_sha,
            "attempt": self.args.attempt,
            "sealed_at": _utc_now(),
        }
    )
    _atomic_write(self.sealed / "COLLECTED.json", _json_bytes(classification))
    classification["sha256sums_sha256"] = _sha256_manifest(self.sealed)
    return classification

  def upload_sealed(self) -> bool:
    result = self.runner.run(
        [
            "gcloud",
            "storage",
            "rsync",
            "--recursive",
            str(self.sealed),
            self.args.gcs_prefix.rstrip("/") + "/sealed",
        ]
    )
    return result.returncode == 0

  def run(self) -> dict[str, Any]:
    self.prepare()
    last_upload = 0.0
    terminal_seen_at: float | None = None
    while not self.stop_requested:
      try:
        self.snapshot()
      except (CommandError, OSError, ValueError) as error:
        self.poll_errors.append(
            {"at": _utc_now(), "operation": "snapshot", "error": str(error)}
        )
        self._write_live_receipt()
      now = time.monotonic()
      if now - last_upload >= self.args.upload_seconds:
        self.upload_live()
        last_upload = now
      if self.terminal:
        terminal_seen_at = terminal_seen_at or now
        if now - terminal_seen_at >= self.args.terminal_grace_seconds:
          break
      time.sleep(self.args.poll_seconds)
    if self.stop_requested and self.terminal is None:
      self.poll_errors.append(
          {"at": _utc_now(), "operation": "collector", "error": "interrupted"}
      )
      self._write_live_receipt()
    try:
      self.snapshot()
    except (CommandError, OSError, ValueError) as error:
      self.poll_errors.append(
          {"at": _utc_now(), "operation": "final_snapshot", "error": str(error)}
      )
    self.upload_live()
    classification = self.seal()
    if not self.upload_sealed():
      classification["classification"] = "INCONCLUSIVE"
      classification["reasons"].append("sealed_upload_failed")
      (self.sealed / "SHA256SUMS").unlink()
      classification.pop("sha256sums_sha256", None)
      _atomic_write(self.sealed / "COLLECTED.json", _json_bytes(classification))
      classification["sha256sums_sha256"] = _sha256_manifest(self.sealed)
    return classification


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--jobset", required=True)
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--gcs-prefix", required=True)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--namespace", default="default")
  parser.add_argument("--attempt", default=0, type=int)
  parser.add_argument("--expected-workers", default=16, type=int)
  parser.add_argument("--poll-seconds", default=30, type=int)
  parser.add_argument("--upload-seconds", default=120, type=int)
  parser.add_argument("--terminal-grace-seconds", default=30, type=int)
  return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  collector = Collector(args)

  def request_stop(unused_signum: int, unused_frame: Any) -> None:
    del unused_signum, unused_frame
    collector.stop_requested = True

  signal.signal(signal.SIGINT, request_stop)
  signal.signal(signal.SIGTERM, request_stop)
  try:
    classification = collector.run()
  except (CommandError, FileExistsError, FileNotFoundError, ValueError) as error:
    print(f"P57_JOBSET_LOG_COLLECTOR_REFUSED reason={error}", file=sys.stderr)
    return 2
  verdict = classification["classification"]
  print(
      "P57_JOBSET_LOG_COLLECTOR_"
      f"{verdict} workers={len(classification['worker_indices'])}/"
      f"{classification['expected_workers']} terminal="
      f"{classification['terminal_condition']}"
  )
  return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
