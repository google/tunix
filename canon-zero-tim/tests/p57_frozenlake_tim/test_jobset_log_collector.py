"""Host-only contracts for the external P57 JobSet evidence collector."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts"
    / "collect_jobset_logs_to_gcs.py"
)


def _module():
  spec = importlib.util.spec_from_file_location("p57_jobset_log_collector", SCRIPT)
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


collector = _module()


def _pod(name, uid, role, containers, index=None, node="node-a"):
  labels = {"jobset.sigs.k8s.io/replicatedjob-name": role}
  if index is not None:
    labels["batch.kubernetes.io/job-completion-index"] = str(index)
  return {
      "metadata": {"name": name, "uid": uid, "labels": labels},
      "spec": {
          "nodeName": node,
          "containers": [{"name": item} for item in containers],
          "initContainers": [],
      },
  }


class JobSetLogCollectorTest(unittest.TestCase):

  def test_identity_requires_attempt_zero_exact_sha_and_fresh_prefix(self):
    collector.validate_identity("canon-p57-good", "a" * 40, 0, "gs://bucket/run")
    bad = (
        ("Bad_Name", "a" * 40, 0, "gs://bucket/run"),
        ("canon-p57-good", "a" * 39, 0, "gs://bucket/run"),
        ("canon-p57-good", "a" * 40, 1, "gs://bucket/run"),
        ("canon-p57-good", "a" * 40, 0, "/tmp/not-gcs"),
        ("canon-p57-good", "a" * 40, 0, "gs://bucket"),
        ("canon-p57-good", "a" * 40, 0, "gs://bucket/../run"),
    )
    for arguments in bad:
      with self.subTest(arguments=arguments), self.assertRaises(ValueError):
        collector.validate_identity(*arguments)

  def test_discovers_head_sidecars_and_all_worker_indices_by_uid(self):
    head = _pod(
        "run-pathways-head-0-0",
        "head-uid",
        "pathways-head",
        ["jax-tpu"],
    )
    head["spec"]["initContainers"] = [
        {"name": "pathways-proxy"},
        {"name": "pathways-rm"},
    ]
    workers = [
        _pod(
            f"run-pathways-worker-0-{index}",
            f"worker-uid-{index}",
            "pathways-worker",
            ["pathways-worker"],
            index=index,
            node=f"node-{index}",
        )
        for index in range(16)
    ]
    streams = collector.discover_streams({"items": [head, *workers]})
    self.assertEqual(len(streams), 19)
    self.assertEqual(
        {item.index for item in streams if item.role == "pathways-worker"},
        set(range(16)),
    )
    self.assertEqual(
        {item.container for item in streams if item.role == "pathways-head"},
        {"jax-tpu", "pathways-proxy", "pathways-rm"},
    )
    self.assertTrue(all(item.pod_uid in str(item.relative_log) for item in streams))

  def test_worker_index_falls_back_to_job_pod_name(self):
    pod = _pod(
        "run-pathways-worker-0-7",
        "uid-7",
        "pathways-worker",
        ["pathways-worker"],
    )
    self.assertEqual(collector.worker_index(pod), 7)

  def test_terminal_condition_is_explicit_only(self):
    self.assertIsNone(collector.jobset_terminal({"status": {"conditions": []}}))
    self.assertIsNone(
        collector.jobset_terminal(
            {"status": {"conditions": [{"type": "Failed", "status": "False"}]}}
        )
    )
    self.assertEqual(
        collector.jobset_terminal(
            {"status": {"conditions": [{"type": "Completed", "status": "True"}]}}
        ),
        "Completed",
    )

  def test_log_command_is_argument_vector_and_resumes_by_timestamp(self):
    key = collector.StreamKey(
        "worker-uid-2",
        "run-pathways-worker-0-2",
        "pathways-worker",
        2,
        "pathways-worker",
    )
    command = collector.build_log_command(
        "default", key, "2026-08-28T03:32:41.123Z"
    )
    self.assertEqual(command[0], "kubectl")
    self.assertIn("--follow", command)
    self.assertEqual(command[-2:], ["--since-time", "2026-08-28T03:32:41.123Z"])
    self.assertFalse(any(item in command for item in ("|", ">", "&&", ";")))
    previous = collector.build_previous_log_command("default", key)
    self.assertEqual(previous[-1], "--previous")
    self.assertNotIn("--follow", previous)

  def test_previous_log_is_requested_only_after_container_restart(self):
    pod = _pod(
        "run-pathways-worker-0-2",
        "worker-uid-2",
        "pathways-worker",
        ["pathways-worker"],
        index=2,
    )
    streams = collector.discover_streams({"items": [pod]})
    self.assertEqual(collector.restarted_streams({"items": [pod]}, streams), ())
    pod["status"] = {
        "containerStatuses": [
            {"name": "pathways-worker", "restartCount": 1}
        ]
    }
    self.assertEqual(collector.restarted_streams({"items": [pod]}, streams), streams)

  def test_event_filter_keeps_only_jobset_and_pod_events(self):
    events = {
        "items": [
            {"involvedObject": {"name": "run", "uid": "jobset-uid"}},
            {"involvedObject": {"name": "pod-2", "uid": "pod-uid-2"}},
            {"involvedObject": {"name": "other", "uid": "other-uid"}},
        ]
    }
    filtered = collector._filter_events(
        events, {"run", "pod-2"}, {"jobset-uid", "pod-uid-2"}
    )
    self.assertEqual(len(filtered["items"]), 2)

  def test_sealed_classifier_passes_only_complete_16_worker_bundle(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for index in range(16):
        path = root / "logs" / f"worker-{index:02d}" / (
            f"pathways-worker.pod.uid-{index}.pathways-worker.log.gz"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"worker-log")
      head = root / "logs/head/pathways-head.pod.head.jax-tpu.log.gz"
      head.parent.mkdir(parents=True)
      head.write_bytes(b"head-log")
      final = root / "final"
      final.mkdir()
      for name in ("jobset", "pods", "events", "nodes"):
        (final / f"{name}.json").write_text("{}\n", encoding="utf-8")

      result = collector.classify_sealed(root, 16, "Completed", 0)
      self.assertEqual(result["classification"], "PASS")
      self.assertEqual(result["worker_indices"], list(range(16)))

      (root / "logs/worker-02/pathways-worker.pod.uid-2.pathways-worker.log.gz").unlink()
      missing = collector.classify_sealed(root, 16, "Failed", 0)
      self.assertEqual(missing["classification"], "INCONCLUSIVE")
      self.assertTrue(any("missing=2" in reason for reason in missing["reasons"]))

  def test_upload_failure_and_nonterminal_are_fail_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      result = collector.classify_sealed(Path(tmp), 16, None, 1)
    self.assertEqual(result["classification"], "INCONCLUSIVE")
    self.assertIn("jobset_not_terminal", result["reasons"])
    self.assertIn("live_upload_failures=1", result["reasons"])

  def test_checksum_manifest_excludes_itself_and_verifies_contents(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      (root / "nested").mkdir()
      (root / "a.txt").write_text("a\n", encoding="utf-8")
      (root / "nested/b.txt").write_text("b\n", encoding="utf-8")
      digest = collector._sha256_manifest(root)
      lines = (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
      self.assertEqual(len(digest), 64)
      self.assertEqual(len(lines), 2)
      self.assertTrue(all("SHA256SUMS" not in line for line in lines))
      for line in lines:
        expected, relative = line.split("  ", 1)
        actual = collector.hashlib.sha256((root / relative).read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

  def test_seal_copies_live_logs_and_writes_complete_terminal_package(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp) / "evidence"
      args = collector.argparse.Namespace(
          output_dir=root,
          expected_workers=16,
          jobset="canon-p57-test",
          source_sha="a" * 40,
          attempt=0,
      )
      instance = collector.Collector(args)
      instance.live.mkdir(parents=True)
      for index in range(16):
        path = root / "live/logs" / f"worker-{index:02d}" / (
            f"pathways-worker.pod.uid-{index}.pathways-worker.log"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"worker {index}\n", encoding="utf-8")
      head = root / "live/logs/head/pathways-head.pod.uid.jax-tpu.log"
      head.parent.mkdir(parents=True)
      head.write_text("head\n", encoding="utf-8")
      instance.latest = {
          "jobset": {"status": {"conditions": []}},
          "pods": {"items": []},
          "events": {"items": []},
          "nodes": {"items": []},
      }
      instance.terminal = "Failed"

      result = instance.seal()
      self.assertEqual(result["classification"], "PASS")
      self.assertTrue((root / "sealed/SHA256SUMS").is_file())
      stored = json.loads((root / "sealed/COLLECTED.json").read_text())
      self.assertEqual(stored["classification"], "PASS")
      self.assertEqual(stored["terminal_condition"], "Failed")
      self.assertEqual(
          len(list((root / "sealed/logs").rglob("*.log.gz"))), 17
      )

  def test_source_contains_no_mutating_kubectl_verb(self):
    source = SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("kubectl apply", "kubectl delete", "kubectl rollout"):
      self.assertNotIn(forbidden, source)


if __name__ == "__main__":
  unittest.main()
