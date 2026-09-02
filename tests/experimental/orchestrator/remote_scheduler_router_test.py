# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for RemoteSchedulerRouter against a fake scheduler sidecar."""

import http.server
import json
import threading
import time

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import remote_scheduler_router


class _FakeSidecar:
  """Minimal in-process stand-in for the py-inference-scheduler sidecar."""

  def __init__(self):
    self.mode = "pick"  # "pick" | "fallback" | "error"
    self.pick_name = None  # None => first candidate in the payload
    self.payloads = []
    self.paths = []

    sidecar = self

    class Handler(http.server.BaseHTTPRequestHandler):

      def log_message(self, *args):
        del args

      def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        sidecar.paths.append(self.path)
        if self.path == "/reset":
          self._send(200, {"status": "ok", "reset_scorers": 1})
          return
        payload = json.loads(raw) if raw else {}
        sidecar.payloads.append(payload)
        if sidecar.mode == "error":
          self._send(503, {"error": "unavailable"})
        elif sidecar.mode == "fallback":
          self._send(200, {"picked": None, "fallback": True, "scores": {}})
        else:
          picked = sidecar.pick_name or payload["candidates"][0]["name"]
          self._send(
              200, {"picked": picked, "fallback": False, "scores": {picked: 1.0}}
          )

      def _send(self, status, body):
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
    self._thread = threading.Thread(
        target=self.server.serve_forever, daemon=True
    )

  def __enter__(self):
    self._thread.start()
    return self

  def __exit__(self, *exc):
    self.server.shutdown()
    self.server.server_close()


class _FakeActor:

  def __init__(self, worker_id, state=datatypes.WorkerState.READY, **report):
    self.worker_id = worker_id
    self.report = datatypes.HealthReport(state=state, **report)

  def info(self):
    return datatypes.WorkerInfo(
        worker_id=self.worker_id, roles=frozenset({"rollout"})
    )

  def heartbeat(self):
    return self.report


def _router(url, **kwargs):
  defaults = dict(poll_interval_s=0.01, http_timeout_s=1.0, stats_ttl_s=5.0)
  defaults.update(kwargs)
  return remote_scheduler_router.RemoteSchedulerRouter(url, **defaults)


def _call_until(router, actors, predicate, deadline_s=5.0, **kwargs):
  """Routes repeatedly until predicate(result) holds (lets the poller warm up)."""
  deadline = time.monotonic() + deadline_s
  while True:
    result = router(actors, "generate", (), dict(kwargs))
    if predicate() or time.monotonic() > deadline:
      return result
    time.sleep(0.02)


class RemoteSchedulerRouterTest(absltest.TestCase):

  def test_picks_worker_named_by_sidecar(self):
    actors = [_FakeActor("rollout-0"), _FakeActor("rollout-1")]
    with _FakeSidecar() as sidecar:
      sidecar.pick_name = "rollout-1"
      router = _router(sidecar.url)
      try:
        picked = router(actors, "generate", (), {"route_key": "k"})
        self.assertIs(picked, actors[1])
      finally:
        router.stop()

  def test_payload_carries_mapped_stats_after_poll(self):
    actors = [
        _FakeActor("rollout-0", inflight=2, queue_depth=3, policy_version=7),
        _FakeActor("rollout-1", state=datatypes.WorkerState.SYNCING),
    ]
    with _FakeSidecar() as sidecar:
      router = _router(sidecar.url)
      try:
        _call_until(
            router,
            actors,
            lambda: sidecar.payloads
            and sidecar.payloads[-1]["candidates"][0]["attributes"]["state"]
            == "READY",
            prompt="solve this",
            request_id="req-42",
        )
        payload = sidecar.payloads[-1]
        self.assertEqual(payload["request_id"], "req-42")
        self.assertEqual(payload["prompt"], "solve this")
        by_name = {c["name"]: c["attributes"] for c in payload["candidates"]}
        self.assertEqual(
            by_name["rollout-0"]["routing_stats"],
            {"num_running_reqs": 2, "num_waiting_reqs": 3, "kv": 0.0},
        )
        self.assertEqual(by_name["rollout-0"]["queue_len"], 5)
        self.assertEqual(by_name["rollout-0"]["policy_version"], 7)
        self.assertEqual(by_name["rollout-1"]["state"], "SYNCING")
      finally:
        router.stop()

  def test_stale_stats_marked_unknown(self):
    actors = [_FakeActor("rollout-0")]
    with _FakeSidecar() as sidecar:
      router = _router(sidecar.url, stats_ttl_s=0.0)
      try:
        router(actors, "generate", (), {})
        state = sidecar.payloads[-1]["candidates"][0]["attributes"]["state"]
        self.assertEqual(state, "UNKNOWN")
      finally:
        router.stop()

  def test_fallback_on_http_error_round_robins(self):
    actors = [_FakeActor("rollout-0"), _FakeActor("rollout-1")]
    with _FakeSidecar() as sidecar:
      sidecar.mode = "error"
      router = _router(sidecar.url)
      try:
        picks = [router(actors, "generate", (), {}) for _ in range(4)]
        self.assertEqual(picks, [actors[0], actors[1], actors[0], actors[1]])
      finally:
        router.stop()

  def test_fallback_on_http_error_respects_route_key(self):
    actors = [_FakeActor("rollout-0"), _FakeActor("rollout-1")]
    with _FakeSidecar() as sidecar:
      sidecar.mode = "error"
      router = _router(sidecar.url)
      try:
        first = router(actors, "generate", (), {"route_key": "sticky"})
        second = router(actors, "generate", (), {"route_key": "sticky"})
        self.assertIs(first, second)
      finally:
        router.stop()

  def test_fallback_on_decline_response(self):
    actors = [_FakeActor("rollout-0"), _FakeActor("rollout-1")]
    with _FakeSidecar() as sidecar:
      sidecar.mode = "fallback"
      router = _router(sidecar.url)
      try:
        picked = router(actors, "generate", (), {})
        self.assertIn(picked, actors)
      finally:
        router.stop()

  def test_fallback_on_unknown_picked_name(self):
    actors = [_FakeActor("rollout-0")]
    with _FakeSidecar() as sidecar:
      sidecar.pick_name = "no-such-worker"
      router = _router(sidecar.url)
      try:
        picked = router(actors, "generate", (), {})
        self.assertIs(picked, actors[0])
      finally:
        router.stop()

  def test_fallback_when_sidecar_unreachable(self):
    actors = [_FakeActor("rollout-0"), _FakeActor("rollout-1")]
    # Nothing is listening on this port.
    router = _router("http://127.0.0.1:1", http_timeout_s=0.2)
    try:
      picked = router(actors, "generate", (), {})
      self.assertIn(picked, actors)
    finally:
      router.stop()

  def test_notify_weights_synced_posts_reset(self):
    with _FakeSidecar() as sidecar:
      router = _router(sidecar.url)
      try:
        router.notify_weights_synced()
        self.assertIn("/reset", sidecar.paths)
      finally:
        router.stop()

  def test_stop_joins_poller(self):
    actors = [_FakeActor("rollout-0")]
    with _FakeSidecar() as sidecar:
      router = _router(sidecar.url)
      router(actors, "generate", (), {})
      poller = router._poller
      self.assertIsNotNone(poller)
      router.stop()
      self.assertFalse(poller.is_alive())


if __name__ == "__main__":
  absltest.main()
