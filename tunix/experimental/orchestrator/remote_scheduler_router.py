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

"""RoutingActorPool router that delegates rollout picking to a scheduler sidecar.

The sidecar is a py-inference-scheduler decision service (its
``integration/tunix`` FastAPI app): the orchestrator ships each candidate
worker's heartbeat-derived stats inline with a ``POST /schedule`` request and
dispatches to the returned winner over tunix's own transport. The sidecar
never proxies requests or scrapes workers.

Fallback is local (sticky ``route_key`` hash, else round-robin) whenever the
sidecar is unreachable, declines to pick, or names an unknown worker — the
router hook on `RoutingActorPool` cannot decline, so degradation happens here.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Sequence
import urllib.error
import urllib.request

from absl import logging

_WARN_EVERY = 50


class RemoteSchedulerRouter:
  """Load/state-aware rollout worker picker backed by a remote scheduler.

  Compatible with `RoutingActorPool(router=...)`: called as
  ``router(actors, method_name, args, kwargs) -> actor``.

  A background daemon thread polls each actor's ``heartbeat()`` and caches the
  latest `HealthReport` per worker; the routing hot path does no RPC — it
  snapshots the cache, posts one HTTP request to the sidecar (bounded by
  ``http_timeout_s``), and maps the winner's name back to an actor handle.
  """

  def __init__(
      self,
      scheduler_url: str,
      *,
      target_model: str | None = None,
      poll_interval_s: float = 0.25,
      http_timeout_s: float = 0.2,
      stats_ttl_s: float = 2.0,
  ):
    self._scheduler_url = scheduler_url.rstrip("/")
    self._target_model = target_model
    self._poll_interval_s = poll_interval_s
    self._http_timeout_s = http_timeout_s
    self._stats_ttl_s = stats_ttl_s

    self._lock = threading.Lock()
    # candidate name -> (health-ish dict, monotonic timestamp)
    self._snapshots: Dict[str, tuple[Dict[str, Any], float]] = {}
    # id(actor) -> stable candidate name
    self._names: Dict[int, str] = {}
    self._actors: list[Any] = []
    self._poller: threading.Thread | None = None
    self._stop_event = threading.Event()
    self._rr_idx = 0
    self._failures = 0
    self._request_seq = 0

  # ---------------------------------------------------------------- naming

  def _name_for(self, actor: Any, index: int) -> str:
    key = id(actor)
    name = self._names.get(key)
    if name is not None:
      return name
    try:
      info = actor.info()
      name = getattr(info, "worker_id", None)
    except Exception:  # pylint: disable=broad-except
      name = None
    if not name:
      name = getattr(actor, "target_address", None) or f"worker-{index}"
    self._names[key] = str(name)
    return self._names[key]

  # --------------------------------------------------------------- polling

  def _ensure_poller(self, actors: Sequence[Any]) -> None:
    with self._lock:
      self._actors = list(actors)
      if self._poller is None or not self._poller.is_alive():
        self._stop_event.clear()
        self._poller = threading.Thread(
            target=self._poll_loop,
            name="remote-scheduler-router-poller",
            daemon=True,
        )
        self._poller.start()

  def _poll_loop(self) -> None:
    while not self._stop_event.is_set():
      with self._lock:
        actors = list(self._actors)
      for i, actor in enumerate(actors):
        name = self._name_for(actor, i)
        try:
          heartbeat = getattr(actor, "heartbeat", None)
          report = heartbeat() if callable(heartbeat) else actor.submit(
              "heartbeat"
          )
          stats = self._report_to_stats(report)
        except Exception as e:  # pylint: disable=broad-except
          logging.log_every_n(
              logging.WARNING, "Heartbeat poll failed for %s: %s", 20, name, e
          )
          continue
        with self._lock:
          self._snapshots[name] = (stats, time.monotonic())
      self._stop_event.wait(self._poll_interval_s)

  def _report_to_stats(self, report: Any) -> Dict[str, Any]:
    """Maps a tunix HealthReport onto scheduler candidate attributes."""
    state = getattr(report, "state", None)
    state_str = str(getattr(state, "value", state or "UNKNOWN"))
    inflight = int(getattr(report, "inflight", 0) or 0)
    queue_depth = int(getattr(report, "queue_depth", 0) or 0)
    return {
        "state": state_str,
        "policy_version": int(getattr(report, "policy_version", 0) or 0),
        "queue_len": inflight + queue_depth,
        "routing_stats": {
            "num_running_reqs": inflight,
            "num_waiting_reqs": queue_depth,
            "kv": 0.0,  # Sampler.get_load_info() not plumbed yet.
        },
    }

  def _candidates(self, actors: Sequence[Any]) -> list[Dict[str, Any]]:
    now = time.monotonic()
    candidates = []
    with self._lock:
      for i, actor in enumerate(actors):
        name = self._name_for(actor, i)
        snapshot = self._snapshots.get(name)
        if snapshot is None or now - snapshot[1] > self._stats_ttl_s:
          attributes = {
              "state": "UNKNOWN",
              "policy_version": 0,
              "queue_len": 0,
              "routing_stats": {
                  "num_running_reqs": 0,
                  "num_waiting_reqs": 0,
                  "kv": 0.0,
              },
          }
        else:
          attributes = snapshot[0]
        candidates.append({"name": name, "attributes": attributes})
    return candidates

  # --------------------------------------------------------------- routing

  def __call__(
      self,
      actors: Sequence[Any],
      method_name: str | None,
      args: Sequence[Any],
      kwargs: Dict[str, Any],
  ) -> Any:
    kwargs = kwargs or {}
    self._ensure_poller(actors)
    candidates = self._candidates(actors)
    self._request_seq += 1
    payload = {
        "request_id": str(
            kwargs.get("request_id") or f"req-{self._request_seq}"
        ),
        "target_model": self._target_model,
        "prompt": str(kwargs.get("prompt") or ""),
        "candidates": candidates,
    }
    try:
      request = urllib.request.Request(
          f"{self._scheduler_url}/schedule",
          data=json.dumps(payload).encode("utf-8"),
          headers={"Content-Type": "application/json"},
          method="POST",
      )
      with urllib.request.urlopen(
          request, timeout=self._http_timeout_s
      ) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # pylint: disable=broad-except
      self._warn_failure(f"scheduler request failed: {e}")
      return self._fallback(actors, kwargs)

    picked = body.get("picked")
    if not picked or body.get("fallback"):
      self._warn_failure("scheduler declined to pick (fallback response)")
      return self._fallback(actors, kwargs)
    for i, actor in enumerate(actors):
      if self._name_for(actor, i) == picked:
        self._failures = 0
        return actor
    self._warn_failure(f"scheduler picked unknown worker '{picked}'")
    return self._fallback(actors, kwargs)

  def _fallback(self, actors: Sequence[Any], kwargs: Dict[str, Any]) -> Any:
    route_key = kwargs.get("route_key")
    if route_key is not None:
      return actors[hash(route_key) % len(actors)]
    actor = actors[self._rr_idx % len(actors)]
    self._rr_idx += 1
    return actor

  def _warn_failure(self, message: str) -> None:
    self._failures += 1
    if self._failures == 1 or self._failures % _WARN_EVERY == 0:
      logging.warning(
          "RemoteSchedulerRouter falling back to local routing (%d failures):"
          " %s",
          self._failures,
          message,
      )

  # -------------------------------------------------------------- lifecycle

  def notify_weights_synced(self) -> None:
    """Best-effort prefix-cache invalidation on the sidecar after weight sync."""
    try:
      request = urllib.request.Request(
          f"{self._scheduler_url}/reset", data=b"{}", method="POST"
      )
      with urllib.request.urlopen(request, timeout=self._http_timeout_s):
        pass
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Scheduler /reset failed (ignored): %s", e)

  def stop(self) -> None:
    self._stop_event.set()
    poller = self._poller
    if poller is not None and poller.is_alive():
      poller.join(timeout=2 * self._poll_interval_s + 1.0)
    self._poller = None
