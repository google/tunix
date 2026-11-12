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

"""Utilities for managing Kubernetes Agent Sandboxes and lookahead pre-warming."""

# pylint: disable=g-import-not-at-top,protected-access

from __future__ import annotations

import atexit
import collections
import dataclasses
import hashlib
import logging
import os
import re
import threading
from typing import Any, Callable
import numpy as np


@dataclasses.dataclass(frozen=True)
class _FallbackTask:
  """Minimal typed task record when `agent_sandbox_rl` is not installed."""

  id: str
  image: str
  metadata: dict[str, Any]


_GLOBAL_FLEET = None
_FLEET_LOCK = threading.RLock()
_PATCH_LOCK = threading.Lock()
_R2EGYM_PATCHED = False


def patch_r2egym_for_agent_sandbox() -> None:
  """In-memory compatibility patch for r2egym to route 'kubernetes-sandbox' backend."""
  global _R2EGYM_PATCHED
  with _PATCH_LOCK:
    if _R2EGYM_PATCHED:
      return
    _R2EGYM_PATCHED = True
  try:
    import huggingface_hub  # pyrefly: ignore[missing-import]

    if not hasattr(huggingface_hub, "HfFolder"):
      try:
        from huggingface_hub._login import HfFolder  # pyrefly: ignore[missing-import]

        huggingface_hub.HfFolder = HfFolder
      except Exception:  # pylint: disable=broad-exception-caught

        class DummyHfFolder:

          @staticmethod
          def get_token():
            return None

        huggingface_hub.HfFolder = DummyHfFolder
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[SandboxFleet] HfFolder patch note: %s", e)

  try:
    from r2egym.agenthub.runtime import docker as docker_mod  # pyrefly: ignore[missing-import]

    orig_init = getattr(docker_mod.DockerRuntime, "_orig_init", None)
    if orig_init is None:
      docker_mod.DockerRuntime._orig_init = docker_mod.DockerRuntime.__init__

      def _patched_init(self, *args, **kwargs):
        if kwargs.get("backend") == "kubernetes-sandbox":
          kwargs["backend"] = "kubernetes"
          self._actual_backend = "kubernetes-sandbox"
        return self._orig_init(*args, **kwargs)

      docker_mod.DockerRuntime.__init__ = _patched_init

    orig_start = getattr(
        docker_mod.DockerRuntime, "_orig_start_container", None
    )
    if orig_start is None:
      docker_mod.DockerRuntime._orig_start_container = (
          docker_mod.DockerRuntime.start_container
      )

      def _patched_start_container(
          self, docker_image, command, ctr_name, **kwargs
      ):
        if (
            getattr(self, "_actual_backend", None) == "kubernetes-sandbox"
            or getattr(self, "backend", None) == "kubernetes-sandbox"
        ):
          return self._start_kubernetes_sandbox()
        return self._orig_start_container(
            docker_image, command, ctr_name, **kwargs
        )

      docker_mod.DockerRuntime.start_container = _patched_start_container
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[SandboxFleet] r2egym in-memory patch note: %s", e)


def get_image_rewrite_fn(
    image_rewrite: Callable[[str], str] | None = None,
) -> Callable[[str], str] | None:
  """Retrieve or construct the image rewrite function from prefix if configured."""
  if image_rewrite is not None:
    return image_rewrite
  prefix = os.getenv("IMAGE_REWRITE_PREFIX")
  if prefix:
    prefix = prefix.rstrip("/")
    return lambda img: f"{prefix}/{img.split('/')[-1]}"
  return None


def extract_task_image(item: dict[str, Any]) -> Any | None:
  """Safely extract docker_image or image name from an item dictionary."""
  # 1. Direct docker_image / image
  img = item.get("docker_image")
  if img is None:
    img = item.get("image")
  if img is not None:
    return img

  # 2. Inside metadata
  meta = item.get("metadata")
  if isinstance(meta, dict):
    img = meta.get("docker_image")
    if img is None:
      img = meta.get("image")
    if img is not None:
      return img
    env_cfg = meta.get("env_config")
    if isinstance(env_cfg, dict):
      entry = env_cfg.get("entry")
      if isinstance(entry, dict):
        img = entry.get("docker_image")
        if img is None:
          img = entry.get("image")
        if img is not None:
          return img

  # 3. Inside env_config
  env_cfg = item.get("env_config")
  if isinstance(env_cfg, dict):
    entry = env_cfg.get("entry")
    if isinstance(entry, dict):
      img = entry.get("docker_image")
      if img is None:
        img = entry.get("image")
      if img is not None:
        return img

  return None


def _decode_scalar(val: Any, default: str) -> str:
  """Extracts a single string from a scalar, bytes, sequence, or ndarray."""
  if isinstance(val, (list, tuple, np.ndarray)):
    flat = np.asarray(val).reshape(-1).tolist()
    val = flat[0] if flat else None
  if val is None:
    return default
  decoded = val.decode("utf-8") if isinstance(val, bytes) else str(val)
  return decoded or default


def normalize_tasks_for_fleet(
    tasks: Any,
    scaffold: str = "r2egym",
    image_rewrite: Any | None = None,
) -> list[Any]:
  """Normalize heterogeneous dataset entries into Task objects for SandboxFleet."""
  del scaffold
  try:
    from agent_sandbox_rl import Task as task_cls  # pyrefly: ignore[missing-import]
  except ImportError:
    task_cls = _FallbackTask

  rewrite_fn = get_image_rewrite_fn(image_rewrite)
  normalized = []
  for item in tasks:
    if not isinstance(item, dict):
      normalized.append(item)
      continue
    str_img = _decode_scalar(extract_task_image(item), "default")
    if rewrite_fn is not None:
      str_img = rewrite_fn(str_img)
    raw_id = next(
        (
            item[k]
            for k in ("instance_id", "id", "prompt_id")
            if item.get(k) is not None
        ),
        None,
    )
    str_id = _decode_scalar(raw_id, str_img)
    normalized.append(task_cls(id=str_id, image=str_img, metadata={"ds": item}))
  return normalized


def ensure_tasks_in_fleet_plan(
    fleet: Any,
    tasks: list[Any] | None,
    *,
    scaffold: str = "r2egym",
    image_rewrite: Callable[[str], str] | None = None,
) -> None:
  """Registers `tasks` in `fleet.plan_` so `acquire()` hits warm pools (`on_demand=False`)."""
  if fleet is None or not tasks:
    return
  with _FLEET_LOCK:
    image_rewrite_fn = get_image_rewrite_fn(
        image_rewrite or fleet._image_rewrite_fn
    )
    normalized_tasks = normalize_tasks_for_fleet(
        tasks, scaffold=scaffold, image_rewrite=image_rewrite_fn
    )
    if not normalized_tasks:
      return

    plan = fleet.plan_
    if plan is not None:
      missing = [t for t in normalized_tasks if plan.for_image(t.image) is None]
      if not missing:
        return
    else:
      missing = list(normalized_tasks)

    existing_tasks = list(fleet.tasks) if fleet.tasks else []
    existing_images = {t.image for t in existing_tasks}
    new_tasks = []
    for t in missing:
      if t.image not in existing_images:
        new_tasks.append(t)
        existing_images.add(t.image)
    if not new_tasks and plan is not None:
      return

    fleet.load_tasks(existing_tasks + new_tasks)
    fleet.plan()


def _job_fleet_scope(job_id: str, scaffold: str) -> tuple[str, str]:
  """Derives a deterministic 8-char hex `run_id` and DNS-1123 template prefix."""
  stripped = job_id.strip()
  clean_job = (
      re.sub(r"[^a-z0-9-]+", "-", stripped.removesuffix("-orch").lower())
      .strip("-")[:24]
      .rstrip("-")
  )
  if not clean_job:
    raise ValueError(
        f"job_id {job_id!r} must contain at least one alphanumeric character."
    )
  run_id = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:8]
  prefix = "swe" if scaffold == "sweagent" else "r2e"
  return run_id, f"{prefix}-{clean_job}-"


def init_global_fleet(
    tasks: list[Any] | None = None,
    max_concurrency: int = 128,
    num_generations: int = 8,
    batch_size: int = 8,
    max_warmpool_replicas: int | None = None,
    namespace: str | None = None,
    scaffold: str = "r2egym",
    image_rewrite: Callable[[str], str] | None = None,
    node_selector: dict[str, str] | None = None,
    job_id: str | None = None,
    warm_on_init: bool = False,
    teardown_on_exit: bool = True,
) -> Any:
  """Initialize the process-wide SandboxFleet instance once upfront."""
  global _GLOBAL_FLEET
  with _FLEET_LOCK:
    if _GLOBAL_FLEET is not None:
      if tasks is not None:
        ensure_tasks_in_fleet_plan(
            _GLOBAL_FLEET,
            tasks,
            scaffold=scaffold,
            image_rewrite=image_rewrite,
        )
      return _GLOBAL_FLEET

    patch_r2egym_for_agent_sandbox()

    try:
      from agent_sandbox_rl import ClusterConfig  # pyrefly: ignore[missing-import]
      from agent_sandbox_rl import FleetConfig  # pyrefly: ignore[missing-import]
      from agent_sandbox_rl import SandboxFleet  # pyrefly: ignore[missing-import]
    except ImportError as e:
      raise ImportError(
          "use_agent_sandbox=True strictly requires the 'agent_sandbox_rl'"
          " package. Install via: pip install"
          " git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl"
      ) from e

    fleet_ns = namespace or os.getenv("NAMESPACE", "rl-tunix-swebench")
    key = os.environ.get("NODE_SELECTOR_KEY")
    val = os.environ.get("NODE_SELECTOR_VAL")
    node_sel = node_selector or ({key: val} if (key and val) else None)
    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    effective_max_concurrent = max(
        max_concurrency, batch_size * num_generations * 2
    )
    target_warmpool_size = (
        max_warmpool_replicas
        if max_warmpool_replicas is not None
        else num_generations
    )

    fleet_kwargs: dict[str, Any] = {
        "clusters": [
            ClusterConfig(
                name="default",
                namespace=fleet_ns,
                node_selector=node_sel,
                in_cluster=in_cluster,
            )
        ],
        "max_concurrent": effective_max_concurrent,
        "window_size": batch_size,
        "max_warmpool_size": target_warmpool_size,
        "warm_per_task": True,
        "install_teardown_hooks": teardown_on_exit,
    }
    if job_id is not None:
      shared_run_id, template_prefix = _job_fleet_scope(job_id, scaffold)
      fleet_kwargs["run_id"] = shared_run_id
      fleet_kwargs["template_name_prefix"] = template_prefix
      fleet_kwargs["labels"] = {
          "app": "agent-sandbox-rl",
          "agent-sandbox-rl-run": shared_run_id,
          "app.kubernetes.io/created-by": job_id.strip(),
      }

    fleet_cfg = FleetConfig(**fleet_kwargs)
    fleet_inst = SandboxFleet(fleet_cfg)

    image_rewrite_fn = get_image_rewrite_fn(image_rewrite)
    fleet_inst._image_rewrite_fn = image_rewrite_fn

    if tasks is not None:
      normalized_tasks = normalize_tasks_for_fleet(
          tasks, scaffold=scaffold, image_rewrite=image_rewrite_fn
      )
      fleet_inst.load_tasks(normalized_tasks)

    logging.info(
        "[SandboxFleet] Initialized fleet in namespace=%s (node_selector=%s,"
        " max_concurrent=%d, window_size=%d, max_warmpool_replicas=%d,"
        " job_id=%s).",
        fleet_ns,
        node_sel,
        effective_max_concurrent,
        batch_size,
        target_warmpool_size,
        job_id,
    )
    if fleet_cfg.install_teardown_hooks:
      fleet_inst._install_teardown_hooks()
    fleet_inst._torndown = False
    fleet_inst.plan()
    _GLOBAL_FLEET = fleet_inst
    if warm_on_init:
      warm_global_fleet()
    if teardown_on_exit:
      atexit.register(teardown_global_fleet)
    return _GLOBAL_FLEET


def warm_global_fleet() -> None:
  """Warms all planned task images in the global SandboxFleet and waits until ready."""
  fleet_inst = get_global_fleet()
  entries = fleet_inst.plan_.entries
  images = [e.image for e in entries]
  if images:
    target_replicas = fleet_inst.config.max_warmpool_size
    fleet_inst.warm_images(images, replicas_override=target_replicas, wait=True)
    logging.info(
        "[SandboxFleet] Started initial warmpools for %d image(s) (%s"
        " replicas each, wait=True).",
        len(images),
        target_replicas,
    )


def get_global_fleet() -> Any:
  """Retrieve the active process-wide SandboxFleet instance."""
  if _GLOBAL_FLEET is None:
    raise RuntimeError(
        "SandboxFleet has not been initialized. Call init_global_fleet() first."
    )
  return _GLOBAL_FLEET


def teardown_global_fleet() -> None:
  """Atexit handler to cleanly tear down warm pools on process exit."""
  global _GLOBAL_FLEET
  if _GLOBAL_FLEET is not None:
    logging.info(
        "[SandboxFleet] Automatically tearing down warm pools on exit..."
    )
    try:
      _GLOBAL_FLEET.teardown()
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("[SandboxFleet] Teardown note: %s", e)
    _GLOBAL_FLEET = None


class PrewarmDatasetIterator:
  """Lookahead dataset iterator: pre-warms Agent Sandboxes on Kubernetes.

  Maintains two queues:
    - current_batch: samples for the current batch being processed.
    - next_batch: samples for the next batch being pre-warmed.
  A dictionary maintains the sample counts of both queues.
  After the dictionary is updated, we interact with the fleet:
    * New image key: fleet.warm_image(img, replicas, wait=False)
    * Changed count: fleet.set_pool_replicas(img, replicas)
    * Deleted image key (count 0): fleet.unwarm_image(img)
  Cleans up all warm pools upon iteration completion or close().
  """

  def __init__(
      self,
      dataset: Any,
      fleet: Any | None = None,
      num_generations: int = 8,
      batch_size: int = 8,
      lookahead_steps: int = 1,
      max_warmpool_replicas: int | None = None,
      max_staleness: int = 0,
      unwarm_on_exhaustion: bool = False,
      scaffold: str = "r2egym",
      image_rewrite: Any | None = None,
  ):
    del lookahead_steps
    if (
        isinstance(max_staleness, bool)
        or not isinstance(max_staleness, int)
        or max_staleness < 0
    ):
      raise ValueError(
          f"max_staleness must be a non-negative int, got {max_staleness!r}."
      )
    self.scaffold = scaffold
    self.dataset_iter = iter(dataset)
    self.fleet = fleet or get_global_fleet()
    self.num_generations = num_generations
    self.batch_size = max(1, batch_size)
    self.max_warmpool_replicas = max_warmpool_replicas
    self.max_staleness = max_staleness
    self.unwarm_on_exhaustion = unwarm_on_exhaustion
    self.image_rewrite = get_image_rewrite_fn(
        image_rewrite or getattr(self.fleet, "_image_rewrite_fn", None)
    )

    self.current_batch: collections.deque[tuple[Any, dict[str, int], int]] = (
        collections.deque()
    )
    self.next_batch: collections.deque[tuple[Any, dict[str, int], int]] = (
        collections.deque()
    )
    self._current_batch_counts: dict[str, int] = {}
    self._next_batch_counts: dict[str, int] = {}
    self._previous_batches: collections.deque[dict[str, int]] = (
        collections.deque(maxlen=max_staleness + 1)
    )
    self._image_counts: dict[str, int] = {}
    self._active_replicas: dict[str, int] = {}
    self.unwarm_calls: list[str] = []
    self._exhausted = False

    # 1. Fill current_batch queue up to batch_size
    self._fill_batch(self.current_batch, self._current_batch_counts)

    # 2. Fill next_batch queue up to batch_size
    self._fill_batch(self.next_batch, self._next_batch_counts)

    # 3. Dict maintains the samples of active batches
    self._update_image_counts()

    # 4. After the dict updated, we interact the fleet
    if self._image_counts:
      logging.info(
          "[PrewarmDatasetIterator] Priming initial sandboxes on K8s..."
      )
      self._interact_fleet(wait=False)

  def _extract_item_image_counts(self, item: Any) -> dict[str, int]:
    """Extracts a dict mapping docker_image -> count for a dataset item."""
    counts: dict[str, int] = {}
    if item is None:
      return counts
    if isinstance(item, (list, tuple)):
      for sub in item:
        for img, cnt in self._extract_item_image_counts(sub).items():
          counts[img] = counts.get(img, 0) + cnt
      return counts

    raw_images = []
    if isinstance(item, dict):
      # 1. Top-level docker_image or image
      raw = item.get("docker_image")
      if raw is None:
        raw = item.get("image")
      if raw is not None:
        if isinstance(raw, (list, tuple, np.ndarray)):
          raw_images.extend(np.array(raw).flatten().tolist())
        elif isinstance(raw, (str, bytes)):
          raw_images.append(raw)

      # 2. Check metadata if raw_images is still empty
      if not raw_images:
        meta = item.get("metadata")
        if isinstance(meta, dict):
          raw = meta.get("docker_image")
          if raw is None:
            raw = meta.get("image")
          if raw is not None:
            if isinstance(raw, (list, tuple, np.ndarray)):
              raw_images.extend(np.array(raw).flatten().tolist())
            elif isinstance(raw, (str, bytes)):
              raw_images.append(raw)
          else:
            env_cfg = meta.get("env_config")
            if isinstance(env_cfg, dict):
              entry = env_cfg.get("entry")
              if isinstance(entry, dict):
                raw = entry.get("docker_image")
                if raw is None:
                  raw = entry.get("image")
                if raw is not None:
                  if isinstance(raw, (list, tuple, np.ndarray)):
                    raw_images.extend(np.array(raw).flatten().tolist())
                  elif isinstance(raw, (str, bytes)):
                    raw_images.append(raw)

      # 3. Check env_config if raw_images is still empty
      if not raw_images:
        env_cfg = item.get("env_config")
        if isinstance(env_cfg, dict):
          entry = env_cfg.get("entry")
          if isinstance(entry, dict):
            raw = entry.get("docker_image")
            if raw is None:
              raw = entry.get("image")
            if raw is not None:
              if isinstance(raw, (list, tuple, np.ndarray)):
                raw_images.extend(np.array(raw).flatten().tolist())
              elif isinstance(raw, (str, bytes)):
                raw_images.append(raw)

    rewrite_fn = getattr(self, "image_rewrite", None) or get_image_rewrite_fn()
    for img in raw_images:
      if img is not None:
        str_img = img.decode("utf-8") if hasattr(img, "decode") else str(img)
        if rewrite_fn is not None:
          str_img = rewrite_fn(str_img)
        counts[str_img] = counts.get(str_img, 0) + 1
    return counts

  def _extract_images(self, batch: Any) -> list[str]:
    """Extracts unique docker image strings from a batch."""
    counts = self._extract_item_image_counts(batch)
    return list(counts.keys())

  def _fill_batch(
      self,
      queue: collections.deque[tuple[Any, dict[str, int], int]],
      counts_dict: dict[str, int],
  ) -> int:
    """Consumes from dataset and adds item(s) to queue until samples >= batch_size."""
    current_samples = sum(s_cnt for _, _, s_cnt in queue)
    flat_items: list[Any] = []
    while current_samples < self.batch_size and not self._exhausted:
      try:
        item = next(self.dataset_iter)
      except StopIteration:
        self._exhausted = True
        break
      if isinstance(item, (list, tuple)):
        flat_items.extend(item)
      else:
        flat_items.append(item)
      item_counts = self._extract_item_image_counts(item)
      img_count_sum = sum(item_counts.values())
      if img_count_sum > 0:
        sample_count = img_count_sum
      elif isinstance(item, (list, tuple)):
        sample_count = max(1, len(item))
      else:
        sample_count = 1
      queue.append((item, item_counts, sample_count))
      for img, count in item_counts.items():
        counts_dict[img] = counts_dict.get(img, 0) + count
      current_samples += sample_count
    if flat_items:
      ensure_tasks_in_fleet_plan(
          self.fleet,
          flat_items,
          scaffold=self.scaffold,
          image_rewrite=self.image_rewrite,
      )
    return current_samples

  def _update_image_counts(self) -> None:
    """Updates dict to maintain samples of active batches."""
    self._image_counts.clear()
    for prev_counts in self._previous_batches:
      for img, count in prev_counts.items():
        self._image_counts[img] = self._image_counts.get(img, 0) + count
    for img, count in self._current_batch_counts.items():
      self._image_counts[img] = self._image_counts.get(img, 0) + count
    for img, count in self._next_batch_counts.items():
      self._image_counts[img] = self._image_counts.get(img, 0) + count

  def _interact_fleet(self, wait: bool = False) -> None:
    """Interacts with the fleet to reconcile warm pools with self._image_counts."""
    if not self.fleet:
      return

    desired: dict[str, int] = {}
    for img, count in self._image_counts.items():
      if count > 0:
        reps = count * self.num_generations
        if self.max_warmpool_replicas is not None:
          reps = min(reps, self.max_warmpool_replicas)
        desired[img] = reps

    # 1. Warm new keys or scale existing keys
    for img, target_reps in desired.items():
      if img not in self._active_replicas:
        try:
          self.fleet.warm_image(img, replicas_override=target_reps, wait=wait)
          self._active_replicas[img] = target_reps
          logging.info(
              "[PrewarmDatasetIterator] Warmed new pool on K8s: %s"
              " (replicas=%d, wait=%s)",
              img,
              target_reps,
              wait,
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning(
              "[PrewarmDatasetIterator] Warm note for %s: %s", img, e
          )
      elif self._active_replicas[img] != target_reps:
        try:
          self.fleet.set_pool_replicas(img, target_reps)
          logging.info(
              "[PrewarmDatasetIterator] Scaled pool on K8s: %s (replicas %d ->"
              " %d)",
              img,
              self._active_replicas[img],
              target_reps,
          )
          self._active_replicas[img] = target_reps
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning(
              "[PrewarmDatasetIterator] Set replicas note for %s: %s", img, e
          )

    # 2. Delete / unwarm keys no longer in desired
    to_delete = [img for img in self._active_replicas if img not in desired]
    for img in to_delete:
      try:
        self.fleet.unwarm_image(img)
        self.unwarm_calls.append(img)
        logging.info(
            "[PrewarmDatasetIterator] Unwarmed retired pool on K8s: %s", img
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "[PrewarmDatasetIterator] Unwarm note for %s: %s", img, e
        )
      del self._active_replicas[img]

  def __iter__(self):
    return self

  def __next__(self):
    if not self.current_batch:
      if not self.next_batch:
        if self.unwarm_on_exhaustion:
          self.close()
        raise StopIteration

      # The previous current_batch is now in-flight/running on the cluster.
      # Retain up to (max_staleness + 1) previous batches so their warm pools
      # stay alive while workers connect and claim sandboxes.
      self._previous_batches.append(self._current_batch_counts)

      # Shift next_batch to current_batch
      self.current_batch = self.next_batch
      self._current_batch_counts = self._next_batch_counts

      # Refill new next_batch from dataset
      self.next_batch = collections.deque()
      self._next_batch_counts = {}
      self._fill_batch(self.next_batch, self._next_batch_counts)

      # Dict maintains the samples of active batches
      self._update_image_counts()

      # After the dict updated, we interact the fleet
      self._interact_fleet(wait=False)

    item, _, _ = self.current_batch.popleft()
    return item

  def close(self) -> None:
    """Explicitly tears down active warm pools managed by this iterator."""
    for img in list(self._active_replicas):
      if self.fleet:
        try:
          self.fleet.unwarm_image(img)
          self.unwarm_calls.append(img)
          logging.info(
              "[PrewarmDatasetIterator] Cleaned up warm pool on K8s: %s", img
          )
        except Exception as e:  # pylint: disable=broad-exception-caught
          logging.warning("[PrewarmDatasetIterator] Final unwarm note: %s", e)
    self._active_replicas.clear()
    self._image_counts.clear()
    self._previous_batches.clear()
    self._current_batch_counts.clear()
    self._next_batch_counts.clear()
    self.current_batch.clear()
    self.next_batch.clear()
