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
import logging
import os
import threading
from typing import Any, Callable
import numpy as np

_GLOBAL_FLEET = None
_FLEET_LOCK = threading.Lock()
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

        huggingface_hub.HfFolder = DummyHfFolder  # pyrefly: ignore[bad-assignment]
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


def normalize_tasks_for_fleet(
    tasks: Any,
    scaffold: str = "r2egym",
    image_rewrite: Any | None = None,
) -> list[Any]:
  """Normalize heterogeneous dataset entries into Task objects for SandboxFleet."""
  del scaffold
  try:
    from agent_sandbox_rl import Task  # pyrefly: ignore[missing-import]
  except ImportError:
    return list(tasks)

  rewrite_fn = get_image_rewrite_fn(image_rewrite)
  normalized = []
  for item in tasks:
    if hasattr(item, "image") and hasattr(item, "id"):
      normalized.append(item)
    elif isinstance(item, dict):
      img = extract_task_image(item) or "default"
      if isinstance(img, (list, np.ndarray)):
        img = img[0] if len(img) > 0 else "default"
      str_img = img.decode("utf-8") if hasattr(img, "decode") else str(img)
      if rewrite_fn is not None:
        str_img = rewrite_fn(str_img)

      t_id = (
          item.get("instance_id")
          or item.get("id")
          or item.get("prompt_id")
          or str_img
      )
      if isinstance(t_id, (list, np.ndarray)):
        t_id = t_id[0] if len(t_id) > 0 else "default"
      str_id = t_id.decode("utf-8") if hasattr(t_id, "decode") else str(t_id)
      normalized.append(Task(id=str_id, image=str_img, metadata={"ds": item}))
    else:
      normalized.append(item)
  return normalized


def init_global_fleet(
    tasks: list[Any] | None = None,
    max_concurrency: int = 128,
    num_generations: int = 8,
    batch_size: int = 8,
    max_warmpool_replicas: int | None = None,
    namespace: str | None = None,
    scaffold: str = "r2egym",
    image_rewrite: Any | None = None,
    node_selector: dict[str, str] | None = None,
) -> Any:
  """Initialize the process-wide SandboxFleet instance once upfront."""
  global _GLOBAL_FLEET
  with _FLEET_LOCK:
    if _GLOBAL_FLEET is not None:
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

    scaffold_env = os.getenv("SCAFFOLD")
    if scaffold == "r2egym" and scaffold_env:
      scaffold = scaffold_env
    elif not scaffold:
      scaffold = scaffold_env or "r2egym"

    fleet_ns = namespace or os.getenv("NAMESPACE", "rl-tunix-swebench")
    if node_selector is None:
      key = os.environ.get("NODE_SELECTOR_KEY")
      val = os.environ.get("NODE_SELECTOR_VAL")
      node_sel = {key: val} if (key and val) else None
    else:
      node_sel = node_selector

    logging.info(
        "[SandboxFleet] Initializing SandboxFleet in namespace '%s' with"
        " node_selector=%s",
        fleet_ns,
        node_sel,
    )

    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    effective_max_concurrent = max(
        max_concurrency, batch_size * num_generations * 2
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
        "max_warmpool_size": (
            max_warmpool_replicas
            if max_warmpool_replicas is not None
            else num_generations
        ),
        "warm_per_task": True,
    }

    try:
      from examples.deepswe import template as template_mod  # pyrefly: ignore[missing-import]

      template = template_mod.get_template(scaffold, node_sel)
      if template is not None:
        fleet_kwargs["template"] = template
      if scaffold == "openhands":
        fleet_kwargs["template_name_prefix"] = "oh-img-"
    except (ImportError, AttributeError):
      pass

    fleet_cfg = FleetConfig(**fleet_kwargs)
    fleet_inst = SandboxFleet(fleet_cfg)

    image_rewrite_fn = get_image_rewrite_fn(image_rewrite)
    fleet_inst._image_rewrite_fn = image_rewrite_fn

    if tasks is not None:
      normalized_tasks = normalize_tasks_for_fleet(
          tasks, scaffold=scaffold, image_rewrite=image_rewrite_fn
      )
      if image_rewrite_fn is not None:
        try:
          fleet_inst.load_tasks(
              normalized_tasks, image_rewrite=image_rewrite_fn
          )
        except TypeError:
          fleet_inst.load_tasks(normalized_tasks)
      else:
        fleet_inst.load_tasks(normalized_tasks)

    msg = (
        f"[SandboxFleet] Initializing pipelined fleet in namespace={fleet_ns}"
        f" (max_concurrent={effective_max_concurrent},"
        f" window_size={batch_size},"
        f" max_warmpool_replicas={fleet_kwargs['max_warmpool_size']},"
        " warm_per_task=True)..."
    )
    logging.info(msg)
    if getattr(fleet_cfg, "install_teardown_hooks", False):
      fleet_inst._install_teardown_hooks()
    fleet_inst._torndown = False
    if hasattr(fleet_inst, "preflight"):
      fleet_inst.preflight()
    if hasattr(fleet_inst, "plan"):
      fleet_inst.plan()
      entries = (
          getattr(getattr(fleet_inst, "plan_", None), "entries", None) or []
      )
      images = [e.image for e in entries]
      if images and hasattr(fleet_inst, "warm_images"):
        target_replicas = fleet_kwargs["max_warmpool_size"]
        fleet_inst.warm_images(
            images, replicas_override=target_replicas, wait=False
        )
        logging.info(
            "[SandboxFleet] Started initial warmpools for %d image(s) (%d"
            " replicas each).",
            len(images),
            target_replicas,
        )
    _GLOBAL_FLEET = fleet_inst
    atexit.register(teardown_global_fleet)
    return _GLOBAL_FLEET


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
      unwarm_on_exhaustion: bool = False,
      scaffold: str = "r2egym",
      image_rewrite: Any | None = None,
      wait_initial: bool = True,
  ):
    del lookahead_steps
    self.scaffold = scaffold
    self.dataset_iter = iter(dataset)
    self.fleet = fleet or get_global_fleet()
    self.num_generations = num_generations
    self.batch_size = max(1, batch_size)
    self.max_warmpool_replicas = max_warmpool_replicas
    self.unwarm_on_exhaustion = unwarm_on_exhaustion
    self.wait_initial = wait_initial
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
    self._previous_batch_counts: dict[str, int] = {}
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
          "[PrewarmDatasetIterator] Priming initial sandboxes on K8s (wait=%s)...",
          self.wait_initial,
      )
      self._interact_fleet(wait=self.wait_initial)

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
    while current_samples < self.batch_size and not self._exhausted:
      try:
        item = next(self.dataset_iter)
      except StopIteration:
        self._exhausted = True
        break
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
    return current_samples

  def _update_image_counts(self) -> None:
    """Updates dict to maintain samples of active batches."""
    self._image_counts.clear()
    for img, count in self._previous_batch_counts.items():
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
      # Retain its counts in _previous_batch_counts so its warm pool stays alive
      # while workers connect and claim sandboxes.
      self._previous_batch_counts = self._current_batch_counts

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
    self._previous_batch_counts.clear()
    self._current_batch_counts.clear()
    self._next_batch_counts.clear()
    self.current_batch.clear()
    self.next_batch.clear()
