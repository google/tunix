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

import collections
from collections.abc import Callable, Iterable, Iterator
import hashlib
import logging
import os
import re
import threading
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PlanEntryLike(Protocol):
  """Typed protocol for `agent_sandbox_rl.fleet.PlanEntry`."""

  image: str
  replicas: int


@runtime_checkable
class FleetPlanLike(Protocol):
  """Typed protocol for `agent_sandbox_rl.fleet.FleetPlan`."""

  entries: list[PlanEntryLike]

  def for_image(self, image: str) -> PlanEntryLike | None:
    ...


@runtime_checkable
class FleetConfigLike(Protocol):
  """Typed protocol for `agent_sandbox_rl.config.FleetConfig`."""

  max_warmpool_size: int
  install_teardown_hooks: bool
  template_name_prefix: str
  labels: dict[str, str]


@runtime_checkable
class SandboxFleetLike(Protocol):
  """Typed protocol for `agent_sandbox_rl.fleet.SandboxFleet` at commit 9afee7039342."""

  config: FleetConfigLike
  run_id: str
  tasks: list[object]
  plan_: FleetPlanLike | None

  def load_tasks(
      self,
      source: list[object],
      *,
      image_rewrite: Callable[[str], str] | None = None,
  ) -> list[object]:
    ...

  def preflight(self) -> dict[str, object]:
    ...

  def plan(self) -> FleetPlanLike:
    ...

  def warm_images(
      self,
      images: list[str],
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    ...

  def warm_image(
      self,
      image: str,
      *,
      replicas_override: int | None = None,
      wait: bool = True,
  ) -> None:
    ...

  def set_pool_replicas(self, image: str, replicas: int) -> None:
    ...

  def unwarm_image(self, image: str) -> None:
    ...

  def teardown(self) -> dict[str, object] | None:
    ...

  def _install_teardown_hooks(self) -> None:
    ...


_GLOBAL_FLEET: SandboxFleetLike | None = None
_GLOBAL_IMAGE_REWRITE_FN: Callable[[str], str] | None = None
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
          def get_token() -> None:
            return None

        huggingface_hub.HfFolder = DummyHfFolder
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[SandboxFleet] HfFolder patch note: %s", e)

  try:
    from r2egym.agenthub.runtime import docker as docker_mod  # pyrefly: ignore[missing-import]

    if not hasattr(docker_mod.DockerRuntime, "_orig_init"):
      docker_mod.DockerRuntime._orig_init = docker_mod.DockerRuntime.__init__

      def _patched_init(self, *args, **kwargs):
        if "backend" in kwargs and kwargs["backend"] == "kubernetes-sandbox":
          kwargs["backend"] = "kubernetes"
          self._actual_backend = "kubernetes-sandbox"
        return self._orig_init(*args, **kwargs)

      docker_mod.DockerRuntime.__init__ = _patched_init

    if not hasattr(docker_mod.DockerRuntime, "_orig_start_container"):
      docker_mod.DockerRuntime._orig_start_container = (
          docker_mod.DockerRuntime.start_container
      )

      def _patched_start_container(
          self, docker_image, command, ctr_name, **kwargs
      ):
        if (
            hasattr(self, "_actual_backend")
            and self._actual_backend == "kubernetes-sandbox"
        ) or (
            hasattr(self, "backend") and self.backend == "kubernetes-sandbox"
        ):
          return self._start_kubernetes_sandbox()
        return self._orig_start_container(
            docker_image, command, ctr_name, **kwargs
        )

      docker_mod.DockerRuntime.start_container = _patched_start_container
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug("[SandboxFleet] r2egym in-memory patch note: %s", e)

  patch_agent_sandbox_rl_templates()


def patch_agent_sandbox_rl_templates() -> None:
  """Patches generated SandboxTemplates to disable per-template NetworkPolicy churn."""
  try:
    from agent_sandbox_rl import resources as asrl_resources  # pyrefly: ignore[missing-import]

    if not hasattr(asrl_resources.Resources, "_orig_template_manifest") and (
        hasattr(asrl_resources.Resources, "_template_manifest")
    ):
      asrl_resources.Resources._orig_template_manifest = (
          asrl_resources.Resources._template_manifest
      )

      def _patched_template_manifest(self, image, template_name, template):
        manifest = self._orig_template_manifest(image, template_name, template)
        spec = manifest["spec"]
        spec["networkPolicyManagement"] = "Unmanaged"
        pod_spec = spec["podTemplate"]["spec"]
        pod_spec["automountServiceAccountToken"] = False
        pod_spec["shareProcessNamespace"] = False
        return manifest

      asrl_resources.Resources._template_manifest = _patched_template_manifest
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.debug(
        "[SandboxFleet] agent_sandbox_rl template patch note: %s", e
    )


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


def _coerce_str(val: object) -> str:
  if isinstance(val, (list, tuple, np.ndarray)):
    flat = list(np.asarray(val).reshape(-1))
    val = flat[0] if flat else "default"
  if isinstance(val, bytes):
    return val.decode("utf-8")
  return str(val)


def extract_task_image(item: dict[str, object]) -> object | None:
  """Extract `docker_image` or `image` from a dataset item dictionary."""
  if "docker_image" in item and item["docker_image"] is not None:
    return item["docker_image"]
  if "image" in item and item["image"] is not None:
    return item["image"]

  if "metadata" in item and isinstance(item["metadata"], dict):
    meta = item["metadata"]
    if "docker_image" in meta and meta["docker_image"] is not None:
      return meta["docker_image"]
    if "image" in meta and meta["image"] is not None:
      return meta["image"]
    if "env_config" in meta and isinstance(meta["env_config"], dict):
      env_cfg = meta["env_config"]
      if "entry" in env_cfg and isinstance(env_cfg["entry"], dict):
        entry = env_cfg["entry"]
        if "docker_image" in entry and entry["docker_image"] is not None:
          return entry["docker_image"]
        if "image" in entry and entry["image"] is not None:
          return entry["image"]

  if "env_config" in item and isinstance(item["env_config"], dict):
    env_cfg = item["env_config"]
    if "entry" in env_cfg and isinstance(env_cfg["entry"], dict):
      entry = env_cfg["entry"]
      if "docker_image" in entry and entry["docker_image"] is not None:
        return entry["docker_image"]
      if "image" in entry and entry["image"] is not None:
        return entry["image"]

  return None


def normalize_tasks_for_fleet(
    tasks: Iterable[object],
    scaffold: str = "r2egym",
    image_rewrite: Callable[[str], str] | None = None,
) -> list[object]:
  """Normalize dataset entries into `Task` objects for `SandboxFleet`."""
  del scaffold
  try:
    from agent_sandbox_rl import Task  # pyrefly: ignore[missing-import]
  except ImportError:
    return list(tasks)

  rewrite_fn = get_image_rewrite_fn(image_rewrite)
  normalized: list[object] = []
  for item in tasks:
    if isinstance(item, dict):
      raw_img = extract_task_image(item)
      str_img = _coerce_str(raw_img) if raw_img is not None else "default"
      if rewrite_fn is not None:
        str_img = rewrite_fn(str_img)

      raw_id: object = str_img
      for key in ("instance_id", "id", "prompt_id"):
        if key in item and item[key]:
          raw_id = item[key]
          break
      str_id = _coerce_str(raw_id)
      normalized.append(Task(id=str_id, image=str_img, metadata={"ds": item}))
    else:
      normalized.append(item)
  return normalized


def derive_job_run_id_and_prefix(job_id: str) -> tuple[str, str]:
  """Derives a deterministic 8-char hex `run_id` and DNS-1123 `template_name_prefix`."""
  stripped = job_id.strip()
  if not stripped:
    raise ValueError("job_id must be a non-empty string when provided.")
  clean_job = (
      re.sub(r"[^a-z0-9-]+", "-", stripped.removesuffix("-orch").lower())
      .strip("-")[:24]
      .rstrip("-")
  )
  if not clean_job:
    raise ValueError(
        f"job_id {job_id!r} produced an empty DNS-1123 prefix after"
        " sanitization."
    )
  shared_run_id = hashlib.sha256(stripped.encode("utf-8")).hexdigest()[:8]
  return shared_run_id, f"r2e-{clean_job}-"


def _task_image_str(task: object) -> str | None:
  """Extracts the normalized image string from a `Task` object or dict."""
  if hasattr(task, "image"):
    img = getattr(task, "image")
    return str(img) if img is not None else None
  if isinstance(task, dict):
    raw = extract_task_image(task)
    return _coerce_str(raw) if raw is not None else None
  return None


def ensure_tasks_in_fleet_plan(
    fleet: SandboxFleetLike,
    tasks: Iterable[object],
    *,
    scaffold: str = "r2egym",
    image_rewrite: Callable[[str], str] | None = None,
) -> None:
  """Registers `tasks` in `fleet.plan_` so `acquire()` hits warm pools (`on_demand=False`)."""
  rewrite_fn = get_image_rewrite_fn(image_rewrite or _GLOBAL_IMAGE_REWRITE_FN)
  normalized_tasks = normalize_tasks_for_fleet(
      tasks, scaffold=scaffold, image_rewrite=rewrite_fn
  )
  if not normalized_tasks:
    return
  needs_replan = fleet.plan_ is None
  if not needs_replan and fleet.plan_ is not None:
    for t in normalized_tasks:
      img = _task_image_str(t)
      if img and fleet.plan_.for_image(img) is None:
        needs_replan = True
        break
  if needs_replan:
    existing = list(fleet.tasks) if fleet.tasks else []
    fleet.load_tasks(existing + normalized_tasks)
    fleet.plan()


def init_global_fleet(
    tasks: list[object] | None = None,
    max_concurrency: int = 128,
    num_generations: int = 8,
    batch_size: int = 8,
    max_warmpool_replicas: int | None = None,
    namespace: str | None = None,
    scaffold: str = "r2egym",
    image_rewrite: Callable[[str], str] | None = None,
    node_selector: dict[str, str] | None = None,
    job_id: str | None = None,
    warm_on_init: bool = True,
    teardown_on_exit: bool = True,
) -> SandboxFleetLike:
  """Initialize or reuse the process-wide `SandboxFleet` instance."""
  global _GLOBAL_FLEET, _GLOBAL_IMAGE_REWRITE_FN
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
          " git+https://github.com/kubernetes-sigs/agent-sandbox.git@9afee7039342f18fa9ad402929cabf8b66eebaa7#subdirectory=examples/agent-sandbox-rl"
      ) from e

    fleet_ns = namespace or os.getenv("NAMESPACE", "rl-tunix-swebench")
    if node_selector is None:
      key = os.environ.get("NODE_SELECTOR_KEY")
      val = os.environ.get("NODE_SELECTOR_VAL")
      node_sel = {key: val} if (key and val) else None
    else:
      node_sel = node_selector

    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    effective_max_concurrent = max(
        max_concurrency, batch_size * num_generations * 2
    )
    target_warmpool_size = (
        max_warmpool_replicas
        if max_warmpool_replicas is not None
        else num_generations
    )

    fleet_kwargs: dict[str, object] = {
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
    shared_run_id = ""
    template_prefix = ""
    if job_id is not None:
      shared_run_id, template_prefix = derive_job_run_id_and_prefix(job_id)
      fleet_kwargs["run_id"] = shared_run_id
      fleet_kwargs["template_name_prefix"] = template_prefix
      fleet_kwargs["labels"] = {
          "app": "agent-sandbox-rl",
          "agent-sandbox-rl-run": shared_run_id,
          "app.kubernetes.io/created-by": job_id.strip(),
      }
      logging.info(
          "[SandboxFleet] Initializing job-scoped SandboxFleet in namespace"
          " '%s' (job_id=%s, run_id=%s, template_name_prefix=%s,"
          " node_selector=%s, warm_on_init=%s, teardown_on_exit=%s)",
          fleet_ns,
          job_id,
          shared_run_id,
          template_prefix,
          node_sel,
          warm_on_init,
          teardown_on_exit,
      )
    else:
      logging.info(
          "[SandboxFleet] Initializing SandboxFleet in namespace '%s' with"
          " node_selector=%s",
          fleet_ns,
          node_sel,
      )

    fleet_cfg = FleetConfig(**fleet_kwargs)
    fleet_inst: SandboxFleetLike = SandboxFleet(fleet_cfg)
    if job_id is not None and (
        fleet_inst.run_id != shared_run_id
        or fleet_inst.config.template_name_prefix != template_prefix
    ):
      raise RuntimeError(
          "SandboxFleet did not honor shared run_id/template_name_prefix:"
          f" got run_id={fleet_inst.run_id!r} (expected {shared_run_id!r}),"
          " template_name_prefix="
          f"{fleet_inst.config.template_name_prefix!r} (expected"
          f" {template_prefix!r})."
      )

    image_rewrite_fn = get_image_rewrite_fn(image_rewrite)
    _GLOBAL_IMAGE_REWRITE_FN = image_rewrite_fn
    # Exposed for `SWEEnv._init_agent_sandbox_env` in `oss/examples/deepswe`.
    setattr(fleet_inst, "_image_rewrite_fn", image_rewrite_fn)

    if tasks is not None:
      normalized_tasks = normalize_tasks_for_fleet(
          tasks, scaffold=scaffold, image_rewrite=image_rewrite_fn
      )
      fleet_inst.load_tasks(normalized_tasks)

    if fleet_cfg.install_teardown_hooks:
      fleet_inst._install_teardown_hooks()
    setattr(fleet_inst, "_torndown", False)

    if tasks is not None:
      plan = fleet_inst.plan()
      if warm_on_init:
        images = [e.image for e in plan.entries]
        if images:
          fleet_inst.warm_images(
              images, replicas_override=target_warmpool_size, wait=False
          )
          logging.info(
              "[SandboxFleet] Started initial warmpools for %d image(s) (%d"
              " replicas each).",
              len(images),
              target_warmpool_size,
          )

    _GLOBAL_FLEET = fleet_inst
    return _GLOBAL_FLEET


def get_global_fleet() -> SandboxFleetLike:
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
    finally:
      _GLOBAL_FLEET = None


class PrewarmDatasetIterator:
  """Lookahead dataset iterator: pre-warms Agent Sandboxes on Kubernetes.

  Maintains two queues:
    - current_batch: samples for the current batch being processed.
    - next_batch: samples for the next batch being pre-warmed.
  A dictionary maintains the sample counts of active batches (including up to
  `max_staleness + 1` in-flight previous batches).
  After the dictionary is updated, we interact with the fleet:
    * New image: fleet.warm_image(img, replicas_override=replicas, wait=wait)
    * Changed count: fleet.set_pool_replicas(img, replicas)
    * Deleted image key (count 0): fleet.unwarm_image(img)
  Cleans up all warm pools upon iteration completion or close().
  """

  def __init__(
      self,
      dataset: Iterable[object],
      fleet: SandboxFleetLike | None = None,
      num_generations: int = 8,
      batch_size: int = 8,
      lookahead_steps: int = 1,
      max_warmpool_replicas: int | None = None,
      max_staleness: int = 0,
      unwarm_on_exhaustion: bool = False,
      scaffold: str = "r2egym",
      image_rewrite: Callable[[str], str] | None = None,
  ):
    del lookahead_steps
    self.scaffold = scaffold
    self.dataset_iter: Iterator[object] = iter(dataset)
    self.fleet: SandboxFleetLike = (
        fleet if fleet is not None else get_global_fleet()
    )
    self.num_generations = num_generations
    self.batch_size = max(1, batch_size)
    self.max_warmpool_replicas = max_warmpool_replicas
    self.max_staleness = max(0, max_staleness)
    self.unwarm_on_exhaustion = unwarm_on_exhaustion
    self.image_rewrite = get_image_rewrite_fn(
        image_rewrite or _GLOBAL_IMAGE_REWRITE_FN
    )

    self.current_batch: collections.deque[
        tuple[object, dict[str, int], int]
    ] = collections.deque()
    self.next_batch: collections.deque[
        tuple[object, dict[str, int], int]
    ] = collections.deque()
    self._current_batch_counts: dict[str, int] = {}
    self._next_batch_counts: dict[str, int] = {}
    self._previous_batches: collections.deque[dict[str, int]] = (
        collections.deque(maxlen=max(1, self.max_staleness + 1))
    )
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
          "[PrewarmDatasetIterator] Priming initial sandboxes on K8s..."
      )
      self._interact_fleet(wait=False)

  def _extract_item_image_counts(self, item: object) -> dict[str, int]:
    """Extracts a dict mapping docker_image -> count for a dataset item."""
    counts: dict[str, int] = {}
    if item is None:
      return counts
    if isinstance(item, (list, tuple)):
      for sub in item:
        for img, cnt in self._extract_item_image_counts(sub).items():
          counts[img] = counts.get(img, 0) + cnt
      return counts

    raw_images: list[object] = []
    if isinstance(item, dict):
      found = extract_task_image(item)
      if found is not None:
        if isinstance(found, (list, tuple, np.ndarray)):
          raw_images.extend(np.asarray(found).reshape(-1).tolist())
        elif isinstance(found, (str, bytes)):
          raw_images.append(found)

    rewrite_fn = self.image_rewrite or get_image_rewrite_fn()
    for img in raw_images:
      if img is not None:
        str_img = _coerce_str(img)
        if rewrite_fn is not None:
          str_img = rewrite_fn(str_img)
        counts[str_img] = counts.get(str_img, 0) + 1
    return counts

  def _extract_images(self, batch: object) -> list[str]:
    """Extracts unique docker image strings from a batch."""
    counts = self._extract_item_image_counts(batch)
    return list(counts.keys())

  def _fill_batch(
      self,
      queue: collections.deque[tuple[object, dict[str, int], int]],
      counts_dict: dict[str, int],
  ) -> int:
    """Consumes from dataset and adds item(s) to queue until samples >= batch_size."""
    current_samples = sum(s_cnt for _, _, s_cnt in queue)
    new_items: list[object] = []
    while current_samples < self.batch_size and not self._exhausted:
      try:
        item = next(self.dataset_iter)
      except StopIteration:
        self._exhausted = True
        break
      new_items.append(item)
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
    if new_items:
      flat_items: list[object] = []
      for it in new_items:
        if isinstance(it, (list, tuple)):
          flat_items.extend(it)
        else:
          flat_items.append(it)
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
    for img, count in self._previous_batch_counts.items():
      self._image_counts[img] = self._image_counts.get(img, 0) + count
    for img, count in self._current_batch_counts.items():
      self._image_counts[img] = self._image_counts.get(img, 0) + count
    for img, count in self._next_batch_counts.items():
      self._image_counts[img] = self._image_counts.get(img, 0) + count

  def _interact_fleet(self, wait: bool = False) -> None:
    """Interacts with the fleet to reconcile warm pools with self._image_counts."""
    desired: dict[str, int] = {}
    for img, count in self._image_counts.items():
      if count > 0:
        reps = count * self.num_generations
        if self.max_warmpool_replicas is not None:
          reps = min(reps, self.max_warmpool_replicas)
        desired[img] = reps

    # 1. Warm new keys or scale existing keys (fail loud on errors).
    for img, target_reps in desired.items():
      if img not in self._active_replicas:
        self.fleet.warm_image(img, replicas_override=target_reps, wait=wait)
        self._active_replicas[img] = target_reps
        logging.info(
            "[PrewarmDatasetIterator] Warmed new pool on K8s: %s"
            " (replicas=%d, wait=%s)",
            img,
            target_reps,
            wait,
        )
      elif self._active_replicas[img] != target_reps:
        self.fleet.set_pool_replicas(img, target_reps)
        logging.info(
            "[PrewarmDatasetIterator] Scaled pool on K8s: %s (replicas %d ->"
            " %d)",
            img,
            self._active_replicas[img],
            target_reps,
        )
        self._active_replicas[img] = target_reps

    # 2. Delete / unwarm keys no longer in desired (fail loud on errors).
    to_delete = [img for img in self._active_replicas if img not in desired]
    for img in to_delete:
      self.fleet.unwarm_image(img)
      self.unwarm_calls.append(img)
      logging.info(
          "[PrewarmDatasetIterator] Unwarmed retired pool on K8s: %s", img
      )
      del self._active_replicas[img]

  def __iter__(self) -> PrewarmDatasetIterator:
    return self

  def __next__(self) -> object:
    if not self.current_batch:
      if not self.next_batch:
        if self.unwarm_on_exhaustion:
          self.close()
        raise StopIteration

      # The previous current_batch is now in-flight/running on the cluster.
      # Retain up to (max_staleness + 1) previous batches so their warm pools
      # stay alive while workers connect and claim sandboxes.
      if self._current_batch_counts:
        self._previous_batches.append(self._current_batch_counts)
      self._previous_batch_counts = {}
      for prev_counts in self._previous_batches:
        for img, count in prev_counts.items():
          self._previous_batch_counts[img] = (
              self._previous_batch_counts.get(img, 0) + count
          )

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
      self.fleet.unwarm_image(img)
      self.unwarm_calls.append(img)
      logging.info(
          "[PrewarmDatasetIterator] Cleaned up warm pool on K8s: %s", img
      )
    self._active_replicas.clear()
    self._image_counts.clear()
    self._previous_batches.clear()
    self._previous_batch_counts.clear()
    self._current_batch_counts.clear()
    self._next_batch_counts.clear()
    self.current_batch.clear()
    self.next_batch.clear()
