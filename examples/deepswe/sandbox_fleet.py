"""Bounded Agent Sandbox Fleet integration for DeepSWE R2E environments.

This module is deliberately lazy: importing DeepSWE in the default ``direct``
mode must not import or require Agent Sandbox.  The Fleet owns only sandbox
lifecycle.  Prompt, tool, reward, trajectory, token, numerical, loss, backward,
and optimizer semantics remain in their existing owners.
"""

from __future__ import annotations

import collections
import functools
import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from examples.deepswe.r2egym_runtime_patch import (
    _kubernetes_label,
    _optional_queue_name,
    resolve_node_selector_value,
)


DIRECT = "direct"
FLEET = "fleet"
RUNTIME_MODES = frozenset((DIRECT, FLEET))
AGENT_SANDBOX_COMMIT = "7935857fee859bb18752ee04d8948b975e47ff20"


@dataclass(frozen=True)
class FleetPermission:
  """One Kubernetes authorization required by the client-side Fleet path."""

  api_group: str
  resource: str
  verb: str
  namespaced: bool = True
  resource_name: str | None = None


def required_fleet_permissions(
    *, image_pull_secret: str | None = None
) -> tuple[FleetPermission, ...]:
  """Returns the exact client-side RBAC surface exercised by DeepSWE Fleet."""
  permissions = [
      # Resources.ensure_template / create_warmpool / current-batch retirement
      # and the run-scoped cleanup sweep.
      FleetPermission("extensions.agents.x-k8s.io", "sandboxtemplates", verb)
      for verb in ("create", "delete", "get", "list")
  ]
  permissions.extend(
      FleetPermission("extensions.agents.x-k8s.io", "sandboxwarmpools", verb)
      for verb in ("create", "delete", "get", "list", "patch", "watch")
  )
  permissions.extend(
      FleetPermission("extensions.agents.x-k8s.io", "sandboxclaims", verb)
      for verb in ("create", "delete", "get", "list", "watch")
  )
  permissions.extend(
      FleetPermission("agents.x-k8s.io", "sandboxes", verb)
      for verb in ("get", "list", "watch")
  )
  permissions.extend((
      FleetPermission("", "pods", "get"),
      FleetPermission("", "pods", "list"),
      # The pinned router-free R2E adapter calls
      # connect_get_namespaced_pod_exec, so the required RBAC verb is get.
      FleetPermission("", "pods/exec", "get"),
  ))
  for resource_name in (
      "sandboxclaims.extensions.agents.x-k8s.io",
      "sandboxtemplates.extensions.agents.x-k8s.io",
      "sandboxwarmpools.extensions.agents.x-k8s.io",
      "sandboxes.agents.x-k8s.io",
  ):
    permissions.append(FleetPermission(
        "apiextensions.k8s.io",
        "customresourcedefinitions",
        "get",
        namespaced=False,
        resource_name=resource_name,
    ))
  if image_pull_secret:
    permissions.append(FleetPermission(
        "", "secrets", "get", resource_name=image_pull_secret
    ))
  return tuple(permissions)


def verify_fleet_permissions(
    *,
    namespace: str,
    image_pull_secret: str | None = None,
    authorization_api: Any | None = None,
) -> int:
  """Fails before provisioning unless the current identity can drive Fleet.

  Agent Sandbox upstream intentionally treats some manifest dry-run RBAC
  failures as warnings. DeepSWE cannot: a late 403 wastes the TPU allocation
  and can leave a partially warmed run. SelfSubjectAccessReview is read-only
  and evaluates the exact identity used by the head Pod.
  """
  if not namespace:
    raise ValueError("Fleet authorization check requires a namespace")
  from kubernetes import client, config

  if authorization_api is None:
    try:
      config.load_incluster_config()
    except config.ConfigException:
      config.load_kube_config()
    authorization_api = client.AuthorizationV1Api()

  denied = []
  permissions = required_fleet_permissions(
      image_pull_secret=image_pull_secret
  )
  for permission in permissions:
    attributes = client.V1ResourceAttributes(
        group=permission.api_group,
        resource=permission.resource,
        verb=permission.verb,
        namespace=namespace if permission.namespaced else None,
        name=permission.resource_name,
    )
    body = client.V1SelfSubjectAccessReview(
        spec=client.V1SelfSubjectAccessReviewSpec(
            resource_attributes=attributes
        )
    )
    result = authorization_api.create_self_subject_access_review(
        body=body, _request_timeout=60
    )
    status = result.status
    if getattr(status, "allowed", False) is not True:
      scope = namespace if permission.namespaced else "cluster"
      target = permission.resource
      if permission.resource_name:
        target += f"/{permission.resource_name}"
      denied.append(
          f"{permission.verb} {permission.api_group or 'core'}/{target} "
          f"scope={scope} reason={getattr(status, 'reason', '') or '-'} "
          f"evaluation_error={getattr(status, 'evaluation_error', '') or '-'}"
      )
  if denied:
    raise PermissionError(
        "DeepSWE Fleet Kubernetes authorization denied: " + "; ".join(denied)
    )
  print(
      "[DEEPSWE.SANDBOX] RBAC_PASS "
      f"namespace={namespace} checks={len(permissions)}",
      flush=True,
  )
  return len(permissions)


def runtime_mode(environ: Mapping[str, str] | None = None) -> str:
  """Returns the single DeepSWE sandbox runtime selector."""
  env = os.environ if environ is None else environ
  mode = env.get("CANON_DEEPSWE_SANDBOX_RUNTIME", "") or DIRECT
  if mode not in RUNTIME_MODES:
    raise ValueError(
        "CANON_DEEPSWE_SANDBOX_RUNTIME must be exactly direct or fleet; "
        f"got {mode!r}"
    )
  return mode


def _positive_int(
    env: Mapping[str, str], name: str, default: int | None = None
) -> int:
  raw = env.get(name, "")
  if not raw and default is not None:
    return default
  try:
    value = int(raw)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be a positive integer, got {raw!r}") from error
  if value <= 0:
    raise ValueError(f"{name} must be a positive integer, got {value}")
  return value


@dataclass(frozen=True)
class FleetAdmission:
  """Validated, non-semantic infrastructure contract for one training run."""

  batch_size: int
  num_generations: int
  max_concurrency: int
  total_capacity: int
  active_trajectories: int
  replacement_warm: int
  minimum_total: int
  lookahead_batches: int

  @classmethod
  def from_values(
      cls,
      *,
      batch_size: int,
      num_generations: int,
      max_concurrency: int,
      environ: Mapping[str, str] | None = None,
  ) -> "FleetAdmission":
    env = os.environ if environ is None else environ
    for name, value in (
        ("batch_size", batch_size),
        ("num_generations", num_generations),
        ("max_concurrency", max_concurrency),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Fleet {name} must be a positive integer, got {value!r}")
    active = batch_size * num_generations
    if max_concurrency != active:
      raise ValueError(
          "Fleet requires one complete prompt batch in flight: "
          f"batch_size*num_generations={active} != max_concurrency={max_concurrency}"
      )
    total_capacity = _positive_int(env, "R2E_SANDBOX_CAPACITY")
    # A claimed warm sandbox leaves the pool; the controller replenishes it.
    # Consequently a C-wide current batch occupies C active + C replacement.
    replacement = active
    minimum = active + replacement
    if total_capacity < minimum:
      raise ValueError(
          "Fleet sandbox capacity is below the no-lookahead minimum: "
          f"capacity={total_capacity} active={active} replacement_warm={replacement} "
          f"minimum_total={minimum}"
      )
    return cls(
        batch_size=batch_size,
        num_generations=num_generations,
        max_concurrency=max_concurrency,
        total_capacity=total_capacity,
        active_trajectories=active,
        replacement_warm=replacement,
        minimum_total=minimum,
        lookahead_batches=0,
    )


def _flat_values(value: Any) -> list[Any]:
  if isinstance(value, np.ndarray):
    return value.reshape(-1).tolist()
  if isinstance(value, (list, tuple)):
    return list(value)
  return [value]


def _decode(value: Any) -> str:
  if isinstance(value, bytes):
    return value.decode("utf-8")
  return str(value)


def batch_images(batch: Any) -> list[str]:
  """Extracts ordered prompt-level docker images from a Grain batch."""
  if not isinstance(batch, Mapping) or "docker_image" not in batch:
    raise ValueError("Fleet batch must be a mapping containing docker_image")
  images = [_decode(value) for value in _flat_values(batch["docker_image"])]
  if not images or any(not image for image in images):
    raise ValueError("Fleet batch contains an empty docker_image")
  return images


def _dataset_tasks(dataset: Iterable[Mapping[str, Any]], task_type: Any) -> list[Any]:
  tasks = []
  for ordinal, raw in enumerate(dataset):
    row = dict(raw)
    image = _decode(row.get("docker_image", ""))
    if not image:
      raise ValueError(f"Fleet dataset row {ordinal} has no docker_image")
    task_id = _decode(
        row.get("instance_id") or row.get("id") or f"row-{ordinal}"
    )
    tasks.append(task_type(id=task_id, image=image, metadata={"ds": row}))
  if not tasks:
    raise ValueError("Fleet dataset is empty")
  return tasks


def patch_r2egym_fleet_backend() -> None:
  """Adds a complete start/stop bridge for the pinned R2E-Gym backend enum.

  The pinned R2E-Gym accepts only ``docker|kubernetes`` while the upstream
  adapter constructs ``kubernetes-sandbox``.  Initialization temporarily uses
  ``kubernetes`` so R2E creates its Kubernetes client; start and stop then
  dispatch to the adapter subclass.  Patching only init/start is insufficient:
  close would otherwise call R2E's direct Pod deletion path.
  """
  from r2egym.agenthub.runtime import docker as docker_mod

  runtime_cls = docker_mod.DockerRuntime
  if getattr(runtime_cls, "_tunix_fleet_backend_patch", False):
    return
  required = ("__init__", "start_container", "stop_container")
  missing = [name for name in required if not hasattr(runtime_cls, name)]
  if missing:
    raise RuntimeError(
        "R2E-Gym Fleet backend API drifted; missing " + ",".join(missing)
    )
  original_init = runtime_cls.__init__
  original_start = runtime_cls.start_container
  original_stop = runtime_cls.stop_container

  @functools.wraps(original_init)
  def patched_init(self, *args, **kwargs):
    fleet_backend = kwargs.get("backend") == "kubernetes-sandbox"
    if fleet_backend:
      kwargs["backend"] = "kubernetes"
      self._tunix_fleet_backend = True
    original_init(self, *args, **kwargs)
    if fleet_backend:
      self.backend = "kubernetes-sandbox"

  @functools.wraps(original_start)
  def patched_start(self, docker_image, command, ctr_name, **kwargs):
    if getattr(self, "_tunix_fleet_backend", False):
      return self._start_kubernetes_sandbox()
    return original_start(self, docker_image, command, ctr_name, **kwargs)

  @functools.wraps(original_stop)
  def patched_stop(self, *args, **kwargs):
    if getattr(self, "_tunix_fleet_backend", False):
      return self._stop_kubernetes_sandbox()
    return original_stop(self, *args, **kwargs)

  runtime_cls.__init__ = patched_init
  runtime_cls.start_container = patched_start
  runtime_cls.stop_container = patched_stop
  runtime_cls._tunix_fleet_backend_patch = True


class DeepSWESandboxFleet:
  """Run-owned, current-batch Agent Sandbox manager for R2E-Gym."""

  def __init__(
      self,
      dataset: Iterable[Mapping[str, Any]],
      *,
      batch_size: int,
      num_generations: int,
      max_concurrency: int,
      scaffold: str = "r2egym",
      environ: Mapping[str, str] | None = None,
  ):
    self._env = dict(os.environ if environ is None else environ)
    if runtime_mode(self._env) != FLEET:
      raise ValueError("DeepSWESandboxFleet requires sandbox runtime fleet")
    if scaffold != "r2egym":
      raise ValueError("DeepSWE Fleet currently admits only scaffold=r2egym")
    actual_commit = self._env.get("CANON_AGENT_SANDBOX_COMMIT", "")
    if actual_commit != AGENT_SANDBOX_COMMIT:
      raise ValueError(
          "Fleet requires the exact Agent Sandbox source commit: "
          f"expected={AGENT_SANDBOX_COMMIT} actual={actual_commit or 'absent'}"
      )
    run_id = self._env.get("CANON_RUN_ID", "")
    if not run_id:
      raise ValueError("Fleet requires nonempty CANON_RUN_ID for resource ownership")
    self.admission = FleetAdmission.from_values(
        batch_size=batch_size,
        num_generations=num_generations,
        max_concurrency=max_concurrency,
        environ=self._env,
    )
    self._active_deadline = _positive_int(
        self._env, "R2E_ACTIVE_DEADLINE_SECONDS", 5100
    )
    self._delete_timeout = _positive_int(
        self._env, "R2E_POD_DELETE_TIMEOUT_SECONDS", 300
    )
    self._ready_timeout = _positive_int(
        self._env, "R2E_POD_START_TIMEOUT_SECONDS", 1200
    )
    self._lock = threading.Lock()
    self._local = threading.local()
    self._prepared_counts: collections.Counter[str] = collections.Counter()
    self._claimed_counts: collections.Counter[str] = collections.Counter()
    self._prepared_images: set[str] = set()
    self._closed = False

    try:
      from agent_sandbox_rl import (
          ClusterConfig,
          FleetConfig,
          ResourceSpec,
          SandboxFleet,
          Task,
          TemplateSpec,
      )
    except ImportError as error:
      raise ImportError(
          "Fleet mode requires the pinned Agent Sandbox packages; run Canon "
          "step 36_install_agent_sandbox.sh"
      ) from error

    patch_r2egym_fleet_backend()
    node_key = self._env.get(
        "NODE_SELECTOR_KEY", "cloud.google.com/gke-nodepool"
    )
    node_value = resolve_node_selector_value(self._env)
    if not node_key or not node_value:
      raise ValueError("Fleet requires a nonempty sandbox node selector")
    namespace = self._env.get("R2E_K8S_NAMESPACE", "default")
    if not namespace:
      raise ValueError("R2E_K8S_NAMESPACE must not be empty")
    queue_name = _optional_queue_name(
        self._env.get("R2E_K8S_QUEUE_NAME", "")
    )
    lineage = _kubernetes_label(run_id, fallback="unknown")
    self._lineage = lineage
    labels = {
        "app.kubernetes.io/name": "r2egym",
        "app.kubernetes.io/managed-by": "tunix-deepswe",
        "canon.zero-tim/run-id": lineage,
        "canon.zero-tim/resume-tag": _kubernetes_label(
            self._env.get("CANON_P46_RESUME_TAG", run_id),
            fallback="unknown",
        ),
    }
    if queue_name:
      labels["kueue.x-k8s.io/queue-name"] = queue_name
    pull_secret = self._env.get("IMAGE_PULL_SECRET", "dockerhub-pro")
    verify_fleet_permissions(
        namespace=namespace, image_pull_secret=pull_secret or None
    )
    pod_spec = {
        "activeDeadlineSeconds": self._active_deadline,
        "restartPolicy": "Never",
        "tolerations": [{
            "key": "node.kubernetes.io/disk-pressure",
            "operator": "Exists",
            "effect": "NoExecute",
            "tolerationSeconds": 10800,
        }],
    }
    template = TemplateSpec(
        resources=ResourceSpec(
            cpu=self._env.get("R2E_K8S_CPU", "2"),
            memory=self._env.get("R2E_K8S_MEM", "4Gi"),
        ),
        node_selector={node_key: node_value},
        image_pull_secret=pull_secret or None,
        image_pull_policy="IfNotPresent",
        colocate_replicas=True,
        extra_pod_spec=pod_spec,
    )
    prefix = f"r2e-{hashlib.sha256(run_id.encode()).hexdigest()[:10]}-"
    in_cluster = bool(self._env.get("KUBERNETES_SERVICE_HOST"))
    config = FleetConfig(
        clusters=[ClusterConfig(
            name="deepswe",
            namespace=namespace,
            node_selector={node_key: node_value},
            image_pull_secret=pull_secret or None,
            in_cluster=in_cluster,
            # This is only the upstream warm-replica hint. Total-Pod admission
            # is enforced independently by FleetAdmission above.
            max_replicas=self.admission.replacement_warm,
        )],
        max_concurrent=self.admission.max_concurrency,
        # A batch can repeat one image across prompts.  The explicit warm call
        # then needs prompt_count(image)*G replicas, up to the full active C.
        # Keep the upstream per-image hard cap honest even though its current
        # replicas_override path does not clamp the requested value.
        max_warmpool_size=self.admission.active_trajectories,
        warm_per_task=False,
        window_size=self.admission.batch_size,
        template=template,
        template_name_prefix=prefix,
        labels=labels,
    )
    self._fleet = SandboxFleet(config)
    self._task_type = Task
    self._labels = labels
    self._limits = {
        "cpu": self._env.get("R2E_K8S_CPU_LIMIT", "4"),
        "memory": self._env.get("R2E_K8S_MEM_LIMIT", "8Gi"),
    }
    self._install_manifest_contract()
    self._install_claim_deadline()
    tasks = _dataset_tasks(dataset, Task)
    self._fleet.load_tasks(tasks)
    self._fleet.preflight()
    plan = self._fleet.plan()
    if not plan.entries:
      raise RuntimeError("Fleet produced an empty image plan")
    print(
        "[DEEPSWE.SANDBOX] ADMISSION_PASS "
        f"mode=fleet source={AGENT_SANDBOX_COMMIT} "
        f"batch_size={self.admission.batch_size} "
        f"num_generations={self.admission.num_generations} "
        f"active={self.admission.active_trajectories} "
        f"replacement_warm={self.admission.replacement_warm} "
        f"minimum_total={self.admission.minimum_total} "
        f"admitted_capacity={self.admission.total_capacity} "
        "lookahead_batches=0 scaffold=r2egym",
        flush=True,
    )

  @property
  def fleet(self) -> Any:
    return self._fleet

  def _install_manifest_contract(self) -> None:
    """Preserves the direct R2E Pod resource, queue, and lineage contract."""
    for cluster in self._fleet.registry:
      resources = cluster.resources
      original = resources._template_manifest

      @functools.wraps(original)
      def manifest(image, template_name, template, _original=original):
        body = _original(image, template_name, template)
        pod_template = body["spec"]["podTemplate"]
        pod_template.setdefault("metadata", {}).setdefault("labels", {}).update(
            self._labels
        )
        containers = pod_template["spec"].get("containers", [])
        if len(containers) != 1:
          raise RuntimeError(
              "Agent Sandbox template must contain exactly one runtime container"
          )
        containers[0].setdefault("resources", {})["limits"] = dict(self._limits)
        return body

      resources._template_manifest = manifest

  def _install_claim_deadline(self) -> None:
    """Forwards an atomic claim TTL omitted by SandboxFleet.acquire()."""
    for cluster in self._fleet.registry:
      client = cluster.sandbox_client
      original = client.create_sandbox

      @functools.wraps(original)
      def create(*args, _original=original, **kwargs):
        remaining = getattr(self._local, "ready_timeout", self._ready_timeout)
        kwargs["sandbox_ready_timeout"] = max(
            1, min(int(kwargs.get("sandbox_ready_timeout", remaining)), remaining)
        )
        kwargs["shutdown_after_seconds"] = max(
            1, getattr(self._local, "shutdown_after_seconds", self._active_deadline)
        )
        return _original(*args, **kwargs)

      client.create_sandbox = create

  def prepare_batch(self, batch: Any) -> None:
    """Retires the prior pool and synchronously warms exactly one prompt batch."""
    images = batch_images(batch)
    if len(images) != self.admission.batch_size:
      raise ValueError(
          "Fleet prompt batch size drifted: "
          f"expected={self.admission.batch_size} actual={len(images)}"
      )
    with self._lock:
      if self._closed:
        raise RuntimeError("Fleet manager is closed")
      handles = list(self._fleet.handles())
      if handles:
        raise RuntimeError(
            "Cannot advance Fleet batch while sandbox handles remain active: "
            f"active_handles={len(handles)}"
        )
      previous = sorted(self._prepared_images)
    for image in previous:
      self._fleet.unwarm_image(image)

    counts = collections.Counter(images)
    by_replicas: dict[int, list[str]] = collections.defaultdict(list)
    for image, prompt_count in counts.items():
      by_replicas[prompt_count * self.admission.num_generations].append(image)
    total_warm = sum(
        replicas * len(group) for replicas, group in by_replicas.items()
    )
    if total_warm != self.admission.active_trajectories:
      raise RuntimeError(
          "Fleet warm replica arithmetic drifted: "
          f"expected={self.admission.active_trajectories} actual={total_warm}"
      )
    started = time.perf_counter()
    for replicas, group in sorted(by_replicas.items()):
      self._fleet.warm_images(
          sorted(group), replicas_override=replicas, wait=True
      )
    elapsed = time.perf_counter() - started
    with self._lock:
      self._prepared_counts = counts
      self._claimed_counts.clear()
      self._prepared_images = set(counts)
    print(
        "[DEEPSWE.SANDBOX] BATCH_READY "
        f"prompts={len(images)} images={len(counts)} "
        f"warm_replicas={total_warm} elapsed_secs={elapsed:.3f}",
        flush=True,
    )

  def acquire(
      self, entry: Mapping[str, Any], *, batch_started_monotonic: float | None
  ) -> tuple[Any, float]:
    """Acquires one prepared sandbox with a deadline-aware retry budget."""
    image = _decode(entry.get("docker_image", ""))
    if not image:
      raise ValueError("Fleet trajectory entry has no docker_image")
    with self._lock:
      expected = self._prepared_counts.get(image, 0) * self.admission.num_generations
      claimed = self._claimed_counts.get(image, 0)
      if expected <= 0:
        raise RuntimeError(f"Fleet image was not prepared for this batch: {image}")
      if claimed >= expected:
        raise RuntimeError(
            f"Fleet image claim count exceeded prepared replicas: {claimed}/{expected}"
        )
      self._claimed_counts[image] += 1

    elapsed_from_batch = 0.0
    if batch_started_monotonic is not None:
      elapsed_from_batch = max(0.0, time.perf_counter() - batch_started_monotonic)
    ready_budget = min(
        self._ready_timeout,
        max(0, int(math.floor(self._active_deadline - elapsed_from_batch))),
    )
    if ready_budget <= 0:
      with self._lock:
        self._claimed_counts[image] -= 1
      raise TimeoutError("Fleet claim began after the shared sandbox deadline")
    self._local.ready_timeout = ready_budget
    self._local.shutdown_after_seconds = max(
        1, int(math.ceil(self._active_deadline - elapsed_from_batch))
    )
    task = self._task_type(
        id=_decode(entry.get("instance_id") or entry.get("id") or image),
        image=image,
        metadata={"ds": dict(entry)},
    )
    started = time.perf_counter()
    try:
      handle = self._fleet.acquire(task)
    except BaseException:
      with self._lock:
        self._claimed_counts[image] -= 1
      raise
    finally:
      self._local.ready_timeout = self._ready_timeout
      self._local.shutdown_after_seconds = self._active_deadline
    return handle, time.perf_counter() - started

  def make_repo_env(
      self,
      handle: Any,
      *,
      command_files: list[str],
      step_timeout: int,
      reward_timeout: int,
      verbose: bool,
  ) -> Any:
    from agent_sandbox_rl.adapters.r2egym import make_fleet_repo_env

    return make_fleet_repo_env(
        handle,
        command_files=command_files,
        step_timeout=step_timeout,
        reward_timeout=reward_timeout,
        verbose=verbose,
    )

  def _resource_absent(
      self, cluster: Any, *, group: str, version: str, plural: str, name: str
  ) -> bool:
    from kubernetes import client

    try:
      cluster.custom_api.get_namespaced_custom_object(
          group=group,
          version=version,
          namespace=cluster.namespace,
          plural=plural,
          name=name,
      )
    except client.ApiException as error:
      if error.status == 404:
        return True
      raise
    return False

  def _pod_absent(self, cluster: Any, name: str) -> bool:
    from kubernetes import client

    try:
      cluster.core_api.read_namespaced_pod(
          name=name, namespace=cluster.namespace, _request_timeout=60
      )
    except client.ApiException as error:
      if error.status == 404:
        return True
      raise
    return False

  def _confirm_released(self, handle: Any) -> None:
    from agent_sandbox_rl import constants

    cluster = self._fleet.registry.get(handle.cluster_name)
    deadline = time.monotonic() + self._delete_timeout
    while time.monotonic() < deadline:
      claim_gone = self._resource_absent(
          cluster,
          group=constants.GROUP,
          version=constants.VERSION,
          plural=constants.CLAIMS_PLURAL,
          name=handle.claim_name,
      )
      sandbox_gone = self._resource_absent(
          cluster,
          group=constants.SANDBOX_GROUP,
          version=constants.SANDBOX_VERSION,
          plural=constants.SANDBOXES_PLURAL,
          name=handle.sandbox_id,
      )
      pod_gone = self._pod_absent(cluster, handle.pod_name)
      if claim_gone and sandbox_gone and pod_gone:
        return
      time.sleep(2)
    raise TimeoutError(
        "Agent Sandbox cleanup did not remove claim/sandbox/pod within "
        f"{self._delete_timeout}s: claim={handle.claim_name} "
        f"sandbox={handle.sandbox_id} pod={handle.pod_name}"
    )

  def release(self, handle: Any) -> float:
    """Releases one handle and confirms all remote resources are absent."""
    started = time.perf_counter()
    release_error = None
    try:
      self._fleet.release(handle)
    except BaseException as error:
      release_error = error
      # SandboxFleet removes bookkeeping before remote deletion. Retry the
      # exactly-scoped idempotent claim deletion so a transient API failure does
      # not turn into an untracked leak. The original failure remains fatal.
      cluster = self._fleet.registry.get(handle.cluster_name)
      cluster.resources.delete_claim(handle.claim_name)
    self._confirm_released(handle)
    if release_error is not None:
      raise release_error
    return time.perf_counter() - started

  def _delete_run_owned_resources(self) -> None:
    """Deletes only this run's claims, warm pools, and templates.

    Upstream ``SandboxFleet.teardown()`` selects every resource carrying the
    package-wide managed-by label.  That is unsafe in a shared DeepSWE
    namespace because it can delete another run's pools.  The Canon wrapper
    adds a run lineage label to every object and scopes every sweep to it.
    """
    selector = f"canon.zero-tim/run-id={self._lineage}"
    for cluster in self._fleet.registry:
      resources = cluster.resources
      for claim in resources.list_claims(label_selector=selector):
        resources.delete_claim(claim)
      for pool in resources.list_warmpools(label_selector=selector):
        resources.delete_warmpool(pool)
      for template in resources.list_templates(label_selector=selector):
        resources.delete_template(template)

      deadline = time.monotonic() + self._delete_timeout
      while time.monotonic() < deadline:
        remaining = (
            resources.list_claims(label_selector=selector)
            + resources.list_warmpools(label_selector=selector)
            + resources.list_templates(label_selector=selector)
        )
        pods = cluster.core_api.list_namespaced_pod(
            namespace=cluster.namespace,
            label_selector=selector,
            _request_timeout=60,
        ).items
        if not remaining and not pods:
          break
        time.sleep(2)
      else:
        raise TimeoutError(
            "Agent Sandbox run-scoped teardown did not finish within "
            f"{self._delete_timeout}s: selector={selector}"
        )

  def close(self) -> None:
    """Hard normal-exit cleanup for handles and run-owned warm resources."""
    with self._lock:
      if self._closed:
        return
      self._closed = True
    for handle in list(self._fleet.handles()):
      self.release(handle)
    self._delete_run_owned_resources()
    if self._fleet.handles():
      raise RuntimeError("Fleet close returned with active handles")
    self._prepared_images.clear()


class CurrentBatchPrewarmIterator:
  """Warms exactly the batch being delivered; never prefetches another batch."""

  def __init__(self, dataset: Iterable[Any], manager: DeepSWESandboxFleet):
    self._dataset = iter(dataset)
    self._manager = manager

  def __iter__(self) -> "CurrentBatchPrewarmIterator":
    return self

  def __next__(self) -> Any:
    batch = next(self._dataset)
    self._manager.prepare_batch(batch)
    return batch
