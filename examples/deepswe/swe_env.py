import atexit
import collections
import json
import logging
import os
import threading
from typing import Any, Optional, cast
import numpy as np

_GLOBAL_FLEET = None
_FLEET_LOCK = threading.Lock()
_PATCH_LOCK = threading.Lock()
_R2EGYM_PATCHED = False


def _patch_r2egym_for_agent_sandbox() -> None:
  """In-memory compatibility patch for r2egym to route 'kubernetes-sandbox' backend."""
  global _R2EGYM_PATCHED
  with _PATCH_LOCK:
    if _R2EGYM_PATCHED:
      return
    _R2EGYM_PATCHED = True
  try:
    import huggingface_hub  # pytype: disable=import-error

    if not hasattr(huggingface_hub, "HfFolder"):
      try:
        from huggingface_hub._login import HfFolder  # pytype: disable=import-error

        huggingface_hub.HfFolder = HfFolder
      except Exception:

        class DummyHfFolder:

          @staticmethod
          def get_token():
            return None

        huggingface_hub.HfFolder = DummyHfFolder  # pytype: disable=bad-assignment
  except Exception as e:
    logging.debug("[SandboxFleet] HfFolder patch note: %s", e)

  try:
    from r2egym.agenthub.runtime import docker as docker_mod  # pytype: disable=import-error

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
  except Exception as e:
    logging.debug("[SandboxFleet] r2egym in-memory patch note: %s", e)


def _normalize_tasks_for_fleet(tasks: Any) -> list[Any]:
  """Normalize heterogeneous dataset entries into Task objects for SandboxFleet."""
  try:
    from agent_sandbox_rl import Task  # pytype: disable=import-error
  except ImportError:
    return list(tasks)

  normalized = []
  for item in tasks:
    if isinstance(item, Task):
      normalized.append(item)
    elif isinstance(item, dict):
      img = item.get("docker_image") or item.get("image")
      if not img and "metadata" in item and isinstance(item["metadata"], dict):
        img = item["metadata"].get("docker_image") or item["metadata"].get(
            "env_config", {}
        ).get("entry", {}).get("docker_image")
      if (
          not img
          and "env_config" in item
          and isinstance(item["env_config"], dict)
      ):
        img = item["env_config"].get("entry", {}).get("docker_image")
      img = img or "default"
      if isinstance(img, (list, np.ndarray)):
        img = img[0] if len(img) > 0 else "default"
      t_id = (
          item.get("instance_id")
          or item.get("id")
          or item.get("prompt_id")
          or img
      )
      if isinstance(t_id, (list, np.ndarray)):
        t_id = t_id[0] if len(t_id) > 0 else "default"
      normalized.append(
          Task(id=str(t_id), image=str(img), metadata={"ds": item})
      )
    else:
      normalized.append(item)
  return normalized


def _init_global_fleet(
    tasks: list[Any],
    max_concurrency: int = 128,
    num_generations: int = 8,
    batch_size: int = 8,
    max_warmpool_replicas: int | None = None,
) -> Any:
  """Initialize the process-wide SandboxFleet instance once upfront."""
  global _GLOBAL_FLEET
  with _FLEET_LOCK:
    if _GLOBAL_FLEET is not None:
      return _GLOBAL_FLEET

    _patch_r2egym_for_agent_sandbox()

    try:
      from agent_sandbox_rl import (  # pytype: disable=import-error
          ClusterConfig,
          FleetConfig,
          SandboxFleet,
          Task,
          TemplateSpec,
      )
    except ImportError as e:
      raise ImportError(
          "use_agent_sandbox=True strictly requires the 'agent_sandbox_rl'"
          " package. Install via: pip install"
          " git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl"
      ) from e

    fleet_ns = os.getenv("NAMESPACE", "rl-tunix-swebench")
    key = os.environ.get("NODE_SELECTOR_KEY")
    val = os.environ.get("NODE_SELECTOR_VAL")
    node_sel = {key: val} if (key and val) else None

    in_cluster = os.getenv("KUBERNETES_SERVICE_HOST") is not None
    effective_max_concurrent = max(
        max_concurrency, batch_size * num_generations * 2
    )
    fleet_cfg = FleetConfig(
        clusters=[
            ClusterConfig(
                name="default",
                namespace=fleet_ns,
                node_selector=node_sel,
                in_cluster=in_cluster,
            )
        ],
        max_concurrent=effective_max_concurrent,
        window_size=batch_size,
        max_warmpool_size=max_warmpool_replicas
        if max_warmpool_replicas is not None
        else num_generations,
        warm_per_task=True,
    )
    fleet_inst = SandboxFleet(fleet_cfg)
    if tasks is not None:
      fleet_inst.load_tasks(_normalize_tasks_for_fleet(tasks))
    msg = (
        f"[SandboxFleet] Initializing pipelined fleet in namespace={fleet_ns}"
        f" (max_concurrent={effective_max_concurrent},"
        f" window_size={batch_size},"
        f" max_warmpool_replicas={max_warmpool_replicas if max_warmpool_replicas is not None else num_generations},"
        " warm_per_task=True)..."
    )
    logging.info(msg)
    if fleet_cfg.install_teardown_hooks:
      fleet_inst._install_teardown_hooks()
    fleet_inst._torndown = False
    fleet_inst.preflight()
    fleet_inst.plan()
    _GLOBAL_FLEET = fleet_inst
    atexit.register(_teardown_global_fleet)
    return _GLOBAL_FLEET


def _get_global_fleet() -> Any:
  """Retrieve the active process-wide SandboxFleet instance."""
  if _GLOBAL_FLEET is None:
    raise RuntimeError(
        "SandboxFleet has not been initialized. Call _init_global_fleet()"
        " first."
    )
  return _GLOBAL_FLEET


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
  ):
    del lookahead_steps
    self.dataset_iter = iter(dataset)
    self.fleet = fleet or _get_global_fleet()
    self.num_generations = num_generations
    self.batch_size = max(1, batch_size)
    self.max_warmpool_replicas = max_warmpool_replicas
    self.unwarm_on_exhaustion = unwarm_on_exhaustion

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
      if "docker_image" in item:
        raw = item["docker_image"]
        if isinstance(raw, (list, tuple, np.ndarray)):
          raw_images.extend(np.array(raw).flatten().tolist())
        elif isinstance(raw, (str, bytes)):
          raw_images.append(raw)
      elif "metadata" in item and isinstance(item["metadata"], dict):
        meta = item["metadata"]
        if "docker_image" in meta:
          raw_images.append(meta["docker_image"])
        elif "env_config" in meta and isinstance(meta["env_config"], dict):
          entry = meta["env_config"].get("entry", {})
          if isinstance(entry, dict) and "docker_image" in entry:
            raw_images.append(entry["docker_image"])
      elif "env_config" in item and isinstance(item["env_config"], dict):
        entry = item["env_config"].get("entry", {})
        if isinstance(entry, dict) and "docker_image" in entry:
          raw_images.append(entry["docker_image"])

    for img in raw_images:
      if img is not None:
        str_img = img.decode("utf-8") if hasattr(img, "decode") else str(img)
        counts[str_img] = counts.get(str_img, 0) + 1
    return counts

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
      sample_count = max(1, sum(item_counts.values()))
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
          self.fleet.warm_image(
              img, replicas_override=target_reps, wait=wait
          )
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


def _teardown_global_fleet() -> None:
  """Atexit handler to cleanly tear down warm pools on process exit."""
  global _GLOBAL_FLEET
  if _GLOBAL_FLEET is not None:
    logging.info(
        "[SandboxFleet] Automatically tearing down warm pools on exit..."
    )
    try:
      _GLOBAL_FLEET.teardown()
    except Exception as e:
      logging.warning("[SandboxFleet] Teardown note: %s", e)
    _GLOBAL_FLEET = None


try:
  import r2egym  # pytype: disable=import-error
  from r2egym.agenthub.action import Action  # pytype: disable=import-error  # pytype: disable=import-error
  from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error  # pytype: disable=import-error
except ImportError:
  r2egym = cast(Any, None)
  EnvArgs = cast(Any, None)
  RepoEnv = cast(Any, None)
  Action = cast(Any, None)

from tunix.rl.agentic.environments.base_environment import BaseTaskEnv, EnvStepResult

if r2egym:
  R2EGYM_PATH = os.path.dirname(r2egym.__file__)
else:
  R2EGYM_PATH = ""
# List of tools to be used in the environment.
R2EGYM_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/file_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/search.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/r2egym/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/finish.py"),
]

SWEAGENT_COMMAND_FILES = [
    os.path.join(R2EGYM_PATH, "agenthub/tools/str_replace_editor.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/execute_bash.py"),
    os.path.join(R2EGYM_PATH, "agenthub/tools/submit.py"),
]


def _unpack_entry(entry: dict) -> dict:
  """Utility to clean up and unpack the dataset entry."""
  unpacked_entry = {}
  for k, v in entry.items():
    if isinstance(v, np.ndarray):
      unpacked_entry[k] = v.item()
    elif isinstance(v, list):
      if len(v) != 1:
        raise ValueError(
            f"Can only convert a list of size 1; got size {len(v)}"
        )
      unpacked_entry[k] = v[0]
    else:
      unpacked_entry[k] = v
  return unpacked_entry


class SWEEnv(BaseTaskEnv):
  """Software Engineering Environment for code-related tasks."""

  def __init__(
      self,
      entry: dict,
      group_id: int | None = None,
      pair_index: int | None = None,
      step_timeout: int = 30 * 60,
      reward_timeout: int = 30 * 60,
      backend: str = "kubernetes",
      delete_image: bool = False,
      verbose: bool = False,
      scaffold: str = "r2egym",
      max_steps: int = 1,
      use_agent_sandbox: bool = False,
      fleet: Any | None = None,
  ):
    """Initialize the SWE environment.

    Args:
        entry: Dataset containing the tasks. If None, uses default dataset.
        group_id: ID of the group to which the task belongs.
        pair_index: Index of the pair to use. If None, selects a random pair.
        step_timeout: Timeout for each step in seconds.
        reward_timeout: Timeout for reward computation in seconds.
        backend: Backend to use for the environment.
        delete_image: Whether to delete the Docker image after closing.
        verbose: Verbose output toggle.
        scaffold: Scaffold tool set ('r2egym' or 'sweagent').
        max_steps: Maximum interaction steps.
        use_agent_sandbox: If True, strictly forces SandboxFleet and
          AgentSandboxRuntime.
        fleet: Optional SandboxFleet instance to use.
    """
    self.entry = _unpack_entry(entry)
    self.step_timeout = step_timeout
    self.reward_timeout = reward_timeout
    self.total_steps = 0
    self.delete_image = delete_image
    self.backend = backend
    self.env = None
    self.handle = None
    self.verbose = verbose
    self.scaffold = scaffold
    self.use_agent_sandbox = use_agent_sandbox
    self.fleet = fleet

    assert scaffold in [
        "r2egym",
        "sweagent",
    ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent']"
    super().__init__(max_steps=max_steps)

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}

    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _initial_observation(self) -> Any:
    if not self.env:
      if self.use_agent_sandbox:
        _patch_r2egym_for_agent_sandbox()
        from agent_sandbox_rl import Task  # pytype: disable=import-error
        from agent_sandbox_rl.adapters.r2egym import (  # pytype: disable=import-error
            make_fleet_repo_env,
            r2egym_command_files,
        )

        fleet = self.fleet or _get_global_fleet()
        msg = (
            "[SWEEnv] Acquiring SandboxHandle from SandboxFleet and"
            " constructing FleetRepoEnv!"
        )
        logging.info(msg)
        task = Task(
            id=str(
                self.entry.get(
                    "instance_id", self.entry.get("docker_image", "default")
                )
            ),
            image=self.entry.get("docker_image", "default"),
            metadata={"ds": self.entry},
        )
        self.handle = fleet.acquire(task)
        # TODO(wuhao): Revisit command_files once other harnesses (such as OpenHands) are supported.
        cmd_files = r2egym_command_files()
        self.env = make_fleet_repo_env(self.handle, command_files=cmd_files)
      else:
        # Initialize standard local Docker RepoEnv
        global EnvArgs, RepoEnv, Action
        if EnvArgs is None:
          from r2egym.agenthub.action import Action  # pytype: disable=import-error
          from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # pytype: disable=import-error
        env_args = EnvArgs(ds=self.entry)
        self.env = RepoEnv(
            env_args,
            backend=self.backend,
            step_timeout=self.step_timeout,
            reward_timeout=self.reward_timeout,
            verbose=self.verbose,
        )
        if self.scaffold == "r2egym":
          self.env.add_commands(R2EGYM_COMMAND_FILES)
        else:
          self.env.add_commands(SWEAGENT_COMMAND_FILES)
    else:
      self.env.reset()

    self.final_reward_fn = self.env.compute_reward  # pytype: disable=attribute-error
    self.total_steps = 0

    # Polls docker runtime to get task instruction.
    return self.env.get_task_instruction()  # pytype: disable=attribute-error

  def _step_impl(self, action: Any) -> EnvStepResult:
    global Action
    if Action is None:
      from r2egym.agenthub.action import Action  # pytype: disable=import-error
    if isinstance(action, str):
      action_obj = Action.from_string(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return EnvStepResult(observation="", reward=0, done=False, info={})

    # RepoEnv always returns 0 reward, must be evaluated by DockerRuntime.
    if not self.env:
      raise ValueError("Environment not initialized")
    obs, reward, done, info = self.env.step(action_obj)

    self.total_steps += 1

    return EnvStepResult(
        observation=str(obs), reward=reward, done=done, info=info
    )

  def close(self) -> None:
    """Close the environment and clean up resources."""
    if self.env is not None:
      self.env.close()

    fleet = self.fleet or _GLOBAL_FLEET
    if (
        hasattr(self, "handle")
        and self.handle is not None
        and fleet is not None
    ):
      msg = "[SWEEnv] Releasing SandboxHandle back to SandboxFleet."
      logging.info(msg)
      fleet.release(self.handle)
      self.handle = None

    if (
        self.delete_image
        and not self.use_agent_sandbox
        and self.env
        and hasattr(self.env, "runtime")
    ):
      docker_image = getattr(self.env.runtime, "docker_image", None)
      if docker_image:
        os.system(f"docker rmi {docker_image}")

  @staticmethod
  def from_dict(extra_info: dict | str) -> "SWEEnv":  # pyrefly: ignore[bad-override]
    """Create an environment instance from JSON configuration.

    Args:
        extra_info: Dictionary containing configuration parameters. The entire
          dict will be used as 'entry', and any keys matching __init__
          parameters will be extracted and passed.

    Returns:
        Initialized SWEEnv instance
    """
    import inspect

    if isinstance(extra_info, str):
      extra_info = json.loads(extra_info)

    sig = inspect.signature(SWEEnv.__init__)
    init_params = {}
    for param_name, param in sig.parameters.items():
      if param_name == "self":
        continue
      if param_name in extra_info:
        init_params[param_name] = extra_info[param_name]
      # else if param has default value, use the default value
    init_params["entry"] = extra_info
    return SWEEnv(**init_params)
