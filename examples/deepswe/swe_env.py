import atexit
import json
import logging
import os
import re
import threading
import time
from typing import Any, Optional, cast
import numpy as np

try:
  from examples.deepswe import template as template_mod
except ImportError:
  import template as template_mod  # pytype: disable=import-error

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


def _get_image_rewrite_fn(image_rewrite: Any | None = None) -> Any | None:
  """Retrieve or construct the image rewrite function from prefix if configured."""
  if image_rewrite is not None:
    return image_rewrite
  if os.getenv("IMAGE_REWRITE_PREFIX"):
    prefix = os.environ["IMAGE_REWRITE_PREFIX"].rstrip("/")
    return lambda img: f"{prefix}/{img.split('/')[-1]}"
  return None


def _normalize_tasks_for_fleet(
    tasks: Any, scaffold: str = "r2egym"
) -> list[Any]:
  """Normalize heterogeneous dataset entries into Task objects for SandboxFleet."""
  TaskCls = None
  try:
    from agent_sandbox_rl import Task  # pytype: disable=import-error
    if isinstance(Task, type):
      TaskCls = Task
  except ImportError:
    pass

  if TaskCls is None:
    from dataclasses import dataclass

    @dataclass
    class TaskCls:  # pytype: disable=reimported
      id: str
      image: str
      metadata: dict[str, Any]

  agent_server_override = (
      os.getenv("AGENT_SERVER_IMAGE") if scaffold == "openhands" else None
  )
  normalized = []
  for item in tasks:
    if hasattr(item, "image") and hasattr(item, "id"):
      normalized.append(item)
    elif isinstance(item, dict):
      img = (
          agent_server_override
          or item.get("docker_image")
          or item.get("image", "default")
      )
      if isinstance(img, (list, np.ndarray)):
        img = img[0] if len(img) > 0 else "default"
      t_id = item.get("instance_id") or item.get("id") or img
      if isinstance(t_id, (list, np.ndarray)):
        t_id = t_id[0] if len(t_id) > 0 else "default"
      normalized.append(
          TaskCls(id=str(t_id), image=str(img), metadata={"ds": item})
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
    scaffold: str = "r2egym",
    image_rewrite: Any | None = None,
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

    template = template_mod.get_template(scaffold, node_sel)

    is_shared_image = scaffold == "openhands" and bool(
        os.getenv("AGENT_SERVER_IMAGE")
    )
    default_warmpool_size = (
        effective_max_concurrent if is_shared_image else num_generations
    )
    fleet_kwargs = {
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
            else default_warmpool_size
        ),
        "warm_per_task": True,
    }
    if template is not None:
      fleet_kwargs["template"] = template
    if scaffold == "openhands":
      fleet_kwargs["template_name_prefix"] = "oh-img-"
    fleet_cfg = FleetConfig(**fleet_kwargs)
    fleet_inst = SandboxFleet(fleet_cfg)
    image_rewrite_fn = _get_image_rewrite_fn(image_rewrite)
    fleet_inst._image_rewrite_fn = image_rewrite_fn
    if tasks is not None:
      normalized_tasks = _normalize_tasks_for_fleet(tasks, scaffold=scaffold)
      should_rewrite = (
          image_rewrite_fn is not None
          and (scaffold != "openhands" or not os.getenv("AGENT_SERVER_IMAGE"))
      )
      if should_rewrite:
        fleet_inst.load_tasks(normalized_tasks, image_rewrite=image_rewrite_fn)
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
  """2-slot Lookahead iterator: guarantees the upcoming batch is always pre-warming ahead on K8s."""

  def __init__(
      self,
      dataset: Any,
      fleet: Any | None = None,
      num_generations: int = 8,
      batch_size: int = 8,
      max_warmpool_replicas: int | None = None,
      scaffold: str = "r2egym",
      image_rewrite: Any | None = None,
  ):
    self.dataset_iter = iter(dataset)
    self.num_generations = num_generations
    self.batch_size = batch_size
    self.max_warmpool_replicas = max_warmpool_replicas
    self.scaffold = scaffold
    self.fleet = fleet or _get_global_fleet()
    self.image_rewrite = _get_image_rewrite_fn(
        image_rewrite or getattr(self.fleet, "_image_rewrite_fn", None)
    )
    self.current_batch = None
    self.next_batch = None
    self.prev_batch_images: list[str] = []

    # 1. Prime Slot 1 (Current Batch - wait until pods are ready before training starts)
    try:
      self.current_batch = next(self.dataset_iter)
      logging.info(
          "[PrewarmDatasetIterator] Warming initial batch on K8s and waiting"
          " for pods to be ready..."
      )
      self._warm_batch(self.current_batch, wait=True)
    except StopIteration:
      pass

    # 2. Prime Slot 2 (Next Batch - Pre-warming in background!)
    try:
      self.next_batch = next(self.dataset_iter)
      self._warm_batch(self.next_batch, wait=False)
    except StopIteration:
      pass

  def _extract_images(self, batch: Any) -> list[str]:
    agent_server_override = (
        os.getenv("AGENT_SERVER_IMAGE")
        if self.scaffold == "openhands"
        else None
    )
    if agent_server_override:
      return [agent_server_override]

    raw_images = []
    if isinstance(batch, dict) and "docker_image" in batch:
      raw = batch["docker_image"]
      if isinstance(raw, (list, np.ndarray)):
        raw_images = np.array(raw).flatten().tolist()
      elif isinstance(raw, str):
        raw_images = [raw]
      elif hasattr(raw, "decode"):
        raw_images = [raw]
    elif isinstance(batch, list):
      raw_images = [
          item.get("docker_image")
          for item in batch
          if isinstance(item, dict) and item.get("docker_image")
      ]

    # Safely decode/stringify all elements and apply rewrite if configured
    rewrite_fn = getattr(self, "image_rewrite", None) or _get_image_rewrite_fn()
    str_images = []
    for img in raw_images:
      s = img.decode("utf-8") if hasattr(img, "decode") else str(img)
      if rewrite_fn is not None:
        s = rewrite_fn(s)
      str_images.append(s)

    return list(dict.fromkeys(str_images))

  def _warm_batch(self, batch: Any, wait: bool = False):
    images = self._extract_images(batch)
    if images and self.fleet:
      is_shared_image = self.scaffold == "openhands" and bool(
          os.getenv("AGENT_SERVER_IMAGE")
      )
      default_replicas = (
          getattr(self.fleet.config, "max_concurrent", self.num_generations)
          if is_shared_image
          else self.num_generations
      )
      target_replicas = self.max_warmpool_replicas or default_replicas
      try:
        self.fleet.warm_images(
            images, replicas_override=target_replicas, wait=wait
        )
        logging.info(
            "[PrewarmDatasetIterator] Pre-warming %d image(s) (%d replicas"
            " each) on K8s: %s",
            len(images),
            target_replicas,
            images[:3],
        )
      except Exception as e:
        logging.warning("[PrewarmDatasetIterator] Warm note: %s", e)

  def _unwarm_batch(self, images: list[str]):
    if images and self.fleet:
      for img in images:
        if self.scaffold == "openhands" and os.getenv("AGENT_SERVER_IMAGE"):
          continue
        try:
          self.fleet.unwarm_image(img)
          logging.info(
              "[PrewarmDatasetIterator] Unwarmed finished pool on K8s: %s",
              img,
          )
        except Exception as e:
          logging.warning("[PrewarmDatasetIterator] Unwarm note: %s", e)

  def __iter__(self):
    return self

  def __next__(self):
    if self.current_batch is None:
      raise StopIteration

    # 1. Deliver current batch to Tunix
    batch_to_return = self.current_batch
    current_images = self._extract_images(batch_to_return)

    # 2. 🧹 Unwarm previous batch (which has completed its execution)
    if self.prev_batch_images:
      active_images = set(
          current_images + self._extract_images(self.next_batch)
      )
      for old_img in self.prev_batch_images:
        if old_img not in active_images:
          self._unwarm_batch([old_img])

    # 3. Shift window: next becomes current
    self.prev_batch_images = current_images
    self.current_batch = self.next_batch

    # 4. 🚀 Pull fresh next batch and kick off background pre-warm on K8s!
    try:
      self.next_batch = next(self.dataset_iter)
      self._warm_batch(self.next_batch, wait=False)
    except StopIteration:
      self.next_batch = None

    return batch_to_return


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
  Action = None


class _ActionFallback:
  """Minimal Action parser fallback when r2egym is not installed."""

  def __init__(self, function_name: str, parameters: dict[str, str]):
    self.function_name = function_name
    self.parameters = parameters

  @classmethod
  def from_string(cls, action_str: str) -> "_ActionFallback":
    fn_match = re.search(r"<function\s*=\s*([^>]+)>", action_str)
    function_name = fn_match.group(1).strip() if fn_match else ""
    pattern = r"<parameter\s*=\s*([^>]+)>(.*?)</parameter>"
    param_matches = re.findall(pattern, action_str, flags=re.DOTALL)
    params = {k.strip(): v.strip() for k, v in param_matches}
    return cls(function_name, params)


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
        scaffold: Scaffold tool set ('r2egym', 'sweagent', or 'openhands').
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
    self.workspace = None
    self.handle = None
    self.verbose = verbose
    self.scaffold = scaffold
    self.use_agent_sandbox = use_agent_sandbox
    self.fleet = fleet
    self._cached_reward = None

    assert scaffold in [
        "r2egym",
        "sweagent",
        "openhands",
    ], f"Invalid scaffold: {scaffold}, must be one of ['r2egym', 'sweagent', 'openhands']"
    super().__init__(max_steps=max_steps)

    if not hasattr(self, "extra_kwargs"):
      self.extra_kwargs = {}

    self.extra_kwargs["group_id"] = group_id
    self.extra_kwargs["pair_index"] = pair_index

  def _initial_observation(self) -> Any:
    if not self.env and not self.workspace:
      if self.use_agent_sandbox:
        _patch_r2egym_for_agent_sandbox()
        from agent_sandbox_rl import Task  # pytype: disable=import-error

        fleet = self.fleet or _get_global_fleet()
        msg = (
            "[SWEEnv] Acquiring SandboxHandle from SandboxFleet!"
        )
        logging.info(msg)
        task_id = str(
            self.entry.get(
                "instance_id", self.entry.get("docker_image", "default")
            )
        )
        task = None
        if hasattr(fleet, "tasks") and fleet.tasks:
          for t in fleet.tasks:
            if t.id == task_id:
              task = t
              break
        if task is None:
          task_img = (
              os.getenv("AGENT_SERVER_IMAGE")
              if self.scaffold == "openhands" and os.getenv("AGENT_SERVER_IMAGE")
              else self.entry.get("docker_image", "default")
          )
          if isinstance(task_img, (list, np.ndarray)):
            task_img = task_img[0] if len(task_img) > 0 else "default"
          task_img_str = str(task_img)
          rewrite_fn = _get_image_rewrite_fn(
              getattr(fleet, "_image_rewrite_fn", None)
          )
          if rewrite_fn and not (
              self.scaffold == "openhands" and os.getenv("AGENT_SERVER_IMAGE")
          ):
            task_img_str = rewrite_fn(task_img_str)
          task = Task(
              id=task_id,
              image=task_img_str,
              metadata={"ds": self.entry},
          )
        max_acquire_retries = 5
        for attempt in range(max_acquire_retries):
          try:
            self.handle = fleet.acquire(task)
            break
          except Exception as e:
            if attempt < max_acquire_retries - 1:
              logging.warning(
                  "[SWEEnv] fleet.acquire failed (attempt %d/%d): %s; retrying in %ds...",
                  attempt + 1,
                  max_acquire_retries,
                  e,
                  5 * (attempt + 1),
              )
              time.sleep(5 * (attempt + 1))
            else:
              raise
        if self.scaffold == "openhands":
          from agent_sandbox_rl.adapters.openhands import make_handle_workspace  # pytype: disable=import-error
          ws_kwargs = {}
          if os.getenv("SANDBOX_SESSION_KEY"):
            ws_kwargs["api_key"] = os.getenv("SANDBOX_SESSION_KEY")
          if os.getenv("ROUTER_URL"):
            ws_kwargs["router_url"] = os.getenv("ROUTER_URL")
          if os.getenv("ROUTER_AUTH_TOKEN"):
            ws_kwargs["router_auth_token"] = os.getenv("ROUTER_AUTH_TOKEN")
          ws_kwargs["working_dir"] = os.getenv(
              "OPENHANDS_WORKING_DIR", "/testbed"
          )
          self.workspace = make_handle_workspace(self.handle, **ws_kwargs)
          try:
            from agent_sandbox_rl.adapters.r2egym import (  # pytype: disable=import-error
                make_fleet_repo_env,
                r2egym_command_files,
            )
            cmd_files = r2egym_command_files()
            self.env = make_fleet_repo_env(
                self.handle,
                command_files=cmd_files,
                step_timeout=self.step_timeout,
                reward_timeout=self.reward_timeout,
                verbose=self.verbose,
            )
          except Exception as e:
            logging.debug(
                "[SWEEnv] Could not bind FleetRepoEnv alongside OpenHands"
                " workspace: %s",
                e,
            )
          self._setup_openhands_workspace()
        else:
          from agent_sandbox_rl.adapters.r2egym import (  # pytype: disable=import-error
              make_fleet_repo_env,
              r2egym_command_files,
          )
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
        elif self.scaffold == "sweagent":
          self.env.add_commands(SWEAGENT_COMMAND_FILES)
    else:
      if self.env is not None:
        self.env.reset()

    self.final_reward_fn = self.compute_reward
    self._cached_reward = None
    self.total_steps = 0

    if self.workspace is not None:
      return str(
          self.entry.get("problem_statement")
          or self.entry.get("instruction")
          or ""
      )

    # Polls docker runtime to get task instruction.
    return self.env.get_task_instruction()  # pytype: disable=attribute-error

  def _setup_openhands_workspace(self) -> None:
    """Configure repository environment in the OpenHands workspace."""
    if self.workspace is None:
      return

    entry = getattr(self, "entry", None) or {}
    repo = (
        entry.get("repo_name")
        or entry.get("repo")
        or ""
    )
    commit = (
        entry.get("commit_hash")
        or entry.get("base_commit")
        or ""
    )
    if isinstance(repo, (list, np.ndarray)):
      repo = repo[0] if len(repo) > 0 else ""
    if isinstance(commit, (list, np.ndarray)):
      commit = commit[0] if len(commit) > 0 else ""

    repo_map = {
        "pandas": "https://github.com/pandas-dev/pandas.git",
        "numpy": "https://github.com/numpy/numpy.git",
        "pillow": "https://github.com/python-pillow/Pillow.git",
        "tornado": "https://github.com/tornadoweb/tornado.git",
        "orange3": "https://github.com/biolab/orange3.git",
        "datalad": "https://github.com/datalad/datalad.git",
        "aiohttp": "https://github.com/aio-libs/aiohttp.git",
        "pyramid": "https://github.com/Pylons/pyramid.git",
        "scrapy": "https://github.com/scrapy/scrapy.git",
        "coveragepy": "https://github.com/nedbat/coveragepy.git",
    }
    if repo in repo_map:
      repo_url = repo_map[repo]
    elif "/" in repo:
      repo_url = (
          f"https://github.com/{repo}.git"
          if not repo.startswith("http")
          else repo
      )
    elif repo:
      repo_url = f"https://github.com/{repo}/{repo}.git"
    else:
      repo_url = ""

    setup_cmds = [
        "git config --global user.email 'openhands@agent.sandbox'",
        "git config --global user.name 'OpenHands Agent'",
        "git config --global --add safe.directory /testbed 2>/dev/null || true",
        "git config --global --add safe.directory /workspace 2>/dev/null || true",
    ]
    if repo_url:
      setup_cmds.append(
          f"if [ ! -d /testbed/.git ] && [ ! -d /workspace/.git ]; then "
          f"  git clone {repo_url} /tmp/repo_clone && "
          f"  cp -a /tmp/repo_clone/. /workspace/ && "
          f"  rm -rf /tmp/repo_clone && "
          f"  (([ ! -d /testbed ] || rmdir /testbed 2>/dev/null || true) && ln -s /workspace /testbed 2>/dev/null || true); "
          f"elif [ ! -e /testbed ] && [ -d /workspace ]; then "
          f"  ln -s /workspace /testbed 2>/dev/null || true; "
          f"fi"
      )
      if commit:
        setup_cmds.append(
            f"if [ ! -d /testbed/.git ] || [ -L /testbed ]; then "
            f"  if [ -d /workspace/.git ]; then "
            f"    cd /workspace && git fetch origin 2>/dev/null || true && "
            f"    git checkout -f {commit} 2>/dev/null || true; "
            f"  fi; "
            f"fi"
        )
    else:
      setup_cmds.append(
          f"if [ ! -e /testbed ] && [ -d /workspace ]; then "
          f"  ln -s /workspace /testbed 2>/dev/null || true; "
          f"fi"
      )

    full_setup_cmd = " && ".join(setup_cmds)
    try:
      logging.info(
          "[SWEEnv] Configuring repository environment for %s...",
          repo,
      )
      res = self.workspace.execute_command(full_setup_cmd, timeout=180.0)
      if res.exit_code != 0:
        logging.warning(
            "[SWEEnv] Repository setup exit code %s: %s",
            res.exit_code,
            res.stderr or res.stdout,
        )
      else:
        logging.info(
            "[SWEEnv] Successfully configured repository environment for %s", repo
        )
    except Exception as e:
      logging.warning(
          "[SWEEnv] Failed to set up repository in workspace: %s", e
      )

  def compute_reward(self, verbose: bool = False) -> float:
    """Compute task reward for the current environment state."""
    if self._cached_reward is not None:
      return self._cached_reward

    if self.env is not None and hasattr(self.env, "compute_reward"):
      try:
        reward = float(self.env.compute_reward(verbose=verbose))
      except TypeError:
        reward = float(self.env.compute_reward())
      self._cached_reward = reward
      return reward

    if self.workspace is not None:
      test_patch = self.entry.get("test_patch")
      if test_patch:
        if hasattr(test_patch, "decode"):
          test_patch = test_patch.decode("utf-8")
        apply_cmd = (
            f"cat << '__EOF_PATCH__' > /tmp/test_patch.diff\n{test_patch}\n__EOF_PATCH__\n"
            "(cd /testbed 2>/dev/null || cd /workspace) && "
            "(git apply --whitespace=nowarn /tmp/test_patch.diff 2>/dev/null || patch -p1 < /tmp/test_patch.diff)"
        )
        try:
          self.workspace.execute_command(apply_cmd, timeout=60.0)
        except Exception as e:
          logging.warning("[SWEEnv] Failed to apply test_patch: %s", e)

      eval_script = (
          self.entry.get("eval_script")
          or self.entry.get("run_tests_regression")
      )
      if eval_script:
        if hasattr(eval_script, "decode"):
          eval_script = eval_script.decode("utf-8")
        eval_cmd = (
            f"cat << '__EOF_EVAL__' > /tmp/eval.sh\n{eval_script}\n__EOF_EVAL__\n"
            "chmod +x /tmp/eval.sh && (cd /testbed 2>/dev/null || cd /workspace) && /tmp/eval.sh"
        )
        try:
          res = self.workspace.execute_command(
              eval_cmd, timeout=float(self.reward_timeout)
          )
          reward = 1.0 if res.exit_code == 0 else 0.0
          self._cached_reward = reward
          return reward
        except Exception as e:
          logging.warning("[SWEEnv] Reward computation error: %s", e)
          self._cached_reward = 0.0
          return 0.0

      test_cmd = self.entry.get("test_command")
      if test_cmd:
        try:
          res = self.workspace.execute_command(
              f"(cd /testbed 2>/dev/null || cd /workspace) && {test_cmd}",
              timeout=float(self.reward_timeout),
          )
          reward = 1.0 if res.exit_code == 0 else 0.0
          self._cached_reward = reward
          return reward
        except Exception as e:
          logging.warning("[SWEEnv] Reward computation error: %s", e)
          self._cached_reward = 0.0
          return 0.0

      try:
        run_cmd = (
            "if [ -f /testbed/run_tests.sh ]; then "
            "cd /testbed && bash /testbed/run_tests.sh; "
            "elif [ -f /run_tests.sh ]; then "
            "(cd /testbed 2>/dev/null || cd /workspace) && bash /run_tests.sh; "
            "else exit 1; fi"
        )
        res = self.workspace.execute_command(
            run_cmd,
            timeout=float(self.reward_timeout),
        )
        reward = 1.0 if res.exit_code == 0 else 0.0
        self._cached_reward = reward
        return reward
      except Exception as e:
        logging.warning("[SWEEnv] Reward computation error: %s", e)
        self._cached_reward = 0.0
        return 0.0

    return 0.0

  def _step_impl(self, action: Any) -> EnvStepResult:
    global Action
    if Action is None:
      try:
        from r2egym.agenthub.action import Action  # pytype: disable=import-error
      except ImportError:
        Action = _ActionFallback
    if isinstance(action, str):
      action_obj = Action.from_string(action)
    else:
      action_obj = action

    if not action_obj.function_name:
      return EnvStepResult(
          observation="",
          reward=0,
          done=False,
          info={"max_steps": self.max_steps},
      )

    if self.scaffold == "openhands" and self.workspace is not None:
      if action_obj.function_name in ("finish", "submit"):
        return EnvStepResult(
            observation="Task submitted.",
            reward=0,
            done=True,
            info={"max_steps": self.max_steps},
        )

      if action_obj.function_name == "str_replace_editor" and self.env is not None:
        obs, reward, done, info = self.env.step(action_obj)
        self.total_steps += 1
        return EnvStepResult(
            observation=str(obs),
            reward=0,
            done=done,
            info={"max_steps": self.max_steps},
        )

      if action_obj.function_name != "execute_bash":
        return EnvStepResult(
            observation=(
                f"ERROR: Tool '{action_obj.function_name}' is not recognized. "
                "Only 'execute_bash', 'str_replace_editor', and 'submit' are available."
            ),
            reward=0,
            done=False,
            info={"max_steps": self.max_steps},
        )

      cmd = action_obj.parameters.get("command") or action_obj.parameters.get(
          "cmd"
      )
      if not cmd:
        return EnvStepResult(
            observation="ERROR: No command specified for execute_bash.",
            reward=0,
            done=False,
            info={"max_steps": self.max_steps},
        )

      try:
        wrapped_cmd = f"(cd /testbed 2>/dev/null || cd /workspace) && {cmd}"
        result = self.workspace.execute_command(
            wrapped_cmd, timeout=float(self.step_timeout)
        )
        obs = (
            result.stdout
            if result.exit_code == 0
            else f"{result.stdout}\n{result.stderr}"
        )
      except Exception as e:
        obs = f"Command execution failed: {e}"
      self.total_steps += 1
      return EnvStepResult(
          observation=obs,
          reward=0,
          done=False,
          info={"max_steps": self.max_steps},
      )

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
    self._cached_reward = None
    if self.env is not None:
      self.env.close()

    if getattr(self, "workspace", None) is not None:
      try:
        self.workspace.cleanup()
      except Exception as e:
        logging.warning("[SWEEnv] Workspace cleanup note: %s", e)
      self.workspace = None

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
