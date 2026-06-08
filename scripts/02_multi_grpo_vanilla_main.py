from absl import app
from absl import logging
from dataclasses import dataclass
import importlib
import multiprocessing
import multiprocess as mp
import os
import numpy as np
import threading
import time
import traceback
import threading
from typing import Any, Optional, Sequence, Tuple
from tqdm import tqdm

import jax
import jaxtyping
from flax import nnx

import tunix
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import utils
# from tunix.google.examples.remote_copy import main_proc
# from tunix.google.examples.remote_copy import rl_proc
# from tunix.google.examples.remote_copy import rollout_proc
# from tunix.google.examples.remote_copy import resources
# from tunix.google.examples.remote_copy import channels
from tunix.cli import grpo_main

JaxDevice = Any

def import_symbol(fqn: str, split: int = 1) -> Any:
    """Imports a symbol (class or function) from its fully qualified name."""
    if "." not in fqn:
        # If no dots, assume it's a built-in or already in the global namespace
        # (Though usually everything should be fully qualified in this system)
        import builtins
        return getattr(builtins, fqn, None)
    module_path, *symbol_names = fqn.rsplit(".", split)
    symbol = importlib.import_module(module_path)
    for symbol_name in symbol_names:
        symbol = getattr(symbol, symbol_name)
    return symbol

@dataclass(frozen=True)
class MeshSpec:
    """Specification for reconstructing a JAX Mesh locally."""
    mesh_shape: Sequence[int]
    axis_names: Sequence[str]

def serialize_mesh(mesh: jax.sharding.Mesh) -> MeshSpec:
    """Serializes a JAX Mesh into a MeshSpec."""
    return MeshSpec(
        mesh_shape=mesh.devices.shape,
        axis_names=mesh.axis_names,
    )

def deserialize_mesh(spec: MeshSpec, devices: Sequence[JaxDevice]) -> jax.sharding.Mesh:
    """Reconstructs a JAX Mesh locally using available devices."""
    return jax.make_mesh(axis_shapes=spec.mesh_shape,
                         axis_names=spec.axis_names,
                         axis_types=(jax.sharding.AxisType.Auto,) * len(spec.axis_names),
                         devices=devices)

def serialize_sharding(sharding: Any) -> dict[str, Any]:
    """Serializes a JAX Sharding for transport."""
    spec = getattr(sharding, "spec", jax.sharding.PartitionSpec())
    mesh_axis_names = []
    if hasattr(sharding, "mesh"):
      mesh_axis_names = sharding.mesh.axis_names
    return {
        "spec": spec,
        "mesh_axis_names": mesh_axis_names,
    }

def deserialize_sharding(sharding_dict: dict[str, Any], mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    """Deserializes a JAX NamedSharding from its dictionary representation."""
    # Note: This assumes the mesh is already reconstructed locally.
    return jax.sharding.NamedSharding(mesh, sharding_dict["spec"])

@dataclass(frozen=True)
class ModelSpec:
    """Specification for loading a model via fully qualified names."""
    model_id: str
    model_class: str   # FQN, e.g., "tunix.models.llama3.model.Model"
    model_params: str  # FQN, e.g., "tunix.models.llama3.params"
    model_config: str  # FQN, e.g., "tunix.models.llama3.model.ModelConfig.llama3p2_1b"

@dataclass(frozen=True)
class RolloutWorkerCreate:
    """Request from rl_proc to main_proc to spawn a new worker."""
    worker_type: str  # e.g., "vanilla"
    worker_config: base_rollout.RolloutConfig
    model_spec: ModelSpec
    mesh_spec: MeshSpec

class RolloutGenerateRequest:
    """Request from rl_proc to rollout_proc to perform inference."""
    # prompts: Sequence[str]
    # response_queue: Any  # for RolloutOutput
    # rollout_config: base_rollout.RolloutConfig | None = None
    def __init__(self, prompts: Sequence[str], response_queue: Any):
      super().__init__()
      self.prompts = prompts
      self.response_queue = response_queue

class RolloutUpdateParamsRequest:
    """Request from rl_proc to rollout_proc to update model weights."""
    # sender_info: Any  # sender_info from experimental_remote_copy_prepare
    # shardings: Any    # List of shardings or serialized sharding info
    def __init__(self, sender_info: Any, shardings: Any):
      super().__init__()
      self.sender_info = sender_info
      self.shardings = shardings

class RolloutPadIdRequest:
    def __init__(self, response_queue: Any):
      super().__init__()
      self.response_queue = response_queue

class RolloutEosIdRequest:
    def __init__(self, response_queue: Any):
      super().__init__()
      self.response_queue = response_queue

class ProcessLogger(threading.Thread):
  def __init__(self):
    super().__init__()
    self._sink = mp.Queue()
    self._stop_event = threading.Event()

  @property
  def sink(self):
    return self._sink

  def run(self):
    while not self._stop_event.is_set():
      while not self._sink.empty():
        print(self._sink.get(), flush=True)
      time.sleep(0.1)

  def stop(self):
    self._stop_event.set()

class NoopLogger(threading.Thread):
  def __init__(self):
    super().__init__()
    self._sink = mp.Queue()
    self._stop_event = threading.Event()

  @property
  def sink(self):
    return self._sink

  def run(self):
    while not self._stop_event.is_set():
      while not self._sink.empty():
        self._sink.get()
      time.sleep(0.1)

  def stop(self):
    self._stop_event.set()

class PathwaysClientProcess:
  def __init__(
      self,
      *,
      name: str | None = None,
      logger: Any,
      backend_config: dict[str, Any],
      **kwargs,
  ):
    super().__init__()
    self.name = name if name else self.__class__.__name__
    self.logger = logger
    self.backend_config = backend_config
    self.kwargs = kwargs

  def __call__(self):
    try:
      import logging
      import sys
      if "verbose" in self.kwargs and self.kwargs["verbose"]:
        self.logger.put(f"[{self.name}] intercepting stdout and logging...")
        class StdoutInterceptor(object):
          def __init__(self, name, logger):
            self.name = name
            self.logger = logger
          def write(self, buf):
            for line in buf.rstrip().splitlines():
              # if "INFO:absl:[0] /jax/core/compile" not in line:
              self.logger.put(f"[{self.name}] {time.strftime('%Y-%m-%d %H:%M:%S')} | {line.rstrip()}")
          def flush(self):
              pass
        sys.stdout = StdoutInterceptor(self.name, self.logger)
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

      import jax
      from pathwaysutils.experimental.shared_pathways_service import isc_pathways

      config = self.backend_config
      with isc_pathways.connect(
            cluster=config["cluster"],
            project=config["project"],
            region=config["region"],
            gcs_bucket=config["gcs_bucket"],
            pathways_service=config["pathways_service"],
            expected_tpu_instances={config["tpu_type"]: config["tpu_slices"]},
            proxy_server_image=config["proxy_server_image"],
            # **kwargs,
        ):
          jax.experimental.compilation_cache.compilation_cache.set_cache_dir(config["jax_compilation_cache_dir"])
          self.logger.put(f"[{self.name}] set compilation cache dir {config["jax_compilation_cache_dir"]}")
          devices = jax.devices()
          self.logger.put(f"[{self.name}] connected to Pathways. Devices: {len(devices)}")
          try:
            self._main()
          except Exception as e:
            self.logger.put(f"[{self.name}] error in _main: {e}\n{traceback.format_exc()}")
    except Exception as e:
      self.logger.put(f"[{self.name}] error connecting to pathways: {e}\n{traceback.format_exc()}")

  def _main(self):
    pass

class RolloutWorkerProc(PathwaysClientProcess):
  def __init__(
      self,
      *,
      name: str | None = None,
      logger: Any,
      backend_config: Any,
      worker_request_channel: Any,
      worker_ready_channel: Any,
      model_spec: ModelSpec,
      mesh_spec: MeshSpec,
      rollout_config: base_rollout.RolloutConfig,
      **kwargs,
  ):
    super().__init__(name=name, logger=logger, backend_config=backend_config, **kwargs)
    self.worker_request_channel = worker_request_channel
    self.worker_ready_channel = worker_ready_channel
    self.model_spec = model_spec
    self.mesh_spec = mesh_spec
    self.rollout_config = rollout_config

  def _main(self):
    self.logger.put(f"[{self.name}] importing libraries...")
    from tunix.rl.rollout import vanilla_rollout
    from tunix.rl.rollout import vllm_rollout
    from flax import nnx
    import numpy as np
    import jax
    # from google3.third_party.pathways.jax.ifrt import client as ifrt_client
    import transformers

    self.logger.put(f"[{self.name}] importing model symbols...")
    # model_class = import_symbol(self.model_spec.model_class)
    model_params = import_symbol(self.model_spec.model_params)
    model_config = import_symbol(self.model_spec.model_config, 2)
    if callable(model_config):
        model_config = model_config()

    self.logger.put(f"[{self.name}] reconstructing mesh...")
    model_mesh = deserialize_mesh(self.mesh_spec, jax.devices()[:np.prod(self.mesh_spec.mesh_shape)])

    self.logger.put(f"[{self.name}] initializing model...")
    # show_hbm_usage()
    model = model_params.create_model_from_safe_tensors(
        os.path.join(self.backend_config["model_file_path"], self.model_spec.model_id),
        model_config,
        model_mesh
    )
    # show_hbm_usage()

    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_spec.model_id)

    # self.rollout = vanilla_rollout.VanillaRollout(
    #     model=model,
    #     tokenizer=tokenizer, 
    #     cache_config_or_size=base_rollout.CacheConfig(
    #         cache_size=self.rollout_config.kv_cache_size,
    #         num_layers=model.config.num_layers,
    #         num_kv_heads=model.config.num_kv_heads,
    #         head_dim=model.config.head_dim,
    #     )
    # )
    self.rollout = vllm_rollout.VllmRollout(
        model=model,
        tokenizer=tokenizer, 
        cache_config_or_size=self.rollout_config.kv_cache_size,
        mesh=model_mesh,
        rollout_config=self.rollout_config,
    )
    self.logger.put(f"[{self.name}] ready to handle requests.")
    self.logger.put(f"[{self.name}] test prompt output: what is 1+1? => {self.rollout.generate(prompts=["what is 1+1?"], rollout_config=self.rollout_config).text[0].splitlines()[0]}")
    self.worker_ready_channel.put(True)

    # while True:
    #   request = self.worker_request_channel.get()
    #   if request is None: # Termination signal
    #     self.logger.put(f"[{self.name}] termination signal received.")
    #     break
    # return

    while True:
      request = self.worker_request_channel.get()
      if request is None: # Termination signal
        self.logger.put(f"[{self.name}] termination signal received.")
        break

      if isinstance(request, RolloutGenerateRequest):
        # self.logger.put(f"[{self.name}] handling RolloutGenerateRequest")
        output = self.rollout.generate(
            prompts=request.prompts,
            rollout_config=self.rollout_config # or request.rollout_config
        )
        # self.logger.put(f"[{self.name}] generation: {output}")
        if request.response_queue:
          request.response_queue.put(output)
        # self.logger.put(f"[{self.name}] generation complete.")

      elif isinstance(request, RolloutUpdateParamsRequest):
        self.logger.put(f"[{self.name}] handling RolloutUpdateParamsRequest")
        # _, abstract_weights = nnx.split(model)
        # _, treedef = jax.tree_util.tree_flatten(abstract_weights)

        # target_shardings = [
        #     deserialize_sharding(s, model_mesh) 
        #     for s in request.shardings
        # ]
        # received_arrays = ifrt_client.experimental_remote_copy_recv(
        #     sender_info=request.sender_info,
        #     out_shardings=target_shardings,
        # )

        # new_params = jax.tree_util.tree_unflatten(treedef, received_arrays)
        # self.rollout.update_params(new_params)
        # self.logger.put(f"[{self.name}] params updated.")
      elif isinstance(request, RolloutPadIdRequest):
        self.logger.put(f"[{self.name}] handling RolloutPadIdRequest")
        if request.response_queue:
          request.response_queue.put(self.rollout.pad_id())
      elif isinstance(request, RolloutEosIdRequest):
        self.logger.put(f"[{self.name}] handling RolloutEosIdRequest")
        if request.response_queue:
          request.response_queue.put(self.rollout.eos_id())
      else:
        self.logger.put(f"[{self.name}] unknown request: {request}")

class RolloutEngineClient:
  def __init__(self,
               worker_request_channels: list[Any]):
    # must be pickle-able
    self.worker_request_channels = worker_request_channels
    self.num_workers = len(worker_request_channels)
    self.dispatch_id = 0
    self.semaphore = None
    self.manager = None
    self.cache = {}

  def post_init(self):
    self.semaphore = threading.Semaphore(self.num_workers * 20)
    self.manager = mp.Manager()

  def generate(
      self, prompts, rollout_config: base_rollout.RolloutConfig, **kwargs
  ) -> base_rollout.RolloutOutput:
    if "trajectory_id" in kwargs:
      worker_id = int(kwargs["trajectory_id"] % self.num_workers)
    else:
      worker_id = self.dispatch_id % self.num_workers
      self.dispatch_id += 1

    with self.semaphore:
      # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] generate() to worker {worker_id}")
      response = self.manager.Queue()
      self.worker_request_channels[worker_id].put(
          RolloutGenerateRequest(
              prompts=prompts,
              response_queue=response,
              # rollout_config=rollout_config,
          )
      )
      return response.get()

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
      **kwargs
  ) -> jax.Array:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    self.logger.put(f"[engine] ignore update_params().")
    pass

  def pad_id(self) -> int:
    if "pad_id" not in self.cache:
      response = self.manager.Queue()
      self.worker_request_channels[0].put(
          RolloutPadIdRequest(
              response_queue=response,
          )
      )
      self.cache["pad_id"] = response.get()
      print(f"pad_id() => {self.cache["pad_id"]}")
    return self.cache["pad_id"]

  def eos_id(self) -> int:
    if "eos_id" not in self.cache:
      response = self.manager.Queue()
      self.worker_request_channels[0].put(
          RolloutEosIdRequest(
              response_queue=response,
          )
      )
      self.cache["eos_id"] = response.get()
      print(f"eos_id() => {self.cache["eos_id"]}")
    return self.cache["eos_id"]

  def model(self) -> Any:
    return None

class RolloutEngine:
  def __init__(self,
               logger: Any,
               num_workers: int,
               init_parallelism: int,
               backend_config: dict[str, Any],
               worker_config: RolloutWorkerCreate,
               **kwargs):
    self.logger = logger
    self.num_workers = num_workers
    self.init_parallelism = init_parallelism
    self.backend_config = backend_config
    self.worker_config = worker_config
    self.kwargs = kwargs

    self.workers = []
    self.worker_request_channels = []
    self.worker_ready_channel = None

  def start(self) -> RolloutEngineClient:
    if self.workers:
      raise ValueError("Workers already started.")

    logger = self.logger
    num_workers = self.num_workers
    init_parallelism = self.init_parallelism
    backend_config = self.backend_config
    worker_config = self.worker_config

    worker_request_channels = []
    for _ in range(num_workers):
      worker_request_channels.append(mp.Queue())
    worker_ready_channel = mp.Queue(num_workers)

    print(f"{num_workers=}, {init_parallelism=}")
    assert num_workers % init_parallelism == 0
    workers = []
    worker_id = 0
    # with tqdm(total=num_workers, desc="worker") as pbar:
    for _ in range(num_workers // init_parallelism):
      for _ in range(init_parallelism):
        worker = mp.Process(
          target=RolloutWorkerProc(
            name=f"worker{worker_id}",
            logger=logger,
            backend_config=backend_config,
            worker_request_channel=worker_request_channels[worker_id],
            worker_ready_channel=worker_ready_channel,
            model_spec=worker_config.model_spec,
            mesh_spec=worker_config.mesh_spec,
            rollout_config=worker_config.worker_config,
            verbose=True,
          )
        )
        worker_id += 1
        worker.start()
        workers.append(worker)
      for _ in range(init_parallelism):
        worker_ready_channel.get()
        # pbar.update(1)

    self.workers = workers
    self.worker_request_channels = worker_request_channels
    self.worker_ready_channel = worker_ready_channel
    return RolloutEngineClient(worker_request_channels)

  def stop(self):
    for channel in self.worker_request_channels:
      channel.put(None)
    for worker in self.workers:
      worker.join()

    self.workers = []
    self.worker_request_channels = []
    self.worker_ready_channel = None

class GrpoMainProc(PathwaysClientProcess):
  def __init__(
      self,
      *,
      name: str | None = None,
      logger: Any,
      backend_config: Any,
      argv: Any,
      **kwargs,
  ):
    super().__init__(name=name, logger=logger, backend_config=backend_config, **kwargs)
    self.argv = argv

  def _main(self):
    self.logger.put(f"[{self.name}] start GRPO main.")
    pipeline = grpo_main.GrpoPipeline(self.argv, **self.kwargs)
    logging.info(
        "--- Launching GRPO pipeline with following config ---\n"
        "%r\n--------------------------",
        pipeline.config,
    )
    runner = pipeline.prepare_grpo_trainer()
    runner()

def main(argv, **kwargs):
  jax_backend_config = {
    # "project": "cloud-tpu-multipod-dev",
    # "cluster": "bodaborg-super-alpha-cluster",
    "project": "tpu-prod-env-automated",
    "cluster": "tunix-v7x-64",
    # "project": "tpu-prod-env-automated",
    # "cluster": "tunix-v7x-8",

    "region": "us-central1",
    "gcs_bucket": "gs://yangmu-us-central1",
    "pathways_service": "yangmu-ws-pathways-head-0-0.yangmu-ws:29001",
    "tpu_type": "tpu7x:2x2x1",
    "tpu_slices": 1,

    "proxy_server_image": "us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_proxy_server@sha256:c243fdd5ee52ef0f1d21165cf737540de54760882f82be358897f0b444c104bd",

    "jax_compilation_cache_dir": "/tmp/.jax_cache",

    "model_file_path": "/mnt/disks/github/.models", # TODO: remove
  }
  logger = ProcessLogger()
  logger.start()
  logger.sink.put("begin")

  # 1. start rollout engine
  model_spec = ModelSpec(
      # model_id="meta-llama/Llama-3.2-1B-Instruct",
      # model_class="tunix.models.llama3.model.Llama3",
      # model_params="tunix.models.llama3.params",
      # model_config="tunix.models.llama3.model.ModelConfig.llama3p2_1b",

      model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
      model_class="tunix.models.qwen2.model.Qwen2",
      model_params="tunix.models.qwen2.params",
      model_config="tunix.models.qwen2.model.ModelConfig.deepseek_r1_distill_qwen_1p5b"
  )
  mesh_spec = MeshSpec(
      mesh_shape=(4, 2),
      axis_names=("fsdp", "tp"),
  )
  rollout_config = base_rollout.RolloutConfig(
      max_tokens_to_generate=768,
      max_prompt_length=256,
      kv_cache_size=256 + 768 + 256,
      temperature=0.9,
      top_p=1.0,
      top_k=50,
      data_parallel_size=mesh_spec.mesh_shape[0],
      tensor_parallel_size=mesh_spec.mesh_shape[1],
  )
  worker_config: RolloutWorkerCreate = RolloutWorkerCreate(
    worker_type="vanilla",
    worker_config=rollout_config,
    model_spec=model_spec,
    mesh_spec=mesh_spec,
  )
  rollout_engine = RolloutEngine(
    logger=logger.sink,
    num_workers=1,
    init_parallelism=1,
    backend_config=jax_backend_config,
    worker_config=worker_config,
    **kwargs,
  )

  rollout_client = None
  main_proc = None
  try:
    rollout_client = rollout_engine.start() # block until all workers ready

    # 2. set rollout_config.rollout_engine

    # 3. start main proc
    main_proc = mp.Process(
      target=GrpoMainProc(
        name=f"grpo",
        logger=logger.sink,
        backend_config=jax_backend_config,
        multi_rollout_client=rollout_client, # collected by kwargs
        verbose=True,
        argv=argv,
        **kwargs,
      )
    )
    main_proc.start()

    # 4. wait main ready

    # 5. start trainer

  finally:
    # 6. cleanup
    if main_proc:
      main_proc.join()
    if rollout_engine:
      rollout_engine.stop()
    logger.sink.put("end")
    time.sleep(1)
    logger.stop()
    logger.join()

if __name__ == "__main__":
  try:
    multiprocessing.set_start_method("spawn")
  except RuntimeError:
    pass
  app.run(main)
