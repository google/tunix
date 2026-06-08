import os
import sys
import time

from flax import nnx
import jax
import numpy as np

import tunix
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import utils
# from tunix.google.examples.remote_copy import main_proc
# from tunix.google.examples.remote_copy import rl_proc
# from tunix.google.examples.remote_copy import rollout_proc
# from tunix.google.examples.remote_copy import resources
# from tunix.google.examples.remote_copy import channels

import multiprocessing

from absl import app
from absl import logging
import threading
import time
import traceback
from typing import Any
import multiprocess as mp

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
        print(self._sink.get())
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
      log: Any,
      config: dict[str, Any],
      **kwargs,
  ):
    super().__init__()
    self.name = name if name else self.__class__.__name__
    self.log = log
    self.config = config
    self.kwargs = kwargs

  def __call__(self):
    try:
      import logging
      import sys
      if "verbose" in self.kwargs and self.kwargs["verbose"]:
        self.log.put(f"[{self.name}] intercepting stdout and logging...")
        class StdoutInterceptor(object):
          def __init__(self, name, log):
            self.name = name
            self.log = log
          def write(self, buf):
            for line in buf.rstrip().splitlines():
              # if "INFO:absl:[0] /jax/core/compile" not in line:
              self.log.put(f"[{self.name}] {line.rstrip()}")
          def flush(self):
              pass
        sys.stdout = StdoutInterceptor(self.name, self.log)
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

      import jax
      from pathwaysutils.experimental.shared_pathways_service import isc_pathways

      config = self.config
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
          self.log.put(f"[{self.name}] set compilation cache dir {config["jax_compilation_cache_dir"]}")
          devices = jax.devices()
          self.log.put(f"[{self.name}] connected to Pathways. Devices: {len(devices)}")
          try:
            self._main()
          except Exception as e:
            self.log.put(f"[{self.name}] error in _main: {e}\n{traceback.format_exc()}")
    except Exception as e:
      self.log.put(f"[{self.name}] error connecting to pathways: {e}\n{traceback.format_exc()}")

  def _main(self):
    pass

from tunix.cli import grpo_main

class GrpoMainProc(PathwaysClientProcess):
  def __init__(
      self,
      *,
      name: str | None = None,
      log: Any,
      config: Any,
      argv: Any,
      **kwargs,
  ):
    super().__init__(name=name, log=log, config=config, **kwargs)
    self.argv = argv

  def _main(self):
    self.log.put(f"[{self.name}] start GRPO main.")
    pipeline = grpo_main.GrpoPipeline(self.argv, **self.kwargs)
    logging.info(
        "--- Launching GRPO pipeline with following config ---\n"
        "%r\n--------------------------",
        pipeline.config,
    )
    runner = pipeline.prepare_grpo_trainer()
    runner()

grpo_main_config = {
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
  # "proxy_server_image": "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/wenxindong/unsanitized_proxy_server@sha256:86fedb263c8221bb878c2d301cb45e7c93f54f62872c5c79b055a267da780f42",
  # "proxy_server_image": "us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/gke/ksadi/unsanitized_proxy_server@sha256:e5ad4ef0ec907ba2378394f59c4ba074a82231112c03d7f80d7c4a38b19c043c",
  "proxy_server_image": "us-central1-docker.pkg.dev/cloud-tpu-multipod-dev/yangmu/tunix/unsanitized_proxy_server@sha256:c243fdd5ee52ef0f1d21165cf737540de54760882f82be358897f0b444c104bd",

  "jax_compilation_cache_dir": "/tmp/.jax_cache",
  "model_file_path": "/mnt/disks/github/.models"
}

def main(argv, **kwargs):
  logger = ProcessLogger()
  logger.start()
  logger.sink.put("begin")

  main_proc = mp.Process(
    target=GrpoMainProc(
      name=f"grpo",
      log=logger.sink,
      config=grpo_main_config,
      verbose=False,
      argv=argv,
      **kwargs,
    )
  )
  main_proc.start()

  main_proc.join()
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
