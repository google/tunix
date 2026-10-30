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

"""Reference inference worker node for distributed RL.

Launched as
`python -m tunix.dist_rl.distributed.runtime.main --discovery_addrs=...
--process_main=tunix.dist_rl.launch.nodes.inference.main <config overrides>`.
The overrides are resolved once by `resolve_config.initialize`, and the frozen
reference model and worker are built directly from that validated config.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from tunix.dist_rl.distributed.runtime import context as context_lib
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.worker import inference_worker as exp_inference_worker
from tunix.models import automodel
from tunix.rl.inference import inference_worker as rl_inference_worker

LOCAL_WORKER_ID = "reference-inference-0"

_REFERENCE_KEY = "reference_model_config"


def main(
    argv: list[str], context: context_lib.ProcessContext | None = None
) -> None:
  """Serves a reference InferenceWorker over gRPC until SIGINT/SIGTERM.

  Args:
    argv: `override_config_file=...` and `section.key=value` overrides.
    context: Runtime process context providing JAX init and discovery.
  """
  process_context = node_utils.require_discovery_context(context)
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [InferenceNode] %(message)s",
      force=True,
  )

  hp = resolve_config.initialize(argv)
  logging.info("Resolved config: %s", hp.config)

  process_context.jax.initialize()

  ref = hp.config_mapping(_REFERENCE_KEY)
  worker_id = node_utils.worker_id(hp, "inference", LOCAL_WORKER_ID)
  port = node_utils.service_port(hp, "inference")

  model_dir = node_utils.ensure_local_safetensors(
      str(ref["model_download_path"]),
      str(ref["model_id"]),
      model_key=_REFERENCE_KEY,
  )
  logging.info("Prepared reference safetensors directory: %s", model_dir)

  pad_id, eos_id = node_utils.pad_and_eos_ids(node_utils.load_hf_tokenizer(hp))

  logging.info("Creating inference mesh...")
  mesh = node_utils.create_mesh(hp, _REFERENCE_KEY, "Inference")
  logging.info("Inference mesh: %s", mesh)

  logging.info("Loading frozen reference model...")
  with mesh:
    reference_model, _ = automodel.AutoModel.from_pretrained(
        model_id=str(ref["model_name"]),
        mesh=mesh,
        model_path=model_dir,
        dtype=str(ref["dtype"]),
        load_dtype=str(ref["load_dtype"]),
    )
    core = rl_inference_worker.InferenceWorker({"reference": reference_model})

  worker_service = exp_inference_worker.InferenceWorker(
      core,
      worker_id=worker_id,
      pad_id=pad_id,
      eos_id=eos_id,
      chunk_size=int(
          hp.config_mapping("rl_training_config")[
              "compute_logps_micro_batch_size"
          ]
      ),
      execution_context=mesh,
  )

  asyncio.run(
      node_utils.serve_worker(
          worker_service,
          process_context=process_context,
          role="inference",
          worker_id_str=worker_id,
          port=port,
      )
  )


if __name__ == "__main__":
  main(sys.argv[1:])
