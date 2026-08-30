# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XManager launcher for distributed GSM8K GRPO on 1P Pathways and McJAX."""

import os
from typing import Any, Dict

from absl import app
from absl import flags
from absl import logging
from GOOGLE_INTERNAL_PACKAGE_PATH.learning.deepmind.xmanager2.client.launch.borg import gcl_utils
from GOOGLE_INTERNAL_PACKAGE_PATH.third_party.pathways.google.xmanager import service_lib
from xmanager import xm
from xmanager import xm_abc
from xmanager import xm_flags
from xmanager.contrib.internal import addressing
from xmanager.contrib.internal import xm_jax

_EXP_TITLE = flags.DEFINE_string(
    "exp_title",
    "tunix_math_gsm8k_dist_grpo",
    "Title for the XManager experiment.",
)

_CELL = flags.DEFINE_string(
    "cell",
    "cj",
    "Borg cell to run the experiment in.",
)

_PRIORITY = flags.DEFINE_integer(
    "priority",
    200,
    "Borg priority for jobs.",
)

_TRAINER_PLATFORM = flags.DEFINE_string(
    "trainer_platform",
    "vlp=2x2",
    "TPU platform and topology for the Pathways trainer service (e.g. 'vlp=2x2', 'glp=2x2', 'vf=2x2x1').",
)

_ROLLOUT_PLATFORM = flags.DEFINE_string(
    "rollout_platform",
    "vlp=2x2",
    "TPU platform and topology for the rollout worker (e.g. 'vlp=2x2', 'glp=2x2', 'vf=2x2x1').",
)

_MODEL_ID = flags.DEFINE_string(
    "model_id",
    "Qwen/Qwen3-0.6B",
    "HuggingFace or local path for the model.",
)

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "Qwen3-0.6B",
    "Model name corresponding to models.py registry (e.g. 'Qwen3-0.6B').",
)

_SAMPLER = flags.DEFINE_string(
    "sampler",
    "vanilla",
    "Rollout sampler implementation ('vanilla' or 'inprocess_vllm').",
)

_WEIGHT_SYNC_BACKEND = flags.DEFINE_string(
    "weight_sync_backend",
    "raiden",
    "Weight sync backend ('raiden' or 'none').",
)

_MAX_STEPS = flags.DEFINE_integer(
    "max_steps",
    10,
    "Maximum RL training steps.",
)

_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    4,
    "Prompt batch size per step.",
)

_MINI_BATCH_SIZE = flags.DEFINE_integer(
    "mini_batch_size",
    2,
    "Mini-batch size for trainer gradient updates.",
)

_NUM_GENERATIONS = flags.DEFINE_integer(
    "num_generations",
    2,
    "Number of generations per prompt in GRPO.",
)

_BETA = flags.DEFINE_float(
    "beta",
    0.0,
    "KL penalty coefficient in GRPO. Set 0.0 to run without reference model.",
)

_RESOURCE_MANAGER_RAM_GB = flags.DEFINE_integer(
    "resource_manager_ram_gb",
    25,
    "Host RAM in GB for the Pathways Resource Manager job.",
)

_TRAINER_RAM_GB = flags.DEFINE_integer(
    "trainer_ram_gb",
    32,
    "Host RAM in GB for the Trainer client job.",
)

_ROLLOUT_RAM_GB = flags.DEFINE_integer(
    "rollout_ram_gb",
    64,
    "Host RAM in GB for the Rollout worker job.",
)

_ORCHESTRATOR_RAM_GB = flags.DEFINE_integer(
    "orchestrator_ram_gb",
    32,
    "Host RAM in GB for the Orchestrator coordinator job.",
)


def _parse_platform(platform_str: str) -> Dict[str, Any]:
  """Parses platform strings like 'vlp=2x2' or 'vf=2x2x1' into kwargs for xm.JobRequirements."""
  parts = platform_str.split("=")
  if len(parts) == 2:
    return {parts[0]: parts[1]}
  return {"accelerator": platform_str}


def _create_pathways_service(
    work_unit: xm.WorkUnit,
    requirements: xm.JobRequirements,
) -> service_lib.Service:
  """Creates a Pathways service for the Trainer cluster."""
  enable_ti_vm = xm_flags.XM_ENABLE_BORG_TI_VM.value
  workers = [
      service_lib.TpuWorkerJobConfig(
          name="pathways_server_trainer",
          platform=requirements.accelerator,  # pyrefly: ignore[bad-argument-type]
          topology=requirements.topology,  # pyrefly: ignore[bad-argument-type]
          cell=_CELL.value,
          priority=_PRIORITY.value,
          enable_ti_vm=enable_ti_vm,
      ),
  ]

  borg_parent = gcl_utils.borg_token("trainer")
  resource_manager = service_lib.ResourceManagerJobConfig(
      cell=_CELL.value,
      ram=_RESOURCE_MANAGER_RAM_GB.value * xm.GiB,
      priority=_PRIORITY.value,
      enable_ti_vm=enable_ti_vm,
  )
  service_config = service_lib.ServiceConfig(
      workers=workers,
      resource_manager=resource_manager,
      borg_parent=borg_parent,
  )
  return service_lib.create_service(service_config, work_unit)


async def _launch_experiment():
  """Sets up and launches the distributed GSM8K GRPO experiment."""
  experiment_title = _EXP_TITLE.value

  async with xm_abc.create_experiment(
      experiment_title=experiment_title
  ) as experiment:
    bazel_args = xm_abc.bazel_args.tpu() + (
        "--modify_execution_info=PostMark=+requires-mem:24g,PostMarking=+requires-mem:24g",
    )

    # Package binaries
    executables = experiment.package([
        xm.bazel_binary(
            label="//third_party/py/tunix/experimental/examples/math_gsm8k_dist:run_trainer_node",
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=bazel_args,
        ),
        xm.bazel_binary(
            label="//third_party/py/tunix/experimental/examples/math_gsm8k_dist:run_rollout_node",
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=bazel_args,
        ),
        xm.bazel_binary(
            label="//third_party/py/tunix/experimental/examples/math_gsm8k_dist:run_gsm8k_dist_grpo",
            executor_spec=xm_abc.Borg.Spec(),
            bazel_args=bazel_args,
        ),
    ])
    trainer_exec, rollout_exec, orch_exec = executables[0], executables[1], executables[2]

    job_requirements_kwargs = {
        "location": _CELL.value,
        "priority": _PRIORITY.value,
    }

    # Trainer TPU platform
    trainer_tpu_reqs = xm.JobRequirements(
        **_parse_platform(_TRAINER_PLATFORM.value),
        **job_requirements_kwargs,
    )

    # Rollout TPU platform
    rollout_tpu_reqs = xm.JobRequirements(
        ram=_ROLLOUT_RAM_GB.value * xm.GiB,
        tmp_ram_fs=30 * xm.GiB,
        **_parse_platform(_ROLLOUT_PLATFORM.value),
        **job_requirements_kwargs,
    )

    async def make_jobs(work_unit: xm.WorkUnit):
      jobs: Dict[str, Any] = {}

      # 1. Create Pathways service for Trainer
      pw_service = _create_pathways_service(
          work_unit=work_unit,
          requirements=trainer_tpu_reqs,
      )
      pathways_bns = pw_service.backend_target
      jobs.update(**pw_service.jobs)

      # 2. Derive Orchestrator BNS address for discovery
      orch_bns = addressing.bns_address(
          cell=_CELL.value,
          borguser=os.environ.get("USER", "lancewang"),
          job_name="orchestrator",
          experiment_id=experiment.experiment_id,
          work_unit_id=work_unit.work_unit_id,
      )
      discovery_addr = f"{orch_bns}:20000"

      # 3. Trainer CPU client job connected to Pathways
      trainer_args = [
          f"--pathways_bns={pathways_bns}",
          f"--discovery_addrs={discovery_addr}",
          "--port=20001",
          "--worker_id=trainer-0",
          "--mesh_fsdp=4",
          "--mesh_tp=1",
          f"--model_id={_MODEL_ID.value}",
          f"--model_name={_MODEL_NAME.value}",
      ]
      trainer_env = {
          "JAX_PLATFORMS": "proxy,cpu",
          "JAX_BACKEND_TARGET": f"grpc://{pathways_bns}",
          "USE_RAIDEN_FFI": "1",
          "MODEL_DOWNLOAD_DIR": "/tmp/artifacts/qwen3_dist_gsm8k/models",
      }
      trainer_executor = xm_abc.Borg(
          logs_read_access_roles=["all"],
          requirements=xm.JobRequirements(
              ram=_TRAINER_RAM_GB.value * xm.GiB,
              tmp_ram_fs=30 * xm.GiB,
              cpu=8,
              **job_requirements_kwargs,
          ),
      )
      jobs["trainer"] = xm.Job(
          executable=trainer_exec,
          args=trainer_args,
          env_vars=trainer_env,
          executor=trainer_executor,
      )

      # 4. Rollout worker running McJAX vanilla sampler on physical TPU
      rollout_args = [
          f"--sampler={_SAMPLER.value}",
          f"--discovery_addrs={discovery_addr}",
          "--port=20002",
          "--worker_id=rollout-0",
          "--mesh_fsdp=1",
          "--mesh_tp=4",
          f"--model_id={_MODEL_ID.value}",
          f"--model_name={_MODEL_NAME.value}",
          f"--weight_sync_mode={_WEIGHT_SYNC_BACKEND.value}",
      ]
      rollout_env = {
          "MODEL_DOWNLOAD_DIR": "/tmp/artifacts/qwen3_dist_gsm8k/models",
      }
      rollout_executor = xm_abc.Borg(
          logs_read_access_roles=["all"],
          requirements=rollout_tpu_reqs,
      )
      jobs["rollout"] = xm.Job(
          executable=rollout_exec,
          args=rollout_args,
          env_vars=rollout_env,
          executor=rollout_executor,
      )

      # 5. Orchestrator coordinating RL program
      orch_args = [
          "--discovery_id=orch",
          "--discovery_port=20000",
          f"--model_id={_MODEL_ID.value}",
          f"--model_name={_MODEL_NAME.value}",
          f"--batch_size={_BATCH_SIZE.value}",
          f"--mini_batch_size={_MINI_BATCH_SIZE.value}",
          f"--num_generations={_NUM_GENERATIONS.value}",
          f"--max_steps={_MAX_STEPS.value}",
          f"--beta={_BETA.value}",
          f"--weight_sync_backend={_WEIGHT_SYNC_BACKEND.value}",
          "--stop_workers_on_exit",
      ]
      orch_env = {
          "MODEL_DOWNLOAD_DIR": "/tmp/artifacts/qwen3_dist_gsm8k/models",
      }
      orch_executor = xm_abc.Borg(
          logs_read_access_roles=["all"],
          requirements=xm.JobRequirements(
              ram=_ORCHESTRATOR_RAM_GB.value * xm.GiB,
              tmp_ram_fs=30 * xm.GiB,
              cpu=8,
              **job_requirements_kwargs,
          ),
      )
      jobs["orchestrator"] = xm.Job(
          executable=orch_exec,
          args=orch_args,
          env_vars=orch_env,
          executor=orch_executor,
      )

      work_unit.add(xm.JobGroup(**jobs))

    experiment.add(make_jobs)


import asyncio

def main(_):
  asyncio.run(_launch_experiment())


if __name__ == "__main__":
  app.run(main)
