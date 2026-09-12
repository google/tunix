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
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Asynchronous vLLM Sampler engine implementation for tpu-inference.

This module provides a standalone vLLM serving and rollout sampler (`RLVllmSampler`)
tailored for Reinforcement Learning (RL) workloads on TPUs.
Satisfies the open-source Tunix `Sampler` Protocol defined in:
https://github.com/google/tunix/blob/main/tunix/experimental/sampler/sampler.py
without importing or depending on Tunix.
"""

import asyncio
import logging
import os
import time

os.environ["VLLM_USE_V1"] = "0"
from types import SimpleNamespace
from typing import Any

import numpy as np
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams as VllmSamplingParams

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Helper to extract an attribute or dict key seamlessly from any duck-typed request.

    Falls back to `default` both when `key` is absent and when it's present
    but explicitly `None` -- request dataclasses across callers (e.g. tunix's
    rollout requests) commonly default optional sampling fields to `None`
    rather than omitting them, and vLLM's `SamplingParams` rejects `None` for
    fields like `top_k` that it compares numerically.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return default if val is None else val


class RLVllmSampler:
    """Asynchronous vLLM sampler for RL inside `tpu-inference`.

  Satisfies the open-source Tunix `Sampler` Protocol without importing Tunix.
  Handles:

    - Direct binding to vLLM's `AsyncLLMEngine`.
    - Dynamic attribute extraction (`_get_val`) for arbitrary request inputs.
    - Non-blocking asynchronous batch generation (`sample`).
    - Sticky routing header support (`route_key`) for multi-turn prefix cache hits.
    - Native `TPUWorker` 3-phase weight synchronization (`pre_weight_sync`, `weight_sync`, `post_weight_sync`).
  """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine_args = engine_args
        self._engine: Any | None = None
        self._is_running = False
        self._is_paused = False
        self._cache_valid = True
        self._mesh: Any | None = None
        self._transfer_statuses: dict[str, str] = {}
        self._policy_version = 0

    def _get_tpu_workers(self) -> list[Any]:
        """Retrieves active TPUWorker instances from underlying model executor."""
        if self._engine is None:
            return []
        llm_engine = getattr(self._engine, "engine", None)
        model_executor = getattr(llm_engine, "model_executor",
                                 None) if llm_engine else None
        if model_executor is None:
            return []
        workers = getattr(model_executor, "workers", None)
        if workers:
            return list(workers)
        driver_worker = getattr(model_executor, "driver_worker", None)
        return [driver_worker] if driver_worker else []

    async def start(self, **kwargs: Any) -> None:
        """Initializes the vLLM engine and execution environment."""
        if self._is_running:
            logger.warning("RLVllmSampler is already running.")
            return

        logger.info("Initializing RLVllmSampler with model: %s",
                    self.engine_args.model)

        self._engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        self._is_running = True

        logger.info("RLVllmSampler started successfully.")

    async def stop(self, **kwargs: Any) -> None:
        """Stops the sampler and releases resources."""
        if not self._is_running:
            return
        logger.info("Stopping RLVllmSampler...")
        self._is_paused = True
        self._engine = None
        self._is_running = False
        logger.info("RLVllmSampler stopped.")

    async def pause(self, **kwargs: Any) -> None:
        """Pauses request intake and drains active batch iterations during weight updates."""
        if self._is_paused:
            return
        self._is_paused = True
        # Note: vLLM's pause_background_loop stops the scheduler from taking new requests from the queue.
        # Ongoing requests in the batch will be completed or drained depending on internal vLLM state.
        if self._engine and hasattr(self._engine, "pause_background_loop"):
            logger.info(
                "Pausing RLVllmSampler inference intake for weight sync...")
            await self._engine.pause_background_loop()
        await asyncio.sleep(0.01)

    async def resume(self, **kwargs: Any) -> None:
        """Resumes inference processing after weight sync completion."""
        if not self._is_paused:
            return
        if self._engine and hasattr(self._engine, "resume_background_loop"):
            logger.info("Resuming RLVllmSampler inference serving...")
            await self._engine.resume_background_loop()
        self._is_paused = False

    async def get_mesh(self, **kwargs: Any) -> Any | None:
        """Returns the JAX device mesh."""
        return self._mesh

    # ----------------------------------------------------------------------------
    # Helper Methods for Sample Stream Processing
    # ----------------------------------------------------------------------------

    def _build_vllm_params(
        self,
        req: Any,
        kwargs: dict[str, Any],
    ) -> VllmSamplingParams:
        """Builds a vLLM SamplingParams object from any duck-typed request."""
        sparams = _get_val(req, "sampling_params")
        # `kwargs` goes through `_get_val` too: callers pass unset fields as
        # explicit `None`, so `.get(key, default)` returns `None` rather than
        # the default, defeating it before `_get_val` can coalesce.
        return VllmSamplingParams(
            temperature=_get_val(sparams, "temperature",
                                 _get_val(kwargs, "temperature", 0.7)),
            top_p=_get_val(sparams, "top_p", _get_val(kwargs, "top_p", 0.95)),
            top_k=_get_val(sparams, "top_k", _get_val(kwargs, "top_k", -1)),
            max_tokens=_get_val(sparams, "max_tokens",
                                _get_val(kwargs, "max_tokens", 128)),
            stop=_get_val(sparams, "stop_sequences")
            or _get_val(sparams, "stop") or kwargs.get("stop"),
            logprobs=1
            if _get_val(sparams, "return_logprobs",
                        kwargs.get("return_logprobs", False)) else None,
        )

    async def _process_request_output(
        self,
        req_id: str,
        task: Any,
    ) -> SimpleNamespace:
        """Consumes an AsyncLLMEngine output stream and formats output result."""
        try:
            final_output = None
            async for step_output in task:
                final_output = step_output

            if final_output and final_output.outputs:
                output_choice = final_output.outputs[0]
                text = output_choice.text
                token_ids_arr = np.array(output_choice.token_ids,
                                         dtype=np.int32)
                # On the RequestOutput, not the CompletionOutput. Tunix feeds
                # this into np.asarray(..., dtype=np.int32), so omitting it
                # surfaces as "int() argument must be ... not 'NoneType'".
                prompt_token_ids_arr = np.array(
                    getattr(final_output, "prompt_token_ids", None) or [],
                    dtype=np.int32)
                cum_logprob = float(
                    getattr(output_choice, "cumulative_logprob", 0.0) or 0.0)

                logprobs_arr = None
                if getattr(output_choice, "logprobs", None):
                    logprob_vals = [
                        lp_dict[next(iter(lp_dict.keys()))].logprob
                        if lp_dict else 0.0
                        for lp_dict in output_choice.logprobs
                    ]
                    logprobs_arr = np.array(logprob_vals, dtype=np.float32)

                routed_experts = getattr(output_choice, "routed_experts", None)
                if routed_experts is not None and not isinstance(
                        routed_experts, np.ndarray):
                    routed_experts = np.array(routed_experts)

                return SimpleNamespace(
                    request_id=req_id,
                    text=text,
                    token_ids=token_ids_arr,
                    prompt_token_ids=prompt_token_ids_arr,
                    logprobs=logprobs_arr,
                    cumulative_logprob=cum_logprob,
                    routed_experts=routed_experts,
                    finish_reason=getattr(output_choice, "finish_reason",
                                          "stop") or "stop",
                    error=None,
                )

            err_obj = SimpleNamespace(
                error_type="EmptyOutput",
                message="No output generated by vLLM",
                retryable=False,
            )
            return SimpleNamespace(
                request_id=req_id,
                text="",
                token_ids=np.zeros(0, dtype=np.int32),
                prompt_token_ids=np.zeros(0, dtype=np.int32),
                logprobs=None,
                cumulative_logprob=0.0,
                routed_experts=None,
                finish_reason="stop",
                error=err_obj,
            )
        except Exception as e:
            logger.exception("Error generating sampling result for req_id=%s",
                             req_id)
            err_obj = SimpleNamespace(
                error_type=type(e).__name__,
                message=str(e),
                retryable=True,
            )
            return SimpleNamespace(
                request_id=req_id,
                text="",
                token_ids=np.zeros(0, dtype=np.int32),
                prompt_token_ids=np.zeros(0, dtype=np.int32),
                logprobs=None,
                cumulative_logprob=0.0,
                routed_experts=None,
                finish_reason="stop",
                error=err_obj,
            )

    # ----------------------------------------------------------------------------
    # Dynamic Batch Sampling
    # ----------------------------------------------------------------------------

    async def sample(
        self,
        sampling_requests: Any,
        **kwargs: Any,
    ) -> Any:
        """Generates completions for a batch of sampling requests or raw prompts.

    Accepts raw string lists, dictionaries, or duck-typed objects from callers
    (such as `tunix.experimental.common.datatypes.SamplingRequest`).
    """
        while self._is_paused:
            await asyncio.sleep(0.05)

        if not self._is_running or self._engine is None:
            await self.start()

        raw_input_mode = False
        if isinstance(sampling_requests, (str, list)) and (
                isinstance(sampling_requests, str) or not sampling_requests
                or isinstance(sampling_requests[0], str) or
            (isinstance(sampling_requests[0], list)
             and not _get_val(sampling_requests[0], "prompt"))):
            raw_input_mode = True
            items = sampling_requests if isinstance(
                sampling_requests, list) else [sampling_requests]
            req_list = [{
                "prompt": item,
                "request_id": f"req_{idx}_{time.time_ns()}",
            } for idx, item in enumerate(items)]
        elif _get_val(sampling_requests, "prompt") is not None:
            req_list = [sampling_requests]
        else:
            req_list = list(sampling_requests)

        pending_tasks = []
        for idx, req in enumerate(req_list):
            vllm_params = self._build_vllm_params(req, kwargs)
            req_id = _get_val(req,
                              "request_id") or f"req_{time.time_ns()}_{idx}"
            prompt_val = _get_val(req, "prompt")
            prompt_text = prompt_val if isinstance(prompt_val,
                                                   str) else str(prompt_val)

            task_gen = self._engine.generate(prompt_text,
                                             vllm_params,
                                             request_id=req_id)
            pending_tasks.append((req_id, task_gen))

        results = await asyncio.gather(*[
            self._process_request_output(req_id, task_gen)
            for req_id, task_gen in pending_tasks
        ])

        if raw_input_mode:
            return [r.text for r in results]
        return list(results)

    # ----------------------------------------------------------------------------
    # Cache Management
    # ----------------------------------------------------------------------------

    async def _clear_prefix_cache(self) -> None:
        """Purges prefix caching blocks from HBM."""
        logger.info("Clearing vLLM RadixAttention prefix cache...")
        self._cache_valid = False
        if self._engine and hasattr(self._engine, "reset_prefix_cache"):
            await self._engine.reset_prefix_cache()

    # ----------------------------------------------------------------------------
    # Weight Synchronization (Integrated with TPUWorker)
    # ----------------------------------------------------------------------------

    async def get_weight_sync_metadata(self, **kwargs: Any) -> dict[str, Any]:
        """Returns PyTree of weight sharding rules, dtype, and shape across devices."""
        return self.get_weight_metadata()

    def get_weight_metadata(self) -> dict[str, Any]:
        """Synchronous alias for get_weight_sync_metadata."""
        worker_ips = [
            getattr(w, "ip_address", getattr(w, "ip", "127.0.0.1"))
            for w in self._get_tpu_workers()
        ]
        tp_size = getattr(self.engine_args, "tensor_parallel_size", 1)
        dp_size = getattr(self.engine_args, "data_parallel_size", 1)
        ep_size = getattr(self.engine_args, "expert_parallel_size", 1)
        return {
            "sharding": {
                "tensor_parallel_size": tp_size,
                "data_parallel_size": dp_size,
                "expert_parallel_size": ep_size,
            },
            "dtype": str(getattr(self.engine_args, "dtype", "bfloat16")),
            "model_path": getattr(self.engine_args, "model", ""),
            "policy_version": self._policy_version,
            "cache_valid": self._cache_valid,
            "tpu_worker_ips": worker_ips,
        }

    async def _call_worker_method(self, method_name: str, *args: Any,
                                  **kwargs: Any) -> list[Any]:
        """Dispatches a method call across TPU workers via collective_rpc.

        `AsyncLLMEngine` (an alias of `vllm.v1.engine.async_llm.AsyncLLM`)
        always exposes an async `collective_rpc`.
        """
        if self._engine is None:
            return []
        return await self._engine.collective_rpc(method_name,
                                                 args=args,
                                                 kwargs=kwargs)

    async def get_weights_state(self) -> list[Any]:
        """Returns the PyTree of weights or state from active TPU workers."""
        return await self._call_worker_method("get_weights_state")

    async def bind_raiden_sync(self,
                               worker_index: int = 0,
                               parallelism: int = 4,
                               job_name: str = "rollout") -> None:
        """Binds Raiden to each TPU worker's live weights, in-process.

        `get_weights_state()` cannot back a rollout-side weight sync: the
        live `nnx.State` it returns has to be dispatched back across
        `collective_rpc` into this (parent) process to be useful, but vLLM's
        RPC transport can't serialize `nnx.State`/live TPU arrays, and even a
        pickle-based fallback would bind Raiden to a disconnected host copy
        rather than the device buffers actually being served. Binding must
        happen in the worker subprocess instead -- see
        `tpu_worker.TPUWorker.bind_raiden_sync`.
        """
        await self._call_worker_method("bind_raiden_sync", worker_index,
                                       parallelism, job_name)

    async def refresh_model_state_leaves(self) -> None:
        """Re-points each worker's dispatch view after an h2d weight update."""
        await self._call_worker_method("refresh_model_state_leaves")

    async def get_raiden_metadata(self) -> list[dict]:
        """Wire-safe registration metadata for each worker's current Raiden binding."""
        return await self._call_worker_method("get_raiden_metadata")

    async def raiden_h2d(self) -> list[dict]:
        """Blocks each worker until its just-landed transfer is visible on-device.

        Returns each worker's checksums dict (empty unless VERIFY_WEIGHTS=true).
        """
        return await self._call_worker_method("raiden_h2d")

    async def raiden_metrics(self) -> list[dict]:
        return await self._call_worker_method("raiden_metrics")

    async def pre_weight_sync(
        self,
        sync_request: Any = None,
        free_kv_cache: bool = True,
        **kwargs: Any,
    ) -> None:
        """Phase 1: Pauses intake, clears prefix cache, and drops KV cache via delete_kv_cache()."""
        self._policy_version = _get_val(sync_request, "policy_version",
                                        self._policy_version)
        logger.info(
            "Executing pre_weight_sync (policy_version=%d, free_kv_cache=%s)",
            self._policy_version, free_kv_cache)

        if sync_request is not None:
            rid = _get_val(sync_request, "req_id")
            if rid:
                self._transfer_statuses[str(rid)] = "IN_PROGRESS"

        await self.pause()
        await self._clear_prefix_cache()

        # `AsyncLLMEngine.start_weight_update()` takes no arguments, so it
        # can't forward `free_kv_cache` to the worker -- go through
        # collective_rpc directly instead of the engine-level wrapper.
        await self._call_worker_method("start_weight_update",
                                       free_kv_cache=free_kv_cache)

    async def weight_sync(
        self,
        sync_request: Any = None,
        **kwargs: Any,
    ) -> None:
        """Phase 2: Coordinates weight synchronization barrier across TPU workers."""
        logger.info("Executing weight_sync update on TPU workers...")
        u_info = _get_val(sync_request, "extra_config") or {}
        await self._call_worker_method("update_weights", u_info)
        await asyncio.sleep(0.01)

    async def post_weight_sync(
        self,
        sync_request: Any = None,
        **kwargs: Any,
    ) -> None:
        """Phase 3: Restores KV cache via reinitialize_kv_cache() and resumes serving."""
        rid = None
        if sync_request is not None:
            rid = _get_val(sync_request, "req_id")
        logger.info("Executing post_weight_sync (req_id=%s)...", rid)

        await self._call_worker_method("finish_weight_update")

        self._cache_valid = True
        if rid:
            self._transfer_statuses[str(rid)] = "SUCCESS"
        await self.resume()

    async def get_transfer_status(self, req_id: Any, **kwargs: Any) -> str:
        """Returns status of weight transfer or KV-cache migration request."""
        return self._transfer_statuses.get(str(req_id), "MISSING")

    def get_transfer_controller_info(self) -> tuple[str, str]:
        """Returns (dst_controller_id, dst_controller_ip) for state transfer coordination."""
        return self._dst_controller_id, self._dst_controller_ip

    async def migrate_kv_cache(
        self,
        route_key: str,
        source_server_id: str,
        target_server_id: str,
        token_ids: list[int],
    ) -> bool:
        """Triggers Raiden P2P KV-cache block transfer from source to target sampler slice."""
        raise NotImplementedError(
            "P2P KV-cache migration is pending live TPU Raiden controller integration."
        )

    async def get_load_info(self, **kwargs: Any) -> Any:
        """Returns current load info (num_requests_waiting, num_requests_running, kv_cache_usage_perc)."""
        num_waiting = 0
        num_running = 0
        kv_cache_usage_perc = 0.0

        if self._engine:
            engine_obj = getattr(self._engine, "engine", self._engine)

            # If running under DPScheduler, get aggregated stats across DP ranks
            if hasattr(engine_obj, "make_stats"):
                stats = engine_obj.make_stats()
                if stats:
                    num_waiting = getattr(stats, "num_waiting_reqs", 0)
                    num_running = getattr(stats, "num_running_reqs", 0)
                    kv_usage = getattr(stats, "kv_cache_usage", 0.0)
                    kv_cache_usage_perc = (
                        kv_usage *
                        100.0) if kv_usage <= 1.0 else float(kv_usage)
            # Otherwise, read directly from standard vLLM scheduler(s)
            elif getattr(engine_obj, "scheduler", None):
                schedulers = engine_obj.scheduler
                sched_list = schedulers if isinstance(schedulers,
                                                      list) else [schedulers]
                for sched in sched_list:
                    num_waiting += len(getattr(sched, "waiting", []))
                    num_running += len(getattr(sched, "running", []))

                    block_manager = getattr(sched, "block_manager", None)
                    if block_manager:
                        get_free = getattr(block_manager,
                                           "get_num_free_gpu_blocks", None)
                        get_total = getattr(block_manager,
                                            "get_num_total_gpu_blocks", None)
                        if get_free and get_total:
                            total_blocks = get_total()
                            free_blocks = get_free()
                            if total_blocks > 0:
                                used_ratio = 1.0 - (free_blocks / total_blocks)
                                kv_cache_usage_perc = max(
                                    kv_cache_usage_perc, used_ratio * 100.0)

        return SimpleNamespace(
            num_requests_waiting=num_waiting,
            num_requests_running=num_running,
            kv_cache_usage_perc=kv_cache_usage_perc,
        )
