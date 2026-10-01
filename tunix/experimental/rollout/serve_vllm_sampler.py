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
"""HTTP Server script to serve VllmSampler for agentic benchmarking and OpenAI-compatible API workloads.

Exposes OpenAI-compatible /v1/chat/completions and /v1/completions endpoints
powered directly by VllmSampler in tpu-inference.
"""

import argparse
import json
import logging
import os
import time

os.environ["VLLM_USE_V1"] = "0"
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from vllm.engine.arg_utils import AsyncEngineArgs

from tunix.experimental.rollout.vllm_sampler_v2 import RLVllmSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RLVllmSampler OpenAI-Compatible HTTP Server")
sampler_instance: Optional[RLVllmSampler] = None

# ==============================================================================
# Pydantic Request Models
# ==============================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen2.5-1.5B"
    messages: List[Union[ChatMessage, Dict[str, Any]]]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    route_key: Optional[str] = None
    ignore_eos: Optional[bool] = False


class CompletionRequest(BaseModel):
    model: Optional[str] = "Qwen/Qwen2.5-1.5B"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    route_key: Optional[str] = None
    ignore_eos: Optional[bool] = False


# ==============================================================================
# Helper Functions
# ==============================================================================


def format_messages_to_prompt(
        messages: List[Union[ChatMessage, Dict[str, Any]]]) -> str:
    """Formats a list of chat messages into a single prompt string."""
    formatted_parts = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            role, content = msg.role, msg.content
        elif isinstance(msg, dict):
            role, content = msg.get("role", "user"), msg.get("content", "")
        else:
            role, content = getattr(msg, "role",
                                    "user"), getattr(msg, "content", "")
        formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    formatted_parts.append("<|im_start|>assistant\n")
    return "\n".join(formatted_parts)


# ==============================================================================
# API Endpoints
# ==============================================================================


@app.get("/health")
@app.get("/v1/models")
async def health_check():
    """Health check endpoint."""
    if sampler_instance is None or not sampler_instance._is_running:
        raise HTTPException(status_code=503,
                            detail="RLVllmSampler is not running.")
    model_name = sampler_instance.engine_args.model
    return {
        "status": "ok",
        "model": model_name,
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model"
        }],
    }


async def stream_chat_response(
    prompt_text: str,
    req_id: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    route_key: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Streams chat completion deltas as Server-Sent Events (SSE)."""
    assert sampler_instance is not None and sampler_instance._engine is not None

    vllm_params = sampler_instance._build_vllm_params(
        SimpleNamespace(sampling_params=SimpleNamespace(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            return_logprobs=False,
        )),
        {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "return_logprobs": False
        },
    )

    created_time = int(time.time())
    task_gen = sampler_instance._engine.generate(prompt_text,
                                                 vllm_params,
                                                 request_id=req_id)
    previous_text = ""

    try:
        async for step_output in task_gen:
            if step_output and step_output.outputs:
                choice = step_output.outputs[0]
                current_text = choice.text
                delta_text = current_text[len(previous_text):]
                previous_text = current_text

                chunk = {
                    "id":
                    req_id,
                    "object":
                    "chat.completion.chunk",
                    "created":
                    created_time,
                    "model":
                    model_name,
                    "choices": [{
                        "index":
                        0,
                        "delta": {
                            "role": "assistant",
                            "content": delta_text
                        },
                        "finish_reason":
                        getattr(choice, "finish_reason", None),
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        completion_tokens = len(
            choice.token_ids) if ('choice' in locals() and choice
                                  and hasattr(choice, "token_ids")) else 0
        prompt_tokens = len(prompt_text) // 4
        usage_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception("Error in streaming response for req_id=%s", req_id)
        err_chunk = {
            "error": {
                "message": str(e),
                "type": type(e).__name__,
            }
        }
        yield f"data: {json.dumps(err_chunk)}\n\n"
        yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, req: Request):
    """OpenAI-compatible chat completion endpoint."""
    if sampler_instance is None or not sampler_instance._is_running:
        raise HTTPException(status_code=503,
                            detail="RLVllmSampler server is not running.")

    prompt_text = format_messages_to_prompt(request.messages)
    req_id = f"chatcmpl-{time.time_ns()}"
    route_key = request.route_key or req.headers.get("x-route-key")

    if request.stream:
        return StreamingResponse(
            stream_chat_response(
                prompt_text=prompt_text,
                req_id=req_id,
                model_name=request.model or sampler_instance.engine_args.model,
                max_tokens=request.max_tokens or 128,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.95,
                route_key=route_key,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming mode
    req_obj = SimpleNamespace(
        prompt=prompt_text,
        request_id=req_id,
        route_key=route_key,
        sampling_params=SimpleNamespace(
            max_tokens=request.max_tokens or 128,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            return_logprobs=False,
        ),
    )

    results = await sampler_instance.sample([req_obj])
    if not results or getattr(results[0], "error", None) is not None:
        err_msg = str(results[0].error) if results and hasattr(
            results[0], "error") else "Generation failed"
        raise HTTPException(status_code=500, detail=err_msg)

    res = results[0]
    response_payload = {
        "id":
        req_id,
        "object":
        "chat.completion",
        "created":
        int(time.time()),
        "model":
        request.model or sampler_instance.engine_args.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": res.text
            },
            "finish_reason": getattr(res, "finish_reason", "stop"),
        }],
        "usage": {
            "prompt_tokens":
            0,
            "completion_tokens":
            len(res.token_ids)
            if getattr(res, "token_ids", None) is not None else 0,
            "total_tokens":
            len(res.token_ids)
            if getattr(res, "token_ids", None) is not None else 0,
        },
    }
    return JSONResponse(content=response_payload)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, req: Request):
    """OpenAI-compatible raw prompt completion endpoint."""
    if sampler_instance is None or not sampler_instance._is_running:
        raise HTTPException(status_code=503,
                            detail="RLVllmSampler server is not running.")

    prompt_text = request.prompt if isinstance(request.prompt,
                                               str) else request.prompt[0]
    req_id = f"cmpl-{time.time_ns()}"
    route_key = request.route_key or req.headers.get("x-route-key")

    if request.stream:
        return StreamingResponse(
            stream_chat_response(
                prompt_text=prompt_text,
                req_id=req_id,
                model_name=request.model or sampler_instance.engine_args.model,
                max_tokens=request.max_tokens or 128,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 0.95,
                route_key=route_key,
            ),
            media_type="text/event-stream",
        )

    req_obj = SimpleNamespace(
        prompt=prompt_text,
        request_id=req_id,
        route_key=route_key,
        sampling_params=SimpleNamespace(
            max_tokens=request.max_tokens or 128,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.95,
            return_logprobs=False,
        ),
    )

    results = await sampler_instance.sample([req_obj])
    if not results or getattr(results[0], "error", None) is not None:
        err_msg = str(results[0].error) if results and hasattr(
            results[0], "error") else "Generation failed"
        raise HTTPException(status_code=500, detail=err_msg)

    res = results[0]
    response_payload = {
        "id":
        req_id,
        "object":
        "text_completion",
        "created":
        int(time.time()),
        "model":
        request.model or sampler_instance.engine_args.model,
        "choices": [{
            "index": 0,
            "text": res.text,
            "finish_reason": getattr(res, "finish_reason", "stop"),
        }],
    }
    return JSONResponse(content=response_payload)


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch RLVllmSampler HTTP Server for Agentic Benchmarking."
    )
    parser.add_argument("--host",
                        type=str,
                        default="0.0.0.0",
                        help="Host IP to bind HTTP server.")
    parser.add_argument("--port",
                        type=int,
                        default=8000,
                        help="Port to bind HTTP server.")
    parser.add_argument("--model_path",
                        type=str,
                        default="Qwen/Qwen2.5-1.5B",
                        help="Model path or HF ID.")
    parser.add_argument("--tensor_parallel_size",
                        type=int,
                        default=1,
                        help="Tensor parallelism size.")
    parser.add_argument("--data_parallel_size",
                        type=int,
                        default=1,
                        help="Data parallelism size.")
    parser.add_argument("--max_num_seqs",
                        type=int,
                        default=256,
                        help="Max batched sequences in vLLM.")
    parser.add_argument("--max_num_batched_tokens",
                        type=int,
                        default=8192,
                        help="Max batched tokens.")
    parser.add_argument("--hbm_utilization",
                        type=float,
                        default=0.80,
                        help="Target HBM utilization fraction.")
    parser.add_argument("--enable_prefix_caching",
                        action="store_true",
                        default=True,
                        help="Enable vLLM prefix cache.")
    parser.add_argument("--weight_dtype",
                        type=str,
                        default="bfloat16",
                        help="Model weight dtype.")

    args = parser.parse_args()

    global sampler_instance
    engine_args = AsyncEngineArgs(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.hbm_utilization,
        enable_prefix_caching=args.enable_prefix_caching,
        dtype=args.weight_dtype,
    )

    logger.info("Initializing RLVllmSampler for server deployment...")
    sampler_instance = RLVllmSampler(engine_args=engine_args)

    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting RLVllmSampler engine...")
        await sampler_instance.start()
        logger.info("RLVllmSampler engine online and ready to serve requests.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Stopping RLVllmSampler engine...")
        await sampler_instance.stop()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
