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
"""OpenAI-compatible TPU serving gateway for full-paper Qwen3.5-9B R2Orch."""

from __future__ import annotations

import contextlib
import gc
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import math
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any
import uuid

import jax
import numpy as np
from tunix.dist_rl.swarm import trace as swarm_trace
from tunix.generate import mappings as mappings_lib
from tunix.models.qwen3p5 import mapping_vllm_jax

_FC_TLS = threading.local()
_VW_TLS = threading.local()
_TORCHAX_LOCK = threading.RLock()


class ThreadLocalVllmContexts:
  """Thread-local replacements for vLLM & tpu-inference global contexts."""

  vwc_cls: Any = None
  orig_env_enter: Any = None
  orig_env_exit: Any = None

  @staticmethod
  def get_fc() -> Any:
    ctx = getattr(_FC_TLS, "ctx", None)
    if ctx is None:
      raise AssertionError("Forward context is not set.")
    return ctx

  @staticmethod
  def is_fc() -> bool:
    return getattr(_FC_TLS, "ctx", None) is not None

  @staticmethod
  @contextlib.contextmanager
  def override_fc(forward_context: Any):
    prev = getattr(_FC_TLS, "ctx", None)
    _FC_TLS.ctx = forward_context
    try:
      yield
    finally:
      _FC_TLS.ctx = prev

  @staticmethod
  def get_vwc() -> Any:
    ctx = getattr(_VW_TLS, "ctx", None)
    if ctx is None:
      raise AssertionError("VllmModelWrapperContext is not set.")
    return ctx

  @staticmethod
  @contextlib.contextmanager
  def set_vwc(
      *,
      kv_caches: Any,
      mesh: Any,
      layer_name_to_kvcache_index: Any = None,
      vllm_config: Any = None,
      shared_attn_metadata: Any = None,
  ):
    prev = getattr(_VW_TLS, "ctx", None)
    _VW_TLS.ctx = ThreadLocalVllmContexts.vwc_cls(
        kv_caches=kv_caches,
        mesh=mesh,
        layer_name_to_kvcache_index=layer_name_to_kvcache_index,
        vllm_config=vllm_config,
        shared_attn_metadata=shared_attn_metadata,
    )
    try:
      yield
    finally:
      _VW_TLS.ctx = prev

  @staticmethod
  def env_enter(env: Any) -> Any:
    _TORCHAX_LOCK.acquire()
    try:
      return ThreadLocalVllmContexts.orig_env_enter(env)
    except BaseException:
      _TORCHAX_LOCK.release()
      raise

  @staticmethod
  def env_exit(env: Any, *exc: Any) -> Any:
    try:
      return ThreadLocalVllmContexts.orig_env_exit(env, *exc)
    finally:
      _TORCHAX_LOCK.release()


def install_thread_local_vllm_contexts() -> None:
  """Patches vLLM/tpu-inference/torchax tracing contexts for thread safety."""
  import torchax.tensor as tx_tensor  # pylint: disable=g-import-not-at-top
  from tpu_inference.models.vllm import vllm_model_wrapper_context as vwc  # pylint: disable=g-import-not-at-top
  import vllm.forward_context as vfc  # pylint: disable=g-import-not-at-top

  if getattr(vfc, "_tunix_tls_patched", False):
    return
  ThreadLocalVllmContexts.vwc_cls = vwc.VllmModelWrapperContext
  ThreadLocalVllmContexts.orig_env_enter = tx_tensor.Environment.__enter__
  ThreadLocalVllmContexts.orig_env_exit = tx_tensor.Environment.__exit__
  tx_tensor.Environment.__enter__ = ThreadLocalVllmContexts.env_enter
  tx_tensor.Environment.__exit__ = ThreadLocalVllmContexts.env_exit
  pairs = (
      (vfc.get_forward_context, ThreadLocalVllmContexts.get_fc),
      (vfc.is_forward_context_available, ThreadLocalVllmContexts.is_fc),
      (vfc.override_forward_context, ThreadLocalVllmContexts.override_fc),
      (vwc.get_vllm_model_wrapper_context, ThreadLocalVllmContexts.get_vwc),
      (vwc.set_vllm_model_wrapper_context, ThreadLocalVllmContexts.set_vwc),
  )
  by_id = {id(old): (old, new) for old, new in pairs}
  for mod in list(sys.modules.values()):
    ns = getattr(mod, "__dict__", None)
    if isinstance(ns, dict):
      for key, val in list(ns.items()):
        hit = by_id.get(id(val))
        if hit is not None and val is hit[0]:
          ns[key] = hit[1]
  vfc._tunix_tls_patched = True


def build_qwen3p5_vllm_sampler(
    model_path: str | Path,
    tokenizer: Any,
    *,
    mesh: jax.sharding.Mesh,
    max_model_len: int = 32768,
    hbm_utilization: float = 0.56,
    free_kv_cache_during_weight_sync: bool = False,
    return_logprobs: bool = True,
    eos_tokens: list[int] | None = None,
) -> Any:
  """Creates a thread-safe server-mode `VllmSampler` on `mesh` (`tp=1`)."""
  from tunix.generate import vllm_sampler as vllm_lib  # pylint: disable=g-import-not-at-top

  install_thread_local_vllm_contexts()

  dp_size = int(mesh.devices.size)
  mapping_cfg = mappings_lib.MappingConfig(**mapping_vllm_jax.VLLM_JAX_MAPPING)
  engine_kwargs: dict[str, Any] = {
      "model": str(model_path),
      "max_model_len": int(max_model_len),
      "generation_config": "vllm",
      "enable_prefix_caching": False,
      "hf_overrides": {"architectures": ["Qwen3_5ForCausalLM"]},
      "max_num_seqs": 64,
      "max_num_batched_tokens": 8192,
      "trust_remote_code": True,
  }
  vllm_cfg = vllm_lib.VllmConfig(
      server_mode=True,
      mapping_config=mapping_cfg,
      return_logprobs=bool(return_logprobs),
      eos_tokens=eos_tokens,
      init_with_random_weights=True,
      delete_dst_buffers=False,
      hbm_utilization=float(hbm_utilization),
      mesh=mesh,
      tensor_parallel_size=1,
      data_parallel_size=dp_size,
      free_kv_cache_during_weight_sync=free_kv_cache_during_weight_sync,
      engine_kwargs=engine_kwargs,
  )
  return vllm_lib.VllmSampler(tokenizer=tokenizer, config=vllm_cfg)


def qwen3p5_eos_ids(hf_tokenizer: Any) -> list[int]:
  ids: list[int] = []
  for tok in ("<|im_end|>", "<|endoftext|>"):
    tid = hf_tokenizer.convert_tokens_to_ids(tok)
    if isinstance(tid, int) and tid >= 0 and tid not in ids:
      ids.append(tid)
  if hf_tokenizer.eos_token_id is not None and int(hf_tokenizer.eos_token_id) not in ids:
    ids.append(int(hf_tokenizer.eos_token_id))
  return ids


class R2OrchTpuGatewayState:
  """Holds the frozen Worker vLLM, trainable Director vLLM, and LearnerHub."""

  def __init__(
      self,
      *,
      hf_tokenizer: Any,
      director_jinja: str,
      hub: Any,
      worker_sampler: Any,
      director_sampler: Any,
      max_model_len: int = 32768,
      base_name: str = "r2orch-base",
      proposer_name: str = "r2orch-proposer",
      director_name: str = "r2orch-director",
  ) -> None:
    self.hf_tokenizer = hf_tokenizer
    self.director_jinja = director_jinja
    self.hub = hub
    self.worker_sampler = worker_sampler
    self.director_sampler = director_sampler
    self.max_model_len = int(max_model_len)
    self.eos_ids = set(qwen3p5_eos_ids(hf_tokenizer))
    self.base_name = base_name
    self.proposer_name = proposer_name
    self.director_name = director_name
    self._lock = threading.RLock()
    self.loaded_models: set[str] = {base_name, proposer_name, director_name}
    self.call_counter = 0
    self.recent_traces: list[swarm_trace.CallTrace] = []

  def render_and_encode(
      self,
      messages: list[dict[str, Any]],
      *,
      add_generation_prompt: bool = True,
      chat_template_kwargs: dict[str, Any] | None = None,
  ) -> list[int]:
    kwargs = dict(chat_template_kwargs or {})
    rendered = self.hf_tokenizer.apply_chat_template(
        messages,
        chat_template=self.director_jinja,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return [
        int(t)
        for t in self.hf_tokenizer.encode(rendered, add_special_tokens=False)
    ]

  def _sync_director_adapter(self, lora_name: str, lora_path: str) -> None:
    self.hub.snapshot_adapter(lora_name, "solver", lora_path=lora_path)
    sync_dtype = getattr(self.director_sampler, "weight_sync_dtype", None)
    merged = (
        self.hub.merged_state_for_vllm(
            "solver", served_name=lora_name, dtype=sync_dtype
        )
        if sync_dtype is not None
        else self.hub.merged_state_for_vllm("solver", served_name=lora_name)
    )
    drv = getattr(self.director_sampler, "_driver", None)
    if drv is not None and hasattr(drv, "pause"):
      drv.pause(mode="wait", timeout_s=300.0)
    try:
      self.director_sampler.update_params(merged)
    finally:
      _delete_jax_leaves(merged, keep_tree=self.hub.base_state)
      del merged
      gc.collect()
      if drv is not None and hasattr(drv, "resume"):
        drv.resume()

  def load_adapter(self, lora_name: str, lora_path: str) -> None:
    with self._lock:
      if "proposer" in lora_name:
        self.hub.snapshot_adapter(lora_name, "proposer", lora_path=lora_path)
      elif "director" in lora_name:
        self._sync_director_adapter(lora_name, lora_path)
      if len(self.hub.served_snapshots) > 4:
        for old_key in list(self.hub.served_snapshots.keys())[:-4]:
          self.hub.served_snapshots.pop(old_key, None)
      self.loaded_models.add(lora_name)

  def unload_adapter(self, lora_name: str) -> None:
    with self._lock:
      self.loaded_models.discard(lora_name)
      self.hub.served_snapshots.pop(lora_name, None)

  def reset_prefix_cache(self) -> None:
    if hasattr(self.director_sampler, "reset_prefix_cache"):
      self.director_sampler.reset_prefix_cache()

  def _record_trace(
      self,
      model_name: str,
      prompt_ids: list[int],
      comp_ids: list[int],
      logprobs: list[float],
      text: str = "",
  ) -> None:
    if not prompt_ids or not comp_ids:
      return
    trainable = "director" in model_name or "proposer" in model_name
    role = (
        "proposer"
        if "proposer" in model_name
        else ("director" if "director" in model_name else "worker")
    )
    ver_tail = model_name.rsplit("-v", 1)[1] if "-v" in model_name else ""
    version = int(ver_tail) if ver_tail.isdigit() else 0
    with self._lock:
      self.call_counter += 1
      if len(self.recent_traces) >= 256:
        self.recent_traces.pop(0)
      self.recent_traces.append(
          swarm_trace.CallTrace(
              call_id=f"call-{self.call_counter:08d}",
              role=role,
              policy_id=model_name,
              policy_version=version,
              prompt_ids=tuple(prompt_ids),
              completion_ids=tuple(comp_ids),
              rollout_logprobs=tuple(min(0.0, float(x)) for x in logprobs),
              loss_mask=tuple((1 if trainable else 0) for _ in comp_ids),
              text=text,
              trainable=trainable,
          )
      )

  def handle_constrained_choice(
      self,
      body: dict[str, Any],
      prompt_ids: list[int],
      extra: dict[str, Any],
  ) -> dict[str, Any]:
    model_name = str(body.get("model", self.base_name))
    allowed = [int(t) for t in extra["allowed_token_ids"]]
    lp_ids = [int(t) for t in extra.get("logprob_token_ids", allowed)]
    role = (
        "proposer"
        if "proposer" in model_name
        else ("solver" if "director" in model_name else None)
    )
    logits = self.hub.predict_last_logits(
        prompt_ids, role=role, served_name=model_name
    )
    shifted = logits.astype(np.float64) - float(np.max(logits))
    full_lp = shifted - math.log(float(np.sum(np.exp(shifted))))
    temp = float(body.get("temperature", 1.0))
    sampled_id = _sample_constrained_token(
        logits, allowed, temperature=temp, seed=body.get("seed")
    )
    sampled_str = self.hf_tokenizer.decode(
        [sampled_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    top_items = [
        self._token_logprob_entry(tid, float(full_lp[tid])) for tid in lp_ids
    ]
    sampled_lp = float(full_lp[sampled_id])
    self._record_trace(model_name, prompt_ids, [sampled_id], [sampled_lp])
    return _format_openai_completion(
        model_name=model_name,
        prompt_ids=prompt_ids,
        comp_ids=[sampled_id],
        content=sampled_str,
        finish_reason="length",
        content_logprobs=[{
            **self._token_logprob_entry(sampled_id, sampled_lp),
            "top_logprobs": top_items,
        }],
    )

  def _token_logprob_entry(self, tid: int, lp: float) -> dict[str, Any]:
    tok_str = self.hf_tokenizer.decode(
        [tid], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return {
        "token": tok_str,
        "logprob": lp,
        "bytes": list(tok_str.encode("utf-8")),
    }

  def handle_chat_completion(self, body: dict[str, Any]) -> dict[str, Any]:
    nested_extra = dict(body.get("extra_body") or {})
    extra = {**body, **nested_extra}
    tmpl_kwargs = dict(extra.get("chat_template_kwargs") or {})
    prompt_ids = self.render_and_encode(
        list(body["messages"]),
        add_generation_prompt=True,
        chat_template_kwargs=tmpl_kwargs,
    )
    if "allowed_token_ids" in extra:
      return self.handle_constrained_choice(body, prompt_ids, extra)
    model_name = str(body.get("model", self.base_name))
    sampler = (
        self.director_sampler if "director" in model_name else self.worker_sampler
    )
    gen_args = _resolve_sampling_kwargs(
        body, extra, self.max_model_len, len(prompt_ids)
    )
    out = sampler(prompt_token_ids=[prompt_ids], **gen_args)
    return self._decode_sampler_response(
        model_name, prompt_ids, out, want_lp=bool(body.get("logprobs"))
    )

  def _decode_sampler_response(
      self,
      model_name: str,
      prompt_ids: list[int],
      out: Any,
      *,
      want_lp: bool,
  ) -> dict[str, Any]:
    comp_ids = [int(t) for t in np.asarray(out.tokens[0]).reshape(-1)]
    raw_lps = out.logprobs[0] if out.logprobs else None
    lps = [float(v) for v in raw_lps] if raw_lps is not None else [0.0] * len(comp_ids)
    stopped = bool(comp_ids and comp_ids[-1] in self.eos_ids)
    visible_ids = comp_ids[:-1] if stopped else comp_ids
    content = self.hf_tokenizer.decode(
        visible_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    lp_payload = (
        [
            {**self._token_logprob_entry(tid, lp), "top_logprobs": []}
            for tid, lp in zip(comp_ids, lps)
        ]
        if want_lp
        else None
    )
    self._record_trace(model_name, prompt_ids, comp_ids, lps)
    return _format_openai_completion(
        model_name=model_name,
        prompt_ids=prompt_ids,
        comp_ids=comp_ids,
        content=content,
        finish_reason="stop" if stopped else "length",
        content_logprobs=lp_payload,
    )


def _delete_jax_leaves(tree: Any, *, keep_tree: Any = None) -> None:
  keep_ids = (
      {id(x) for x in jax.tree.leaves(keep_tree)} if keep_tree is not None else set()
  )
  for leaf in jax.tree.leaves(tree):
    if (
        isinstance(leaf, jax.Array)
        and id(leaf) not in keep_ids
        and not leaf.is_deleted()
    ):
      leaf.delete()


def _resolve_sampling_kwargs(
    body: dict[str, Any],
    extra: dict[str, Any],
    max_model_len: int,
    prompt_len: int,
) -> dict[str, Any]:
  req_max = max(1, int(body.get("max_tokens") or 1024))
  max_steps = min(req_max, max(1, max_model_len - prompt_len))
  temp = float(body.get("temperature", 0.0))
  raw_topk = extra.get("top_k", body.get("top_k"))
  topk_val = (
      int(raw_topk)
      if (temp > 0.0 and raw_topk is not None and int(raw_topk) > 0)
      else None
  )
  return {
      "max_generation_steps": max_steps,
      "temperature": temp,
      "top_p": float(body.get("top_p", 1.0)) if temp > 0.0 else None,
      "top_k": topk_val,
  }


def _sample_constrained_token(
    logits: np.ndarray,
    allowed_ids: list[int],
    *,
    temperature: float,
    seed: Any,
) -> int:
  sub = np.asarray([float(logits[i]) for i in allowed_ids], dtype=np.float64)
  if temperature <= 1e-6:
    return allowed_ids[int(np.argmax(sub))]
  scaled = (sub - float(np.max(sub))) / float(temperature)
  weights = np.exp(scaled)
  rng = random.Random(int(seed)) if seed is not None else random.SystemRandom()
  draw = rng.random() * float(np.sum(weights))
  acc = 0.0
  for tid, w in zip(allowed_ids, weights):
    acc += float(w)
    if draw <= acc:
      return tid
  return allowed_ids[-1]


def _format_openai_completion(
    *,
    model_name: str,
    prompt_ids: list[int],
    comp_ids: list[int],
    content: str,
    finish_reason: str,
    content_logprobs: list[dict[str, Any]] | None,
) -> dict[str, Any]:
  choice: dict[str, Any] = {
      "index": 0,
      "message": {"role": "assistant", "content": content},
      "finish_reason": finish_reason,
      "token_ids": comp_ids,
      "logprobs": (
          {"content": content_logprobs}
          if content_logprobs is not None
          else None
      ),
  }
  return {
      "id": f"chatcmpl-{uuid.uuid4().hex}",
      "object": "chat.completion",
      "created": int(time.time()),
      "model": model_name,
      "prompt_token_ids": prompt_ids,
      "choices": [choice],
      "usage": {
          "prompt_tokens": len(prompt_ids),
          "completion_tokens": len(comp_ids),
          "total_tokens": len(prompt_ids) + len(comp_ids),
      },
  }


class _GatewayHttpServer(ThreadingHTTPServer):
  request_queue_size = 512
  daemon_threads = True
  state: R2OrchTpuGatewayState


class _GatewayHandler(BaseHTTPRequestHandler):
  """HTTP handler implementing vLLM's OpenAI + LoRA + tokenize endpoints."""

  protocol_version = "HTTP/1.1"

  def _send_json(self, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    try:
      self.send_response(code)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(body)))
      self.end_headers()
      self.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
      return

  def _read_body(self) -> dict[str, Any]:
    n = int(self.headers.get("Content-Length", "0"))
    raw = self.rfile.read(n).decode("utf-8") if n > 0 else "{}"
    return json.loads(raw) if raw.strip() else {}

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    srv: _GatewayHttpServer = self.server  # type: ignore[assignment]
    if self.path.startswith("/health"):
      self._send_json(200, {"status": "ok"})
      return
    if self.path.startswith("/v1/models"):
      with srv.state._lock:
        models = sorted(srv.state.loaded_models)
      self._send_json(
          200,
          {
              "object": "list",
              "data": [{"id": m, "object": "model"} for m in models],
          },
      )
      return
    self._send_json(404, {"error": f"unknown GET {self.path}"})

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    srv: _GatewayHttpServer = self.server  # type: ignore[assignment]
    try:
      body = self._read_body()
      if self.path.startswith("/tokenize"):
        ids = srv.state.render_and_encode(
            list(body.get("messages") or []),
            add_generation_prompt=bool(body.get("add_generation_prompt", True)),
            chat_template_kwargs=dict(body.get("chat_template_kwargs") or {}),
        )
        self._send_json(
            200,
            {
                "count": len(ids),
                "tokens": ids,
                "max_model_len": srv.state.max_model_len,
            },
        )
        return
      if self.path.startswith("/v1/chat/completions"):
        self._send_json(200, srv.state.handle_chat_completion(body))
        return
      if self.path.startswith("/v1/load_lora_adapter"):
        srv.state.load_adapter(
            str(body["lora_name"]), str(body.get("lora_path", ""))
        )
        self._send_json(200, {"status": "ok"})
        return
      if self.path.startswith("/v1/unload_lora_adapter"):
        srv.state.unload_adapter(str(body["lora_name"]))
        self._send_json(200, {"status": "ok"})
        return
      if self.path.startswith("/reset_prefix_cache"):
        srv.state.reset_prefix_cache()
        self._send_json(200, {"status": "ok"})
        return
      self._send_json(404, {"error": f"unknown POST {self.path}"})
    except Exception as exc:  # pylint: disable=broad-exception-caught
      import traceback  # pylint: disable=g-import-not-at-top

      traceback.print_exc()
      self._send_json(500, {"error": {"message": repr(exc), "type": type(exc).__name__}})

  def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
    del format, args


def start_tpu_openai_gateway(
    state: R2OrchTpuGatewayState,
    *,
    host: str = "127.0.0.1",
    port: int = 18600,
) -> _GatewayHttpServer:
  """Starts the background OpenAI-compatible serving gateway on `host:port`."""
  server = _GatewayHttpServer((host, port), _GatewayHandler)
  server.state = state
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server
