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
"""JAX TPU Wiki-18 exact inner-product retrieval server (`POST /retrieve`)."""

from __future__ import annotations

import functools
import gc
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mmap
from pathlib import Path
import queue
import threading
import time
from typing import Any

import jax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

EMBED_DIM = 768
SHORTLIST_K = 50
BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
LENGTH_BUCKETS: tuple[int, ...] = (64, 128, 192, 256)


def _local_topk(
    queries_f32: jax.Array,
    local_mat_bf16: jax.Array,
    local_ids: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  scores = jnp.matmul(
      queries_f32.astype(jnp.bfloat16),
      local_mat_bf16.T,
      preferred_element_type=jnp.float32,
  )
  valid = local_ids >= 0
  scores = jnp.where(valid[None, :], scores, -1e9)
  k = min(SHORTLIST_K, int(local_mat_bf16.shape[0]))
  vals, idx = jax.lax.top_k(scores, k)
  chosen_ids = jnp.take(local_ids, idx, axis=0)
  return vals[None, :, :], chosen_ids[None, :, :]


def _build_sharded_search_fn(mesh: jax.sharding.Mesh):
  @functools.partial(
      jax.jit,
      in_shardings=(
          NamedSharding(mesh, P(None, None)),
          NamedSharding(mesh, P("ret", None)),
          NamedSharding(mesh, P("ret")),
      ),
      out_shardings=(
          NamedSharding(mesh, P(None, None)),
          NamedSharding(mesh, P(None, None)),
      ),
  )
  def _search_fn(
      queries_f32: jax.Array,
      matrix_bf16: jax.Array,
      row_ids: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    mapped = shard_map(
        _local_topk,
        mesh=mesh,
        in_specs=(P(None, None), P("ret", None), P("ret")),
        out_specs=(P("ret", None, None), P("ret", None, None)),
        check_rep=False,
    )
    local_vals_3d, local_ids_3d = mapped(
        queries_f32, matrix_bf16, row_ids
    )
    # Transpose (num_shards, B, k) -> (B, num_shards * k)
    merged_vals = jnp.transpose(local_vals_3d, (1, 0, 2)).reshape(
        queries_f32.shape[0], -1
    )
    merged_ids = jnp.transpose(local_ids_3d, (1, 0, 2)).reshape(
        queries_f32.shape[0], -1
    )
    k = min(SHORTLIST_K, int(merged_vals.shape[1]))
    best_vals, order = jax.lax.top_k(merged_vals, k)
    best_ids = jnp.take_along_axis(merged_ids, order, axis=1)
    return best_vals, best_ids

  return _search_fn


class TpuWiki18Retriever:
  """Shards 21M Wiki-18 E5 vectors across TPU HBM for exact top-k IP search."""

  def __init__(
      self,
      wiki18_dir: str | Path,
      e5_model_dir: str | Path,
      *,
      tpu_devices: list[Any] | None = None,
  ) -> None:
    import torch  # pylint: disable=g-import-not-at-top
    from transformers import AutoModel  # pylint: disable=g-import-not-at-top
    from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

    self._torch = torch
    root = Path(wiki18_dir)
    self._vectors_path = root / "vectors_fp16.npy"
    self.vectors: Any = np.load(self._vectors_path, mmap_mode="r")
    self.total = int(self.vectors.shape[0])
    self.offsets = np.load(root / "wiki-18.offsets.npy")
    self._corpus_file = (root / "wiki-18.jsonl").open("rb")
    self.corpus = mmap.mmap(
        self._corpus_file.fileno(), 0, access=mmap.ACCESS_READ
    )
    self.tokenizer = AutoTokenizer.from_pretrained(str(e5_model_dir))
    self.encoder = AutoModel.from_pretrained(
        str(e5_model_dir), torch_dtype=torch.float32
    ).eval()
    self.timing: dict[str, float] = {}
    devices = list(tpu_devices) if tpu_devices else list(jax.devices())
    self.mesh = jax.sharding.Mesh(np.asarray(devices), ("ret",))
    self._init_sharded_matrix(len(devices))
    self.release_vectors_file(unlink=False)
    self._search_fn = _build_sharded_search_fn(self.mesh)

  def release_vectors_file(self, *, unlink: bool = False) -> None:
    """Closes the host mmap of `vectors_fp16.npy` (already sharded in TPU HBM)."""
    vec = getattr(self, "vectors", None)
    if vec is not None:
      mm = getattr(vec, "_mmap", None)
      if mm is not None:
        try:
          mm.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      self.vectors = None
      gc.collect()
    if unlink and self._vectors_path.is_file():
      self._vectors_path.unlink(missing_ok=True)

  def _init_sharded_matrix(self, num_shards: int) -> None:
    pad_rows = (-self.total) % num_shards
    padded_total = self.total + pad_rows
    rows_per_shard = padded_total // num_shards
    shard_arrays = []
    id_arrays = []
    for s_idx, dev in enumerate(self.mesh.devices.flat):
      start = s_idx * rows_per_shard
      end = min(self.total, start + rows_per_shard)
      valid_n = max(0, end - start)
      buf = np.zeros((rows_per_shard, EMBED_DIM), dtype=np.float16)
      ids = np.full((rows_per_shard,), -1, dtype=np.int32)
      if valid_n > 0:
        buf[:valid_n] = np.asarray(self.vectors[start:end], dtype=np.float16)
        ids[:valid_n] = np.arange(start, end, dtype=np.int32)
      shard_arrays.append(jax.device_put(jnp.asarray(buf, dtype=jnp.bfloat16), dev))
      id_arrays.append(jax.device_put(jnp.asarray(ids, dtype=jnp.int32), dev))
    mat_shd = NamedSharding(self.mesh, P("ret", None))
    id_shd = NamedSharding(self.mesh, P("ret"))
    self.matrix_bf16 = jax.make_array_from_single_device_arrays(
        (padded_total, EMBED_DIM), mat_shd, shard_arrays
    )
    self.row_ids = jax.make_array_from_single_device_arrays(
        (padded_total,), id_shd, id_arrays
    )

  def encode_queries(self, texts: list[str]) -> np.ndarray:
    torch = self._torch
    rows = next(b for b in BATCH_BUCKETS if b >= len(texts))
    padded = [f"query: {t}" for t in texts] + ["query: "] * (rows - len(texts))
    probe = self.tokenizer(
        padded, max_length=256, padding=True, truncation=True, return_tensors="pt"
    )
    seq_len = int(probe["input_ids"].shape[1])
    length = next(b for b in LENGTH_BUCKETS if b >= seq_len)
    batch = self.tokenizer(
        padded,
        max_length=length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
      hidden = self.encoder(**batch).last_hidden_state
      mask = batch["attention_mask"].unsqueeze(-1).float()
      pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
      normed = torch.nn.functional.normalize(pooled.float(), dim=-1)
    return normed.cpu().numpy().astype(np.float32)

  def warm_up(self) -> None:
    for rows in (1, 4, 16, 64, 128):
      dummy = jnp.zeros((rows, EMBED_DIM), dtype=jnp.float32)
      _, ids = self._search_fn(dummy, self.matrix_bf16, self.row_ids)
      jax.block_until_ready(ids)

  def search(self, texts: list[str], topk: int) -> list[list[dict[str, Any]]]:
    limit = BATCH_BUCKETS[-1]
    if len(texts) > limit:
      merged: list[list[dict[str, Any]]] = []
      for start in range(0, len(texts), limit):
        merged.extend(self.search(texts[start : start + limit], topk))
      return merged
    t0 = time.monotonic()
    q_bucket = self.encode_queries(texts)
    t_enc = time.monotonic()
    shortlist_vals_jax, shortlist_jax = self._search_fn(
        jnp.asarray(q_bucket), self.matrix_bf16, self.row_ids
    )
    shortlist_vals = np.asarray(jax.device_get(shortlist_vals_jax))[: len(texts)]
    shortlist_ids = np.asarray(jax.device_get(shortlist_jax))[: len(texts)]
    t_ip = time.monotonic()
    results = self._rescore_and_fetch(
        q_bucket[: len(texts)],
        shortlist_ids,
        topk,
        shortlist_vals=shortlist_vals,
    )
    t_done = time.monotonic()
    self.timing["encode_s"] = self.timing.get("encode_s", 0.0) + (t_enc - t0)
    self.timing["search_s"] = self.timing.get("search_s", 0.0) + (t_ip - t_enc)
    self.timing["fetch_s"] = self.timing.get("fetch_s", 0.0) + (t_done - t_ip)
    return results

  def _rank_candidates(
      self,
      q_vec: np.ndarray,
      cands: np.ndarray,
      cand_vals: np.ndarray | None,
      topk: int,
  ) -> list[tuple[int, float]]:
    if self.vectors is not None:
      valid = cands[(cands >= 0) & (cands < self.total)]
      unique_ids = np.sort(np.unique(valid))
      exact_vecs = np.asarray(self.vectors[unique_ids], dtype=np.float32)
      exact_scores = exact_vecs @ q_vec
      keep = np.argsort(-exact_scores, kind="stable")[:topk]
      return [(int(unique_ids[idx]), float(exact_scores[idx])) for idx in keep]
    best_by_id: dict[int, float] = {}
    vals = cand_vals if cand_vals is not None else np.zeros_like(cands, dtype=np.float32)
    for raw_id, raw_val in zip(cands, vals):
      pid = int(raw_id)
      if 0 <= pid < self.total:
        score = float(raw_val)
        if pid not in best_by_id or score > best_by_id[pid]:
          best_by_id[pid] = score
    ranked = sorted(best_by_id.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:topk]

  def _rescore_and_fetch(
      self,
      queries: np.ndarray,
      shortlist_ids: np.ndarray,
      topk: int,
      shortlist_vals: np.ndarray | None = None,
  ) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    val_rows = shortlist_vals if shortlist_vals is not None else [None] * len(queries)
    for q_vec, cands, cand_vals in zip(queries, shortlist_ids, val_rows):
      hits: list[dict[str, Any]] = []
      for pid, score in self._rank_candidates(q_vec, cands, cand_vals, topk):
        line = self.corpus[int(self.offsets[pid]) : int(self.offsets[pid + 1])]
        rec = json.loads(line)
        title, _, text = str(rec["contents"]).partition("\n")
        hits.append({
            "id": rec["id"],
            "title": title.strip('"'),
            "text": text,
            "score": round(score, 5),
        })
      out.append(hits)
    return out

  def close(self) -> None:
    self.corpus.close()
    self._corpus_file.close()


class RetrievalBatcher:
  """Micro-batches concurrent HTTP `/retrieve` calls within a 15ms window."""

  def __init__(
      self,
      retriever: TpuWiki18Retriever,
      *,
      window_ms: float = 15.0,
      max_batch: int = 128,
  ) -> None:
    self.retriever = retriever
    self.window_s = window_ms / 1000.0
    self.max_batch = max_batch
    self.inbox: queue.Queue[Any] = queue.Queue()
    self.stats: dict[str, Any] = {
        "requests": 0,
        "queries": 0,
        "batches": 0,
        "busy_seconds": 0.0,
        "started": time.time(),
    }
    self._stop = False
    self._ready = threading.Event()
    self._thread = threading.Thread(target=self._run_loop, daemon=True)
    self._thread.start()
    self._ready.wait()

  def submit(self, queries: list[str], topk: int) -> list[list[dict[str, Any]]]:
    done = threading.Event()
    box: dict[str, Any] = {}
    self.inbox.put((queries, topk, done, box))
    done.wait()
    if "error" in box:
      raise RuntimeError(box["error"])
    return box["result"]

  def _drain_window(self, first_item: Any) -> list[Any]:
    items = [first_item]
    deadline = time.monotonic() + self.window_s
    while sum(len(it[0]) for it in items) < self.max_batch:
      rem = deadline - time.monotonic()
      if rem <= 0:
        break
      try:
        nxt = self.inbox.get(timeout=rem)
      except queue.Empty:
        break
      if nxt is None:
        self._stop = True
        break
      items.append(nxt)
    return items

  def _run_loop(self) -> None:
    self.retriever.warm_up()
    self.stats["started"] = time.time()
    self._ready.set()
    while not self._stop:
      first = self.inbox.get()
      if first is None:
        break
      items = self._drain_window(first)
      t0 = time.monotonic()
      self._execute_batch(items)
      self.stats["busy_seconds"] += time.monotonic() - t0
      self.stats["batches"] += 1
      self.stats["requests"] += len(items)
      self.stats["queries"] += sum(len(it[0]) for it in items)

  def _execute_batch(self, items: list[Any]) -> None:
    try:
      max_k = max(it[1] for it in items)
      flat = [q for it in items for q in it[0]]
      found = self.retriever.search(flat, max_k)
      cursor = 0
      for queries, k, done, box in items:
        chunk = found[cursor : cursor + len(queries)]
        box["result"] = [hits[:k] for hits in chunk]
        cursor += len(queries)
        done.set()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      for _, _, done, box in items:
        box["error"] = repr(exc)
        done.set()

  def shutdown(self) -> None:
    self._stop = True
    self.inbox.put(None)


class _RetrievalHttpServer(ThreadingHTTPServer):
  request_queue_size = 512
  daemon_threads = True
  batcher: RetrievalBatcher
  retriever: TpuWiki18Retriever


class _RetrievalHandler(BaseHTTPRequestHandler):
  """HTTP handler for `/health` and `/retrieve`."""

  protocol_version = "HTTP/1.1"

  def _write_json(self, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    srv: _RetrievalHttpServer = self.server  # type: ignore[assignment]
    if not self.path.startswith("/health"):
      self._write_json(404, {"error": "not found"})
      return
    uptime = max(1.0, time.time() - srv.batcher.stats["started"])
    self._write_json(
        200,
        {
            "ok": True,
            "passages": srv.retriever.total,
            **srv.batcher.stats,
            "timing": {k: round(v, 3) for k, v in srv.retriever.timing.items()},
            "busy_fraction": round(
                srv.batcher.stats["busy_seconds"] / uptime, 4
            ),
        },
    )

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    srv: _RetrievalHttpServer = self.server  # type: ignore[assignment]
    if not self.path.startswith("/retrieve"):
      self._write_json(404, {"error": "not found"})
      return
    try:
      nbytes = int(self.headers.get("Content-Length", "0"))
      req = json.loads(self.rfile.read(nbytes))
      queries = [str(q)[:2000] for q in req["queries"]]
      topk = max(1, min(20, int(req.get("topk", 3))))
      self._write_json(200, {"result": srv.batcher.submit(queries, topk)})
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._write_json(500, {"error": repr(exc)})

  def log_message(self, format: str, *args: Any) -> None:  # pylint: disable=redefined-builtin
    del format, args


def start_tpu_retrieval_server(
    wiki18_dir: str | Path,
    e5_model_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 18610,
    tpu_devices: list[Any] | None = None,
) -> tuple[_RetrievalHttpServer, TpuWiki18Retriever]:
  """Starts the background TPU Wiki-18 retrieval HTTP server."""
  retriever = TpuWiki18Retriever(
      wiki18_dir, e5_model_dir, tpu_devices=tpu_devices
  )
  batcher = RetrievalBatcher(retriever)
  server = _RetrievalHttpServer((host, port), _RetrievalHandler)
  server.batcher = batcher
  server.retriever = retriever
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server, retriever
