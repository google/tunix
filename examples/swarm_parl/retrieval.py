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
"""Multi-corpus retrieval & document-open HTTP server (Wiki-18 + BrowseComp-Plus)."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mmap
from pathlib import Path
import threading
from typing import Any

import numpy as np

from examples.swarm_r2orch import paper_retrieval


class LocalDenseCorpusRetriever:
  """Exact inner-product dense retriever + O(1) mmap `open_doc` for ~100K corpora."""

  def __init__(
      self,
      corpus_dir: str | Path,
      query_encoder: Any,
  ) -> None:
    root = Path(corpus_dir)
    self._encode_queries = query_encoder
    self.offsets = np.load(root / "corpus.offsets.npy")
    self.total = int(self.offsets.shape[0])
    self._corpus_file = (root / "corpus.jsonl").open("rb")
    self.corpus = mmap.mmap(
        self._corpus_file.fileno(), 0, access=mmap.ACCESS_READ
    )
    docid_map_path = root / "docid_to_idx.json"
    self.docid_to_idx: dict[str, int] = (
        json.loads(docid_map_path.read_text(encoding="utf-8"))
        if docid_map_path.is_file()
        else {}
    )
    vec_path = root / "vectors_fp16.npy"
    if not vec_path.is_file():
      raise FileNotFoundError(f"Missing dense index vectors: {vec_path}")
    self.vectors_f32 = np.asarray(np.load(vec_path), dtype=np.float32)
    self._lock = threading.Lock()

  def _read_doc_at(self, idx: int) -> dict[str, Any]:
    if idx < 0 or idx >= self.total:
      return {}
    start = int(self.offsets[idx])
    end = (
        int(self.offsets[idx + 1])
        if idx + 1 < self.total
        else self.corpus.size()
    )
    raw = self.corpus[start:end].decode("utf-8", errors="replace").strip()
    return json.loads(raw) if raw else {}

  def search(self, texts: list[str], topk: int = 5) -> list[list[dict[str, Any]]]:
    with self._lock:
      q_vecs = self._encode_queries(texts)[: len(texts)]
    scores = q_vecs @ self.vectors_f32.T
    k = max(1, min(int(topk), self.total))
    out: list[list[dict[str, Any]]] = []
    for row in scores:
      part = np.argpartition(-row, k - 1)[:k]
      order = part[np.argsort(-row[part], kind="stable")]
      hits: list[dict[str, Any]] = []
      for idx in order:
        doc = self._read_doc_at(int(idx))
        doc.setdefault("docid", str(int(idx)))
        hits.append({"document": doc, "score": float(row[int(idx)])})
      out.append(hits)
    return out

  def open_doc(self, docid: str, *, max_chars: int = 3000) -> dict[str, Any]:
    key = str(docid or "").strip()
    idx = self.docid_to_idx.get(key)
    if idx is None and key.isdigit():
      idx = int(key)
    if idx is None:
      return {"found": False, "docid": key, "title": "", "url": "", "contents": ""}
    doc = self._read_doc_at(idx)
    return {
        "found": bool(doc),
        "docid": str(doc.get("docid", key)),
        "title": str(doc.get("title", "")),
        "url": str(doc.get("url", "")),
        "contents": str(doc.get("contents", doc.get("text", "")))[:max_chars],
    }


class _MultiCorpusHandler(BaseHTTPRequestHandler):
  wiki_retriever: Any = None
  browsecomp_retriever: LocalDenseCorpusRetriever | None = None

  def log_message(self, fmt: str, *args: Any) -> None:
    del fmt, args

  def _send_json(self, code: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    self.send_response(code)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # pylint: disable=invalid-name
    if self.path != "/health":
      self._send_json(404, {"error": "not found"})
      return
    bc = self.browsecomp_retriever
    self._send_json(
        200,
        {
            "status": "ok",
            "wiki18": getattr(self.wiki_retriever, "total", 0),
            "browsecomp": bc.total if bc is not None else 0,
        },
    )

  def _handle_retrieve(self, req: dict[str, Any]) -> None:
    queries = [str(q) for q in (req.get("queries") or [])]
    topk = int(req.get("topk", 3))
    corpus = str(req.get("corpus", "wiki18")).strip().lower()
    bc = self.browsecomp_retriever
    if corpus == "browsecomp" and bc is not None:
      res = bc.search(queries, topk=topk)
    else:
      res = self.wiki_retriever.search(queries, topk=topk)
    self._send_json(200, {"result": res})

  def _handle_open_doc(self, req: dict[str, Any]) -> None:
    bc = self.browsecomp_retriever
    if bc is None:
      self._send_json(404, {"error": "browsecomp corpus not loaded"})
      return
    doc = bc.open_doc(
        str(req.get("docid", "")), max_chars=int(req.get("max_chars", 3000))
    )
    self._send_json(200, doc)

  def do_POST(self) -> None:  # pylint: disable=invalid-name
    length = int(self.headers.get("Content-Length", "0"))
    req = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
    if self.path == "/retrieve":
      self._handle_retrieve(req)
    elif self.path == "/open_doc":
      self._handle_open_doc(req)
    else:
      self._send_json(404, {"error": "not found"})


def start_multi_corpus_retrieval_server(
    wiki18_dir: str | Path,
    e5_model_dir: str | Path,
    *,
    browsecomp_dir: str | Path | None = None,
    port: int = 18610,
    tpu_devices: list[Any] | None = None,
) -> tuple[ThreadingHTTPServer, Any, LocalDenseCorpusRetriever | None]:
  """Starts a unified HTTP server backing both Wiki-18 and BrowseComp-Plus."""
  wiki_ret = paper_retrieval.TpuWiki18Retriever(
      wiki18_dir, e5_model_dir, tpu_devices=tpu_devices
  )
  wiki_ret.warm_up()
  bc_ret: LocalDenseCorpusRetriever | None = None
  if browsecomp_dir is not None and (Path(browsecomp_dir) / "vectors_fp16.npy").is_file():
    bc_ret = LocalDenseCorpusRetriever(browsecomp_dir, wiki_ret.encode_queries)
  handler_cls = type(
      "_BoundMultiCorpusHandler",
      (_MultiCorpusHandler,),
      {"wiki_retriever": wiki_ret, "browsecomp_retriever": bc_ret},
  )
  server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  return server, wiki_ret, bc_ret
