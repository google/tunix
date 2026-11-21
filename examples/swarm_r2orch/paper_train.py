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
"""Full-paper Qwen3.5-9B R2Orch TPU preflight, 200-cycle trainer, and evaluator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import urllib.request

import jax
import numpy as np

from examples.swarm_r2orch import paper_eval
from examples.swarm_r2orch import paper_fp32_director
from examples.swarm_r2orch import paper_retrieval
from examples.swarm_r2orch import paper_serving
from examples.swarm_r2orch import paper_trainer
from tunix.dist_rl.swarm import contracts as swarm_contracts
from tunix.dist_rl.swarm import step_context as swarm_step


@dataclass(frozen=True)
class TpuSlicePartition:
  """Disjoint TPU device partition across learner, retrieval, Director, and Worker."""

  train_mesh: jax.sharding.Mesh
  retrieval_devices: list[Any]
  director_mesh: jax.sharding.Mesh
  worker_mesh: jax.sharding.Mesh


def _maybe_init_pathways() -> None:
  if "proxy" in os.environ.get("JAX_PLATFORMS", ""):
    import pathwaysutils  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-import]

    pathwaysutils.initialize()


def _sort_devices_by_z_slab(devs: list[Any]) -> list[Any]:
  if devs and hasattr(devs[0], "coords"):
    return sorted(
        devs,
        key=lambda d: (
            tuple(reversed(tuple(d.coords))),
            getattr(d, "core_on_chip", 0),
        ),
    )
  return devs


def partition_tpu_devices(devices: list[Any] | None = None) -> TpuSlicePartition:
  _maybe_init_pathways()
  raw_devs = list(devices) if devices is not None else list(jax.devices())
  devs = _sort_devices_by_z_slab(raw_devs)
  n = len(devs)
  if n >= 256:
    # Contiguous z-slabs on 4x8x8: train=32, ret=16, fp32_director=64, worker=144
    n_train, n_ret, n_dir = 32, 16, 64
  elif n >= 128:
    # Contiguous z-slabs on 4x4x8 (16 chips per z-plane): 1 + 1 + 2 + 4 = 8 planes
    n_train, n_ret, n_dir = 16, 16, 32
  elif n >= 64:
    # Contiguous z-slabs on 4x4x4 (16 chips per z-plane): 1 + 1 + 1 + 1 = 4 planes
    n_train, n_ret, n_dir = 16, 16, 16
  elif n >= 16:
    n_train, n_ret, n_dir = 4, 4, 4
  else:
    raise ValueError(f"R2Orch Qwen3.5-9B TPU layout requires >= 16 chips, got {n}")
  train_devs = np.asarray(devs[:n_train]).reshape(n_train // 4, 4)
  ret_devs = devs[n_train : n_train + n_ret]
  dir_start = n_train + n_ret
  dir_devs = np.asarray(devs[dir_start : dir_start + n_dir]).reshape(n_dir, 1)
  wrk_devs = np.asarray(devs[dir_start + n_dir :]).reshape(
      n - dir_start - n_dir, 1
  )
  return TpuSlicePartition(
      train_mesh=jax.sharding.Mesh(train_devs, ("fsdp", "tp")),
      retrieval_devices=ret_devs,
      director_mesh=jax.sharding.Mesh(dir_devs, ("fsdp", "tp")),
      worker_mesh=jax.sharding.Mesh(wrk_devs, ("fsdp", "tp")),
  )


def render_paper_toml(
    *,
    upstream_root: Path,
    data_root: Path,
    run_root: Path,
    alfworld_python: str,
    rollout_workers: int,
    vllm_port: int = 18600,
    retrieval_port: int = 18610,
) -> Path:
  src_toml = upstream_root / "configs" / "r2orch.toml"
  text = src_toml.read_text(encoding="utf-8")
  text = (
      text.replace("@RUN_ROOT@", str(run_root))
      .replace("@DATA_ROOT@", str(data_root))
      .replace("@CODE_ROOT@", str(upstream_root))
  )
  text = text.replace(
      f'alfworld_python = "{data_root}/envs/alfworld/bin/python"',
      f'alfworld_python = "{alfworld_python}"',
  )
  text = text.replace(
      f'evalplus_python = "{data_root}/envs/evalplus/bin/python"',
      f'evalplus_python = "{sys.executable}"',
  )
  text = text.replace(
      'seed_skills_path = "configs/skills/seed_cards.json"',
      f'seed_skills_path = "{upstream_root}/configs/skills/seed_cards.json"',
  )
  text = text.replace("rollout_workers = 120", f"rollout_workers = {int(rollout_workers)}")
  text = text.replace("aime_rollout_wall_time_s = 360.0", "aime_rollout_wall_time_s = 1800.0")
  text = text.replace("alfworld_rollout_wall_time_s = 360.0", "alfworld_rollout_wall_time_s = 1800.0")
  text = text.replace("rollout_wall_time_s = 400.0", "rollout_wall_time_s = 1200.0")
  text = text.replace("timeout_s = 600.0", "timeout_s = 1800.0")
  text = text.replace("timeout_s = 360.0", "timeout_s = 1800.0")
  text = text.replace("vllm_port = 18600", f"vllm_port = {int(vllm_port)}")
  text = text.replace("127.0.0.1:18600", f"127.0.0.1:{int(vllm_port)}")
  text = text.replace("retrieval_port = 18610", f"retrieval_port = {int(retrieval_port)}")
  text = text.replace("127.0.0.1:18610", f"127.0.0.1:{int(retrieval_port)}")
  out_path = run_root / "r2orch_tpu.toml"
  out_path.parent.mkdir(parents=True, exist_ok=True)
  out_path.write_text(text, encoding="utf-8")
  return out_path


def configure_paper_environment(data_root: Path, upstream_root: Path) -> None:
  src_dir = str(upstream_root / "src")
  if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
  os.environ.setdefault("R2ORCH_DIRECTOR_CONTEXT_MODE", "append_only")
  os.environ.setdefault("R2ORCH_OUTPUT_CEILING", "4096")
  os.environ.setdefault("R2ORCH_RECOVERY_ATTEMPTS", "0")
  os.environ.setdefault("R2ORCH_ALLOWED_PHYSICAL_GPUS", "0,1,2,3")
  os.environ["ALFWORLD_DATA"] = str(Path(data_root) / "alfworld")


def _validate_r2orch_option5_ownership(
    model_path: Path, director_fidelity: str
) -> swarm_contracts.ServingFidelity:
  base_snap = swarm_contracts.SnapshotRef(
      model_id=model_path.name, revision=0, digest=str(model_path)
  )
  fidelity_mode = swarm_contracts.ServingFidelityMode(director_fidelity)
  dir_fidelity = swarm_contracts.ServingFidelity(
      mode=fidelity_mode,
      max_top1_mismatch_rate=0.0 if fidelity_mode == swarm_contracts.ServingFidelityMode.FP32_NATIVE else 0.15,
      max_logprob_abs_diff=1e-3 if fidelity_mode == swarm_contracts.ServingFidelityMode.FP32_NATIVE else 0.25,
      allow_lossy_lora_merge=(fidelity_mode == swarm_contracts.ServingFidelityMode.MERGED_BF16),
  )
  learners = {
      "director": swarm_contracts.LearnerSpec.lora(
          "director", base=base_snap, adapter="director", rank=64, alpha=128.0, fidelity=dir_fidelity
      ),
      "proposer": swarm_contracts.LearnerSpec.lora(
          "proposer", base=base_snap, adapter="proposer", rank=64, alpha=128.0, fidelity=dir_fidelity
      ),
  }
  bindings = {
      "director": swarm_contracts.PolicySpec.follow("director", fidelity=dir_fidelity),
      "worker": swarm_contracts.PolicySpec.fixed(base_snap),
  }
  swarm_contracts.validate_ownership(learners, bindings)
  return dir_fidelity


def boot_tpu_services(
    *,
    data_root: Path,
    upstream_root: Path,
    vllm_port: int = 18600,
    retrieval_port: int = 18610,
    max_model_len: int = 32768,
    hbm_utilization: float = 0.56,
    director_fidelity: str = "fp32_native",
    unlink_staged_safetensors: bool = False,
) -> dict[str, Any]:
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  model_path = Path(data_root) / "models" / "Qwen3.5-9B"
  e5_path = Path(data_root) / "models" / "e5-base-v2"
  wiki18_path = Path(data_root) / "wiki18"
  jinja_path = upstream_root / "configs" / "templates" / "director_append_only.jinja"
  dir_fidelity = _validate_r2orch_option5_ownership(model_path, director_fidelity)

  part = partition_tpu_devices()
  print(
      f"[tpu-boot] chips={len(jax.devices())}"
      f" train={part.train_mesh.devices.size}"
      f" ret={len(part.retrieval_devices)}"
      f" director_devs={part.director_mesh.devices.size}"
      f" worker_dp={part.worker_mesh.devices.size}"
      f" director_fidelity={dir_fidelity.mode.value}",
      flush=True,
  )
  hf_tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
  eos_ids = paper_serving.qwen3p5_eos_ids(hf_tok)
  t0 = time.monotonic()
  hub = paper_trainer.SharedTunixLearnerHub(model_path, mesh=part.train_mesh)
  print(f"[tpu-boot] loaded Qwen3.5-9B learner in {time.monotonic() - t0:.1f}s", flush=True)

  t_ret = time.monotonic()
  ret_server, retriever = paper_retrieval.start_tpu_retrieval_server(
      wiki18_path,
      e5_path,
      port=retrieval_port,
      tpu_devices=part.retrieval_devices,
  )
  print(
      f"[tpu-boot] Wiki-18 ({retriever.total} passages) ready on :{retrieval_port}"
      f" in {time.monotonic() - t_ret:.1f}s",
      flush=True,
  )
  if unlink_staged_safetensors:
    retriever.release_vectors_file(unlink=True)
    print("[tpu-boot] freed staged Wiki-18 vectors (31 GiB) from tmpfs", flush=True)

  t_vllm = time.monotonic()
  worker_sampler = paper_serving.build_qwen3p5_vllm_sampler(
      model_path,
      hf_tok,
      mesh=part.worker_mesh,
      max_model_len=max_model_len,
      hbm_utilization=hbm_utilization,
      free_kv_cache_during_weight_sync=False,
      return_logprobs=False,
      eos_tokens=eos_ids,
  )
  base_bf16 = hub.merged_state_for_vllm("solver")
  worker_sampler.update_params(base_bf16)
  if unlink_staged_safetensors:
    for sf in model_path.glob("*.safetensors"):
      sf.unlink(missing_ok=True)
    print("[tpu-boot] freed staged Qwen3.5-9B safetensors from tmpfs", flush=True)

  if dir_fidelity.mode == swarm_contracts.ServingFidelityMode.FP32_NATIVE:
    del base_bf16
    gc.collect()
    director_sampler: Any = paper_fp32_director.BatchingFp32DirectorSampler(
        hub.base_state,
        hf_tok,
        list(part.director_mesh.devices.flat),
        eos_ids,
        cache_size=max_model_len,
    )
  else:
    director_sampler = paper_serving.build_qwen3p5_vllm_sampler(
        model_path,
        hf_tok,
        mesh=part.director_mesh,
        max_model_len=max_model_len,
        hbm_utilization=hbm_utilization,
        free_kv_cache_during_weight_sync=False,
        eos_tokens=eos_ids,
    )
    director_sampler.update_params(base_bf16)
  print(f"[tpu-boot] synced base weights to serving pools in {time.monotonic() - t_vllm:.1f}s", flush=True)

  fidelity_report = paper_fp32_director.run_director_fidelity_gate(
      hub, director_sampler, hf_tok, fidelity=dir_fidelity
  )
  print(
      f"[option5-L3] ServingFidelity PASS mode={fidelity_report.mode.value} "
      f"top1={fidelity_report.top1_agreement:.4f} "
      f"max_lp_diff={fidelity_report.max_logprob_abs_diff:.3e}",
      flush=True,
  )

  gw_state = paper_serving.R2OrchTpuGatewayState(
      hf_tokenizer=hf_tok,
      director_jinja=jinja_path.read_text(encoding="utf-8"),
      hub=hub,
      worker_sampler=worker_sampler,
      director_sampler=director_sampler,
      max_model_len=max_model_len,
  )
  t_warm = time.monotonic()
  warm_msgs = [{"role": "user", "content": "Reply with OK."}]
  gw_state.handle_chat_completion({
      "model": "r2orch-base",
      "messages": warm_msgs,
      "temperature": 0.3,
      "max_tokens": 4,
  })
  gw_state.handle_chat_completion({
      "model": "r2orch-director",
      "messages": warm_msgs,
      "temperature": 1.0,
      "max_tokens": 4,
      "logprobs": True,
      "chat_template_kwargs": {"enable_thinking": False, "director_append_only": True},
  })
  print(f"[tpu-boot] warmed up Director & Worker pools in {time.monotonic() - t_warm:.1f}s", flush=True)
  gw_server = paper_serving.start_tpu_openai_gateway(gw_state, port=vllm_port)
  return {
      "partition": part,
      "hf_tokenizer": hf_tok,
      "hub": hub,
      "retrieval_server": ret_server,
      "retriever": retriever,
      "gateway_state": gw_state,
      "gateway_server": gw_server,
      "fidelity_report": fidelity_report,
  }


def run_numerical_and_env_preflight(
    services: dict[str, Any],
    *,
    data_root: Path,
    upstream_root: Path,
    retrieval_port: int = 18610,
) -> dict[str, Any]:
  """Verifies vLLM-vs-JAX logprob parity, Wiki-18 TPU search, and ALFWorld."""
  gw: paper_serving.R2OrchTpuGatewayState = services["gateway_state"]
  hub: paper_trainer.SharedTunixLearnerHub = services["hub"]

  # 1. Sampler vs Trainer forward logprob parity on Qwen3.5-9B hybrid GDN+GQA
  msgs = [{"role": "user", "content": "Compute 17 * 23 and give only the integer."}]
  resp = gw.handle_chat_completion({
      "model": "r2orch-director",
      "messages": msgs,
      "temperature": 0.7,
      "max_tokens": 24,
      "logprobs": True,
      "extra_body": {"chat_template_kwargs": {"enable_thinking": False, "director_append_only": True}},
  })
  prompt_ids = resp["prompt_token_ids"]
  choice = resp["choices"][0]
  comp_ids = choice["token_ids"]
  vllm_lps = np.asarray([it["logprob"] for it in choice["logprobs"]["content"]], dtype=np.float32)
  full_seq = list(prompt_ids) + list(comp_ids)
  pred_positions = list(range(len(prompt_ids) - 1, len(full_seq) - 1))
  trainer_logits = hub.eval_selected_logits(full_seq, pred_positions, reference=True)
  trainer_lps = []
  for row, tid in zip(trainer_logits, comp_ids):
    r64 = row.astype(np.float64) - float(np.max(row))
    lse = float(np.log(np.sum(np.exp(r64))))
    trainer_lps.append(float(r64[tid] - lse))
  trainer_lps_arr = np.asarray(trainer_lps, dtype=np.float32)
  logp_abs_diff = float(np.mean(np.abs(vllm_lps - trainer_lps_arr)))
  logp_max_diff = float(np.max(np.abs(vllm_lps - trainer_lps_arr)))
  print(
      f"[preflight] vLLM vs Qwen3p5 trainer logp diff: mean={logp_abs_diff:.5f}"
      f" max={logp_max_diff:.5f} len={len(comp_ids)} reply={choice['message']['content']!r}",
      flush=True,
  )
  if logp_abs_diff > 0.12:
    raise RuntimeError(f"vLLM vs Qwen3p5 logprob mismatch too high: {logp_abs_diff}")

  # 2. Constrained binary `off/on` + Proposer `A..H` HTTP wire-format check
  off_id = gw.hf_tokenizer.encode("off", add_special_tokens=False)[0]
  on_id = gw.hf_tokenizer.encode("on", add_special_tokens=False)[0]
  bin_http = urllib.request.Request(
      "http://127.0.0.1:18600/v1/chat/completions",
      data=json.dumps({
          "model": "r2orch-director",
          "messages": msgs,
          "temperature": 1.0,
          "max_tokens": 1,
          "logprobs": True,
          "seed": 42,
          "allowed_token_ids": [off_id, on_id],
          "logprob_token_ids": [off_id, on_id],
          "chat_template_kwargs": {
              "enable_thinking": False,
              "director_append_only": True,
          },
      }).encode(),
      headers={"Content-Type": "application/json"},
      method="POST",
  )
  with urllib.request.urlopen(bin_http, timeout=600) as r:
    bin_resp = json.loads(r.read().decode())
  assert bin_resp["choices"][0]["message"]["content"] in {"off", "on"}

  # 3. Wiki-18 TPU retrieval latency & accuracy check
  t_q = time.monotonic()
  req = urllib.request.Request(
      f"http://127.0.0.1:{retrieval_port}/retrieve",
      data=json.dumps({"queries": ["Who directed the 2010 science fiction film Inception?"], "topk": 3}).encode(),
      headers={"Content-Type": "application/json"},
      method="POST",
  )
  with urllib.request.urlopen(req, timeout=300) as r:
    ret_body = json.loads(r.read().decode())
  ret_ms = (time.monotonic() - t_q) * 1000.0
  hits = ret_body["result"][0]
  joined_hits = " ".join(h["title"] + " " + h["text"] for h in hits)
  print(f"[preflight] Wiki-18 top1={hits[0]['title']!r} latency={ret_ms:.1f}ms", flush=True)
  if "Nolan" not in joined_hits and "Inception" not in joined_hits:
    raise RuntimeError(f"Unexpected Wiki-18 retrieval result: {hits}")

  # 4. ALFWorld session check
  from r2orch.alfworld_bridge import ALFWorldBridgeClient  # pylint: disable=g-import-not-at-top

  held_alf = json.loads(
      (upstream_root / "data/r2orch/heldout/alfworld.jsonl").read_text().splitlines()[0]
  )
  game_file = Path(data_root) / "alfworld/json_2.1.1" / held_alf["metadata"]["game_path"]
  alf_client = ALFWorldBridgeClient("/opt/r2orch_alfworld_venv/bin/python")
  sess = alf_client.create_session(str(game_file), max_steps=50)
  alf_client.close_session(sess["session_id"])
  alf_client.close()
  print(f"[preflight] ALFWorld verified cmds={len(sess['admissible_commands'])}", flush=True)

  return {
      "vllm_trainer_logp_mean_abs_diff": round(logp_abs_diff, 6),
      "vllm_trainer_logp_max_abs_diff": round(logp_max_diff, 6),
      "wiki18_latency_ms": round(ret_ms, 2),
      "wiki18_top1_title": hits[0]["title"],
      "alfworld_initial_cmds": len(sess["admissible_commands"]),
  }


def _num(
    mapping: dict[str, Any] | None, key: str, default: float = 0.0
) -> float:
  val = (mapping or {}).get(key)
  return float(default if val is None else val)


def build_live_cycle_row(
    cyc: int, cycle_metrics: dict[str, Any]
) -> dict[str, Any]:
  """Extracts scalar live telemetry from an `r2orch_metrics@1` cycle record."""
  losses = cycle_metrics.get("losses") or {}
  dir_l = losses.get("director")
  prop_l = losses.get("proposer")
  cost = cycle_metrics.get("cost")
  datasets = cycle_metrics.get("datasets") or {}
  ds_accs = [
      float(v["acc_independent"])
      for v in datasets.values()
      if isinstance(v, dict) and v.get("acc_independent") is not None
  ]
  return {
      "cycle": cyc,
      "timestamp": time.time(),
      "train/acc_independent_mean": (
          sum(ds_accs) / len(ds_accs) if ds_accs else 0.0
      ),
      "train/director_loss": _num(dir_l, "loss", 0.0),
      "train/director_kl": _num(dir_l, "kl", 0.0),
      "train/director_tis_mean": _num(dir_l, "tis_weight_mean", 1.0),
      "train/proposer_loss": _num(prop_l, "pair_loss", 0.0),
      "train/proposer_p_ads": _num(prop_l, "p_ads_policy", 0.0),
      "train/proposer_grad_norm": _num(prop_l, "grad_norm_pre_clip_mean", 0.0),
      "train/cycle_elapsed_s": _num(cost, "cycle_wall_s", 0.0),
      "train/collection_wall_s": _num(cost, "collection_wall_s", 0.0),
      "train/training_wall_s": _num(cost, "training_wall_s", 0.0),
  }


class _LiveMetricsAndHeldoutTracker:
  """Streams per-cycle metrics + periodic zero-skill Heldout accuracy."""

  def __init__(
      self,
      *,
      run_root: Path,
      upstream_root: Path,
      rendered_toml: Path,
      eval_every: int = 25,
      eval_concurrency: int = 32,
  ) -> None:
    self.run_root = run_root
    self.upstream_root = upstream_root
    self.rendered_toml = rendered_toml
    self.eval_every = int(eval_every)
    self.eval_concurrency = int(eval_concurrency)
    self.live_jsonl = run_root / "tunix_live_metrics.jsonl"
    self.manifest_store = swarm_step.ManifestStore(run_root / "option5_manifest")

  def start(self, meta: dict[str, Any]) -> None:
    (self.run_root / "tunix_tracker_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    self._backfill_resumed_cycles()

  def _backfill_resumed_cycles(self) -> None:
    metrics_p = self.run_root / "r2orch_metrics.jsonl"
    if not metrics_p.is_file():
      return
    logged: set[int] = set()
    if self.live_jsonl.is_file():
      for line in self.live_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
          logged.add(int(json.loads(line).get("cycle", -1)))
    for line in metrics_p.read_text(encoding="utf-8").splitlines():
      if not line.strip():
        continue
      rec = json.loads(line)
      if int(rec.get("cycle", -1)) not in logged:
        self.log_cycle(rec)

  def log_cycle(self, cycle_metrics: dict[str, Any]) -> None:
    cyc = int(cycle_metrics.get("cycle", 0))
    r2m_path = self.run_root / f"cycle-{cyc:04d}" / "r2orch_metrics.json"
    if "losses" not in cycle_metrics and r2m_path.is_file():
      cycle_metrics = json.loads(r2m_path.read_text(encoding="utf-8"))
    row = build_live_cycle_row(cyc, cycle_metrics)
    if self.eval_every > 0 and ((cyc + 1) % self.eval_every == 0 or cyc == 0):
      held_report = self._run_heldout_eval(cyc)
      row["heldout/mean_acc"] = held_report["mean_acc"]
      for sname, sdata in held_report["suites"].items():
        row[f"heldout/{sname}_acc"] = sdata["acc_mean"]
        row[f"heldout/{sname}_multi_pct"] = sdata["multi_agent_pct"]
    with self.live_jsonl.open("a", encoding="utf-8") as f:
      f.write(json.dumps(row, ensure_ascii=False) + "\n")
    prev_m = self.manifest_store.load_committed()
    prev_step = -1 if prev_m is None else int(prev_m["step_id"])
    if cyc > prev_step:
      self.manifest_store.commit_step(
          step_id=cyc,
          expected_prev_step=prev_step,
          learner_versions={"director": cyc + 1, "proposer": cyc + 1},
          recipe_state={"completed_cycles": cyc + 1},
          cursor=cyc + 1,
          metrics={k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
      )
    self._prune_transient_cycle_files(cyc)
    print(f"[cycle-{cyc:04d}] {json.dumps(row)}", flush=True)

  def _prune_transient_cycle_files(self, cyc: int) -> None:
    old = cyc - 3
    if old >= 0:
      old_dir = self.run_root / f"cycle-{old:04d}"
      (old_dir / "solver_batch.json").unlink(missing_ok=True)
      (old_dir / "rollout_attempts.jsonl").unlink(missing_ok=True)
    trace_p = self.run_root / "state" / "traces.jsonl"
    if trace_p.is_file() and trace_p.stat().st_size > (256 << 20):
      trace_p.write_text("", encoding="utf-8")

  def _run_heldout_eval(self, cyc: int) -> dict[str, Any]:
    from r2orch.application import load_adaptive_config  # pylint: disable=g-import-not-at-top
    from r2orch.selfplay_runtime import HuggingFaceTokenizer  # pylint: disable=g-import-not-at-top

    cfg = load_adaptive_config(self.rendered_toml)
    served_path = self.run_root / f"cycle-{cyc:04d}" / "served_adapters.json"
    served = (
        json.loads(served_path.read_text(encoding="utf-8"))
        if served_path.is_file()
        else {}
    )
    dir_name = str(served.get("director") or cfg.solver_model.served_model)
    tok = HuggingFaceTokenizer(cfg.proposer_model.base_model_path)
    held_dir = self.upstream_root / "data" / "r2orch" / "heldout"
    suites = {p.stem: p for p in sorted(held_dir.glob("*.jsonl"))}
    return paper_eval.evaluate_zero_skill_suites(
        cfg,
        tok,
        suites,
        output_dir=self.run_root / "heldout_evals" / f"cycle-{cyc:04d}",
        served_director=dir_name,
        concurrency=self.eval_concurrency,
        runs=1,
    )


def build_selfplay_args(
    *,
    rendered_toml: Path,
    upstream_root: Path,
    run_root: Path,
    cycles: int,
    async_next_cycle: bool,
) -> argparse.Namespace:
  from r2orch import cli as r2_cli  # pylint: disable=g-import-not-at-top

  parser = r2_cli.build_parser()
  argv = [
      "selfplay-experiment",
      "--config",
      str(rendered_toml),
      "--task-pool",
      str(upstream_root / "data/r2orch/train/train_pool.jsonl"),
      "--curriculum-profile",
      str(upstream_root / "configs/curriculum/r2orch.toml"),
      "--output",
      str(run_root),
      "--cycles",
      str(int(cycles)),
      "--max-sequence-length",
      "35000",
      "--max-micro-batch-tokens",
      "35000",
      "--micro-batch-size",
      "1",
      "--raw-policy-backward-mode",
      "timeline",
      "--lora-dropout-override",
      "0",
      "--fuse-old-logprobs",
      "--matmul-precision",
      "high",
      "--kl-reference-stride",
      "4",
      "--solver-learning-rate",
      "1e-5",
      "--proposer-learning-rate",
      "1e-5",
      "--kl-term-cap",
      "10",
      "--resume",
  ]
  if async_next_cycle:
    argv.append("--async-next-cycle-rollouts")
  return parser.parse_args(argv)


def run_full_evaluations(
    *,
    rendered_toml: Path,
    upstream_root: Path,
    run_root: Path,
    hub: paper_trainer.SharedTunixLearnerHub,
    gw_state: paper_serving.R2OrchTpuGatewayState,
    concurrency: int,
    iid_runs: int,
    ood_runs: int,
    eval_base: bool,
    limit_per_suite: int = 0,
) -> dict[str, Any]:
  prev_ceiling = os.environ.get("R2ORCH_OUTPUT_CEILING")
  os.environ["R2ORCH_OUTPUT_CEILING"] = "12288"
  try:
    return _run_full_evaluations_body(
        rendered_toml=rendered_toml,
        upstream_root=upstream_root,
        run_root=run_root,
        hub=hub,
        gw_state=gw_state,
        concurrency=concurrency,
        iid_runs=iid_runs,
        ood_runs=ood_runs,
        eval_base=eval_base,
        limit_per_suite=limit_per_suite,
    )
  finally:
    if prev_ceiling is None:
      os.environ.pop("R2ORCH_OUTPUT_CEILING", None)
    else:
      os.environ["R2ORCH_OUTPUT_CEILING"] = prev_ceiling


def _run_full_evaluations_body(
    *,
    rendered_toml: Path,
    upstream_root: Path,
    run_root: Path,
    hub: paper_trainer.SharedTunixLearnerHub,
    gw_state: paper_serving.R2OrchTpuGatewayState,
    concurrency: int,
    iid_runs: int,
    ood_runs: int,
    eval_base: bool,
    limit_per_suite: int = 0,
) -> dict[str, Any]:
  from r2orch.application import load_adaptive_config  # pylint: disable=g-import-not-at-top
  from r2orch.selfplay_runtime import HuggingFaceTokenizer  # pylint: disable=g-import-not-at-top
  from r2orch.vllm_serving import latest_adapter  # pylint: disable=g-import-not-at-top
  from r2orch.vllm_serving import versioned_name  # pylint: disable=g-import-not-at-top

  cfg = load_adaptive_config(rendered_toml)
  tok = HuggingFaceTokenizer(cfg.proposer_model.base_model_path)
  iid_dir = upstream_root / "data/r2orch/test/iid"
  ood_dir = upstream_root / "data/r2orch/test/ood"
  iid_suites = {p.stem: p for p in sorted(iid_dir.glob("*.jsonl"))}
  ood_suites = {
      name: ood_dir / f"{name}.jsonl"
      for name in ("triviaqa", "musique", "math_hard")
      if (ood_dir / f"{name}.jsonl").is_file()
  }
  results: dict[str, Any] = {}
  if eval_base:
    results["base_iid"] = paper_eval.evaluate_zero_skill_suites(
        cfg,
        tok,
        iid_suites,
        output_dir=run_root / "final_eval" / "base_iid",
        served_director="r2orch-base",
        concurrency=concurrency,
        runs=1,
        limit_per_suite=limit_per_suite,
    )
    results["base_ood"] = paper_eval.evaluate_zero_skill_suites(
        cfg,
        tok,
        ood_suites,
        output_dir=run_root / "final_eval" / "base_ood",
        served_director="r2orch-base",
        concurrency=concurrency,
        runs=1,
        limit_per_suite=limit_per_suite,
    )
  dir_ckpt, dir_step = latest_adapter(cfg.solver_model.checkpoint_path)
  served_dir = "r2orch-director"
  if dir_ckpt is not None:
    from r2orch.training import PolicyTrainingConfig  # pylint: disable=g-import-not-at-top

    dummy_cfg = PolicyTrainingConfig(
        cfg.solver_model.base_model_path,
        cfg.solver_model.checkpoint_path,
        lora_rank=int(cfg.r2orch.lora.rank),
        lora_alpha=int(cfg.r2orch.lora.alpha),
    )
    hub.ensure_role("solver", dummy_cfg, seed=0)
    served_dir = versioned_name("r2orch-director", dir_step)
    gw_state.load_adapter(served_dir, str(dir_ckpt))
  results["trained_iid"] = paper_eval.evaluate_zero_skill_suites(
      cfg,
      tok,
      iid_suites,
      output_dir=run_root / "final_eval" / "trained_iid",
      served_director=served_dir,
      concurrency=concurrency,
      runs=iid_runs,
      limit_per_suite=limit_per_suite,
  )
  results["trained_ood"] = paper_eval.evaluate_zero_skill_suites(
      cfg,
      tok,
      ood_suites,
      output_dir=run_root / "final_eval" / "trained_ood",
      served_director=served_dir,
      concurrency=concurrency,
      runs=ood_runs,
      limit_per_suite=limit_per_suite,
  )
  (run_root / "final_eval" / "paper_comparison_report.json").write_text(
      json.dumps(results, indent=2) + "\n", encoding="utf-8"
  )
  return results


def parse_cli() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--mode", choices=("preflight", "train", "eval"), required=True)
  p.add_argument("--data-root", type=Path, required=True)
  p.add_argument("--upstream-root", type=Path, required=True)
  p.add_argument("--run-root", type=Path, required=True)
  p.add_argument("--alfworld-python", default="/opt/r2orch_alfworld_venv/bin/python")
  p.add_argument("--cycles", type=int, default=200)
  p.add_argument("--rollout-workers", type=int, default=96)
  p.add_argument("--eval-every", type=int, default=25)
  p.add_argument("--eval-concurrency", type=int, default=48)
  p.add_argument("--iid-runs", type=int, default=1)
  p.add_argument("--ood-runs", type=int, default=1)
  p.add_argument("--eval-base", action="store_true")
  p.add_argument("--eval-limit", type=int, default=0)
  p.add_argument("--async-next-cycle", action="store_true")
  p.add_argument("--skip-final-eval", action="store_true")
  p.add_argument("--hbm-utilization", type=float, default=0.56)
  p.add_argument(
      "--director-fidelity",
      choices=("fp32_native", "merged_bf16"),
      default="fp32_native",
  )
  p.add_argument("--unlink-staged-safetensors", action="store_true")
  return p.parse_args()


def _reconcile_sqlite_pats(db_p: Path, done: int) -> None:
  import sqlite3  # pylint: disable=g-import-not-at-top

  if not db_p.is_file():
    return
  conn = sqlite3.connect(str(db_p))
  row = conn.execute("SELECT payload FROM pats_state WHERE id=1").fetchone()
  if row and int(json.loads(row[0]).get("step", -1)) > done:
    ps = {**json.loads(row[0]), "step": done}
    conn.execute(
        "UPDATE pats_state SET payload=? WHERE id=1",
        (json.dumps(ps, separators=(",", ":"), sort_keys=True),),
    )
    stale_ids = [
        (cid,)
        for cid, cpay in conn.execute("SELECT id, payload FROM pats_cycles").fetchall()
        if int(json.loads(cpay).get("step", 0)) > done
    ]
    conn.executemany("DELETE FROM pats_cycles WHERE id=?", stale_ids)
    conn.commit()
  conn.close()


def reconcile_resume_state(run_root: Path) -> None:
  """Aligns SQLite PATS step & SkillLedger with committed `completed_cycles`."""
  import shutil  # pylint: disable=g-import-not-at-top

  prog_p = Path(run_root) / "experiment_progress.json"
  if not prog_p.is_file():
    return
  done = int(json.loads(prog_p.read_text(encoding="utf-8")).get("completed_cycles", 0))
  _reconcile_sqlite_pats(Path(run_root) / "state" / "director_skill_cases.sqlite3", done)
  sl_p = Path(run_root) / "state" / "skill_ledger.json"
  if sl_p.is_file():
    sl = json.loads(sl_p.read_text(encoding="utf-8"))
    if any(int(k) >= done for k in sl.get("cycles", {})):
      sl["records"] = [r for r in sl.get("records", []) if int(r["cycle"]) < done]
      sl["cycles"] = {k: v for k, v in sl["cycles"].items() if int(k) < done}
      sl_p.write_text(json.dumps(sl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  shutil.rmtree(Path(run_root) / f"cycle-{done:04d}", ignore_errors=True)


def main() -> int:
  args = parse_cli()
  reconcile_resume_state(args.run_root)
  configure_paper_environment(args.data_root, args.upstream_root)
  rendered = render_paper_toml(
      upstream_root=args.upstream_root,
      data_root=args.data_root,
      run_root=args.run_root,
      alfworld_python=args.alfworld_python,
      rollout_workers=args.rollout_workers,
  )
  services = boot_tpu_services(
      data_root=args.data_root,
      upstream_root=args.upstream_root,
      hbm_utilization=args.hbm_utilization,
      director_fidelity=args.director_fidelity,
      unlink_staged_safetensors=args.unlink_staged_safetensors,
  )
  hub: paper_trainer.SharedTunixLearnerHub = services["hub"]
  gw_state: paper_serving.R2OrchTpuGatewayState = services["gateway_state"]

  import r2orch.training as r2_training  # pylint: disable=g-import-not-at-top
  import r2orch.training_cycle as r2_cycle  # pylint: disable=g-import-not-at-top

  r2_training.TransformersGRPOTrainer = hub.factory  # type: ignore[assignment]
  r2_cycle.R2ORCH_AIME_ROLLOUT_WALL_TIME_S = 1200.0
  r2_cycle.R2ORCH_ALFWORLD_ROLLOUT_WALL_TIME_S = 1200.0
  run_r2orch_experiment = r2_cycle.run_r2orch_experiment

  if args.mode == "preflight":
    pf_checks = run_numerical_and_env_preflight(
        services, data_root=args.data_root, upstream_root=args.upstream_root
    )
    (args.run_root / "preflight_checks.json").write_text(
        json.dumps(pf_checks, indent=2) + "\n", encoding="utf-8"
    )
    exp_args = build_selfplay_args(
        rendered_toml=rendered,
        upstream_root=args.upstream_root,
        run_root=args.run_root,
        cycles=min(args.cycles, 2),
        async_next_cycle=False,
    )
    tracker = _LiveMetricsAndHeldoutTracker(
        run_root=args.run_root,
        upstream_root=args.upstream_root,
        rendered_toml=rendered,
        eval_every=0,
        eval_concurrency=16,
    )
    rc = run_r2orch_experiment(exp_args, tracker)
    if rc != 0:
      return rc
    eval_rep = run_full_evaluations(
        rendered_toml=rendered,
        upstream_root=args.upstream_root,
        run_root=args.run_root,
        hub=hub,
        gw_state=gw_state,
        concurrency=16,
        iid_runs=1,
        ood_runs=1,
        eval_base=True,
        limit_per_suite=args.eval_limit or 4,
    )
    print("[preflight] COMPLETE:", json.dumps({"checks": pf_checks, "eval_sample": eval_rep}), flush=True)
    return 0

  if args.mode == "train":
    exp_args = build_selfplay_args(
        rendered_toml=rendered,
        upstream_root=args.upstream_root,
        run_root=args.run_root,
        cycles=args.cycles,
        async_next_cycle=args.async_next_cycle,
    )
    tracker = _LiveMetricsAndHeldoutTracker(
        run_root=args.run_root,
        upstream_root=args.upstream_root,
        rendered_toml=rendered,
        eval_every=args.eval_every,
        eval_concurrency=args.eval_concurrency,
    )
    rc = run_r2orch_experiment(exp_args, tracker)
    if rc != 0 or args.skip_final_eval:
      return rc
    run_full_evaluations(
        rendered_toml=rendered,
        upstream_root=args.upstream_root,
        run_root=args.run_root,
        hub=hub,
        gw_state=gw_state,
        concurrency=args.eval_concurrency,
        iid_runs=args.iid_runs,
        ood_runs=args.ood_runs,
        eval_base=args.eval_base,
        limit_per_suite=args.eval_limit,
    )
    return 0

  run_full_evaluations(
      rendered_toml=rendered,
      upstream_root=args.upstream_root,
      run_root=args.run_root,
      hub=hub,
      gw_state=gw_state,
      concurrency=args.eval_concurrency,
      iid_runs=args.iid_runs,
      ood_runs=args.ood_runs,
      eval_base=args.eval_base,
      limit_per_suite=args.eval_limit,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
