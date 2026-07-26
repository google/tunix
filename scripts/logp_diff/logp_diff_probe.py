"""Remote probe: real Qwen3-32B rollout-vs-train per-token logp diff, DISAGGREGATED.

Runs on TPU (Phase 3b). Reuses the CPU-proven numpy harness (harness.py / sources.py).

WHY DISAGGREGATED (Phase 3b): deepswe RL is a two-mesh disaggregated setup
(train_deepswe_nb.py:648-654,894-897): ROLLOUT (vLLM) runs on rollout_mesh
(fsdp8xtp8, chips 0-63, vLLM data_parallel=8) and ACTOR (tunix) runs on train_mesh
(fsdp8xtp8, chips 64-127, real weight sharding) -- PHYSICALLY DISJOINT chips. The real
per-token diff RL pays (A-vs-C, deepswe measured mean 0.002 / max 0.92) is therefore
kernel + mesh + decode STACKED. A single-mesh probe structurally cannot see the mesh
term. This probe reproduces the true two-mesh topology so A-vs-C reproduces the real
diff AND decomposes it.

PATH FIDELITY (critical): the probe CALLS the exact functions RL training uses, so the
measured diff IS the training-time diff (not a reimplementation):
  A = vLLM-decode  : logprobs returned DURING generation, on rollout_mesh
                     == rollout old_logprobs (agentic_grpo_learner.py:542)
  C = tunix-forward: common.compute_per_token_logps(graphdef, state, ...), on train_mesh
                     == the trainer path (rl_cluster.py:1122 get_actor_per_token_logps)
  B = vLLM-forward : same tokens re-run through vLLM prefill (prompt_logprobs), rollout_mesh

Decomposition (one disaggregated run):
  A-vs-C = REAL total (reproduce 0.92)   A-vs-B = decode effect (clean, both @rollout)
  B-vs-C = kernel+mesh (both forward)    additivity: per-token A-C == (A-B)+(B-C) (align guard)
  C-vs-C2 (optional) = tunix's own fsdp-sharding sensitivity (bounds the mesh term)

CPU boundary (goal.md): py_compile + --dry_run + probe_wiring_test on CPU (jax/vllm/tunix lazy
or mocked). Real numerics need metrax+TPU. _REMOTE_VERIFY_ marks cluster-only checks.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import harness as H
import sources as S  # noqa: F401  (kept for Phase-2 toy compare reuse / activation attribution)


def _parse_args(argv=None):
  p = argparse.ArgumentParser(description="Qwen3-32B rollout-vs-train logp-diff probe (disaggregated)")
  p.add_argument("--model_path", default="/mnt/disks/linchai_data/models/Qwen3-32B")
  p.add_argument("--model_version", default="Qwen3-32B")
  p.add_argument("--dataset", default="SWE-bench/SWE-smith-trajectories")
  p.add_argument("--n_prompt", type=int, default=2048, help="prompt tokens taken from a real trajectory")
  p.add_argument("--n_gen", type=int, default=512, help="tokens to generate (defines A's decode logp span)")
  # Disaggregated meshes (mirror deepswe: rollout on devices[:R], train on devices[R:R+T]).
  p.add_argument("--rollout_mesh_fsdp", type=int, default=8, help="vLLM data_parallel replicas")
  p.add_argument("--rollout_mesh_tp", type=int, default=8, help="vLLM tensor_parallel")
  p.add_argument("--train_mesh_fsdp", type=int, default=8, help="tunix actor fsdp shard")
  p.add_argument("--train_mesh_tp", type=int, default=8, help="tunix actor tensor_parallel")
  p.add_argument("--temperature", type=float, default=1.0, help="MUST match rollout+trainer (deepswe=1.0)")
  # dtype fidelity (deepswe train_deepswe_nb.py:251-267 DEFAULTS -- yaml does NOT override):
  #   config_dtype is the COMPUTE dtype (weights downcast to it at matmul, qwen3/model.py:328) -> the
  #   real numerics lever = bfloat16. param_dtype is STORAGE only = float32 (minor effect).
  p.add_argument("--config_dtype", default="bfloat16", help="COMPUTE dtype (deepswe config.dtype=bfloat16) -- the numerics lever")
  p.add_argument("--param_dtype", default="float32", help="actor weight STORAGE dtype (deepswe --param_dtype=float32; downcast to config_dtype at matmul)")
  # vLLM engine fidelity (deepswe rollout_vllm_dict, train_deepswe_nb.py:851-872):
  p.add_argument("--vllm_hbm_util", type=float, default=0.6, help="deepswe --vllm_utilization=0.6")
  p.add_argument("--vllm_max_num_seqs", type=int, default=64, help="deepswe --rollout_vllm_max_num_seqs=64")
  p.add_argument("--vllm_max_num_batched_tokens", type=int, default=8192, help="deepswe --max_num_batched_tokens=8192")
  p.add_argument("--enable_prefix_caching", action="store_true", default=False,
                 help="deepswe = False (APC on triggers the known RoPE logp corruption; Phase3 Issue6)")
  p.add_argument("--pairs", default="A-vs-C,A-vs-B,B-vs-C", help="which source pairs to compare")
  p.add_argument("--mesh_sensitivity", action="store_true",
                 help="also run C2 = tunix @ (fsdp1 x train_tp) to bound the mesh/sharding term")
  p.add_argument("--out", default="gs://yuxzhang-tunix-models/logp-diff/report.json")
  p.add_argument("--dry_run", action="store_true")
  return p.parse_args(argv)


# ---------------------------------------------------------------- meshes (disaggregated)
def build_meshes(rollout_fsdp, rollout_tp, train_fsdp, train_tp):
  """Two PHYSICALLY DISJOINT meshes, mirroring train_deepswe_nb.py:648-654.

  rollout_mesh = devices[:R]  (R = rollout_fsdp*rollout_tp)   -> vLLM
  train_mesh   = devices[R:R+T]  (T = train_fsdp*train_tp)    -> tunix actor
  On the 256-chip deepswe slice with 8x8+8x8, R=64,T=64 -> 128 used, 128 idle (faithful to prod).
  Returns (rollout_mesh, train_mesh, devices_list) so a C2 sub-mesh can be carved from train chips.
  """
  import jax                                                    # lazy
  from jax.sharding import Mesh
  devices = list(jax.devices())
  R, T = rollout_fsdp * rollout_tp, train_fsdp * train_tp
  if R + T > len(devices):
    raise ValueError(f"need R({R})+T({T})={R+T} devices, have {len(devices)}")
  rollout_devices = np.asarray(devices[:R]).reshape(rollout_fsdp, rollout_tp)
  train_devices = np.asarray(devices[R:R + T]).reshape(train_fsdp, train_tp)
  rollout_mesh = Mesh(rollout_devices, ("fsdp", "tp"))
  train_mesh = Mesh(train_devices, ("fsdp", "tp"))
  idle = len(devices) - (R + T)
  print(f"[probe] devices={len(devices)} rollout_mesh={rollout_mesh.shape} "
        f"train_mesh={train_mesh.shape} idle={idle}  # _REMOTE_VERIFY_ disjoint + vLLM confinement")
  return rollout_mesh, train_mesh, devices


# ---------------------------------------------------------------- data
def load_prompt_tokens(dataset, n_prompt, model_path):
  from transformers import AutoTokenizer  # lazy
  tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
  if dataset.endswith(".txt"):
    text = open(dataset).read()
  else:
    from datasets import load_dataset  # lazy
    ds = load_dataset(dataset, streaming=True)
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    row = next(iter(ds[split_name]))
    text = row.get("text") or json.dumps(row.get("messages") or row)
  ids = tok(text, return_tensors=None)["input_ids"]
  ids = (ids * (n_prompt // max(1, len(ids)) + 1))[:n_prompt]
  return np.asarray(ids, np.int32), tok


# ---------------------------------------------------------------- vLLM (A + B, on rollout_mesh)
def run_vllm(model_path, rollout_mesh, prompt_ids, n_gen, temperature,
             hbm_util, max_num_seqs, max_num_batched_tokens, enable_prefix_caching):
  """Real rollout on rollout_mesh: generate n_gen tokens -> (full_tokens, A_decode_logp, sampler).

  Mirrors deepswe's vLLM rollout (train_deepswe_nb.py:851-872 rollout_vllm_dict):
    server_mode=True, async_scheduling=False, hbm_utilization=0.6, max_num_seqs=64,
    max_num_batched_tokens=8192, enable_prefix_caching=False, tpu_backend_type=jax.
  Confinement (VERIFIED in code): VllmConfig.mesh -> device_indexes = mesh.device_ids
  (vllm_sampler.py:295) -> vLLM is placed on EXACTLY rollout_mesh's chips (0-63), so it does
  NOT collide with tunix on train_mesh. deepswe wires this via rl_cluster role_to_mesh; the
  standalone probe passes mesh=rollout_mesh directly (same effect). _REMOTE_VERIFY_: confirm
  both engines coexist on the cluster.
  init_with_random_weights=False is the ONE deliberate deviation (deepswe=True + syncs from
  actor; the probe has no training loop so it loads real weights directly -- weight equality
  is the go/no-go pre-check).
  """
  from tunix.generate import vllm_sampler  # noqa: F401  (lazy; wired on remote)
  import numpy as np
  from transformers import AutoTokenizer

  rollout_tp = int(rollout_mesh.shape["tp"])
  rollout_dp = int(rollout_mesh.shape["fsdp"])   # vLLM interprets the rollout "fsdp" axis as data_parallel

  config = vllm_sampler.VllmConfig(
      server_mode=True,                # deepswe rollout_vllm_server_mode=True
      return_logprobs=True,
      init_with_random_weights=False,  # deliberate: probe loads real weights (no training loop to sync)
      tpu_backend_type="jax",
      mesh=rollout_mesh,
      hbm_utilization=hbm_util,        # deepswe 0.6
      tensor_parallel_size=rollout_tp,
      data_parallel_size=rollout_dp,
      engine_kwargs={
          "model": model_path,
          "trust_remote_code": True,
          "max_model_len": max(len(prompt_ids) + n_gen + 128, 8192),
          "max_num_seqs": max_num_seqs,                     # deepswe 64
          "max_num_batched_tokens": max_num_batched_tokens, # deepswe 8192
          "async_scheduling": False,                        # deepswe rollout_vllm_async_scheduling=False
          "enable_prefix_caching": enable_prefix_caching,   # deepswe False (APC-on = known RoPE logp corruption)
      }
  )

  tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
  sampler = vllm_sampler.VllmSampler(tokenizer=tok, config=config)

  # FIDELITY: feed the prompt as TOKENS, not decode->re-encode (BPE round-trip is not identity;
  # a text-in path would shift A's prefill vs C -> contaminate A-vs-C). Mirrors source B.
  from vllm.inputs import TokensPrompt
  from vllm.sampling_params import SamplingParams
  params = SamplingParams(temperature=temperature, max_tokens=n_gen, logprobs=1)
  tp = [TokensPrompt(prompt_token_ids=prompt_ids.tolist())]
  if sampler._driver is not None:
    outs = sampler._generate_server_mode(tp, params)
  else:
    outs = sampler.llm.generate(tp, sampling_params=params, use_tqdm=False)
  gen = outs[0].outputs[0]
  gen_token_ids = np.asarray(gen.token_ids, dtype=np.int32)
  # decode logp of each SAMPLED token == source A == training's old_logprobs
  a_decode_logp = np.asarray(
      [gen.logprobs[i][int(gen_token_ids[i])].logprob for i in range(len(gen_token_ids))],
      dtype=np.float32)
  full_tokens = np.concatenate([prompt_ids, gen_token_ids], axis=0)
  return full_tokens, a_decode_logp, sampler


def vllm_prefill_logp(sampler, full_tokens, n_completion):
  """Source B: same tokens re-run through vLLM prefill; prompt_logprobs -> per-token logp.

  ALIGNMENT: prompt_logprobs gives logp for ALL tokens; SLICE to the last ``n_completion``
  (the generated span) so B is on the SAME completion tokens as A and C.
  _REMOTE_VERIFY_: sampling_params.prompt_logprobs=1 (vllm_sampler.py:470); numerics (b/428730696).
  """
  from vllm.inputs import TokensPrompt
  from vllm.sampling_params import SamplingParams
  import numpy as np

  params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
  if sampler._driver is not None:
    outputs = sampler._generate_server_mode([TokensPrompt(prompt_token_ids=full_tokens.tolist())], params)
  else:
    outputs = sampler.llm.generate([TokensPrompt(prompt_token_ids=full_tokens.tolist())],
                                   sampling_params=params, use_tqdm=False)
  prompt_logprobs = outputs[0].prompt_logprobs
  res = []
  start_idx = len(full_tokens) - n_completion
  for i in range(start_idx, len(full_tokens)):
    token_id = full_tokens[i]
    res.append(prompt_logprobs[i][token_id].logprob)
  return np.array(res, dtype=np.float32)


# ---------------------------------------------------------------- tunix trainer forward (C / C2)
_DTYPE_MAP = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32"}  # str -> jnp attr


def tunix_forward_logp(model_path, model_version, mesh, prompt_ids, completion_ids, tok, temperature,
                       param_dtype="float32", config_dtype="bfloat16"):
  """Source C: the EXACT trainer path -- common.compute_per_token_logps on (graphdef, state).

  Mirrors rl_cluster.get_actor_per_token_logps (rl_cluster.py:1122): nnx.split the actor,
  call compute_per_token_logps with the same pad/eos/temperature. Fidelity by construction.
  `mesh` selects the sharding: train_mesh for C; a (fsdp1 x tp) sub-mesh for C2 (sharding sensitivity).

  DTYPE FIDELITY (deepswe train_deepswe_nb.py:588,677 + qwen3/model.py:151,328):
    - COMPUTE dtype = config.dtype. The forward DOWNCASTS weights to config.dtype before
      every matmul (model.py:328 `w = jnp.astype(self.w.value, self.dtype)`), so config.dtype
      is what actually sets the forward numerics. deepswe sets config.dtype=BFLOAT16 (train:588).
      call_model_config() defaults config.dtype to FLOAT32 (ModelConfig default) -- if left
      unset the actor would compute in fp32 (too precise vs deepswe's bf16), so we set bf16.
    - STORAGE dtype = param_dtype (create_model dtype arg) = float32 to match deepswe (:677);
      minor numerical effect since compute downcasts to config.dtype anyway.
  """
  import jax, jax.numpy as jnp                                   # lazy
  from flax import nnx
  from tunix.models.automodel import create_model_from_safe_tensors, call_model_config
  from tunix.rl import common

  cfg = call_model_config(model_version)   # authoritative (train_deepswe_nb.py:575)
  cfg.dtype = getattr(jnp, _DTYPE_MAP[config_dtype])          # activations/compute (deepswe --dtype=bfloat16)
  pdt = getattr(jnp, _DTYPE_MAP[param_dtype])                 # actor weights (deepswe --param_dtype=float32)
  model = create_model_from_safe_tensors(model_version, model_path, cfg, mesh, dtype=pdt)
  graphdef, state = nnx.split(model)                            # == trainer split
  lp = common.compute_per_token_logps(
      graphdef, state,
      prompt_tokens=jnp.asarray(prompt_ids)[None, :],
      completion_tokens=jnp.asarray(completion_ids)[None, :],
      pad_id=tok.pad_token_id if tok.pad_token_id is not None else 0,
      eos_id=tok.eos_token_id,
      stop_gradient=True, return_logits=False, temperature=temperature,
  )
  return np.asarray(jax.device_get(lp)).reshape(-1)


# ---------------------------------------------------------------- decomposition
def decompose(logps):
  """Pairwise stats + per-token additivity guard. logps: {name: per-token logp array}.

  additivity: per token, (A-C) == (A-B)+(B-C) EXACTLY -> residual>~1e-5 means the three
  sources are NOT aligned to the same completion span (a wiring bug); a loud guard for the
  expensive run. Not a scientific result, an alignment check.
  """
  out = {}
  if all(k in logps for k in ("A", "B", "C")):
    a, b, c = (np.asarray(logps[k], np.float64) for k in ("A", "B", "C"))
    resid = float(np.abs((a - c) - ((a - b) + (b - c))).max()) if a.size else 0.0
    out["real_total(A-vs-C)"] = H.logp_diff_stats(a, c)
    out["decode(A-vs-B)"] = H.logp_diff_stats(a, b)
    out["kernel+mesh(B-vs-C)"] = H.logp_diff_stats(b, c)
    out["additivity_residual_max"] = resid
    out["lengths"] = {k: int(np.asarray(v).size) for k, v in logps.items()}
    out["note"] = ("per-token A-C==(A-B)+(B-C) exact; residual>1e-5 => A/B/C misaligned. "
                   "kernel+mesh is entangled (cross-engine); C-vs-C2 bounds the sharding part.")
  if "C2" in logps and "C" in logps:
    out["tunix_sharding_sensitivity(C-vs-C2)"] = H.logp_diff_stats(logps["C"], logps["C2"])
  return out


# ---------------------------------------------------------------- main
def main(argv=None):
  args = _parse_args(argv)
  plan = {"model": args.model_version, "n_prompt": args.n_prompt, "n_gen": args.n_gen,
          "rollout_mesh": f"fsdp{args.rollout_mesh_fsdp}xtp{args.rollout_mesh_tp}",
          "train_mesh": f"fsdp{args.train_mesh_fsdp}xtp{args.train_mesh_tp}",
          "temperature": args.temperature, "pairs": args.pairs,
          "mesh_sensitivity": args.mesh_sensitivity,
          "actor_param_dtype": args.param_dtype, "config_dtype": args.config_dtype,
          "vllm": {"hbm_util": args.vllm_hbm_util, "max_num_seqs": args.vllm_max_num_seqs,
                   "max_num_batched_tokens": args.vllm_max_num_batched_tokens,
                   "enable_prefix_caching": args.enable_prefix_caching, "server_mode": True},
          "flow": "generate->A(decode@rollout); B(prefill@rollout)+C(compute_per_token_logps@train) on same tokens"}
  print("[probe] plan:", json.dumps(plan))
  if args.dry_run:
    print("[probe] --dry_run OK (CPU boundary: jax/vllm/tunix not imported).")
    return plan

  rollout_mesh, train_mesh, devices = build_meshes(
      args.rollout_mesh_fsdp, args.rollout_mesh_tp, args.train_mesh_fsdp, args.train_mesh_tp)
  R = args.rollout_mesh_fsdp * args.rollout_mesh_tp

  prompt_ids, tok = load_prompt_tokens(args.dataset, args.n_prompt, args.model_path)
  # A: real rollout generate on rollout_mesh -> full tokens + decode logp
  full_tokens, a_decode_logp, sampler = run_vllm(
      args.model_path, rollout_mesh, prompt_ids, args.n_gen, args.temperature,
      args.vllm_hbm_util, args.vllm_max_num_seqs, args.vllm_max_num_batched_tokens,
      args.enable_prefix_caching)
  completion_ids = full_tokens[len(prompt_ids):]
  ncomp = len(completion_ids)

  # Evaluate each source's completion-span logp ONCE (cache) -- the real vLLM prefill /
  # tunix forward are expensive; the old compare() called each up to 3x per pair.
  logps = {
      "A": np.asarray(a_decode_logp, np.float32),
      "B": vllm_prefill_logp(sampler, full_tokens, ncomp),
      "C": tunix_forward_logp(args.model_path, args.model_version, train_mesh,
                              prompt_ids, completion_ids, tok, args.temperature,
                              args.param_dtype, args.config_dtype),
  }
  if args.mesh_sensitivity:
    from jax.sharding import Mesh
    train_tp = args.train_mesh_tp
    c2_mesh = Mesh(np.asarray(devices[R:R + train_tp]).reshape(1, train_tp), ("fsdp", "tp"))
    logps["C2"] = tunix_forward_logp(args.model_path, args.model_version, c2_mesh,
                                     prompt_ids, completion_ids, tok, args.temperature,
                                     args.param_dtype, args.config_dtype)

  # Length guard (loud): all sources must be on the same ncomp completion tokens.
  bad = {k: int(np.asarray(v).size) for k, v in logps.items() if np.asarray(v).size != ncomp}
  if bad:
    print(f"[probe] !! LENGTH MISMATCH (expected {ncomp}): {bad}  # _REMOTE_VERIFY_ alignment bug")

  report = {"plan": plan, "comparisons": {}, "decomposition": {}}
  for pair in args.pairs.split(","):
    x, y = pair.split("-vs-")
    report["comparisons"][pair] = H.logp_diff_stats(logps[x], logps[y])
  report["decomposition"] = decompose(logps)
  print("[probe] report:", json.dumps(report, indent=2, default=float))

  # _REMOTE_VERIFY_: write report to args.out (GCS needs gcsfs, TPU-only; local path on CPU).
  if args.out.startswith("gs://"):
    from etils import epath
    epath.Path(args.out).write_text(json.dumps(report, indent=2, default=float))
  else:
    with open(args.out, "w") as f:
      json.dump(report, f, indent=2, default=float)
  return report


if __name__ == "__main__":
  main()
