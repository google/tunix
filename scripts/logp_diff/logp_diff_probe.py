"""Remote probe: real Qwen3-32B rollout-vs-train per-token logp diff + op attribution.

Runs on TPU (Phase 3). Reuses the CPU-proven numpy harness (harness.py / sources.py).

PATH FIDELITY (critical): the probe CALLS the exact functions RL training uses, so the
measured diff IS the training-time diff (not a reimplementation):
  A = vLLM-decode  : logprobs returned DURING generation  == rollout's old_logprobs
                     (what training uses as rollout_per_token_logps; agentic_grpo_learner.py:542)
  C = tunix-forward: common.compute_per_token_logps(graphdef, state, prompt, completion, pad, eos, T)
                     == the trainer path (rl_cluster.py:1122 get_actor_per_token_logps)
  B = vLLM-forward : same tokens re-run through vLLM prefill (prompt_logprobs) -- kernel-isolation diagnostic

Faithful flow = generate once -> A(decode logp); then B & C on the SAME generated tokens.
  A vs C = the REAL training diff ; B vs C = pure kernel ; A vs B = decode-vs-forward effect.

CPU boundary (goal.md): py_compile + --help + --dry_run on CPU (jax/vllm/tunix lazy).
Real numerics need metrax+TPU; fidelity is guaranteed by CALLING the real functions.
_REMOTE_VERIFY_ marks the parts only checkable on the cluster.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

import harness as H
import sources as S


def _parse_args(argv=None):
  p = argparse.ArgumentParser(description="Qwen3-32B rollout-vs-train logp-diff probe")
  p.add_argument("--model_path", default="/mnt/disks/linchai_data/models/Qwen3-32B")
  p.add_argument("--model_version", default="Qwen3-32B")
  p.add_argument("--dataset", default="SWE-bench/SWE-smith-trajectories")
  p.add_argument("--n_prompt", type=int, default=2048, help="prompt tokens taken from a real trajectory")
  p.add_argument("--n_gen", type=int, default=512, help="tokens to generate (defines A's decode logp span)")
  p.add_argument("--mesh_tp", type=int, default=8)
  p.add_argument("--mesh_fsdp", type=int, default=1)
  p.add_argument("--temperature", type=float, default=1.0, help="MUST match rollout+trainer (deepswe=1.0)")
  p.add_argument("--pairs", default="A-vs-C,B-vs-C,A-vs-B", help="which source pairs to compare")
  p.add_argument("--out", default="gs://yuxzhang-tunix-models/logp-diff/report.json")
  p.add_argument("--dry_run", action="store_true")
  return p.parse_args(argv)


# ---------------------------------------------------------------- data
def load_prompt_tokens(dataset, n_prompt, model_path):
  from transformers import AutoTokenizer  # lazy
  tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
  if dataset.endswith(".txt"):
    text = open(dataset).read()
  else:
    from datasets import load_dataset  # lazy
    row = next(iter(load_dataset(dataset, split="train", streaming=True)))
    text = row.get("text") or json.dumps(row.get("messages") or row)
  ids = tok(text, return_tensors=None)["input_ids"]
  ids = (ids * (n_prompt // max(1, len(ids)) + 1))[:n_prompt]
  return np.asarray(ids, np.int32), tok


# ---------------------------------------------------------------- vLLM (A + B, generate + prefill)
def run_vllm(model_path, mesh_tp, mesh_fsdp, prompt_ids, n_gen, temperature):
  """Real rollout: generate n_gen tokens (return_logprobs) -> (full_tokens, A_decode_logp).

  _REMOTE_VERIFY_: construct VllmSampler(return_logprobs=True); generate from prompt_ids;
  the per-token logprobs of the GENERATED tokens == source A (rollout decode logp),
  the exact array training stores as trajectory 'old_logprobs' (agentic_grpo_learner.py:542).
  """
  from tunix.generate import vllm_sampler  # noqa: F401  (lazy; wired on remote)
  import jax
  import numpy as np
  from jax.sharding import Mesh
  from transformers import AutoTokenizer

  # Use the first tp*fsdp devices -> works on a dedicated small slice (exactly tp*fsdp
  # chips) AND on a larger atomic slice (e.g. the 256-chip deepswe cluster: use 8, rest idle).
  mesh = Mesh(np.asarray(jax.devices()[: mesh_fsdp * mesh_tp]).reshape(mesh_fsdp, mesh_tp), ("fsdp", "tp"))

  config = vllm_sampler.VllmConfig(
      return_logprobs=True,
      init_with_random_weights=False,  # CRITICAL: real weights
      tpu_backend_type="jax",
      mesh=mesh,
      tensor_parallel_size=mesh_tp,
      data_parallel_size=mesh_fsdp,
      engine_kwargs={
          "model": model_path,
          "trust_remote_code": True,
          "max_model_len": max(8192, len(prompt_ids) + n_gen + 128),
      }
  )

  tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
  sampler = vllm_sampler.VllmSampler(tokenizer=tok, config=config)

  # FIDELITY: feed the prompt as TOKENS, not decode->re-encode. Text-in
  # (sampler(input_strings=[tok.decode(prompt_ids)])) re-tokenizes internally
  # (vllm_sampler.py:519) and the BPE round-trip is NOT identity, so the prompt
  # vLLM actually prefilled could differ from prompt_ids -> A's generation context
  # would mismatch C's -> A-vs-C contaminated by a prompt-token shift, not just
  # kernel/decode. Use the token-in engine path (same as source B).
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

  ALIGNMENT (must match A and C): prompt_logprobs gives logp for ALL tokens; SLICE to the
  last ``n_completion`` (the generated span) so B is on the SAME completion tokens as
  A (decode logp of generated tokens) and C (compute_per_token_logps -> completion-only).
  TEMPERATURE: A was sampled at T and C applies temperature=T; ensure B's prompt_logprobs
  use the same convention (deepswe T=1.0 -> no scaling; verify for T!=1).
  _REMOTE_VERIFY_: sampling_params.prompt_logprobs=1 (vllm_sampler.py:470); extract & align (b/428730696).
  """
  from vllm.inputs import TokensPrompt
  from vllm.sampling_params import SamplingParams
  import numpy as np

  # Source B prefill logp
  params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
  
  if sampler._driver is not None:
    outputs = sampler._generate_server_mode([TokensPrompt(prompt_token_ids=full_tokens.tolist())], params)
  else:
    outputs = sampler.llm.generate([TokensPrompt(prompt_token_ids=full_tokens.tolist())], sampling_params=params, use_tqdm=False)

  out = outputs[0]
  prompt_logprobs = out.prompt_logprobs
  
  res = []
  start_idx = len(full_tokens) - n_completion
  for i in range(start_idx, len(full_tokens)):
    token_id = full_tokens[i]
    logp = prompt_logprobs[i][token_id].logprob
    res.append(logp)
  
  return np.array(res, dtype=np.float32)


# ---------------------------------------------------------------- tunix trainer forward (C)
def tunix_forward_logp(model_path, model_version, mesh, prompt_ids, completion_ids, tok, temperature):
  """Source C: the EXACT trainer path -- common.compute_per_token_logps on (graphdef, state).

  Mirrors rl_cluster.get_actor_per_token_logps (rl_cluster.py:1122): nnx.split the actor,
  call compute_per_token_logps with the same pad/eos/temperature. Fidelity by construction.
  """
  import jax, jax.numpy as jnp                                   # lazy
  from flax import nnx
  from tunix.models.automodel import create_model_from_safe_tensors, call_model_config
  from tunix.rl import common

  # Authoritative config lookup (train_deepswe_nb.py:575: config = call_model_config(MODEL_VERSION)).
  cfg = call_model_config(model_version)
  model = create_model_from_safe_tensors(model_version, model_path, cfg, mesh, dtype=jnp.bfloat16)
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


# ---------------------------------------------------------------- main
def main(argv=None):
  args = _parse_args(argv)
  plan = {"model": args.model_version, "n_prompt": args.n_prompt, "n_gen": args.n_gen,
          "mesh": f"tp{args.mesh_tp}xfsdp{args.mesh_fsdp}", "temperature": args.temperature,
          "pairs": args.pairs, "flow": "generate->A(decode); B(prefill)+C(compute_per_token_logps) on same tokens"}
  print("[probe] plan:", json.dumps(plan))
  if args.dry_run:
    print("[probe] --dry_run OK (CPU boundary: jax/vllm/tunix not imported).")
    return plan

  import jax                                                    # lazy
  from jax.sharding import Mesh
  n = args.mesh_fsdp * args.mesh_tp   # use first n devices (small dedicated slice OR subset of 256)
  mesh = Mesh(np.asarray(jax.devices()[:n]).reshape(args.mesh_fsdp, args.mesh_tp), ("fsdp", "tp"))
  print(f"[probe] devices={len(jax.devices())} using={n} mesh={mesh.shape}  # _REMOTE_VERIFY_ idle-chip check")

  prompt_ids, tok = load_prompt_tokens(args.dataset, args.n_prompt, args.model_path)
  # A: real rollout generate -> full tokens + decode logp
  full_tokens, a_decode_logp, sampler = run_vllm(
      args.model_path, args.mesh_tp, args.mesh_fsdp, prompt_ids, args.n_gen, args.temperature)
  completion_ids = full_tokens[len(prompt_ids):]

  # All three sources are aligned to the SAME completion span (the n_gen generated tokens):
  #   A = decode logp of generated tokens; C = compute_per_token_logps -> completion-only;
  #   B = prefill logp sliced to the last len(completion_ids).
  srcA = S.Source("A(vllm-decode)", get_logp=lambda t: a_decode_logp)
  srcB = S.Source("B(vllm-fwd)", get_logp=lambda t: vllm_prefill_logp(sampler, full_tokens, len(completion_ids)))
  srcC = S.Source("C(tunix-fwd)",
                  get_logp=lambda t: tunix_forward_logp(args.model_path, args.model_version, mesh,
                                                        prompt_ids, completion_ids, tok, args.temperature))
  reg = {"A": srcA, "B": srcB, "C": srcC}
  report = {"plan": plan, "comparisons": {}}
  for pair in args.pairs.split(","):
    x, y = pair.split("-vs-")
    report["comparisons"][pair] = S.compare(reg[x], reg[y], full_tokens, atol=1e-6)
  print("[probe] report:", json.dumps(report, indent=2, default=float))
  # _REMOTE_VERIFY_: write report to args.out (GCS).
  if args.out.startswith("gs://"):
    from etils import epath
    epath.Path(args.out).write_text(json.dumps(report, indent=2, default=float))
  return report


if __name__ == "__main__":
  main()
