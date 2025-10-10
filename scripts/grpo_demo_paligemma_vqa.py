import os, io, json, time, argparse, math, random, re, dataclasses
import sys
sys.path.insert(0, "/home/grads/tianjiao/tunix")

from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm

from orbax import checkpoint as ocp

# --- Tunix bits ---
# Vision encoder
from tunix.models.siglip import model as siglip_model
from tunix.models.siglip import params as siglip_params

# PaLI-Gemma (vision+projector+text)
from tunix.models.paligemma import model as pali_model   # your PaLI-Gemma wrapper
# (no params module required if you're composing from components)

# Text backbone config (Gemma3 here; switch to Gemma2 if that’s what you saved!)
from tunix.models.gemma3 import model as gemma_model

# Tokenizer
from tunix.models.gemma import data as gemma_tokenizer_lib
from tunix.generate import tokenizer_adapter as tok_adapt
# VLM sampler/rollout/GRPO
from tunix.generate.vlm_sampler import VLMSampler
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner

from tunix.sft import metrics_logger


# ----------------------
# Image preprocessing
# ----------------------
def load_and_preprocess_image(path: str, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    # Return raw uint8 array, no normalization
    x = np.array(img, dtype=np.uint8)
    return x


# ----------------------
# Simple VQA dataset
# ----------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def make_vqa_iter(jsonl_path: str, image_size: int, batch_size: int):
    """Yields batches: dict(prompts, images, answer, question)."""
    data = read_jsonl(jsonl_path)
    random.Random(42).shuffle(data)

    def prompt_fmt(q: str) -> str:
        return (
            "<start_of_turn>user\n"
            "You are a helpful visual assistant. Answer the question based on the image.\n\n"
            f"Question: {q}\n"
            "<end_of_turn>\n"
            "<start_of_turn>model"
        )

    batch = {"prompts": [], "images": [], "answer": [], "question": []}
    for ex in data:
        try:
            img = load_and_preprocess_image(ex["image"], image_size)
        except Exception:
            continue
        batch["prompts"].append(prompt_fmt(ex["question"]))
        batch["images"].append(img)
        batch["answer"].append(ex.get("answer", ""))
        batch["question"].append(ex["question"])
        if len(batch["prompts"]) == batch_size:
            yield {
                "prompts": batch["prompts"],
                "images": np.stack(batch["images"], axis=0).astype(np.uint8),
                "answer": batch["answer"],
                "question": batch["question"],
            }
            batch = {"prompts": [], "images": [], "answer": [], "question": []}
    if batch["prompts"]:
        yield {
            "prompts": batch["prompts"],
            "images": np.stack(batch["images"], axis=0).astype(np.uint8),
            "answer": batch["answer"],
            "question": batch["question"],
        }


# ----------------------
# Rewards (simple)
# ----------------------
REASON_S = "<reasoning>"
REASON_E = "</reasoning>"
ANS_S = "<answer>"
ANS_E = "</answer>"

FORMAT_RE = re.compile(
    rf"^\s*{REASON_S}.+?{REASON_E}.*?{ANS_S}(.+?){ANS_E}\s*$",
    flags=re.MULTILINE | re.DOTALL,
)
NUM_RE = re.compile(rf"{ANS_S}.*?([\d\.]+)")

def reward_format_exact(prompts, completions, **k):
    return [3.0 if FORMAT_RE.search(c) else 0.0 for c in completions]

def reward_format_soft(prompts, completions, **k):
    scores = []
    for c in completions:
        sc = 0.0
        sc += 0.5 if c.count(REASON_S) == 1 else -0.5
        sc += 0.5 if c.count(REASON_E) == 1 else -0.5
        sc += 0.5 if c.count(ANS_S) == 1 else -0.5
        sc += 0.5 if c.count(ANS_E) == 1 else -0.5
        scores.append(sc)
    return scores

def reward_answer(prompts, completions, answer, **k):
    scores = []
    for c, a in zip(completions, answer):
        s = 0.0
        m = FORMAT_RE.search(c)
        if m:
            guess = m.group(1)
            try:
                g = float(guess.strip())
                t = float(str(a).strip())
                if g == t:
                    s += 3.0
                else:
                    ratio = g / max(t, 1e-6)
                    if 0.9 <= ratio <= 1.1:
                        s += 0.5
                    elif 0.8 <= ratio <= 1.2:
                        s += 0.25
                    else:
                        s -= 1.0
            except:
                s -= 0.5
        scores.append(s)
    return scores

def reward_number(prompts, completions, answer, **k):
    scores = []
    for c, a in zip(completions, answer):
        try:
            g = NUM_RE.search(c)
            if not g:
                scores.append(0.0); continue
            gg = float(g.group(1).strip())
            tt = float(str(a).strip())
            scores.append(1.5 if gg == tt else 0.0)
        except:
            scores.append(0.0)
    return scores


# ----------------------
# SigLIP loader
# ----------------------
def ensure_cfg_divisible(cfg):
    rem = cfg.image_size % cfg.patch_size
    if rem != 0:
        cfg = dataclasses.replace(cfg, image_size=cfg.image_size - rem)
    return cfg

def load_siglip(siglip_dir: str, forced_cfg: str | None, mesh):
    print("Loading SigLIP encoder...")
    if forced_cfg:
        cfg = getattr(siglip_model.SigLIPConfig, forced_cfg)()
    else:
        print("Inferring SigLIP config from folder...")
        cfg = None
    if cfg is None:
        cfg = siglip_model.SigLIPConfig.so400m_patch14_384()
    cfg = ensure_cfg_divisible(cfg)

    enc = siglip_params.create_model_from_safe_tensors(siglip_dir, cfg, mesh)
    # Warmup-compile
    dummy = enc.get_model_input()
    _ = nnx.jit(enc)(**dummy)
    return enc, cfg


# ----------------------
# PaLI-Gemma helpers: config + state copy
# ----------------------
def build_pali_cfg() -> pali_model.PaLIGemmaConfig:
    """Compose PaLI-Gemma config from your available factories."""
    sig = siglip_model.SigLIPConfig.so400m_patch14_384()
    # IMPORTANT: pick the Gemma *family* matching the checkpoint you saved.
    # If you saved Gemma2, switch to the Gemma2 config/factory instead.
    txt = gemma_model.Gemma3Config.gemma3_4b()
    return pali_model.PaLIGemmaConfig(vision=sig, text=txt).with_sharding()

def _path_to_tuple(path):
    """Serialize JAX KeyPath to a stable tuple so we can match leaves by path."""
    out = []
    for k in path:
        # JAX currently uses GetAttrKey(name), DictKey(key), SequenceKey(idx)
        # but we guard generically.
        if hasattr(k, "name"):        # GetAttrKey
            out.append(("attr", k.name))
        elif hasattr(k, "key"):       # DictKey
            out.append(("dict", k.key))
        elif hasattr(k, "idx"):       # SequenceKey
            out.append(("seq", k.idx))
        else:
            out.append((type(k).__name__, None))
    return tuple(out)

def _copy_matching_tensors(src_tree, dst_tree):
    """
    Copy leaves from src_tree into dst_tree when (path, shape, dtype) match.
    Returns (new_dst_tree, {'copied': int, 'skipped': int}).
    """
    # 1) Build a map from serialized path -> src leaf
    src_map = {}
    def _collect_src(path, x):
        src_map[_path_to_tuple(path)] = x
        return x  # value is ignored
    jax.tree.map_with_path(_collect_src, src_tree)

    # 2) Replace leaves in dst if we find a compatible src leaf at same path
    copied = 0
    skipped = 0

    def _maybe_replace(path, a):
        nonlocal copied, skipped
        key = _path_to_tuple(path)
        b = src_map.get(key, None)
        if b is not None and hasattr(a, "shape") and hasattr(b, "shape"):
            if a.shape == b.shape and getattr(a, "dtype", None) == getattr(b, "dtype", None):
                copied += 1
                return b
        skipped += 1
        return a

    new_dst = jax.tree.map_with_path(_maybe_replace, dst_tree)
    return new_dst, {"copied": copied, "skipped": skipped}

def _update_module_state(module, new_state_tree):
    nnx.update(module, new_state_tree)


# ----------------------
# PaLI-Gemma loaders (two paths)
# ----------------------
def build_pali_from_components(siglip_dir: str, gemma_ckpt_dir: str, mesh):
    print("Building PaLI-Gemma from components (SigLIP + Gemma ckpt)…")

    # 1) Load SigLIP encoder (real arrays) + config
    siglip, vision_cfg = load_siglip(siglip_dir, forced_cfg=None, mesh=mesh)

    # 2) Build a REAL PaLI-Gemma (not eval_shape!)
    text_cfg = gemma_model.Gemma3Config.gemma3_4b()  # pick your factory
    pali_cfg = pali_model.PaLIGemmaConfig.from_components(vision_cfg, text_cfg, max_vision_tokens=256)
    pali = pali_model.PaLIGemma(pali_cfg, rngs=nnx.Rngs(params=0))

    # 3) Copy SigLIP -> pali.vision
    print("Loading SigLIP encoder...")
    sig_state = nnx.state(siglip)
    pali_vis_state = nnx.state(pali.vision)
    new_vis_state, stats_v = _copy_matching_tensors(sig_state, pali_vis_state)
    nnx.update(pali.vision, new_vis_state)
    print(f"[PaLI] Vision copy: copied={stats_v['copied']} skipped={stats_v['skipped']}")

    # 4) Restore Gemma text weights (REAL arrays) and copy -> pali.text
    #    (Your Gemma NNX checkpoint dir probably has a 'state' subdir)
    ckpt_path_try = os.path.join(gemma_ckpt_dir, "state")
    ckpt_path = ckpt_path_try if os.path.exists(ckpt_path_try) else gemma_ckpt_dir

    # Build a REAL Gemma model to use as a target for restore
    ref_text = gemma_model.Gemma3(text_cfg, rngs=nnx.Rngs(params=0))
    ref_text_state = nnx.state(ref_text)

    ckptr = ocp.StandardCheckpointer()
    try:
        restored_text_state = ckptr.restore(ckpt_path, target=ref_text_state)
    except Exception:
        # Fall back to unsafe restore only if necessary (not ideal)
        print("[WARN] Gemma restore with target failed; attempting without target…")
        restored_text_state = ckptr.restore(ckpt_path)

    pali_text_state = nnx.state(pali.text)
    new_text_state, stats_t = _copy_matching_tensors(restored_text_state, pali_text_state)
    nnx.update(pali.text, new_text_state)
    print(f"[PaLI] Text copy:   copied={stats_t['copied']} skipped={stats_t['skipped']}")

    # 5) Shard PaLI state
    with mesh:
        st = nnx.state(pali)
        ps = nnx.get_partition_spec(st)
        nnx.update(pali, jax.lax.with_sharding_constraint(st, ps))

    return pali, pali_cfg

def restore_pali_ckpt(pali_ckpt: str, cfg: pali_model.PaLIGemmaConfig):
    """Restore a full PaLI-Gemma from its own checkpoint directory."""
    abs_pali = nnx.eval_shape(lambda: pali_model.PaLIGemma(cfg, rngs=nnx.Rngs(params=0)))
    abs_state = nnx.state(abs_pali)
    abs_state_struct = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), abs_state)
    restored = ocp.StandardCheckpointer().restore(pali_ckpt, target=abs_state_struct)
    pali = nnx.merge(nnx.split(abs_pali)[0], restored)
    return pali


# ----------------------
# LoRA application (via Qwix)
# ----------------------
def apply_lora(base_model, mesh, rank=16, alpha=16.0):
    try:
        import qwix
    except Exception:
        raise RuntimeError("Qwix not found: please install or plug your LoRA applier.")
    lora_provider = qwix.LoraProvider(
        module_path=(".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
                     ".*gate_proj|.*down_proj|.*up_proj"),
        rank=rank,
        alpha=alpha,
    )
    model_input = base_model.get_model_input()
    lora_model = qwix.apply_lora_to_model(base_model, lora_provider, **model_input)
    with mesh:
        state = nnx.state(lora_model)
        pspecs = nnx.get_partition_spec(state)
        sharded = jax.lax.with_sharding_constraint(state, pspecs)
        nnx.update(lora_model, sharded)
    return lora_model


# ----------------------
# Evaluation (simple)
# ----------------------
def make_vlm_sampler(policy_model, tokenizer, image_size: int):
    return VLMSampler(
        transformer=policy_model,
        tokenizer=tokenizer,
        image_size=image_size,
    )

def generate_batch(prompts, images, sampler, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    out = sampler(
        input_strings=prompts,
        images=images,
        max_generation_steps=256,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        echo=False,
        seed=seed,
    )
    return out.text

def quick_eval(dataset_iter, sampler, max_batches=10):
    total, corr, fmt = 0, 0, 0
    for i, batch in enumerate(dataset_iter):
        if i >= max_batches: break
        outs = generate_batch(batch["prompts"], batch["images"], sampler, temperature=1e-4, top_k=1, top_p=1.0)
        for pred, ans in zip(outs, batch["answer"]):
            if FORMAT_RE.search(pred): fmt += 1
            m = FORMAT_RE.search(pred)
            if m:
                try:
                    if float(m.group(1).strip()) == float(str(ans).strip()):
                        corr += 1
                except:
                    pass
            total += 1
    acc = (corr / total * 100) if total else 0.0
    fmt_acc = (fmt / total * 100) if total else 0.0
    return dict(total=total, exact_acc=acc, format_acc=fmt_acc)


# ----------------------
# Main
# ----------------------
def main():
    p = argparse.ArgumentParser()

    # Paths (choose one path below)
    p.add_argument("--siglip_dir", default="/home/grads/tianjiao/checkpoints/siglip-so400m-patch14-384",
                   help="Folder with SigLIP safetensors (required for Path A)")
    p.add_argument("--gemma_ckpt", default="/home/grads/tianjiao/checkpoints/gemma_ckpts/intermediate/state",
                   help="Gemma-only NNX checkpoint dir (Path A)")
    p.add_argument("--pali_ckpt", default="", help="PaLI-Gemma checkpoint dir to restore (Path B)")
    p.add_argument("--pali_ckpt_out", default="", help="(Optional) Where to save a fresh PaLI-Gemma checkpoint after Path A")

    # Data
    p.add_argument("--train_jsonl", default="/home/grads/tianjiao/tunix/scripts/dummy_vqa/train.jsonl",
                   help="VQA JSONL for training")
    p.add_argument("--eval_jsonl", default="/home/grads/tianjiao/tunix/scripts/dummy_vqa/eval.jsonl",
                   help="VQA JSONL for eval")

    # Mesh / batch
    p.add_argument("--mesh_fsdp", type=int, default=1)
    p.add_argument("--mesh_tp", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=float, default=16.0)

    # GRPO
    p.add_argument("--num_generations", type=int, default=2)
    p.add_argument("--num_iterations", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.04)
    p.add_argument("--epsilon", type=float, default=0.2)

    # Generation during rollout
    p.add_argument("--max_prompt_len", type=int, default=256)
    p.add_argument("--max_gen_steps", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=50)

    # Train steps / logging / ckpt
    p.add_argument("--max_steps", type=int, default=100)  # demo-short
    p.add_argument("--eval_every_n", type=int, default=20)
    p.add_argument("--ckpt_dir", type=str, default="./ckpts_vlm")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--tb_dir", type=str, default="./tb_vlm")

    args = p.parse_args()

    # Devices & mesh
    local_devs = jax.local_devices()
    print(f"Using {len(local_devs)} device(s).")
    mesh = jax.make_mesh((args.mesh_fsdp, args.mesh_tp), ("fsdp", "tp"))

    # --- Load SigLIP (always needed for sampler + Path A init) ---
    siglip, sigcfg = load_siglip(args.siglip_dir, forced_cfg=None, mesh=mesh)

    # --- Build or Restore PaLI-Gemma ---
    if args.pali_ckpt and os.path.exists(args.pali_ckpt):
        print(f"Restoring PaLI-Gemma from: {args.pali_ckpt}")
        cfg = build_pali_cfg()
        ref_model = restore_pali_ckpt(args.pali_ckpt, cfg)
    else:
        print("Building PaLI-Gemma from components (SigLIP + Gemma ckpt)…")
        ref_model, cfg = build_pali_from_components(args.siglip_dir, args.gemma_ckpt, mesh)
        if args.pali_ckpt_out:
            os.makedirs(args.pali_ckpt_out, exist_ok=True)
            _, state = nnx.split(ref_model)
            ocp.StandardCheckpointer().save(args.pali_ckpt_out, state, force=True)
            print(f"[Saved] Initial PaLI-Gemma checkpoint -> {args.pali_ckpt_out}")

    # --- Apply LoRA to build policy model ---
    policy_model = apply_lora(ref_model, mesh, rank=args.lora_rank, alpha=args.lora_alpha)

    # Tokenizer
    tokenizer = tok_adapt.TokenizerAdapter(gemma_tokenizer_lib.GemmaTokenizer())


    # Sampler for eval
    sampler = make_vlm_sampler(
        policy_model,
        tokenizer,
        sigcfg.image_size,
    )
    # --- Datasets ---
    train_iter = make_vqa_iter(args.train_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size)
    eval_iter_for_eval = lambda: make_vqa_iter(args.eval_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size)

    # --- Pre-train eval ---
    print("Running quick pre-train eval (greedy, few batches)…")
    pre_metrics = quick_eval(eval_iter_for_eval(), sampler, max_batches=5)
    print(f"[Pre] total={pre_metrics['total']}, "
          f"exact_acc={pre_metrics['exact_acc']:.2f}%, "
          f"format_acc={pre_metrics['format_acc']:.2f}%")

    # --- Optimizer & cluster config ---
    lr = 3e-6
    warmup = max(1, int(0.1 * args.max_steps))
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=lr, warmup_steps=warmup,
        decay_steps=args.max_steps, end_value=0.0
    )
    optimizer = optax.chain(optax.clip_by_global_norm(0.1),
                            optax.adamw(schedule, b1=0.9, b2=0.99, weight_decay=0.1))

    ckpt_opts = ocp.CheckpointManagerOptions(save_interval_steps=args.save_every, max_to_keep=4)
    log_opts = metrics_logger.MetricsLoggerOptions(log_dir=args.tb_dir, flush_every_n_steps=20)

    cluster_cfg = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vlm",          # ensure your rollout name matches registration
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=args.eval_every_n,
            max_steps=args.max_steps,
            gradient_accumulation_steps=1,
            metrics_logging_options=log_opts,
            checkpoint_root_directory=args.ckpt_dir,
            checkpointing_options=ckpt_opts,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=args.max_gen_steps,
            max_prompt_length=args.max_prompt_len,
            kv_cache_size=args.max_prompt_len + args.max_gen_steps + 128,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        ),
    )

    # --- RL cluster & learner ---
    rl_cluster = rl_cluster_lib.RLCluster(
        actor=policy_model,
        reference=ref_model,
        tokenizer=tokenizer,
        vision_encoder=siglip,  # if your RLCluster uses it
        cluster_config=cluster_cfg,
    )

    grpo_conf = GrpoConfig(
        num_generations=args.num_generations,
        num_iterations=args.num_iterations,
        beta=args.beta,
        epsilon=args.epsilon,
    )

    learner = GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=[reward_format_exact, reward_format_soft, reward_answer, reward_number],
        grpo_config=grpo_conf,
    )

    # --- Train ---
    print("Starting GRPO training…")
    with mesh:
        def repeat_train():
            while True:
                for b in make_vqa_iter(args.train_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size):
                    yield b
        learner.train(train_ds=repeat_train(), eval_ds=None, skip_jit=False)

    # --- Savepoint info: LoRA-only restore test ---
    print("Training done. Attempting quick LoRA-only restore test…")
    last_step = args.max_steps
    trained_ckpt_path = os.path.join(args.ckpt_dir, str(last_step), "model_params")
    abs_lora = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                            nnx.state(policy_model, nnx.LoRAParam))
    ckptr = ocp.StandardCheckpointer()
    restored_lora = ckptr.restore(trained_ckpt_path, target=abs_lora)
    nnx.update(
        policy_model,
        jax.tree.map(lambda a, b: b, nnx.state(policy_model, nnx.LoRAParam), restored_lora),
    )

    # --- Post-train eval ---
    sampler = make_vlm_sampler(
        policy_model,
        tokenizer,
        sigcfg.image_size,
    )
    post_metrics = quick_eval(eval_iter_for_eval(), sampler, max_batches=5)
    print(f"[Post] total={post_metrics['total']}, "
          f"exact_acc={post_metrics['exact_acc']:.2f}%, "
          f"format_acc={post_metrics['format_acc']:.2f}%")

    print("Done.")


if __name__ == "__main__":
    main()
