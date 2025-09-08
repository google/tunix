
import os, io, json, time, argparse, math, random
from typing import List, Dict, Any

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

# PaLI-Gemma (text)
from tunix.models.paligemma import model as pali_model   # <- adjust if your path differs
from tunix.models.paligemma import params as pali_params # <- adjust if your path differs

# Tokenizer (Gemma tokenizer used by PaLI-Gemma head)
from tunix.models.gemma import data as gemma_tokenizer_lib

# VLM sampler/rollout/GRPO
from tunix.generate.vlm_sampler import VLMSampler            # <- your VLM sampler
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout                   # for RolloutConfig (we'll adapt)
from tunix.rl.grpo.grpo_learner import GrpoConfig, GrpoLearner

from tunix.sft import metrics_logger


# ----------------------
# Image preprocessing
# ----------------------
def load_and_preprocess_image(path: str, image_size: int) -> np.ndarray:
    """Loads an image (RGB), resizes to square image_size, returns float32 [H,W,3]
    normalized approximately like SigLIP (0..1 then mean/std)."""
    img = Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    x = np.array(img, dtype=np.float32) / 255.0
    # SigLIP normalization (close to CLIP): mean/std
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    x = (x - mean) / std
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
    # Shuffle for a touch of randomness
    random.Random(42).shuffle(data)

    def prompt_fmt(q: str) -> str:
        # PaLI-Gemma chat-ish template — tweak to your exact format if needed.
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
            # Skip unreadable images
            continue
        batch["prompts"].append(prompt_fmt(ex["question"]))
        batch["images"].append(img)
        batch["answer"].append(ex.get("answer", ""))     # string or list
        batch["question"].append(ex["question"])
        if len(batch["prompts"]) == batch_size:
            yield {
                "prompts": batch["prompts"],
                "images": np.stack(batch["images"], axis=0).astype(np.float32),
                "answer": batch["answer"],
                "question": batch["question"],
            }
            batch = {"prompts": [], "images": [], "answer": [], "question": []}
    # last partial (drop if empty)
    if batch["prompts"]:
        yield {
            "prompts": batch["prompts"],
            "images": np.stack(batch["images"], axis=0).astype(np.float32),
            "answer": batch["answer"],
            "question": batch["question"],
        }


# ----------------------
# Rewards (simple)
# ----------------------
import re
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
    scores = []
    for c in completions:
        s = 3.0 if FORMAT_RE.search(c) is not None else 0.0
        scores.append(s)
    return scores

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
    # Exact/close numeric checks inside <answer>...</answer>
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
    # Protect against image_size % patch_size != 0
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
        cfg = None  # params loader can infer; fallback to so400m_patch14_384 if needed
    if cfg is None:
        cfg = siglip_model.SigLIPConfig.so400m_patch14_384()
    cfg = ensure_cfg_divisible(cfg)

    enc = siglip_params.create_model_from_safe_tensors(siglip_dir, cfg, mesh)
    # Warmup-compile
    dummy = enc.get_model_input()
    _ = nnx.jit(enc)(**dummy)
    return enc, cfg


# ----------------------
# PaLI-Gemma loader (reference + LoRA policy)
# ----------------------
def load_pali_from_ckpt(ckpt_path: str, mesh):
    # Build abstract module/state, then restore
    model_cfg = pali_model.PaLIGemmaConfig.paligemma_3b()  # adjust to your variant
    abs_m = nnx.eval_shape(lambda: pali_model.PaLIGemma(model_cfg, rngs=nnx.Rngs(params=0)))
    abs_state = nnx.state(abs_m)
    abs_state = jax.tree.map(
        lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
        abs_state,
        nnx.get_named_sharding(abs_state, mesh),
    )
    ckptr = ocp.StandardCheckpointer()
    restored = ckptr.restore(ckpt_path, target=abs_state)
    graph_def, _ = nnx.split(abs_m)
    ref = nnx.merge(graph_def, restored)
    return ref, model_cfg

def apply_lora(base_model, mesh, rank=16, alpha=16.0):
    # Minimal, module-path based LoRA injection via Qwix (if you have it),
    # or your own utility. Here we use a simple helper if present.
    try:
        import qwix
    except Exception:
        qwix = None

    if qwix is None:
        raise RuntimeError("Qwix not found: please install or plug your LoRA applier.")

    lora_provider = qwix.LoraProvider(
        module_path=(
            ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
            ".*gate_proj|.*down_proj|.*up_proj"
        ),
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
def make_vlm_sampler(policy_text_model, tokenizer, vision_encoder, max_prompt_len, max_gen_steps, model_cfg):
    return VLMSampler(
        text_model=policy_text_model,
        tokenizer=tokenizer,
        vision_encoder=vision_encoder,
        # kv-cache sizing
        cache_config=dict(
            cache_size=max_prompt_len + max_gen_steps + 128,
            num_layers=model_cfg.num_layers,
            num_kv_heads=model_cfg.num_kv_heads,
            head_dim=model_cfg.head_dim,
        ),
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
            # format
            if FORMAT_RE.search(pred): fmt += 1
            # numeric exact
            m = FORMAT_RE.search(pred)
            if m:
                try:
                    if float(m.group(1).strip()) == float(str(ans).strip()):
                        corr += 1
                except: pass
            total += 1
    acc = (corr / total * 100) if total else 0.0
    fmt_acc = (fmt / total * 100) if total else 0.0
    return dict(total=total, exact_acc=acc, format_acc=fmt_acc)


# ----------------------
# Main
# ----------------------
def main():
    p = argparse.ArgumentParser()
    # Paths
    p.add_argument("--siglip_dir", default='/home/grads/tianjiao/checkpoints/siglip-so400m-patch14-384', help="Folder with SigLIP safetensors")
    p.add_argument("--pali_ckpt", required=True, help="Orbax state checkpoint path for PaLI-Gemma (reference)")
    p.add_argument("--train_jsonl", default='/home/grads/tianjiao/tunix/scripts//dummy_vqa/train.jsonl', help="VQA JSONL for training (image, question, answer)")
    p.add_argument("--eval_jsonl", default='/home/grads/tianjiao/tunix/scripts/dummy_vqa/eval.jsonl', help="VQA JSONL for eval")

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

    # --- Load models ---
    siglip, sigcfg = load_siglip(args.siglip_dir, forced_cfg=None, mesh=mesh)
    ref_model, text_cfg = load_pali_from_ckpt(args.pali_ckpt, mesh=mesh)
    policy_model = apply_lora(ref_model, mesh, rank=args.lora_rank, alpha=args.lora_alpha)

    # Tokenizer
    tokenizer = gemma_tokenizer_lib.GemmaTokenizer()

    # Sampler for eval
    sampler = make_vlm_sampler(
        policy_text_model=policy_model,
        tokenizer=tokenizer,
        vision_encoder=siglip,
        max_prompt_len=args.max_prompt_len,
        max_gen_steps=args.max_gen_steps,
        model_cfg=text_cfg,
    )

    # --- Datasets ---
    train_iter = make_vqa_iter(args.train_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size)
    eval_iter_for_eval = lambda: make_vqa_iter(args.eval_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size)

    # --- Pre-train eval ---
    print("Running quick pre-train eval (greedy, few batches)...")
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
    optimizer = optax.chain(optax.clip_by_global_norm(0.1), optax.adamw(schedule, b1=0.9, b2=0.99, weight_decay=0.1))

    ckpt_opts = ocp.CheckpointManagerOptions(save_interval_steps=args.save_every, max_to_keep=4)
    log_opts = metrics_logger.MetricsLoggerOptions(log_dir=args.tb_dir, flush_every_n_steps=20)

    cluster_cfg = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vlm",          # <-- ensure your rollout is registered as 'vlm'
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
        vision_encoder=siglip,     # <-- if your RLCluster takes it; else your rollout reads it from sampler
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
    print("Starting GRPO training...")
    with mesh:
        # Our train iterator must be infinite-ish for GRPO;
        # adapt to your loader: wrap as repeat generator.
        def repeat_train():
            while True:
                for b in make_vqa_iter(args.train_jsonl, image_size=sigcfg.image_size, batch_size=args.batch_size):
                    yield b
        learner.train(train_ds=repeat_train(), eval_ds=None, skip_jit=False)

    # --- Savepoint info ---
    print("Training done. Attempting quick LoRA-only restore test...")
    last_step = args.max_steps
    trained_ckpt_path = os.path.join(args.ckpt_dir, str(last_step), "model_params")
    abs_lora = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), nnx.state(policy_model, nnx.LoRAParam))
    ckptr = ocp.StandardCheckpointer()
    restored_lora = ckptr.restore(trained_ckpt_path, target=abs_lora)
    nnx.update(
        policy_model,
        jax.tree.map(lambda a, b: b, nnx.state(policy_model, nnx.LoRAParam), restored_lora),
    )

    # --- Post-train eval ---
    sampler = make_vlm_sampler(policy_model, tokenizer, siglip, args.max_prompt_len, args.max_gen_steps, text_cfg)
    post_metrics = quick_eval(eval_iter_for_eval(), sampler, max_batches=5)
    print(f"[Post] total={post_metrics['total']}, "
          f"exact_acc={post_metrics['exact_acc']:.2f}%, "
          f"format_acc={post_metrics['format_acc']:.2f}%")

    print("Done.")


if __name__ == "__main__":
    main()