"""Admission probe: what MIN_TOKEN_BUCKET does this dp geometry need?

The all-M rule is "decode and prefill must land in the SAME token bucket", because a
different bucket is a different executable and bf16 reassociation makes it a different
number.  On the probe host that was satisfied by MIN_TOKEN_BUCKET=256 with dp=1.

Under data parallelism the arithmetic changes, and it changes silently.  The runner builds
one GLOBAL padding list and then divides:

    num_tokens_paddings = get_token_paddings(
        min_token_size=max(MIN_TOKEN_BUCKET, next_power_of_2(dp_size * kv_packing)),
        max_token_size=max_num_batched_tokens * dp_size,
        padding_gap=VLLM_TPU_BUCKET_PADDING_GAP)
    num_tokens_paddings_per_dp = [p // dp_size for p in num_tokens_paddings]

So MIN_TOKEN_BUCKET is a GLOBAL token count.  At dp=64 a global 256 gives each replica a
bucket of 4, not 256 -- the pinning that the whole result rests on would be silently gone
while every switch still reads "on".  This probe derives the value the target geometry
actually needs, and prefers the engine's own function over a reimplementation so the answer
cannot drift from the code that will run.

    python3 probe_bucket_contract.py

Environment:
    CANON_DP_SIZE          data-parallel width (default 1)
    CANON_TARGET_M         desired PER-REPLICA bucket (default 256)
    CANON_MAX_BATCHED      max_num_batched_tokens per replica (default 256)
    CANON_KV_PACKING       kv packing factor (default 1)
"""
import os


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _load_engine_impl():
    """Prefer the engine's own padding function; report which implementation was used."""
    try:
        from tpu_inference.runner import utils as runner_utils  # type: ignore
        return runner_utils.get_token_paddings, "tpu_inference.runner.utils"
    except Exception:
        pass
    try:
        from tpu_inference.runner.utils import get_token_paddings  # type: ignore
        return get_token_paddings, "tpu_inference.runner.utils"
    except Exception:
        return None, None


def _fallback_paddings(min_token_size, max_token_size, padding_gap):
    """Documented reimplementation, used only when the engine is not importable.

    Powers of two up to the gap, then linear steps of `padding_gap`.  Reported explicitly so
    a reader never mistakes this for the engine's own answer.
    """
    out = []
    v = min_token_size
    while v < max_token_size and (padding_gap <= 0 or v < padding_gap):
        out.append(v)
        v *= 2
    if padding_gap and padding_gap > 0:
        v = max(min_token_size, padding_gap)
        while v < max_token_size:
            out.append(v)
            v += padding_gap
    out.append(max_token_size)
    return sorted(set(x for x in out if x >= min_token_size))


def main():
    dp = int(os.environ.get("CANON_DP_SIZE", "1"))
    target_m = int(os.environ.get("CANON_TARGET_M", "256"))
    max_batched = int(os.environ.get("CANON_MAX_BATCHED", "256"))
    kv_packing = int(os.environ.get("CANON_KV_PACKING", "1"))

    print(f"[bucket] dp_size={dp} target_per_replica_M={target_m} "
          f"max_num_batched_tokens={max_batched} kv_packing={kv_packing}", flush=True)

    required_global = target_m * dp
    floor_from_dp = _next_power_of_2(dp * kv_packing)
    effective_min = max(required_global, floor_from_dp)

    print(f"[bucket] required_global_MIN_TOKEN_BUCKET={required_global}  "
          f"(= target_per_replica_M * dp_size)", flush=True)
    print(f"[bucket] next_power_of_2(dp*kv_packing)={floor_from_dp}  "
          f"(engine raises the floor to at least this)", flush=True)
    print(f"[bucket] SET MIN_TOKEN_BUCKET={effective_min}", flush=True)

    if dp > 1 and required_global != target_m:
        print(f"[bucket] WARNING: copying MIN_TOKEN_BUCKET={target_m} from the dp=1 recipe "
              f"would give each replica a bucket of {target_m // dp if dp <= target_m else 0}"
              f", not {target_m}.  Every canonical switch would still read 'on'.", flush=True)

    fn, src = _load_engine_impl()
    gap = int(os.environ.get("VLLM_TPU_BUCKET_PADDING_GAP", "0"))
    if fn is not None:
        try:
            paddings = fn(min_token_size=effective_min,
                          max_token_size=max_batched * dp,
                          padding_gap=gap)
            impl = src
        except Exception as exc:  # signature drift -> say so, do not guess
            print(f"[bucket] engine function present but not callable with this signature "
                  f"({exc!r}); falling back", flush=True)
            paddings, impl = _fallback_paddings(effective_min, max_batched * dp, gap), "fallback"
    else:
        paddings, impl = _fallback_paddings(effective_min, max_batched * dp, gap), "fallback"

    per_dp = [p // dp for p in paddings]
    print(f"[bucket] impl={impl}", flush=True)
    print(f"[bucket] global_paddings={paddings}", flush=True)
    print(f"[bucket] per_dp_paddings={per_dp}", flush=True)

    ok = target_m in per_dp
    print(f"[bucket] target_per_replica_M_reachable={ok}", flush=True)
    if impl == "fallback":
        print("[bucket] VERDICT: ADVISORY -- the engine's own get_token_paddings was not "
              "importable here, so the padding list is a documented reimplementation.  Re-run "
              "this probe inside the engine container before relying on the list.", flush=True)
        return 0
    print(f"[bucket] VERDICT: {'OK' if ok else 'UNREACHABLE'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
