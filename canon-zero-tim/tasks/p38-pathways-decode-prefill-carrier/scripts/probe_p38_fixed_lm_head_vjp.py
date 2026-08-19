#!/usr/bin/env python3
"""Real-weight one-v5p VJP gate for the P38 fixed lm-head composition."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from p22xi_padded_matmul import matmul as padded_pallas_matmul
from p22xk_vjp_ops import matmul as canonical_vjp_matmul
from p38_fixed_lm_head import (
    FIXED_M,
    HIDDEN,
    LEARNER_M,
    LOCAL_VOCAB,
    TP_SIZE,
    VOCAB,
    VJP_PASS,
    classify_vjp,
    fixed_lm_head,
    preflight,
)
from probe_p38_lm_head import _load_weight


def _promoted_matmul(x, y, **kwargs):
    def forward(a, b):
        return padded_pallas_matmul(a, b, **kwargs)

    return canonical_vjp_matmul(x, y, forward=forward)


def _different_elements(left, right):
    return jnp.sum((left != right).astype(jnp.int32), dtype=jnp.int32)


def _max_abs(left, right):
    return jnp.max(
        jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))
    )


def _finite(value):
    return jnp.all(jnp.isfinite(value))


def _nonzero(value):
    return jnp.sum((value != 0).astype(jnp.int32), dtype=jnp.int32)


def _small_metrics(left, right) -> dict[str, int | float]:
    differing, largest = jax.jit(
        lambda x, y: (_different_elements(x, y), _max_abs(x, y))
    )(left, right)
    return {
        "differing_elements": int(differing),
        "max_abs": float(largest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=3821)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN)
    args = parser.parse_args()

    preflight(require_enabled=True)
    devices = jax.devices()
    if jax.default_backend() != "tpu" or len(devices) != TP_SIZE:
        raise RuntimeError(
            f"P38 fixed lm_head VJP requires exactly {TP_SIZE} TPU devices"
        )
    learner_m = LEARNER_M[0]
    chunks = learner_m // FIXED_M
    if learner_m % FIXED_M or chunks != 16:
        raise RuntimeError("P38 fixed lm_head learner chunk contract drifted")

    mesh = Mesh(np.asarray(devices), ("model",))
    replicated = NamedSharding(mesh, P(None, None))
    rows_replicated = NamedSharding(mesh, P(None))
    vocab_sharded = NamedSharding(mesh, P(None, "model"))
    hidden_size = int(args.hidden_size)
    weight = _load_weight(args.model, vocab_sharded, hidden_size)

    key = jax.random.PRNGKey(args.seed)
    hidden = jax.random.normal(
        key, (learner_m, hidden_size), dtype=jnp.float32
    )
    hidden = jax.device_put(hidden.astype(jnp.bfloat16), replicated)
    # Exercise one repeated selected-token column on every TP shard. Repeating
    # each column across 1024 rows makes the shared dWeight accumulation across
    # the 16 M256 chunks observable instead of accidentally disjoint.
    target_ids = (
        (jnp.arange(learner_m, dtype=jnp.int32) % TP_SIZE) * LOCAL_VOCAB + 17
    )
    target_ids = jax.device_put(target_ids, rows_replicated)

    def selected_loss(hidden_arg, weight_arg, target_arg):
        logits = fixed_lm_head(
            hidden_arg,
            weight_arg,
            mesh=mesh,
            tp_axis="model",
            local_matmul=_promoted_matmul,
        )
        selected = jnp.take_along_axis(
            logits, target_arg[:, None], axis=1
        )[:, 0]
        return jnp.sum(selected.astype(jnp.float32))

    learner_value_and_grad = jax.jit(
        jax.value_and_grad(selected_loss, argnums=(0, 1)),
        in_shardings=(replicated, vocab_sharded, rows_replicated),
    )

    candidate_loss, (candidate_hidden, candidate_weight) = (
        learner_value_and_grad(hidden, weight, target_ids)
    )
    jax.block_until_ready(
        (candidate_loss, candidate_hidden, candidate_weight)
    )
    repeat_loss, (repeat_hidden, repeat_weight) = learner_value_and_grad(
        hidden, weight, target_ids
    )
    jax.block_until_ready((repeat_loss, repeat_hidden, repeat_weight))

    chunk_value_and_grad = jax.jit(
        jax.value_and_grad(selected_loss, argnums=(0, 1)),
        in_shardings=(replicated, vocab_sharded, rows_replicated),
    )
    hidden_parts = []
    reference_weight = jnp.zeros_like(weight)
    reference_loss = 0.0
    for chunk in range(chunks):
        start = chunk * FIXED_M
        stop = start + FIXED_M
        chunk_loss, (chunk_hidden, chunk_weight) = chunk_value_and_grad(
            hidden[start:stop], weight, target_ids[start:stop]
        )
        # This is the reference fixed order: each semantic M256 VJP is complete
        # before its shared dWeight contribution is added to the accumulator.
        next_weight = reference_weight + chunk_weight
        jax.block_until_ready((chunk_loss, chunk_hidden, next_weight))
        hidden_parts.append(chunk_hidden)
        reference_weight = next_weight
        reference_loss += float(chunk_loss)
    reference_hidden = jnp.concatenate(hidden_parts, axis=0)
    jax.block_until_ready((reference_hidden, reference_weight))

    hidden_metrics = _small_metrics(candidate_hidden, reference_hidden)
    weight_metrics = _small_metrics(candidate_weight, reference_weight)
    repeat_hidden_metrics = _small_metrics(candidate_hidden, repeat_hidden)
    repeat_weight_metrics = _small_metrics(candidate_weight, repeat_weight)

    finite_hidden, finite_weight, nonzero_hidden, nonzero_weight = jax.jit(
        lambda dh, dw: (
            _finite(dh),
            _finite(dw),
            _nonzero(dh),
            _nonzero(dw),
        )
    )(candidate_hidden, candidate_weight)
    gradients_finite = bool(finite_hidden) and bool(finite_weight)
    hidden_nonzero = int(nonzero_hidden)
    weight_nonzero = int(nonzero_weight)
    gradients_nonzero = hidden_nonzero > 0 and weight_nonzero > 0

    # A normal finite value is used rather than toggling a BF16 subnormal: TPU
    # flush-to-zero would make the latter a fake negative control.
    poisoned_weight = candidate_weight.at[0, 17].set(jnp.bfloat16(1.0))
    if bool(candidate_weight[0, 17] == jnp.bfloat16(1.0)):
        poisoned_weight = candidate_weight.at[0, 17].set(jnp.bfloat16(2.0))
    negative_differing = int(
        jax.jit(_different_elements)(candidate_weight, poisoned_weight)
    )

    verdict = classify_vjp(
        hidden_differing=hidden_metrics["differing_elements"],
        weight_differing=weight_metrics["differing_elements"],
        repeat_hidden_differing=repeat_hidden_metrics["differing_elements"],
        repeat_weight_differing=repeat_weight_metrics["differing_elements"],
        gradients_finite=gradients_finite,
        gradients_nonzero=gradients_nonzero,
        negative_differing=negative_differing,
    )
    report: dict[str, Any] = {
        "schema": "canon-p38-fixed-lm-head-vjp-v1",
        "verdict": verdict,
        "claim_scope": "onehost-real-weight-fixed-lm-head-vjp-construction-only",
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "semantic_m": learner_m,
        "fixed_m": FIXED_M,
        "chunks": chunks,
        "hidden": hidden_metrics,
        "weight": weight_metrics,
        "repeat_hidden": repeat_hidden_metrics,
        "repeat_weight": repeat_weight_metrics,
        "candidate_loss": float(candidate_loss),
        "repeat_loss": float(repeat_loss),
        "reference_chunk_loss_sum": reference_loss,
        "gradient_finite": gradients_finite,
        "hidden_gradient_nonzero": hidden_nonzero,
        "weight_gradient_nonzero": weight_nonzero,
        "negative_control_differing_elements": negative_differing,
        "weight_shape": [hidden_size, VOCAB],
        "weight_dtype": str(weight.dtype),
        "weight_sharding": str(weight.sharding),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[P38.FIXED_LM_HEAD.VJP] {json.dumps(report, sort_keys=True)}")
    if verdict != VJP_PASS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
