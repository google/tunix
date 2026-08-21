#!/usr/bin/env python3
"""Qwen3-8B TP8 projection contract for the additive P22.XK stack."""

from __future__ import annotations

import os
from dataclasses import dataclass


ENV = "CANON_PALLAS_ALL_PROJ"
CONFLICTS = (
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_POSTRPA_M",
    "CANON_CUT",
    "CANON_TAIL",
)
BM = 128
BN = 128
BK = 128
MATMUL_K_PADDING = {}
# The seven decoder projections remain exact and unpadded.  The only padded
# N is the TP8-local output head: 151936 / 8 = 18992 logical vocabulary rows,
# padded to the registered fixed-head BN256 geometry.
MATMUL_N_PADDING = {18992: 19200}
SWIGLU_FEATURE_PADDING = {}

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
NUM_ATTENTION_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
TP_SIZE = 8

_MODEL_ENV = {
    "CANON_QWEN3_HIDDEN_SIZE": str(HIDDEN_SIZE),
    "CANON_QWEN3_INTERMEDIATE_SIZE": str(INTERMEDIATE_SIZE),
    "CANON_QWEN3_NUM_ATTENTION_HEADS": str(NUM_ATTENTION_HEADS),
    "CANON_QWEN3_NUM_KV_HEADS": str(NUM_KV_HEADS),
    "CANON_QWEN3_HEAD_DIM": str(HEAD_DIM),
    "CANON_QWEN3_TP_SIZE": str(TP_SIZE),
}


@dataclass(frozen=True)
class Site:
    suffix: str
    family: str
    equations: tuple[str, ...]
    k_local: int
    n_local: int
    contract_parallel: bool


SITES = (
    Site(".q_proj", "q_proj", ("TD,DNH->TNH", "TD,NDH->TNH"), 4096, 512, False),
    Site(".k_proj", "k_proj", ("TD,DKH->TKH",), 4096, 128, False),
    Site(".v_proj", "v_proj", ("TD,DKH->TKH",), 4096, 128, False),
    Site(".o_proj", "o_proj", ("TNH,NHD->TD",), 512, 4096, True),
    Site(".gate_proj", "gate_proj", ("mn,np->mp",), 4096, 1536, False),
    Site(".up_proj", "up_proj", ("mn,np->mp",), 4096, 1536, False),
    Site(".down_proj", "down_proj", ("mn,np->mp",), 1536, 4096, True),
)


def validate_qwen8b_env() -> None:
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, expected in _MODEL_ENV.items()
        if os.environ.get(name, "") != expected
    ]
    if wrong:
        raise RuntimeError(
            "Qwen3-8B TP8 P22.XK model contract mismatch: " + ", ".join(wrong)
        )


validate_model_env = validate_qwen8b_env


def validate_manifest(sites) -> None:
    expected_shapes = {
        "q_proj": (4096, 512),
        "k_proj": (4096, 128),
        "v_proj": (4096, 128),
        "o_proj": (512, 4096),
        "gate_proj": (4096, 1536),
        "up_proj": (4096, 1536),
        "down_proj": (1536, 4096),
    }
    if len(sites) != 7:
        raise ValueError(f"expected 7 projection sites, got {len(sites)}")
    suffixes = [site.suffix for site in sites]
    families = [site.family for site in sites]
    if len(set(suffixes)) != 7 or len(set(families)) != 7:
        raise ValueError("projection suffixes/families must be unique")
    if sum(site.contract_parallel for site in sites) != 2:
        raise ValueError("expected exactly o/down contract-parallel sites")
    for site in sites:
        actual = (site.k_local, site.n_local)
        if actual != expected_shapes[site.family]:
            raise ValueError(
                f"{site.family} shape {actual} does not match Qwen3-8B TP8 "
                f"{expected_shapes[site.family]}"
            )
        if site.k_local % BK or site.n_local % BN:
            raise ValueError(
                f"{site.family} local shape {actual} does not divide "
                f"BK/BN={BK}/{BN}"
            )
    if MATMUL_K_PADDING or MATMUL_N_PADDING != {18992: 19200}:
        raise ValueError(
            "Qwen3-8B TP8 permits padding only for the fixed lm_head: "
            "N18992->19200"
        )
    local_feature = INTERMEDIATE_SIZE // TP_SIZE
    if local_feature != 1536 or local_feature % 256 or SWIGLU_FEATURE_PADDING:
        raise ValueError(
            "Qwen3-8B TP8 SwiGLU must remain on the unpadded BF256 path"
        )


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise RuntimeError(f"{ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise RuntimeError(f"{ENV}=1 required")
    if value == "1":
        validate_qwen8b_env()
        if os.environ.get("CANON_FIXED_AR", "") != "1":
            raise RuntimeError("CANON_FIXED_AR=1 required")
        active = [name for name in CONFLICTS if os.environ.get(name, "")]
        if active:
            raise RuntimeError("conflicting diagnostics: " + ",".join(active))
    validate_manifest(SITES)


def match_site(prefix: str, equation: str) -> Site | None:
    matches = [site for site in SITES if prefix.endswith(site.suffix)]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"ambiguous Qwen3-8B TP8 site for {prefix!r}")
    site = matches[0]
    if equation not in site.equations:
        raise RuntimeError(
            f"P22.XF equation mismatch at {prefix}: {equation!r} not in "
            f"{site.equations!r}"
        )
    return site


def self_test() -> None:
    old = {name: os.environ.get(name) for name in _MODEL_ENV}
    old_enabled = os.environ.get(ENV)
    old_fixed_ar = os.environ.get("CANON_FIXED_AR")
    try:
        os.environ.update(_MODEL_ENV)
        os.environ[ENV] = "1"
        os.environ["CANON_FIXED_AR"] = "1"
        preflight(require_enabled=True)
        assert len(SITES) == 7
        assert (BM, BN, BK) == (128, 128, 128)
        assert match_site(
            "model.layers.0.self_attn.q_proj", "TD,DNH->TNH"
        ).n_local == 512
        assert match_site(
            "model.layers.0.mlp.down_proj", "mn,np->mp"
        ).k_local == 1536
        assert MATMUL_K_PADDING == {}
        assert MATMUL_N_PADDING == {18992: 19200}
        assert SWIGLU_FEATURE_PADDING == {}
        os.environ["CANON_QWEN3_TP_SIZE"] = "4"
        try:
            preflight(require_enabled=True)
        except RuntimeError:
            pass
        else:
            raise AssertionError("wrong TP4 model environment was accepted")
    finally:
        for name, previous in old.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        for name, previous in ((ENV, old_enabled), ("CANON_FIXED_AR", old_fixed_ar)):
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
    print("P45_QWEN8B_TP8_CONTRACT_SELFTEST_PASS cases=7/7 tp4_negative=1")


if __name__ == "__main__":
    self_test()
