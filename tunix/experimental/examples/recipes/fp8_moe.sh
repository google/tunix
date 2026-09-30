#!/bin/bash
# ==============================================================================
# FP8 MoE switches, shared by the recipe launchers (mlperf_base.sh,
# gsm_8k_35b_256.sh). Sourced after the recipe has set VLLM_ADDITIONAL_CONFIG and
# MAXTEXT_EXTRA_FLAGS; appends to both, so any recipe can turn FP8 on from the
# command line:
#   ROLLOUT_FP8=true bash mlperf_397b_256_v7x.sh start
#
# ROLLOUT_FP8=true   The rollout stores its routed-expert weights (wi/wo) in
#                    float8_e4m3fn with per-expert, per-output-channel scales, and
#                    the MoE kernel quantizes its activations to FP8 too (W8A8).
#                    All other weights, and the trainer, stay bf16. The trainer's
#                    weight converter quantizes the experts on every sync
#                    (rollout_fp8_moe tells it the rollout expects FP8).
# TRAINER_FP8=true   Experimental; requires ROLLOUT_FP8. The trainer keeps bf16
#                    master weights but rounds its routed experts to the same FP8
#                    grid in the forward pass (straight-through gradients), so its
#                    log-probs use the weights the rollout serves.
#
# Both need a MaxText that has the rollout_fp8_moe / fp8_moe_fake_quant flags.
# ==============================================================================

export ROLLOUT_FP8="${ROLLOUT_FP8:-false}"
export TRAINER_FP8="${TRAINER_FP8:-false}"
for _var in ROLLOUT_FP8 TRAINER_FP8; do
  if [[ "${!_var}" != "true" && "${!_var}" != "false" ]]; then
    echo "ERROR: ${_var} must be true or false, got '${!_var}'" >&2
    exit 1
  fi
done
if [[ "${TRAINER_FP8}" == "true" && "${ROLLOUT_FP8}" != "true" ]]; then
  echo "ERROR: TRAINER_FP8=true matches the trainer to an FP8 rollout; set ROLLOUT_FP8=true too" >&2
  exit 1
fi

if [[ "${ROLLOUT_FP8}" == "true" ]]; then
  VLLM_ADDITIONAL_CONFIG="$(python3 -c '
import json, os
cfg = json.loads(os.environ.get("VLLM_ADDITIONAL_CONFIG") or "{}")
cfg.setdefault("maxtext_config", {})["fp8_moe"] = True
print(json.dumps(cfg, separators=(",", ":")))
')"
  export VLLM_ADDITIONAL_CONFIG
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-} rollout_fp8_moe=true"
fi
if [[ "${TRAINER_FP8}" == "true" ]]; then
  export MAXTEXT_EXTRA_FLAGS="${MAXTEXT_EXTRA_FLAGS:-} fp8_moe_fake_quant=true"
fi
