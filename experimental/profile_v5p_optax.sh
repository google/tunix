#!/bin/bash
# Profile ONLY the optax.MultiSteps grad-accum variant on a single-host TPU VM.
# Thin wrapper over profile_v5p_grad_accum.sh (all env overrides pass through);
# trace -> $TRACE_DEST/${RUN_TAG}_optax (default gs://yuxzhang-tunix-models/
# xprof/v5p_refactor_optax).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VARIANTS=optax exec bash "$SCRIPT_DIR/profile_v5p_grad_accum.sh" "$@"
