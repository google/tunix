#!/usr/bin/env bash
# 28_sync_cache.sh: Pull the persistent JAX/XLA cache before training.
set -uo pipefail
source "$CANON_STATE/env.sh"
# shellcheck disable=SC1091
source "$CANON_PKG/cluster/steps/jax_cache_sync_lib.sh"

canon_jax_cache_sync restore
