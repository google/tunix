#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec python3 -m examples.frozenlake.gemma4_tim.run --arm tis "$@"
