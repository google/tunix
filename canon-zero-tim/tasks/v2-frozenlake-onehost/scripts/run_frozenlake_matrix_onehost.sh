#!/usr/bin/env bash
# Typed entrypoint for the Qwen3-8B FrozenLake four-chip one-host matrix.
set -euo pipefail

workload="${1:?usage: run_frozenlake_matrix_onehost.sh <p45|m15> <dp4-tp1|dp2-tp2|dp1-tp4> <r0|r1|r2|r3> <fresh-label> <measure|certify> [none|capture|replay] [capsule.npz]}"
geometry="${2:?usage: run_frozenlake_matrix_onehost.sh <p45|m15> <dp4-tp1|dp2-tp2|dp1-tp4> <r0|r1|r2|r3> <fresh-label> <measure|certify> [none|capture|replay] [capsule.npz]}"
arm="${3:?usage: run_frozenlake_matrix_onehost.sh <p45|m15> <dp4-tp1|dp2-tp2|dp1-tp4> <r0|r1|r2|r3> <fresh-label> <measure|certify> [none|capture|replay] [capsule.npz]}"
label="${4:?usage: run_frozenlake_matrix_onehost.sh <p45|m15> <dp4-tp1|dp2-tp2|dp1-tp4> <r0|r1|r2|r3> <fresh-label> <measure|certify> [none|capture|replay] [capsule.npz]}"
mode="${5:?usage: run_frozenlake_matrix_onehost.sh <p45|m15> <dp4-tp1|dp2-tp2|dp1-tp4> <r0|r1|r2|r3> <fresh-label> <measure|certify> [none|capture|replay] [capsule.npz]}"
capsule_mode="${6:-none}"
capsule_source="${7:-}"

case "$geometry" in dp4-tp1|dp2-tp2|dp1-tp4) ;; *) echo "invalid geometry: $geometry" >&2; exit 2;; esac
case "$arm" in r0|r1|r2|r3) ;; *) echo "matrix entrypoint admits only r0/r1/r2/r3" >&2; exit 2;; esac
case "$mode" in measure|certify) ;; *) echo "matrix entrypoint admits only measure/certify" >&2; exit 2;; esac
if [ "$geometry" = dp1-tp4 ] && [ "$arm" = r2 ]; then
  echo "DP1 has no reduce-once arm; use r1 or the stream+sort r3 candidate" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "$0")" && pwd)"
export V2_FL_GEOMETRY="$geometry"
exec bash "$script_dir/run_frozenlake_dp2tp2_onehost.sh" \
  "$workload" "$arm" "$label" "$mode" "$capsule_mode" "$capsule_source"
