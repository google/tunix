#!/usr/bin/env bash
# Run the matched P57 exact-TiTO observer off/on pair on one v5p host.
set -euo pipefail

label="${1:?usage: run_tito_onehost_neutrality_pair.sh <unique-label>}"
case "$label" in
  *[!a-zA-Z0-9_-]*|'')
    echo "[P57.TITO.ONEHOST] invalid label: $label" >&2
    exit 2
    ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$script_dir" rev-parse --show-toplevel)"
perf_runner="$repo/canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/run_perf_v2_onehost.sh"
judge="$script_dir/judge_tito_onehost_neutrality.py"
root=/mnt/disks/tunix-data/logp_probe_1host
off_root="$root/p57_perf_v2_${label}_off"
on_root="$root/p57_perf_v2_${label}_on"
result="$root/p57_tito_neutrality_${label}.json"

if [ -n "$(git -C "$repo" status --porcelain)" ]; then
  echo "[P57.TITO.ONEHOST] refusing a dirty worktree" >&2
  exit 2
fi
if [ -e "$off_root" ] || [ -e "$on_root" ] || [ -e "$result" ]; then
  echo "[P57.TITO.ONEHOST] refusing an existing pair label: $label" >&2
  exit 3
fi

idle_samples=0
while [ "$idle_samples" -lt 12 ]; do
  active=""
  while IFS= read -r running_container; do
    case "$running_container" in
      tpu-runtime|instance_agent|vbarcontrolagent|google-runtime-monitor|healthagent|google-collectd|monitoringagent)
        continue
        ;;
    esac
    if [ "$(sudo docker inspect --format '{{.HostConfig.Privileged}}' "$running_container")" = true ]; then
      active="${active}${active:+$'\n'}${running_container}"
    fi
  done < <(sudo docker ps --format '{{.Names}}')
  if [ -n "$active" ]; then
    idle_samples=0
    echo "[P57.TITO.ONEHOST] waiting for an idle host; active privileged containers:" >&2
    printf '%s\n' "$active" >&2
  else
    idle_samples=$((idle_samples + 1))
    echo "[P57.TITO.ONEHOST] idle sample ${idle_samples}/12"
  fi
  if [ "$idle_samples" -lt 12 ]; then
    sleep 10
  fi
done

bash "$perf_runner" "${label}_off" tito-off
bash "$perf_runner" "${label}_on" tito-on
python3 "$judge" \
  --off-root "$off_root" \
  --on-root "$on_root" \
  --output "$result"

echo "P57_TITO_ONEHOST_PAIR_PASS label=$label result=$result"
