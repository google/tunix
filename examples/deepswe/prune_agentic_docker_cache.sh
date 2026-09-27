#!/usr/bin/env bash
# Reclaim R2E task images after their containers have finished.
set -euo pipefail

if ! systemctl is-active --quiet deepswe-agentic-q4-clean.service; then
  exit 0
fi

service_start=$(systemctl show -P ExecMainStartTimestamp deepswe-agentic-q4-clean.service)
service_start_epoch=$(date -d "$service_start" +%s)
while read -r container_id container_name created_at; do
  if [[ "$container_name" != namanjain12-* ]]; then
    continue
  fi
  created_epoch=$(date -d "${created_at% UTC}" +%s) || continue
  if (( created_epoch + 5 < service_start_epoch )); then
    # The training process that owned this container has already exited.
    docker rm -f "$container_id" >/dev/null 2>&1 || true
  fi
done < <(docker ps --filter status=running --format '{{.ID}} {{.Names}} {{.CreatedAt}}')

available_kib=$(df -Pk / | awk 'NR == 2 {print $4}')
if (( available_kib >= 20 * 1024 * 1024 )); then
  exit 0
fi

while read -r container_id container_name; do
  if [[ "$container_name" == namanjain12-* ]]; then
    docker rm "$container_id" >/dev/null 2>&1 || true
  fi
done < <(docker ps -a --filter status=exited --format '{{.ID}} {{.Names}}')

while read -r image_id; do
  # Docker refuses removal while any container still references the image.
  docker image rm "$image_id" >/dev/null 2>&1 || true
done < <(docker image ls --filter 'reference=namanjain12/*' --format '{{.ID}}' | sort -u)

df -h /
