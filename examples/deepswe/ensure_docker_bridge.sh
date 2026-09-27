#!/usr/bin/env bash
# Restore the host's Docker default bridge if it disappeared while dockerd ran.
set -euo pipefail

if ! /usr/sbin/ip link show docker0 >/dev/null 2>&1; then
  /usr/sbin/ip link add name docker0 type bridge
fi
if ! /usr/sbin/ip -4 addr show dev docker0 | /usr/bin/grep -Fq '172.17.0.1/16'; then
  /usr/sbin/ip addr add 172.17.0.1/16 dev docker0
fi
/usr/sbin/ip link set docker0 up
