"""Fail-closed Pathways/JAX registration preflight for T1."""

import os

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax  # noqa: E402  Pathways must register before JAX import.


def main() -> int:
    try:
        devices = jax.devices()
    except Exception as exc:
        print(f"[t1.devices] REFUSING: jax.devices() failed: {exc}", flush=True)
        return 1
    if not devices:
        print("[t1.devices] REFUSING: JAX reported zero devices", flush=True)
        return 1
    expected_raw = os.environ.get("CANON_EXPECT_VISIBLE_DEVICES", "").strip()
    if expected_raw:
        try:
            expected = int(expected_raw)
        except ValueError:
            print(
                "[t1.devices] REFUSING: CANON_EXPECT_VISIBLE_DEVICES must be an integer",
                flush=True,
            )
            return 1
        if len(devices) != expected:
            print(
                f"[t1.devices] REFUSING: expected={expected} actual={len(devices)}",
                flush=True,
            )
            return 1
    first = devices[0]
    print(
        f"[t1.devices] count={len(devices)} kind={first.device_kind} "
        f"platform={first.platform}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
