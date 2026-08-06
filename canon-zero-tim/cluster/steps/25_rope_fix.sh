#!/usr/bin/env bash
# Apply the RoPE compatibility fix -- but only to builds that need it.
#
# The cluster runs carried this as an unconditional inline patch that rewrites qwen3.py.  On a
# build where the fix is already upstream, that patch's string replacements match nothing and
# silently do nothing, while its injected function definitions still get prepended -- shadowing
# the real ones.  A no-op that looks like a success is exactly the failure mode this project
# keeps paying for, so this step decides explicitly and refuses on an unrecognised build.
#
#   applied            the old form is present -> rewrite it
#   not_needed         the new form is present -> leave the file alone
#   unknown_version    neither -> stop.  Do not guess at a build nobody has looked at.
set -euo pipefail
source "$CANON_STATE/env.sh"

SP="$(cat "$CANON_STATE/tpu_inference_path")"
F="$SP/models/jax/qwen3.py"
[ -f "$F" ] || { echo "[rope] missing $F" >&2; exit 1; }

OLD_THETA='self.rope_theta = config.rope_parameters["rope_theta"]'
OLD_SCALING='self.rope_scaling = getattr(config, "rope_scaling", None)'

if grep -qF "$OLD_THETA" "$F"; then
  echo "[rope] old form detected -- applying fix"
  python3 - "$F" <<'PY'
import sys, textwrap
p = sys.argv[1]
code = open(p, encoding="utf-8").read()
injected = textwrap.dedent('''
    from typing import Any, Dict, Optional

    def normalize_rope_scaling(rope_scaling: Any) -> Optional[Dict[str, Any]]:
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            if (rope_scaling.get("rope_type", "default") == "default"
                    and "factor" not in rope_scaling
                    and "scale_factor" not in rope_scaling
                    and "mrope_section" not in rope_scaling):
                rope_scaling = None
            elif "factor" in rope_scaling and "scale_factor" not in rope_scaling:
                rope_scaling["scale_factor"] = rope_scaling.pop("factor")
        return rope_scaling

    def get_rope_scaling(config: Any) -> Optional[Dict[str, Any]]:
        rope_scaling = getattr(config, "rope_parameters", None) or getattr(
            config, "rope_scaling", None)
        return normalize_rope_scaling(rope_scaling)

    def get_rope_theta(config: Any, default: float = 10000.0) -> float:
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is not None and "rope_theta" in rope_parameters:
            return float(rope_parameters["rope_theta"])
        return float(getattr(config, "rope_theta", default))

''')
n = 0
for old, new in (
    ('self.rope_theta = config.rope_parameters["rope_theta"]',
     'self.rope_theta = get_rope_theta(config, default=1000000.0)'),
    ('self.rope_scaling = getattr(config, "rope_scaling", None)',
     'self.rope_scaling = get_rope_scaling(config)'),
):
    if old in code:
        code = code.replace(old, new)
        n += 1
if n == 0:
    sys.exit("[rope] internal error: detector said old form, replacer found none")
open(p, "w", encoding="utf-8").write(injected + code)
print(f"[rope] rewrote {n} site(s)")
PY
  echo "ROPE_FIX=applied" > "$CANON_STATE/rope_fix"
  echo "[rope] ROPE_FIX=applied"
  exit 0
fi

if grep -qE 'get_rope_theta\(|get_rope_scaling\(' "$F"; then
  echo "[rope] new form already present (upstream) -- not patching"
  grep -nE 'from tpu_inference.*rope_interface|self\.rope_(theta|scaling) *=' "$F" \
    | head -5 | sed 's/^/[rope]   /'
  echo "ROPE_FIX=not_needed" > "$CANON_STATE/rope_fix"
  echo "[rope] ROPE_FIX=not_needed"
  exit 0
fi

echo "[rope] neither the old nor the new form was found in $F" >&2
echo "[rope] ROPE_FIX=unknown_version -- REFUSING.  Patching an unrecognised build blind is" >&2
echo "[rope] how a silent no-op gets mistaken for a fix.  Inspect the file first." >&2
echo "ROPE_FIX=unknown_version" > "$CANON_STATE/rope_fix"
exit 1
