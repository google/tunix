"""Hot-patch tpu_inference's qwen3.py for the RoPE-scaling bug (verbatim logic from
experimental/deepswe-256-mlperf.yaml's inline patch). Source A/B (vLLM/tpu_inference) need
this or RoPE is computed wrong. Runs INSIDE the tunix_base_image container (the target file
only exists there); no-op with a message if the file isn't present.

Usage (inside the container): python3 scripts/logp_diff/patch_qwen3.py
"""
import os

FILE_PATH = "/app/vllm_tpu_inference/tpu_inference/tpu_inference/models/jax/qwen3.py"

INJECTED = '''
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

'''


def main():
  if not os.path.exists(FILE_PATH):
    print(f"[patch_qwen3] target not found, skipping: {FILE_PATH}")
    return
  with open(FILE_PATH, "r") as f:
    code = f.read()
  if "def normalize_rope_scaling" in code:
    print("[patch_qwen3] already patched, skipping")
    return
  code = INJECTED + code
  code = code.replace(
      'self.rope_theta = config.rope_parameters["rope_theta"]',
      'self.rope_theta = get_rope_theta(config, default=1000000.0)')
  code = code.replace(
      'self.rope_scaling = getattr(config, "rope_scaling", None)',
      'self.rope_scaling = get_rope_scaling(config)')
  with open(FILE_PATH, "w") as f:
    f.write(code)
  print("[patch_qwen3] applied RoPE fix to", FILE_PATH)


if __name__ == "__main__":
  main()
