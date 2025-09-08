from __future__ import annotations
from flax import nnx
import jax

from tunix.models.siglip import params as siglip_params
from tunix.models.siglip import model as siglip_model
from tunix.models.gemma3 import params as gemma3_params
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.paligemma import model as pali_model

def create_model_from_checkpoints(
    *,
    siglip_dir: str,                       # HF safetensors folder: google/siglip-so400m-patch14-384
    gemma_ckpt_path: str,                  # Orbax path for Gemma3 (or swap with Gemma2 loader)
    gemma_config: gemma3_model.Gemma3Config,  # e.g., gemma3_model.Gemma3Config.gemma3_4b()
    mesh: jax.sharding.Mesh | None = None,
    max_vision_tokens: int = 256,
) -> pali_model.PaLIGemma:
  """Build PaLI-Gemma from SigLIP (HF) + Gemma3 (Orbax)."""
  # Load vision encoder
  siglip = siglip_params.create_model_from_safe_tensors(
      file_dir=siglip_dir, config=None, mesh=mesh
  )
  # Load Gemma3
  gemma = gemma3_params.create_model_from_checkpoint(
      checkpoint_path=gemma_ckpt_path, model_config=gemma_config, mesh=mesh
  )

  # Assemble wrapper with fresh projector
  cfg = pali_model.PaLIGemmaConfig(
      vision=siglip.cfg,  # exploit that SigLIPEngine keeps cfg as .cfg
      text=gemma.config,
      max_vision_tokens=max_vision_tokens,
  )
  pali_abs = nnx.eval_shape(lambda: pali_model.PaLIGemma(cfg, rngs=nnx.Rngs(params=0)))
  # stitch actual submodules in (share loaded params)
  graph_def, state = nnx.split(pali_abs)

  # replace subtrees with loaded weights (vision & text); projector stays randomly init’d
  state_dict = state.to_pure_dict()
  state_dict["vision"] = nnx.state(siglip).to_pure_dict()
  state_dict["text"] = nnx.state(gemma).to_pure_dict()

  return nnx.merge(graph_def, state_dict)
