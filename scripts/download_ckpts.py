import os
import sys
sys.path.insert(0, "/home/grads/tianjiao/tunix")

from flax import nnx
from orbax import checkpoint as ocp
import kagglehub

from tunix.models.gemma import params as params_lib
from tunix.models.gemma import gemma as gemma_lib


INTERMEDIATE_CKPT_DIR = os.path.expanduser(
    "/home/grads/tianjiao/checkpoints/gemma_ckpts/intermediate"
)


def main():
    os.makedirs(INTERMEDIATE_CKPT_DIR, exist_ok=True)

    # Auth (no-op if already configured via ~/.kaggle/kaggle.json)
    if "KAGGLE_USERNAME" not in os.environ or "KAGGLE_KEY" not in os.environ:
        try:
            kagglehub.login()
        except Exception as e:
            print("Kaggle login skipped or not interactive:", e)

    # Download
    path = kagglehub.model_download("google/gemma-2/flax/gemma2-2b-it")
    print("Kaggle model at:", path)

    # Build NNX model from Kaggle params
    params = params_lib.load_and_format_params(os.path.join(path, "gemma2-2b-it"))
    gemma = gemma_lib.Transformer.from_params(params, version="2-2b-it")

    # Split and save
    _, state = nnx.split(gemma)

    # Use AsyncCheckpointer but *wait* and *close* to flush everything
    checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
    save_fut = checkpointer.save(
        os.path.join(INTERMEDIATE_CKPT_DIR, "state"),
        state,
        force=True,
    )
    # Ensure all background work completes
    checkpointer.wait_until_finished()
    checkpointer.close()

    print("Saved NNX state to:", INTERMEDIATE_CKPT_DIR)

    # Free RAM
    del params, gemma, state


if __name__ == "__main__":
    main()