import os
from absl import app
from absl import flags
from tunix.oss import utils as oss_utils

_MODEL_ID = flags.DEFINE_string('model_id', None, 'HF model ID', required=True)
_DOWNLOAD_PATH = flags.DEFINE_string('download_path', None, 'Local download path', required=True)

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  oss_utils.hf_pipeline(_MODEL_ID.value, _DOWNLOAD_PATH.value)

if __name__ == '__main__':
  app.run(main)
