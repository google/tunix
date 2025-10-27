import inspect
from typing import Type

from tunix import models
# Static imports of all allowed models
from tunix.models import t5


# Static mapping of model names to their modules
MODEL_MODULES = {
    't5': t5,
}


def get_model_class(model_name: str) -> Type[models.Model]:
  """Returns the model class for a given model name.

  Args:
    model_name: The name of the model.

  Returns:
    The model class.
  """
  # The model name can be a full HuggingFace model name, like "t5-small". We
  # want to extract the base model name, which is "t5".
  base_model_name = model_name.split('-')[0]
  
  if base_model_name not in MODEL_MODULES:
    raise ValueError(
        f'Unsupported model: {model_name}. '
        f'Allowed models: {sorted(list(MODEL_MODULES.keys()))}'
    )

  module = MODEL_MODULES[base_model_name]
  for _, member in inspect.getmembers(module, inspect.isclass):
    if issubclass(member, models.Model) and member is not models.Model:
      return member
  raise ValueError(f'Could not find model class in module: {module.__name__}')
