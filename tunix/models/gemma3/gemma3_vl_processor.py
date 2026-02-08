"""Gemma3 vision-language processor."""

from tunix.processors import vl_processor


class Gemma3VLProcessor(vl_processor.VLProcessor):
  """Gemma3 vision-language processor."""

  def preprocess_text(self, text: str):
    """Preprocess text."""
    num_mm_tokens_per_image = self.config.num_mm_tokens_per_image
    start_of_image_token = self.config.start_of_image_token
    end_of_image_token = self.config.end_of_image_token
    image_placeholder = self.config.soft_token_placeholder_token
    double_new_line_token = self.config.double_new_line_token

    processed_text = text.replace(
        start_of_image_token,
        f"{double_new_line_token}{start_of_image_token}"
        + image_placeholder * num_mm_tokens_per_image
        + f"{end_of_image_token}{double_new_line_token}",
    )
    return processed_text
