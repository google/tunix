from tunix.generate import sampler

try:
  # This should fail with AttributeError
  sampler.generate_text("test", max_tokens=10)
except AttributeError as e:
  print(f"Caught expected error: {e}")
except Exception as e:
  print(f"Caught unexpected error: {e}")
