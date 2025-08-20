from pathlib import Path
import yaml

def load_prompt_file(filename: str) -> str:
    """
    Load a prompt YAML file that contains only a single bare string.
    """
    path = Path(__file__).parent / filename
    with open(path, "r", encoding="utf-8") as f:
        prompt = yaml.safe_load(f)
    if not isinstance(prompt, str):
        raise ValueError(f"{filename} does not contain a single string prompt")
    return prompt
