"""Pull the configured model if needed, run a test prompt, and confirm it ran on GPU (not CPU)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_runtime.client import get_client  # noqa: E402
from llm_runtime.config import load_config  # noqa: E402


def ensure_model_pulled(model: str) -> None:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    if model not in result.stdout:
        print(f"Model {model} not found locally, pulling...")
        subprocess.run(["ollama", "pull", model], check=True)
    else:
        print(f"Model {model} already present.")


def model_is_on_gpu(model: str) -> bool:
    result = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
    print(result.stdout)
    for line in result.stdout.splitlines()[1:]:
        if line.split() and line.split()[0] == model:
            return "GPU" in line
    return False


def main() -> int:
    config = load_config()
    model = config.model.name

    ensure_model_pulled(model)

    client = get_client(config)
    print("Sending test prompt...")
    response = client.generate("In one sentence, what is a large language model?")
    print(f"Response: {response.text.strip()}")

    if model_is_on_gpu(model):
        print("Confirmed: model is running on GPU.")
        return 0

    print("WARNING: model does not appear to be running on GPU — check `ollama ps` output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
