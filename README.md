# local-LLM-runtime

General-purpose infrastructure for running LLMs locally on your own GPU. Meant to be a reusable
foundation for other local AI projects (RAG pipelines, agents, etc.) — those projects talk to a
stable `LLMClient` interface or a local HTTP API, not to a specific runtime.

## My setup

- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM (driver 610.88, compute capability 8.6)
- OS: Windows 11 Pro
- Python: 3.13
- Runtime: [Ollama](https://ollama.com) — chosen over llama.cpp/vLLM because it ships prebuilt CUDA
  binaries (no CUDA toolkit build step needed), has first-class native Windows support, and already
  exposes a REST API and model manager. vLLM is effectively Linux/WSL2-only and built for batched
  multi-request serving, which is overkill for a single local dev box; raw llama.cpp offers more
  low-level control at the cost of managing GGUF files and CUDA builds yourself.
- Default model: `llama3.1:8b` (~4.9 GB at Q4_K_M — comfortable fit with headroom on 12 GB VRAM)

## Project layout

```
config/config.yaml              # model name, context length, runtime settings
src/llm_runtime/
  client.py                     # LLMClient ABC + get_client() factory (swap runtimes here)
  config.py                     # loads config.yaml into a pydantic AppConfig
  schemas.py                    # shared request/response models
  providers/ollama_client.py    # OllamaClient implementation
  api/server.py                 # FastAPI wrapper exposing the client over HTTP
scripts/
  check_gpu.py                  # confirms the GPU/VRAM is visible via NVML
  test_gpu_inference.py         # pulls the configured model, runs a prompt, confirms GPU (not CPU)
```

## Install

1. Install [Ollama for Windows](https://ollama.com/download) (installs a background server + `ollama` CLI):
   ```powershell
   Invoke-WebRequest -Uri https://ollama.com/download/OllamaSetup.exe -OutFile "$env:TEMP\OllamaSetup.exe"
   Start-Process -FilePath "$env:TEMP\OllamaSetup.exe" -Wait
   ```
2. Create a virtual environment and install this package:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   ```
   or, for an exact reproducible environment:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

## Run a model and confirm GPU usage

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu.py
.\.venv\Scripts\python.exe scripts\test_gpu_inference.py
```

`test_gpu_inference.py` pulls the model from `config/config.yaml` if it isn't already local, sends a
test prompt through the `LLMClient` abstraction, then checks `ollama ps` to confirm the model's
`PROCESSOR` column shows GPU rather than CPU.

To change the model or runtime settings, edit `config/config.yaml` — no code changes needed:

```yaml
runtime: ollama
model:
  name: llama3.1:8b
  context_length: 8192
```

## Run the local API

```powershell
.\.venv\Scripts\python.exe -m uvicorn llm_runtime.api.server:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` — checks the runtime backend is reachable
- `GET /models` — lists locally available models
- `POST /generate` — `{"prompt": "...", "model": null, "temperature": null, "max_tokens": null}` → `{"text": "...", "model": "..."}`
- `POST /chat` — `{"messages": [{"role": "user", "content": "..."}]}` → `{"text": "...", "model": "..."}`

## Calling this from another project (e.g. a RAG pipeline)

**Over HTTP** (no dependency on this repo, just a running server):

```python
import requests

resp = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Summarize: ..."},
)
print(resp.json()["text"])
```

**In-process**, if the other project imports this package directly:

```python
from llm_runtime import load_config, get_client

client = get_client(load_config("path/to/config.yaml"))
response = client.generate("Summarize: ...")
print(response.text)
```

Either path goes through the same `LLMClient` abstraction, so swapping the backend (e.g. adding a
`llamacpp` or `vllm` provider later) only requires implementing `LLMClient` in
`src/llm_runtime/providers/` and adding a branch in `get_client()` — calling code doesn't change.
