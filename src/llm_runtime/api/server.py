from fastapi import FastAPI, HTTPException

from llm_runtime.client import get_client
from llm_runtime.config import load_config
from llm_runtime.schemas import ChatRequest, GenerateRequest, GenerateResponse

app = FastAPI(title="local-LLM-runtime", version="0.1.0")

_config = load_config()
_client = get_client(_config)


@app.get("/health")
def health():
    if not _client.health():
        raise HTTPException(status_code=503, detail="runtime backend unreachable")
    return {"status": "ok", "runtime": _config.runtime}


@app.get("/models")
def models():
    return {"models": _client.list_models()}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    return _client.generate(
        req.prompt,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )


@app.post("/chat", response_model=GenerateResponse)
def chat(req: ChatRequest):
    return _client.chat(
        req.messages,
        model=req.model,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
