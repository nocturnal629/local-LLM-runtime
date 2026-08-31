from typing import Optional

import requests

from llm_runtime.client import LLMClient
from llm_runtime.config import AppConfig
from llm_runtime.schemas import ChatMessage, GenerateResponse


class OllamaClient(LLMClient):
    """Talks to a local Ollama server over its REST API."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._host = config.runtime_settings.ollama.host.rstrip("/")
        self._keep_alive = config.runtime_settings.ollama.keep_alive
        self._default_model = config.model.name

    def _options(self, temperature: Optional[float], max_tokens: Optional[int]) -> dict:
        return {
            "temperature": temperature if temperature is not None else self._config.generation.temperature,
            "num_predict": max_tokens if max_tokens is not None else self._config.generation.max_tokens,
            "num_ctx": self._config.model.context_length,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateResponse:
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        resp = requests.post(f"{self._host}/api/generate", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return GenerateResponse(text=data["response"], model=resolved_model)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateResponse:
        resolved_model = model or self._default_model
        payload = {
            "model": resolved_model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "keep_alive": self._keep_alive,
            "options": self._options(temperature, max_tokens),
        }
        resp = requests.post(f"{self._host}/api/chat", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return GenerateResponse(text=data["message"]["content"], model=resolved_model)

    def list_models(self) -> list[str]:
        resp = requests.get(f"{self._host}/api/tags", timeout=30)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def health(self) -> bool:
        try:
            resp = requests.get(self._host, timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False
