from abc import ABC, abstractmethod
from typing import Optional

from llm_runtime.config import AppConfig
from llm_runtime.schemas import ChatMessage, GenerateResponse


class LLMClient(ABC):
    """Runtime-agnostic interface for local LLM inference.

    Concrete backends (Ollama, and later llama.cpp/vLLM) implement this so
    calling code never depends on a specific runtime's API shape.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateResponse: ...

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GenerateResponse: ...

    @abstractmethod
    def list_models(self) -> list[str]: ...

    @abstractmethod
    def health(self) -> bool: ...


def get_client(config: AppConfig) -> LLMClient:
    """Factory that picks the backend implementation based on config.runtime."""
    if config.runtime == "ollama":
        from llm_runtime.providers.ollama_client import OllamaClient

        return OllamaClient(config)
    raise ValueError(f"Unsupported runtime: {config.runtime!r}")
