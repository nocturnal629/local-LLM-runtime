import os
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel

DEFAULT_CONFIG_PATH = "config/config.yaml"


class ModelConfig(BaseModel):
    name: str
    context_length: int


class GenerationConfig(BaseModel):
    temperature: float = 0.7
    max_tokens: int = 1024


class OllamaSettings(BaseModel):
    host: str = "http://localhost:11434"
    keep_alive: str = "5m"


class RuntimeSettings(BaseModel):
    ollama: OllamaSettings = OllamaSettings()


class AppConfig(BaseModel):
    runtime: str
    model: ModelConfig
    generation: GenerationConfig
    runtime_settings: RuntimeSettings


def load_config(path: Union[str, Path, None] = None) -> AppConfig:
    """Load AppConfig from a YAML file.

    Resolution order: explicit `path` argument, then the LLM_RUNTIME_CONFIG
    env var, then `config/config.yaml` relative to the current working
    directory. Other projects embedding this package should pass an
    explicit path or set the env var.
    """
    resolved = path or os.environ.get("LLM_RUNTIME_CONFIG", DEFAULT_CONFIG_PATH)
    with open(resolved, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
