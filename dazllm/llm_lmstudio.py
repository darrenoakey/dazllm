"""LM Studio provider for dazllm"""

from __future__ import annotations

import json
import keyring
import requests
from typing import Type
from pydantic import BaseModel

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmLmstudio(Llm):
    """LM Studio implementation using OpenAI compatible API"""

    def __init__(self, model: str):
        self.model = model
        self.base_url = self._get_base_url()
        self.headers = {"Content-Type": "application/json"}
        self.check_config()

    @staticmethod
    def default_model() -> str:
        try:
            models = LlmLmstudio.supported_models()
            if models:
                return models[0]
        except Exception:
            pass
        return "lmstudio"

    @staticmethod
    def default_for_type(model_type: str) -> str:
        defaults = {
            "local_small": None,
            "local_medium": None,
            "local_large": None,
            "paid_cheap": None,
            "paid_best": None,
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        return {"chat", "structured"}

    @staticmethod
    def supported_models() -> list[str]:
        try:
            base_url = LlmLmstudio._get_base_url_static()
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
            data = response.json().get("data", [])
            return [m.get("id") for m in data]
        except Exception:
            return []

    @staticmethod
    def check_config():
        try:
            base_url = LlmLmstudio._get_base_url_static()
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConfigurationError(f"LM Studio not accessible: {e}")

    def _get_base_url(self) -> str:
        return self._get_base_url_static()

    @staticmethod
    def _get_base_url_static() -> str:
        url = keyring.get_password("dazllm", "lmstudio_url")
        return url or "http://127.0.0.1:1234"

    def _normalize_conversation(self, conversation: Conversation) -> list:
        if isinstance(conversation, str):
            return [{"role": "user", "content": conversation}]
        return conversation

    def _request(self, payload: dict) -> dict:
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise DazLlmError(f"LM Studio API error: {e}")

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        messages = self._normalize_conversation(conversation)
        payload = {"model": self.model, "messages": messages}
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        data = self._request(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise DazLlmError(f"Unexpected LM Studio response: {e}")

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        messages = self._normalize_conversation(conversation)
        schema_json = schema.model_json_schema()
        system_message = {
            "role": "system",
            "content": "Respond with JSON matching this schema:\n"
            + json.dumps(schema_json, indent=2),
        }
        messages = [system_message] + messages
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if context_size:
            payload["max_tokens"] = context_size
        data = self._request(payload)
        try:
            content = data["choices"][0]["message"]["content"]
            return schema(**json.loads(content))
        except Exception as e:
            raise DazLlmError(f"LM Studio structured chat error: {e}")

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        raise DazLlmError("Image generation not supported by LM Studio")
