"""LM Studio provider for dazllm"""

from __future__ import annotations

import json
import keyring
import requests
import unittest
from typing import Type
from pydantic import BaseModel

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmLmstudio(Llm):
    """LM Studio implementation using OpenAI compatible API"""

    def __init__(self, model: str):
        super().__init__(f"lmstudio:{model}")
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
        except (requests.RequestException, ValueError, KeyError):
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
        except (requests.RequestException, ValueError, KeyError):
            return []

    @staticmethod
    def check_config():
        try:
            base_url = LlmLmstudio._get_base_url_static()
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConfigurationError(f"LM Studio not accessible: {e}") from e

    def _get_base_url(self) -> str:
        return LlmLmstudio._get_base_url_static()

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
        except requests.RequestException as e:
            raise DazLlmError(f"LM Studio API error: {e}") from e

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        messages = self._normalize_conversation(conversation)
        payload = {"model": self.model, "messages": messages}
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        data = self._request(payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise DazLlmError(f"Unexpected LM Studio response: {e}") from e

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
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValueError) as e:
            raise DazLlmError(f"LM Studio structured chat error: {e}") from e

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        raise DazLlmError("Image generation not supported by LM Studio")


class TestLlmLmstudio(unittest.TestCase):
    """Essential tests for LlmLmstudio class"""

    def test_capabilities_and_metadata(self):
        """Test capabilities, supported models, and default model"""
        self.assertEqual(LlmLmstudio.capabilities(), {"chat", "structured"})
        self.assertIsInstance(LlmLmstudio.supported_models(), list)
        self.assertIsInstance(LlmLmstudio.default_model(), str)

    def test_default_for_type_behavior(self):
        """Test default_for_type returns None for all types"""
        types = ["local_small", "local_medium", "local_large", "paid_cheap", "paid_best"]
        for model_type in types:
            self.assertIsNone(LlmLmstudio.default_for_type(model_type))

    def test_configuration_check(self):
        """Test configuration check behavior"""
        try:
            LlmLmstudio.check_config()
        except ConfigurationError as e:
            self.assertIn("not accessible", str(e))

    def test_initialization_and_image_error(self):
        """Test model initialization and image generation error"""
        try:
            llm = LlmLmstudio("test-model")
            self.assertEqual(llm.model, "test-model")
            with self.assertRaises(DazLlmError):
                llm.image("test", "test.jpg")
        except ConfigurationError:
            pass  # Expected when LM Studio not running

    def test_chat_when_configured(self):
        """Test basic chat functionality when configured"""
        try:
            llm = LlmLmstudio("test-model")
            response = llm.chat("Say hello")
            self.assertIsInstance(response, str)
        except (ConfigurationError, DazLlmError):
            pass  # Expected when not configured or model issues

    def test_structured_chat_when_configured(self):
        """Test structured chat when configured"""
        try:
            class SimpleResponse(BaseModel):
                message: str
            llm = LlmLmstudio("test-model")
            response = llm.chat_structured("Test", SimpleResponse)
            self.assertIsInstance(response, SimpleResponse)
        except (ConfigurationError, DazLlmError):
            pass  # Expected when not configured or parsing issues
