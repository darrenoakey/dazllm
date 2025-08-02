"""
Ollama implementation for dazllm
"""

import keyring
import json
import requests
from typing import Type
from pydantic import BaseModel
from jsonschema import validate, ValidationError

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmOllama(Llm):
    """Ollama implementation"""

    def __init__(self, model: str):
        super().__init__(model)
        self.base_url = self._get_base_url()
        self.headers = {"Content-Type": "application/json"}
        self._ensure_model_available()

    @staticmethod
    def default_model() -> str:
        """Default model for Ollama"""
        return "mistral-small"

    @staticmethod
    def default_for_type(model_type: str) -> str:
        """Get default model for a given type"""
        defaults = {
            "local_small": "phi3",
            "local_medium": "mistral-small",
            "local_large": "qwen3:32b",
            "paid_cheap": None,
            "paid_best": None,
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured"}

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports"""
        url = keyring.get_password("dazllm", "ollama_url") or "http://127.0.0.1:11434"
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        return [model["name"] for model in models] if models else []

    @staticmethod
    def check_config():
        """Check if Ollama is properly configured"""
        try:
            base_url = LlmOllama._get_base_url_static()
            response = requests.get(f"{base_url}/api/version", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConfigurationError(f"Ollama not accessible: {e}") from e

    def _get_base_url(self) -> str:
        """Get Ollama base URL from keyring or default"""
        return self._get_base_url_static()

    @staticmethod
    def _get_base_url_static() -> str:
        """Static version of _get_base_url"""
        return keyring.get_password("dazllm", "ollama_url") or "http://127.0.0.1:11434"

    def _ensure_model_available(self):
        """Ensure model is available, pull if necessary"""
        if not self._is_model_available() and not self._pull_model():
            raise ConfigurationError(f"Failed to pull model '{self.model}' from Ollama registry")

    def _is_model_available(self) -> bool:
        """Check if model is available locally"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers, timeout=5)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(model["name"].startswith(self.model) for model in models)
        except (requests.exceptions.RequestException, ValueError, KeyError):
            return False

    def _pull_model(self) -> bool:
        """Pull model from Ollama registry"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": self.model},
                headers=self.headers,
                timeout=300,
            )
            response.raise_for_status()
            return True
        except (requests.exceptions.RequestException, ValueError):
            return False

    def _normalize_conversation(self, conversation: Conversation) -> list:
        """Convert conversation to Ollama message format"""
        return [{"role": "user", "content": conversation}] if isinstance(conversation, str) else conversation

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Ollama API"""
        messages = self._normalize_conversation(conversation)
        data = {"model": self.model, "messages": messages, "stream": False}
        if force_json:
            data["format"] = "json"

        try:
            response = requests.post(f"{self.base_url}/api/chat", json=data, headers=self.headers, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise DazLlmError(f"Ollama API error: {e}") from e
        except KeyError as e:
            raise DazLlmError(f"Unexpected Ollama response structure: {e}") from e

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        messages = self._normalize_conversation(conversation)
        schema_json = schema.model_json_schema()

        system_message = {
            "role": "system",
            "content": (
                f"All responses should be strictly in JSON obeying this schema: {schema_json} "
                "with no accompanying text or delimiters. Do not include the schema in the output. "
                "We want the shortest possible output with no explanations. If there is source code or "
                "other technical output, pay very close attention to proper escaping so the result is valid JSON."
            ),
        }

        conv_with_system = [system_message] + messages
        attempts = 20

        while attempts > 0:
            data = {
                "model": self.model,
                "messages": conv_with_system,
                "stream": False,
                "format": schema_json,
            }
            if context_size > 0:
                data["options"] = {"num_ctx": context_size}

            try:
                response = requests.post(f"{self.base_url}/api/chat", json=data, headers=self.headers, timeout=60)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                parsed_content = self._find_json(content)
                validate(instance=parsed_content, schema=schema_json)
                return schema(**parsed_content)

            except requests.exceptions.RequestException as e:
                raise DazLlmError(f"Ollama API error: {e}") from e
            except json.JSONDecodeError:
                conv_with_system.append({
                    "role": "system",
                    "content": "The previous response was not valid JSON. Please ensure the output is valid JSON strictly following the schema.",
                })
            except ValidationError as e:
                conv_with_system.append({
                    "role": "system",
                    "content": f"Your previous output did not adhere to the JSON schema because: {e}. Please generate a response that strictly follows the schema without any extra text or formatting.",
                })
            except KeyError as e:
                raise DazLlmError(f"Unexpected Ollama response structure: {e}") from e

            attempts -= 1

        raise DazLlmError("Failed to get valid structured response after multiple attempts")

    def _find_json(self, text: str) -> dict:
        """Extract JSON from text response"""
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            json_text = text[start:end].strip()
        else:
            json_text = text
        return json.loads(json_text)

    def image(self, prompt: str, file_name: str, width: int = 1024, height: int = 1024) -> str:
        """Generate image using Ollama (not supported by default)"""
        raise DazLlmError("Image generation not supported by Ollama. Use OpenAI or other providers for image generation.")


import unittest
class TestLlmOllama(unittest.TestCase):
    def test_default_model(self):
        self.assertEqual(LlmOllama.default_model(), "mistral-small")
