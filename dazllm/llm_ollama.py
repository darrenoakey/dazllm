"""
Ollama implementation for dazllm
"""

import keyring
import json
import requests
import subprocess
from typing import Type, Optional
from pydantic import BaseModel
from jsonschema import validate, ValidationError

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmOllama(Llm):
    """Ollama implementation"""

    def __init__(self, model: str):
        self.model = model
        self.base_url = self._get_base_url()
        self.headers = {"Content-Type": "application/json"}

        # Ensure model is available
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
            "paid_cheap": None,  # Ollama is local, no paid models
            "paid_best": None,
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured"}

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports (queries actual Ollama if possible)"""
        try:
            # Try to get actual installed models from Ollama
            url = (
                keyring.get_password("dazllm", "ollama_url") or "http://127.0.0.1:11434"
            )
            response = requests.get(f"{url}/api/tags", timeout=5)
            response.raise_for_status()

            models = response.json().get("models", [])
            installed_models = [model["name"] for model in models]

            if installed_models:
                return installed_models
        except Exception:
            raise

    @staticmethod
    def check_config():
        """Check if Ollama is properly configured"""
        try:
            base_url = LlmOllama._get_base_url_static()
            response = requests.get(f"{base_url}/api/version", timeout=5)
            response.raise_for_status()
        except Exception as e:
            raise ConfigurationError(f"Ollama not accessible: {e}")

    def _get_base_url(self) -> str:
        """Get Ollama base URL from keyring or default"""
        return self._get_base_url_static()

    @staticmethod
    def _get_base_url_static() -> str:
        """Static version of _get_base_url"""
        url = keyring.get_password("dazllm", "ollama_url")
        return url or "http://127.0.0.1:11434"

    def context_length(self) -> int:
        """Return context window from Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                headers=self.headers,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            if "details" in data and "context_length" in data["details"]:
                return data["details"]["context_length"]
            if "context_length" in data:
                return data["context_length"]
            raise DazLlmError("Context length not found in Ollama response")
        except Exception as e:
            raise DazLlmError(f"Ollama context length error: {e}")

    def _ensure_model_available(self):
        """Ensure model is available, pull if necessary"""
        if not self._is_model_available():
            if not self._pull_model():
                raise ConfigurationError(
                    f"Failed to pull model '{self.model}' from Ollama registry"
                )

    def _is_model_available(self) -> bool:
        """Check if model is available locally"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(model["name"].startswith(self.model) for model in models)
        except Exception:
            return False

    def _pull_model(self) -> bool:
        """Pull model from Ollama registry"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": self.model},
                headers=self.headers,
                timeout=300,  # 5 minute timeout for model pulling
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _normalize_conversation(self, conversation: Conversation) -> list:
        """Convert conversation to Ollama message format"""
        if isinstance(conversation, str):
            return [{"role": "user", "content": conversation}]
        return conversation

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Ollama API"""
        messages = self._normalize_conversation(conversation)

        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        if force_json:
            data["format"] = "json"

        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=data, headers=self.headers
            )
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise DazLlmError(f"Ollama API error: {e}")
        except KeyError as e:
            raise DazLlmError(f"Unexpected Ollama response structure: {e}")

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        messages = self._normalize_conversation(conversation)
        schema_json = schema.model_json_schema()

        # Add system message with schema instructions
        system_message = {
            "role": "system",
            "content": (
                f"All responses should be strictly in JSON obeying this schema: {schema_json} "
                "with no accompanying text or delimiters. Do not include the schema in the output. "
                "We want the shortest possible output with no explanations. If there is source code or "
                "other technical output, pay very close attention to proper escaping so the result is valid JSON."
            ),
        }

        conversation_with_system = [system_message] + messages

        attempt_count = 20
        while attempt_count > 0:
            data = {
                "model": self.model,
                "messages": conversation_with_system,
                "stream": False,
                "format": schema_json,
            }

            if context_size > 0:
                data["options"] = {"num_ctx": context_size}

            try:
                response = requests.post(
                    f"{self.base_url}/api/chat", json=data, headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                content = result["message"]["content"]

                # Parse JSON
                parsed_content = self._find_json(content)

                # Validate against schema
                validate(instance=parsed_content, schema=schema_json)

                # Create Pydantic model
                return schema(**parsed_content)

            except requests.exceptions.RequestException as e:
                raise DazLlmError(f"Ollama API error: {e}")
            except json.JSONDecodeError as e:
                conversation_with_system.append(
                    {
                        "role": "system",
                        "content": "The previous response was not valid JSON. Please ensure the output is valid JSON strictly following the schema.",
                    }
                )
            except ValidationError as e:
                conversation_with_system.append(
                    {
                        "role": "system",
                        "content": (
                            f"Your previous output did not adhere to the JSON schema because: {e}. "
                            "Please generate a response that strictly follows the schema without any extra text or formatting."
                        ),
                    }
                )
            except KeyError as e:
                raise DazLlmError(f"Unexpected Ollama response structure: {e}")

            attempt_count -= 1

        raise DazLlmError(
            "Failed to get valid structured response after multiple attempts"
        )

    def _find_json(self, text: str) -> dict:
        """Extract JSON from text response"""
        if "```json" in text:
            start = text.index("```json") + len("```json")
            end = text.index("```", start)
            json_text = text[start:end].strip()
        else:
            json_text = text
        return json.loads(json_text)

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate image using Ollama (not supported by default)"""
        raise DazLlmError(
            "Image generation not supported by Ollama. Use OpenAI or other providers for image generation."
        )
