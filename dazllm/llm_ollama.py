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
import re


class LlmOllama(Llm):
    """Ollama implementation"""

    # Models that don't support the 'format' parameter and need special handling
    MODELS_WITHOUT_FORMAT_SUPPORT = [
        "gpt-oss:20b",
        # Add more models here as needed
    ]

    def __init__(self, model: str):
        super().__init__(model)
        self.base_url = self._get_base_url()
        self.headers = {"Content-Type": "application/json"}
        self._ensure_model_available()
        self._supports_format = self._check_format_support()

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

    def _check_format_support(self) -> bool:
        """Check if the model supports the 'format' parameter"""
        # Check against our list of known models without format support
        for model_pattern in self.MODELS_WITHOUT_FORMAT_SUPPORT:
            if self.model == model_pattern or self.model.startswith(model_pattern.split(':')[0]):
                return False
        return True

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

        if self._supports_format:
            # Use the standard format parameter approach
            return self._chat_structured_with_format(messages, schema, schema_json, context_size)
        else:
            # Use manual schema injection for models without format support
            return self._chat_structured_without_format(messages, schema, schema_json, context_size)

    def _chat_structured_with_format(
        self, messages: list, schema: Type[BaseModel], schema_json: dict, context_size: int
    ) -> BaseModel:
        """Standard structured output using format parameter"""
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

    def _chat_structured_without_format(
        self, messages: list, schema: Type[BaseModel], schema_json: dict, context_size: int
    ) -> BaseModel:
        """Structured output for models without format support"""
        # Inject schema instructions directly into the prompt
        schema_str = json.dumps(schema_json, indent=2)
        
        system_message = {
            "role": "system",
            "content": (
                "You must respond with valid JSON that matches the following schema exactly. "
                "Output ONLY the JSON, wrapped in ```json code blocks. "
                "Do not include any other text, explanations, or the schema itself.\n\n"
                f"Required JSON Schema:\n{schema_str}"
            ),
        }

        # Add schema reminder to the last user message
        modified_messages = messages.copy()
        if modified_messages and modified_messages[-1]["role"] == "user":
            modified_messages[-1] = {
                "role": "user",
                "content": (
                    f"{modified_messages[-1]['content']}\n\n"
                    f"Remember: Return your response as valid JSON matching this schema:\n```json\n{schema_str}\n```"
                ),
            }

        conv_with_system = [system_message] + modified_messages
        attempts = 20

        while attempts > 0:
            data = {
                "model": self.model,
                "messages": conv_with_system,
                "stream": False,
            }
            if context_size > 0:
                data["options"] = {"num_ctx": context_size}

            try:
                response = requests.post(f"{self.base_url}/api/chat", json=data, headers=self.headers, timeout=60)
                response.raise_for_status()
                content = response.json()["message"]["content"]
                
                # Try multiple parsing strategies
                parsed_content = self._parse_json_with_fallbacks(content)
                validate(instance=parsed_content, schema=schema_json)
                return schema(**parsed_content)

            except requests.exceptions.RequestException as e:
                raise DazLlmError(f"Ollama API error: {e}") from e
            except (json.JSONDecodeError, ValueError) as e:
                conv_with_system.append({
                    "role": "system",
                    "content": f"Invalid JSON in previous response. Error: {e}. Please output valid JSON wrapped in ```json blocks.",
                })
            except ValidationError as e:
                conv_with_system.append({
                    "role": "system",
                    "content": f"JSON doesn't match schema. Error: {e}. Please fix and return valid JSON matching the schema.",
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

    def _parse_json_with_fallbacks(self, text: str) -> dict:
        """Parse JSON with multiple fallback strategies"""
        # Strategy 1: Try to find JSON in markdown code blocks
        if "```json" in text:
            try:
                return self._find_json(text)
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 2: Try to parse the entire text as JSON
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 3: Look for JSON-like structures with regex
        json_patterns = [
            r'\{[^{}]*\}',  # Simple single-level JSON
            r'\{.*\}',      # Any JSON object
            r'\[.*\]',      # JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue

        # Strategy 4: Clean up common issues and retry
        cleaned = text.strip()
        # Remove common prefixes/suffixes
        prefixes = ["Here is the JSON:", "JSON:", "Output:", "Result:"]
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove trailing punctuation
        cleaned = cleaned.rstrip('.,;:')
        
        # Try parsing cleaned text
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # If all strategies fail, raise an error with helpful context
        raise ValueError(f"Could not parse JSON from response. Text: {text[:500]}...")

    def image(self, prompt: str, file_name: str, width: int = 1024, height: int = 1024) -> str:
        """Generate image using Ollama (not supported by default)"""
        raise DazLlmError("Image generation not supported by Ollama. Use OpenAI or other providers for image generation.")
    
    def get_context_length(self) -> int:
        """Get the context length for the current Ollama model"""
        try:
            # Try to get model info from Ollama API
            response = requests.get(
                f"{self.base_url}/api/show",
                json={"name": self.model},
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                model_info = response.json()
                # Check if context length is in the model info
                if "parameters" in model_info:
                    params = model_info["parameters"]
                    # Look for common parameter names for context length
                    for key in ["num_ctx", "context_length", "max_context_length", "context_size"]:
                        if key in params:
                            try:
                                return int(params[key])
                            except (ValueError, TypeError):
                                continue
                
                # Check modelfile for context length settings
                if "modelfile" in model_info:
                    modelfile = model_info["modelfile"]
                    # Look for PARAMETER num_ctx lines
                    import re
                    match = re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
            
        except Exception:
            pass  # Fall back to defaults
        
        # Fallback to known context lengths for common models
        context_lengths = {
            # Common Ollama models and their typical context lengths
            "mistral-small": 32768,      # Mistral models typically 32K
            "mistral": 32768,
            "mistral-large": 32768,
            "llama3": 8192,              # Llama 3 models
            "llama3:8b": 8192,
            "llama3:70b": 8192,
            "llama2": 4096,              # Llama 2 models
            "llama2:7b": 4096,
            "llama2:13b": 4096,
            "llama2:70b": 4096,
            "phi3": 4096,                # Phi-3 models
            "phi3:3.8b": 4096,
            "phi3:14b": 4096,
            "qwen3": 32768,              # Qwen models
            "qwen3:32b": 32768,
            "codestral": 32768,          # Code models
            "codeqwen": 32768,
            "gpt-oss:20b": 4096,         # GPT-OSS model
        }
        
        # Try exact match first
        if self.model in context_lengths:
            return context_lengths[self.model]
        
        # Try pattern matching for model families
        model_lower = self.model.lower()
        if "mistral" in model_lower:
            return 32768
        elif "llama3" in model_lower:
            return 8192
        elif "llama2" in model_lower:
            return 4096
        elif "phi3" in model_lower:
            return 4096
        elif "qwen" in model_lower:
            return 32768
        elif "code" in model_lower:
            return 32768
        
        # Conservative default for unknown models
        return 4096


import unittest
class TestLlmOllama(unittest.TestCase):
    def test_default_model(self):
        self.assertEqual(LlmOllama.default_model(), "mistral-small")
