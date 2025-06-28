"""
Google implementation for dazllm
"""

import keyring
import json
import time
import base64
import random
import requests
import os
from io import BytesIO
from typing import Type
from pydantic import BaseModel
from PIL import Image

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmGoogle(Llm):
    """Google implementation"""

    def __init__(self, model: str):
        self.model = model if model != "gemini" else "gemini-2.0-flash"
        self.check_config()

        # Import Google AI client - try multiple approaches
        try:
            # Try the newer google-ai-generativelanguage library first
            from google import genai

            api_key = self._get_api_key()
            self.client = genai.Client(api_key=api_key)
            self._use_new_api = True
        except ImportError:
            try:
                # Fallback to the older google-generativeai library
                import google.generativeai as genai

                api_key = self._get_api_key()
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
                self._use_new_api = False
            except ImportError:
                raise ConfigurationError(
                    "Google AI library not installed. Run either:\n"
                    "  pip install google-ai-generativelanguage  (newer API)\n"
                    "  pip install google-generativeai  (older API)"
                )

    @staticmethod
    def check_config():
        """Check if Google is properly configured"""
        api_key = keyring.get_password("dazllm", "google_api_key")
        if not api_key:
            raise ConfigurationError(
                "Google API key not found in keyring. Set with: keyring set dazllm google_api_key"
            )

    @staticmethod
    def default_model() -> str:
        """Default model for Google"""
        return "gemini-2.0-flash"

    @staticmethod
    def default_for_type(model_type: str) -> str:
        """Get default model for a given type"""
        defaults = {
            "local_small": None,  # Google doesn't have local models
            "local_medium": None,
            "local_large": None,
            "paid_cheap": "gemini-1.5-flash",
            "paid_best": "gemini-2.0-flash",
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured", "image"}  # Added image generation support

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports"""
        return [
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation",  # Image generation model
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-pro",
        ]

    @staticmethod
    def check_config():
        """Check if Google is properly configured"""
        api_key = keyring.get_password("dazllm", "google_api_key")
        if not api_key:
            raise ConfigurationError(
                "Google API key not found in keyring. Set with: keyring set dazllm google_api_key"
            )

    def _get_api_key(self) -> str:
        """Get Google API key from keyring"""
        api_key = keyring.get_password("dazllm", "google_api_key")
        if not api_key:
            raise ConfigurationError("Google API key not found in keyring")
        return api_key

    def _normalize_conversation(self, conversation: Conversation) -> str:
        """Convert conversation to Google format"""
        if isinstance(conversation, str):
            return conversation
        else:
            # Convert list format to simple string
            return "\n".join([msg["content"] for msg in conversation])

    def _generate_content(self, messages: str, config: dict):
        """Generate content with retry logic"""
        retries_left = 5
        delay = 1

        while True:
            try:
                from google import genai

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=config,
                )
                return response
            except genai.errors.ServerError as e:
                if e.code == 503:
                    # Retry on server error
                    if retries_left > 0:
                        retries_left -= 1
                        time.sleep(delay)
                        delay *= 2
                        continue
                raise e

    def _extract_json(self, text: str):
        """Extract JSON from response text"""
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                json_str = text[start:end].strip()
                return json.loads(json_str)
            except Exception:
                pass
        try:
            return json.loads(text)
        except Exception:
            raise DazLlmError(f"Could not parse JSON from response: {text}")

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Google AI API"""
        if self._use_new_api:
            return self._chat_new_api(conversation, force_json)
        else:
            return self._chat_old_api(conversation, force_json)

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        if self._use_new_api:
            return self._chat_structured_new_api(conversation, schema, context_size)
        else:
            return self._chat_structured_old_api(conversation, schema, context_size)

    def _chat_new_api(
        self, conversation: Conversation, force_json: bool = False
    ) -> str:
        """Chat using the new google-ai-generativelanguage API"""
        messages = self._normalize_conversation(conversation)

        # Configure response format
        config = {}
        if force_json:
            config = {"response_mime_type": "application/json"}

        try:
            response = self._generate_content(messages, config)

            if force_json:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError:
                    return self._extract_json(response.text)

            return response.text

        except Exception as e:
            raise DazLlmError(f"Google AI API error: {e}")

    def _chat_old_api(
        self, conversation: Conversation, force_json: bool = False
    ) -> str:
        """Chat using the old google-generativeai API"""
        if isinstance(conversation, str):
            prompt = conversation
        else:
            # Convert conversation to simple text for old API
            parts = []
            for msg in conversation:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            prompt = "\n".join(parts)

        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            raise DazLlmError(f"Google AI API error: {e}")

    def _chat_structured_new_api(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Structured chat using the new API"""
        messages = self._normalize_conversation(conversation)

        # Configure with schema
        config = {
            "response_mime_type": "application/json",
            "response_schema": schema,
        }

        try:
            response = self._generate_content(messages, config)

            # Google returns parsed response directly when using schema
            if hasattr(response, "parsed") and response.parsed:
                return response.parsed
            else:
                # Fallback to manual parsing
                content = response.text
                try:
                    data = json.loads(content)
                    return schema(**data)
                except Exception as e:
                    raise DazLlmError(f"Could not create Pydantic model: {e}")

        except Exception as e:
            raise DazLlmError(f"Google structured chat error: {e}")

    def _chat_structured_old_api(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Structured chat using the old API"""
        # Add schema instruction to conversation
        schema_json = schema.model_json_schema()
        schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema_json, indent=2)}"

        if isinstance(conversation, str):
            prompt = conversation + schema_instruction
        else:
            # Convert to simple prompt with schema
            parts = []
            for msg in conversation:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")

            prompt = "\n".join(parts) + schema_instruction

        try:
            response = self.client.generate_content(prompt)
            content = response.text

            # Try to extract and parse JSON
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                else:
                    # Fallback: try parsing entire response
                    data = json.loads(content)

                return schema(**data)
            except json.JSONDecodeError:
                raise DazLlmError(f"Could not parse JSON response: {content}")
            except Exception as e:
                raise DazLlmError(f"Could not create Pydantic model: {e}")

        except Exception as e:
            raise DazLlmError(f"Google structured chat error: {e}")

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate image using Gemini 2.0 Flash Experimental image generation"""

        # Use the image generation model if not already specified
        image_model = self.model
        if "image-generation" not in self.model:
            image_model = "gemini-2.0-flash-exp-image-generation"

        try:
            api_key = self._get_api_key()

            # Gemini image generation API endpoint
            api_url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{image_model}:generateContent"
            )

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            }

            # Include image size in prompt as per working code
            enhanced_prompt = f"{prompt} (Image size: {width}x{height})"

            payload = {
                "contents": [{"parts": [{"text": enhanced_prompt}]}],
                "generationConfig": {
                    "responseModalities": ["TEXT", "IMAGE"],
                    "temperature": 0.4,
                    "topK": 32,
                    "topP": 1.0,
                    "maxOutputTokens": 2048,
                },
            }

            # Retry logic for server errors
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        f"{api_url}?key={api_key}",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )

                    if response.status_code == 200:
                        image = self._extract_image_from_response(
                            response.json(), (width, height)
                        )
                        self._save_image(image, file_name)
                        return file_name
                    elif response.status_code in (500, 503):
                        # Retry on server errors
                        delay = min(2**attempt + random.random(), 60)
                        time.sleep(delay)
                    else:
                        raise DazLlmError(
                            f"Gemini API error {response.status_code}: {response.text}"
                        )

                except requests.exceptions.RequestException as e:
                    delay = min(2**attempt + random.random(), 60)
                    time.sleep(delay)

            raise DazLlmError("Max retries exceeded - Gemini model may be overloaded")

        except Exception as e:
            raise DazLlmError(f"Google image generation error: {e}")

    def _extract_image_from_response(
        self, data: dict, desired_size: tuple[int, int]
    ) -> Image.Image:
        """Extract base64 image from Gemini response and ensure desired size"""
        try:
            parts = data["candidates"][0]["content"]["parts"]
            b64_data = next(p["inlineData"]["data"] for p in parts if "inlineData" in p)
        except (KeyError, IndexError, StopIteration) as e:
            raise DazLlmError("No image data in Gemini response") from e

        # Decode base64 image
        image = Image.open(BytesIO(base64.b64decode(b64_data)))

        # Resize if needed
        if image.size != desired_size:
            image = image.resize(desired_size, Image.Resampling.LANCZOS)

        return image

    def _save_image(self, image: Image.Image, file_name: str):
        """Save image with appropriate format"""
        # Ensure parent directory exists
        parent_dir = os.path.dirname(os.path.abspath(file_name))
        if parent_dir and not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Save image
        if file_name.lower().endswith(".png"):
            image.save(file_name, "PNG")
        else:
            # Convert to RGB if saving as JPEG (removes transparency)
            if image.mode == "RGBA":
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                rgb_image.save(file_name, "JPEG", quality=95)
            else:
                image.save(file_name, "JPEG", quality=95)
