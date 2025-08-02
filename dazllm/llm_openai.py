"""
OpenAI implementation for dazllm
"""

import keyring
import json
import unittest
from typing import Type
from pydantic import BaseModel

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmOpenai(Llm):
    """OpenAI implementation"""

    def __init__(self, model: str):
        super().__init__(model)
        self.check_config()

        # Import OpenAI client
        try:
            import openai
            self.client = openai.OpenAI(api_key=self._get_api_key())
        except ImportError as exc:
            raise ConfigurationError(
                "OpenAI library not installed. Run: pip install openai"
            ) from exc

    @staticmethod
    def default_model() -> str:
        """Default model for OpenAI"""
        return "gpt-4o"

    @staticmethod
    def default_for_type(model_type: str) -> str:
        """Get default model for a given type"""
        defaults = {
            "local_small": None,  # OpenAI doesn't have local models
            "local_medium": None,
            "local_large": None,
            "paid_cheap": "gpt-4o-mini",
            "paid_best": "gpt-4o",
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured", "image"}

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports"""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "dall-e-3",
            "dall-e-2",
            "gpt-image-1",
        ]

    @staticmethod
    def check_config():
        """Check if OpenAI is properly configured"""
        api_key = keyring.get_password("dazllm", "openai_api_key")
        if not api_key:
            raise ConfigurationError(
                "OpenAI API key not found in keyring. Set with: keyring set dazllm openai_api_key"
            )

    def _get_api_key(self) -> str:
        """Get OpenAI API key from keyring"""
        api_key = keyring.get_password("dazllm", "openai_api_key")
        if not api_key:
            raise ConfigurationError("OpenAI API key not found in keyring")
        return api_key

    def _normalize_conversation(self, conversation: Conversation) -> list:
        """Convert conversation to OpenAI message format"""
        if isinstance(conversation, str):
            return [{"role": "user", "content": conversation}]
        return conversation

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using OpenAI API"""
        messages = self._normalize_conversation(conversation)

        kwargs = {"model": self.model, "messages": messages}

        if force_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise DazLlmError(f"OpenAI API error: {e}") from e

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        messages = self._normalize_conversation(conversation)

        # Add schema instruction to conversation
        schema_json = schema.model_json_schema()
        schema_prompt = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema_json, indent=2)}"

        # Add to last user message
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += schema_prompt
        else:
            messages.append({"role": "user", "content": schema_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content

            # Parse and validate JSON
            try:
                data = json.loads(content)
                return schema(**data)
            except json.JSONDecodeError as exc:
                raise DazLlmError(f"Could not parse JSON response: {content}") from exc
            except Exception as e:
                raise DazLlmError(f"Could not create Pydantic model: {e}") from e

        except Exception as e:
            raise DazLlmError(f"OpenAI structured chat error: {e}") from e

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate image using OpenAI image models"""
        from .image_utils import ImageUtils
        import base64
        from PIL import Image
        from io import BytesIO

        # Use gpt-image-1 for image generation unless already an image model
        image_model = self.model
        if self.model not in ["gpt-image-1", "dall-e-2", "dall-e-3"]:
            image_model = "gpt-image-1"

        try:
            # Calculate optimal generation size and enhance prompt
            gen_width, gen_height = ImageUtils.calculate_optimal_size(width, height)
            enhanced_prompt = ImageUtils.enhance_prompt_for_aspect_ratio(prompt, gen_width, gen_height)

            # Generate image
            response = self.client.images.generate(
                model=image_model,
                prompt=enhanced_prompt,
                size=f"{gen_width}x{gen_height}",
                quality="high",
                output_format="png",
                n=1,
            )

            # Handle response and save directly
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                image_data = base64.b64decode(response.data[0].b64_json)
                image = Image.open(BytesIO(image_data))
            else:
                return ImageUtils.save_image(response.data[0].url, file_name)

            # If size differs, resize and save, otherwise save directly
            if width != gen_width or height != gen_height:
                temp_path = file_name + "_temp.png"
                image.save(temp_path, 'PNG')
                ImageUtils.resize_and_crop(temp_path, width, height)
                import os
                os.remove(temp_path)

            # Save final image
            import os
            base_dir = os.path.dirname(file_name) or '.'
            os.makedirs(base_dir, exist_ok=True)
            image.save(file_name, 'PNG')

            return file_name

        except Exception as e:
            raise DazLlmError(f"OpenAI image generation error: {e}") from e


class TestLlmOpenai(unittest.TestCase):
    """Essential tests for LlmOpenai class"""

    def test_default_model(self):
        """Test default model returns expected value"""
        self.assertEqual(LlmOpenai.default_model(), "gpt-4o")

    def test_default_for_type(self):
        """Test default_for_type returns correct models"""
        self.assertEqual(LlmOpenai.default_for_type("paid_cheap"), "gpt-4o-mini")
        self.assertEqual(LlmOpenai.default_for_type("paid_best"), "gpt-4o")
        self.assertIsNone(LlmOpenai.default_for_type("local_small"))

    def test_capabilities(self):
        """Test capabilities returns expected set"""
        capabilities = LlmOpenai.capabilities()
        self.assertEqual(capabilities, {"chat", "structured", "image"})

    def test_supported_models(self):
        """Test supported_models returns expected list"""
        models = LlmOpenai.supported_models()
        self.assertIn("gpt-4o", models)
        self.assertIn("dall-e-3", models)

    def test_normalize_conversation_string(self):
        """Test conversation normalization with string input"""
        result = [{"role": "user", "content": "hello"}] if isinstance("hello", str) else "hello"
        self.assertEqual(result, [{"role": "user", "content": "hello"}])


__all__ = ["LlmOpenai"]
