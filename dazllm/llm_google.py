"""
Google implementation for dazllm
"""

import keyring
import json
import unittest
from typing import Type
from pydantic import BaseModel

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmGoogle(Llm):
    """Google implementation"""

    def __init__(self, model: str):
        super().__init__(f"google:{model}")
        self.model = model if model != "gemini" else "gemini-2.0-flash"
        self.check_config()

        try:
            import google.generativeai as genai
            api_key = self._get_api_key()
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError as exc:
            raise ConfigurationError(
                "Google AI library not installed. Run: pip install google-generativeai"
            ) from exc

    @staticmethod
    def default_model() -> str:
        """Default model for Google"""
        return "gemini-2.0-flash"

    @staticmethod
    def default_for_type(model_type: str) -> str:
        """Get default model for a given type"""
        defaults = {
            "local_small": None,
            "local_medium": None,
            "local_large": None,
            "paid_cheap": "gemini-1.5-flash",
            "paid_best": "gemini-2.0-flash",
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
            "gemini-2.0-flash",
            "gemini-2.0-flash-exp-image-generation",
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
            return "\n".join([msg["content"] for msg in conversation])

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Google AI API"""
        prompt = self._normalize_conversation(conversation)
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            raise DazLlmError(f"Google AI API error: {e}") from e

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        schema_json = schema.model_json_schema()
        schema_instruction = f"\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema_json, indent=2)}"
        
        if isinstance(conversation, str):
            prompt = conversation + schema_instruction
        else:
            parts = []
            for msg in conversation:
                parts.append(f"{msg['role']}: {msg['content']}")
            prompt = "\n".join(parts) + schema_instruction

        try:
            response = self.client.generate_content(prompt)
            content = response.text

            # Try to extract and parse JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
            else:
                data = json.loads(content)

            return schema(**data)
        except json.JSONDecodeError as exc:
            raise DazLlmError(f"Could not parse JSON response: {content}") from exc
        except Exception as e:
            raise DazLlmError(f"Google structured chat error: {e}") from e

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate image using Google AI with nano model"""
        from .image_utils import ImageUtils
        from PIL import Image
        from io import BytesIO
        import os

        # Use the nano banana model for image generation
        image_model_name = "gemini-2.5-flash-image-preview"

        try:
            import google.generativeai as genai
            # Create a new model instance for image generation
            image_model = genai.GenerativeModel(image_model_name)

            # Calculate optimal generation size and enhance prompt
            gen_width, gen_height = ImageUtils.calculate_optimal_size(width, height)
            enhanced_prompt = ImageUtils.enhance_prompt_for_aspect_ratio(prompt, gen_width, gen_height)

            # Add size hint to prompt
            size_prompt = f"{enhanced_prompt} Image dimensions: {gen_width}x{gen_height}"

            # Generate image using Gemini's generate_content method
            response = image_model.generate_content(size_prompt)

            # Check if response contains image data
            if response.parts and len(response.parts) > 0:
                for part in response.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Extract image data from response
                        if hasattr(part.inline_data, 'data') and part.inline_data.data:
                            # The data is already in bytes format, not base64
                            image_data = part.inline_data.data
                            image = Image.open(BytesIO(image_data))

                            # Ensure output directory exists
                            base_dir = os.path.dirname(file_name) or '.'
                            os.makedirs(base_dir, exist_ok=True)

                            # Resize if needed
                            if width != image.width or height != image.height:
                                # Resize to target dimensions
                                image = image.resize((width, height), Image.Resampling.LANCZOS)

                            # Save the image
                            image.save(file_name, 'PNG')
                            return file_name

            # If no image data found in response, try text-based generation
            # Fallback to generating an image description and returning error
            raise DazLlmError(f"No image data found in response from {image_model_name}")

        except ImportError as exc:
            raise DazLlmError(
                "Google AI library not installed. Run: pip install google-generativeai"
            ) from exc
        except Exception as e:
            raise DazLlmError(f"Google image generation error: {e}") from e
    
    def get_context_length(self) -> int:
        """Get the context length for the current Gemini model"""
        # Known context lengths for Gemini models
        context_lengths = {
            "gemini-2.0-flash": 1000000,      # 1M tokens
            "gemini-1.5-pro": 2000000,        # 2M tokens
            "gemini-1.5-flash": 1000000,      # 1M tokens
            "gemini-1.5-flash-8b": 1000000,   # 1M tokens
            "gemini-1.0-pro": 32768,          # 32K tokens
            "gemini-pro": 32768,              # 32K tokens
            "gemini-pro-vision": 32768,       # 32K tokens
        }
        
        return context_lengths.get(self.model, 32768)  # Default fallback


class TestLlmGoogle(unittest.TestCase):
    """Test cases for LlmGoogle"""

    def test_default_model(self):
        """Test default model"""
        self.assertEqual(LlmGoogle.default_model(), "gemini-2.0-flash")

    def test_default_for_type(self):
        """Test default for type"""
        self.assertEqual(LlmGoogle.default_for_type("paid_best"), "gemini-2.0-flash")
        self.assertIsNone(LlmGoogle.default_for_type("local_small"))

    def test_capabilities(self):
        """Test capabilities"""
        caps = LlmGoogle.capabilities()
        self.assertIn("chat", caps)
        self.assertIn("structured", caps)
        self.assertIn("image", caps)

    def test_supported_models(self):
        """Test supported models"""
        models = LlmGoogle.supported_models()
        self.assertIn("gemini-2.0-flash", models)
        self.assertIsInstance(models, list)

    def test_model_name_handling(self):
        """Test model name handling"""
        try:
            llm = LlmGoogle("gemini")
            self.assertEqual(llm.model, "gemini-2.0-flash")
        except ConfigurationError:
            pass  # Expected without API key

    def test_check_config_behavior(self):
        """Test configuration checking"""
        try:
            LlmGoogle.check_config()
        except ConfigurationError as e:
            self.assertIn("API key", str(e))

    def test_chat_functionality_structure(self):
        """Test that chat method exists and has proper signature"""
        try:
            llm = LlmGoogle("gemini-2.0-flash")
            # Just test that the method exists, don't call it without API key
            self.assertTrue(hasattr(llm, 'chat'))
            self.assertTrue(callable(llm.chat))
        except ConfigurationError:
            pass

    def test_image_functionality_structure(self):
        """Test that image method exists and has proper signature"""
        try:
            llm = LlmGoogle("gemini-2.0-flash")
            # Just test that the method exists, don't call it without API key
            self.assertTrue(hasattr(llm, 'image'))
            self.assertTrue(callable(llm.image))
        except ConfigurationError:
            pass  # Expected without API key
