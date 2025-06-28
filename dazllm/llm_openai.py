"""
OpenAI implementation for dazllm
"""

import keyring
import json
import base64
import requests
from io import BytesIO
from PIL import Image
from typing import Type
from pydantic import BaseModel

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmOpenai(Llm):
    """OpenAI implementation"""

    def __init__(self, model: str):
        self.model = model
        self.check_config()

        # Import OpenAI client
        try:
            import openai

            self.client = openai.OpenAI(api_key=self._get_api_key())
        except ImportError:
            raise ConfigurationError(
                "OpenAI library not installed. Run: pip install openai"
            )

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
            raise DazLlmError(f"OpenAI API error: {e}")

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
            except json.JSONDecodeError:
                raise DazLlmError(f"Could not parse JSON response: {content}")
            except Exception as e:
                raise DazLlmError(f"Could not create Pydantic model: {e}")

        except Exception as e:
            raise DazLlmError(f"OpenAI structured chat error: {e}")

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate image using OpenAI image models"""

        # Automatically use gpt-image-1 for image generation unless already an image model
        image_model = self.model
        if self.model not in ["gpt-image-1", "dall-e-2", "dall-e-3"]:
            image_model = "gpt-image-1"  # Default to gpt-image-1 for image generation

        try:
            # Calculate optimal generation size for gpt-image-1
            gen_width, gen_height = self._calculate_optimal_size(width, height)

            # Enhance prompt for aspect ratio
            enhanced_prompt = self._enhance_prompt_for_aspect_ratio(
                prompt, gen_width, gen_height
            )

            # Generate image
            response = self.client.images.generate(
                model=image_model,
                prompt=enhanced_prompt,
                size=f"{gen_width}x{gen_height}",
                quality="high",  # "high", not "hd"
                output_format="png",
                n=1,
            )

            # Handle response based on format
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                # Base64 response (gpt-image-1 format)
                image_data = base64.b64decode(response.data[0].b64_json)
                image = Image.open(BytesIO(image_data))
            else:
                # URL response (DALL-E format)
                image_url = response.data[0].url
                image_response = requests.get(image_url)
                image = Image.open(BytesIO(image_response.content))

            # Resize and crop if needed
            if width != gen_width or height != gen_height:
                image = self._resize_and_crop(image, width, height)

            # Save image
            self._save_image(image, file_name)

            return file_name

        except Exception as e:
            raise DazLlmError(f"OpenAI image generation error: {e}")

    def _calculate_optimal_size(self, width: int, height: int) -> tuple[int, int]:
        """Calculate optimal size for gpt-image-1 generation"""
        # Available gpt-image-1 sizes
        available_sizes = [
            (1024, 1024),  # Square
            (1024, 1536),  # Portrait
            (1536, 1024),  # Landscape
        ]

        target_ratio = width / height
        best_size = (1024, 1024)
        best_ratio_diff = float("inf")

        for w, h in available_sizes:
            ratio = w / h
            ratio_diff = abs(ratio - target_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_size = (w, h)

        return best_size

    def _enhance_prompt_for_aspect_ratio(
        self, prompt: str, width: int, height: int
    ) -> str:
        """Enhance the prompt to encourage the desired aspect ratio"""
        aspect_ratio_text = f" with aspect ratio {width}:{height}"
        return prompt + aspect_ratio_text

    def _resize_and_crop(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        """Resize and crop image to exact dimensions while maintaining aspect ratio"""
        original_width, original_height = image.size
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height

        # Calculate new size for aspect-ratio-preserving resize
        if original_ratio > target_ratio:
            # Image is wider than target, fit to height
            new_height = target_height
            new_width = int(original_width * (target_height / original_height))
        else:
            # Image is taller than target, fit to width
            new_width = target_width
            new_height = int(original_height * (target_width / original_width))

        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Calculate crop box to center the crop
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        # Crop to exact target size
        cropped_image = resized_image.crop((left, top, right, bottom))

        return cropped_image

    def _save_image(self, image: Image.Image, file_name: str):
        """Save image with appropriate format"""
        if image.mode == "RGBA":
            if file_name.lower().endswith((".jpg", ".jpeg")):
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                rgb_image.save(file_name, "JPEG", quality=95)
            else:
                # Save as PNG to preserve transparency
                if not file_name.lower().endswith(".png"):
                    file_name = file_name.rsplit(".", 1)[0] + ".png"
                image.save(file_name, "PNG")
        else:
            if file_name.lower().endswith(".png"):
                image.save(file_name, "PNG")
            else:
                image.save(file_name, "JPEG", quality=95)

    def _resize_and_crop(
        self, image: Image.Image, target_width: int, target_height: int
    ) -> Image.Image:
        """Resize and crop image to exact dimensions"""
        original_width, original_height = image.size
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height

        if original_ratio > target_ratio:
            # Image is wider, fit to height
            new_height = target_height
            new_width = int(original_width * (target_height / original_height))
        else:
            # Image is taller, fit to width
            new_width = target_width
            new_height = int(original_height * (target_width / original_width))

        # Resize
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Crop to center
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
        right = left + target_width
        bottom = top + target_height

        return resized.crop((left, top, right, bottom))

    def _save_image(self, image: Image.Image, file_name: str):
        """Save image with appropriate format"""
        if image.mode == "RGBA":
            if file_name.lower().endswith((".jpg", ".jpeg")):
                # Convert RGBA to RGB for JPEG
                rgb_image = Image.new("RGB", image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])
                rgb_image.save(file_name, "JPEG", quality=95)
            else:
                # Save as PNG to preserve transparency
                if not file_name.lower().endswith(".png"):
                    file_name = file_name.rsplit(".", 1)[0] + ".png"
                image.save(file_name, "PNG")
        else:
            if file_name.lower().endswith(".png"):
                image.save(file_name, "PNG")
            else:
                image.save(file_name, "JPEG", quality=95)
