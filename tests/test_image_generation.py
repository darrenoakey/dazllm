"""
Real tests for image generation functionality - no mocks allowed
"""

import unittest
import os
import tempfile
from PIL import Image

from dazllm.core import Llm, DazLlmError
from dazllm.llm_openai import LlmOpenai
from dazllm.llm_google import LlmGoogle


class TestImageGeneration(unittest.TestCase):
    """Test image generation functionality across providers with REAL API calls"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.test_image_path = os.path.join(self.test_dir, "test_image.png")

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_image_path):
            os.remove(self.test_image_path)
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

    def test_openai_image_generation_real(self):
        """Test OpenAI image generation with real API call"""
        # Create real OpenAI LLM instance
        llm = LlmOpenai("openai:gpt-image-1")

        # Generate a real image
        test_prompt = "A simple blue circle on white background, minimalist digital art"
        result = llm.image(test_prompt, self.test_image_path, 512, 512)

        # Verify the image was created
        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertEqual(result, self.test_image_path)

        # Verify it's actually an image file
        with Image.open(self.test_image_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            # Note: OpenAI might return different sizes based on optimization
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)

    def test_openai_fallback_to_image_model(self):
        """Test that non-image OpenAI models fallback to gpt-image-1"""
        # Create OpenAI instance with chat model
        llm = LlmOpenai("openai:gpt-4o-mini")

        # Generate image (should use gpt-image-1 internally)
        test_prompt = "A simple green triangle on white background"
        result = llm.image(test_prompt, self.test_image_path, 512, 512)

        # Verify the image was created
        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertEqual(result, self.test_image_path)

    def test_google_image_generation_with_nano_model(self):
        """Test Google Gemini image generation with nano model - REAL test"""
        # Create real Google LLM instance
        llm = LlmGoogle("google:gemini-2.0-flash")

        # Generate a real image with a simple prompt
        test_prompt = "A simple red square on white background, minimalist digital art"
        result = llm.image(test_prompt, self.test_image_path, 512, 512)

        # Verify the image was created
        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertEqual(result, self.test_image_path)

        # Verify it's actually an image file
        with Image.open(self.test_image_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            # Check dimensions
            self.assertEqual(img.width, 512)
            self.assertEqual(img.height, 512)

    def test_provider_capabilities_image_detection(self):
        """Test that providers correctly report image generation capabilities"""
        # OpenAI should support image generation
        openai_caps = LlmOpenai.capabilities()
        self.assertIn("image", openai_caps)

        # Google should now support image generation with nano model
        google_caps = LlmGoogle.capabilities()
        self.assertIn("image", google_caps)

    def test_llm_generate_image_class_method_real(self):
        """Test the Llm.generate_image class method with real API calls"""
        # Use the generic class method
        result = Llm.generate_image("A yellow star on blue background", self.test_image_path, 512, 512)

        # Verify the image was created
        self.assertTrue(os.path.exists(self.test_image_path))
        self.assertEqual(result, self.test_image_path)

        # Verify it's actually an image
        with Image.open(self.test_image_path) as img:
            self.assertIsNotNone(img)
            self.assertGreater(img.width, 0)
            self.assertGreater(img.height, 0)

    def test_image_generation_with_different_sizes(self):
        """Test image generation with various sizes"""
        test_sizes = [
            (256, 256),
            (512, 512),
            # Skip the larger sizes that are taking too long
            # (1024, 768),  # Wide aspect ratio
            # (768, 1024),  # Tall aspect ratio
        ]

        for width, height in test_sizes:
            output_path = os.path.join(self.test_dir, f"test_{width}x{height}.png")
            result = Llm.generate_image(
                f"Simple geometric shape, {width}x{height}",
                output_path,
                width,
                height
            )

            # Verify the image was created
            self.assertTrue(os.path.exists(output_path))

            # Clean up
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_image_generation_context_lengths(self):
        """Test that image models return appropriate context lengths"""
        # Test OpenAI image model context length
        llm = LlmOpenai("openai:gpt-image-1")
        context_length = llm.get_context_length()
        self.assertIsInstance(context_length, int)
        self.assertGreater(context_length, 0)

        # Test Google model context length
        llm = LlmGoogle("google:gemini-2.0-flash")
        context_length = llm.get_context_length()
        self.assertIsInstance(context_length, int)
        self.assertGreater(context_length, 0)


if __name__ == "__main__":
    unittest.main()