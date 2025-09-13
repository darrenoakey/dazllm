"""
Tests for landscape and portrait image generation across providers
"""

import unittest
import os
import tempfile
from PIL import Image

from dazllm.core import Llm
from dazllm.llm_openai import LlmOpenai
from dazllm.llm_google import LlmGoogle


class TestLandscapePortraitGeneration(unittest.TestCase):
    """Test image generation in landscape and portrait formats for both providers"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)

    def test_openai_landscape_generation(self):
        """Test OpenAI image generation in landscape format"""
        llm = LlmOpenai("openai:gpt-image-1")
        output_path = os.path.join(self.test_dir, "openai_landscape.png")

        result = llm.image(
            "A beautiful mountain landscape with trees, wide panoramic view",
            output_path,
            width=1536,  # Wide landscape (OpenAI supported)
            height=1024
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertGreater(img.width, img.height)  # Landscape aspect
            self.assertEqual(img.width, 1536)
            self.assertEqual(img.height, 1024)

    def test_openai_portrait_generation(self):
        """Test OpenAI image generation in portrait format"""
        llm = LlmOpenai("openai:gpt-image-1")
        output_path = os.path.join(self.test_dir, "openai_portrait.png")

        result = llm.image(
            "A tall tower reaching into the sky, vertical composition",
            output_path,
            width=1024,  # Tall portrait (OpenAI supported)
            height=1536
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertGreater(img.height, img.width)  # Portrait aspect
            self.assertEqual(img.width, 1024)
            self.assertEqual(img.height, 1536)

    def test_gemini_landscape_generation(self):
        """Test Gemini image generation in landscape format"""
        llm = LlmGoogle("google:gemini-2.0-flash")
        output_path = os.path.join(self.test_dir, "gemini_landscape.png")

        result = llm.image(
            "A serene lake with mountains in the background, wide landscape view",
            output_path,
            width=1536,  # Wide landscape
            height=1024
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertGreater(img.width, img.height)  # Landscape aspect
            self.assertEqual(img.width, 1536)
            self.assertEqual(img.height, 1024)

    def test_gemini_portrait_generation(self):
        """Test Gemini image generation in portrait format"""
        llm = LlmGoogle("google:gemini-2.0-flash")
        output_path = os.path.join(self.test_dir, "gemini_portrait.png")

        result = llm.image(
            "A majestic waterfall cascading down a cliff, tall vertical scene",
            output_path,
            width=1024,  # Tall portrait
            height=1536
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertGreater(img.height, img.width)  # Portrait aspect
            self.assertEqual(img.width, 1024)
            self.assertEqual(img.height, 1536)

    def test_openai_square_generation(self):
        """Test OpenAI image generation in square format"""
        llm = LlmOpenai("openai:gpt-image-1")
        output_path = os.path.join(self.test_dir, "openai_square.png")

        result = llm.image(
            "A geometric mandala pattern, centered square composition",
            output_path,
            width=1024,  # Square
            height=1024
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertEqual(img.width, img.height)  # Square aspect
            self.assertEqual(img.width, 1024)
            self.assertEqual(img.height, 1024)

    def test_gemini_square_generation(self):
        """Test Gemini image generation in square format"""
        llm = LlmGoogle("google:gemini-2.0-flash")
        output_path = os.path.join(self.test_dir, "gemini_square.png")

        result = llm.image(
            "A circular flower arrangement, perfectly centered composition",
            output_path,
            width=1024,  # Square
            height=1024
        )

        # Verify the image was created
        self.assertTrue(os.path.exists(output_path))
        self.assertEqual(result, output_path)

        # Verify it's actually an image file with correct aspect ratio
        with Image.open(output_path) as img:
            self.assertIsNotNone(img)
            self.assertEqual(img.format, 'PNG')
            self.assertEqual(img.width, img.height)  # Square aspect
            self.assertEqual(img.width, 1024)
            self.assertEqual(img.height, 1024)


if __name__ == "__main__":
    unittest.main()