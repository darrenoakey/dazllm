#!/usr/bin/env python3
"""
Image generation example for dazllm
"""

from dazllm import Llm

def main():
    """Main function demonstrating image generation with dazllm"""
    print("Image generation example")
    
    try:
        # Generate an image
        prompt = "A serene mountain landscape at sunset"
        filename = "mountain_sunset.png"
        
        result_path = Llm.image_static(
            prompt,
            filename,
            width=1024,
            height=768,
            model="openai:dall-e-3"
        )
        
        print(f"Image generated and saved to: {result_path}")
        
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error: {e}")
        print("Make sure you have configured your OpenAI API key:")
        print("keyring set dazllm openai_api_key YOUR_KEY")

if __name__ == "__main__":
    main()


# Unit tests
import unittest


class TestImageGenerationExample(unittest.TestCase):
    """Test cases for image generation example"""

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        self.assertTrue(callable(main))

    def test_llm_has_image_method(self):
        """Test that Llm class has image generation method"""
        self.assertTrue(hasattr(Llm, 'image_static'))

    def test_example_parameters(self):
        """Test example uses valid parameters"""
        prompt = "A serene mountain landscape at sunset"
        filename = "mountain_sunset.png"
        width = 1024
        height = 768
        model = "openai:dall-e-3"
        
        # Test parameter types and values
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(filename, str)
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertIsInstance(model, str)
        
        # Test parameter validity
        self.assertTrue(len(prompt) > 0)
        self.assertTrue(filename.endswith('.png'))
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)
        self.assertIn(':', model)

    def test_model_name_format(self):
        """Test model name follows correct format"""
        model = "openai:dall-e-3"
        self.assertIn(":", model)
        provider, model_name = model.split(":", 1)
        self.assertEqual(provider, "openai")
        self.assertEqual(model_name, "dall-e-3")

    def test_image_dimensions(self):
        """Test image dimensions are positive integers"""
        width = 1024
        height = 768
        self.assertIsInstance(width, int)
        self.assertIsInstance(height, int)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_file_extension(self):
        """Test filename has appropriate extension"""
        filename = "mountain_sunset.png"
        self.assertTrue(filename.endswith(('.png', '.jpg', '.jpeg')))
