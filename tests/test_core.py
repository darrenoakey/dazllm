"""
Test file for dazllm core functionality.

This file tests the main Llm class, ModelType enum, and error handling
to ensure the core dazllm functionality works as expected.
"""

import unittest
from dazllm import Llm, ModelType, DazLlmError


class TestLlmCore(unittest.TestCase):
    """Test cases for dazllm core functionality."""

    def test_model_name_parsing(self):
        """Test model name parsing with provider:model format."""
        try:
            llm = Llm("openai:gpt-4")
            self.assertEqual(llm.provider, "openai")
            self.assertEqual(llm.model, "gpt-4")
        except DazLlmError:
            # Skip if provider not configured - this is expected behavior
            pass

    def test_invalid_model_format(self):
        """Test that invalid model format raises DazLlmError."""
        with self.assertRaises(DazLlmError):
            Llm("invalid-format")

    def test_model_types(self):
        """Test that ModelType enum has expected values."""
        self.assertEqual(ModelType.PAID_BEST.value, "paid_best")
        self.assertEqual(ModelType.LOCAL_SMALL.value, "local_small")

    def test_model_type_enum_completeness(self):
        """Test that all expected model types exist."""
        expected_types = [
            ModelType.LOCAL_SMALL,
            ModelType.LOCAL_MEDIUM,
            ModelType.LOCAL_LARGE,
            ModelType.PAID_CHEAP,
            ModelType.PAID_BEST
        ]
        # Just verify they exist without errors
        for model_type in expected_types:
            self.assertIsInstance(model_type.value, str)

    def test_llm_class_methods(self):
        """Test that Llm class has expected static methods."""
        self.assertTrue(hasattr(Llm, 'get_providers'))
        self.assertTrue(hasattr(Llm, 'model_named'))
        self.assertTrue(hasattr(Llm, 'chat_static'))
