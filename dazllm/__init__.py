"""
dazllm - A simple, unified interface for all major LLMs

This package provides a unified interface to work with various LLM providers
including OpenAI, Anthropic, Google, Ollama, and LM Studio.
"""

import unittest
from .core import (
    Llm, 
    ModelType, 
    Message, 
    Conversation,
    DazLlmError, 
    ConfigurationError, 
    ModelNotFoundError,
    check_configuration
)

__version__ = "0.2.0"
__all__ = [
    'Llm',
    'ModelType', 
    'Message',
    'Conversation',
    'DazLlmError',
    'ConfigurationError',
    'ModelNotFoundError',
    'check_configuration'
]


class TestDazllmInit(unittest.TestCase):
    """Test dazllm package initialization and imports"""

    def test_imports_available(self):
        """Test that all public imports are available"""
        # Test classes
        self.assertTrue(callable(Llm))
        self.assertTrue(issubclass(DazLlmError, Exception))
        self.assertTrue(issubclass(ConfigurationError, DazLlmError))
        self.assertTrue(issubclass(ModelNotFoundError, DazLlmError))
        
        # Test enum
        self.assertTrue(hasattr(ModelType, 'LOCAL_SMALL'))
        self.assertTrue(hasattr(ModelType, 'PAID_BEST'))
        
        # Test function
        self.assertTrue(callable(check_configuration))

    def test_version_defined(self):
        """Test that version is properly defined"""
        self.assertIsInstance(__version__, str)
        self.assertGreater(len(__version__), 0)

    def test_all_exports_exist(self):
        """Test that all items in __all__ are actually defined"""
        # Get module globals
        module_globals = globals()
        
        for export_name in __all__:
            self.assertIn(export_name, module_globals,
                         f"Export '{export_name}' not found in module globals")

    def test_model_types_values(self):
        """Test that ModelType enum has expected values"""
        expected_types = {
            'LOCAL_SMALL', 'LOCAL_MEDIUM', 'LOCAL_LARGE',
            'PAID_CHEAP', 'PAID_BEST'
        }
        actual_types = {member.name for member in ModelType}
        self.assertEqual(expected_types, actual_types)
