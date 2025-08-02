# Tests for dazllm
"""
Tests package for dazllm.
Provides test utilities and ensures the test package is properly structured.
"""
import unittest


class TestPackageStructure(unittest.TestCase):
    """Test the basic structure and importability of the dazllm package."""
    
    def test_package_import(self):
        """Test that the dazllm package can be imported successfully."""
        try:
            import dazllm
            self.assertIsNotNone(dazllm)
        except ImportError as e:
            self.fail(f"Failed to import dazllm package: {e}")
    
    def test_core_module_import(self):
        """Test that the core module can be imported."""
        try:
            import dazllm.core
            self.assertIsNotNone(dazllm.core)
        except ImportError as e:
            self.fail(f"Failed to import dazllm.core: {e}")
