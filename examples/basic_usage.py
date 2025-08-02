#!/usr/bin/env python3
"""
Basic usage example for dazllm
"""

from dazllm import Llm, ModelType

def main():
    """Main function demonstrating basic dazllm usage"""
    print("Basic dazllm usage examples")
    
    try:
        # Simple chat with specific model
        response = Llm.chat_static("What's 2+2?", model="openai:gpt-4o")
        print(f"Math answer: {response}")
        
        # Chat with model type
        response = Llm.chat_static("Tell me a joke", model_type=ModelType.PAID_CHEAP)
        print(f"Joke: {response}")
        
        # Instance-based usage
        llm = Llm.model_named("openai:gpt-4o")
        response = llm.chat("Hello!")
        print(f"Greeting: {response}")
        
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys:")
        print("keyring set dazllm openai_api_key YOUR_KEY")

if __name__ == "__main__":
    main()


# Unit tests
import unittest


class TestBasicUsageExample(unittest.TestCase):
    """Test cases for basic usage example"""

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        self.assertTrue(callable(main))

    def test_imports_work(self):
        """Test that required imports are available"""
        # Use already imported classes to avoid reimport warnings
        self.assertTrue(hasattr(Llm, 'chat_static'))
        self.assertTrue(hasattr(Llm, 'model_named'))
        self.assertTrue(hasattr(ModelType, 'PAID_CHEAP'))

    def test_model_type_enum_values(self):
        """Test ModelType enum has expected values"""
        self.assertEqual(ModelType.PAID_CHEAP.value, "paid_cheap")
        self.assertEqual(ModelType.PAID_BEST.value, "paid_best")
        self.assertEqual(ModelType.LOCAL_SMALL.value, "local_small")

    def test_llm_class_methods_exist(self):
        """Test that Llm class has required methods"""
        self.assertTrue(hasattr(Llm, 'chat_static'))
        self.assertTrue(hasattr(Llm, 'model_named'))
        self.assertTrue(hasattr(Llm, 'get_providers'))

    def test_example_constants(self):
        """Test example uses valid model names and types"""
        model_name = "openai:gpt-4o"
        self.assertIn(":", model_name)
        provider, model = model_name.split(":", 1)
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4o")

    def test_model_type_attributes(self):
        """Test ModelType has all expected attributes"""
        required_attrs = ['PAID_CHEAP', 'PAID_BEST', 'LOCAL_SMALL', 'LOCAL_MEDIUM', 'LOCAL_LARGE']
        for attr in required_attrs:
            self.assertTrue(hasattr(ModelType, attr))
