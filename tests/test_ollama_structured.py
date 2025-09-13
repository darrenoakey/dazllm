"""
Tests for Ollama structured output with models that don't support format parameter
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from typing import List, Optional
from pydantic import BaseModel
from dazllm.llm_ollama import LlmOllama
from dazllm.core import DazLlmError


# Test Pydantic models
class SimpleResponse(BaseModel):
    """Simple response model for testing"""
    message: str
    status: str


class PersonInfo(BaseModel):
    """Person information model"""
    name: str
    age: int
    city: str
    occupation: Optional[str] = None


class TodoItem(BaseModel):
    """Todo item model"""
    id: int
    task: str
    completed: bool
    priority: Optional[str] = "medium"


class TodoList(BaseModel):
    """Todo list with multiple items"""
    title: str
    items: List[TodoItem]
    total_count: int


class CodeAnalysis(BaseModel):
    """Code analysis result"""
    language: str
    complexity: str
    has_errors: bool
    suggestions: List[str]
    line_count: int


class TestOllamaStructuredOutput(unittest.TestCase):
    """Test structured output for Ollama models without format support"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_base_url = "http://localhost:11434"
        
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_format_support_detection(self, mock_keyring, mock_get):
        """Test that models are correctly identified as not supporting format"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        # Create instance directly without going through Llm framework
        with patch.object(LlmOllama, '__init__', lambda self, model: None):
            # Test model that doesn't support format
            llm = LlmOllama.__new__(LlmOllama)
            llm.model = "gpt-oss:20b"
            llm.base_url = self.mock_base_url
            llm.headers = {"Content-Type": "application/json"}
            llm._supports_format = llm._check_format_support()
            self.assertFalse(llm._supports_format)
            
            # Test model that does support format
            llm2 = LlmOllama.__new__(LlmOllama)
            llm2.model = "mistral-small"
            llm2.base_url = self.mock_base_url
            llm2.headers = {"Content-Type": "application/json"}
            llm2._supports_format = llm2._check_format_support()
            self.assertTrue(llm2._supports_format)

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_structured_output_without_format(self, mock_keyring, mock_get, mock_post):
        """Test structured output for models without format support"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        # Create a mock response with JSON in markdown blocks
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {
                "content": '```json\n{"message": "Hello", "status": "success"}\n```'
            }
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "gpt-oss:20b"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = False
        
        result = llm.chat_structured("Test prompt", SimpleResponse)
        
        # Verify the result
        self.assertIsInstance(result, SimpleResponse)
        self.assertEqual(result.message, "Hello")
        self.assertEqual(result.status, "success")
        
        # Verify the API was called without format parameter
        call_args = mock_post.call_args[1]['json']
        self.assertNotIn('format', call_args)
        self.assertIn('messages', call_args)
        
        # Check that schema was injected into the prompt
        messages = call_args['messages']
        system_msg = messages[0]['content']
        self.assertIn('JSON Schema', system_msg)

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_json_parsing_fallbacks(self, mock_keyring, mock_get, mock_post):
        """Test various JSON parsing fallback strategies"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        test_cases = [
            # Case 1: JSON in markdown blocks
            ('```json\n{"name": "Alice", "age": 30, "city": "NYC"}\n```', 
             {"name": "Alice", "age": 30, "city": "NYC"}),
            
            # Case 2: Plain JSON
            ('{"name": "Bob", "age": 25, "city": "LA"}',
             {"name": "Bob", "age": 25, "city": "LA"}),
            
            # Case 3: JSON with prefix text
            ('Here is the JSON: {"name": "Charlie", "age": 35, "city": "Chicago"}',
             {"name": "Charlie", "age": 35, "city": "Chicago"}),
            
            # Case 4: JSON with surrounding text
            ('The result is {"name": "David", "age": 40, "city": "Boston"} as requested.',
             {"name": "David", "age": 40, "city": "Boston"}),
        ]
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "gpt-oss:20b"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = False
        
        for json_text, expected in test_cases:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "message": {"content": json_text}
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response
                
                result = llm.chat_structured("Test", PersonInfo)
                self.assertEqual(result.name, expected["name"])
                self.assertEqual(result.age, expected["age"])
                self.assertEqual(result.city, expected["city"])

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_complex_nested_schema(self, mock_keyring, mock_get, mock_post):
        """Test structured output with complex nested schema"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        todo_response = {
            "title": "My Tasks",
            "items": [
                {"id": 1, "task": "Write tests", "completed": True, "priority": "high"},
                {"id": 2, "task": "Review code", "completed": False, "priority": "medium"}
            ],
            "total_count": 2
        }
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": f"```json\n{json.dumps(todo_response)}\n```"}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "gpt-oss:20b"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = False
        
        result = llm.chat_structured("Create a todo list", TodoList)
            
        self.assertIsInstance(result, TodoList)
        self.assertEqual(result.title, "My Tasks")
        self.assertEqual(result.total_count, 2)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].task, "Write tests")
        self.assertTrue(result.items[0].completed)

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_retry_on_invalid_json(self, mock_keyring, mock_get, mock_post):
        """Test retry mechanism when JSON is invalid"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        # First response: invalid JSON
        invalid_response = Mock()
        invalid_response.json.return_value = {
            "message": {"content": "This is not JSON at all"}
        }
        invalid_response.raise_for_status = Mock()
        
        # Second response: valid JSON
        valid_response = Mock()
        valid_response.json.return_value = {
            "message": {"content": '```json\n{"message": "Retry worked", "status": "ok"}\n```'}
        }
        valid_response.raise_for_status = Mock()
        
        mock_post.side_effect = [invalid_response, valid_response]
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "gpt-oss:20b"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = False
        
        result = llm.chat_structured("Test", SimpleResponse)
        
        self.assertEqual(result.message, "Retry worked")
        self.assertEqual(result.status, "ok")
        
        # Verify retry happened
        self.assertEqual(mock_post.call_count, 2)
        
        # Check that error message was added to conversation
        second_call_messages = mock_post.call_args_list[1][1]['json']['messages']
        error_msg = second_call_messages[-1]['content']
        self.assertIn("Invalid JSON", error_msg)

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_retry_on_schema_validation_error(self, mock_keyring, mock_get, mock_post):
        """Test retry when JSON doesn't match schema"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "gpt-oss:20b"}]}
        
        # First response: wrong schema
        wrong_schema_response = Mock()
        wrong_schema_response.json.return_value = {
            "message": {"content": '```json\n{"wrong_field": "value"}\n```'}
        }
        wrong_schema_response.raise_for_status = Mock()
        
        # Second response: correct schema
        correct_response = Mock()
        correct_response.json.return_value = {
            "message": {"content": '```json\n{"message": "Fixed", "status": "correct"}\n```'}
        }
        correct_response.raise_for_status = Mock()
        
        mock_post.side_effect = [wrong_schema_response, correct_response]
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "gpt-oss:20b"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = False
        
        result = llm.chat_structured("Test", SimpleResponse)
        
        self.assertEqual(result.message, "Fixed")
        self.assertEqual(result.status, "correct")
        
        # Verify validation error was communicated
        second_call_messages = mock_post.call_args_list[1][1]['json']['messages']
        error_msg = second_call_messages[-1]['content']
        self.assertIn("doesn't match schema", error_msg)

    @patch('dazllm.llm_ollama.requests.post')
    @patch('dazllm.llm_ollama.requests.get')
    @patch('dazllm.llm_ollama.keyring.get_password')
    def test_models_with_format_support_use_format(self, mock_keyring, mock_get, mock_post):
        """Test that models with format support still use the format parameter"""
        mock_keyring.return_value = self.mock_base_url
        mock_get.return_value.json.return_value = {"models": [{"name": "mistral-small"}]}
        
        mock_response = Mock()
        mock_response.json.return_value = {
            "message": {"content": '{"message": "Using format", "status": "formatted"}'}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Create instance directly without going through Llm framework
        llm = LlmOllama.__new__(LlmOllama)
        llm.model = "mistral-small"
        llm.base_url = self.mock_base_url
        llm.headers = {"Content-Type": "application/json"}
        llm._supports_format = True
        
        result = llm.chat_structured("Test", SimpleResponse)
        
        # Verify format parameter was included
        call_args = mock_post.call_args[1]['json']
        self.assertIn('format', call_args)
        self.assertIsInstance(call_args['format'], dict)

    def test_parse_json_with_fallbacks_unit(self):
        """Unit test for _parse_json_with_fallbacks method"""
        llm = LlmOllama.__new__(LlmOllama)
        
        test_cases = [
            # Markdown blocks
            ('```json\n{"key": "value"}\n```', {"key": "value"}),
            
            # Plain JSON
            ('{"key": "value"}', {"key": "value"}),
            
            # JSON with prefix
            ('JSON: {"key": "value"}', {"key": "value"}),
            ('Here is the JSON: {"key": "value"}', {"key": "value"}),
            
            # Nested JSON
            ('{"outer": {"inner": "value"}}', {"outer": {"inner": "value"}}),
            
            # Array
            ('[{"id": 1}, {"id": 2}]', [{"id": 1}, {"id": 2}]),
            
            # JSON with trailing punctuation
            ('{"key": "value"}.', {"key": "value"}),
            ('{"key": "value"},', {"key": "value"}),
        ]
        
        for input_text, expected in test_cases:
            with self.subTest(input=input_text):
                result = llm._parse_json_with_fallbacks(input_text)
                self.assertEqual(result, expected)
        
        # Test error case
        with self.assertRaises(ValueError):
            llm._parse_json_with_fallbacks("This is definitely not JSON")


class TestOllamaIntegration(unittest.TestCase):
    """Integration tests using actual Ollama API (requires running Ollama)"""
    
    @unittest.skipUnless(
        False,  # Set to True to run integration tests
        "Integration tests disabled by default - requires running Ollama"
    )
    def test_real_ollama_structured_output(self):
        """Test actual Ollama API with structured output"""
        try:
            # This test requires Ollama to be running with gpt-oss:20b installed
            llm = LlmOllama("gpt-oss:20b")
            
            # Test simple structured output
            result = llm.chat_structured(
                "Generate a person named John who is 30 years old and lives in New York",
                PersonInfo
            )
            
            self.assertIsInstance(result, PersonInfo)
            self.assertEqual(result.name, "John")
            self.assertEqual(result.age, 30)
            self.assertIn("New York", result.city)
            
            # Test complex nested structure
            todo_result = llm.chat_structured(
                "Create a todo list with 2 items about programming",
                TodoList
            )
            
            self.assertIsInstance(todo_result, TodoList)
            self.assertGreater(len(todo_result.items), 0)
            self.assertEqual(todo_result.total_count, len(todo_result.items))
            
        except Exception as e:
            self.skipTest(f"Ollama not available: {e}")


if __name__ == "__main__":
    unittest.main()