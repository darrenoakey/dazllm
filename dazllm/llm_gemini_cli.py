"""
Gemini CLI implementation for dazllm
Uses the gemini command-line tool for LLM interactions
"""

import subprocess
import json
import re
from typing import Type, Optional
from pydantic import BaseModel
from jsonschema import validate, ValidationError

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmGeminiCli(Llm):
    """Gemini CLI implementation"""

    def __init__(self, model: str):
        super().__init__(model)
        self.executable = self._find_executable()
        self._check_executable()

    @staticmethod
    def default_model() -> str:
        """Default model for Gemini CLI"""
        return "gemini-2.0-flash-exp"

    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        """Get default model for a given type"""
        defaults = {
            "local_small": None,  # Gemini doesn't have local models
            "local_medium": None,
            "local_large": None,
            "paid_cheap": "gemini-1.5-flash",
            "paid_best": "gemini-2.0-flash-exp",
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured"}

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports"""
        return [
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.0-pro",
        ]

    @staticmethod
    def check_config():
        """Check if Gemini CLI is properly configured"""
        try:
            result = subprocess.run(
                ["gemini", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise ConfigurationError(f"Gemini CLI not accessible: {result.stderr}")
        except FileNotFoundError:
            raise ConfigurationError("Gemini CLI not found. Please install gemini CLI tool.")
        except subprocess.TimeoutExpired:
            raise ConfigurationError("Gemini CLI timeout")
        except Exception as e:
            raise ConfigurationError(f"Gemini CLI error: {e}") from e

    def _find_executable(self) -> str:
        """Find the gemini executable"""
        import shutil
        executable = shutil.which("gemini")
        if not executable:
            raise ConfigurationError("Gemini CLI not found in PATH")
        return executable

    def _check_executable(self):
        """Ensure the executable is available and working"""
        try:
            result = subprocess.run(
                [self.executable, "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise ConfigurationError(f"Gemini CLI check failed: {result.stderr}")
        except Exception as e:
            raise ConfigurationError(f"Cannot run Gemini CLI: {e}") from e

    def _normalize_conversation(self, conversation: Conversation) -> str:
        """Convert conversation to a prompt string for Gemini CLI"""
        if isinstance(conversation, str):
            return conversation
        
        # Convert message list to a formatted prompt
        prompt_parts = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                # Gemini doesn't have explicit system messages, prepend as instruction
                prompt_parts.insert(0, f"Instructions: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        
        return "\n".join(prompt_parts)

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Gemini CLI"""
        prompt = self._normalize_conversation(conversation)
        
        if force_json:
            prompt = f"{prompt}\n\nPlease respond with valid JSON only, no other text."
        
        try:
            # Use --yolo for headless operation, --model to specify model
            cmd = [
                self.executable,
                "--yolo",  # Auto-approve all actions for headless mode
                "--model", self.model,
                prompt
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                input="",  # Provide empty input to avoid hanging
                env={"GEMINI_DISABLE_SANDBOX": "true"}  # Disable sandbox for headless operation
            )
            
            if result.returncode != 0:
                # Gemini CLI might return non-zero even for successful responses
                # Check if we have output
                if not result.stdout.strip() and result.stderr:
                    raise DazLlmError(f"Gemini CLI error: {result.stderr}")
            
            response = result.stdout.strip()
            
            # If force_json, try to extract JSON from response
            if force_json:
                response = self._extract_json_string(response)
            
            return response
            
        except subprocess.TimeoutExpired:
            raise DazLlmError("Gemini CLI timeout after 120 seconds")
        except Exception as e:
            raise DazLlmError(f"Gemini CLI execution error: {e}") from e

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        schema_json = schema.model_json_schema()
        schema_str = json.dumps(schema_json, indent=2)
        
        # Build prompt with schema instructions
        prompt = self._normalize_conversation(conversation)
        structured_prompt = (
            f"{prompt}\n\n"
            f"You must respond with valid JSON that matches the following schema exactly.\n"
            f"Output ONLY the JSON, no other text or explanation.\n"
            f"Wrap your JSON response in ```json code blocks.\n"
            f"JSON Schema:\n{schema_str}"
        )
        
        attempts = 10
        last_error = None
        current_prompt = structured_prompt
        
        while attempts > 0:
            try:
                # Use --yolo for headless operation
                cmd = [
                    self.executable,
                    "--yolo",
                    "--model", self.model,
                    current_prompt
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    input="",
                    env={"GEMINI_DISABLE_SANDBOX": "true"}
                )
                
                if result.returncode != 0:
                    # Check if we have output despite non-zero return
                    if not result.stdout.strip() and result.stderr:
                        raise DazLlmError(f"Gemini CLI error: {result.stderr}")
                
                response = result.stdout.strip()
                
                if not response:
                    raise DazLlmError("Empty response from Gemini CLI")
                
                # Parse JSON from response
                parsed_json = self._extract_json(response)
                
                # Validate against schema
                validate(instance=parsed_json, schema=schema_json)
                
                # Create and return Pydantic model
                return schema(**parsed_json)
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                current_prompt = (
                    f"{structured_prompt}\n\n"
                    f"Previous attempt failed with error: {e}\n"
                    f"Please respond with ONLY valid JSON wrapped in ```json blocks."
                )
            except ValidationError as e:
                last_error = e
                current_prompt = (
                    f"{structured_prompt}\n\n"
                    f"Previous JSON didn't match schema. Error: {e}\n"
                    f"Please fix and return valid JSON matching the schema exactly."
                )
            except subprocess.TimeoutExpired:
                raise DazLlmError("Gemini CLI timeout after 120 seconds")
            except Exception as e:
                raise DazLlmError(f"Gemini CLI execution error: {e}") from e
            
            attempts -= 1
        
        raise DazLlmError(f"Failed to get valid structured response after multiple attempts. Last error: {last_error}")

    def _extract_json_string(self, text: str) -> str:
        """Extract JSON string from text (for force_json mode)"""
        try:
            parsed = self._extract_json(text)
            return json.dumps(parsed)
        except (json.JSONDecodeError, ValueError):
            return text  # Return original if can't parse

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from text with multiple fallback strategies"""
        # Strategy 1: Look for markdown code blocks
        if "```json" in text:
            match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Strategy 2: Look for any code blocks
        if "```" in text:
            match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Strategy 3: Try to parse the entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Look for JSON-like structures
        json_patterns = [
            r'\{[^{}]*\}',  # Simple JSON object
            r'\{.*?\}(?=\s*$|\s*\n)',  # JSON object at end of line
            r'\{.*\}',      # Any JSON object
            r'\[.*\]',      # JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Strategy 5: Clean common prefixes and try again
        cleaned = text.strip()
        for prefix in ["Here is the JSON:", "JSON:", "Output:", "Result:", "Response:", "Here's the JSON:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")

    def image(self, prompt: str, file_name: str, width: int = 1024, height: int = 1024) -> str:
        """Generate image using Gemini CLI (not supported)"""
        raise DazLlmError("Image generation not supported by Gemini CLI")
    
    def get_context_length(self) -> int:
        """Get the context length for Gemini CLI models"""
        # Gemini CLI uses the same models as the API, so use same context lengths
        context_lengths = {
            "gemini-2.0-flash-exp": 1000000,      # 1M tokens
            "gemini-2.0-flash": 1000000,          # 1M tokens
            "gemini-1.5-pro": 2000000,            # 2M tokens
            "gemini-1.5-flash": 1000000,          # 1M tokens
            "gemini-1.5-flash-8b": 1000000,       # 1M tokens
            "gemini-1.0-pro": 32768,              # 32K tokens
            "gemini-pro": 32768,                  # 32K tokens
            "gemini-pro-vision": 32768,           # 32K tokens
        }
        
        return context_lengths.get(self.model, 32768)  # Default fallback


# Unit tests
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestLlmGeminiCli(unittest.TestCase):
    """Test cases for Gemini CLI implementation"""
    
    def test_default_model(self):
        """Test default model returns expected value"""
        self.assertEqual(LlmGeminiCli.default_model(), "gemini-2.0-flash-exp")
    
    def test_default_for_type(self):
        """Test default_for_type returns correct models"""
        self.assertEqual(LlmGeminiCli.default_for_type("paid_cheap"), "gemini-1.5-flash")
        self.assertEqual(LlmGeminiCli.default_for_type("paid_best"), "gemini-2.0-flash-exp")
        self.assertIsNone(LlmGeminiCli.default_for_type("local_small"))
    
    def test_capabilities(self):
        """Test capabilities returns expected set"""
        caps = LlmGeminiCli.capabilities()
        self.assertIn("chat", caps)
        self.assertIn("structured", caps)
        self.assertNotIn("image", caps)
    
    def test_supported_models(self):
        """Test supported models list"""
        models = LlmGeminiCli.supported_models()
        self.assertIn("gemini-2.0-flash-exp", models)
        self.assertIn("gemini-1.5-flash", models)
        self.assertGreater(len(models), 3)
    
    def test_extract_json_markdown(self):
        """Test JSON extraction from markdown blocks"""
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        
        text = '```json\n{"key": "value", "number": 42}\n```'
        result = llm._extract_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})
    
    def test_extract_json_plain(self):
        """Test JSON extraction from plain text"""
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        
        text = '{"status": "success", "data": [1, 2, 3]}'
        result = llm._extract_json(text)
        self.assertEqual(result, {"status": "success", "data": [1, 2, 3]})
    
    def test_normalize_conversation_string(self):
        """Test conversation normalization with string input"""
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        
        result = llm._normalize_conversation("Hello, world!")
        self.assertEqual(result, "Hello, world!")
    
    def test_normalize_conversation_messages(self):
        """Test conversation normalization with message list"""
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        
        conversation = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = llm._normalize_conversation(conversation)
        expected = "Instructions: You are helpful\nUser: Hello\nAssistant: Hi there\nUser: How are you?"
        self.assertEqual(result, expected)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_chat_success(self, mock_which, mock_run):
        """Test successful chat operation"""
        mock_which.return_value = "/usr/bin/gemini"
        
        # Mock chat result
        chat_result = Mock()
        chat_result.returncode = 0
        chat_result.stdout = "Hello! I'm Gemini, how can I assist you today?"
        chat_result.stderr = ""
        
        mock_run.return_value = chat_result
        
        # Create instance without going through Llm.__init__
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        llm.model = "gemini-2.0-flash-exp"
        llm.executable = "/usr/bin/gemini"
        result = llm.chat("Hello")
        
        self.assertEqual(result, "Hello! I'm Gemini, how can I assist you today?")
        
        # Verify gemini was called with correct args
        args = mock_run.call_args[0][0]
        self.assertIn("--yolo", args)
        self.assertIn("--model", args)
        self.assertIn("gemini-2.0-flash-exp", args)
        self.assertIn("Hello", args)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_chat_structured_success(self, mock_which, mock_run):
        """Test successful structured chat operation"""
        mock_which.return_value = "/usr/bin/gemini"
        
        # Mock structured response
        chat_result = Mock()
        chat_result.returncode = 0
        chat_result.stdout = '```json\n{"name": "GeminiTest", "value": 456}\n```'
        chat_result.stderr = ""
        
        mock_run.return_value = chat_result
        
        from pydantic import BaseModel
        
        class TestSchema(BaseModel):
            name: str
            value: int
        
        # Create instance without going through Llm.__init__
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        llm.model = "gemini-2.0-flash-exp"
        llm.executable = "/usr/bin/gemini"
        result = llm.chat_structured("Generate test data", TestSchema)
        
        self.assertIsInstance(result, TestSchema)
        self.assertEqual(result.name, "GeminiTest")
        self.assertEqual(result.value, 456)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_chat_handles_nonzero_return(self, mock_which, mock_run):
        """Test that chat handles non-zero return codes with valid output"""
        mock_which.return_value = "/usr/bin/gemini"
        
        # Mock result with non-zero return but valid output
        chat_result = Mock()
        chat_result.returncode = 1
        chat_result.stdout = "This is a valid response despite non-zero exit"
        chat_result.stderr = "some warning"
        
        mock_run.return_value = chat_result
        
        # Create instance without going through Llm.__init__
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        llm.model = "gemini-1.5-flash"
        llm.executable = "/usr/bin/gemini"
        result = llm.chat("Hello")
        
        self.assertEqual(result, "This is a valid response despite non-zero exit")
    
    def test_image_raises_error(self):
        """Test that image method raises error"""
        llm = LlmGeminiCli.__new__(LlmGeminiCli)
        
        with self.assertRaises(DazLlmError):
            llm.image("test prompt", "test.jpg")


if __name__ == "__main__":
    unittest.main()