"""
Codex CLI implementation for dazllm
Uses the codex command-line tool for LLM interactions
"""

import subprocess
import json
import re
from typing import Type, Optional
from pydantic import BaseModel
from jsonschema import validate, ValidationError

from .core import Llm, Conversation, ConfigurationError, DazLlmError


class LlmCodexCli(Llm):
    """Codex CLI implementation"""

    def __init__(self, model: str):
        super().__init__(model)
        self.executable = self._find_executable()
        self._check_executable()

    @staticmethod
    def default_model() -> str:
        """Default model for Codex CLI"""
        return "default"

    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        """Get default model for a given type"""
        # Codex CLI doesn't expose model selection, so we use "default"
        defaults = {
            "local_small": "default",
            "local_medium": "default",
            "local_large": "default",
            "paid_cheap": "default",
            "paid_best": "default",
        }
        return defaults.get(model_type)

    @staticmethod
    def capabilities() -> set[str]:
        """Return set of capabilities this provider supports"""
        return {"chat", "structured"}

    @staticmethod
    def supported_models() -> list[str]:
        """Return list of models this provider supports"""
        # Codex CLI doesn't expose available models
        return ["default"]

    @staticmethod
    def check_config():
        """Check if Codex CLI is properly configured"""
        try:
            result = subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise ConfigurationError(f"Codex CLI not accessible: {result.stderr}")
        except FileNotFoundError:
            raise ConfigurationError("Codex CLI not found. Please install codex CLI tool.")
        except subprocess.TimeoutExpired:
            raise ConfigurationError("Codex CLI timeout")
        except Exception as e:
            raise ConfigurationError(f"Codex CLI error: {e}") from e

    def _find_executable(self) -> str:
        """Find the codex executable"""
        import shutil
        executable = shutil.which("codex")
        if not executable:
            raise ConfigurationError("Codex CLI not found in PATH")
        return executable

    def _check_executable(self):
        """Ensure the executable is available and working"""
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0 and "Codex CLI" not in result.stdout:
                raise ConfigurationError(f"Codex CLI check failed: {result.stderr}")
        except Exception as e:
            raise ConfigurationError(f"Cannot run Codex CLI: {e}") from e

    def _normalize_conversation(self, conversation: Conversation) -> str:
        """Convert conversation to a single prompt string for CLI"""
        if isinstance(conversation, str):
            return conversation
        
        # Convert message list to a formatted prompt
        prompt_parts = []
        for msg in conversation:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"User: {content}")
        
        return "\n".join(prompt_parts)

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat using Codex CLI"""
        prompt = self._normalize_conversation(conversation)
        
        if force_json:
            prompt = f"{prompt}\n\nPlease respond with valid JSON only, no other text."
        
        try:
            # Use 'codex exec' for non-interactive execution
            result = subprocess.run(
                [self.executable, "exec", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                raise DazLlmError(f"Codex CLI error: {result.stderr}")
            
            response = result.stdout.strip()
            
            # If force_json, try to extract JSON from response
            if force_json:
                response = self._extract_json(response)
            
            return response
            
        except subprocess.TimeoutExpired:
            raise DazLlmError("Codex CLI timeout after 60 seconds")
        except Exception as e:
            raise DazLlmError(f"Codex CLI execution error: {e}") from e

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
            f"JSON Schema:\n{schema_str}"
        )
        
        attempts = 10
        last_error = None
        
        while attempts > 0:
            try:
                # Use codex exec for non-interactive execution
                result = subprocess.run(
                    [self.executable, "exec", structured_prompt],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    raise DazLlmError(f"Codex CLI error: {result.stderr}")
                
                response = result.stdout.strip()
                
                # Parse JSON from response
                parsed_json = self._extract_json(response)
                
                # Validate against schema
                validate(instance=parsed_json, schema=schema_json)
                
                # Create and return Pydantic model
                return schema(**parsed_json)
                
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                structured_prompt += (
                    f"\n\nThe previous response was not valid JSON. Error: {e}\n"
                    f"Please respond with ONLY valid JSON matching the schema."
                )
            except ValidationError as e:
                last_error = e
                structured_prompt += (
                    f"\n\nThe JSON didn't match the schema. Error: {e}\n"
                    f"Please fix and return valid JSON matching the schema exactly."
                )
            except subprocess.TimeoutExpired:
                raise DazLlmError("Codex CLI timeout after 60 seconds")
            except Exception as e:
                raise DazLlmError(f"Codex CLI execution error: {e}") from e
            
            attempts -= 1
        
        raise DazLlmError(f"Failed to get valid structured response after multiple attempts. Last error: {last_error}")

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
            r'\{.*\}',      # Any JSON object
            r'\[.*\]',      # JSON array
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        # Strategy 5: Clean common prefixes and try again
        cleaned = text.strip()
        for prefix in ["Here is the JSON:", "JSON:", "Output:", "Result:", "Response:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")

    def image(self, prompt: str, file_name: str, width: int = 1024, height: int = 1024) -> str:
        """Generate image using Codex CLI (not supported)"""
        raise DazLlmError("Image generation not supported by Codex CLI")
    
    def get_context_length(self) -> int:
        """Get the context length for Codex CLI models"""
        # Codex CLI uses various underlying models, default to a reasonable size
        return 8192  # Conservative default


# Unit tests
import unittest
from unittest.mock import Mock, patch, MagicMock


class TestLlmCodexCli(unittest.TestCase):
    """Test cases for Codex CLI implementation"""
    
    def test_default_model(self):
        """Test default model returns expected value"""
        self.assertEqual(LlmCodexCli.default_model(), "default")
    
    def test_capabilities(self):
        """Test capabilities returns expected set"""
        caps = LlmCodexCli.capabilities()
        self.assertIn("chat", caps)
        self.assertIn("structured", caps)
        self.assertNotIn("image", caps)
    
    def test_extract_json_markdown(self):
        """Test JSON extraction from markdown blocks"""
        llm = LlmCodexCli.__new__(LlmCodexCli)
        
        text = '```json\n{"key": "value", "number": 42}\n```'
        result = llm._extract_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})
    
    def test_extract_json_plain(self):
        """Test JSON extraction from plain text"""
        llm = LlmCodexCli.__new__(LlmCodexCli)
        
        text = '{"status": "success", "data": [1, 2, 3]}'
        result = llm._extract_json(text)
        self.assertEqual(result, {"status": "success", "data": [1, 2, 3]})
    
    def test_normalize_conversation_string(self):
        """Test conversation normalization with string input"""
        llm = LlmCodexCli.__new__(LlmCodexCli)
        
        result = llm._normalize_conversation("Hello, world!")
        self.assertEqual(result, "Hello, world!")
    
    def test_normalize_conversation_messages(self):
        """Test conversation normalization with message list"""
        llm = LlmCodexCli.__new__(LlmCodexCli)
        
        conversation = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = llm._normalize_conversation(conversation)
        expected = "System: You are helpful\nUser: Hello\nAssistant: Hi there\nUser: How are you?"
        self.assertEqual(result, expected)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_chat_success(self, mock_which, mock_run):
        """Test successful chat operation"""
        mock_which.return_value = "/usr/bin/codex"
        
        # Mock chat result
        chat_result = Mock()
        chat_result.returncode = 0
        chat_result.stdout = "Hello! How can I help you?"
        chat_result.stderr = ""
        
        mock_run.return_value = chat_result
        
        # Create instance without going through Llm.__init__
        llm = LlmCodexCli.__new__(LlmCodexCli)
        llm.model = "default"
        llm.executable = "/usr/bin/codex"
        result = llm.chat("Hello")
        
        self.assertEqual(result, "Hello! How can I help you?")
        
        # Verify codex exec was called
        mock_run.assert_called_with(
            ["/usr/bin/codex", "exec", "Hello"],
            capture_output=True,
            text=True,
            timeout=60
        )
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_chat_structured_success(self, mock_which, mock_run):
        """Test successful structured chat operation"""
        mock_which.return_value = "/usr/bin/codex"
        
        # Mock version check
        version_result = Mock()
        version_result.returncode = 0
        version_result.stdout = "Codex CLI v1.0"
        
        # Mock structured response
        chat_result = Mock()
        chat_result.returncode = 0
        chat_result.stdout = '```json\n{"name": "Test", "value": 123}\n```'
        chat_result.stderr = ""
        
        mock_run.return_value = chat_result
        
        from pydantic import BaseModel
        
        class TestSchema(BaseModel):
            name: str
            value: int
        
        # Create instance without going through Llm.__init__
        llm = LlmCodexCli.__new__(LlmCodexCli)
        llm.model = "default"
        llm.executable = "/usr/bin/codex"
        result = llm.chat_structured("Generate test data", TestSchema)
        
        self.assertIsInstance(result, TestSchema)
        self.assertEqual(result.name, "Test")
        self.assertEqual(result.value, 123)


if __name__ == "__main__":
    unittest.main()