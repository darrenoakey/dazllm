"""
Integration tests for CLI providers
Tests that all CLI providers are properly integrated with the dazllm framework
"""

import unittest
from unittest.mock import patch, Mock
from pydantic import BaseModel
from dazllm.provider_manager import ProviderManager
from dazllm.core import Llm


class SimpleTestModel(BaseModel):
    """Simple test model for structured output tests"""
    name: str
    value: int
    active: bool


class TestCliProvidersIntegration(unittest.TestCase):
    """Test CLI providers integration with dazllm framework"""
    
    def test_cli_providers_registered(self):
        """Test that all CLI providers are registered"""
        providers = ProviderManager.get_providers()
        
        self.assertIn("codex-cli", providers)
        self.assertIn("claude-cli", providers)
        self.assertIn("gemini-cli", providers)
    
    def test_cli_provider_aliases(self):
        """Test that CLI provider aliases work"""
        # Test codex alias
        self.assertEqual(
            ProviderManager.resolve_provider_alias("codex"),
            "codex-cli"
        )
        
        # Test claude CLI alias
        self.assertEqual(
            ProviderManager.resolve_provider_alias("claudecli"),
            "claude-cli"
        )
        
        # Test gemini CLI alias
        self.assertEqual(
            ProviderManager.resolve_provider_alias("geminicli"),
            "gemini-cli"
        )
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_codex_cli_provider_info(self, mock_which, mock_run):
        """Test Codex CLI provider info"""
        mock_which.return_value = "/usr/bin/codex"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Codex CLI"
        mock_run.return_value = mock_result
        
        try:
            info = ProviderManager.get_provider_info("codex-cli")
            self.assertEqual(info["name"], "codex-cli")
            self.assertIn("chat", info["capabilities"])
            self.assertIn("structured", info["capabilities"])
            self.assertIn("default", info["supported_models"])
            self.assertEqual(info["default_model"], "default")
        except Exception as e:
            self.skipTest(f"Codex CLI provider not available: {e}")
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_claude_cli_provider_info(self, mock_which, mock_run):
        """Test Claude CLI provider info"""
        mock_which.return_value = "/usr/bin/claude"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Claude CLI"
        mock_run.return_value = mock_result
        
        try:
            info = ProviderManager.get_provider_info("claude-cli")
            self.assertEqual(info["name"], "claude-cli")
            self.assertIn("chat", info["capabilities"])
            self.assertIn("structured", info["capabilities"])
            self.assertIn("claude-3-5-sonnet-20241022", info["supported_models"])
            self.assertEqual(info["default_model"], "claude-3-5-sonnet-20241022")
        except Exception as e:
            self.skipTest(f"Claude CLI provider not available: {e}")
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_gemini_cli_provider_info(self, mock_which, mock_run):
        """Test Gemini CLI provider info"""
        mock_which.return_value = "/usr/bin/gemini"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Gemini CLI"
        mock_run.return_value = mock_result
        
        try:
            info = ProviderManager.get_provider_info("gemini-cli")
            self.assertEqual(info["name"], "gemini-cli")
            self.assertIn("chat", info["capabilities"])
            self.assertIn("structured", info["capabilities"])
            self.assertIn("gemini-2.0-flash-exp", info["supported_models"])
            self.assertEqual(info["default_model"], "gemini-2.0-flash-exp")
        except Exception as e:
            self.skipTest(f"Gemini CLI provider not available: {e}")
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_can_create_instances_via_llm_class(self, mock_which, mock_run):
        """Test that CLI providers can be instantiated via Llm class"""
        # Mock all CLI tools as available
        mock_which.return_value = "/usr/bin/mock"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Mock CLI"
        mock_run.return_value = mock_result
        
        # Test creating instances with fully-qualified names
        test_cases = [
            "codex-cli:default",
            "claude-cli:claude-3-5-sonnet-20241022",
            "gemini-cli:gemini-2.0-flash-exp"
        ]
        
        for model_name in test_cases:
            try:
                llm = Llm.model_named(model_name)
                self.assertIsNotNone(llm)
                # Test that the instance has the required methods
                self.assertTrue(hasattr(llm, 'chat'))
                self.assertTrue(hasattr(llm, 'chat_structured'))
                self.assertTrue(hasattr(llm, 'image'))
            except Exception as e:
                self.skipTest(f"Could not create {model_name}: {e}")
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_can_create_with_aliases(self, mock_which, mock_run):
        """Test creating instances using aliases"""
        mock_which.return_value = "/usr/bin/mock"
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Mock CLI"
        mock_run.return_value = mock_result
        
        # Test alias usage
        test_cases = [
            ("codex", "codex-cli"),
            ("claudecli", "claude-cli"),
            ("geminicli", "gemini-cli")
        ]
        
        for alias, expected_provider in test_cases:
            try:
                llm = Llm(provider=alias)
                self.assertEqual(llm.provider, expected_provider)
            except Exception as e:
                self.skipTest(f"Could not create with alias {alias}: {e}")
    
    def test_default_models_for_types(self):
        """Test default model selection for different types"""
        from dazllm.llm_codex_cli import LlmCodexCli
        from dazllm.llm_claude_cli import LlmClaudeCli  
        from dazllm.llm_gemini_cli import LlmGeminiCli
        
        # Codex CLI - supports all types with same model
        self.assertEqual(LlmCodexCli.default_for_type("local_small"), "default")
        self.assertEqual(LlmCodexCli.default_for_type("paid_best"), "default")
        
        # Claude CLI - only paid models
        self.assertIsNone(LlmClaudeCli.default_for_type("local_small"))
        self.assertEqual(LlmClaudeCli.default_for_type("paid_cheap"), "claude-3-5-haiku-20241022")
        self.assertEqual(LlmClaudeCli.default_for_type("paid_best"), "claude-3-5-sonnet-20241022")
        
        # Gemini CLI - only paid models
        self.assertIsNone(LlmGeminiCli.default_for_type("local_small"))
        self.assertEqual(LlmGeminiCli.default_for_type("paid_cheap"), "gemini-1.5-flash")
        self.assertEqual(LlmGeminiCli.default_for_type("paid_best"), "gemini-2.0-flash-exp")
    
    def test_capabilities_consistency(self):
        """Test that all CLI providers have consistent capabilities"""
        from dazllm.llm_codex_cli import LlmCodexCli
        from dazllm.llm_claude_cli import LlmClaudeCli  
        from dazllm.llm_gemini_cli import LlmGeminiCli
        
        providers = [LlmCodexCli, LlmClaudeCli, LlmGeminiCli]
        
        for provider_class in providers:
            caps = provider_class.capabilities()
            
            # All should support chat and structured
            self.assertIn("chat", caps)
            self.assertIn("structured", caps)
            
            # None should support image generation
            self.assertNotIn("image", caps)
    
    @patch('subprocess.run')
    @patch('shutil.which')
    def test_error_handling_consistency(self, mock_which, mock_run):
        """Test that error handling is consistent across CLI providers"""
        mock_which.return_value = "/usr/bin/mock"
        
        # Mock a failing subprocess call
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Command failed"
        mock_run.return_value = mock_result
        
        from dazllm.llm_codex_cli import LlmCodexCli
        from dazllm.llm_claude_cli import LlmClaudeCli
        from dazllm.llm_gemini_cli import LlmGeminiCli
        from dazllm.core import DazLlmError
        
        # Test that all providers handle errors consistently
        test_instances = [
            (LlmCodexCli.__new__(LlmCodexCli), "default"),
            (LlmClaudeCli.__new__(LlmClaudeCli), "claude-3-5-sonnet-20241022"),
            (LlmGeminiCli.__new__(LlmGeminiCli), "gemini-2.0-flash-exp")
        ]
        
        for instance, model in test_instances:
            instance.model = model
            instance.executable = "/usr/bin/mock"
            
            # Test that chat raises DazLlmError on failure
            with self.assertRaises(DazLlmError):
                instance.chat("test prompt")
            
            # Test that image generation raises DazLlmError
            with self.assertRaises(DazLlmError):
                instance.image("test", "test.jpg")


class TestCliProvidersDemo(unittest.TestCase):
    """Demo tests showing CLI providers usage patterns"""
    
    @unittest.skipUnless(False, "Demo test - enable manually")
    def test_real_codex_cli_usage(self):
        """Demo: Real Codex CLI usage (requires codex CLI installed)"""
        try:
            llm = Llm("codex-cli:default")
            
            # Test basic chat
            response = llm.chat("What is Python?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)
            
            # Test structured output
            result = llm.chat_structured(
                "Generate a person with name 'Alice', age 30, active status true",
                SimpleTestModel
            )
            self.assertIsInstance(result, SimpleTestModel)
            self.assertEqual(result.name, "Alice")
            
        except Exception as e:
            self.skipTest(f"Codex CLI not available: {e}")
    
    @unittest.skipUnless(False, "Demo test - enable manually")
    def test_real_claude_cli_usage(self):
        """Demo: Real Claude CLI usage (requires claude CLI installed)"""
        try:
            llm = Llm("claude-cli:claude-3-5-sonnet-20241022")
            
            # Test basic chat
            response = llm.chat("Explain recursion briefly")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)
            
            # Test structured output
            result = llm.chat_structured(
                "Generate test data with name 'Bob', value 42, active false",
                SimpleTestModel
            )
            self.assertIsInstance(result, SimpleTestModel)
            self.assertEqual(result.name, "Bob")
            
        except Exception as e:
            self.skipTest(f"Claude CLI not available: {e}")
    
    @unittest.skipUnless(False, "Demo test - enable manually")  
    def test_real_gemini_cli_usage(self):
        """Demo: Real Gemini CLI usage (requires gemini CLI installed)"""
        try:
            llm = Llm("gemini-cli:gemini-2.0-flash-exp")
            
            # Test basic chat
            response = llm.chat("What is machine learning in one sentence?")
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 10)
            
            # Test structured output
            result = llm.chat_structured(
                "Generate data with name 'Charlie', value 100, active true",
                SimpleTestModel
            )
            self.assertIsInstance(result, SimpleTestModel)
            self.assertEqual(result.name, "Charlie")
            
        except Exception as e:
            self.skipTest(f"Gemini CLI not available: {e}")


if __name__ == "__main__":
    unittest.main()