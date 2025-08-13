"""Provider management utilities for dazllm

This module handles provider registration, instantiation, and validation.
It centralizes all provider-related operations including alias resolution,
class loading, and configuration checking.
"""

from __future__ import annotations
import importlib
import unittest
from typing import Dict, Type

PROVIDERS = {
    "openai": {"class": "LlmOpenai", "module": ".llm_openai", "aliases": []},
    "anthropic": {
        "class": "LlmAnthropic",
        "module": ".llm_anthropic",
        "aliases": ["claude"],
    },
    "google": {"class": "LlmGoogle", "module": ".llm_google", "aliases": ["gemini"]},
    "ollama": {"class": "LlmOllama", "module": ".llm_ollama", "aliases": []},
    "lm-studio": {
        "class": "LlmLmstudio",
        "module": ".llm_lmstudio",
        "aliases": ["lmstudio"],
    },
}


class DazLlmError(Exception):
    """Base exception for dazllm"""


class ModelNotFoundError(DazLlmError):
    """Raised when requested model is not available"""


class ConfigurationError(DazLlmError):
    """Raised when configuration is missing or invalid"""


class ProviderManager:
    """Manages provider registration and instantiation"""

    @staticmethod
    def resolve_provider_alias(provider: str) -> str:
        """Resolve provider aliases to canonical provider names"""
        if provider in PROVIDERS:
            return provider
        for real_provider, provider_info in PROVIDERS.items():
            if provider in provider_info.get("aliases", []):
                return real_provider
        return provider

    @staticmethod
    def get_provider_class(provider: str) -> Type:
        """Get the provider class for a given provider name"""
        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")
        provider_info = PROVIDERS[provider]
        module = importlib.import_module(provider_info["module"], package=__package__)
        return getattr(module, provider_info["class"])

    @classmethod
    def create_provider_instance(cls, provider: str, model: str):
        """Create a provider instance for the given provider and model"""
        provider_class = cls.get_provider_class(provider)
        return provider_class(f"{provider}:{model}")

    @classmethod
    def check_provider_config(cls, provider: str):
        """Check if a provider is properly configured"""
        provider_class = cls.get_provider_class(provider)
        provider_class.check_config()

    @staticmethod
    def get_providers() -> list[str]:
        """Get list of all available providers"""
        return list(PROVIDERS.keys())

    @classmethod
    def get_provider_info(cls, provider: str) -> Dict:
        """Get detailed information about a provider"""
        provider = cls.resolve_provider_alias(provider)
        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")
        provider_class = cls.get_provider_class(provider)
        try:
            is_configured = True
            cls.check_provider_config(provider)
        except ConfigurationError:
            is_configured = False
        return {
            "name": provider,
            "configured": is_configured,
            "capabilities": provider_class.capabilities(),
            "supported_models": provider_class.supported_models(),
            "default_model": provider_class.default_model(),
        }

    @classmethod
    def get_all_providers_info(cls) -> Dict[str, Dict]:
        """Get information about all providers"""
        return {
            provider: cls.get_provider_info(provider) for provider in PROVIDERS.keys()
        }


class TestProviderManager(unittest.TestCase):
    """Tests for provider manager functionality"""

    def test_providers_registry(self):
        """Test providers registry structure"""
        self.assertIsInstance(PROVIDERS, dict)
        self.assertIn("openai", PROVIDERS)
        self.assertIn("lm-studio", PROVIDERS)

    def test_get_providers(self):
        """Test getting list of providers"""
        providers = ProviderManager.get_providers()
        self.assertIsInstance(providers, list)
        self.assertIn("openai", providers)

    def test_alias_resolution(self):
        """Test provider alias resolution"""
        self.assertEqual(ProviderManager.resolve_provider_alias("claude"), "anthropic")
        self.assertEqual(ProviderManager.resolve_provider_alias("gemini"), "google")
        self.assertEqual(ProviderManager.resolve_provider_alias("openai"), "openai")

    def test_exception_hierarchy(self):
        """Test exception inheritance"""
        self.assertTrue(issubclass(ConfigurationError, DazLlmError))
        self.assertTrue(issubclass(ModelNotFoundError, DazLlmError))


__all__ = [
    "ProviderManager",
    "PROVIDERS",
    "ModelNotFoundError",
    "ConfigurationError",
    "DazLlmError",
]
