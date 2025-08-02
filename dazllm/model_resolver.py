"""Model resolution utilities for dazllm

This module handles model name parsing, provider resolution, and finding
appropriate models based on type preferences. It centralizes all the logic
for determining which model to use given various input parameters.
"""

from __future__ import annotations
import keyring
import unittest
from typing import Optional
from .core import ModelType
from .provider_manager import ProviderManager, ModelNotFoundError, ConfigurationError, DazLlmError


class ModelResolver:
    """Handles model name resolution and provider selection"""

    @staticmethod
    def parse_model_name(model_name: str) -> tuple[str, str]:
        """Parse a model name into provider and model components"""
        if ":" not in model_name:
            raise ModelNotFoundError(f"Model name must be in format 'provider:model', got: {model_name}")
        provider, model = model_name.split(":", 1)
        provider = ProviderManager.resolve_provider_alias(provider)
        if provider not in ProviderManager.get_providers():
            raise ModelNotFoundError(f"Unknown provider: {provider}")
        return provider, model

    @classmethod
    def resolve_model(cls, model: Optional[str] = None, model_type: Optional[ModelType] = None) -> str:
        """Resolve a model name from various input parameters"""
        if model and model_type:
            raise DazLlmError("Cannot specify both model name and model_type")
        if model:
            if ":" not in model:
                return cls._get_provider_default(model)
            return model
        if model_type:
            return cls._get_default_for_type(model_type)
        default_model = keyring.get_password("dazllm", "default_model")
        if default_model:
            return default_model
        return cls._find_configured_model()

    @classmethod
    def _get_default_for_type(cls, model_type: ModelType) -> str:
        """Get the default model for a specific model type"""
        type_str = model_type.value
        for provider_name in ProviderManager.get_providers():
            try:
                provider_class = ProviderManager.get_provider_class(provider_name)
                provider_default = provider_class.default_for_type(type_str)
                if provider_default:
                    return f"{provider_name}:{provider_default}"
            except (ImportError, AttributeError, ModelNotFoundError):
                continue
        raise ModelNotFoundError(f"No provider supports model type: {model_type.value}")

    @classmethod
    def _get_provider_default(cls, provider: str) -> str:
        """Get the default model for a provider"""
        provider = ProviderManager.resolve_provider_alias(provider)
        provider_class = ProviderManager.get_provider_class(provider)
        default_model = provider_class.default_model()
        return f"{provider}:{default_model}"

    @classmethod
    def _find_configured_model(cls) -> str:
        """Find the first properly configured model"""
        for provider_name in ProviderManager.get_providers():
            try:
                ProviderManager.check_provider_config(provider_name)
                default_model = cls._get_provider_default(provider_name)
                return default_model
            except (ConfigurationError, ModelNotFoundError):
                continue
        raise ConfigurationError("No properly configured models found. Run 'dazllm --check' to verify setup.")


class TestModelResolver(unittest.TestCase):
    """Tests for model resolver functionality"""

    def test_parse_model_name_valid(self):
        """Test parsing valid model names"""
        provider, model = ModelResolver.parse_model_name("openai:gpt-4")
        self.assertEqual(provider, "openai")
        self.assertEqual(model, "gpt-4")

    def test_parse_model_name_alias(self):
        """Test parsing model names with provider aliases"""
        provider, model = ModelResolver.parse_model_name("claude:sonnet")
        self.assertEqual(provider, "anthropic")
        self.assertEqual(model, "sonnet")

    def test_parse_model_name_invalid(self):
        """Test parsing invalid model names"""
        with self.assertRaises(ModelNotFoundError):
            ModelResolver.parse_model_name("invalid_model_name")

    def test_resolve_model_conflicts(self):
        """Test that specifying both model and model_type raises error"""
        with self.assertRaises(DazLlmError):
            ModelResolver.resolve_model("openai:gpt-4", ModelType.PAID_BEST)


__all__ = ["ModelResolver"]
