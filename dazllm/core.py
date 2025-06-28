"""
dazllm - A simple, unified interface for all major LLMs
"""

from __future__ import annotations
import keyring
from typing import Optional, Union, Dict, List, Type, Literal, TypedDict, Set
from pydantic import BaseModel
from enum import Enum


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


Conversation = Union[str, List[Message]]


class ModelType(Enum):
    LOCAL_SMALL = "local_small"  # ~1B params
    LOCAL_MEDIUM = "local_medium"  # ~7B params
    LOCAL_LARGE = "local_large"  # ~14B params
    PAID_CHEAP = "paid_cheap"  # Cost-effective cloud models
    PAID_BEST = "paid_best"  # Best performance cloud models


class DazLlmError(Exception):
    """Base exception for dazllm"""

    pass


class ConfigurationError(DazLlmError):
    """Raised when configuration is missing or invalid"""

    pass


class ModelNotFoundError(DazLlmError):
    """Raised when requested model is not available"""

    pass


# PROVIDER REGISTRY - Add new providers here (one line each!)
PROVIDERS = {
    "openai": {"class": "LlmOpenai", "module": ".llm_openai", "aliases": []},
    "anthropic": {
        "class": "LlmAnthropic",
        "module": ".llm_anthropic",
        "aliases": ["claude"],
    },
    "google": {"class": "LlmGoogle", "module": ".llm_google", "aliases": ["gemini"]},
    "ollama": {"class": "LlmOllama", "module": ".llm_ollama", "aliases": []},
    # Add new providers here - just one line each!
    # "huggingface": {"class": "LlmHuggingface", "module": ".llm_huggingface", "aliases": ["hf"]},
}


class Llm:
    """
    Unified interface for all major LLMs

    Usage:
        # Instance-based
        llm = Llm("openai:gpt-4")
        response = llm.chat("Hello!")

        # Static/module-level
        response = Llm.chat("Hello!", model="openai:gpt-4")
        response = Llm.chat("Hello!", model_type=ModelType.PAID_BEST)
    """

    _cached: Dict[str, Llm] = {}

    def __init__(self, model_name: str):
        """Initialize with full model name like 'openai:gpt-4'"""
        self.model_name = model_name
        self.provider, self.model = self._parse_model_name(model_name)

    @staticmethod
    def _resolve_provider_alias(provider: str) -> str:
        """Resolve provider alias to actual provider name"""
        # Check if it's already a real provider
        if provider in PROVIDERS:
            return provider

        # Look for alias in all providers
        for real_provider, provider_info in PROVIDERS.items():
            if provider in provider_info.get("aliases", []):
                return real_provider

        # Return original if no alias found
        return provider

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse 'provider:model' format"""
        if ":" not in model_name:
            raise ModelNotFoundError(
                f"Model name must be in format 'provider:model', got: {model_name}"
            )

        provider, model = model_name.split(":", 1)

        # Resolve aliases
        provider = self._resolve_provider_alias(provider)

        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")

        return provider, model

    @classmethod
    def _get_provider_class(cls, provider: str):
        """Get provider class by importing it dynamically"""
        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")

        provider_info = PROVIDERS[provider]
        module_name = provider_info["module"]
        class_name = provider_info["class"]

        # Dynamic import
        import importlib

        module = importlib.import_module(module_name, package=__package__)
        return getattr(module, class_name)

    @classmethod
    def _create_provider_instance(cls, model_name: str) -> Llm:
        """Create the appropriate provider instance"""
        provider, model = cls._parse_model_name_static(model_name)
        provider_class = cls._get_provider_class(provider)
        return provider_class(model)

    @staticmethod
    def _parse_model_name_static(model_name: str) -> tuple[str, str]:
        """Static version of _parse_model_name"""
        if ":" not in model_name:
            raise ModelNotFoundError(
                f"Model name must be in format 'provider:model', got: {model_name}"
            )

        provider, model = model_name.split(":", 1)

        # Resolve aliases
        provider = Llm._resolve_provider_alias(provider)

        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")

        return provider, model

    @classmethod
    def _resolve_model(
        cls, model: Optional[str] = None, model_type: Optional[ModelType] = None
    ) -> str:
        """Resolve model from name, type, or defaults"""

        # Error if both model and model_type provided
        if model and model_type:
            raise DazLlmError("Cannot specify both model name and model_type")

        # Direct model name
        if model:
            # Handle provider-only names like "openai" -> use provider default
            if ":" not in model:
                return cls._get_provider_default(model)
            return model

        # Model type - get default for that type
        if model_type:
            return cls._get_default_for_type(model_type)

        # No model specified - try keyring default first
        default_model = keyring.get_password("dazllm", "default_model")
        if default_model:
            return default_model

        # Fall back to trying configured models in order
        return cls._find_configured_model()

    @classmethod
    def _get_default_for_type(cls, model_type: ModelType) -> str:
        """Get default model for a model type by asking each provider"""
        type_str = model_type.value

        # Ask each provider in order for their default for this type
        for provider_name in PROVIDERS.keys():
            try:
                provider_class = cls._get_provider_class(provider_name)
                provider_default = provider_class.default_for_type(type_str)
                if provider_default:
                    return f"{provider_name}:{provider_default}"
            except:
                continue

        raise ModelNotFoundError(f"No provider supports model type: {model_type.value}")

    @classmethod
    def _get_provider_default(cls, provider: str) -> str:
        """Get default model for a provider"""
        # Resolve aliases
        provider = cls._resolve_provider_alias(provider)

        provider_class = cls._get_provider_class(provider)
        default_model = provider_class.default_model()
        return f"{provider}:{default_model}"

    @classmethod
    def _find_configured_model(cls) -> str:
        """Find first model that has proper configuration by trying each provider in order"""
        # Try each provider in order
        for provider_name in PROVIDERS.keys():
            try:
                cls._check_provider_config(provider_name)
                # If configured, get their default model
                default_model = cls._get_provider_default(provider_name)
                return default_model
            except (ConfigurationError, ModelNotFoundError):
                continue

        raise ConfigurationError(
            "No properly configured models found. Run 'dazllm --check' to verify setup."
        )

    @classmethod
    def _check_provider_config(cls, provider: str):
        """Check if provider is configured"""
        provider_class = cls._get_provider_class(provider)
        provider_class.check_config()

    @classmethod
    def model_named(cls, model_name: str) -> Llm:
        """Get or create model instance"""
        if model_name in cls._cached:
            return cls._cached[model_name]

        instance = cls._create_provider_instance(model_name)
        cls._cached[model_name] = instance
        return instance

    # Introspection methods
    @classmethod
    def get_providers(cls) -> List[str]:
        """Get list of available providers"""
        return list(PROVIDERS.keys())

    @classmethod
    def get_provider_info(cls, provider: str) -> Dict:
        """Get information about a provider"""
        # Resolve aliases
        provider = cls._resolve_provider_alias(provider)

        if provider not in PROVIDERS:
            raise ModelNotFoundError(f"Unknown provider: {provider}")

        provider_class = cls._get_provider_class(provider)

        try:
            is_configured = True
            cls._check_provider_config(provider)
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

    # Abstract methods to be implemented by subclasses
    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        """Chat with the LLM"""
        raise NotImplementedError("chat should be implemented by subclasses")

    def chat_structured(
        self, conversation: Conversation, schema: Type[BaseModel], context_size: int = 0
    ) -> BaseModel:
        """Chat with structured output using Pydantic schema"""
        raise NotImplementedError("chat_structured should be implemented by subclasses")

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        """Generate an image"""
        raise NotImplementedError("image should be implemented by subclasses")

    # Provider interface methods (to be implemented by each provider)
    @staticmethod
    def capabilities() -> Set[str]:
        """Return set of capabilities this provider supports"""
        raise NotImplementedError("capabilities should be implemented by subclasses")

    @staticmethod
    def supported_models() -> List[str]:
        """Return list of models this provider supports"""
        raise NotImplementedError(
            "supported_models should be implemented by subclasses"
        )

    @staticmethod
    def default_model() -> str:
        """Return default model for this provider"""
        raise NotImplementedError("default_model should be implemented by subclasses")

    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        """Return default model for a given type"""
        raise NotImplementedError(
            "default_for_type should be implemented by subclasses"
        )

    @staticmethod
    def check_config():
        """Check if provider is properly configured"""
        raise NotImplementedError("check_config should be implemented by subclasses")

    # Static methods for module-level interface
    @classmethod
    def chat(
        cls,
        conversation: Conversation,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        force_json: bool = False,
    ) -> str:
        """Static chat method"""
        model_name = cls._resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.chat(conversation, force_json)

    @classmethod
    def chat_structured(
        cls,
        conversation: Conversation,
        schema: Type[BaseModel],
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        context_size: int = 0,
    ) -> BaseModel:
        """Static structured chat method"""
        model_name = cls._resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.chat_structured(conversation, schema, context_size)

    @classmethod
    def image(
        cls,
        prompt: str,
        file_name: str,
        width: int = 1024,
        height: int = 1024,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
    ) -> str:
        """Static image generation method"""
        model_name = cls._resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.image(prompt, file_name, width, height)


def check_configuration() -> Dict[str, Dict[str, Union[bool, str]]]:
    """
    Check if all providers are properly configured

    Returns:
        Dict mapping provider names to configuration status and details
    """
    status = {}

    for provider in PROVIDERS.keys():
        try:
            Llm._check_provider_config(provider)
            status[provider] = {"configured": True, "error": None}
        except ConfigurationError as e:
            status[provider] = {"configured": False, "error": str(e)}
        except Exception as e:
            status[provider] = {"configured": False, "error": f"Unknown error: {e}"}

    return status


# Convenience exports
__all__ = [
    "Llm",
    "ModelType",
    "Message",
    "Conversation",
    "DazLlmError",
    "ConfigurationError",
    "ModelNotFoundError",
    "check_configuration",
    "PROVIDERS",
]
