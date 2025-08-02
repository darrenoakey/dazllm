"""dazllm - A simple, unified interface for all major LLMs"""

from __future__ import annotations
import unittest
from typing import Optional, Union, Dict, List, Literal, TypedDict, Set
from pydantic import BaseModel
from enum import Enum


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


Conversation = Union[str, List[Message]]


class ModelType(Enum):
    LOCAL_SMALL = "local_small"
    LOCAL_MEDIUM = "local_medium"
    LOCAL_LARGE = "local_large"
    PAID_CHEAP = "paid_cheap"
    PAID_BEST = "paid_best"


# Import all exceptions from provider_manager to maintain hierarchy
from .provider_manager import DazLlmError, ConfigurationError, ModelNotFoundError


class Llm:
    """Unified interface for all major LLMs"""

    _cached: Dict[str, Llm] = {}

    def __init__(self, model_name: str):
        from .model_resolver import ModelResolver

        self.model_name = model_name
        self.provider, self.model = ModelResolver.parse_model_name(model_name)

    @classmethod
    def model_named(cls, model_name: str) -> Llm:
        from .provider_manager import ProviderManager

        print(f"Got model {model_name}")

        if model_name in cls._cached:
            return cls._cached[model_name]
        provider, model = cls._parse_model_name_static(model_name)
        instance = ProviderManager.create_provider_instance(provider, model)
        cls._cached[model_name] = instance
        return instance

    @staticmethod
    def _parse_model_name_static(model_name: str) -> tuple[str, str]:
        from .model_resolver import ModelResolver

        return ModelResolver.parse_model_name(model_name)

    @classmethod
    def get_providers(cls) -> List[str]:
        from .provider_manager import ProviderManager

        return ProviderManager.get_providers()

    @classmethod
    def get_provider_info(cls, provider: str) -> Dict:
        from .provider_manager import ProviderManager

        return ProviderManager.get_provider_info(provider)

    @classmethod
    def get_all_providers_info(cls) -> Dict[str, Dict]:
        from .provider_manager import ProviderManager

        return ProviderManager.get_all_providers_info()

    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        raise NotImplementedError("chat should be implemented by subclasses")

    def chat_structured(
        self, conversation: Conversation, schema: BaseModel, context_size: int = 0
    ) -> BaseModel:
        raise NotImplementedError("chat_structured should be implemented by subclasses")

    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        raise NotImplementedError("image should be implemented by subclasses")

    @staticmethod
    def capabilities() -> Set[str]:
        raise NotImplementedError("capabilities should be implemented by subclasses")

    @staticmethod
    def supported_models() -> List[str]:
        raise NotImplementedError(
            "supported_models should be implemented by subclasses"
        )

    @staticmethod
    def default_model() -> str:
        raise NotImplementedError("default_model should be implemented by subclasses")

    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        raise NotImplementedError(
            "default_for_type should be implemented by subclasses"
        )

    @staticmethod
    def check_config():
        raise NotImplementedError("check_config should be implemented by subclasses")

    @classmethod
    def chat_static(
        cls,
        conversation: Conversation,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        force_json: bool = False,
    ) -> str:
        from .model_resolver import ModelResolver

        model_name = ModelResolver.resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.chat(conversation, force_json)

    @classmethod
    def chat_structured_static(
        cls,
        conversation: Conversation,
        schema: BaseModel,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
        context_size: int = 0,
    ) -> BaseModel:
        from .model_resolver import ModelResolver

        model_name = ModelResolver.resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.chat_structured(conversation, schema, context_size)

    @classmethod
    def image_static(
        cls,
        prompt: str,
        file_name: str,
        width: int = 1024,
        height: int = 1024,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
    ) -> str:
        from .model_resolver import ModelResolver

        model_name = ModelResolver.resolve_model(model, model_type)
        llm = cls.model_named(model_name)
        return llm.image(prompt, file_name, width, height)


def check_configuration() -> Dict[str, Dict[str, Union[bool, str]]]:
    """Check if all providers are properly configured"""
    from .provider_manager import ProviderManager, PROVIDERS

    status = {}
    for provider in PROVIDERS.keys():
        try:
            # Use the public get_provider_info method which handles errors properly
            provider_info = ProviderManager.get_provider_info(provider)
            status[provider] = {
                "configured": provider_info["configured"],
                "error": None,
            }
        except ModelNotFoundError as e:
            status[provider] = {"configured": False, "error": str(e)}
        except (ImportError, AttributeError) as e:
            status[provider] = {"configured": False, "error": f"Import error: {e}"}
    return status


class TestLlmCore(unittest.TestCase):
    """Essential tests for core Llm functionality"""

    def test_model_type_enum(self):
        """Test ModelType enum has correct values"""
        self.assertEqual(ModelType.LOCAL_SMALL.value, "local_small")
        self.assertEqual(ModelType.PAID_BEST.value, "paid_best")

    def test_exception_hierarchy(self):
        """Test exception class hierarchy"""
        self.assertTrue(issubclass(ConfigurationError, DazLlmError))
        self.assertTrue(issubclass(ModelNotFoundError, DazLlmError))

    def test_get_providers(self):
        """Test getting list of providers"""
        providers = Llm.get_providers()
        self.assertIsInstance(providers, list)
        self.assertIn("openai", providers)

    def test_check_configuration_function(self):
        """Test check_configuration function"""
        status = check_configuration()
        self.assertIsInstance(status, dict)

    def test_message_structure(self):
        """Test Message TypedDict structure"""
        msg = {"role": "user", "content": "Hello"}
        self.assertIn("role", msg)
        self.assertIn("content", msg)

    def test_conversation_types(self):
        """Test Conversation type usage"""
        conv_str = "Hello"
        conv_list = [{"role": "user", "content": "Hello"}]
        self.assertIsInstance(conv_str, (str, list))
        self.assertIsInstance(conv_list, (str, list))


__all__ = [
    "Llm",
    "ModelType",
    "Message",
    "Conversation",
    "DazLlmError",
    "ConfigurationError",
    "ModelNotFoundError",
    "check_configuration",
]
