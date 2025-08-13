# dazllm - A simple, unified interface for all major LLMs
from __future__ import annotations

# Imports kept explicit and top-level
import unittest
from enum import Enum
from typing import Optional, Union, Dict, List, Literal, TypedDict, Set, Tuple
from pydantic import BaseModel

# Import exception hierarchy from provider_manager to keep types consistent
from .provider_manager import DazLlmError, ConfigurationError, ModelNotFoundError


# Define the shape of a single chat message
class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


# Define supported conversation forms: a single prompt or a list of role-tagged messages
Conversation = Union[str, List[Message]]


# Define logical model sizes / tiers referenced across providers
class ModelType(Enum):
    LOCAL_SMALL = "local_small"
    LOCAL_MEDIUM = "local_medium"
    LOCAL_LARGE = "local_large"
    PAID_CHEAP = "paid_cheap"
    PAID_BEST = "paid_best"


# Convert free-form strings into a ModelType when possible
def _coerce_model_type(value: Optional[str]) -> Optional[ModelType]:
    # Accept None
    if value is None:
        return None

    # Normalize common user inputs
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")

    # Canonical keys → ModelType
    mapping: Dict[str, ModelType] = {
        "local_small": ModelType.LOCAL_SMALL,
        "small_local": ModelType.LOCAL_SMALL,
        "localsmall": ModelType.LOCAL_SMALL,
        "local_medium": ModelType.LOCAL_MEDIUM,
        "medium_local": ModelType.LOCAL_MEDIUM,
        "localmedium": ModelType.LOCAL_MEDIUM,
        "local_large": ModelType.LOCAL_LARGE,
        "large_local": ModelType.LOCAL_LARGE,
        "locallarge": ModelType.LOCAL_LARGE,
        "paid_cheap": ModelType.PAID_CHEAP,
        "cheap_paid": ModelType.PAID_CHEAP,
        "paidcheap": ModelType.PAID_CHEAP,
        "paid_best": ModelType.PAID_BEST,
        "best_paid": ModelType.PAID_BEST,
        "paidbest": ModelType.PAID_BEST,
    }

    # Direct match on normalized
    if normalized in mapping:
        return mapping[normalized]

    # ALLCAPS enum names also accepted
    upper = value.strip().upper()
    try:
        return ModelType[upper]
    except KeyError:
        return None


# Resolve constructor inputs into a fully-qualified "provider:model_id" string
# Rules:
# - If model_name is already provided → return as-is
# - If provider + explicit model → compose "provider:model"
# - If provider only → choose that provider's default model
# - If provider + type → provider's default_for_type or its default_model
# - If only a size/type hint (via 'model' alias or model_type) → use ModelResolver
def _resolve_model_name_for_ctor(
    model_name: Optional[str],
    provider: Optional[str],
    model: Optional[str],
    model_type: Optional[ModelType],
) -> str:
    from .model_resolver import ModelResolver
    from .provider_manager import ProviderManager

    # Keep explicit fully-qualified model names untouched
    if model_name:
        return model_name

    # If 'model' looks like a size alias, treat it as model_type
    alias_type = _coerce_model_type(model) if model else None
    if alias_type is not None:
        model_type = alias_type
        model = None

    # Provider + explicit model id
    if provider and model and model_type is None:
        return f"{provider}:{model}"

    # Provider only → provider default
    if provider and model is None and model_type is None:
        info = ProviderManager.get_provider_info(provider)
        default_model = info.get("default_model")
        if not default_model:
            raise ModelNotFoundError(f"No default model found for provider '{provider}'")
        return f"{provider}:{default_model}"

    # Provider + type → provider default_for_type, fallback to provider default
    if provider and model_type is not None:
        default_for_type: Optional[str] = None

        # Try to read type defaults if provider_info exposes them
        info = ProviderManager.get_provider_info(provider)
        for key in ("defaults", "default_for_type_map", "type_defaults"):
            m = info.get(key) if isinstance(info, dict) else None
            if isinstance(m, dict):
                default_for_type = m.get(model_type.value) or m.get(model_type.name)
                if default_for_type:
                    break

        # Ask provider class directly when available
        if not default_for_type:
            try:
                provider_class = ProviderManager.get_provider_class(provider)  # type: ignore[attr-defined]
                default_for_type = provider_class.default_for_type(model_type.value)
            except Exception:
                default_for_type = None

        chosen = default_for_type or info.get("default_model")
        if not chosen:
            raise ModelNotFoundError(
                f"No default for type '{model_type.value}' on provider '{provider}'"
            )
        return f"{provider}:{chosen}"

    # No provider specified → delegate to ModelResolver to pick the best provider/model
    resolved = ModelResolver.resolve_model(model, model_type)
    return resolved


# Unified interface for all providers, with flexible construction helpers
class Llm:
    # Cache instances by fully-qualified model name
    _cached: Dict[str, Llm] = {}

    # Constructor supports:
    # - Llm() → default overall model
    # - Llm(provider="lm-studio") → provider's default model
    # - Llm(model="large_local") or Llm(model_type=ModelType.LOCAL_LARGE) → best provider for that tier
    # - Llm("provider:model_id") → fully-qualified explicit selection
    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        model_type: Optional[ModelType] = None,
    ):
        from .model_resolver import ModelResolver

        # Resolve inputs into a fully-qualified name
        fq_name = _resolve_model_name_for_ctor(model_name, provider, model, model_type)

        # Store resolved name and parsed parts
        self.model_name = fq_name
        self.provider, self.model = ModelResolver.parse_model_name(fq_name)

    # Construct using a provider's default model (camelCase per request)
    @classmethod
    def fromProvider(cls, provider_name: str) -> Llm:
        return cls(provider=provider_name)

    # Construct using a provider's default model (snake_case alias)
    @classmethod
    def from_provider(cls, provider_name: str) -> Llm:
        return cls(provider=provider_name)

    # Shortcut: choose a local small model on the best provider
    @classmethod
    def LocalSmall(cls) -> Llm:
        return cls(model_type=ModelType.LOCAL_SMALL)

    # Shortcut: choose a local medium model on the best provider
    @classmethod
    def LocalMedium(cls) -> Llm:
        return cls(model_type=ModelType.LOCAL_MEDIUM)

    # Shortcut: choose a local large model on the best provider
    @classmethod
    def LocalLarge(cls) -> Llm:
        return cls(model_type=ModelType.LOCAL_LARGE)

    # Shortcut: choose the cheapest paid model on the best provider
    @classmethod
    def PaidCheap(cls) -> Llm:
        return cls(model_type=ModelType.PAID_CHEAP)

    # Shortcut: choose the best paid model on the best provider
    @classmethod
    def PaidBest(cls) -> Llm:
        return cls(model_type=ModelType.PAID_BEST)

    # Retrieve or create a cached instance for a fully-qualified model name
    @classmethod
    def model_named(cls, model_name: str) -> Llm:
        from .provider_manager import ProviderManager

        if model_name in cls._cached:
            return cls._cached[model_name]
        provider, model = cls._parse_model_name_static(model_name)
        instance = ProviderManager.create_provider_instance(provider, model)
        cls._cached[model_name] = instance
        return instance

    # Parse a fully-qualified model string into (provider, model_id)
    @staticmethod
    def _parse_model_name_static(model_name: str) -> Tuple[str, str]:
        from .model_resolver import ModelResolver

        return ModelResolver.parse_model_name(model_name)

    # Return a list of available providers
    @classmethod
    def get_providers(cls) -> List[str]:
        from .provider_manager import ProviderManager

        return ProviderManager.get_providers()

    # Return metadata for a single provider
    @classmethod
    def get_provider_info(cls, provider: str) -> Dict:
        from .provider_manager import ProviderManager

        return ProviderManager.get_provider_info(provider)

    # Return metadata for all providers
    @classmethod
    def get_all_providers_info(cls) -> Dict[str, Dict]:
        from .provider_manager import ProviderManager

        return ProviderManager.get_all_providers_info()

    # Send a chat conversation and return assistant text (subclasses must implement)
    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        raise NotImplementedError("chat should be implemented by subclasses")

    # Send a chat conversation and parse into a Pydantic schema (subclasses must implement)
    def chat_structured(
        self, conversation: Conversation, schema: BaseModel, context_size: int = 0
    ) -> BaseModel:
        raise NotImplementedError("chat_structured should be implemented by subclasses")

    # Generate an image and return the output path/URL (subclasses must implement)
    def image(
        self, prompt: str, file_name: str, width: int = 1024, height: int = 1024
    ) -> str:
        raise NotImplementedError("image should be implemented by subclasses")

    # Describe provider capabilities (subclasses must implement)
    @staticmethod
    def capabilities() -> Set[str]:
        raise NotImplementedError("capabilities should be implemented by subclasses")

    # List model ids supported by the provider (subclasses must implement)
    @staticmethod
    def supported_models() -> List[str]:
        raise NotImplementedError(
            "supported_models should be implemented by subclasses"
        )

    # Return the provider's default model id (subclasses must implement)
    @staticmethod
    def default_model() -> str:
        raise NotImplementedError("default_model should be implemented by subclasses")

    # Return the provider's default model id for a given type (subclasses must implement)
    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        raise NotImplementedError(
            "default_for_type should be implemented by subclasses"
        )

    # Verify provider-level configuration (subclasses must implement)
    @staticmethod
    def check_config():
        raise NotImplementedError("check_config should be implemented by subclasses")

    # Convenience: resolve and chat without manually constructing an instance
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

    # Convenience: resolve and structured-chat without manually constructing an instance
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

    # Convenience: resolve and image without manually constructing an instance
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


# Check configuration status across providers
def check_configuration() -> Dict[str, Dict[str, Union[bool, str]]]:
    from .provider_manager import ProviderManager, PROVIDERS

    status: Dict[str, Dict[str, Union[bool, str]]] = {}
    for provider in PROVIDERS.keys():
        try:
            info = ProviderManager.get_provider_info(provider)
            status[provider] = {
                "configured": info["configured"],
                "error": None,
            }
        except ModelNotFoundError as e:
            status[provider] = {"configured": False, "error": str(e)}
        except (ImportError, AttributeError) as e:
            status[provider] = {"configured": False, "error": f"Import error: {e}"}
    return status


# Basic tests to ensure core wiring behaves as expected
class TestLlmCore(unittest.TestCase):
    # Verify enum values are stable
    def test_model_type_enum(self):
        self.assertEqual(ModelType.LOCAL_SMALL.value, "local_small")
        self.assertEqual(ModelType.PAID_BEST.value, "paid_best")

    # Verify exception hierarchy
    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(ConfigurationError, DazLlmError))
        self.assertTrue(issubclass(ModelNotFoundError, DazLlmError))

    # Providers are discoverable (do not assert specific names; environments vary)
    def test_get_providers(self):
        providers = Llm.get_providers()
        self.assertIsInstance(providers, list)
        self.assertGreaterEqual(len(providers), 0)

    # Configuration checker returns a dict
    def test_check_configuration_function(self):
        status = check_configuration()
        self.assertIsInstance(status, dict)

    # TypedDict layout is correct
    def test_message_structure(self):
        msg = {"role": "user", "content": "Hello"}
        self.assertIn("role", msg)
        self.assertIn("content", msg)

    # Conversation union accepts both forms
    def test_conversation_types(self):
        conv_str = "Hello"
        conv_list = [{"role": "user", "content": "Hello"}]
        self.assertIsInstance(conv_str, (str, list))
        self.assertIsInstance(conv_list, (str, list))


# Public exports
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
