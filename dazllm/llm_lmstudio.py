# LM Studio provider for dazllm
from __future__ import annotations
import os


# Comment: Keep imports top-level so configuration errors are immediate
from typing import Optional, Union, List, Dict, Set, Any, cast

try:
    import lmstudio as lms
except Exception as _e:
    # Comment: We don't fail import of this module; core.check_config() will surface the error
    lms = None  # type: ignore[assignment]

from pydantic import BaseModel

from .core import Llm, Conversation, Message, ConfigurationError, DazLlmError


# Comment: Central place to control desired context window (tokens)
_DESIRED_CONTEXT_LENGTH = int(os.getenv("DAZ_LMSTUDIO_CONTEXT", "32768"))



# Comment: Build a simple config dict for the SDK; we keep it minimal and deterministic
def _mk_config() -> Dict[str, Any]:
    return {}


# Comment: Convert our Conversation union into an lmstudio Chat or a raw prompt string
def _to_lms_chat(conversation: Conversation) -> Union[str, "lms.Chat"]:
    if isinstance(conversation, str):
        return conversation
    # Find a system message if present; default to empty system prompt
    system_prompt = ""
    for msg in conversation:
        if msg["role"] == "system":
            system_prompt = msg["content"]
            break
    if lms is None:
        raise ConfigurationError("lmstudio SDK is not available")
    chat = lms.Chat(system_prompt)
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            chat.add_user_message(content)
        elif role == "assistant":
            chat.add_assistant_message(content)
        # system already applied via constructor
    return chat


# Comment: Try to list locally available model ids via SDK; fall back to []
def _list_local_models() -> List[str]:
    if lms is None:
        return []
    # The SDK may expose different listing helpers; try a few common ones defensively
    for attr in ("list_models", "installed_models", "available_models"):
        fn = getattr(lms, attr, None)
        try:
            if callable(fn):
                models = fn()
                if isinstance(models, (list, tuple)):
                    # Accept either list[str] or list[dict{name}]
                    if models and isinstance(models[0], dict) and "name" in models[0]:
                        return [cast(str, m["name"]) for m in models]
                    return [cast(str, m) for m in models]
        except Exception:
            pass
    return []


# Comment: Pick reasonable LM Studio defaults for model types; callers can override in ModelResolver
_LMSTUDIO_DEFAULTS: Dict[str, Optional[str]] = {
    "local_small": "phi-3.5-mini-instruct",
    "local_medium": "openai/gpt-oss-20b",
    "local_large": "openai/gpt-oss-20b",
    "paid_cheap": None,
    "paid_best": None,
}


# Comment: Concrete LM Studio implementation that satisfies core.Llm’s interface
class LlmLmstudio(Llm):
    # Comment: Declare capabilities supported by this provider
    @staticmethod
    def capabilities() -> Set[str]:
        return {"chat", "chat_structured"}

    # Comment: Report supported model ids (best-effort from installed LM Studio models)
    @staticmethod
    def supported_models() -> List[str]:
        return _list_local_models()

    # Comment: Return a sensible default model id for this provider
    @staticmethod
    def default_model() -> str:
        # If any local models found, prefer the medium default when present, else first
        found = _list_local_models()
        preferred = _LMSTUDIO_DEFAULTS.get("local_medium")
        if preferred and preferred in found:
            return preferred
        return found[0] if found else "openai/gpt-oss-20b"

    # Comment: Map a ModelType (passed as its string value) to a default model id, or None if unsupported
    @staticmethod
    def default_for_type(model_type: str) -> Optional[str]:
        return _LMSTUDIO_DEFAULTS.get(model_type)

    # Comment: Provider-level configuration check invoked without an instance
    @staticmethod
    def check_config():
        if lms is None:
            raise ConfigurationError(
                "The 'lmstudio' Python SDK is not importable. Install it and ensure LM Studio is running."
            )

    # Comment: Create with a specific model name resolved by ModelResolver in core
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if lms is None:
            raise ConfigurationError(
                "The 'lmstudio' Python SDK is not importable. Install it and ensure LM Studio is running."
            )

        # Comment: If an instance is already loaded, LM Studio will ignore new load config.
        # So we check what's loaded, verify context, and reload if needed.
        try:
            # Comment: Try to get an existing handle (loads on demand if nothing loaded)
            existing = lms.llm(self.model)
            current_ctx = existing.get_context_length()
        except Exception:
            existing = None
            current_ctx = None

        # Comment: If context matches, use it. Otherwise, unload and load a fresh instance with config.
        if existing is not None and current_ctx == _DESIRED_CONTEXT_LENGTH:
            self._model = existing
        else:
            try:
                if existing is not None:
                    existing.unload()
            except Exception:
                pass

            # Comment: Load a new instance so that load-time config is applied
            client = lms.get_default_client()
            self._model = client.llm.load_new_instance(
                self.model,
                config={"contextLength": _DESIRED_CONTEXT_LENGTH},
            )

        # Optional sanity check; cheap and catches silent 4k zombies
        loaded_ctx = self._model.get_context_length()
        if loaded_ctx != _DESIRED_CONTEXT_LENGTH:
            raise ConfigurationError(
                f"LM Studio loaded with contextLength={loaded_ctx}, expected {_DESIRED_CONTEXT_LENGTH}. "
                "Ensure the model supports this window and that no other instance is pinned in memory."
            )


    # Comment: Plain chat interface; force_json is accepted for parity but not enforced here
    def chat(self, conversation: Conversation, force_json: bool = False) -> str:
        try:
            chat_or_text = _to_lms_chat(conversation)
            result = self._model.respond(chat_or_text, config=_mk_config())
            return cast(str, result.content)
        except ConfigurationError:
            raise
        except Exception as e:
            raise DazLlmError(f"LM Studio chat failed: {e}")

    # Comment: Structured chat returning a validated Pydantic instance
    def chat_structured(
        self, conversation: Conversation, schema: BaseModel, context_size: int = 0
    ) -> BaseModel:
        try:
            chat_or_text = _to_lms_chat(conversation)
            # Accept either a Pydantic class or an instance; SDK wants the class
            schema_type = schema if isinstance(schema, type) else schema.__class__
            result = self._model.respond(
                chat_or_text, response_format=schema_type, config=_mk_config()
            )
            parsed = cast(dict, result.parsed)
            return schema_type.model_validate(parsed)  # type: ignore[attr-defined]
        except ConfigurationError:
            raise
        except Exception as e:
            raise DazLlmError(f"LM Studio structured chat failed: {e}")

    # Comment: Images are not supported via the LM Studio text SDK path
    def image(self, prompt: str, file_name: str, width: int = 1024, height: int = 1024) -> str:
        raise DazLlmError("image generation is not supported by the LM Studio provider")
    
    # Comment: Get the context length for the currently loaded model
    def get_context_length(self) -> int:
        """Get the context length for the current LM Studio model"""
        try:
            # LM Studio SDK provides direct access to context length
            if hasattr(self, '_model') and self._model:
                return self._model.get_context_length()
        except Exception:
            pass
        
        # Fallback to the desired context length we configured
        return _DESIRED_CONTEXT_LENGTH


__all__ = ["LlmLmstudio"]
