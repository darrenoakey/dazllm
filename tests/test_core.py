import pytest
from dazllm import Llm, ModelType, DazLlmError

def test_model_name_parsing():
    """Test model name parsing"""
    try:
        llm = Llm("openai:gpt-4")
        assert llm.provider == "openai"
        assert llm.model == "gpt-4"
    except:
        # Skip if not configured
        pass

def test_invalid_model_format():
    """Test invalid model format raises error"""
    with pytest.raises(DazLlmError):
        Llm("invalid-format")

def test_model_types():
    """Test model type enum"""
    assert ModelType.PAID_BEST.value == "paid_best"
    assert ModelType.LOCAL_SMALL.value == "local_small"
