#!/usr/bin/env python3
"""
Demo script to test get_context_length() functionality across all providers
"""

from dazllm.core import Llm

def test_context_lengths():
    """Test context length detection for all providers"""
    
    print("Testing context length detection across all providers...")
    print("="*60)
    
    # Test models for each provider
    test_models = [
        ("openai:gpt-4o", "OpenAI GPT-4o"),
        ("openai:gpt-image-1", "OpenAI gpt-image-1"),
        ("anthropic:claude-3-5-sonnet-20241022", "Anthropic Claude 3.5 Sonnet"),
        ("google:gemini-2.0-flash", "Google Gemini 2.0 Flash"),
        ("ollama:mistral-small", "Ollama Mistral Small"),
        ("lm-studio:default", "LM Studio"),
        ("codex-cli:default", "Codex CLI"),
        ("claude-cli:claude-3-5-sonnet-20241022", "Claude CLI"),
        ("gemini-cli:gemini-2.0-flash-exp", "Gemini CLI"),
    ]
    
    results = []
    
    for model_name, display_name in test_models:
        try:
            llm = Llm.model_named(model_name)
            context_length = llm.get_context_length()
            
            print(f"✅ {display_name:30} -> {context_length:,} tokens")
            results.append((display_name, context_length, "✅"))
            
        except Exception as e:
            print(f"❌ {display_name:30} -> Error: {e}")
            results.append((display_name, 0, f"❌ {e}"))
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    for display_name, context_length, status in results:
        if status == "✅":
            print(f"{display_name:30} {context_length:>10,} tokens")
        else:
            print(f"{display_name:30} {status}")
    
    print("="*60)

def test_image_capabilities():
    """Test image generation capability detection"""
    
    print("\nTesting image generation capabilities...")
    print("="*60)
    
    providers = Llm.get_providers()
    
    for provider in providers:
        try:
            info = Llm.get_provider_info(provider)
            capabilities = info.get('capabilities', set())
            configured = info.get('configured', False)
            
            has_image = 'image' in capabilities
            status = "✅ SUPPORTS" if has_image else "❌ No support"
            config_status = "CONFIGURED" if configured else "not configured"
            
            print(f"{provider:15} -> {status:15} ({config_status})")
            
        except Exception as e:
            print(f"{provider:15} -> ❌ Error: {e}")

if __name__ == "__main__":
    print("dazllm Context Length & Image Capabilities Test")
    print("="*60)
    
    test_context_lengths()
    test_image_capabilities()
    
    print("\n🎉 Testing completed!")