#!/usr/bin/env python3
"""
Basic usage example for dazllm
"""

from dazllm import Llm, ModelType

def main():
    print("Basic dazllm usage examples")
    
    try:
        # Simple chat with specific model
        response = Llm.chat("What's 2+2?", model="openai:gpt-4o")
        print(f"Math answer: {response}")
        
        # Chat with model type
        response = Llm.chat("Tell me a joke", model_type=ModelType.PAID_CHEAP)
        print(f"Joke: {response}")
        
        # Instance-based usage
        llm = Llm.model_named("openai:gpt-4o")
        response = llm.chat("Hello!")
        print(f"Greeting: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys:")
        print("keyring set dazllm openai_api_key YOUR_KEY")

if __name__ == "__main__":
    main()
