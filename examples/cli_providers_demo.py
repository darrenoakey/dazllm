#!/usr/bin/env python3
"""
Demo script showcasing the three new CLI-based LLM providers:
- Codex CLI (codex-cli)
- Claude CLI (claude-cli) 
- Gemini CLI (gemini-cli)

All providers support both regular chat and structured output via Pydantic models.
"""

from pydantic import BaseModel
from typing import List, Optional
from dazllm import Llm


# Example Pydantic models for structured output
class PersonInfo(BaseModel):
    """Person information structure"""
    name: str
    age: int
    occupation: str
    city: str
    interests: List[str]


class CodeAnalysis(BaseModel):
    """Code analysis result"""
    language: str
    complexity_score: int  # 1-10 scale
    has_bugs: bool
    suggestions: List[str]
    estimated_time_hours: float


class CreativeIdea(BaseModel):
    """Creative writing idea"""
    title: str
    genre: str
    setting: str
    main_character: str
    conflict: str
    estimated_length_pages: int


def demo_basic_chat():
    """Demonstrate basic chat functionality with all CLI providers"""
    print("=" * 60)
    print("BASIC CHAT DEMO")
    print("=" * 60)
    
    providers_to_test = [
        ("codex-cli:default", "Codex CLI"),
        ("claude-cli:claude-3-5-sonnet-20241022", "Claude CLI"),
        ("gemini-cli:gemini-2.0-flash-exp", "Gemini CLI")
    ]
    
    question = "What is the difference between machine learning and artificial intelligence? Answer in 2 sentences."
    
    for model_name, display_name in providers_to_test:
        print(f"\n{display_name} ({model_name}):")
        print("-" * 50)
        
        try:
            llm = Llm.model_named(model_name)
            response = llm.chat(question)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure the CLI tool is installed and accessible")


def demo_structured_output():
    """Demonstrate structured output with Pydantic models"""
    print("\n" + "=" * 60)
    print("STRUCTURED OUTPUT DEMO")
    print("=" * 60)
    
    providers_to_test = [
        ("codex-cli:default", "Codex CLI"),
        ("claude-cli:claude-3-5-sonnet-20241022", "Claude CLI"), 
        ("gemini-cli:gemini-2.0-flash-exp", "Gemini CLI")
    ]
    
    for model_name, display_name in providers_to_test:
        print(f"\n{display_name} - Person Info Generation:")
        print("-" * 50)
        
        try:
            llm = Llm.model_named(model_name)
            
            prompt = "Generate information for a fictional software engineer named Alex who lives in San Francisco"
            result = llm.chat_structured(prompt, PersonInfo)
            
            print(f"Name: {result.name}")
            print(f"Age: {result.age}")
            print(f"Occupation: {result.occupation}")
            print(f"City: {result.city}")
            print(f"Interests: {', '.join(result.interests)}")
            
        except Exception as e:
            print(f"Error: {e}")


def demo_complex_structured_output():
    """Demonstrate complex structured output with nested data"""
    print("\n" + "=" * 60)
    print("COMPLEX STRUCTURED OUTPUT DEMO")
    print("=" * 60)
    
    # Test with Claude CLI for code analysis
    print("\nClaude CLI - Code Analysis:")
    print("-" * 50)
    
    try:
        llm = Llm("claude-cli:claude-3-5-sonnet-20241022")
        
        code_prompt = """
        Analyze this Python function:
        
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        """
        
        analysis = llm.chat_structured(code_prompt, CodeAnalysis)
        
        print(f"Language: {analysis.language}")
        print(f"Complexity Score: {analysis.complexity_score}/10")
        print(f"Has Bugs: {analysis.has_bugs}")
        print(f"Suggestions: {', '.join(analysis.suggestions)}")
        print(f"Estimated Fix Time: {analysis.estimated_time_hours} hours")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with Gemini CLI for creative writing
    print("\nGemini CLI - Creative Idea Generation:")
    print("-" * 50)
    
    try:
        llm = Llm("gemini-cli:gemini-2.0-flash-exp")
        
        creative_prompt = "Generate a creative writing idea for a sci-fi short story about time travel"
        
        idea = llm.chat_structured(creative_prompt, CreativeIdea)
        
        print(f"Title: {idea.title}")
        print(f"Genre: {idea.genre}")
        print(f"Setting: {idea.setting}")
        print(f"Main Character: {idea.main_character}")
        print(f"Conflict: {idea.conflict}")
        print(f"Estimated Length: {idea.estimated_length_pages} pages")
        
    except Exception as e:
        print(f"Error: {e}")


def demo_using_aliases():
    """Demonstrate using provider aliases"""
    print("\n" + "=" * 60)
    print("PROVIDER ALIASES DEMO")
    print("=" * 60)
    
    aliases_to_test = [
        ("codex", "Codex"),
        ("claudecli", "Claude CLI"),
        ("geminicli", "Gemini CLI")
    ]
    
    for alias, display_name in aliases_to_test:
        print(f"\n{display_name} (using alias '{alias}'):")
        print("-" * 50)
        
        try:
            # Using the alias with Llm constructor
            llm = Llm(provider=alias)
            
            question = "What is Python? Answer in one sentence."
            response = llm.chat(question)
            print(f"Response: {response}")
            print(f"Actual provider: {llm.provider}")
            
        except Exception as e:
            print(f"Error: {e}")


def demo_conversation_history():
    """Demonstrate conversation history handling"""
    print("\n" + "=" * 60)
    print("CONVERSATION HISTORY DEMO")
    print("=" * 60)
    
    print("\nCodex CLI - Multi-turn conversation:")
    print("-" * 50)
    
    try:
        llm = Llm("codex-cli:default")
        
        # Create a conversation with history
        conversation = [
            {"role": "system", "content": "You are a helpful programming assistant."},
            {"role": "user", "content": "What is a list in Python?"},
            {"role": "assistant", "content": "A list in Python is an ordered collection of items that can store multiple values."},
            {"role": "user", "content": "How do I add an item to it?"}
        ]
        
        response = llm.chat(conversation)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    """Run all demo functions"""
    print("CLI LLM Providers Demo")
    print("This demo showcases the three new CLI-based providers:")
    print("- codex-cli (uses 'codex' command)")
    print("- claude-cli (uses 'claude' command)")  
    print("- gemini-cli (uses 'gemini' command)")
    print("\nAll providers are headless and support both chat() and chat_structured() methods.")
    
    # Run demos (comment out any that fail due to missing CLI tools)
    try:
        demo_basic_chat()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return
    
    try:
        demo_structured_output()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return
    
    try:
        demo_complex_structured_output()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return
        
    try:
        demo_using_aliases()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return
        
    try:
        demo_conversation_history()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        return
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Features Implemented:")
    print("✓ Full dazllm framework integration")
    print("✓ Both chat() and chat_structured() support")
    print("✓ Headless operation (no user prompts)")
    print("✓ Provider aliases (codex, claudecli, geminicli)")
    print("✓ Robust JSON parsing with fallbacks")
    print("✓ Conversation history handling")
    print("✓ Error handling and retries")
    print("✓ Pydantic model validation")
    print("\nTo use in your code:")
    print("  from dazllm import Llm")
    print("  llm = Llm('codex-cli:default')")
    print("  response = llm.chat('Your question here')")


if __name__ == "__main__":
    main()