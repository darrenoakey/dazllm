#!/usr/bin/env python3
"""
Structured output example for dazllm
"""

from dazllm import Llm
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

class People(BaseModel):
    people: list[Person]

class MathResult(BaseModel):
    question: str
    answer: int
    explanation: str

def main():
    print("Structured output examples")
    
    try:
        # Extract structured data
        text = "John Smith is 30 and works as a teacher. Jane Doe is 25 and is a doctor."
        result = Llm.chat_structured(
            f"Extract person info from: {text}",
            People,
            model="openai:gpt-4o-mini"
        )
        
        print("Extracted people:")
        for person in result.people:
            print(f"- {person.name}, {person.age}, {person.occupation}")
        
        # Math with explanation
        math_result = Llm.chat_structured(
            "What's 15 * 7?",
            MathResult,
            model_type="paid_cheap"
        )
        
        print(f"\nMath: {math_result.question}")
        print(f"Answer: {math_result.answer}")
        print(f"Explanation: {math_result.explanation}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys")

if __name__ == "__main__":
    main()
