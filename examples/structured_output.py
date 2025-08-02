#!/usr/bin/env python3
"""
Structured output example for dazllm
"""

from dazllm import Llm
from pydantic import BaseModel

class Person(BaseModel):
    """Person model for structured output"""
    name: str
    age: int
    occupation: str

class People(BaseModel):
    """Collection of people for structured output"""
    people: list[Person]

class MathResult(BaseModel):
    """Math result with explanation for structured output"""
    question: str
    answer: int
    explanation: str

def main():
    """Main function demonstrating structured output with dazllm"""
    print("Structured output examples")
    
    try:
        # Extract structured data
        text = "John Smith is 30 and works as a teacher. Jane Doe is 25 and is a doctor."
        result = Llm.chat_structured_static(
            f"Extract person info from: {text}",
            People,
            model="openai:gpt-4o-mini"
        )
        
        print("Extracted people:")
        for person in result.people:
            print(f"- {person.name}, {person.age}, {person.occupation}")
        
        # Math with explanation
        math_result = Llm.chat_structured_static(
            "What's 15 * 7?",
            MathResult,
            model_type="paid_cheap"
        )
        
        print(f"\nMath: {math_result.question}")
        print(f"Answer: {math_result.answer}")
        print(f"Explanation: {math_result.explanation}")
        
    except (ImportError, AttributeError, ValueError) as e:
        print(f"Error: {e}")
        print("Make sure you have configured your API keys")

if __name__ == "__main__":
    main()


# Unit tests
import unittest


class TestStructuredOutputExample(unittest.TestCase):
    """Test cases for structured output example"""

    def test_main_function_exists(self):
        """Test that main function exists and is callable"""
        self.assertTrue(callable(main))

    def test_pydantic_models_defined(self):
        """Test that Pydantic models are properly defined"""
        # Test Person model
        person = Person(name="Test", age=25, occupation="Developer")
        self.assertEqual(person.name, "Test")
        self.assertEqual(person.age, 25)
        self.assertEqual(person.occupation, "Developer")
        
        # Test People model
        people = People(people=[person])
        self.assertEqual(len(people.people), 1)
        self.assertEqual(people.people[0].name, "Test")
        
        # Test MathResult model
        math_result = MathResult(question="2+2", answer=4, explanation="Basic addition")
        self.assertEqual(math_result.question, "2+2")
        self.assertEqual(math_result.answer, 4)
        self.assertEqual(math_result.explanation, "Basic addition")

    def test_llm_has_structured_method(self):
        """Test that Llm class has structured output method"""
        self.assertTrue(hasattr(Llm, 'chat_structured_static'))

    def test_model_fields(self):
        """Test that models have correct field types"""
        # Test Person fields
        person_fields = Person.model_fields
        self.assertIn('name', person_fields)
        self.assertIn('age', person_fields)
        self.assertIn('occupation', person_fields)
        
        # Test MathResult fields
        math_fields = MathResult.model_fields
        self.assertIn('question', math_fields)
        self.assertIn('answer', math_fields)
        self.assertIn('explanation', math_fields)

    def test_model_validation(self):
        """Test that models validate input correctly"""
        # Valid Person
        person = Person(name="John", age=30, occupation="Teacher")
        self.assertEqual(person.name, "John")
        
        # Test type validation
        with self.assertRaises(ValueError):
            Person(name="John", age="thirty", occupation="Teacher")  # Invalid age type

    def test_example_data(self):
        """Test example data formats"""
        text = "John Smith is 30 and works as a teacher. Jane Doe is 25 and is a doctor."
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)
        
        # Test math question
        question = "What's 15 * 7?"
        self.assertIsInstance(question, str)
        self.assertIn("15", question)
        self.assertIn("7", question)
