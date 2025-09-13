"""
Example of using Ollama with structured output for models without format support
"""

from pydantic import BaseModel
from typing import List, Optional
from dazllm import Llm


# Define some Pydantic models for structured output
class MovieReview(BaseModel):
    """Movie review structure"""
    title: str
    rating: float  # 0-10
    genre: str
    summary: str
    recommended: bool


class RecipeIngredient(BaseModel):
    """Recipe ingredient"""
    name: str
    amount: str
    unit: str


class Recipe(BaseModel):
    """Complete recipe structure"""
    name: str
    cuisine: str
    prep_time_minutes: int
    cook_time_minutes: int
    servings: int
    ingredients: List[RecipeIngredient]
    instructions: List[str]


class WeatherReport(BaseModel):
    """Weather report structure"""
    location: str
    temperature_celsius: float
    condition: str
    humidity_percent: int
    wind_speed_kmh: float
    forecast: str


def example_basic_structured_output():
    """Example using basic structured output"""
    print("Example 1: Movie Review")
    print("-" * 40)
    
    # Create LLM instance for gpt-oss:20b (doesn't support format)
    llm = Llm("ollama:gpt-oss:20b")
    
    # Get structured movie review
    review = llm.chat_structured(
        "Review the movie 'The Matrix' from 1999",
        MovieReview
    )
    
    print(f"Title: {review.title}")
    print(f"Rating: {review.rating}/10")
    print(f"Genre: {review.genre}")
    print(f"Summary: {review.summary}")
    print(f"Recommended: {'Yes' if review.recommended else 'No'}")
    print()


def example_nested_structured_output():
    """Example with nested structured output"""
    print("Example 2: Recipe Generation")
    print("-" * 40)
    
    llm = Llm("ollama:gpt-oss:20b")
    
    # Generate a complex nested structure
    recipe = llm.chat_structured(
        "Create a recipe for chocolate chip cookies",
        Recipe
    )
    
    print(f"Recipe: {recipe.name}")
    print(f"Cuisine: {recipe.cuisine}")
    print(f"Prep Time: {recipe.prep_time_minutes} minutes")
    print(f"Cook Time: {recipe.cook_time_minutes} minutes")
    print(f"Servings: {recipe.servings}")
    print("\nIngredients:")
    for ing in recipe.ingredients:
        print(f"  - {ing.amount} {ing.unit} {ing.name}")
    print("\nInstructions:")
    for i, instruction in enumerate(recipe.instructions, 1):
        print(f"  {i}. {instruction}")
    print()


def example_with_conversation():
    """Example using conversation history"""
    print("Example 3: Weather Report with Context")
    print("-" * 40)
    
    llm = Llm("ollama:gpt-oss:20b")
    
    # Create a conversation
    conversation = [
        {"role": "system", "content": "You are a weather reporter."},
        {"role": "user", "content": "What's the weather like in San Francisco today? Make it sound realistic but you can make up the specific numbers."}
    ]
    
    weather = llm.chat_structured(
        conversation,
        WeatherReport
    )
    
    print(f"Location: {weather.location}")
    print(f"Temperature: {weather.temperature_celsius}°C")
    print(f"Condition: {weather.condition}")
    print(f"Humidity: {weather.humidity_percent}%")
    print(f"Wind Speed: {weather.wind_speed_kmh} km/h")
    print(f"Forecast: {weather.forecast}")
    print()


def example_compare_models():
    """Compare output between models with and without format support"""
    print("Example 4: Comparing Models")
    print("-" * 40)
    
    prompt = "Generate a movie review for 'Inception'"
    
    # Model without format support
    print("Using gpt-oss:20b (no format support):")
    llm1 = Llm("ollama:gpt-oss:20b")
    review1 = llm1.chat_structured(prompt, MovieReview)
    print(f"  Title: {review1.title}, Rating: {review1.rating}")
    
    # Model with format support (if available)
    print("\nUsing mistral-small (with format support):")
    try:
        llm2 = Llm("ollama:mistral-small")
        review2 = llm2.chat_structured(prompt, MovieReview)
        print(f"  Title: {review2.title}, Rating: {review2.rating}")
    except Exception as e:
        print(f"  Error: {e}")
    print()


def example_error_handling():
    """Example showing error handling and retries"""
    print("Example 5: Error Handling")
    print("-" * 40)
    
    llm = Llm("ollama:gpt-oss:20b")
    
    # This should work even if the model initially returns invalid JSON
    # The implementation will retry with clarifications
    try:
        review = llm.chat_structured(
            "Give me a brief movie review",
            MovieReview
        )
        print(f"Successfully got review: {review.title}")
    except Exception as e:
        print(f"Error after retries: {e}")


if __name__ == "__main__":
    print("Ollama Structured Output Examples")
    print("=" * 40)
    print("These examples demonstrate using Ollama with models")
    print("that don't support the 'format' parameter (like gpt-oss:20b)")
    print()
    
    # Run examples (comment out any that fail due to missing models)
    try:
        example_basic_structured_output()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    try:
        example_nested_structured_output()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    try:
        example_with_conversation()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    try:
        example_compare_models()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    try:
        example_error_handling()
    except Exception as e:
        print(f"Example 5 failed: {e}\n")
    
    print("Examples complete!")