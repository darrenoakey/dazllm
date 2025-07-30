![](banner.jpg)

# dazllm 🚀

**Simple, unified interface for all major LLMs**

Stop juggling different APIs and libraries. `dazllm` gives you a clean, consistent way to chat with any LLM - from GPT-4 and Claude to local Ollama or LM Studio models.

## Features

✨ **Unified API** - Same interface for OpenAI, Anthropic, Google, and local models (Ollama, LM Studio)
🔧 **Smart Model Selection** - Choose by name, type, or let it auto-select  
🔐 **Secure Configuration** - API keys stored safely in system keyring  
📝 **Structured Output** - Get Pydantic models directly from LLM responses  
🎨 **Image Generation** - Create images with DALL-E and more  
💻 **CLI & Python API** - Use from command line or import in your code  

## Quick Start

### Installation

```bash
pip install dazllm
```

### Setup

Configure your API keys using keyring:

```bash
keyring set dazllm openai_api_key YOUR_OPENAI_KEY
keyring set dazllm anthropic_api_key YOUR_ANTHROPIC_KEY
keyring set dazllm google_api_key YOUR_GOOGLE_KEY
keyring set dazllm ollama_url http://localhost:11434
keyring set dazllm lmstudio_url http://localhost:1234
```

Check everything is working:

```bash
dazllm --check
```

### Usage

#### Command Line

```bash
# Simple chat
dazllm chat "What's the capital of France?"

# Use specific model  
dazllm chat --model openai:gpt-4 "Explain quantum computing"

# Use model type (auto-selects best available)
dazllm chat --model-type paid_best "Write a poem"

# Use provider default
dazllm chat --model openai "Tell me about AI"

# Structured output
dazllm structured "List 3 colors" --schema '{"type":"array","items":{"type":"string"}}'

# Generate images
dazllm image "a red cat wearing a hat" cat.png

# From file
dazllm chat --file prompt.txt --output response.txt
```

#### Python API

```python
from dazllm import Llm, ModelType
from pydantic import BaseModel

# Instance-based usage
llm = Llm("openai:gpt-4")
response = llm.chat("Hello!")

# Static/module-level usage
response = Llm.chat("Hello!", model="anthropic:claude-3-5-sonnet-20241022")
response = Llm.chat("Hello!", model_type=ModelType.PAID_BEST)

# Structured output with Pydantic
class ColorList(BaseModel):
    colors: list[str]

result = Llm.chat_structured("List 3 colors", ColorList)
print(result.colors)  # ['red', 'green', 'blue']

# Image generation
Llm.image("a sunset over mountains", "sunset.png")

# Conversation history
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "What's your name?"}
]
response = Llm.chat(conversation, model="ollama:mistral-small")
```

## Model Types

Instead of remembering model names, use semantic types:

- `local_small` - ~1B parameter models (fast, basic)
- `local_medium` - ~7B parameter models (good balance)  
- `local_large` - ~14B parameter models (best local quality)
- `paid_cheap` - Cost-effective cloud models
- `paid_best` - Highest quality cloud models

## Model Format

All models use the format `provider:model`:

- **OpenAI**: `openai:gpt-4o`, `openai:gpt-4o-mini`, `openai:dall-e-3`
- **Anthropic**: `anthropic:claude-3-5-sonnet-20241022`, `anthropic:claude-3-haiku-20240307`
- **Google**: `google:gemini-pro`, `google:gemini-flash`
- **Ollama**: `ollama:mistral-small`, `ollama:llama3:8b`, `ollama:codellama:7b`
- **LM Studio**: `lm-studio:mistral`, `lm-studio:llama3`

You can also use just the provider name (e.g., `openai`) to use that provider's default model.

## Configuration

API keys are stored securely in your system keyring:

```bash
# Set API keys
keyring set dazllm openai_api_key YOUR_OPENAI_KEY
keyring set dazllm anthropic_api_key YOUR_ANTHROPIC_KEY
keyring set dazllm google_api_key YOUR_GOOGLE_KEY
keyring set dazllm ollama_url http://localhost:11434
keyring set dazllm lmstudio_url http://localhost:1234

# Set default model (optional)
keyring set dazllm default_model openai:gpt-4o

# Check what's configured
dazllm --check
```

## Examples

### Building a Chatbot

```python
from dazllm import Llm

def chatbot():
    llm = Llm.model_named("openai:gpt-4o")
    conversation = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        conversation.append({"role": "user", "content": user_input})
        response = llm.chat(conversation)
        conversation.append({"role": "assistant", "content": response})
        
        print(f"AI: {response}")

chatbot()
```

### Data Extraction

```python
from dazllm import Llm
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    city: str

class People(BaseModel):
    people: list[Person]

text = "John Doe, age 30, lives in New York. Jane Smith, age 25, lives in LA."

result = Llm.chat_structured(
    f"Extract people info from: {text}",
    People,
    model="openai:gpt-4o-mini"
)

for person in result.people:
    print(f"{person.name} is {person.age} years old and lives in {person.city}")
```

### Image Generation Pipeline

```python
from dazllm import Llm

# Generate image description
description = Llm.chat(
    "Describe a serene mountain landscape in detail",
    model_type="paid_cheap"
)

# Generate the image
image_path = Llm.image(description, "mountain.png", width=1024, height=768)
print(f"Image saved to {image_path}")
```

## Requirements

- Python 3.8+
- API keys for desired providers (OpenAI, Anthropic, Google)
- Ollama or LM Studio installed for local models

## License

MIT License

## Contributing

Contributions welcome! Please see the GitHub repository for guidelines.

---

**dazllm** - Making LLMs accessible to everyone! 🚀