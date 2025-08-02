# DazLLM Usage Guide

## Installation

```bash
pip install dazllm
```

## Quick Setup

Before using dazllm, configure your API keys using the keyring:

```bash
# Required for specific providers
keyring set dazllm openai_api_key YOUR_OPENAI_KEY
keyring set dazllm anthropic_api_key YOUR_ANTHROPIC_KEY  
keyring set dazllm google_api_key YOUR_GOOGLE_KEY

# Optional: Set a default model
keyring set dazllm default_model openai:gpt-4o

# Check configuration
dazllm --check
```

## Core Concepts

### Model Identification

Models are identified using the format `provider:model`:

**OpenAI Models:**
- `openai:gpt-4o` - Latest GPT-4 Omni
- `openai:gpt-4o-mini` - Smaller, faster GPT-4
- `openai:gpt-4-turbo` - GPT-4 Turbo
- `openai:gpt-3.5-turbo` - GPT-3.5 Turbo

**Anthropic Models:**
- `anthropic:claude-3-5-sonnet-20241022` - Latest Claude 3.5 Sonnet
- `anthropic:claude-3-opus-20240229` - Claude 3 Opus (most capable)
- `anthropic:claude-3-haiku-20240307` - Claude 3 Haiku (fastest)

**Google Models:**
- `google:gemini-1.5-pro` - Gemini 1.5 Pro
- `google:gemini-1.5-flash` - Gemini 1.5 Flash (faster)

**Local Models (via Ollama):**
- `ollama:llama3.1:8b` - Llama 3.1 8B
- `ollama:llama3.1:70b` - Llama 3.1 70B
- `ollama:codellama:13b` - Code Llama 13B

**Local Models (via LM Studio):**
- `lmstudio:model-name` - Any model loaded in LM Studio

### Model Types

Instead of specifying exact models, you can use model types:

- `ModelType.LOCAL_SMALL` - ~1B parameter models (fast, basic)
- `ModelType.LOCAL_MEDIUM` - ~7B parameter models (good balance)  
- `ModelType.LOCAL_LARGE` - ~14B parameter models (best local quality)
- `ModelType.PAID_CHEAP` - Cost-effective cloud models
- `ModelType.PAID_BEST` - Highest quality cloud models

## Basic Usage

### Import and Basic Chat

```python
from dazllm import Llm, ModelType

# Simple chat with specific model
response = Llm.chat("What's the capital of France?", model="openai:gpt-4o")
print(response)  # "The capital of France is Paris."

# Chat using model type
response = Llm.chat("Tell me a joke", model_type=ModelType.PAID_CHEAP)
print(response)

# Use default model (from keyring or auto-detected)
response = Llm.chat("Hello!")
print(response)
```

### Instance-Based Usage

```python
# Create a model instance for reuse
llm = Llm.model_named("anthropic:claude-3-5-sonnet-20241022")
response1 = llm.chat("Hello!")
response2 = llm.chat("How are you?")

# Alternative constructor
llm = Llm("openai:gpt-4o")
response = llm.chat("What's 2+2?")
```

## Core Functions

### `Llm.chat(conversation, model=None, model_type=None, force_json=False)`

**Purpose:** Basic text chat with an LLM

**Parameters:**
- `conversation` (str or List[Message]): The conversation to send
- `model` (str, optional): Specific model like "openai:gpt-4o"
- `model_type` (ModelType, optional): Model type instead of specific model
- `force_json` (bool): Force JSON-formatted response

**Returns:** `str` - The LLM's response

**Examples:**
```python
# Simple string prompt
response = Llm.chat("Explain quantum physics")

# Multi-turn conversation
conversation = [
    {"role": "system", "content": "You are a helpful math tutor"},
    {"role": "user", "content": "What's 15 * 7?"},
    {"role": "assistant", "content": "15 * 7 = 105"},
    {"role": "user", "content": "How did you calculate that?"}
]
response = Llm.chat(conversation, model="openai:gpt-4o")

# Force JSON response
response = Llm.chat("List 3 colors", force_json=True)
```

### `Llm.chat_structured(conversation, schema, model=None, model_type=None, context_size=0)`

**Purpose:** Get structured output using Pydantic schemas

**Parameters:**
- `conversation` (str or List[Message]): The conversation
- `schema` (Type[BaseModel]): Pydantic model class defining the structure
- `model` (str, optional): Specific model
- `model_type` (ModelType, optional): Model type
- `context_size` (int): Context window size hint

**Returns:** Instance of the provided Pydantic model

**Examples:**
```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

class People(BaseModel):
    people: list[Person]

# Extract structured data
text = "John Smith is 30 and works as a teacher. Jane Doe is 25 and is a doctor."
result = Llm.chat_structured(
    f"Extract person info from: {text}",
    People,
    model="openai:gpt-4o-mini"
)

for person in result.people:
    print(f"{person.name}, {person.age}, {person.occupation}")
```

### `Llm.image(prompt, file_name, width=1024, height=1024, model=None, model_type=None)`

**Purpose:** Generate images (only supported by some providers like OpenAI)

**Parameters:**
- `prompt` (str): Description of the image to generate
- `file_name` (str): Path where to save the generated image
- `width` (int): Image width in pixels (default: 1024)
- `height` (int): Image height in pixels (default: 1024)
- `model` (str, optional): Specific model
- `model_type` (ModelType, optional): Model type

**Returns:** `str` - Path to the saved image file

**Examples:**
```python
# Generate an image
image_path = Llm.image(
    "a red cat sitting on a blue chair",
    "cat.png",
    width=512,
    height=512,
    model="openai:dall-e-3"
)
print(f"Image saved to: {image_path}")
```

## Introspection Functions

### `Llm.get_providers()`

**Purpose:** Get list of all available providers

**Returns:** `List[str]` - List of provider names

```python
providers = Llm.get_providers()
print(providers)  # ['openai', 'anthropic', 'google', 'ollama', 'lmstudio']
```

### `Llm.get_provider_info(provider)`

**Purpose:** Get detailed information about a specific provider

**Parameters:**
- `provider` (str): Provider name

**Returns:** `Dict` with provider information

```python
info = Llm.get_provider_info("openai")
print(info["configured"])  # True/False
print(info["capabilities"])  # {'chat', 'structured', 'image'}
print(info["supported_models"])  # List of models
print(info["default_model"])  # Default model name
```

### `Llm.get_all_providers_info()`

**Purpose:** Get information about all providers

**Returns:** `Dict[str, Dict]` - Information for all providers

```python
all_info = Llm.get_all_providers_info()
for provider, info in all_info.items():
    print(f"{provider}: {info['configured']}")
```

### `check_configuration()`

**Purpose:** Check configuration status of all providers

**Returns:** `Dict[str, Dict]` - Configuration status for each provider

```python
from dazllm import check_configuration

status = check_configuration()
for provider, config in status.items():
    if config["configured"]:
        print(f"✓ {provider}: Ready")
    else:
        print(f"✗ {provider}: {config['error']}")
```

## Constructing a Large Local Model

For large local models, you have several options:

### Option 1: Use Ollama

```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull a large model
ollama pull llama3.1:70b
```

```python
# Use the large model
response = Llm.chat(
    "Explain the theory of relativity", 
    model="ollama:llama3.1:70b"
)

# Or use model type
response = Llm.chat(
    "Write a complex analysis", 
    model_type=ModelType.LOCAL_LARGE
)
```

### Option 2: Use LM Studio

```bash
# Download and install LM Studio
# Load a model in the LM Studio interface
# Start the local server
```

```python
# Configure LM Studio URL (if not default)
# keyring set dazllm lmstudio_url http://localhost:1234

# Use LM Studio model
response = Llm.chat(
    "Complex reasoning task",
    model="lmstudio:your-loaded-model"
)
```

### Option 3: Model Type Selection

```python
# Let dazllm choose the best available large local model
large_llm = Llm.model_named("ollama:llama3.1:70b")  # If available
response = large_llm.chat("Complex analysis task")

# Or use model type to automatically select
response = Llm.chat(
    "Difficult reasoning problem",
    model_type=ModelType.LOCAL_LARGE
)
```

## Command Line Usage

DazLLM also provides a powerful CLI:

```bash
# Basic chat
dazllm chat "What's the weather like?"

# Specify model
dazllm chat "Explain AI" --model anthropic:claude-3-5-sonnet-20241022

# Use model type
dazllm chat "Tell me a story" --model-type paid_best

# Structured output
dazllm structured "List 3 colors" --schema '{"type":"array","items":{"type":"string"}}'

# Generate image
dazllm image "a sunset over mountains" sunset.png --width 1024 --height 1024

# Check configuration
dazllm --check

# List all models
dazllm models
```

## Provider Capabilities

| Provider | Chat | Structured | Image | Local |
|----------|------|------------|-------|-------|
| OpenAI | ✓ | ✓ | ✓ | ✗ |
| Anthropic | ✓ | ✓ | ✗ | ✗ |
| Google | ✓ | ✓ | ✗ | ✗ |
| Ollama | ✓ | ✓ | ✗ | ✓ |
| LM Studio | ✓ | ✓ | ✗ | ✓ |

## Error Handling

```python
from dazllm import DazLlmError, ConfigurationError, ModelNotFoundError

try:
    response = Llm.chat("Hello", model="nonexistent:model")
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except DazLlmError as e:
    print(f"General dazllm error: {e}")
```

## Configuration Management

```bash
# Set API keys
keyring set dazllm openai_api_key sk-...
keyring set dazllm anthropic_api_key sk-ant-...
keyring set dazllm google_api_key AIza...

# Set custom endpoints
keyring set dazllm ollama_url http://localhost:11434
keyring set dazllm lmstudio_url http://localhost:1234

# Set default model
keyring set dazllm default_model openai:gpt-4o

# Check what's configured
dazllm --check
```

This provides a complete interface for using any LLM through a single, consistent API!