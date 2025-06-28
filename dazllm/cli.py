#!/usr/bin/env python3
"""
dazllm command line interface with colorama support
"""

import argparse
import sys
import json
import os
from typing import Optional, Type
from pydantic import BaseModel

# Colorama for colored output
from colorama import init, Fore, Back, Style

init(autoreset=True)

from .core import Llm, ModelType, check_configuration
from .core import DazLlmError, ConfigurationError, ModelNotFoundError


def success(msg: str):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")


def error(msg: str):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {msg}{Style.RESET_ALL}", file=sys.stderr)


def warning(msg: str):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")


def info(msg: str):
    """Print info message in blue"""
    print(f"{Fore.BLUE}ℹ {msg}{Style.RESET_ALL}")


def header(msg: str):
    """Print header message in bold cyan"""
    print(f"{Fore.CYAN}{Style.BRIGHT}{msg}{Style.RESET_ALL}")


def cmd_chat(args):
    """Handle chat command"""
    try:
        # Build conversation from arguments
        if args.file:
            with open(args.file, "r") as f:
                conversation = f.read().strip()
        else:
            conversation = args.prompt

        if not conversation:
            error("No prompt provided. Use --prompt or --file")
            return 1

        # Make the call
        response = Llm.chat(
            conversation,
            model=args.model,
            model_type=ModelType(args.model_type) if args.model_type else None,
            force_json=args.json,
        )

        # Output
        if args.output:
            with open(args.output, "w") as f:
                f.write(response)
            success(f"Response saved to {args.output}")
        else:
            print(response)

        return 0

    except (DazLlmError, FileNotFoundError) as e:
        error(str(e))
        return 1


def cmd_structured(args):
    """Handle structured chat command"""
    try:
        # Load schema class dynamically or create simple schema
        if args.schema_class:
            # Import schema class dynamically
            module_name, class_name = args.schema_class.rsplit(".", 1)
            import importlib

            module = importlib.import_module(module_name)
            schema_class = getattr(module, class_name)
        else:
            # Create simple schema from JSON
            if args.schema_file:
                with open(args.schema_file, "r") as f:
                    schema_dict = json.load(f)
            else:
                try:
                    schema_dict = json.loads(args.schema)
                except json.JSONDecodeError:
                    error("Invalid JSON schema")
                    return 1

            # Create dynamic Pydantic model
            schema_class = create_dynamic_model(schema_dict)

        # Build conversation
        if args.file:
            with open(args.file, "r") as f:
                conversation = f.read().strip()
        else:
            conversation = args.prompt

        if not conversation:
            error("No prompt provided. Use --prompt or --file")
            return 1

        # Make the call
        try:
            response = Llm.chat_structured(
                conversation,
                schema_class,
                model=args.model,
                model_type=ModelType(args.model_type) if args.model_type else None,
                context_size=args.context_size,
            )
        except Exception as e:
            error(f"LLM call failed: {e}")
            return 1

        # Output - handle different model types
        if hasattr(response, "data"):
            # Array model - output the data directly
            output_json = json.dumps(response.data, indent=2)
        else:
            # Regular model - output the full model
            output_json = response.model_dump_json(indent=2)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            success(f"Response saved to {args.output}")
        else:
            print(output_json)

        return 0

    except (
        DazLlmError,
        FileNotFoundError,
        json.JSONDecodeError,
        ImportError,
        AttributeError,
    ) as e:
        error(str(e))
        return 1


def create_dynamic_model(schema_dict: dict) -> Type[BaseModel]:
    """Create a dynamic Pydantic model from JSON schema"""
    from pydantic import create_model

    # Handle array schemas at root level
    if schema_dict.get("type") == "array":
        # For array schemas, create a wrapper model with a 'data' field
        items_type = list
        if "items" in schema_dict:
            item_schema = schema_dict["items"]
            if item_schema.get("type") == "string":
                items_type = list[str]
            elif item_schema.get("type") == "integer":
                items_type = list[int]
            elif item_schema.get("type") == "number":
                items_type = list[float]
            elif item_schema.get("type") == "boolean":
                items_type = list[bool]
            else:
                items_type = list

        return create_model("ArrayModel", data=(items_type, ...))

    # Handle object schemas
    fields = {}
    properties = schema_dict.get("properties", {})

    # If no properties, create a simple model with a 'result' field
    if not properties:
        return create_model("SimpleModel", result=(str, ...))

    for field_name, field_schema in properties.items():
        field_type = str  # Default to string
        if field_schema.get("type") == "integer":
            field_type = int
        elif field_schema.get("type") == "number":
            field_type = float
        elif field_schema.get("type") == "boolean":
            field_type = bool
        elif field_schema.get("type") == "array":
            field_type = list
        elif field_schema.get("type") == "object":
            field_type = dict

        fields[field_name] = (field_type, ...)

    return create_model("DynamicModel", **fields)


def cmd_image(args):
    """Handle image generation command"""
    try:
        # Generate image
        result_path = Llm.image(
            args.prompt,
            args.output,
            width=args.width,
            height=args.height,
            model=args.model,
            model_type=ModelType(args.model_type) if args.model_type else None,
        )

        success(f"Image saved to {result_path}")
        return 0

    except DazLlmError as e:
        error(str(e))
        return 1


def cmd_check(args):
    """Check configuration status"""
    header("Checking dazllm configuration...\n")

    status = check_configuration()

    all_good = True
    for provider, provider_status in status.items():
        if provider_status["configured"]:
            success(f"{provider.upper()}: Configured")
        else:
            error(f"{provider.upper()}: {provider_status['error']}")
            all_good = False

    print()

    if all_good:
        success("All providers are configured!")
        info("To set a default model, add it to keyring:")
        print("  keyring set dazllm default_model openai:gpt-4o")
    else:
        warning("Some providers need configuration.")
        info("Configure providers using keyring:")
        print("  keyring set dazllm openai_api_key YOUR_KEY")
        print("  keyring set dazllm anthropic_api_key YOUR_KEY")
        print("  keyring set dazllm google_api_key YOUR_KEY")
        print("  keyring set dazllm ollama_url http://localhost:11434")

    return 0 if all_good else 1


def cmd_models(args):
    """List all available models from all providers"""
    header("dazllm Available Models\n")

    try:
        # Get all provider information
        all_providers_info = Llm.get_all_providers_info()

        for provider_name, provider_info in all_providers_info.items():
            print(f"{Fore.CYAN}{provider_name.upper()}:{Style.RESET_ALL}")

            if not provider_info["configured"]:
                print(f"  {Fore.YELLOW}Not configured{Style.RESET_ALL}")
                print()
                continue

            # Show capabilities
            capabilities = ", ".join(provider_info["capabilities"])
            print(f"  {Fore.GREEN}Capabilities:{Style.RESET_ALL} {capabilities}")
            print(
                f"  {Fore.GREEN}Default model:{Style.RESET_ALL} {provider_info['default_model']}"
            )

            # Show models
            models = provider_info["supported_models"]
            if models:
                print(
                    f"  {Fore.GREEN}Available models ({len(models)}):{Style.RESET_ALL}"
                )
                for model in sorted(models):
                    print(f"    {provider_name}:{model}")
            else:
                print(f"  {Fore.YELLOW}No models available{Style.RESET_ALL}")

            print()

        # Show model type information
        print(f"{Fore.CYAN}MODEL TYPES:{Style.RESET_ALL}")
        print(
            f"  {Fore.GREEN}local_small{Style.RESET_ALL}  - ~1B parameter models (fast, basic)"
        )
        print(
            f"  {Fore.GREEN}local_medium{Style.RESET_ALL} - ~7B parameter models (good balance)"
        )
        print(
            f"  {Fore.GREEN}local_large{Style.RESET_ALL}  - ~14B parameter models (best local quality)"
        )
        print(
            f"  {Fore.GREEN}paid_cheap{Style.RESET_ALL}   - Cost-effective cloud models"
        )
        print(
            f"  {Fore.GREEN}paid_best{Style.RESET_ALL}    - Highest quality cloud models"
        )

        print()
        info(
            "Use model types with --model-type or specific models with --model provider:model"
        )

    except Exception as e:
        error(f"Failed to get model information: {e}")
        return 1

    return 0


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="dazllm - Simple, unified interface for all major LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Fore.CYAN}Examples:{Style.RESET_ALL}
  dazllm chat "What's the capital of France?"
  dazllm chat --model openai:gpt-4 --file prompt.txt
  dazllm chat --model-type paid_best "Explain quantum computing"
  dazllm structured "List 3 colors" --schema '{{"type":"array","items":{{"type":"string"}}}}'
  dazllm image "a red cat" cat.png --width 512 --height 512
  dazllm --check
        """,
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"dazllm {Fore.GREEN}0.1.0{Style.RESET_ALL}",
    )
    parser.add_argument(
        "--check", action="store_true", help="Check configuration status"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an LLM")
    chat_parser.add_argument("prompt", nargs="?", help="Prompt text")
    chat_parser.add_argument("--model", help="Specific model name (provider:model)")
    chat_parser.add_argument(
        "--model-type", choices=[t.value for t in ModelType], help="Model type"
    )
    chat_parser.add_argument("--file", help="Read prompt from file")
    chat_parser.add_argument("--output", help="Save response to file")
    chat_parser.add_argument("--json", action="store_true", help="Force JSON output")

    # Structured chat command
    structured_parser = subparsers.add_parser(
        "structured", help="Chat with structured output"
    )
    structured_parser.add_argument("prompt", nargs="?", help="Prompt text")
    structured_parser.add_argument(
        "--model", help="Specific model name (provider:model)"
    )
    structured_parser.add_argument(
        "--model-type", choices=[t.value for t in ModelType], help="Model type"
    )
    structured_parser.add_argument("--file", help="Read prompt from file")
    structured_parser.add_argument("--schema", help="JSON schema string")
    structured_parser.add_argument("--schema-file", help="JSON schema file")
    structured_parser.add_argument(
        "--schema-class", help="Pydantic model class (module.ClassName)"
    )
    structured_parser.add_argument(
        "--context-size", type=int, default=0, help="Context window size"
    )
    structured_parser.add_argument("--output", help="Save response to file")

    # Image command
    image_parser = subparsers.add_parser("image", help="Generate an image")
    image_parser.add_argument("prompt", help="Image description")
    image_parser.add_argument("output", help="Output image file")
    image_parser.add_argument("--model", help="Specific model name (provider:model)")
    image_parser.add_argument(
        "--model-type", choices=[t.value for t in ModelType], help="Model type"
    )
    image_parser.add_argument("--width", type=int, default=1024, help="Image width")
    image_parser.add_argument("--height", type=int, default=1024, help="Image height")

    args = parser.parse_args()

    # Handle global options
    if args.check:
        return cmd_check(args)

    # Handle subcommands
    if args.command == "chat":
        return cmd_chat(args)
    elif args.command == "structured":
        return cmd_structured(args)
    elif args.command == "image":
        return cmd_image(args)
    else:
        # Default to help if no command
        if len(sys.argv) == 1:
            header("Welcome to dazllm! 🚀")
            info("Run 'dazllm --help' for usage information")
            info("Run 'dazllm --check' to verify configuration")
            return 0
        else:
            parser.print_help()
            return 1


if __name__ == "__main__":
    sys.exit(main())
