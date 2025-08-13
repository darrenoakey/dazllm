#!/usr/bin/env python3
"""
dazllm command line interface with colorama support
"""

import argparse
import sys
import unittest
from typing import Type
from pydantic import BaseModel

# Colorama for colored output
from colorama import init, Fore, Style

init(autoreset=True)

from .core import Llm, ModelType, check_configuration
from .core import DazLlmError


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
        if hasattr(args, 'file') and args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                conversation = f.read().strip()
        else:
            conversation = args.prompt

        if not conversation:
            error("No prompt provided. Use --prompt or --file")
            return 1

        # Make the call
        response = Llm.chat_static(
            conversation,
            model=getattr(args, 'model', None),
            model_type=ModelType(args.model_type) if hasattr(args, 'model_type') and args.model_type else None,
            force_json=getattr(args, 'json', False),
        )

        # Output
        if hasattr(args, 'output') and args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(response)
            success(f"Response saved to {args.output}")
        else:
            print(response)

        return 0

    except (DazLlmError, FileNotFoundError) as e:
        error(str(e))
        return 1


def cmd_check(_args):
    """Check configuration status"""
    header("Checking dazllm configuration...\\n")

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


def create_dynamic_model(schema_dict: dict) -> Type[BaseModel]:
    """Create a dynamic Pydantic model from JSON schema"""
    from pydantic import create_model

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


def get_version():
    """Get version from __init__.py in the same directory"""
    try:
        # Try to import the version directly first
        from . import __version__
        return __version__
    except ImportError:
        # Fallback to reading the file directly
        import os
        import re
        version_file = os.path.join(os.path.dirname(__file__), "__init__.py")
        try:
            with open(version_file, "r", encoding="utf-8") as f:
                content = f.read()
                version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
                if version_match:
                    return version_match.group(1)
        except FileNotFoundError:
            pass
    # Final fallback - should never be reached if __init__.py exists
    return "unknown"  # fallback version  # fallback version


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="dazllm - Simple, unified interface for all major LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Fore.CYAN}Examples:{Style.RESET_ALL}
  dazllm chat "What's the capital of France?"
  dazllm chat --model openai:gpt-4 --file prompt.txt
  dazllm --check
        """,
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"dazllm {Fore.GREEN}{get_version()}{Style.RESET_ALL}",
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

    args = parser.parse_args()

    # Handle global options
    if args.check:
        return cmd_check(args)

    # Handle subcommands
    if args.command == "chat":
        return cmd_chat(args)
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


class TestCli(unittest.TestCase):
    """Test CLI functions"""

    def test_cli_functions(self):
        """Test that CLI functions exist"""
        self.assertTrue(callable(main))
        self.assertTrue(callable(cmd_chat))
        self.assertTrue(callable(cmd_check))
        self.assertTrue(callable(create_dynamic_model))

    def test_get_version(self):
        """Test version function"""
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)


if __name__ == "__main__":
    sys.exit(main())
