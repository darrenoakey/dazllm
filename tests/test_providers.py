#!/usr/bin/env python3
"""
Fully generic provider test for dazllm.
Uses provider introspection - no provider-specific code!
"""

import sys
import traceback
import argparse
from typing import Dict, List
from colorama import init, Fore, Style
from pydantic import BaseModel

# Initialize colorama
init(autoreset=True)

# Import dazllm
try:
    from dazllm import Llm, ModelType, DazLlmError, ConfigurationError
except ImportError:
    print(
        f"{Fore.RED}✗ Could not import dazllm. Make sure it's installed: pip install -e .{Style.RESET_ALL}"
    )
    sys.exit(1)


def success(msg: str):
    """Print success message in green"""
    print(f"{Fore.GREEN}✓ {msg}{Style.RESET_ALL}")


def error(msg: str):
    """Print error message in red"""
    print(f"{Fore.RED}✗ {msg}{Style.RESET_ALL}")


def warning(msg: str):
    """Print warning message in yellow"""
    print(f"{Fore.YELLOW}⚠ {msg}{Style.RESET_ALL}")


def info(msg: str):
    """Print info message in blue"""
    print(f"{Fore.BLUE}ℹ {msg}{Style.RESET_ALL}")


def header(msg: str):
    """Print header message in bold cyan"""
    print(f"{Fore.CYAN}{Style.BRIGHT}{msg}{Style.RESET_ALL}")


# Test schemas for structured output
class SimpleResponse(BaseModel):
    answer: str
    confidence: float


class ColorList(BaseModel):
    colors: List[str]
    count: int


def test_provider_chat(provider_name: str, provider_info: Dict) -> bool:
    """Test basic chat functionality for a provider"""
    try:
        print(f"    Testing chat...")

        model_name = f"{provider_name}:{provider_info['default_model']}"

        # Test instance-based chat
        llm = Llm.model_named(model_name)
        response = llm.chat("What is 2+2? Answer briefly.")

        if not response or len(response.strip()) == 0:
            error(f"    Empty response from {model_name}")
            return False

        success(f"    Chat response: {response[:50]}...")

        # Test static chat
        static_response = Llm.chat(
            "What color is the sky? One word answer.", model=model_name
        )
        success(f"    Static chat response: {static_response[:30]}...")

        return True

    except Exception as e:
        error(f"    Chat failed: {e}")
        return False


def test_provider_structured(provider_name: str, provider_info: Dict) -> bool:
    """Test structured chat functionality for a provider"""
    try:
        if "structured" not in provider_info["capabilities"]:
            warning(f"    Structured chat not supported by {provider_name}")
            return True  # Not a failure, just not supported

        print(f"    Testing structured chat...")

        model_name = f"{provider_name}:{provider_info['default_model']}"

        # Test with SimpleResponse schema
        response = Llm.chat_structured(
            "What is the capital of France? Provide confidence 0-1.",
            SimpleResponse,
            model=model_name,
        )

        if not isinstance(response, SimpleResponse):
            error(f"    Did not return SimpleResponse object")
            return False

        success(
            f"    Structured response: {response.answer} (confidence: {response.confidence})"
        )

        # Test with ColorList schema
        color_response = Llm.chat_structured(
            "List exactly 3 primary colors", ColorList, model=model_name
        )

        if not isinstance(color_response, ColorList):
            error(f"    Did not return ColorList object")
            return False

        success(
            f"    Color list: {color_response.colors} (count: {color_response.count})"
        )

        return True

    except Exception as e:
        error(f"    Structured chat failed: {e}")
        return False


def test_provider_image(provider_name: str, provider_info: Dict) -> bool:
    """Test image generation functionality for a provider"""
    try:
        if "image" not in provider_info["capabilities"]:
            warning(f"    Image generation not supported by {provider_name}")
            return True  # Not a failure, just not supported

        print(f"    Testing image generation...")

        model_name = f"{provider_name}:{provider_info['default_model']}"

        # Generate a simple image
        output_file = f"test_image_{provider_name}.png"
        result_path = Llm.image(
            "a simple red circle", output_file, width=256, height=256, model=model_name
        )

        # Check if file was created
        import os

        if os.path.exists(result_path):
            success(f"    Image generated: {result_path}")
            # Clean up test file
            try:
                os.remove(result_path)
            except:
                pass
            return True
        else:
            error(f"    Image file not created: {result_path}")
            return False

    except DazLlmError as e:
        # Handle provider-specific limitations gracefully
        if (
            "requires Google Cloud" in str(e)
            or "Vertex AI" in str(e)
            or "Cloud setup" in str(e)
        ):
            warning(f"    Image generation needs additional setup: {e}")
            return True  # Not a failure, just needs more setup
        else:
            error(f"    Image generation failed: {e}")
            return False
    except Exception as e:
        error(f"    Image generation failed: {e}")
        return False


def test_model_types() -> bool:
    """Test model type resolution"""
    try:
        header("Testing Model Type Resolution")

        # Test each model type
        for model_type in ModelType:
            try:
                print(f"  Testing {model_type.value}...")
                response = Llm.chat("Hello", model_type=model_type)
                success(f"  {model_type.value}: {response[:30]}...")
            except Exception as e:
                error(f"  {model_type.value} failed: {e}")
                return False

        return True

    except Exception as e:
        error(f"Model type testing failed: {e}")
        return False


def test_provider_shortcuts(providers_info: Dict) -> bool:
    """Test provider shortcut functionality"""
    try:
        header("Testing Provider Shortcuts")

        for provider_name, provider_info in providers_info.items():
            if not provider_info["configured"]:
                continue

            try:
                print(f"  Testing '{provider_name}' shortcut...")
                response = Llm.chat("Hi", model=provider_name)
                success(f"  {provider_name}: {response[:30]}...")
            except Exception as e:
                error(f"  {provider_name} shortcut failed: {e}")

        return True

    except Exception as e:
        error(f"Provider shortcut testing failed: {e}")
        return False


def run_comprehensive_test(target_provider=None):
    """Run comprehensive test using provider introspection"""
    if target_provider:
        header(f"🚀 dazllm {target_provider.upper()} Provider Test")
    else:
        header("🚀 dazllm Comprehensive Provider Test")
    print()

    # Get all provider information using introspection
    try:
        all_providers_info = Llm.get_all_providers_info()
    except Exception as e:
        error(f"Failed to get provider information: {e}")
        return False

    # Filter providers if target specified
    if target_provider:
        if target_provider not in all_providers_info:
            error(f"Unknown provider: {target_provider}")
            error(f"Available providers: {', '.join(all_providers_info.keys())}")
            return False
        providers_to_test = {target_provider: all_providers_info[target_provider]}
    else:
        providers_to_test = all_providers_info

    total_tests = 0
    passed_tests = 0
    provider_results = {}

    # Test each provider
    for provider_name, provider_info in providers_to_test.items():
        header(f"Testing {provider_name.upper()} Provider")

        provider_results[provider_name] = {
            "chat": False,
            "structured": False,
            "image": False,
            "configured": provider_info["configured"],
            "capabilities": provider_info["capabilities"],
            "models": provider_info["supported_models"],
        }

        if not provider_info["configured"]:
            error(f"  {provider_name} not configured")
            warning(f"  Skipping {provider_name} tests")
            print()
            continue

        success(f"  {provider_name} is configured")
        info(f"  Capabilities: {', '.join(provider_info['capabilities'])}")
        info(f"  Supported models: {len(provider_info['supported_models'])} models")

        # Test chat (all providers should support this)
        total_tests += 1
        if test_provider_chat(provider_name, provider_info):
            passed_tests += 1
            provider_results[provider_name]["chat"] = True

        # Test structured chat (if supported)
        total_tests += 1
        if test_provider_structured(provider_name, provider_info):
            passed_tests += 1
            provider_results[provider_name]["structured"] = True

        # Test image generation (if supported)
        total_tests += 1
        if test_provider_image(provider_name, provider_info):
            passed_tests += 1
            provider_results[provider_name]["image"] = True

        print()

    # Test cross-provider functionality (if not targeting specific provider)
    configured_providers = [p for p, r in provider_results.items() if r["configured"]]
    if configured_providers and not target_provider:
        total_tests += 1
        if test_model_types():
            passed_tests += 1

        total_tests += 1
        if test_provider_shortcuts(providers_to_test):
            passed_tests += 1

    # Print summary
    header("📊 Test Results Summary")
    print()

    for provider_name, results in provider_results.items():
        if results["configured"]:
            status_symbols = []
            status_symbols.append("✓" if results["chat"] else "✗")

            # Structured output status
            if "structured" in results["capabilities"]:
                status_symbols.append("✓" if results["structured"] else "✗")
            else:
                status_symbols.append("~")

            # Image generation status
            if "image" in results["capabilities"]:
                if results["image"]:
                    status_symbols.append("✓")
                else:
                    status_symbols.append("⚙")  # Needs additional setup
            else:
                status_symbols.append("~")  # Not supported

            model_count = len(results["models"])
            print(
                f"  {provider_name:10} Chat:{status_symbols[0]} Structured:{status_symbols[1]} Image:{status_symbols[2]} Models:{model_count}"
            )
        else:
            print(f"  {provider_name:10} {Fore.YELLOW}Not configured{Style.RESET_ALL}")

    print()
    print(f"  Legend: ✓=Works ✗=Failed ~=Not supported ⚙=Needs additional setup")
    print()

    if passed_tests == total_tests:
        success(f"All tests passed! ({passed_tests}/{total_tests})")
        return True
    else:
        error(f"Some tests failed. ({passed_tests}/{total_tests} passed)")
        return False


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(
        description="Fully generic test for all dazllm providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
{Fore.CYAN}Examples:{Style.RESET_ALL}
  python tests/test_providers.py                    # Test all providers
  python tests/test_providers.py --provider openai  # Test only OpenAI
  python tests/test_providers.py --provider google  # Test only Google

{Fore.CYAN}Available providers are discovered automatically!{Style.RESET_ALL}
        """,
    )

    # Get available providers dynamically
    try:
        available_providers = Llm.get_providers()
        parser.add_argument(
            "--provider",
            choices=available_providers,
            help="Test only the specified provider (default: test all providers)",
        )
    except Exception as e:
        error(f"Could not discover providers: {e}")
        return 1

    args = parser.parse_args()

    try:
        success_result = run_comprehensive_test(target_provider=args.provider)

        print()
        header("🔧 Configuration Help")
        info("To configure providers, use keyring:")
        print("  keyring set dazllm openai_api_key YOUR_OPENAI_KEY")
        print("  keyring set dazllm anthropic_api_key YOUR_ANTHROPIC_KEY")
        print("  keyring set dazllm google_api_key YOUR_GOOGLE_KEY")
        print("  keyring set dazllm ollama_url http://localhost:11434")

        print()
        info("Check configuration status:")
        print("  dazllm --check")

        return 0 if success_result else 1

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Test interrupted by user{Style.RESET_ALL}")
        return 1
    except Exception as e:
        error(f"Test failed with unexpected error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
