#!/usr/bin/env python3
"""Demo script to test Gemini image generation with nano model"""

import os
import sys
from dazllm import Llm
from colorama import init, Fore, Style

init(autoreset=True)


def main():
    """Main function to test Gemini image generation"""
    print(f"{Fore.CYAN}=== Gemini Image Generation Demo ==={Style.RESET_ALL}\n")

    # Check if API key is configured
    from dazllm.llm_google import LlmGoogle
    LlmGoogle.check_config()
    print(f"{Fore.GREEN}✓ Google API key found in keyring{Style.RESET_ALL}")

    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Test prompts
    test_prompts = [
        ("A serene mountain landscape at sunset with vibrant colors", "landscape.png"),
        ("A cute robot learning to paint, digital art style", "robot_painter.png"),
        ("Abstract geometric patterns in blue and gold", "abstract.png"),
    ]

    for prompt, filename in test_prompts:
        output_path = os.path.join(output_dir, filename)
        print(f"\n{Fore.CYAN}Generating: {prompt}{Style.RESET_ALL}")
        print(f"Output: {output_path}")

        # Create Google LLM instance
        llm = Llm.model_named("google:gemini-2.0-flash")

        # Generate image
        result = llm.image(prompt, output_path, width=1024, height=1024)
        print(f"{Fore.GREEN}✓ Image generated successfully: {result}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Demo completed!{Style.RESET_ALL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())