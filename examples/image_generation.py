#!/usr/bin/env python3
"""
Image generation example for dazllm
"""

from dazllm import Llm

def main():
    print("Image generation example")
    
    try:
        # Generate an image
        prompt = "A serene mountain landscape at sunset"
        filename = "mountain_sunset.png"
        
        result_path = Llm.image(
            prompt,
            filename,
            width=1024,
            height=768,
            model="openai:dall-e-3"
        )
        
        print(f"Image generated and saved to: {result_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have configured your OpenAI API key:")
        print("keyring set dazllm openai_api_key YOUR_KEY")

if __name__ == "__main__":
    main()
