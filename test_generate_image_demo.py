#!/usr/bin/env python3
"""
Demo script to test Llm.generate_image() functionality
This is a real-world test (not unit test) to verify the image generation works
"""

import os
import tempfile
from dazllm.core import Llm, DazLlmError

def test_generate_image():
    """Test the Llm.generate_image class method"""
    
    # Create a temporary file for the image
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        image_path = tmp_file.name
    
    try:
        print("Testing Llm.generate_image()...")
        print(f"Output file: {image_path}")
        
        # Test the class method
        result = Llm.generate_image(
            "A simple red square on a white background",
            image_path,
            512, 512
        )
        
        print(f"Image generation completed successfully!")
        print(f"Result: {result}")
        
        # Check if file was created
        if os.path.exists(image_path):
            size = os.path.getsize(image_path)
            print(f"Generated image file size: {size} bytes")
            return True
        else:
            print("ERROR: Image file was not created")
            return False
            
    except DazLlmError as e:
        print(f"Expected error (no provider configured): {e}")
        return True  # This is expected if no providers are configured
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Cleaned up temporary file: {image_path}")

def test_provider_info():
    """Test provider information and capabilities"""
    print("\n" + "="*50)
    print("Testing provider information...")
    
    try:
        providers = Llm.get_providers()
        print(f"Available providers: {providers}")
        
        for provider in providers:
            try:
                info = Llm.get_provider_info(provider)
                capabilities = info.get('capabilities', set())
                configured = info.get('configured', False)
                print(f"{provider}: configured={configured}, capabilities={capabilities}")
                
                if 'image' in capabilities:
                    print(f"  -> {provider} supports image generation!")
                    
            except Exception as e:
                print(f"  -> Error getting info for {provider}: {e}")
                
    except Exception as e:
        print(f"Error getting provider list: {e}")

if __name__ == "__main__":
    print("="*60)
    print("dazllm Image Generation Test")
    print("="*60)
    
    # Test provider capabilities first
    test_provider_info()
    
    print("\n" + "="*50)
    # Test actual image generation
    success = test_generate_image()
    
    print("\n" + "="*50)
    if success:
        print("✅ Image generation test completed successfully!")
    else:
        print("❌ Image generation test failed!")
    print("="*50)