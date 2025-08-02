"""Image processing utilities for OpenAI provider

This module contains image processing functions including size calculation,
aspect ratio enhancement, resizing, cropping, and saving operations.
"""

from __future__ import annotations
import os
import unittest
from typing import Tuple
from PIL import Image
import requests


class ImageUtils:
    """Utility functions for image processing"""

    @staticmethod
    def calculate_optimal_size(width: int, height: int) -> Tuple[int, int]:
        """Calculate optimal size within OpenAI constraints"""
        openai_sizes = [
            (1024, 1024), (1152, 896), (1216, 832), (1344, 768),
            (1536, 640), (640, 1536), (768, 1344), (832, 1216),
            (896, 1152)
        ]
        
        target_ratio = width / height
        best_size = (1024, 1024)
        min_diff = float('inf')
        
        for size in openai_sizes:
            size_ratio = size[0] / size[1]
            diff = abs(target_ratio - size_ratio)
            if diff < min_diff:
                min_diff = diff
                best_size = size
        
        return best_size

    @staticmethod
    def enhance_prompt_for_aspect_ratio(prompt: str, width: int, height: int) -> str:
        """Enhance prompt based on aspect ratio"""
        ratio = width / height
        if ratio > 1.5:
            return f"{prompt}. Wide landscape format, panoramic view."
        elif ratio < 0.67:
            return f"{prompt}. Tall portrait format, vertical composition."
        return prompt

    @staticmethod
    def resize_and_crop(image_path: str, target_width: int, target_height: int) -> str:
        """Resize and crop image to target dimensions"""
        with Image.open(image_path) as img:
            img_ratio = img.width / img.height
            target_ratio = target_width / target_height
            
            if img_ratio > target_ratio:
                new_height = target_height
                new_width = int(new_height * img_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                left = (new_width - target_width) // 2
                img = img.crop((left, 0, left + target_width, target_height))
            else:
                new_width = target_width
                new_height = int(new_width / img_ratio)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                top = (new_height - target_height) // 2
                img = img.crop((0, top, target_width, top + target_height))
            
            processed_path = image_path.replace('.', '_processed.')
            img.save(processed_path, 'PNG')
            return processed_path

    @staticmethod
    def save_image(response_url: str, file_name: str) -> str:
        """Download and save image from URL"""
        response = requests.get(response_url, timeout=30)
        response.raise_for_status()
        
        base_dir = os.path.dirname(file_name) or '.'
        os.makedirs(base_dir, exist_ok=True)
        
        with open(file_name, 'wb') as f:
            f.write(response.content)
        
        return file_name


class TestImageUtils(unittest.TestCase):
    """Tests for image utility functions"""

    def test_calculate_optimal_size(self):
        """Test optimal size calculation"""
        size = ImageUtils.calculate_optimal_size(1200, 800)
        self.assertIn(size, [(1024, 1024), (1152, 896), (1216, 832)])

    def test_enhance_prompt_logic(self):
        """Test prompt enhancement for different ratios"""
        wide_prompt = ImageUtils.enhance_prompt_for_aspect_ratio("test", 1600, 900)
        self.assertIn("landscape", wide_prompt)
        
        tall_prompt = ImageUtils.enhance_prompt_for_aspect_ratio("test", 600, 1000)
        self.assertIn("portrait", tall_prompt)


__all__ = ["ImageUtils"]
