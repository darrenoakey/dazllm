#!/usr/bin/env python3
"""
Test utilities for dazllm provider testing.
Provides common formatting and helper functions for test output.
"""

import unittest
from colorama import init, Fore, Style
from pydantic import BaseModel
from typing import List

# Initialize colorama
init(autoreset=True)


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
    """Simple response model for testing."""
    answer: str
    confidence: float


class ColorList(BaseModel):
    """Color list model for testing."""
    colors: List[str]
    count: int


class TestUtilities(unittest.TestCase):
    """Test the utility functions."""
    
    def test_schema_creation(self):
        """Test that test schemas can be created."""
        response = SimpleResponse(answer="test", confidence=0.9)
        self.assertEqual(response.answer, "test")
        self.assertEqual(response.confidence, 0.9)
        
        colors = ColorList(colors=["red", "blue"], count=2)
        self.assertEqual(len(colors.colors), 2)
        self.assertEqual(colors.count, 2)
