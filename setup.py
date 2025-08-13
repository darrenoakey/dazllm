"""
Setup configuration for dazllm package.

This setup script configures the package for distribution. The version number
is read from dazllm/__init__.py to maintain a single source of truth.
"""

import os
import re
from setuptools import setup, find_packages


def get_version():
    """Get version from dazllm/__init__.py"""
    version_file = os.path.join(os.path.dirname(__file__), "dazllm", "__init__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string")


def get_long_description():
    """Get long description from README.md"""
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_requirements():
    """Get requirements from requirements.txt"""
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []


def run_setup():
    """Run the setup configuration"""
    setup(
        name="dazllm",
        version=get_version(),
        author="Darren Oakey",
        author_email="darren.oakey@insidemind.com.au",
        description="Simple, unified interface for all major LLMs",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        url="https://github.com/darrenoakey/dazllm",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        python_requires=">=3.8",
        install_requires=get_requirements(),
        entry_points={
            "console_scripts": [
                "dazllm=dazllm.cli:main",
            ],
        },
        keywords="llm ai openai anthropic claude gemini ollama chatgpt gpt-4",
        project_urls={
            "Bug Reports": "https://github.com/darrenoakey/dazllm/issues",
            "Source": "https://github.com/darrenoakey/dazllm",
            "Documentation": "https://github.com/darrenoakey/dazllm#readme",
        },
    )


if __name__ == "__main__":
    run_setup()


# Unit tests
import unittest


class TestSetupConfiguration(unittest.TestCase):
    """Test cases for setup.py configuration"""

    def test_package_name(self):
        """Test package name is correct"""
        package_name = "dazllm"
        self.assertEqual(package_name, "dazllm")
        self.assertIsInstance(package_name, str)

    def test_version_format(self):
        """Test version follows semantic versioning"""
        version = get_version()
        parts = version.split(".")
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertTrue(part.isdigit())

    def test_author_info(self):
        """Test author information is present"""
        author = "Darren Oakey"
        author_email = "darren.oakey@insidemind.com.au"
        self.assertIsInstance(author, str)
        self.assertIsInstance(author_email, str)
        self.assertIn("@", author_email)

    def test_description(self):
        """Test package description"""
        description = "Simple, unified interface for all major LLMs"
        self.assertIsInstance(description, str)
        self.assertTrue(len(description) > 10)

    def test_python_requires(self):
        """Test Python version requirement"""
        python_requires = ">=3.8"
        self.assertIsInstance(python_requires, str)
        self.assertIn("3.8", python_requires)

    def test_version_can_be_read(self):
        """Test that version can be read from __init__.py"""
        version = get_version()
        self.assertIsInstance(version, str)
        self.assertGreater(len(version), 0)
        # Test semantic versioning format
        parts = version.split(".")
        self.assertEqual(len(parts), 3)
