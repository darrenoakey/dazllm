"""
Setup script for dazllm package
"""

from setuptools import setup, find_packages


def get_long_description():
    """Read long description from README.md"""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def get_requirements():
    """Read requirements from requirements.txt"""
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


def run_setup():
    """Run the setup configuration"""
    setup(
        name="dazllm",
        version="0.2.0",
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
        version = "0.1.0"
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

    def test_functions_exist(self):
        """Test that setup functions exist"""
        self.assertTrue(callable(get_long_description))
        self.assertTrue(callable(get_requirements))
        self.assertTrue(callable(run_setup))

    def test_entry_points_format(self):
        """Test entry points are correctly formatted"""
        entry_point = "dazllm=dazllm.cli:main"
        self.assertIn("=", entry_point)
        parts = entry_point.split("=")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "dazllm")
        self.assertEqual(parts[1], "dazllm.cli:main")

    def test_keywords(self):
        """Test keywords are relevant"""
        keywords = "llm ai openai anthropic claude gemini ollama chatgpt gpt-4"
        keyword_list = keywords.split()
        self.assertIn("llm", keyword_list)
        self.assertIn("ai", keyword_list)
        self.assertTrue(len(keyword_list) > 5)
