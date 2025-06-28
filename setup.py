from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="dazllm",
    version="0.1.0",
    author="Darren Oakey",
    author_email="darren.oakey@insidemind.com.au",
    description="Simple, unified interface for all major LLMs",
    long_description=long_description,
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
    install_requires=requirements,
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
