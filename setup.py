from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bridge-lm",
    version="1.0.0",
    author="Chaz Hyseni",
    author_email="chaz.hyseni@gmail.com",
    description="BRIDGE-LM: Biological Regulatory Integration & Detection through Genomic Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chazhyseni/BRIDGE-LM",
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.9,<3.11",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "biopython>=1.81",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "rich>=13.6.0",
        "peft>=0.6.0",
        "accelerate>=0.24.0",
        "einops>=0.7.0",
    ],
    entry_points={
        "console_scripts": [
            "bridge=bridge.cli:main",
            "bridge-api=bridge.api.server:main",
        ],
    },
)
