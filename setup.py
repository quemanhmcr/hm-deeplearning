from setuptools import setup, find_packages

setup(
    name="huggingface-project",
    version="0.1.0",
    description="Deep Learning Project for HuggingFace Challenge",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
    ],
)