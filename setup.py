"""
Setup script for VLM Content Moderation System
"""
from setuptools import setup, find_packages

setup(
    name="vlm-content-moderation",
    version="0.1.0",
    description="Content moderation system for detecting embedded hateful content in images",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
        "pytesseract>=0.3.10",
        "easyocr>=1.7.0",
        "hypothesis>=6.82.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ]
    },
)
