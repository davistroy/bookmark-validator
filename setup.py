#!/usr/bin/env python3
"""
Setup configuration for Bookmark Validation and Enhancement Tool
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bookmark-processor",
    version="1.0.0",
    author="Troy Davis",
    author_email="",
    description="A powerful Windows tool that processes raindrop.io bookmark exports to validate URLs, generate AI-enhanced descriptions, and create optimized tagging systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/davistroy/bookmark-validator",
    packages=find_packages(),
    package_data={
        "bookmark_processor": [
            "config/*.ini",
            "data/*.txt",
            "data/*.json",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "bookmark-processor=bookmark_processor.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)