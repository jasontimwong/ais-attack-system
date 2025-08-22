#!/usr/bin/env python3
"""
AIS Attack Generation System Setup Configuration
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ais-attack-system",
    version="1.0.0",
    author="Jason Tim Wong",
    author_email="jason@example.com",
    description="Advanced AIS Attack Generation and Visualization System for Maritime Cybersecurity Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jasontimwong/ais-attack-system",
    project_urls={
        "Bug Tracker": "https://github.com/jasontimwong/ais-attack-system/issues",
        "Documentation": "https://github.com/jasontimwong/ais-attack-system/docs",
        "Source Code": "https://github.com/jasontimwong/ais-attack-system",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Security",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=21.9.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.2.0",
            "sphinx-rtd-theme>=1.0.0",
            "mkdocs>=1.2.0",
            "mkdocs-material>=7.3.0",
        ],
        "bridge": [
            "pyserial>=3.5",
        ],
        "performance": [
            "numba>=0.54.0",
            "cython>=0.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ais-attack=core.cli:main",
            "ais-visualize=visualization.cli:main",
            "ais-batch=tools.batch_runner.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "datasets": ["scenarios/*.yaml", "labels/*.json"],
        "visualization": ["templates/*.html", "static/*"],
        "configs": ["*.yaml"],
    },
    zip_safe=False,
    keywords=[
        "ais", "maritime", "cybersecurity", "attack-simulation", 
        "vessel-tracking", "colregs", "ecdis", "maritime-security"
    ],
)
