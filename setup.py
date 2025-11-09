"""
Setup script for the DiDA package.
"""

from setuptools import setup, find_packages

setup(
    name="dida",
    version="0.1.0",
    author="Manus AI",
    description="An educational implementation of Discrete Diffusion Adaptation (DiDA) from the Emu3.5 paper.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hwilner/discrete-diffusion-adaptation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "torch",
    ],
)
