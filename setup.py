from setuptools import setup, find_packages

setup(
    name="MNLI-MultiModel-Benchmark",
    version="0.1.0",
    author="Ilyes DJERFAF, Nazim KESKES",
    description="A project for Multi Natural Language Inference (MNLI) MultiModel Benchmark.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mlengineershub/MNLI-MultiModel-Benchmark",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "transformers"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
