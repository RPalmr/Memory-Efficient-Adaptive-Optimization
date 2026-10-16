from setuptools import setup, find_packages

setup(
    name="meao",
    version="0.1.0",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=["numpy>=1.15.0"],
    python_requires=">=3.9",
    description="Implementation of Memory-Efficient Adaptive Optimization"
)
