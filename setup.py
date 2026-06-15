from setuptools import find_packages, setup


setup(
    name="tcc-core",
    version="0.1.0",
    description="Minimal RH20T multi-view pretraining code.",
    packages=find_packages(include=["xirl", "xirl.*"]),
    install_requires=[
        "numpy",
        "pillow",
        "pyyaml",
        "torch",
        "torchvision",
    ],
    python_requires=">=3.10",
)
