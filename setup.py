from setuptools import find_packages, setup


setup(
    name="hralign-reproduction",
    version="0.1.0",
    description="Release-compatible reproduction of R3M-Align-L.",
    packages=find_packages(include=["hralign", "hralign.*"]),
    install_requires=[
        "numpy>=1.24",
        "pillow>=10.0",
        "pyyaml>=6.0",
        "torch>=2.1",
        "torchvision>=0.16",
        "transformers>=4.40",
    ],
    python_requires=">=3.10",
)
