from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deeparguing",
    version="1.0.0",
    author="Adam Gould",
    author_email="adam.gould19@imperial.ac.uk",
    description="CLArg's Implementation of Deep Arguing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheAdamG/Gradual-AACBR",
    project_urls={
        "Bug Tracker": "https://github.com/TheAdamG/Gradual-AACBR/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    install_requires=[
        "matplotlib>=3.10.3,<3.11",
        "networkx>=3.4.2,<3.5",
        "numpy>=2.2.6,<2.3",
        "torch>=2.7.0,<2.8",
        "tqdm>=4.67.1,<4.68",
        "scikit-learn>=1.6.1,<1.7",
        "torchviz>=0.0.3,<0.0.4",
        "pandas>=2.2.3,<2.3",
        "ipykernel>=6.29.5,<6.30",
    ],
    dependency_links=["https://download.pytorch.org/whl/cu118"],
    python_requires=">=3.12",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
