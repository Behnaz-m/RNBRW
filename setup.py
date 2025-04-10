from setuptools import setup, find_packages

setup(
    name="rnbrw",
    version="0.1.6.1",
    author="Behnaz Moradi-Jamei",
    description="Renewal Non-Backtracking Random Walk (RNBRW) for community detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rnbrw",
    packages=find_packages(),
    install_requires=[
        "numpy", "scipy", "matplotlib", "networkx", "joblib", "python-louvain"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
