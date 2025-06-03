from setuptools import setup, find_packages

setup(
    name="tonic",
    version="0.3.0",
    packages=["tonic"],  # Explicitly include only the 'tonic' package
    install_requires=[
        "gym>=0.21.0",
        "matplotlib",
        "numpy",
        "pandas",
        "pyyaml",
        "termcolor",
    ],
    author="Fabio Pardo",
    description="A deep reinforcement learning library.",
    url="https://github.com/fabiopardo/tonic",
)
