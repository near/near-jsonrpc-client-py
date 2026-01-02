from setuptools import setup, find_packages
import os

PACKAGE_VERSION = os.getenv("PACKAGE_VERSION", "0.1.0")

setup(
    name="near-jsonrpc-client",
    version=PACKAGE_VERSION,
    description="A Python client for interacting with NEAR JSON-RPC API",
    long_description="This is a Python client for interacting with the NEAR Protocol using JSON-RPC.",
    packages=find_packages(include=["near_jsonrpc_client*", "near_jsonrpc_models*"]),
    install_requires=[
        "httpx>=0.24",
        "pydantic>=2.0",
    ],
    python_requires=">=3.9"
)
