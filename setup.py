from setuptools import setup

import os
PACKAGE_VERSION = os.getenv("PACKAGE_VERSION", "0.0.1")

setup(
    name="near_jsonrpc_client",
    version=PACKAGE_VERSION,
    packages=["client", "models"],
    install_requires=[
        "pydantic>=2.0",
        "httpx>=0.24",
    ],
    python_requires=">=3.11",
)