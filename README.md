# NEAR JSON-RPC Python Client

[![Build Status](https://img.shields.io/github/actions/workflow/status/near/near-jsonrpc-client-kotlin/ci-cd.yml?branch=main)](https://github.com/hosseinkarami-dev/near-jsonrpc-client-py/actions)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Type Safe](https://img.shields.io/badge/type--safe-yes-success.svg)
![Release Badge](https://img.shields.io/github/tag/hosseinkarami-dev/near-jsonrpc-client-py.svg?label=release)

A **type-safe**, Pythonic client for the NEAR Protocol JSON-RPC API.

---

## Table of contents

* [Overview](#-overview)
* [Features](#-features)
* [Requirements](#-requirements)
* [Installation](#-installation)
* [Quickstart](#-quickstart)
* [Basic Usage](#-basic-usage)
* [Handling Responses & Errors](#-handling-responses--errors)
* [Testing](#-testing)
* [Contributing](#-contributing)
* [Deployment Guide](#-deployment-guide)
* [License](#-license)
* [References](#-references)

---

## 📖 Overview

This library provides a **type-safe**, developer-friendly Python interface for interacting with the NEAR Protocol JSON-RPC API.

* Fully typed request & response models
* Clean separation between transport, RPC layer, and domain models
* Designed for both scripting and production use

The client is inspired by official NEAR JSON-RPC client in [Kotlin](https://github.com/near/near-jsonrpc-client-kotlin).

| Module       | Description                                                                      |
|--------------|----------------------------------------------------------------------------------|
| `client`     | Python JSON-RPC client supporting both sync and async usage, with full NEAR RPC method wrappers (auto-generated) |
| `models`     | Typed Python classes for RPC requests and responses using Pydantic (auto-generated)            |
| `generator`  | Tools for generating Python client and Pydantic models from NEAR’s OpenAPI specification     |

---

## ✨ Features

🎯 **Type-Safe API**
All RPC requests and responses are represented as typed Python models (dataclasses / Pydantic), reducing runtime errors.

⚡ **Simple & Explicit Design**
No magic. Each RPC method maps directly to a NEAR JSON-RPC endpoint.

🛡️ **Structured Error Handling**
Clear distinction between:

* JSON-RPC errors
* HTTP errors
* Network failures
* Serialization issues

🔄 **Sync & Async Friendly**

* Synchronous client for scripts & backend services using `httpx.Client`
* Optional async client for asyncio-based applications using `httpx.AsyncClient`

📦 **Minimal Dependencies**
Built on top of well-known Python libraries (`httpx` and `pydantic`).


🧪 **Testable by Design**
Easy to mock transport layer for unit & integration tests.

---

## ⚙️ Requirements

* Python **3.9+**
* `httpx` (used for both sync and async transports)
* `pydantic` (for type-safe request/response models)

---

## 📦 Installation

```bash
pip install near-jsonrpc-client httpx pydantic
```

---

## 🚀 Quickstart

### Async Client

```python
import asyncio
from near_jsonrpc_client import NearClientAsync
from near_jsonrpc_models import RpcBlockRequest, BlockId, RpcBlockRequestBlockId, BlockIdBlockHeight


async def main():
    client = NearClientAsync(rpc_urls="https://rpc.mainnet.near.org")

    params = RpcBlockRequest(
        RpcBlockRequestBlockId(
            block_id=BlockId(BlockIdBlockHeight(178682261))
        )
    )

    block = await client.block(params=params)
    print(block)

    await client.close()


asyncio.run(main())
```

### Sync Client

```python
from near_jsonrpc_client import NearClientSync
from near_jsonrpc_models import RpcBlockRequest, BlockId, RpcBlockRequestBlockId, BlockIdBlockHeight

client = NearClientSync(rpc_urls="https://rpc.mainnet.near.org")

params = RpcBlockRequest(
    RpcBlockRequestBlockId(
        block_id=BlockId(BlockIdBlockHeight(178682261))
    )
)

block = client.block(params=params)
print(block)

client.close()
```

---

## 📝 Basic Usage

* Create request models for each RPC method.
* Call the method on the appropriate client (async or sync).
* Receive typed response models.

```python
from near_jsonrpc_models import RpcBlockRequest, RpcBlockRequestBlockId, BlockIdBlockHeight, BlockId

params = RpcBlockRequest(RpcBlockRequestBlockId(block_id=BlockId(BlockIdBlockHeight(178682261))))
response = client.block(params=params)
print(response)
```

---

## ⚠️ Handling Responses & Errors

The client raises structured exceptions:

* `RpcError` – returned from NEAR JSON-RPC
* `RpcHttpError` – HTTP errors with status code and body
* `RpcTimeoutError` – request timeout
* `RpcClientError` – unexpected or invalid responses

Example:

```python
from near_jsonrpc_client import RpcError, RpcHttpError, RpcTimeoutError, RpcClientError

try:
    block = client.block(params=params)
except RpcError as e:
    print(f"RPC error: {e.error}")
except RpcHttpError as e:
    print(f"HTTP error: {e.status_code}, {e.body}")
except RpcTimeoutError as e:
    print("Request timed out")
except RpcClientError as e:
    print("Invalid response", e)
```

---

## 🧪 Testing

* Simply run `pytest` to execute all tests.
* The transport layer (`HttpTransportAsync` or `HttpTransportSync`) is mocked internally, so no actual network calls are made.

---

## 🤝 Contributing

* Fork the repository
* Create a feature branch
* Submit a pull request with tests

---

## 📜 License

This project is licensed under the Apache-2.0 License. See LICENSE for details.

---

## 📦 Deployment Guide

For detailed instructions on project structure, CI/CD workflow, versioning, and deployment steps, see the [DEPLOYMENT.md](./DEPLOYMENT.md) file.

---

## 📚 References

* [NEAR Protocol JSON-RPC](https://docs.near.org/docs/api/rpc)
* [httpx Documentation](https://www.python-httpx.org/)
* [Pydantic Documentation](https://docs.pydantic.dev/)

