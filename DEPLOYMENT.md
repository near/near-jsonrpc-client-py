# 🧩 Project Structure & Overview

This project — **NEAR JSON-RPC Client (Python)** — is a fully automated **Python client and model generator**
designed to interact with the [NEAR Protocol JSON-RPC API](https://docs.near.org/api/rpc/introduction).

It uses the **official OpenAPI specification** from the NEAR core repository to generate:
- Strongly-typed **Pydantic v2 models**
- A complete **async/sync JSON-RPC client** built on top of `httpx`

The goal is to provide a **reliable, up-to-date, and fully automated Python client**
for developers building tooling, analytics, SDKs, or backend integrations on NEAR.

---

## 📁 Project Structure

```
near-jsonrpc-client-python/
├── generator/                    # OpenAPI → Python code generator
│   ├── scripts/
│   └── README.md
│
├── near_jsonrpc_client/          # Runtime client implementation (async + sync)
│   ├── __init__.py
│   ├── client.py
│   ├── api_methods_async.py
│   ├── api_methods_sync.py
│   ├── base_client.py
│   ├── transport.py
│   ├── errors.py
│   └── cli.py
│
├── near_jsonrpc_models/          # Generated Pydantic models (requests & responses)
│   ├── __init__.py
│   └── *.py
│
├── tests/                        # Unit & integration tests
│
├── examples/                     # Usage examples
│
├── pyproject.toml                # Packaging & build configuration
├── setup.py                      # Legacy setuptools compatibility
├── DEPLOYMENT.md                 # Deployment & CI/CD documentation
└── README.md                     # Project introduction
```

---

## ⚙️ Module Overview

| Module | Description |
|------|------------|
| **generator** | Fetches the latest NEAR OpenAPI spec and regenerates Python models & client code |
| **near_jsonrpc_client** | Core RPC client (async & sync), transports, errors, and CLI |
| **near_jsonrpc_models** | Auto-generated Pydantic v2 models for all NEAR RPC methods |
| **tests** | Validation tests for models and client behavior |
| **examples** | Practical usage samples |

---

## 🧠 How It Works (High-Level)

1. The **generator** downloads the official NEAR JSON-RPC OpenAPI specification.
2. Python code is generated:
   - Pydantic request/response models
   - RPC method wrappers
3. CI/CD detects changes in generated sources.
4. Changes are committed, merged into `main`, and versioned.
5. A new GitHub Release is created and published to **PyPI**.

---

## ⚙️ CI/CD Workflow & Versioning

### Overview
This repository uses a **fully automated GitHub Actions pipeline** to:
- Regenerate client & models
- Commit and merge changes
- Create semantic versioned tags
- Publish releases to **GitHub** and **PyPI**

### Workflow Triggers
- **`workflow_dispatch`** — Manual trigger
- **`schedule`** — Daily automated run

### Pipeline Steps (Simplified)
1. Checkout repository (with full history & tags)
2. Run generator against latest OpenAPI spec
3. Detect changes in generated code
4. Auto-commit to a regeneration branch
5. Auto-merge into `main`
6. Determine next semantic version
7. Create Git tag (`vX.Y.Z`)
8. Build Python distribution (`sdist` + `wheel`)
9. Publish package to **PyPI**

---

## 🔢 Versioning Strategy

- Git tags follow **Semantic Versioning**:
  ```
  vMAJOR.MINOR.PATCH
  ```
- Default behavior increments **PATCH**
- Version source of truth:
  - Git tag → injected into build via `PACKAGE_VERSION`
- PyPI version always matches Git tag (without `v` prefix)

### Example
| Git Tag | PyPI Version |
|-------|--------------|
| v1.0.0 | 1.0.0 |
| v1.0.1 | 1.0.1 |

---

## 🧩 Commit Convention

This project follows **Conventional Commits**:

| Prefix | Meaning | Version Impact |
|------|--------|---------------|
| `feat:` | New feature | Minor |
| `fix:` | Bug fix | Patch |
| `refactor:` | Internal refactor | None |
| `docs:` | Documentation | None |
| `chore:` | Generator / CI changes | None |
| `test:` | Tests | None |

---

## 🛠 Environment / Prerequisites

### Local Development
- Python **3.9+**
- pip / virtualenv recommended
- Internet access (OpenAPI fetch)

### CI Environment
- Ubuntu GitHub runner
- Python 3.11
- GitHub Actions secrets:
  - `PAT_TOKEN`
  - `PYPI_API_TOKEN`

---

## 🚀 Publishing & Deployment

### Local Build
```bash
python -m pip install build
python -m build --sdist --wheel
```

### Upload to PyPI
```bash
python -m pip install twine
python -m twine upload dist/*
```

### Install from PyPI
```bash
pip install near-jsonrpc-client
```

---

## 📦 Usage Example

```python
import asyncio
from near_jsonrpc_client import NearClientAsync

async def main():
    client = NearClientAsync("https://rpc.mainnet.near.org")
    status = await client.status()
    print(status)

asyncio.run(main())
```

---

## ⚠️ Common Issues & Notes

- `__pycache__` and `*.egg-info` must **never** be committed
- Generator output should only include:
  - `near_jsonrpc_client/`
  - `near_jsonrpc_models/`
- Always use a **virtual environment** when testing locally
- PyPI version mismatches usually indicate missing version injection

---

## 📖 Release Notes Policy

Each GitHub Release includes:
- Generated tag
- Automated description
- Commit summary since last release

Example:
```
🚀 Automated release
- chore: regenerate client from OpenAPI
- fix: correct block response parsing
```

---

## 📞 Support

For issues related to:
- Generator
- CI/CD
- Versioning
- Publishing

Open an issue on GitHub or contact maintainers directly.
