from pydantic import BaseModel


class ClientError(Exception):
    """Base error for all NEAR client related failures."""
    pass


class TransportError(ClientError):
    """Network-level errors (timeout, DNS, connection, etc)."""
    pass


class HttpError(ClientError):
    """Non-200 HTTP responses."""

    def __init__(self, status_code: int, body: str | None = None):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP error {status_code}")


class RpcError(ClientError):
    """JSON-RPC error object wrapping the Pydantic error model."""

    def __init__(self, error: BaseModel):
        self.error = error
        super().__init__(getattr(error, "message", "RPC Error"))


class RequestTimeoutError(ClientError):
    """Timeout Error"""

    def __init__(self):
        super().__init__("Timeout Error")
