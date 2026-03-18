"""Controls how the transaction nonce is validated against the access key nonce.monotonic: Any nonce strictly greater than the current access key nonce (default behavior).strict: Nonce must be exactly `ak_nonce + 1` (sequential ordering)."""

from pydantic import RootModel
from typing import Literal


class NonceMode(RootModel[Literal['monotonic', 'strict']]):
    pass

