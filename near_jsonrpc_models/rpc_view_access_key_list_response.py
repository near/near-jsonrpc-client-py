"""Lists access keys"""

from near_jsonrpc_models.access_key_info_view import AccessKeyInfoView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.public_key_handle import PublicKeyHandle
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcViewAccessKeyListResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    keys: List[AccessKeyInfoView]
    # Pagination cursor. When `Some`, the listing was truncated and the caller
    # should issue another request with `after_key` set to this handle to fetch
    # the next page. `None` means this was the last page.
    last_key: PublicKeyHandle | None = None
