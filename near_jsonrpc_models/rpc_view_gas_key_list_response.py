from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.gas_key_info_view import GasKeyInfoView
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcViewGasKeyListResponse(BaseModel):
    block_hash: CryptoHash
    block_height: conint(ge=0, le=18446744073709551615)
    keys: List[GasKeyInfoView]
