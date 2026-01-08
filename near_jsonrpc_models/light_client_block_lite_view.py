from near_jsonrpc_models.block_header_inner_lite_view import BlockHeaderInnerLiteView
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class LightClientBlockLiteView(BaseModel):
    inner_lite: BlockHeaderInnerLiteView
    inner_rest_hash: CryptoHash
    prev_block_hash: CryptoHash
