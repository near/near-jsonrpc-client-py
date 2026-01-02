from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.block_header_view import BlockHeaderView
from near_jsonrpc_models.chunk_header_view import ChunkHeaderView
from pydantic import BaseModel
from typing import List


class RpcBlockResponse(BaseModel):
    # The AccountId of the author of the Block
    author: AccountId
    chunks: List[ChunkHeaderView]
    header: BlockHeaderView
