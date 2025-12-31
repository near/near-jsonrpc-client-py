from models.account_id import AccountId
from models.block_header_view import BlockHeaderView
from models.chunk_header_view import ChunkHeaderView
from pydantic import BaseModel
from typing import List


class RpcBlockResponse(BaseModel):
    # The AccountId of the author of the Block
    author: AccountId
    chunks: List[ChunkHeaderView]
    header: BlockHeaderView
