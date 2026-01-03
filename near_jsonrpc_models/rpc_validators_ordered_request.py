from near_jsonrpc_models.block_id import BlockId
from pydantic import BaseModel


class RpcValidatorsOrderedRequest(BaseModel):
    block_id: BlockId | None = None
