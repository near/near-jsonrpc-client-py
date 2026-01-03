from near_jsonrpc_models.chunk_hash import ChunkHash
from near_jsonrpc_models.shard_id import ShardId
from pydantic import BaseModel
from pydantic import RootModel
from typing import Any
from typing import Dict
from typing import Literal
from typing import Union


class RpcChunkErrorInternalErrorInfo(BaseModel):
    error_message: str

class RpcChunkErrorInternalError(BaseModel):
    info: RpcChunkErrorInternalErrorInfo
    name: Literal['INTERNAL_ERROR']

class RpcChunkErrorUnknownBlock(BaseModel):
    info: Dict[str, Any]
    name: Literal['UNKNOWN_BLOCK']

class RpcChunkErrorInvalidShardIdInfo(BaseModel):
    shard_id: ShardId

class RpcChunkErrorInvalidShardId(BaseModel):
    info: RpcChunkErrorInvalidShardIdInfo
    name: Literal['INVALID_SHARD_ID']

class RpcChunkErrorUnknownChunkInfo(BaseModel):
    chunk_hash: ChunkHash

class RpcChunkErrorUnknownChunk(BaseModel):
    info: RpcChunkErrorUnknownChunkInfo
    name: Literal['UNKNOWN_CHUNK']

class RpcChunkError(RootModel[Union[RpcChunkErrorInternalError, RpcChunkErrorUnknownBlock, RpcChunkErrorInvalidShardId, RpcChunkErrorUnknownChunk]]):
    pass

