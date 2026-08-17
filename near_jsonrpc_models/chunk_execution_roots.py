"""Merkle leaf committing to a single chunk's certified execution roots.
The `chunk_execution_root` in a spice block header is the merkle root over
these leaves, sorted by `chunk_id`."""

from near_jsonrpc_models.chunk_execution_roots_v1 import ChunkExecutionRootsV1
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from typing import Union


class ChunkExecutionRootsV1Option(StrictBaseModel):
    V1: ChunkExecutionRootsV1

class ChunkExecutionRoots(RootModel[Union[ChunkExecutionRootsV1Option]]):
    pass

