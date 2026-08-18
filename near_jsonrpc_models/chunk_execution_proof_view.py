"""Proof that a chunk's certified execution roots are committed by a spice block
that a light client can trust via its `light_client_head`.

`roots_proof` recomputes the certifying block's `chunk_execution_root` from the leaf;
`certifying_block_proof` places the certifying block into the head's block merkle tree."""

from near_jsonrpc_models.chunk_execution_roots import ChunkExecutionRoots
from near_jsonrpc_models.light_client_block_lite_view import LightClientBlockLiteView
from near_jsonrpc_models.merkle_path_item import MerklePathItem
from pydantic import BaseModel
from typing import List


class ChunkExecutionProofView(BaseModel):
    certifying_block_header_lite: LightClientBlockLiteView
    certifying_block_proof: List[MerklePathItem]
    roots: ChunkExecutionRoots
    roots_proof: List[MerklePathItem]
