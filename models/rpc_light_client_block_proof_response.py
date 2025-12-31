from models.light_client_block_lite_view import LightClientBlockLiteView
from models.merkle_path_item import MerklePathItem
from pydantic import BaseModel
from typing import List


class RpcLightClientBlockProofResponse(BaseModel):
    block_header_lite: LightClientBlockLiteView
    block_proof: List[MerklePathItem]
