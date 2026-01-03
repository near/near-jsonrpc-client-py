"""A state for the current head of a light client. More info [here](https://nomicon.io/ChainSpec/LightClient)."""

from near_jsonrpc_models.block_header_inner_lite_view import BlockHeaderInnerLiteView
from near_jsonrpc_models.crypto_hash import CryptoHash
from near_jsonrpc_models.signature import Signature
from near_jsonrpc_models.validator_stake_view import ValidatorStakeView
from pydantic import BaseModel
from typing import List


class RpcLightClientNextBlockResponse(BaseModel):
    approvals_after_next: List[Signature | None] = None
    # Inner part of the block header that gets hashed, split into two parts, one that is sent
    #    to light clients, and the rest
    inner_lite: BlockHeaderInnerLiteView = None
    inner_rest_hash: CryptoHash = None
    next_block_inner_hash: CryptoHash = None
    next_bps: List[ValidatorStakeView] | None = None
    prev_block_hash: CryptoHash = None
