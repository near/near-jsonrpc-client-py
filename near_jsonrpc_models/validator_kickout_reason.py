"""Reasons for removing a validator from the validator set."""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Literal
from typing import Union


"""Deprecated"""
class ValidatorKickoutReasonUnusedSlashed(RootModel[Literal['_UnusedSlashed']]):
    pass

class ValidatorKickoutReasonNotEnoughBlocksPayload(BaseModel):
    expected: conint(ge=0, le=18446744073709551615)
    produced: conint(ge=0, le=18446744073709551615)

class ValidatorKickoutReasonNotEnoughBlocks(StrictBaseModel):
    """Validator didn't produce enough blocks."""
    NotEnoughBlocks: ValidatorKickoutReasonNotEnoughBlocksPayload

class ValidatorKickoutReasonNotEnoughChunksPayload(BaseModel):
    expected: conint(ge=0, le=18446744073709551615)
    produced: conint(ge=0, le=18446744073709551615)

class ValidatorKickoutReasonNotEnoughChunks(StrictBaseModel):
    """Validator didn't produce enough chunks."""
    NotEnoughChunks: ValidatorKickoutReasonNotEnoughChunksPayload

"""Validator unstaked themselves."""
class ValidatorKickoutReasonUnstaked(RootModel[Literal['Unstaked']]):
    pass

class ValidatorKickoutReasonNotEnoughStakePayload(BaseModel):
    stake_u128: NearToken
    threshold_u128: NearToken

class ValidatorKickoutReasonNotEnoughStake(StrictBaseModel):
    """Validator stake is now below threshold"""
    NotEnoughStake: ValidatorKickoutReasonNotEnoughStakePayload

"""Enough stake but is not chosen because of seat limits."""
class ValidatorKickoutReasonDidNotGetASeat(RootModel[Literal['DidNotGetASeat']]):
    pass

class ValidatorKickoutReasonNotEnoughChunkEndorsementsPayload(BaseModel):
    expected: conint(ge=0, le=18446744073709551615)
    produced: conint(ge=0, le=18446744073709551615)

class ValidatorKickoutReasonNotEnoughChunkEndorsements(StrictBaseModel):
    """Validator didn't produce enough chunk endorsements."""
    NotEnoughChunkEndorsements: ValidatorKickoutReasonNotEnoughChunkEndorsementsPayload

class ValidatorKickoutReasonProtocolVersionTooOldPayload(BaseModel):
    network_version: conint(ge=0, le=4294967295)
    version: conint(ge=0, le=4294967295)

class ValidatorKickoutReasonProtocolVersionTooOld(StrictBaseModel):
    """Validator's last block proposal was for a protocol version older than
the network's voted protocol version."""
    ProtocolVersionTooOld: ValidatorKickoutReasonProtocolVersionTooOldPayload

class ValidatorKickoutReason(RootModel[Union[ValidatorKickoutReasonUnusedSlashed, ValidatorKickoutReasonNotEnoughBlocks, ValidatorKickoutReasonNotEnoughChunks, ValidatorKickoutReasonUnstaked, ValidatorKickoutReasonNotEnoughStake, ValidatorKickoutReasonDidNotGetASeat, ValidatorKickoutReasonNotEnoughChunkEndorsements, ValidatorKickoutReasonProtocolVersionTooOld]]):
    pass

