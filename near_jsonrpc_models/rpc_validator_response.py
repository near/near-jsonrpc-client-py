"""Information about this epoch validators and next epoch validators"""

from near_jsonrpc_models.current_epoch_validator_info import CurrentEpochValidatorInfo
from near_jsonrpc_models.next_epoch_validator_info import NextEpochValidatorInfo
from near_jsonrpc_models.validator_kickout_view import ValidatorKickoutView
from near_jsonrpc_models.validator_stake_view import ValidatorStakeView
from pydantic import BaseModel
from pydantic import conint
from typing import List


class RpcValidatorResponse(BaseModel):
    # Fishermen for the current epoch
    current_fishermen: List[ValidatorStakeView]
    # Proposals in the current epoch
    current_proposals: List[ValidatorStakeView]
    # Validators for the current epoch
    current_validators: List[CurrentEpochValidatorInfo]
    # Epoch height
    epoch_height: conint(ge=0, le=18446744073709551615)
    # Epoch start block height
    epoch_start_height: conint(ge=0, le=18446744073709551615)
    # Fishermen for the next epoch
    next_fishermen: List[ValidatorStakeView]
    # Validators for the next epoch
    next_validators: List[NextEpochValidatorInfo]
    # Kickout in the previous epoch
    prev_epoch_kickout: List[ValidatorKickoutView]
