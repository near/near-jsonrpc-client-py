"""This action allows to execute the inner actions behalf of the defined sender."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.non_delegate_action import NonDelegateAction
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel
from pydantic import conint
from typing import List


class DelegateAction(BaseModel):
    # List of actions to be executed.
    # 
    # With the meta transactions MVP defined in NEP-366, nested
    # DelegateActions are not allowed. A separate type is used to enforce it.
    actions: List[NonDelegateAction]
    # The maximal height of the block in the blockchain below which the given DelegateAction is valid.
    max_block_height: conint(ge=0, le=18446744073709551615)
    # Nonce to ensure that the same delegate action is not sent twice by a
    # relayer and should match for given account's `public_key`.
    # After this action is processed it will increment.
    nonce: conint(ge=0, le=18446744073709551615)
    # Public key used to sign this delegated action.
    public_key: PublicKey
    # Receiver of the delegated actions.
    receiver_id: AccountId
    # Signer of the delegated actions
    sender_id: AccountId
