"""Delegate action with gas key support: `nonce` selects either the access
key's nonce or one of a gas key's parallel nonces by index, mirroring
`TransactionV1`."""

from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.non_delegate_action import NonDelegateAction
from near_jsonrpc_models.public_key import PublicKey
from near_jsonrpc_models.transaction_nonce import TransactionNonce
from pydantic import BaseModel
from pydantic import conint
from typing import List


class DelegateActionV2(BaseModel):
    # List of actions to be executed.
    actions: List[NonDelegateAction]
    # The maximal height of the block in the blockchain below which the given DelegateActionV2 is valid.
    max_block_height: conint(ge=0, le=18446744073709551615)
    # Nonce of the signing key, advanced when this action is processed. For
    # a gas key it also selects which of the parallel nonces to advance.
    nonce: TransactionNonce
    # Public key used to sign this delegated action.
    public_key: PublicKey
    # Receiver of the delegated actions.
    receiver_id: AccountId
    # Signer of the delegated actions
    sender_id: AccountId
