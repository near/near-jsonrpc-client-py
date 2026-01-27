"""Withdraw NEAR from a gas key's balance to the account"""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class WithdrawFromGasKeyAction(BaseModel):
    # Amount of NEAR to transfer from the gas key
    amount: NearToken
    # The public key of the gas key to withdraw from
    public_key: PublicKey
