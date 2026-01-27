"""Transfer NEAR to a gas key's balance"""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class TransferToGasKeyAction(BaseModel):
    # Amount of NEAR to transfer to the gas key
    deposit: NearToken
    # The public key of the gas key to fund
    public_key: PublicKey
