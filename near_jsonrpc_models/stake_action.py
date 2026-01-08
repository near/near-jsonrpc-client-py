"""An action which stakes signer_id tokens and setup's validator public key"""

from near_jsonrpc_models.near_token import NearToken
from near_jsonrpc_models.public_key import PublicKey
from pydantic import BaseModel


class StakeAction(BaseModel):
    # Validator key which will be used to sign transactions on behalf of signer_id
    public_key: PublicKey
    # Amount of tokens to stake.
    stake: NearToken
