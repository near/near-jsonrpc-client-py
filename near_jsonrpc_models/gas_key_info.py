from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel
from pydantic import conint


class GasKeyInfo(BaseModel):
    balance: NearToken
    num_nonces: conint(ge=0, le=4294967295)
