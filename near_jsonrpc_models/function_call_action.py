from near_jsonrpc_models.near_gas import NearGas
from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel


class FunctionCallAction(BaseModel):
    args: str
    deposit: NearToken
    gas: NearGas
    method_name: str
