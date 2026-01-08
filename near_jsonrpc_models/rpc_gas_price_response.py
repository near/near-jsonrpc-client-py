from near_jsonrpc_models.near_token import NearToken
from pydantic import BaseModel


class RpcGasPriceResponse(BaseModel):
    gas_price: NearToken
