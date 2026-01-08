from near_jsonrpc_models.account_id import AccountId
from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from typing import Literal
from typing import Union


class RpcLightClientExecutionProofRequestTransaction(BaseModel):
    light_client_head: CryptoHash
    sender_id: AccountId
    transaction_hash: CryptoHash
    type: Literal['transaction']

class RpcLightClientExecutionProofRequestReceipt(BaseModel):
    light_client_head: CryptoHash
    receipt_id: CryptoHash
    receiver_id: AccountId
    type: Literal['receipt']

class RpcLightClientExecutionProofRequest(RootModel[Union[RpcLightClientExecutionProofRequestTransaction, RpcLightClientExecutionProofRequestReceipt]]):
    pass

