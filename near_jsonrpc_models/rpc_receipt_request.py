from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel


class RpcReceiptRequest(BaseModel):
    receipt_id: CryptoHash
