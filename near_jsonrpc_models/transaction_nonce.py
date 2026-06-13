from near_jsonrpc_models.strict_model import StrictBaseModel
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class TransactionNonceNoncePayload(BaseModel):
    nonce: conint(ge=0, le=18446744073709551615)

class TransactionNonceNonce(StrictBaseModel):
    """Simple nonce without index, used by ordinary access keys"""
    Nonce: TransactionNonceNoncePayload

class TransactionNonceGasKeyNoncePayload(BaseModel):
    nonce: conint(ge=0, le=18446744073709551615)
    nonce_index: conint(ge=0, le=65535)

class TransactionNonceGasKeyNonce(StrictBaseModel):
    """Nonce with index, used by gas keys"""
    GasKeyNonce: TransactionNonceGasKeyNoncePayload

class TransactionNonce(RootModel[Union[TransactionNonceNonce, TransactionNonceGasKeyNonce]]):
    pass

