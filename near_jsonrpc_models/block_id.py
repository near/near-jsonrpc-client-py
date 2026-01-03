from near_jsonrpc_models.crypto_hash import CryptoHash
from pydantic import BaseModel
from pydantic import RootModel
from pydantic import conint
from typing import Union


class BlockIdBlockHeight(RootModel[conint(ge=0, le=18446744073709551615)]):
    pass

class BlockIdCryptoHash(CryptoHash):
    pass

class BlockId(RootModel[Union[BlockIdBlockHeight, BlockIdCryptoHash]]):
    pass

