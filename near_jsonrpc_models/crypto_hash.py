from pydantic import BaseModel
from pydantic import RootModel


class CryptoHash(RootModel[str]):
    pass

