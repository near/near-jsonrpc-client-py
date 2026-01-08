from pydantic import BaseModel
from pydantic import RootModel


class PublicKey(RootModel[str]):
    pass

