from pydantic import BaseModel
from pydantic import RootModel


class PublicKeyHandle(RootModel[str]):
    pass

