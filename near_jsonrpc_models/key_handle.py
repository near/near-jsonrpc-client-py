from pydantic import BaseModel
from pydantic import RootModel


class KeyHandle(RootModel[str]):
    pass

