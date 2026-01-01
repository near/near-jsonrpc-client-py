from pydantic import BaseModel
from pydantic import RootModel


class Signature(RootModel[str]):
    pass

