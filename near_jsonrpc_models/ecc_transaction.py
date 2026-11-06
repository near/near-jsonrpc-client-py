from pydantic import BaseModel
from pydantic import RootModel


class EccTransaction(RootModel[str]):
    pass

