from pydantic import BaseModel
from pydantic import RootModel


class SignedTransaction(RootModel[str]):
    pass

